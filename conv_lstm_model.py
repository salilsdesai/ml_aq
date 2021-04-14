import torch

from time import strftime
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Tuple, Dict, Optional, Any

from .conv_model import ConvModel, ConvParams, ConvReceptorData
from .model import Model
from .utils import Receptor, Link, Features, DEVICE, MetStation, partition, \
	lambda_to_string, A, B, TRANSFORM_OUTPUT, TRANSFORM_OUTPUT_INV, \
	train_val_split

NOTEBOOK_NAME = 'conv_lstm_model.py'
DIRECTORY = '.'

class ConvLSTMReceptorData(ConvReceptorData):
	def __init__(self, distances: Tensor, closest_filter: Tensor):
		super(ConvLSTMReceptorData, self).__init__(distances=distances)
		self.closest_filter: Tensor = closest_filter

class ConvLSTMParams(ConvParams):
	def __init__(
		self,
		hidden_size: int,
		batch_size: int,
		transform_output_src: str,
		transform_output_inv_src: str,
		concentration_threshold: float,
		distance_threshold: float,
		link_features: List[str],
		approx_bin_size: float,
		kernel_size: int,
		time_periods: List[str],
		distance_feature_stats: Optional[Features.FeatureStats],
		num_out_channels: int,
	):
		super(ConvLSTMParams, self).__init__(
			hidden_size=hidden_size,
			batch_size=batch_size,
			transform_output_src=transform_output_src,
			transform_output_inv_src=transform_output_inv_src,
			concentration_threshold=concentration_threshold,
			distance_threshold=distance_threshold,
			link_features=link_features,
			approx_bin_size=approx_bin_size,
			kernel_size=kernel_size,
			time_periods=time_periods,
			distance_feature_stats=distance_feature_stats,
		)
		self.num_out_channels: int = num_out_channels
	
	@staticmethod
	def from_dict(d: Dict[str, Any]) -> 'ConvLSTMParams':
		return ConvLSTMParams(
			hidden_size = d['hidden_size'],
			batch_size = d['batch_size'],
			transform_output_src = d['transform_output_src'],
			transform_output_inv_src = d['transform_output_inv_src'],
			concentration_threshold = d['concentration_threshold'],
			distance_threshold = d['distance_threshold'],
			link_features = d['link_features'],
			approx_bin_size = d['approx_bin_size'],
			kernel_size = d['kernel_size'],
			time_periods = d['time_periods'],
			distance_feature_stats = Features.FeatureStats(
				mean = d['distances_feature_stats'][0], 
				std_dev = d['distances_feature_stats'][1]
			)  if d['distances_feature_stats'] is not None else None,
			num_out_channels = d['num_out_channels'],
		)
	
	def child_dict(self) -> Dict[str, Any]:
		return {
			'num_out_channels': self.num_out_channels,
		}

class ConvLSTMCell(torch.nn.Module):
	"""
	Code based on:
	https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
	"""
	def __init__(self, input_dim, hidden_dim, kernel_size, bias):
		"""
		Initialize ConvLSTM cell.

		Parameters
		----------
		input_dim: int
			Number of channels of input tensor.
		hidden_dim: int
			Number of channels of hidden state.
		kernel_size: (int, int)
			Size of the convolutional kernel.
		bias: bool
			Whether or not to add the bias.
		"""

		super(ConvLSTMCell, self).__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim

		self.kernel_size = kernel_size
		self.padding = kernel_size[0] // 2, kernel_size[1] // 2
		self.bias = bias

		self.conv = torch.nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
							  out_channels=4 * self.hidden_dim,
							  kernel_size=self.kernel_size,
							  padding=self.padding,
							  bias=self.bias).to(DEVICE)

	def forward(self, input_tensor, cur_state):
		h_cur, c_cur = cur_state

		combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

		combined_conv = self.conv(combined)
		cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
		i = torch.sigmoid(cc_i)
		f = torch.sigmoid(cc_f)
		o = torch.sigmoid(cc_o)
		g = torch.tanh(cc_g)

		c_next = f * c_cur + i * g
		h_next = o * torch.tanh(c_next)

		return h_next, c_next

	def init_hidden(self, batch_size, image_size):
		height, width = image_size
		return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
				torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class EncoderDecoderConvLSTM(torch.nn.Module):
	"""
	Code based on: 
	https://gist.github.com/holmdk/b2c5ac479366e3eed3345dd677738626#file-gistfile1-txt
	"""
	def __init__(self, in_chan, out_chan, kernel_size):
		torch.nn.Module.__init__(self)

		""" ARCHITECTURE

		# Encoder (ConvLSTM)
		# Encoder Vector (final hidden state of encoder)
		# Decoder (3D CNN) - produces regression predictions for our model

		"""
		self.encoder_convlstm = ConvLSTMCell(input_dim=in_chan,
											   hidden_dim=out_chan,
											   kernel_size=(kernel_size, kernel_size),
											   bias=True)

		self.decoder_CNN = torch.nn.Conv3d(in_channels=out_chan,
									 out_channels=1,
									 kernel_size=(1, kernel_size, kernel_size),
									 padding=(0, 1, 1)).to(DEVICE)
	

	def autoencoder(self, x, seq_len, future_step, h_t, c_t):

		outputs = []

		# encoder
		for t in range(seq_len):
			h_t, c_t = self.encoder_convlstm(input_tensor=x[:, t, :, :],
											   cur_state=[h_t, c_t])  # we could concat to provide skip conn here

		# encoder_vector
		encoder_vector = h_t
		outputs += [h_t]

		outputs = torch.stack(outputs, 1)
		outputs = outputs.permute(0, 2, 1, 3, 4)
		outputs = self.decoder_CNN(outputs)
		outputs = torch.nn.Sigmoid()(outputs)

		return outputs

	def forward(self, x, future_seq=0, hidden_state=None):

		"""
		Parameters
		----------
		input_tensor:
			5-D Tensor of shape (b, t, c, h, w)		#   batch, time, channel, height, width
		"""

		# find size of different input dimensions
		b, seq_len, _, h, w = x.size()

		# initialize hidden states
		h_t, c_t = self.encoder_convlstm.init_hidden(batch_size=b, image_size=(h, w))
		
		# autoencoder forward
		outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t)

		return outputs

class ConvLSTMModel(EncoderDecoderConvLSTM, ConvModel):
	def __init__(self, params: ConvLSTMParams):
		ConvModel.__init__(self, params)
		self.params: ConvLSTMParams = params
		EncoderDecoderConvLSTM.__init__(
			self,
			in_chan = self.input_size,
			out_chan = self.params.num_out_channels,
			kernel_size = self.params.kernel_size
		)

	def get_time_periods(self) -> List[str]:
		"""
		Override
		"""
		return self.params.time_periods
	
	def fix_channel_dims(self, channels: Tensor) -> Tensor:
		"""
		Override
		"""
		return channels
	
	
	def make_receptor_data(self, distances: Tensor) -> ConvLSTMReceptorData:
		"""
		Override
		"""
		distances = distances.unsqueeze(dim=1).unsqueeze(dim=1)
		closest_filter = (distances <= distances.amin(dim=(3, 4)).unsqueeze(dim=3).unsqueeze(dim=4))
		return ConvLSTMReceptorData(distances, closest_filter)


	def forward_batch(self, receptors: ConvLSTMReceptorData) -> Tensor:
		"""
		Override
		"""
		x = torch.cat((receptors.distances.repeat(1, len(self.get_time_periods()), 1, 1, 1), self.link_data.channels), dim=2)
		fwd = self.forward(x)
		filtered = (fwd * receptors.closest_filter).squeeze(1) * self.link_data.bin_counts
		return filtered.sum(dim=2).sum(dim=2)
	
	@staticmethod
	def load(filepath: str) -> Tuple['ConvLSTMModel', Optimizer]:
		return Model.load(filepath, ConvLSTMModel, ConvLSTMParams)

if __name__ == '__main__':
	model = ConvLSTMModel(
		ConvLSTMParams(
			hidden_size = 8,
			batch_size = 1000,
			transform_output_src = lambda_to_string(
				TRANSFORM_OUTPUT,
				[('A', str(A)), ('B', str(B))]
			),
			transform_output_inv_src = lambda_to_string(
				TRANSFORM_OUTPUT_INV,
				[('A', str(A)), ('B', str(B))]
			),
			concentration_threshold = 0.01,
			distance_threshold = 500,
			link_features = [
				Features.VMT, Features.TRAFFIC_SPEED, Features.FLEET_MIX_LIGHT,
				Features.FLEET_MIX_MEDIUM, Features.FLEET_MIX_HEAVY,
				Features.FLEET_MIX_COMMERCIAL, Features.FLEET_MIX_BUS,
				Features.WIND_DIRECTION, Features.WIND_SPEED,
				Features.UP_DOWN_WIND_EFFECT,
			],
			approx_bin_size = 1000,
			kernel_size = 3,
			time_periods = [''],
			distance_feature_stats = None,
			num_out_channels = 64,
		)
	)

	links_list = Link.load_links(DIRECTORY + '/data/link_data.csv')
	met_data = MetStation.load_met_data(DIRECTORY + '/data/met_data.csv')

	model.set_link_data(links_list, met_data)

	receptors_list = model.filter_receptors(Receptor.load_receptors(DIRECTORY + '/data/receptor_data.csv'))
	train_receptors_list, val_receptors_list = train_val_split(receptors_list)

	train_batches = model.make_receptor_batches(partition(train_receptors_list, model.params.batch_size))
	val_batches = model.make_receptor_batches(partition(val_receptors_list, model.params.batch_size))

	save_location = DIRECTORY + '/Model Saves/model_save_' + strftime('%m-%d-%y %H:%M:%S') + ' ' + NOTEBOOK_NAME
	print('Save Location: ' + save_location)

	model.train(
		optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001),
		num_epochs = 1000,
		train_batches = train_batches,
		val_batches = val_batches,
		save_location = save_location,
		make_graphs = True
	)

	best_model, _ = ConvLSTMModel.load(save_location)
	best_model.set_link_data(links_list, met_data)

	batch = val_batches[1]
	fwd = best_model.forward_batch(batch.receptors)
	for i in range(5):
		print((best_model.params.transform_output_inv(fwd[i].item(), batch.nearest_link_distances[i].item()), best_model.params.transform_output_inv(batch.y[i].item(), batch.nearest_link_distances[i].item())))

	print(sum([best_model.get_error(batch.receptors, batch.y) for batch in val_batches])/len(val_batches))

	best_model.graph_prediction_error(val_batches)

	best_model.export_predictions(val_batches, save_location[save_location.rindex('model_save_') + 11:save_location.rindex('.')] + ' Predictions.csv')

	best_model.print_batch_errors(val_batches)