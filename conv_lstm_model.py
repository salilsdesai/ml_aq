import torch

from time import strftime
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Tuple, Dict, Optional, Any

from .model import Model, ReceptorBatch, ModelParams
from .utils import Receptor, Link, Coordinate, Features, DEVICE, MetStation, \
	lambda_to_string, A, B, TRANSFORM_OUTPUT, TRANSFORM_OUTPUT_INV, \
	partition, train_val_split, flatten

NOTEBOOK_NAME = 'conv_lstm_model.py'
DIRECTORY = '.'

class LinkData():
	def __init__(self, channels: Tensor, num_time_periods: int, bin_counts: Tensor, bin_centers: Tensor):
		self.channels: Tensor = channels
		self.num_time_periods: int = num_time_periods
		self.bin_counts: Tensor = bin_counts
		self.bin_centers: Tensor = bin_centers

class ReceptorData():
	def __init__(self, distances: Tensor, closest_filter: Tensor):
		self.distances: Tensor = distances
		self.closest_filter: Tensor = closest_filter

class ConvLSTMReceptorBatch(ReceptorBatch):
	def __init__(
		self, 
		receptors: ReceptorData, 
		y: Tensor, 
		nearest_link_distances: Tensor,
		coordinates: List[Coordinate],
	):
		ReceptorBatch.__init__(self, receptors, y, nearest_link_distances)
		self.coordinates: List[Coordinate] = coordinates

	def coordinate(self, i: int) -> Coordinate:
		"""
		returns the coordinates of the receptor at index i in the batch
		"""
		return self.coordinates[i]
	def size(self) -> int:
		return self.receptors.distances.shape[0]

class ConvLSTMModelParams(ModelParams):
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
		num_out_channels: int,
		time_periods: List[str],
		distance_feature_stats: Optional[Features.FeatureStats]
	):
		ModelParams.__init__(
			self,
			hidden_size=hidden_size,
			batch_size=batch_size,
			transform_output_src=transform_output_src,
			transform_output_inv_src=transform_output_inv_src,
			concentration_threshold=concentration_threshold,
			distance_threshold=distance_threshold,
			link_features=link_features
		)
		self.approx_bin_size: float = approx_bin_size
		self.kernel_size: int = kernel_size
		self.num_out_channels: int = num_out_channels
		self.time_periods: List[str] = time_periods
		self.distance_feature_stats: Optional[Features.FeatureStats] = \
			distance_feature_stats 
	
	@staticmethod
	def from_dict(d: Dict[str, Any]) -> 'ConvLSTMModelParams':
		return ConvLSTMModelParams(
			hidden_size = d['hidden_size'],
			batch_size = d['batch_size'],
			transform_output_src = d['transform_output_src'],
			transform_output_inv_src = d['transform_output_inv_src'],
			concentration_threshold = d['concentration_threshold'],
			distance_threshold = d['distance_threshold'],
			link_features = d['link_features'],
			approx_bin_size = d['approx_bin_size'],
			kernel_size = d['kernel_size'],
			num_out_channels = d['num_out_channels'],
			time_periods = d['time_periods'],
			distance_feature_stats = Features.FeatureStats(
				mean = d['distances_feature_stats'][0], 
				std_dev = d['distances_feature_stats'][1]
			)  if d['distances_feature_stats'] is not None else None,
		)
	
	def child_dict(self) -> Dict[str, Any]:
		return {
			'approx_bin_size': self.approx_bin_size,
			'kernel_size': self.kernel_size,
			'num_out_channels': self.num_out_channels,
			'time_periods': self.time_periods,
			'distances_feature_stats': (
				self.distance_feature_stats.mean, 
				self.distance_feature_stats.std_dev
			) if self.distance_feature_stats is not None else None,
		}

class CumulativeStats:
	"""
	Code based on
	http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
	"""
	def __init__(self):
		self.mean = Tensor([0]).to(DEVICE)
		self.stddev = Tensor([0]).to(DEVICE)
		self.n = 0

	def update(self, data):
		m1 = self.mean
		s1 = self.stddev
		n1 = self.n

		m2 = torch.mean(data)
		s2 = torch.std(data, unbiased=False)
		n2 = torch.numel(data)
		
		self.n = n1 + n2
		self.mean = m1 * (n1 / self.n) + m2 * (n2 / self.n)
		self.stddev = ((s1 ** 2) * (n1 / self.n) + (s2 ** 2) * (n2 / self.n) + ((n1 * n2) / ((n1 + n2) ** 2)) * ((m1 - m2) ** 2)) ** 0.5


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

class ConvLSTMModel(EncoderDecoderConvLSTM, Model):
	def __init__(self, params: ConvLSTMModelParams):
		Model.__init__(self, params)
		self.params: ConvLSTMModelParams = params
		self.input_size: int = 1 + len(self.params.link_features)  # +1 for distance
		self.approx_bin_size: float = params.approx_bin_size
		self.num_out_channels: float = params.num_out_channels
		self.kernel_size: int = params.kernel_size
		EncoderDecoderConvLSTM.__init__(
			self,
			in_chan = self.input_size,
			out_chan = self.num_out_channels,
			kernel_size = self.kernel_size
		)
	
	def get_receptor_locations(self, receptors_list: List[Receptor]) -> Tuple[List[List[float]], Tensor]:
		"""
		Returns 
		- A list of the x and y coordinates of each receptor represented as a
		list of size 2
		- 3 dimensional tensor, where the value at index (i, j, k) is the 
		distance from receptor i to the center of bin (j, k)
		"""
		if self.link_data is None:
			raise Exception('Link Data Not Set')
		
		coords_list = [[r.x, r.y] for r in receptors_list]
		receptor_coords_tensor = Tensor(coords_list).to(DEVICE).unsqueeze(dim=1).unsqueeze(dim=1)
		distances = torch.linalg.norm(receptor_coords_tensor - self.link_data.bin_centers, dim=3)
		return coords_list, distances 
	
	def make_receptor_batches(self, receptors: List[List[Receptor]]) -> List[ReceptorBatch]:  
		if self.link_data is None:
			raise Exception('Link Data Not Set')

		if len(receptors) == 0:
			return []

		cumulative_stats = CumulativeStats() if self.params.distance_feature_stats is None else None

		def make_batch(receptors_list: List[Receptor]) -> ReceptorBatch:
			coords_list, distances = self.get_receptor_locations(receptors_list)
			distances = distances.unsqueeze(dim=1).unsqueeze(dim=1)
			if cumulative_stats is not None:
				cumulative_stats.update(distances)
			
			closest_filter = (distances <= distances.amin(dim=(3, 4)).unsqueeze(dim=3).unsqueeze(dim=4))

			data = ReceptorData(distances, closest_filter)
			y = Tensor([[self.params.transform_output(r.pollution_concentration, r.nearest_link_distance)] for r in receptors_list]).to(DEVICE)
			nearest_link_distances = Tensor([[r.nearest_link_distance] for r in receptors_list]).to(DEVICE)
			coordinates = [Coordinate(c[0], c[1]) for c in coords_list]

			return ConvLSTMReceptorBatch(
				receptors = data,
				y = y,
				nearest_link_distances = nearest_link_distances,
				coordinates = coordinates
			)
		
		batches = [make_batch(receptors_list) for receptors_list in receptors]

		if cumulative_stats is not None:
			self.params.distance_feature_stats = Features.FeatureStats(
				mean = cumulative_stats.mean.item(),
				std_dev = cumulative_stats.stddev.item(),
			)

		for batch in batches:
			batch.receptors.distances = \
				(batch.receptors.distances - self.params.distance_feature_stats.mean) / (self.params.distance_feature_stats.std_dev)
		
		return batches

	def filter_receptors(self, receptors: List[Receptor]) -> List[Receptor]:
		"""
		Override
		In addition to parent method filters, also remove receptors with
		no links in the closest bin to them
		"""
		receptors = super().filter_receptors(receptors)
		partitions = partition(receptors, self.params.batch_size)

		def filter_no_link_in_closest_bin(receptors_list: List[Receptor]) -> List[Receptor]:
			_, distances = self.get_receptor_locations(receptors_list)
			min_indices = distances.flatten(start_dim=1, end_dim=2).min(dim=1).indices
			min_indices_i = (min_indices / distances.shape[2]).long()
			min_indices_j = (min_indices % distances.shape[2]).long()
			closest_bin_link_counts = self.link_data.bin_counts[min_indices_i].gather(1, min_indices_j.view(-1,1))
			return [receptors_list[i] for i in range(len(receptors_list)) if closest_bin_link_counts[i] > 0]

		return flatten([filter_no_link_in_closest_bin(p) for p in partitions])

	def forward_batch(self, receptors: ReceptorData) -> Tensor:
		"""
		Override
		"""
		x = torch.cat((receptors.distances.repeat(1, self.link_data.num_time_periods, 1, 1, 1), self.link_data.channels), dim=2)
		fwd = self.forward(x)
		filtered = (fwd * receptors.closest_filter).squeeze(1) * self.link_data.bin_counts
		return filtered.sum(dim=2).sum(dim=2)
	
	def set_link_data(self, links: List[Link], met_data: Dict[int, MetStation]) -> None:
		"""
		Override
		"""
		min_x, max_x, min_y, max_y = links[0].x, links[0].x, links[0].y, links[0].y
		for link in links:
			if link.x < min_x:
				min_x = link.x
			if link.x > max_x:
				max_x = link.x
			if link.y < min_y:
				min_y = link.y
			if link.y > max_y:
				max_y = link.y
		
		# So the links on the top and right boundaries fit in the bins
		max_x += 1
		max_y += 1
		
		spreads = (max_x - min_x, max_y - min_y)
		num_x, num_y = round(spreads[0] / self.params.approx_bin_size), round(spreads[1] / self.params.approx_bin_size)
		size_x, size_y = spreads[0] / num_x, spreads[1] / num_y

		channels = torch.zeros(
			len(self.params.time_periods), 
			len(self.params.link_features), 
			num_x,
			num_y,
		).to(DEVICE)
		counts = torch.zeros(num_x, num_y).to(DEVICE)

		for link in links:
			bin_x = int((link.x - min_x) / size_x)
			bin_y = int((link.y - min_y) / size_y)
			for i in range(channels.shape[0]):
				for j in range(channels.shape[1]):
					channels[i][j][bin_x][bin_y] += \
						Features.GET_FEATURE_WITH_SUFFIX[
							self.params.link_features[j]
						](link, self.params.time_periods[i], met_data)
			counts[bin_x][bin_y] += 1
		
		for bin_x in range(num_x):
			for bin_y in range(num_y):
				if counts[bin_x][bin_y] != 0:
					for i in range(channels.shape[0]):
						for j in range(channels.shape[1]):
							channels[i][j][bin_x][bin_y] /= counts[bin_x][bin_y]

		mean = channels.mean(dim=(0, 2, 3), keepdim=True)
		std = channels.std(dim=(0, 2, 3), keepdim=True)

		channels = ((channels - mean) / std).unsqueeze(dim=0).repeat(self.params.batch_size, 1, 1, 1, 1)

		x_centers = Tensor([min_x + ((i + 0.5) * size_x) for i in range(num_x)]).to(DEVICE)
		y_centers = Tensor([min_y + ((j + 0.5) * size_y) for j in range(num_y)]).to(DEVICE)
		centers = torch.cartesian_prod(x_centers, y_centers).reshape(x_centers.shape[0], y_centers.shape[0], 2).unsqueeze(dim=0)

		self.link_data = LinkData(
			channels = channels,
			num_time_periods = len(self.params.time_periods),
			bin_counts = counts,
			bin_centers = centers,
		)
		
	@staticmethod
	def load(filepath: str) -> Tuple['ConvLSTMModel', Optimizer]:
		return Model.load(filepath, ConvLSTMModel, ConvLSTMModelParams)

if __name__ == '__main__':
	model = ConvLSTMModel(
		ConvLSTMModelParams(
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
			approx_bin_size = 2000,
			kernel_size = 3,
			num_out_channels = 64,
			time_periods = [''],
			distance_feature_stats = None
		)
	)

	links_list = Link.load_links(DIRECTORY + '/data/link_data.csv')
	met_data = MetStation.load_met_data(DIRECTORY + '/data/met_data.csv')
	feature_stats = Features.get_all_feature_stats(DIRECTORY + '/data/feature_stats.csv')

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