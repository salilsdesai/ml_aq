import torch

from time import strftime
from torch import Tensor
from torch.nn.functional import relu
from torch.optim.optimizer import Optimizer
from typing import List, Tuple, Dict, Optional, Any

from .conv_model import ConvModel, ConvParams, ConvReceptorData
from .model import Model
from .utils import Receptor, Link, Features, DEVICE, MetStation, partition, \
	lambda_to_string, A, B, TRANSFORM_OUTPUT, TRANSFORM_OUTPUT_INV, \
	train_val_split

NOTEBOOK_NAME = 'conv_lstm_model.py'
DIRECTORY = '.'

class CNNParams(ConvParams):
	def __init__(
		self,
		batch_size: int,
		transform_output_src: str,
		transform_output_inv_src: str,
		concentration_threshold: float,
		distance_threshold: float,
		link_features: List[str],
		approx_bin_size: float,
		kernel_size: int,
		distance_feature_stats: Optional[Features.FeatureStats],
		pool_size: int,
		conv_hidden_sizes: List[int],
		linear_hidden_sizes: List[int],
		x_dim: Optional[int],
		y_dim: Optional[int],
	):
		super(CNNParams, self).__init__(
			batch_size=batch_size,
			transform_output_src=transform_output_src,
			transform_output_inv_src=transform_output_inv_src,
			concentration_threshold=concentration_threshold,
			distance_threshold=distance_threshold,
			link_features=link_features,
			approx_bin_size=approx_bin_size,
			kernel_size=kernel_size,
			distance_feature_stats=distance_feature_stats,
		)
		self.pool_size: int = pool_size
		self.conv_hidden_sizes: List[int] = conv_hidden_sizes
		self.linear_hidden_sizes: List[int] = linear_hidden_sizes
		self.x_dim: Optional[int] = x_dim
		self.y_dim: Optional[int] = y_dim
	
	@staticmethod
	def from_dict(d: Dict[str, Any]) -> 'CNNParams':
		return CNNParams(
			batch_size = d['batch_size'],
			transform_output_src = d['transform_output_src'],
			transform_output_inv_src = d['transform_output_inv_src'],
			concentration_threshold = d['concentration_threshold'],
			distance_threshold = d['distance_threshold'],
			link_features = d['link_features'],
			approx_bin_size = d['approx_bin_size'],
			kernel_size = d['kernel_size'],
			distance_feature_stats = Features.FeatureStats(
				mean = d['distances_feature_stats'][0], 
				std_dev = d['distances_feature_stats'][1]
			)  if d['distances_feature_stats'] is not None else None,
			pool_size = d['pool_size'],
			conv_hidden_sizes = d['conv_hidden_sizes'],
			linear_hidden_sizes = d['linear_hidden_sizes'],
			x_dim = d['x_dim'],
			y_dim = d['y_dim'],
		)
	
	def as_dict(self) -> Dict[str, Any]:
		"""
		Override
		"""
		d = super().as_dict()
		d.update({
			'pool_size': self.pool_size,
			'conv_hidden_sizes': self.conv_hidden_sizes,
			'linear_hidden_sizes': self.linear_hidden_sizes,
			'x_dim': self.x_dim,
			'y_dim': self.y_dim,
		})
		return d

class CNNModel(ConvModel):
	def __init__(self, params: CNNParams):
		super(CNNModel, self).__init__(params)
		self.params: CNNParams = params
		# Set up convolutional layers
		hidden_sizes = [self.input_size] + self.params.conv_hidden_sizes
		for i in range(1, len(hidden_sizes)):
			setattr(
				self,
				'conv' + str(i - 1),
				torch.nn.Conv2d(
					in_channels = hidden_sizes[i - 1],
					out_channels = hidden_sizes[i],
					kernel_size = self.params.kernel_size,
					padding = self.params.kernel_size // 2
				).to(DEVICE)
			)
		self.pool = torch.nn.MaxPool2d(kernel_size=self.params.pool_size).to(DEVICE)

		if self.params.x_dim is not None and self.params.y_dim is not None:
			self.set_up_linear_layers()
	
	def set_up_linear_layers(self) -> None:
		if self.params.x_dim is None or self.params.y_dim is None:
			raise Exception('x dim or y dim is None')

		x_dim = self.params.x_dim
		y_dim = self.params.y_dim

		for _ in range(len(self.params.conv_hidden_sizes)):
			x_dim = x_dim // self.params.pool_size
			y_dim = y_dim // self.params.pool_size
		self.linear0_in_size = self.params.conv_hidden_sizes[-1] * x_dim * y_dim
		hidden_sizes = [self.linear0_in_size] + self.params.linear_hidden_sizes + [1]
		for i in range(1, len(hidden_sizes)):
			setattr(
				self,
				'linear' + str(i - 1),
				torch.nn.Linear(
					in_features = hidden_sizes[i - 1],
					out_features = hidden_sizes[i],
				).to(DEVICE)
			)
	
	def forward(self, x: Tensor):
		for i in range(len(self.params.conv_hidden_sizes)):
			x = self.pool(relu(getattr(self, 'conv' + str(i))(x)))
		x = x.view(-1, self.linear0_in_size)
		for i in range(len(self.params.linear_hidden_sizes)):
			x = relu(getattr(self, 'linear' + str(i))(x))
		x = getattr(self, 'linear' + str(len(self.params.linear_hidden_sizes)))(x)
		return x

	def get_time_periods(self) -> List[str]:
		"""
		Override
		"""
		return ['']
	
	def set_up_on_channel_dims(self, channels: Tensor) -> Tensor:
		"""
		Override
		"""
		if self.params.x_dim is None or self.params.y_dim is None:
			self.params.x_dim = channels.shape[3]
			self.params.y_dim = channels.shape[4]
			self.set_up_linear_layers()
		else:
			if self.params.x_dim != channels.shape[3] or self.params.y_dim != channels.shape[4]:
				raise Exception('Input and Expected Dims Mismatch: ' + str((channels.shape[3], channels.shape[4])) + ' vs ' + str((self.params.x_dim, self.params.y_dim)))
		return channels.squeeze(dim=1)
	
	
	def make_receptor_data(self, distances: Tensor) -> ConvReceptorData:
		"""
		Override
		"""
		return ConvReceptorData(distances=distances.unsqueeze(dim=1))


	def forward_batch(self, receptors: ConvReceptorData) -> Tensor:
		"""
		Override
		"""
		return self.forward(
			torch.cat(
				tensors = (receptors.distances, self.link_data.channels), 
				dim = 1,
			)
		)
	
	@staticmethod
	def load(filepath: str) -> Tuple['CNNModel', Optimizer]:
		return Model.load(filepath, CNNModel, CNNParams)

if __name__ == '__main__':
	model = CNNModel(
		CNNParams(
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
			kernel_size = 5,
			distance_feature_stats = None,
			pool_size = 2,
			conv_hidden_sizes = [6, 16],
			linear_hidden_sizes = [120, 84],
			x_dim = None,
			y_dim = None,
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

	best_model, _ = CNNModel.load(save_location)
	best_model.set_link_data(links_list, met_data)

	batch = val_batches[1]
	fwd = best_model.forward_batch(batch.receptors)
	for i in range(5):
		print((best_model.params.transform_output_inv(fwd[i].item(), batch.nearest_link_distances[i].item()), best_model.params.transform_output_inv(batch.y[i].item(), batch.nearest_link_distances[i].item())))

	print(sum([best_model.get_error(batch.receptors, batch.y) for batch in val_batches])/len(val_batches))

	best_model.graph_prediction_error(val_batches)

	best_model.export_predictions(val_batches, save_location[save_location.rindex('model_save_') + 11:save_location.rindex('.')] + ' Predictions.csv')

	best_model.print_batch_errors(val_batches)