import torch

from torch import Tensor
from torch.nn.functional import relu
from torch.optim.optimizer import Optimizer
from typing import List, Tuple, Dict, Optional, Any, Callable

from .conv_model import ConvModel, ConvParams, ConvReceptorData
from .model import Model, ReceptorBatch
from .utils import Features, DEVICE, lambda_to_string, A, B, TRANSFORM_OUTPUT, \
	TRANSFORM_OUTPUT_INV

class CNNParams(ConvParams):
	def __init__(
		self,
		batch_size: int,
		transform_output_src: str,
		transform_output_inv_src: str,
		concentration_threshold: float,
		distance_threshold: float,
		link_features: List[str],
		receptor_features: List[str],
		subtract_features: List[str],
		approx_bin_size: float,
		kernel_size: int,
		distance_feature_stats: Optional[Features.FeatureStats],
		receptor_feature_stats: Optional[Features.FeatureStats],
		subtract_feature_stats: Optional[Features.FeatureStats],
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
			receptor_features=receptor_features,
			subtract_features=subtract_features,
			approx_bin_size=approx_bin_size,
			kernel_size=kernel_size,
			distance_feature_stats=distance_feature_stats,
			receptor_feature_stats=receptor_feature_stats,
			subtract_feature_stats=subtract_feature_stats,
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
			receptor_features = d['receptor_features'],
			subtract_features = d['subtract_features'],
			approx_bin_size = d['approx_bin_size'],
			kernel_size = d['kernel_size'],
			distance_feature_stats = Features.FeatureStats.deserialize(d['distance_feature_stats']),
			receptor_feature_stats = Features.FeatureStats.deserialize(d['receptor_feature_stats']),
			subtract_feature_stats = Features.FeatureStats.deserialize(d['subtract_feature_stats']),
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
	
	
	def make_receptor_data(self, distances: Tensor, keep: Tensor, subtract: Tensor) -> ConvReceptorData:
		"""
		Override
		"""
		return ConvReceptorData(
			distances=distances.unsqueeze(dim=1),
			keep=keep,
			subtract=subtract,
		)
	
	def set_up_subtract(self, subtract: Tensor) -> Tensor:
		"""
		Override
		"""
		return subtract


	def forward_batch(self, receptors: ConvReceptorData) -> Tensor:
		"""
		Override
		"""
		subtract = self.link_data.subtract - receptors.subtract
		# TODO: Track "Subtract" stats properly (this way just assumes there's at most one)
		if len(self.params.subtract_features) > 0:
			subtract = (subtract - self.params.subtract_feature_stats.mean) / self.params.subtract_feature_stats.std_dev
		
		return self.forward(
			torch.cat(
				tensors = (
					receptors.distances, 
					self.link_data.channels,
					receptors.keep.repeat(1, 1, self.link_data.bin_counts.shape[0], self.link_data.bin_counts.shape[1]),
					subtract,
				), 
				dim = 1,
			)
		)
	
	@staticmethod
	def load(filepath: str) -> Tuple['CNNModel', Optimizer]:
		return Model.load_with_classes(filepath, CNNModel, CNNParams)

	@staticmethod
	def run_experiment(
		params: CNNParams, 
		make_optimizer: Callable[[torch.nn.Module], Optimizer], 
		show_results: bool
	) -> Tuple['Model', List[ReceptorBatch], Dict[str, float], str]:
		return Model.run_experiment(
			base_class = CNNModel, 
			params = params, 
			make_optimizer = make_optimizer,
			show_results = show_results,
		)

if __name__ == '__main__':
	_ = CNNModel.run_experiment(
		params = CNNParams(
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
			receptor_features = [
				Features.NEAREST_LINK_DISTANCE,
			],
			subtract_features = [
				Features.ELEVATION_DIFFERENCE,
			],
			approx_bin_size = 1000,
			kernel_size = 5,
			distance_feature_stats = None,
			receptor_feature_stats = None,
			subtract_feature_stats = None,
			pool_size = 2,
			conv_hidden_sizes = [6, 16],
			linear_hidden_sizes = [120, 84],
			x_dim = None,
			y_dim = None,
		),
		make_optimizer = lambda m: torch.optim.AdamW(m.parameters(), lr=0.0001),
		show_results = True,
	)