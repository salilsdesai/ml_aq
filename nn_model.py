import torch

from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Tuple, Dict, Optional, Any

from .model import Model, ReceptorBatch, Params
from .utils import Receptor, Link, Coordinate, Features, DEVICE, MetStation, \
	lambda_to_string, A, B, TRANSFORM_OUTPUT, TRANSFORM_OUTPUT_INV, \
	Paths

class NNLinkData():
	def __init__(self, coords: Tensor, coords_dot: Tensor, subtract: Tensor, keep: Tensor):
		self.coords = coords
		self.coords_dot = coords_dot
		self.subtract = subtract
		self.keep = keep

class NNReceptorData():
	def __init__(self, coords: Tensor, coords_dot: Tensor, subtract: Tensor):
		self.coords = coords
		self.coords_dot = coords_dot
		self.subtract = subtract

class NNReceptorBatch(ReceptorBatch):
	def __init__(
		self, 
		receptors: NNReceptorData, 
		y: Tensor, 
		nearest_link_distances: Tensor
	):
		ReceptorBatch.__init__(self, receptors, y, nearest_link_distances)
	
	def coordinate(self, i: int) -> Coordinate:
		"""
		returns the coordinates of the receptor at index i in the batch
		"""
		return Coordinate(
			self.receptors.coords[i][0].cpu(), 
			self.receptors.coords[i][1].cpu(),
		)

	def size(self) -> int:
		return self.receptors.coords.shape[0]

class NNParams(Params):
	def __init__(
		self,
		batch_size: int,
		transform_output_src: str,
		transform_output_inv_src: str,
		concentration_threshold: float,
		distance_threshold: float,
		link_features: List[str],
		hidden_size: int,
		subtract_features: List[str],
	):
		Params.__init__(
			self,
			batch_size=batch_size,
			transform_output_src=transform_output_src,
			transform_output_inv_src=transform_output_inv_src,
			concentration_threshold=concentration_threshold,
			distance_threshold=distance_threshold,
			link_features=link_features
		)
		self.hidden_size: int = hidden_size
		self.subtract_features: List[str] = subtract_features
	
	@staticmethod
	def from_dict(d: Dict[str, Any]) -> 'NNParams':
		return NNParams(
			batch_size = d['batch_size'],
			transform_output_src = d['transform_output_src'],
			transform_output_inv_src =d['transform_output_inv_src'],
			concentration_threshold = d['concentration_threshold'],
			distance_threshold = d['distance_threshold'],
			link_features = d['link_features'],
			hidden_size = d['hidden_size'],
			subtract_features = d['subtract_features'],
		)
	
	def as_dict(self) -> Dict[str, Any]:
		"""
		Override
		"""
		d = super().as_dict()
		d.update({
			'hidden_size': self.hidden_size,
			'subtract_features': self.subtract_features,
		})
		return d

class NNModel(Model):
	def __init__(self, params: NNParams):
		Model.__init__(self, params)
		self.params: NNParams = params
		self.feature_stats: Optional[Features.FeatureStats] = None
		self.input_size = 1 + len(self.params.subtract_features) + len(self.params.link_features)
		self.layer_1 = torch.nn.Linear(self.input_size, params.hidden_size).to(DEVICE)
		self.activ = torch.nn.Sigmoid().to(DEVICE)
		self.layer_2 = torch.nn.Linear(self.params.hidden_size, 1).to(DEVICE)

	def forward(self, x: Tensor):
		s1 = self.layer_1(x)
		a1 = self.activ(s1)
		s2 = self.layer_2(a1)
		return s2.abs()

	def make_x(self, receptors: NNReceptorData) -> Tuple[Tensor, Tensor]:
		if self.feature_stats is None:
			raise Exception('Feature Stats Not Set')
		distances_inv = (self.link_data.coords_dot - (2 * (receptors.coords @ self.link_data.coords)) + receptors.coords_dot).sqrt().reciprocal().unsqueeze(dim=2).type(Tensor).to(DEVICE)
		subtracted = self.link_data.subtract - receptors.subtract
		kept = self.link_data.keep
		cat = torch.cat((distances_inv, subtracted, kept), dim=-1)
		normalized = (cat - self.feature_stats.mean)/self.feature_stats.std_dev
		filter = distances_inv > (1/self.params.distance_threshold)
		return (normalized, filter)
	
	def make_receptor_batches(self, receptors: List[List[Receptor]]) -> List[ReceptorBatch]:
		def make_batch(receptors_list: List[Receptor]) -> ReceptorBatch:
			coords = torch.DoubleTensor([[r.x, r.y] for r in receptors_list]).to(DEVICE)
			data = NNReceptorData(
				coords, 
				(coords * coords).sum(dim=1).unsqueeze(dim=-1), 
				Tensor([[[
					Features.GET_FEATURE_DIFFERENCE_RECEPTOR_DATA[f](r) \
						for f in self.params.subtract_features
				]] for r in receptors_list]).to(DEVICE)
			)
			return NNReceptorBatch(
				data,
				Tensor([[self.params.transform_output(r.pollution_concentration, r.nearest_link_distance)] for r in receptors_list]).to(DEVICE),
				Tensor([[r.nearest_link_distance] for r in receptors_list]).to(DEVICE)
			)
		return [make_batch(receptors_list) for receptors_list in receptors]
		
	def set_link_data(self, links: List[Link], met_data: Dict[int, MetStation]) -> None:
		coords = torch.DoubleTensor([[l.x, l.y] for l in links]).T.to(DEVICE)
		coords_dot = (coords * coords).sum(dim=0)
		subtract = Tensor([[
			Features.GET_FEATURE_DIFFERENCE_LINK_DATA[f](l) \
				for f in self.params.subtract_features
		] for l in links]).to(DEVICE)
		keep = Tensor([[[
			Features.GET_FEATURE[f](link, met_data) \
				for f in self.params.link_features
		] for link in links]]).to(DEVICE).repeat(self.params.batch_size, 1, 1)
		# If too much memory or need multiple batch sizes, move repeat to make_x (distances.shape[0])
		self.link_data = NNLinkData(coords, coords_dot, subtract, keep)

	def set_feature_stats(self, all_feature_stats: Dict[str, Features.FeatureStats]) -> None:
		stats = \
			[all_feature_stats['distance_inverse']] + \
			[all_feature_stats[sf] for sf in self.params.subtract_features] + \
			[all_feature_stats[lf] for lf in self.params.link_features]

		self.feature_stats = Features.FeatureStats(
			Tensor([s.mean for s in stats]).to(DEVICE),
			Tensor([s.std_dev for s in stats]).to(DEVICE),
		)

	def forward_batch(self, receptors: NNReceptorData) -> Tensor:
		"""
		Override
		"""
		(x, filter) = self.make_x(receptors)
		return (self.forward(x) * filter).sum(dim=1)

	@staticmethod
	def load(filepath: str) -> Tuple['NNModel', Optimizer]:
		return Model.load(filepath, NNModel, NNParams)
	
	def prep_experiment(self, directory: str) -> None:
		"""
		Override
		"""
		feature_stats = Features.get_all_feature_stats(Paths.feature_stats(directory))
		self.set_feature_stats(feature_stats)
	
	@staticmethod
	def run_experiment(params: NNParams, show_results: bool) -> Tuple['Model', Optimizer, str, Dict[str, float], List[ReceptorBatch], List[ReceptorBatch]]:
		return Model.run_experiment(
			base_class = NNModel, 
			params = params, 
			show_results = show_results,
		)

if __name__ == '__main__':
	_ = NNModel.run_experiment(
		params = NNParams(
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
			hidden_size = 8,
			subtract_features = [Features.ELEVATION_DIFFERENCE],
		),
		show_results = True,
	)
