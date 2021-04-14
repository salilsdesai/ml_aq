import torch

from torch import Tensor
from typing import List, Tuple, Dict, Optional, Any

from .model import Model, ReceptorBatch, Params
from .utils import Receptor, Link, Coordinate, Features, DEVICE, MetStation, \
	partition, flatten, CumulativeStats

class ConvLinkData():
	def __init__(self, channels: Tensor, bin_counts: Tensor, bin_centers: Tensor):
		self.channels: Tensor = channels
		self.bin_counts: Tensor = bin_counts
		self.bin_centers: Tensor = bin_centers

class ConvReceptorData():
	def __init__(self, distances: Tensor):
		self.distances: Tensor = distances

class ConvReceptorBatch(ReceptorBatch):
	def __init__(
		self, 
		receptors: ConvReceptorData, 
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

class ConvParams(Params):
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
		distance_feature_stats: Optional[Features.FeatureStats]
	):
		Params.__init__(
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
		self.time_periods: List[str] = time_periods
		self.distance_feature_stats: Optional[Features.FeatureStats] = \
			distance_feature_stats 
	
	@staticmethod
	def from_dict(d: Dict[str, Any]) -> 'ConvParams':
		return ConvParams(
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
		)
	
	def child_dict(self) -> Dict[str, Any]:
		return {
			'approx_bin_size': self.approx_bin_size,
			'kernel_size': self.kernel_size,
			'time_periods': self.time_periods,
			'distances_feature_stats': (
				self.distance_feature_stats.mean, 
				self.distance_feature_stats.std_dev
			) if self.distance_feature_stats is not None else None,
		}

class ConvModel(Model):
	def __init__(self, params: ConvParams):
		Model.__init__(self, params)
		self.params: ConvParams = params
		self.input_size: int = 1 + len(self.params.link_features)  # +1 for distance
		self.approx_bin_size: float = params.approx_bin_size
		self.kernel_size: int = params.kernel_size

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
		"""
		Override
		"""
		if self.link_data is None:
			raise Exception('Link Data Not Set')

		if len(receptors) == 0:
			return []

		cumulative_stats = CumulativeStats() if self.params.distance_feature_stats is None else None

		def make_batch(receptors_list: List[Receptor]) -> ReceptorBatch:
			coords_list, distances = self.get_receptor_locations(receptors_list)

			data = self.make_receptor_data(distances)
			y = Tensor([[self.params.transform_output(r.pollution_concentration, r.nearest_link_distance)] for r in receptors_list]).to(DEVICE)
			nearest_link_distances = Tensor([[r.nearest_link_distance] for r in receptors_list]).to(DEVICE)
			coordinates = [Coordinate(c[0], c[1]) for c in coords_list]

			if cumulative_stats is not None:
				cumulative_stats.update(distances)

			return ConvReceptorBatch(
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

		time_periods = self.get_time_periods()

		channels = torch.zeros(
			len(time_periods), 
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
						](link, time_periods[i], met_data)
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

		self.link_data = ConvLinkData(
			channels = self.fix_channel_dims(channels),
			bin_counts = counts,
			bin_centers = centers,
		)
	
	def fix_channel_dims(self, channels: Tensor) -> Tensor:
		raise NotImplementedError
	
	def make_receptor_data(self, distances: Tensor) -> ConvReceptorData:
		raise NotImplementedError()
	
	def get_time_periods(self) -> List[str]:
		raise NotImplementedError()