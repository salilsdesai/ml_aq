import torch

from torch import Tensor
from typing import List, Tuple, Dict, Optional, Any

from .model import Model, ReceptorBatch, Params
from .utils import Receptor, Link, Coordinate, Features, DEVICE, MetStation, \
	partition, flatten, CumulativeStats

class ConvLinkData():
	def __init__(self, channels: Tensor, bin_counts: Tensor, bin_centers: Tensor, subtract: Tensor):
		self.channels: Tensor = channels
		self.bin_counts: Tensor = bin_counts
		self.bin_centers: Tensor = bin_centers
		self.subtract: Tensor = subtract

class ConvReceptorData():
	def __init__(self, distances: Tensor, keep: Tensor, subtract: Tensor):
		self.distances: Tensor = distances
		self.keep: Tensor = keep
		self.subtract: Tensor = subtract

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
	):
		Params.__init__(
			self,
			batch_size=batch_size,
			transform_output_src=transform_output_src,
			transform_output_inv_src=transform_output_inv_src,
			concentration_threshold=concentration_threshold,
			distance_threshold=distance_threshold,
			link_features=link_features,
			receptor_features=receptor_features,
			subtract_features=subtract_features,
		)
		self.approx_bin_size: float = approx_bin_size
		self.kernel_size: int = kernel_size
		self.distance_feature_stats: Optional[Features.FeatureStats] = \
			distance_feature_stats 
		self.receptor_feature_stats: Optional[Features.FeatureStats] = \
			receptor_feature_stats 
		self.subtract_feature_stats: Optional[Features.FeatureStats] = \
			subtract_feature_stats 
	
	@staticmethod
	def from_dict(d: Dict[str, Any]) -> 'ConvParams':
		return ConvParams(
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
		)
	
	def as_dict(self) -> Dict[str, Any]:
		"""
		Override
		"""
		d = super().as_dict()
		d.update({
			'approx_bin_size': self.approx_bin_size,
			'kernel_size': self.kernel_size,
			'distance_feature_stats': Features.FeatureStats.serialize(self.distance_feature_stats),
			'receptor_feature_stats': Features.FeatureStats.serialize(self.receptor_feature_stats),
			'subtract_feature_stats': Features.FeatureStats.serialize(self.subtract_feature_stats),
		})
		return d

class ConvModel(Model):
	def __init__(self, params: ConvParams):
		Model.__init__(self, params)
		self.params: ConvParams = params
		self.input_size: int = 1 + len(self.params.link_features) + len(self.params.receptor_features) + len(self.params.subtract_features) # +1 for distance

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

		# TODO: Track "Keep" and "Subtract" stats properly (this way just assumes there's at most one)
		distance_cs, keep_cs, subtract_cs = (CumulativeStats(), CumulativeStats(), CumulativeStats()) \
			if self.params.distance_feature_stats is None \
			else (None, None, None)

		def make_batch(receptors_list: List[Receptor]) -> ReceptorBatch:
			coords_list, distances = self.get_receptor_locations(receptors_list)

			keep = Tensor([[
				Features.GET_RECEPTOR_FEATURE[f](r) \
					for f in self.params.receptor_features
			] for r in receptors_list]).to(DEVICE)  # num receptors x num features
			keep = keep.unsqueeze(dim=2).unsqueeze(dim=3)

			subtract = Tensor([[
				Features.GET_FEATURE_DIFFERENCE_RECEPTOR_DATA[f](r) \
					for f in self.params.subtract_features
			] for r in receptors_list]).to(DEVICE) # num receptors x num features
			subtract = subtract.unsqueeze(dim=2).unsqueeze(dim=3) 

			data = self.make_receptor_data(distances=distances, keep=keep, subtract=subtract)

			y = Tensor([[self.params.transform_output(r.pollution_concentration, r.nearest_link_distance)] for r in receptors_list]).to(DEVICE)
			nearest_link_distances = Tensor([[r.nearest_link_distance] for r in receptors_list]).to(DEVICE)
			coordinates = [Coordinate(c[0], c[1]) for c in coords_list]

			if distance_cs is not None:
				distance_cs.update(data.distances)

			# TODO: Track "Keep" and "Subtract" stats properly (this way just assumes there's at most one)
			if keep_cs is not None and len(self.params.receptor_features) > 0:
				keep_cs.update(keep)
			if subtract_cs is not None and len(self.params.subtract_features) > 0:
				subtract_cs.update(self.link_data.subtract - data.subtract)

			return ConvReceptorBatch(
				receptors = data,
				y = y,
				nearest_link_distances = nearest_link_distances,
				coordinates = coordinates
			)
		
		batches = [make_batch(receptors_list) for receptors_list in receptors]

		if distance_cs is not None:
			self.params.distance_feature_stats = Features.FeatureStats(
				mean = distance_cs.mean.item(),
				std_dev = distance_cs.stddev.item(),
			)
		
		# TODO: Track "Keep" and "Subtract" stats properly (this way just assumes there's at most one)
		if keep_cs is not None:
			self.params.receptor_feature_stats = Features.FeatureStats(
				mean = keep_cs.mean.item(),
				std_dev = keep_cs.stddev.item(),
			)
		if subtract_cs is not None:
			self.params.subtract_feature_stats = Features.FeatureStats(
				mean = subtract_cs.mean.item(),
				std_dev = subtract_cs.stddev.item()
			)
		

		for batch in batches:
			batch.receptors.distances = \
				(batch.receptors.distances - self.params.distance_feature_stats.mean) / (self.params.distance_feature_stats.std_dev)
			
			# TODO: Track "Keep" stats properly (this way just assumes there's at most one)
			if len(self.params.receptor_features) > 0:
				batch.receptors.keep = \
					(batch.receptors.keep - self.params.receptor_feature_stats.mean) / self.params.receptor_feature_stats.std_dev

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

		subtract = torch.zeros(len(self.params.subtract_features), num_x, num_y).to(DEVICE)

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
			for i in range(subtract.shape[0]):
				subtract[i][bin_x][bin_y] += Features.GET_FEATURE_DIFFERENCE_LINK_DATA[self.params.subtract_features[i]](link)
			counts[bin_x][bin_y] += 1
		
		for bin_x in range(num_x):
			for bin_y in range(num_y):
				if counts[bin_x][bin_y] != 0:
					for i in range(channels.shape[0]):
						for j in range(channels.shape[1]):
							channels[i][j][bin_x][bin_y] /= counts[bin_x][bin_y]
					for i in range(subtract.shape[0]):
						subtract[i][bin_x][bin_y] /= counts[bin_x][bin_y]

		mean = channels.mean(dim=(0, 2, 3), keepdim=True)
		std = channels.std(dim=(0, 2, 3), keepdim=True)

		channels = ((channels - mean) / std).unsqueeze(dim=0).repeat(self.params.batch_size, 1, 1, 1, 1)

		x_centers = Tensor([min_x + ((i + 0.5) * size_x) for i in range(num_x)]).to(DEVICE)
		y_centers = Tensor([min_y + ((j + 0.5) * size_y) for j in range(num_y)]).to(DEVICE)
		centers = torch.cartesian_prod(x_centers, y_centers).reshape(x_centers.shape[0], y_centers.shape[0], 2).unsqueeze(dim=0)

		subtract = subtract.unsqueeze(0).repeat(self.params.batch_size, 1, 1, 1)

		self.link_data = ConvLinkData(
			channels = self.set_up_on_channel_dims(channels),
			bin_counts = counts,
			bin_centers = centers,
			subtract = self.set_up_subtract(subtract),
		)
	
	def prep_experiment(self, directory: str) -> None:
		pass  # Don't need to do anything
	
	def set_up_on_channel_dims(self, channels: Tensor) -> Tensor:
		"""
		Set up the model based on dimensions of convolutional channels.
		Implemented by subclasses
		"""
		raise NotImplementedError
	
	def make_receptor_data(self, distances: Tensor, keep: Tensor, subtract: Tensor) -> ConvReceptorData:
		"""
		Distances is (# receptors) x (num bins x) x (num bins y)
		Keep is (# receptors) x (# channels) x (1) x (1)
		Subtract is (# receptors) x (# channels) x (1) x (1)
		"""
		raise NotImplementedError()
	
	def get_time_periods(self) -> List[str]:
		"""
		Get a list of time periods which data passed into the model is divided
		across. Used to set up convolutional channels
		"""
		raise NotImplementedError()
	
	def set_up_subtract(self, subtract: Tensor) -> Tensor:
		raise NotImplementedError()