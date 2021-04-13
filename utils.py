import gc
import inspect
import numpy as np
import os
import pandas as pd
import torch
import sys

from functools import reduce
from math import sin, pi
from operator import iconcat
from torch import Tensor, is_tensor
from torch.types import Number
from typing import Any, Dict, List, Union, Callable, Tuple

DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

Value = Union[Tensor, Number]

MSE_SUM = torch.nn.MSELoss(reduction='sum')

def mre(y_hat: Tensor, y: Tensor) -> Tensor:
	return ((y_hat - y) / y).abs().mean()

def loss_function(y_hat: Tensor, y: Tensor) -> Tensor:
	return MSE_SUM(y_hat, y)/2

error_function = mre

# Relative Error without absolute value
def graph_error_function(y_hat, y) -> Value:
	return ((y_hat - y) / y) 

def mult_factor_error(y_hat: Tensor, y: Tensor) -> Value:
	return (torch.max(y_hat, y) / torch.min(y_hat, y)).mean()

A, B = (0.8, 0.10297473583824722) # TODO: Move this
TRANSFORM_OUTPUT = lambda y, nld: y * (np.exp(B * (nld ** 0.5)) / A)
TRANSFORM_OUTPUT_INV = lambda y, nld: y / (np.exp(B * (nld ** 0.5)) / A)

def lambda_to_string(f: Callable, replacements: List[Tuple[str, str]]) -> str:
	src = inspect.getsourcelines(f)[0][0]
	src = src[src.index('lambda'):]
	src = src.replace('\n', '')
	for (before, after) in replacements:
		src = src.replace(before, after)
	return src

OUTLIER_LINK_ID: int = 246543  # This one is way off in the distance

TIME_PERIODS: List[str] = ['_morning', '_midday', '_pm', '_nd']

class Link():

	TEMPORAL_FIELDS: List[str] = [
		'traffic_speed', 'traffic_flow', 'fleet_mix_light', 'fleet_mix_medium',
		'fleet_mix_heavy', 'fleet_mix_commercial', 'fleet_mix_bus'
	]

	def __init__(self, data: Dict[str, Any]):
		self.id: int = data['id']
		self.x: float = data['x']
		self.y: float = data['y']
		self.nearest_met_station_id: int = data['nearest_met_station_id']
		self.nearest_met_station_distance: float = \
			data['nearest_met_station_distance']
		self.nearest_met_station_angle: float = \
			data['nearest_met_station_angle']
		self.elevation_mean: float = data['elevation_mean']
		self.traffic_speed: float = data['traffic_speed']
		self.traffic_flow: float = data['traffic_flow']
		self.link_length: float = data['link_length']
		self.fleet_mix_light: float = data['fleet_mix_light']
		self.fleet_mix_medium: float = data['fleet_mix_medium']
		self.fleet_mix_heavy: float = data['fleet_mix_heavy']
		self.fleet_mix_commercial: float = data['fleet_mix_commercial']
		self.fleet_mix_bus: float = data['fleet_mix_bus']
		self.angle: float = data['angle']
		self.population_density: float = data['population_density']
		for tp in TIME_PERIODS:
			for tf in Link.TEMPORAL_FIELDS:
				attr = tf + tp 
				setattr(self, attr, data[attr])

	@staticmethod
	def load_links(filepath: str) -> List['Link']:
		return [l for l in construct_from_csv(Link, filepath) if l.id != OUTLIER_LINK_ID]
		

class Receptor():
	def __init__(self, data):
		self.id: int = data['id']
		self.x: float = data['x']
		self.y: float = data['y']
		self.elevation: float = data['elevation']
		self.pollution_concentration: float = data['pollution_concentration']
		self.nearest_link_distance: float = data['nearest_link_distance']
	
	@staticmethod
	def load_receptors(filepath: str) -> List['Receptor']:
		return construct_from_csv(Receptor, filepath)

class MetStation():
	def __init__(self, data):
		self.id: int = data['id']
		self.name: str = data['name']
		self.latitude: float = data['latitude']
		self.longitude: float = data['longitude']
		self.wind_direction: float = data['wind_direction']
		self.wind_speed: float = data['wind_speed']
		self.temperature: float = data['temperature']
		self.sensible_heat_flux: float = data['sensible_heat_flux']
		self.surface_friction_velocity: float = \
			data['surface_friction_velocity']
		self.convective_velocity_scale: float = \
			data['convective_velocity_scale']
		self.potential_temperature_gradient_above_the_mixing_height: float = \
			data['potential_temperature_gradient_above_the_mixing_height']
		self.convectively_driven_mixing_height: float = \
			data['convectively_driven_mixing_height']
		self.mechanically_driven_mixing_height: float = \
			data['mechanically_driven_mixing_height']
		self.monin_obukhov_length: float = data['monin_obukhov_length']
		self.surface_roughness_length: float = data['surface_roughness_length']
		self.bowen_ratio: float = data['bowen_ratio']
		self.albedo: float = data['albedo']
		self.wind_speed_stage_3: float = data['wind_speed_stage_3']
		self.wind_direction_stage_3: float = data['wind_direction_stage_3']
		self.temperature_stage_3: float = data['temperature_stage_3']
		self.precipitation_type: float = data['precipitation_type']
		self.precipitation_amount: float = data['precipitation_amount']
		self.relative_humidity: float = data['relative_humidity']
		self.station_pressure: float = data['station_pressure']
		self.cloud_cover: float = data['cloud_cover']
		self.wind_direction_morning: float = data['wind_direction_morning']
		self.wind_speed_morning: float = data['wind_speed_morning']
		self.wind_direction_midday: float = data['wind_direction_midday']
		self.wind_speed_midday: float = data['wind_speed_midday']
		self.wind_direction_pm: float = data['wind_direction_pm']
		self.wind_speed_pm: float = data['wind_speed_pm']
		self.wind_direction_nd: float = data['wind_direction_nd']
		self.wind_speed_nd: float = data['wind_speed_nd']
	
	@staticmethod
	def load(filepath: str) -> List['MetStation']:
		return construct_from_csv(MetStation, filepath)
	
	@staticmethod
	def load_met_data(filepath: str) -> Dict[int, 'MetStation']:
		return {s.id: s for s in MetStation.load(filepath)}

def construct_from_csv(constructor, filepath: str) -> list:
	df = pd.read_csv(filepath)
	return [constructor(dict(row[1])) for row in df.iterrows()]

class Coordinate():
	def __init__(self, x: float, y: float):
		self.x = x
		self.y = y

class Features():
	
	VMT = 'vmt'
	TRAFFIC_SPEED = 'traffic_speed'
	FLEET_MIX_LIGHT = 'fleet_mix_light'
	FLEET_MIX_MEDIUM = 'fleet_mix_medium'
	FLEET_MIX_HEAVY = 'fleet_mix_heavy'
	FLEET_MIX_COMMERCIAL = 'fleet_mix_commercial'
	FLEET_MIX_BUS = 'fleet_mix_bus'
	WIND_DIRECTION = 'wind_direction'
	WIND_SPEED = 'wind_speed'
	UP_DOWN_WIND_EFFECT = 'up_down_wind_effect'
	
	ELEVATION_DIFFERENCE = 'elevation_difference'

	GET_FEATURE: Dict[str, Callable[[Link, Dict[int, MetStation]], float]] = {
		VMT: lambda link, met_data: link.traffic_flow * link.link_length,
		TRAFFIC_SPEED: lambda link, met_data: link.traffic_speed,
		FLEET_MIX_LIGHT: lambda link, met_data: link.fleet_mix_light,
		FLEET_MIX_MEDIUM: lambda link, met_data: link.fleet_mix_medium,
		FLEET_MIX_HEAVY: lambda link, met_data: link.fleet_mix_heavy,
		FLEET_MIX_COMMERCIAL: lambda link, met_data: link.fleet_mix_commercial,
		FLEET_MIX_BUS: lambda link, met_data: link.fleet_mix_bus, 
		WIND_DIRECTION: lambda link, met_data: met_data[link.nearest_met_station_id].wind_direction,
		WIND_SPEED: lambda link, met_data: met_data[link.nearest_met_station_id].wind_speed,
		UP_DOWN_WIND_EFFECT: lambda link, met_data: abs(sin((met_data[link.nearest_met_station_id].wind_direction - link.angle) * pi / 180)),
	}

	GET_FEATURE_DIFFERENCE_LINK_DATA: Dict[str, Callable[[Link], float]] = {
		ELEVATION_DIFFERENCE: (lambda link: link.elevation_mean)
	}

	GET_FEATURE_DIFFERENCE_RECEPTOR_DATA: Dict[str, Callable[[Receptor], float]] = {
		ELEVATION_DIFFERENCE: (lambda receptor: receptor.elevation)
	}

	GET_FEATURE_WITH_SUFFIX: Dict[str, Callable[[Link, str, Dict[int, MetStation]], float]] = {
		VMT: lambda link, suffix, met_data: getattr(link, 'traffic_flow' + suffix) * link.link_length,
		TRAFFIC_SPEED: lambda link, suffix, met_data: getattr(link, Features.TRAFFIC_SPEED + suffix),
		FLEET_MIX_LIGHT: lambda link, suffix, met_data: getattr(link, Features.FLEET_MIX_LIGHT + suffix),
		FLEET_MIX_MEDIUM: lambda link, suffix, met_data: getattr(link, Features.FLEET_MIX_MEDIUM + suffix),
		FLEET_MIX_HEAVY: lambda link, suffix, met_data: getattr(link, Features.FLEET_MIX_HEAVY + suffix),
		FLEET_MIX_COMMERCIAL: lambda link, suffix, met_data: getattr(link, Features.FLEET_MIX_COMMERCIAL + suffix),
		FLEET_MIX_BUS: lambda link, suffix, met_data: getattr(link, Features.FLEET_MIX_BUS + suffix), 
		WIND_DIRECTION: lambda link, suffix, met_data: getattr(met_data[link.nearest_met_station_id], Features.WIND_DIRECTION + suffix),
		WIND_SPEED: lambda link, suffix, met_data: getattr(met_data[link.nearest_met_station_id], Features.WIND_SPEED + suffix),
		UP_DOWN_WIND_EFFECT: lambda link, suffix, met_data: abs(sin((getattr(met_data[link.nearest_met_station_id], Features.WIND_DIRECTION + suffix) - link.angle) * pi / 180)),
	}

	class FeatureStats():
		def __init__(self, mean, std_dev):
			self.mean = mean
			self.std_dev = std_dev

	@staticmethod
	def get_all_feature_stats(filepath: str) -> Dict[str, FeatureStats]:
		"""
		returns {Feature Name: (Mean, Std Dev)}
		"""
		df = pd.read_csv(filepath)
		return {
			r[1]['feature']: Features.FeatureStats(
				r[1]['mean'], 
				r[1]['std_dev']
			) for r in df.iterrows()
		}
			
def partition(l: List, size: int) -> List[List]:
	return [l[i:i + size] for i in range(0, len(l) - size, size)]

def train_val_split(l: List) -> Tuple[List, List]:
	train = [l[i] for i in range(len(l)) if (i % 5) < 3] # 60%
	val = [l[i] for i in range(len(l)) if (i % 5) == 3] # 20%
	return (train, val)

def flatten(l: List[List[Any]]) -> List[Any]:
	return reduce(iconcat, l, [])

def get_memory_usage() -> Tuple[Dict[Tuple[str, str], int], int]:
	"""
	Returns a tuple of 
	- Dict mapping from (type, size) -> count of number of objects
	- Total number of values stored in tensors
	"""
	
	d: Dict[Tuple[str, str], int] = {}
	count: int = 0
	for obj in gc.get_objects():
		try:
			# Suppress deprecation warning
			sys.stderr = open(os.devnull, 'w')
			is_tensor = torch.is_tensor(obj)
			sys.stderr = sys.__stderr__

			if (is_tensor or (hasattr(obj, 'data') and torch.is_tensor(obj.data))) and (DEVICE in str(obj.device)):
				type_string = str(type(obj))
				obj_type = type_string[type_string.find("torch.") + 6:-2]
				size_list = list(obj.size())
				tup = (obj_type, str(size_list))
				count += reduce(lambda x, y: x * y, size_list, 1)

				if tup not in d:
					d[tup] = 1
				else:
					d[tup] += 1
		except:
			pass
	return (d, count)