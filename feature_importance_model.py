import torch

from captum.attr import IntegratedGradients
from torch import Tensor
from torch.optim import Optimizer
from torch.types import Number
from typing import List, Tuple

from .model import Model
from .nn_model import NNModel, NNParams, NNReceptorData
from .utils import Features

class FeatureImportanceModel(NNModel):
	def __init__(self, params: NNParams):
		super(FeatureImportanceModel, self).__init__(params)
	
	def forward(self, x: Tensor):
		"""
		Override
		"""
		return super(FeatureImportanceModel, self).forward(x).sum(dim=1)
	
	def forward_batch(self, receptors: NNReceptorData) -> Tensor:
		raise Exception(
			'Do not call forward_batch with FeatureImportanceModel. \
			call forward directly on normalized tensor created by make_x.'
		)
	
	@staticmethod
	def load(filepath: str) -> Tuple['FeatureImportanceModel', Optimizer]:
		return Model.load_with_classes(filepath, FeatureImportanceModel, NNParams)

	@staticmethod
	def run_feature_importance(nn_model_save_location: str) -> List[Tuple[str, Number]]:
		model, _ = FeatureImportanceModel.load(nn_model_save_location)
		_, val_batches, _ = model.quick_setup()

		ig = IntegratedGradients(model)
		attributions_list = []
		for batch in val_batches:
			attr = ig.attribute(
				inputs = model.make_x(batch.receptors)[0], 
				target = 0, 
				return_convergence_delta = True
			)[0]
			with torch.no_grad():
				attributions_list.append(attr.sum(dim=1))
		
		importances = torch.cat(tensors=attributions_list, dim=0).mean(dim=0)
		feature_names = ['distance_inverse' if model.params.invert_distance else 'distance'] + model.params.subtract_features + model.params.link_features + model.params.receptor_features
		if len(feature_names) != importances.shape[0]:
			feature_names = ['?' for _ in range(importances.shape[0])]
		return [(feature_names[i], importances[i].item()) for i in range(importances.shape[0])]
		

if __name__ == '__main__':
	_, _, _, nn_model_save_location = NNModel.run_experiment(
		params = NNParams(
			batch_size = 20,
			transform_output_src = 'lambda y, nld: y',
			transform_output_inv_src = 'lambda y, nld: y',
			concentration_threshold = 0.01,
			distance_threshold = 500,
			link_features = [
				Features.VMT, Features.TRAFFIC_SPEED, Features.FLEET_MIX_LIGHT,
				Features.FLEET_MIX_MEDIUM, Features.FLEET_MIX_HEAVY,
				Features.FLEET_MIX_COMMERCIAL, Features.FLEET_MIX_BUS,
				Features.WIND_SPEED, Features.UP_DOWN_WIND_EFFECT,
				Features.POPULATION_DENSITY, Features.ELEVATION_MEAN,
				Features.NEAREST_MET_STATION_DISTANCE, Features.TEMPERATURE,
				Features.RELATIVE_HUMIDITY,
			],
			receptor_features = [
				Features.NEAREST_LINK_DISTANCE,
			],
			hidden_size = 8,
			subtract_features = [
				Features.ELEVATION_DIFFERENCE,
			],
			invert_distance = False,
		),
		make_optimizer = lambda m: torch.optim.AdamW(m.parameters(), lr=0.001),
		show_results = True,
	)
	importances = FeatureImportanceModel.run_feature_importance(nn_model_save_location)
	for l in importances:
		print(l)