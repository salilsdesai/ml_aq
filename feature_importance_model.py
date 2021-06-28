import torch

from captum.attr import IntegratedGradients
from matplotlib import pyplot as plt
from torch import Tensor
from torch.optim import Optimizer
from torch.types import Number
from typing import List, Tuple

from .model import Model
from .nn_model import NNModel, NNParams, NNReceptorData
from .utils import Features

class FeatureImportanceModel(NNModel):
	def __init__(self, params: NNParams):
		params.batch_size = 20  # Large batch sizes cause memory issues
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
	nn_model_save_location = 'TODO: Fill in'

	importances = FeatureImportanceModel.run_feature_importance(nn_model_save_location)
	for l in importances:
		print(l)

	plt.barh(range(len(importances)), list(map(lambda t: t[1], importances)), tick_label=list(map(lambda t: t[0], importances)))
	plt.title('Integrated Gradients Scores')
	plt.show()

	plt.barh(range(len(importances)), list(map(lambda t: abs(t[1]), importances)), tick_label=list(map(lambda t: t[0], importances)))
	plt.title('Integrated Gradients Scores (Absolute Value)')
	plt.show()