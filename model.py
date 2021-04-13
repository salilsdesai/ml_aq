import numpy as np
import torch

from functools import reduce
from matplotlib import pyplot as plt
from matplotlib import colors, ticker
from time import time, strftime, localtime
from torch import Tensor
from torch.optim import Optimizer
from torch.types import Number
from typing import Callable, List, Tuple, Optional, Dict, Any, TypeVar, Generic

from .utils import MetStation, mre, loss_function, error_function, Value, \
	graph_error_function, mult_factor_error, Link, Receptor, Coordinate, \
	flatten

ReceptorData = TypeVar('ReceptorData')
LinkData = TypeVar('LinkData')

class ReceptorBatch(Generic[ReceptorData]):
	def __init__(
		self, 
		receptors: ReceptorData, 
		y: Tensor, 
		nearest_link_distances: Tensor
	):
		self.receptors: ReceptorData = receptors
		self.y: Tensor = y
		self.nearest_link_distances: Tensor = nearest_link_distances
	
	def coordinate(self, i: int) -> Coordinate:
		"""
		returns the coordinates of the receptor at index i in the batch
		"""
		raise NotImplementedError

	def size(self) -> int:
		raise NotImplementedError


class ModelParams():
	def __init__(
		self,
		hidden_size: int,
		batch_size: int,
		transform_output_src: str,
		transform_output_inv_src: str,
		concentration_threshold: float,
		distance_threshold: float,
		link_features: List[str]
	):
		self.hidden_size: int = hidden_size
		self.batch_size: int = batch_size
		self.transform_output_src: str = transform_output_src
		transform_output: Callable[[Value, Value], Value] = eval(transform_output_src)
		self.transform_output: Callable[[Value, Value], Value] = transform_output
		self.transform_output_inv_src: str = transform_output_inv_src
		transform_output_inv: Callable[[Value, Value], Value] = eval(transform_output_inv_src)
		self.transform_output_inv: Callable[[Value, Value], Value] = transform_output_inv
		self.concentration_threshold = concentration_threshold
		self.distance_threshold = distance_threshold
		self.link_features = link_features
	
	def as_dict(self) -> Dict[str, Any]:
		d = {
			'hidden_size': self.hidden_size,
			'batch_size': self.batch_size,
			'transform_output_src': self.transform_output_src,
			'transform_output_inv_src': self.transform_output_inv_src,
			'concentration_threshold': self.concentration_threshold,
			'distance_threshold': self.distance_threshold,
			'link_features': self.link_features
		}
		d.update(self.child_dict())
		return d

	def child_dict(self) -> Dict[str, Any]:
		raise NotImplementedError


class Model(torch.nn.Module, Generic[LinkData, ReceptorData]):
	def __init__(self, params: ModelParams):	
		super(Model, self).__init__()
		self.link_data: Optional[LinkData] = None
		self.params = params
	
	def filter_receptors(self, receptors: List[Receptor]) -> List[Receptor]:
		"""
		Remove receptors with negligible concentration or no nearby links
		"""
		return [r for r in receptors if (r.nearest_link_distance <= self.params.distance_threshold and r.pollution_concentration >= self.params.concentration_threshold)]
	
	def make_receptor_batches(self, receptors: List[List[Receptor]]) -> ReceptorBatch:
		raise NotImplementedError 

	def forward_batch(self, receptors: ReceptorData) -> Tensor:
		raise NotImplementedError

	def set_link_data(self, links: List[Link], met_data: Dict[int, MetStation]) -> None:
		raise NotImplementedError
		
	def get_error(self, receptors: ReceptorData, y: Tensor) -> Number:
		"""
		[y] is Tensor of corresponding true pollution concentrations
		"""
		y_hat = self.forward_batch(receptors)
		return error_function(y_hat, y).item()

	def print_batch_errors(self, batches: List[ReceptorBatch]) -> None:
		err_funcs = [
			('Mult Factor', mult_factor_error),
			('MSE', torch.nn.MSELoss()),
			('MRE', mre),
			('MAE', torch.nn.L1Loss()),
		]
		errors = [0] * len(err_funcs)
		for batch in batches:
			nld_cpu = batch.nearest_link_distances.cpu()
			y_final = self.params.transform_output_inv(batch.y.cpu(), nld_cpu)
			y_hat_final = self.params.transform_output_inv(
				self.forward_batch(batch.receptors).cpu(), 
				nld_cpu
			)
			for i in range(len(err_funcs)):
				errors[i] += err_funcs[i][1](y_hat_final, y_final).item() / len(batches)
			y_hat_final = None  # Prevents memory leak
		print('Final Errors:')
		for i in range(len(err_funcs)):
			print(err_funcs[i][0] + ': ' + str(errors[i]))

	def save(self, filepath: str, optimizer: Optimizer) -> None:
		torch.save({
			'model_params': self.params.as_dict(),
			'model_state_dict': self.state_dict(),
			'optimizer_class': optimizer.__class__,
			'optimizer_state_dict': optimizer.state_dict(),
			'time': strftime('%m-%d-%y %H:%M:%S', localtime(time())),
		}, filepath)
	
	@staticmethod
	def load(filepath: str, base_class, params_class) -> Tuple[Any, Optimizer]:
		loaded = torch.load(filepath)
		model = base_class(params_class.from_dict(loaded['model_params']))
		model.load_state_dict(loaded['model_state_dict'])
		optimizer = loaded['optimizer_class'](model.parameters())
		optimizer.load_state_dict(loaded['optimizer_state_dict'])
		return (model, optimizer)

	def train(
		self, 
		optimizer: Optimizer, 
		num_epochs: int,
		train_batches: List[ReceptorBatch], 
		val_batches: List[ReceptorBatch], 
		save_location: Optional[str], 
		make_graphs: bool
	):
		if self.link_data is None:
			raise Exception('Link Data Not Set')

		start_time = time()

		losses = []
		val_errors = []

		def get_stats(loss):
			losses.append(loss / len(train_batches))
			print('Loss: ' + str(losses[-1]))
			val_errors.append(sum([self.get_error(batch.receptors, batch.y) for batch in val_batches])/len(val_batches))
			print('Val Error: ' + str(val_errors[-1]))

		get_stats(sum([loss_function(self.forward_batch(batch.receptors), batch.y).item() for batch in train_batches]))
		
		if save_location is not None:
			self.save(save_location, optimizer)
			print('Saved Model!')

		curr_epoch = 0
		stop_training = False
		while not stop_training:
			epoch_loss = 0
			for batch in train_batches:
				optimizer.zero_grad()
				y_hat = self.forward_batch(batch.receptors)
				loss = loss_function(y_hat, batch.y)
				epoch_loss += float(loss.item())
				loss.backward()
				optimizer.step()
			print('---------------------------------------------')
			print('Finished Epoch ' + str(curr_epoch) + ' (' + str(time() - start_time) + ' seconds)')
			get_stats(epoch_loss)
			min_error = min(val_errors)
			if save_location is not None and val_errors[-1] == min_error:
				self.save(save_location, optimizer)
				print('Saved Model!')
			curr_epoch += 1
			# Stop when we've surpassed num_epochs or we haven't improved upon the min val error for three iterations
			stop_training = (curr_epoch >= num_epochs) or ((curr_epoch >= 3) and (val_errors[-1] > min_error) and (val_errors[-2] > min_error) and (val_errors[-3] > min_error))
		
		if make_graphs:
			titles = iter(('Loss', 'Val Error'))
			for l in (losses, val_errors):
				plt.plot(range(len(l)), l)
				plt.title(next(titles))
				plt.show()

	def graph_prediction_error(self, batches: List[ReceptorBatch]) -> None:
		class Prediction():
			def __init__(
				self, 
				nearest_link_distance: Number, 
				coordinate: Coordinate, 
				predicted_value: Value, 
				actual_value: Value,
			):
				self.nearest_link_distance: Number = nearest_link_distance
				self.coordinate: Coordinate = coordinate
				self.predicted_value: Any = predicted_value
				self.actual_value: Any = actual_value
				self.graph_error: Any = graph_error_function(
					y_hat = predicted_value, 
					y = actual_value,
				)
			
		def predict_batch(batch: ReceptorBatch) -> List[Prediction]:
			"""
			returns list of [(nearest link distance, coordinates, graph error)]
			for each receptor in the batch
			"""
			fwd = self.forward_batch(batch.receptors).detach()
			return [Prediction(
				nearest_link_distance = batch.nearest_link_distances[i].item(), 
				coordinate = batch.coordinate(i),
				predicted_value = self.params.transform_output_inv(
					fwd[i].item(), 
					batch.nearest_link_distances[i].item(),
				),
				actual_value = self.params.transform_output_inv(
					batch.y[i].item(), 
					batch.nearest_link_distances[i].item()
				),
			) for i in range(batch.size())]

		predictions: List[Prediction] = flatten([predict_batch(batch) for batch in batches])

		cutoff = 0.05
		original_size = len(predictions)

		# Scatter plot

		# Remove upper ~5% of error predictions (absolute value) (outliers)
		predictions.sort(key=lambda prediction: abs(prediction.graph_error))
		predictions = [predictions[i] for i in range(int(original_size*(1-cutoff)))]
		print('Upper ' + (str(100 * (original_size - len(predictions)) / len(predictions)) + "	 ")[:5] + "% of predictions removed as outliers before drawing plot")

		predictions.sort(key=lambda prediction: prediction.nearest_link_distance)
		X = [p.nearest_link_distance for p in predictions]
		Y = [p.graph_error for p in predictions]
		plt.scatter(X, Y, s=1)
		plt.show()
		plt.yscale('symlog')
		plt.scatter(X, Y, s=1)
		plt.show()

		# Plot average error across bins of NLD
		
		bin_size = 5
		predictions.sort(key=lambda prediction: prediction.nearest_link_distance)

		plot_functions: List[Callable[[Prediction], float]] = [
			lambda p: p.graph_error,
			lambda p: p.predicted_value,
			lambda p: p.actual_value,
		]

		# ((Total error, receptor count) for each bin) for each plot function
		bins: List[List[List[float]]] = [[[0, 0] for _ in range(int(predictions[-1].nearest_link_distance / bin_size) + 1)] for _ in range(len(plot_functions))]

		for p in predictions:
			for i in range(len(plot_functions)):
				j = int(p.nearest_link_distance / bin_size)
				bins[i][j][0] += plot_functions[i](p)
				bins[i][j][1] += 1
		
		non_empty_js = [[j for j in range(len(bins[i])) if bins[i][j][1] > 0] for i in range(len(plot_functions))]
		X = [[bin_size * j for j in non_empty_js[i]] for i in range(len(plot_functions))]
		Y = [[bins[i][j][0]/bins[i][j][1] for j in non_empty_js[i]] for i in range(len(plot_functions))]
		
		plt.title('Average Error vs Nearest Link Distance')
		plt.plot(X[0], Y[0])
		plt.show()

		plt.title('Predicted and Actual Concentrations vs Nearest Link Distance')
		plt.plot(X[1], Y[1], label = 'Predicted')
		plt.plot(X[2], Y[2], label = 'Actual')
		plt.legend()
		plt.show()

		# Map

		# Remove upper and lower ~5% of error predictions (absolute value) (outliers)
		predictions.sort(key=lambda prediction: abs(prediction.graph_error))
		predictions = [predictions[i] for i in range(int(original_size*cutoff), len(predictions))]
		print('Upper and lower combined ' + (str(100 * (original_size - len(predictions)) / len(predictions)) + "	 ")[:5] + "% of predictions removed as outliers before drawing map")

		plt.figure(figsize=(6,9))
		most_extreme, least_extreme = reduce(lambda m, p: (max(abs(p.graph_error), m[0]), min(abs(p.graph_error), m[1])), predictions, (0, 1000000))

		# These do nothing for now, but potentially use log and exp respectively if errors not distributed well over color range
		transform = lambda x: x
		transform_inv = lambda x: x

		max_transformed, min_transformed = transform(most_extreme), transform(least_extreme)

		positive_light = np.asarray(colors.to_rgb('#E0ECFF'))
		positive_dark = np.asarray(colors.to_rgb('#0064FF'))
		negative_light = np.asarray(colors.to_rgb('#FFE0E0'))
		negative_dark = np.asarray(colors.to_rgb('#FF0000'))

		def get_color(x):
			(sign, light, dark) = (1, positive_light, positive_dark) if x > 0 else (-1, negative_light, negative_dark)
			proportion = (transform(sign * x) - min_transformed)/(max_transformed - min_transformed)
			return ((1 - proportion) * light) + (proportion * dark)

		plt.scatter([p.coordinate.x for p in predictions], [p.coordinate.y for p in predictions], s=1, c=[get_color(p.graph_error) for p in predictions])

		plt.show()

		# Gradient Key
		n = 500
		plt.xscale('symlog')
		ax = plt.axes()

		r = lambda x: round(x * 100) / 100  # Round x to the hundredths place

		ax.set_xticks([-r(most_extreme/3), -r(most_extreme*2/3), -r(most_extreme), r(most_extreme/3), r(most_extreme*2/3), r(most_extreme), 0])
		ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
		for i in range(-n, n+1):
			x = transform_inv(min_transformed + ((abs(i)/n) * (max_transformed - min_transformed))) * (1 if i > 0 else -1)
			ax.axvline(x, c=get_color(x), linewidth=4)

	def export_predictions(self, batches: List[ReceptorBatch], filepath: str) -> None:
		def predict_batch(batch: ReceptorBatch) -> List[
				Tuple[float, float, Value, Value, Value]
			]:
			"""
			returns list of [(x, y, prediction, actual, graph error)]
			for each receptor in the batch
			"""
			fwd = self.forward_batch(batch.receptors).detach()
			def make_tuple(i: int) -> Tuple[float, float, Value, Value, Value]:
				prediction = self.params.transform_output_inv(
					fwd[i].item(), 
					batch.nearest_link_distances[i].item()
				)
				actual = self.params.transform_output_inv(
					batch.y[i].item(), 
					batch.nearest_link_distances[i].item()
				)
				coord = batch.coordinate(i)
				return (
					coord.x,
					coord.y,
					prediction,
					actual,
					graph_error_function(prediction, actual)
				)

			return [make_tuple(i) for i in range(batch.size())]
		
		predictions = flatten([predict_batch(batch) for batch in batches])
		
		out_file = open(filepath, 'w')
		format = lambda l: \
			str(l).replace('\'', '').replace(', ', ',')[1:-1] + '\n'
		out_file.write(format(['x', 'y', 'prediction', 'actual', 'error']))
		for prediction in predictions:
			out_file.write(format(prediction))
		out_file.close()