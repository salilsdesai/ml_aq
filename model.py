import numpy as np
import torch

from functools import reduce
from matplotlib import pyplot as plt
from matplotlib import colors, ticker
from scipy.stats import norm, kstest
from time import time, strftime, localtime
from torch import Tensor
from torch.optim import Optimizer
from torch.types import Number
from typing import Callable, List, Tuple, Optional, Dict, Any, TypeVar, Generic, Type

from .utils import CumulativeStats, MetStation, mre, loss_function, error_function, Value, \
	graph_error_function, mult_factor_error, Link, Receptor, Coordinate, \
	flatten, Paths, NOTEBOOK_NAME, BASE_DIRECTORY, train_val_test_split, \
	partition

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


class Params():
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
	):
		self.batch_size: int = batch_size
		self.transform_output_src: str = transform_output_src
		transform_output: Callable[[Value, Value], Value] = eval(transform_output_src)
		self.transform_output: Callable[[Value, Value], Value] = transform_output
		self.transform_output_inv_src: str = transform_output_inv_src
		transform_output_inv: Callable[[Value, Value], Value] = eval(transform_output_inv_src)
		self.transform_output_inv: Callable[[Value, Value], Value] = transform_output_inv
		self.concentration_threshold: float = concentration_threshold
		self.distance_threshold: float = distance_threshold
		self.link_features: List[str] = link_features
		self.receptor_features: List[str] = receptor_features
		self.subtract_features: List[str] = subtract_features
	
	def as_dict(self) -> Dict[str, Any]:
		"""
		Used to serialize the parameters when saving to a file
		"""
		return {
			'batch_size': self.batch_size,
			'transform_output_src': self.transform_output_src,
			'transform_output_inv_src': self.transform_output_inv_src,
			'concentration_threshold': self.concentration_threshold,
			'distance_threshold': self.distance_threshold,
			'link_features': self.link_features,
			'receptor_features': self.receptor_features,
			'subtract_features': self.subtract_features,
		}


class Model(torch.nn.Module, Generic[LinkData, ReceptorData]):
	def __init__(self, params: Params):	
		super(Model, self).__init__()
		self.link_data: Optional[LinkData] = None
		self.params = params
	
	def filter_receptors(self, receptors: List[Receptor]) -> List[Receptor]:
		"""
		Remove receptors with negligible concentration or no nearby links
		"""
		return [r for r in receptors if (r.nearest_link_distance <= self.params.distance_threshold and r.pollution_concentration >= self.params.concentration_threshold)]
	
	def make_receptor_batches(self, receptors: List[List[Receptor]]) -> List[ReceptorBatch]:
		"""
		Prepare lists of receptors into batches to be passed into the model
		"""
		raise NotImplementedError 

	def forward_batch(self, receptors: ReceptorData) -> Tensor:
		"""
		Pass a batch of receptors into the model for prediction
		"""
		raise NotImplementedError

	def set_link_data(self, links: List[Link], met_data: Dict[int, MetStation]) -> None:
		"""
		Configure the model to use the provided links and meteorological data
		for prediction
		"""
		raise NotImplementedError
		
	def get_error(self, receptors: ReceptorData, y: Tensor) -> Number:
		"""
		[y] is Tensor of corresponding true pollution concentrations
		"""
		y_hat = self.forward_batch(receptors)
		return error_function(y_hat, y).item()

	def get_errors(self, batches: List[ReceptorBatch]) -> Dict[str, float]:
		"""
		Compute various metrics, as well as variance and modeling time.
		Return a dict with keys as names of the metric and values as the values
		"""	
		err_funcs = [
			('Mult Factor', mult_factor_error),
			('MSE', torch.nn.MSELoss()),
			('MRE', mre),
			('MAE', torch.nn.L1Loss()),
		]
		errors = [0] * len(err_funcs)
		variance_tracker = CumulativeStats()
		avg_modeling_time = 0
		with torch.no_grad():
			for batch in batches:
				y_final = self.params.transform_output_inv(batch.y, batch.nearest_link_distances).cpu()
				start_time = time()
				y_hat_final = self.params.transform_output_inv(
					self.forward_batch(batch.receptors), 
					batch.nearest_link_distances
				)
				end_time = time()
				avg_modeling_time += (end_time - start_time) / len(batches)
				y_hat_final = y_hat_final.cpu()
				for i in range(len(err_funcs)):
					errors[i] += err_funcs[i][1](y_hat_final, y_final).item() / len(batches)
				variance_tracker.update(y_hat_final)
		errors_dict: Dict[str, float] = {err_funcs[i][0]: errors[i] for i in range(len(err_funcs))}
		errors_dict.update({
			'Variance': float(variance_tracker.stddev.item() ** 2),
			'Average Modeling Time Per Batch': avg_modeling_time,
		})
		return errors_dict

	def save(self, filepath: str, optimizer: Optimizer) -> None:
		"""
		Save a model and optimizer to file to load for continued training/
		prediction in the future
		"""
		torch.save({
			'model_params': self.params.as_dict(),
			'model_state_dict': self.state_dict(),
			'optimizer_class': optimizer.__class__,
			'optimizer_state_dict': optimizer.state_dict(),
			'time': strftime('%m-%d-%y %H:%M:%S', localtime(time())),
		}, filepath)
	
	@staticmethod
	def load_with_classes(filepath: str, base_class, params_class) -> Tuple[Any, Optimizer]:
		"""
		Helper function used to load model/optimizer from file
		"""
		loaded = torch.load(filepath)
		model = base_class(params_class.from_dict(loaded['model_params']))
		model.load_state_dict(loaded['model_state_dict'])
		optimizer = loaded['optimizer_class'](model.parameters())
		optimizer.load_state_dict(loaded['optimizer_state_dict'])
		return (model, optimizer)
	
	@staticmethod
	def load(filepath: str) -> Tuple[Any, Optimizer]:
		"""
		Load a model/optimizer from file
		"""
		raise NotImplementedError

	def train(
		self, 
		optimizer: Optimizer, 
		num_epochs: int,
		train_batches: List[ReceptorBatch], 
		val_batches: List[ReceptorBatch], 
		save_location: Optional[str], 
		make_graphs: bool,
		print_results: bool,
	):
		"""
		Train a model. Training ends when num_epochs epochs pass or when 
		validation error does not decrease for three consecutive epochs. Save
		best model after each iteration a file if save_location is not None.
		print_results and make_graphs can be set to print errors during training
		and show graphs of error during the training process.
		"""
		if self.link_data is None:
			raise Exception('Link Data Not Set')

		start_time = time()

		losses = []
		val_errors = []
	
		def get_stats(loss):
			losses.append(loss / len(train_batches))
			if print_results:
				print('Loss: ' + str(losses[-1]))
			val_errors.append(sum([self.get_error(batch.receptors, batch.y) for batch in val_batches])/len(val_batches))
			if print_results:
				print('Val Error: ' + str(val_errors[-1]))
		
		with torch.no_grad():
			get_stats(sum([loss_function(self.forward_batch(batch.receptors), batch.y).item() for batch in train_batches]))
		
		if save_location is not None:
			self.save(save_location, optimizer)
			if print_results:
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
			if print_results:
				print('---------------------------------------------')
				print('Finished Epoch ' + str(curr_epoch) + ' (' + str(time() - start_time) + ' seconds)')
			get_stats(epoch_loss)
			min_error = min(val_errors)
			if save_location is not None and val_errors[-1] == min_error:
				self.save(save_location, optimizer)
				if print_results:
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
		"""
		Display graphs which show the model's performance on a group of batches
		- 	Scatter plot of graph error vs nearest link distance, on linear and 
			logarithmic scales
		- 	Average graph error vs nearest link distance (average taken over 
			bins of size 5 meters)
		- 	Average Predicted and Actual concentrations vs nearest link distance
			on the same plot
		-	Map of NYC with receptors plotted on it various colors, where darker
			red corresponds to more intense underprediction, and darker blue 
			more intense overprediction
		"""
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

		with torch.no_grad():
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
			returns list of [(x, y, prediction, actual, MRE)]
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
					mre(Tensor([prediction]), Tensor([actual])).item()
				)

			return [make_tuple(i) for i in range(batch.size())]
		
		with torch.no_grad():
			predictions = flatten([predict_batch(batch) for batch in batches])
		
		out_file = open(filepath, 'w')
		format = lambda l: \
			str(l).replace('\'', '').replace(', ', ',')[1:-1] + '\n'
		out_file.write(format(['x', 'y', 'prediction', 'actual', 'error']))
		for prediction in predictions:
			out_file.write(format(prediction))
		out_file.close()
	
	def analyze_absolute_error(self, batches: List[ReceptorBatch]):
		"""
		Runs Kolmogorov-Smirnov test to measure how closely distribution of 
		absolute error (ŷ - y) follows normal distribution
		"""
		def get_absolute_errors(batch: ReceptorBatch) -> List[float]:
			y_final = self.params.transform_output_inv(batch.y, batch.nearest_link_distances).cpu()
			y_hat_final = self.params.transform_output_inv(
				self.forward_batch(batch.receptors), 
				batch.nearest_link_distances
			).cpu()
			return (y_hat_final - y_final).tolist()

		absolute_errors = None
		with torch.no_grad():
			absolute_errors = np.asarray(flatten([get_absolute_errors(b) for b in batches]))
		
		print('Mean: ' + str(np.mean(absolute_errors)))
		print('Standard Deviation: ' + str(np.std(absolute_errors)))
		print('Kolmogorov-Smirnov Test Result: ' + str(kstest(absolute_errors, norm.cdf)))

		plt.hist(absolute_errors, bins=1000)
		plt.title('Distribution of Absolute Errors')
		plt.show()

		plt.hist(absolute_errors, bins=1000, range=[-0.5, 0.5])
		plt.title('Distribution of Absolute Errors')
		plt.show()

	def prep_experiment(self, directory: str) -> None:
		"""
		Subclass specific model setup which must be done before running
		experiments
		"""
		raise NotImplementedError

	def quick_setup(self) -> Tuple[List[ReceptorBatch], List[ReceptorBatch], List[ReceptorBatch]]:
		"""
		Set up model for training/prediction using default data 
		locations/directory after initializing it
		Returns tuple of (Train Batches, Val Batches, Test Batches)
		"""
		links_list = Link.load_links(Paths.link_data(BASE_DIRECTORY))
		met_data = MetStation.load_met_data(Paths.met_data(BASE_DIRECTORY))
		
		self.set_link_data(links_list, met_data)

		self.prep_experiment(BASE_DIRECTORY)

		receptors_list = Receptor.load_receptors(Paths.receptor_data(BASE_DIRECTORY))
		train_receptors_list, val_receptors_list, test_receptors_list = train_val_test_split(receptors_list)

		train_batches = self.make_receptor_batches(partition(self.filter_receptors(train_receptors_list), self.params.batch_size))
		val_batches = self.make_receptor_batches(partition(self.filter_receptors(val_receptors_list), self.params.batch_size))
		test_batches = self.make_receptor_batches(partition(self.filter_receptors(test_receptors_list), self.params.batch_size))

		return (train_batches, val_batches, test_batches)

	@staticmethod
	def run_experiment(
		base_class: Type['Model'], 
		params: Params,
		make_optimizer: Callable[[torch.nn.Module], Optimizer], 
		show_results: bool
	) -> Tuple['Model', List[ReceptorBatch], Dict[str, float], str]:
		"""
		Run an experiment by training and evaluating a model with provided 
		parameters. Uses quick_setup function to get data
		Returns
		- The best model
		- Test receptor batches
		- Dict mapping error function name to error value
		- Model save location
		"""
		
		model = base_class(params)
		(train_batches, val_batches, test_batches) = model.quick_setup()

		save_location = Paths.save(BASE_DIRECTORY, NOTEBOOK_NAME)

		if show_results:
			print('Save Location: ' + str(save_location))
		
		model.train(
			optimizer = make_optimizer(model),
			num_epochs = 1000,
			train_batches = train_batches,
			val_batches = val_batches,
			save_location = save_location,
			make_graphs = True,
			print_results = show_results,
		)

		model_optim: Tuple[Model, Optimizer] = base_class.load(save_location)
		model: Model = model_optim[0]
		_, _, test_batches = model.quick_setup()

		errors = model.get_errors(test_batches)

		if show_results:
			model.graph_prediction_error(test_batches)
			for (k, v) in errors.items():
				print(k + ': ' + str(v))
		
		return (model, test_batches, errors, save_location)

	def predict(self, receptors: List[Receptor]) -> List[float]:
		"""
		Use the model to predict pollution concentration for a group of
		receptors
		This combines make_receptor_batches, forward_batch, and 
		transform_ouput_inv into a single function which is easy to use
		"""
		bs = self.params.batch_size
		
		# Duplicate the last element in the list until its length is a multiple
		# of batch size
		recep = receptors + [receptors[-1]] * ((bs - (len(receptors) % bs)) % bs)

		batches = self.make_receptor_batches(partition(recep, bs))

		predictions = None

		with torch.no_grad():
			predictions = flatten(list(map(
				lambda batch: (
					self.params.transform_output_inv(
						self.forward_batch(batch.receptors), 
						batch.nearest_link_distances,
					).cpu().tolist()
				),
				batches
			)))		

		return predictions[:len(receptors)]
