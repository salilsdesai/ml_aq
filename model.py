import torch
import pandas as pd
from time import time, strftime, localtime
from random import shuffle
from matplotlib import pyplot as plt
from matplotlib import patches, colors, ticker
from functools import reduce
from operator import iconcat
from math import log, log10, exp
import numpy as np
from requests import get

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE != 'cuda':
	print('WARNING: Not using GPU')

MSE_SUM = torch.nn.MSELoss(reduction='sum')
MRE = lambda y_hat, y: ((y_hat - y) / y).abs().mean()

HIDDEN_SIZE = 8
BATCH_SIZE = 1000

LOSS_FUNCTION = lambda y_hat, y: MSE_SUM(y_hat, y)/2
ERROR_FUNCTION = MRE
GRAPH_ERROR_FUNCTION = lambda y_hat, y: ((y_hat - y) / y) # Relative Error without absolute value

TRANSFORM_OUTPUT = lambda y, nld: y
TRANSFORM_OUTPUT_INV = lambda y, nld: y

CONCENTRATION_THRESHOLD = 0.01
DISTANCE_THRESHOLD = 500
LEARNING_RATE = 0.001

NOTEBOOK_NAME = 'model.py'
DIRECTORY = '.'

# Used to load means and std devs for feature normilization
# Must be in the same order as the features in the tensor passed into the model
FEATURES = [
	'distance_inverse', 'elevation_difference', 'vmt', 'traffic_speed', 
	'fleet_mix_light', 'fleet_mix_medium', 'fleet_mix_heavy', 
	'fleet_mix_commercial', 'fleet_mix_bus', 'wind_direction', 'wind_speed',
]
INPUT_SIZE = len(FEATURES)

def load_feature_stats(filepath):
	"""
	Load the mean and std dev of each feature from a file
	Used to normalize data
	"""
	feature_stats_dict = {(row[1]['feature']):(type('FeatureStats', (object,), dict(row[1]))) for row in pd.read_csv(filepath).iterrows()}
	feature_stats = torch.Tensor(2, INPUT_SIZE)
	for i in range(INPUT_SIZE):
		(feature_stats[0][i], feature_stats[1][i]) = (lambda f: (f.mean, f.std_dev))(feature_stats_dict[FEATURES[i]])
	return feature_stats.to(DEVICE)

FEATURE_STATS = load_feature_stats(DIRECTORY + '/data/feature_stats.csv')
MET_DATA = {(int(row[1]['id'])):(type('MetStation', (object,), dict(row[1]))) for row in pd.read_csv(DIRECTORY + '/data/met_data.csv').iterrows()}

def extract_list(filepath, class_name=''):
	df = pd.read_csv(filepath)
	return [type(class_name, (object,), dict(row[1])) for row in df.iterrows()]

def prep_links(links=None):
	if links is None:
		links = extract_list(DIRECTORY + '/data/link_data.csv', 'link')
	coords = torch.DoubleTensor([[l.x, l.y] for l in links]).T.to(DEVICE)
	coords_dot = (coords * coords).sum(dim=0)
	subtract = torch.Tensor([[l.elevation_mean] for l in links]).to(DEVICE)
	keep = torch.Tensor([[[
		link.traffic_flow * link.link_length,
		link.traffic_speed,
		link.fleet_mix_light,
		link.fleet_mix_medium,
		link.fleet_mix_heavy,
		link.fleet_mix_commercial,
		link.fleet_mix_bus,
		MET_DATA[link.nearest_met_station_id].wind_direction,
		MET_DATA[link.nearest_met_station_id].wind_speed,
	] for link in links]]).to(DEVICE).repeat(BATCH_SIZE, 1, 1)
	# If too much memory or need multiple batch sizes, move repeat to make_x (distances.shape[0])
	return (coords, coords_dot, subtract, keep)

def prep_receptors(receptors=None):
	if receptors is None:
		receptors = extract_list(DIRECTORY + '/data/receptor_data.csv', 'receptor')
	coords = torch.DoubleTensor([[r.x, r.y] for r in receptors]).to(DEVICE)
	coords_dot = (coords * coords).sum(dim=1).unsqueeze(dim=-1)
	subtract = torch.Tensor([[[r.elevation]] for r in receptors]).to(DEVICE)
	return (coords, coords_dot, subtract)
  
def prep_batch(receptors):
	prepped = prep_receptors(receptors)
	y = torch.Tensor([[TRANSFORM_OUTPUT(r.pollution_concentration, r.nearest_link_distance)] for r in receptors]).to(DEVICE)
	nearest_link_distances = torch.Tensor([[r.nearest_link_distance] for r in receptors]).to(DEVICE)
	return (prepped, y, nearest_link_distances)

class Model(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.layer_1 = torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZE).to(DEVICE)
		self.activ = torch.nn.Sigmoid().to(DEVICE)
		self.layer_2 = torch.nn.Linear(HIDDEN_SIZE, 1).to(DEVICE)

	def forward(self, x):
		s1 = self.layer_1(x)
		a1 = self.activ(s1)
		s2 = self.layer_2(a1)
		return s2.abs()

	@staticmethod
	def make_x(links, receptors):
		distances_inv = (links[1] - (2 * (receptors[0] @ links[0])) + receptors[1]).sqrt().reciprocal().unsqueeze(dim=2).type(torch.Tensor).to(DEVICE)
		subtracted = links[2] - receptors[2]
		kept = links[3]
		cat = torch.cat((distances_inv, subtracted, kept), dim=-1)
		normalized = (cat - FEATURE_STATS[0])/FEATURE_STATS[1]
		filter = distances_inv > (1/DISTANCE_THRESHOLD)
		return (normalized, filter)

	def forward_batch(self, links, receptors):
		(x, filter) = Model.make_x(links, receptors)
		return (self.forward(x) * filter).sum(dim=1)

	def get_error(self, links, receptors, y):
		"""
		[y] is Tensor of corresponding true pollution concentrations
		"""
		y_hat = self.forward_batch(links, receptors)
		return ERROR_FUNCTION(y_hat, y).item()

	def print_batch_errors(self, links, batches):
		err_funcs = [
			('Mult Factor', lambda y_hat, y: (torch.max(y_hat, y) / torch.min(y_hat, y)).mean()),
			('MSE', torch.nn.MSELoss()),
			('MRE', MRE),
			('MAE', torch.nn.L1Loss()),
		]

		errors = [0] * len(err_funcs)
		for (receptors, y, nld) in batches:
			y_final = TRANSFORM_OUTPUT_INV(y, nld)
			y_hat_final = TRANSFORM_OUTPUT_INV(self.forward_batch(links, receptors), nld)
			for i in range(len(err_funcs)):
				errors[i] += err_funcs[i][1](y_hat_final, y_final).item() / len(batches)
		print('Final Errors:')
		for i in range(len(err_funcs)):
			print(err_funcs[i][0] + ': ' + str(errors[i]))

	def save(self, filepath, optimizer=None):
		torch.save({
			'model_state_dict': self.state_dict(),
			'optimizer_class': optimizer.__class__,
			'optimizer_state_dict': optimizer.state_dict(),
			'time': strftime('%m-%d-%y %H:%M:%S', localtime(time())),
		}, filepath)
	
	@staticmethod
	def load(filepath):
		loaded = torch.load(filepath)
		model = Model()
		model.load_state_dict(loaded['model_state_dict'])
		optimizer = loaded['optimizer_class'](model.parameters())
		optimizer.load_state_dict(loaded['optimizer_state_dict'])
		return (model, optimizer)

	def train(
		self, 
		optimizer, 
		num_epochs, 
		links, 
		train_batches, 
		val_batches, 
		save_location=None, 
		make_graphs=False
	):
		start_time = time()

		losses = []
		val_errors = []

		def get_stats(loss):
			losses.append(loss / len(train_batches))
			print('Loss: ' + str(losses[-1]))
			val_errors.append(sum([self.get_error(links, receptors, y) for (receptors, y, _) in val_batches])/len(val_batches))
			print('Val Error: ' + str(val_errors[-1]))

		get_stats(sum([LOSS_FUNCTION(self.forward_batch(links, receptors), y).item() for (receptors, y, _) in train_batches]))
		
		self.save(save_location, optimizer)
		print('Saved Model!')

		curr_epoch = 0
		stop_training = False
		while not stop_training:
			epoch_loss = 0
			for (receptors, y, _) in train_batches:
				optimizer.zero_grad()
				y_hat = self.forward_batch(links, receptors)
				loss = LOSS_FUNCTION(y_hat, y)
				epoch_loss += loss.item()
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

	def graph_prediction_error(self, links, batches):
		def predict_batch(batch):
			"""
			returns list of [(nearest link distance, coordinates, graph error)]
			for each receptor in the batch
			"""
			(receptors, y, nld) = batch
			fwd = self.forward_batch(links, receptors).detach()
			return [(nld[i].item(), tuple(receptors[0][i].tolist()), GRAPH_ERROR_FUNCTION(TRANSFORM_OUTPUT_INV(fwd[i].item(), nld[i].item()), TRANSFORM_OUTPUT_INV(y[i].item(), nld[i].item()))) for i in range(receptors[0].shape[0])]

		predictions = reduce(iconcat, [predict_batch(batch) for batch in batches], [])

		cutoff = 0.05
		original_size = len(predictions)

		# Scatter plot

		# Remove upper ~5% of error predictions (absolute value) (outliers)
		predictions.sort(key=lambda prediction: abs(prediction[2]))
		predictions = [predictions[i] for i in range(int(original_size*(1-cutoff)))]
		print('Upper ' + (str(100 * (original_size - len(predictions)) / len(predictions)) + "	 ")[:5] + "% of predictions removed as outliers before drawing plot")

		predictions.sort(key=lambda prediction: prediction[0])
		X = [p[0] for p in predictions]
		Y = [p[2] for p in predictions]
		plt.scatter(X, Y, s=1)
		plt.show()
		plt.yscale('symlog')
		plt.scatter(X, Y, s=1)
		plt.show()

		# Map

		# Remove upper and lower ~5% of error predictions (absolute value) (outliers)
		predictions.sort(key=lambda prediction: abs(prediction[2]))
		predictions = [predictions[i] for i in range(int(original_size*cutoff), len(predictions))]
		print('Upper and lower combined ' + (str(100 * (original_size - len(predictions)) / len(predictions)) + "	 ")[:5] + "% of predictions removed as outliers before drawing map")

		plt.figure(figsize=(6,9))
		most_extreme, least_extreme = reduce(lambda m, p: (max(abs(p[2]), m[0]), min(abs(p[2]), m[1])), predictions, (0, 1000000))

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

		plt.scatter([p[1][0] for p in predictions], [p[1][1] for p in predictions], s=1, c=[get_color(p[2]) for p in predictions])

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

	def export_predictions(self, links, batches, filepath):
		def predict_batch(batch):
			"""
			returns list of [(x, y, prediction, actual, graph error)]
			for each receptor in the batch
			"""
			(receptors, y, nld) = batch
			fwd = self.forward_batch(links, receptors).detach()
			def make_tuple(i):
				prediction = TRANSFORM_OUTPUT_INV(fwd[i].item(), nld[i].item())
				actual = TRANSFORM_OUTPUT_INV(y[i].item(), nld[i].item())
				return (receptors[0][i][0].item(), receptors[0][i][1].item(), prediction, actual, GRAPH_ERROR_FUNCTION(prediction, actual))
			return [make_tuple(i) for i in range(receptors[0].shape[0])]
		
		predictions = reduce(iconcat, [predict_batch(batch) for batch in batches], [])
		
		out_file = open(filepath, 'w')
		format = lambda l: str(l).replace('\'', '').replace(', ', ',')[1:-1] + '\n'
		out_file.write(format(['x', 'y', 'prediction', 'actual', 'error']))
		for prediction in predictions:
			out_file.write(format(prediction))
		out_file.close()

if __name__ == "__main__":
	links_list = extract_list(DIRECTORY + '/data/link_data.csv', 'link')
	receptors_list = [r for r in extract_list(DIRECTORY + '/data/receptor_data.csv', 'receptor') if (r.nearest_link_distance <= DISTANCE_THRESHOLD and r.pollution_concentration >= CONCENTRATION_THRESHOLD)]
	shuffle(links_list)
	shuffle(receptors_list)

	links = prep_links(links_list)

	def make_batches(condition):
		unprepped = [receptors_list[i] for i in range(len(receptors_list)) if condition(i)]
		batches = [prep_batch(unprepped[i:i+BATCH_SIZE]) for i in range(0, len(unprepped) - BATCH_SIZE, BATCH_SIZE)]
		return batches

	train_batches = make_batches(lambda i: (i % 5 < 3)) # 60%
	val_batches = make_batches(lambda i: (i % 5 == 3)) # 20%

	model = Model()
	optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)
	num_epochs = 1000
	save_location = DIRECTORY + '/Model Saves/model_save_' + strftime('%m-%d-%y %H:%M:%S') + ' ' + NOTEBOOK_NAME
	print('Save Location: ' + save_location)

	model.train(
		optimizer, 
		num_epochs, 
		links, 
		train_batches, 
		val_batches,
		save_location,
		True,
	)

	best_model, _ = Model.load(save_location)
	(r, y, nld) = val_batches[1]
	fwd = best_model.forward_batch(links, r)
	for i in range(5):
		print((TRANSFORM_OUTPUT_INV(fwd[i].item(), nld[i].item()), TRANSFORM_OUTPUT_INV(y[i].item(), nld[i].item())))

	print(sum([best_model.get_error(links, receptors, y) for (receptors, y, _) in val_batches])/len(val_batches))

	best_model.graph_prediction_error(links, val_batches)

	best_model.export_predictions(links, val_batches, save_location[save_location.rindex('model_save_') + 11:save_location.rindex('.')] + ' Predictions.csv')

	best_model.print_batch_errors(links, val_batches)