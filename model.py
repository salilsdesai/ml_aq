import torch
import pandas as pd
from math import isnan
from functools import partial, reduce
from time import time
import numpy as np
from random import shuffle

INPUT_SIZE = 9
HIDDEN_SIZE = 8
LOSS_FUNCTION = torch.nn.MSELoss()
ERROR_FUNCTION = torch.nn.MSELoss()
BATCH_SIZE = 1000

def extract_list(filepath, class_name=''):
	df = pd.read_csv(filepath)
	return [type(class_name, (object,), dict(row[1])) for row in df.iterrows()]

def prep_links(links=None):
	if links is None:
		links = extract_list('data/link_data.csv', 'link')
	coords = torch.DoubleTensor([[l.x, l.y] for l in links]).T
	coords_dot = (coords * coords).sum(dim=0)
	subtract = torch.Tensor([[l.elevation_mean] for l in links])
	keep = torch.Tensor([[[
	  link.traffic_flow * link.link_length,
	  link.traffic_speed,
	  link.fleet_mix_light,
	  link.fleet_mix_medium,
	  link.fleet_mix_heavy,
	  link.fleet_mix_commercial,
	  link.fleet_mix_bus
	] for link in links]])
	return (coords, coords_dot, subtract, keep)
  
def prep_receptors(receptors=None):
	if receptors is None:
		receptors = extract_list('data/receptor_data.csv', 'receptor')
	coords = torch.DoubleTensor([[r.x, r.y] for r in receptors])
	coords_dot = (coords * coords).sum(dim=1).unsqueeze(dim=-1)
	subtract = torch.Tensor([[[r.elevation]] for r in receptors])
	return (coords, coords_dot, subtract)

def load_feature_stats(filepath):
	"""
	Load the mean and std dev of each feature from a file
	Used to normalize data
	"""
	f = open(filepath, 'r')
	lines = f.readlines()
	assert (len(lines) == INPUT_SIZE)
	feature_stats = torch.zeros((2, INPUT_SIZE))
	for i in range(len(lines)):
		splits = lines[i].split(',')
		feature_stats[0][i] = float(splits[1])
		feature_stats[1][i] = float(splits[2])
	f.close()
	return feature_stats

FEATURE_STATS = load_feature_stats('data/feature_stats.txt')

class Model(torch.nn.Module):
	def __init__(self, in_size, hidden_size):
		super().__init__()
		self.layer_1 = torch.nn.Linear(in_size, hidden_size)
		self.activ = torch.nn.Sigmoid()
		self.layer_2 = torch.nn.Linear(hidden_size, 1)

	def forward(self, x):
		s1 = self.layer_1(x)
		a1 = self.activ(s1)
		s2 = self.layer_2(a1)
		return s2

	@staticmethod
	def make_x(links, receptors):
		distances = (links[1] - (2 * (receptors[0] @ links[0])) + receptors[1]).sqrt().unsqueeze(dim=2).type(torch.Tensor)
		subtracted = links[2] - receptors[2]
		kept = links[3].repeat(distances.shape[0], 1, 1)
		cat = torch.cat((distances, subtracted, kept), dim=-1)
		return (cat - FEATURE_STATS[0])/FEATURE_STATS[1]

	def forward_batch(self, links, receptors):
		return self.forward(Model.make_x(links, receptors)).sum(dim=1)

def get_error(model, links, receptors, y):
	"""
	[y] is Tensor of corresponding true pollution concentrations
	"""
	y_hat = model.forward_batch(links, receptors)
	return ERROR_FUNCTION(y_hat, y).item()

def train(model):
	links_list = extract_list('data/link_data.csv', 'link')
	receptors_list = extract_list('data/receptor_data.csv', 'receptor')

	shuffle(links_list)
	shuffle(receptors_list)

	links = prep_links(links_list)

	def make_batches(condition):
		unprepped = [receptors_list[i] for i in range(len(receptors_list)) if condition(i)]
		batches = []
		for i in range(0, len(unprepped), BATCH_SIZE):
			if (i + BATCH_SIZE <= len(unprepped)):
				b = unprepped[i:i+BATCH_SIZE]
				receptors = prep_receptors(b)
				y = torch.Tensor([[r.pollution_concentration] for r in b])
				batches.append((receptors, y))
		return batches

	train_batches = make_batches(lambda i: (i % 5 < 3)) # 60%
	val_batches = make_batches(lambda i: (i % 5 == 3)) # 20%

	optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01)
	num_epochs = 10
	start_time = time()
	for curr_epoch in range(num_epochs):
		epoch_loss = 0
		for (receptors, y) in train_batches:
			optimizer.zero_grad()
			y_hat = model.forward_batch(links, receptors)
			loss = LOSS_FUNCTION(y_hat, y)
			epoch_loss += loss.item()
			loss.backward()
			optimizer.step()
		print('---------------------------------------------')
		print('Finished Epoch ' + str(curr_epoch) + ' (' + str(time() - start_time) + ' seconds)')
		print('Epoch Loss: ' + str(epoch_loss / len(train_batches)))
		print('Val Error: ' + str(sum([get_error(model, links, receptors, y) for (receptors, y) in val_batches])/len(val_batches)))
		print('Train Error: ' + str(sum([get_error(model, links, receptors, y) for (receptors, y) in train_batches])/len(train_batches)))

if __name__ == "__main__":
	model = Model(in_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
	train(model)

# TODO: meteorological data, use GPU?, hyperparameters
