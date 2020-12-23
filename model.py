import torch
import pandas as pd

INPUT_SIZE = 9
HIDDEN_SIZE = 8
LOSS_FUNCTION = torch.nn.MSELoss()
ERROR_FUNCTION = torch.nn.MSELoss()

class Model(torch.nn.Module):
	def __init__(self, in_size, hidden_size):
		super().__init__()
		self.layer_1 = torch.nn.Linear(in_size, hidden_size)
		self.activ = torch.nn.Sigmoid()
		self.layer_2 = torch.nn.Linear(hidden_size, 1)

	def forward(self, x):
		s1 = self.layer_1(x)
		a1 = self.activ(s1)
		s2 = self.layer_2(s1)
		return s2
	
	@staticmethod
	def make_x(link, receptor):
		"""
		Uses the following three fields in [receptor]
		- x
		- y
		- elevation
		"""
		dist = (((link.x - receptor.x) ** 2) + ((link.y - receptor.y) ** 2)) ** 0.5
		elevation_diff = receptor.elevation - link.elevation_mean
		vmt = link.traffic_flow * link.link_length
		x = [
			dist,
			elevation_diff,
			vmt,
			link.traffic_speed,
			link.fleet_mix_light,
			link.fleet_mix_medium,
			link.fleet_mix_heavy,
			link.fleet_mix_commercial,
			link.fleet_mix_bus
		]
		assert (len(x) == INPUT_SIZE)
		return x

	def forward_links(self, links, receptor):
		x = torch.Tensor([Model.make_x(link, receptor) for link in links])
		return torch.sum(self.forward(x), axis=0)

def extract_dict(filepath, class_name=''):
	df = pd.read_csv(filepath)
	d = {}
	for row in df.iterrows():
		obj = type(class_name, (object,), dict(row[1]))
		d[obj.id] = obj
	return d

def extract_list(filepath, class_name=''):
	df = pd.read_csv(filepath)
	return [type(class_name, (object,), dict(row[1])) for row in df.iterrows()]

ALL_LINKS = extract_list('data/link_data.csv', 'link')
ALL_RECEPTORS = extract_dict('data/receptor_data.csv', 'receptor')

def get_error(model, links, receptors, y):
	"""
	[links] is a python list of all links which contribute to emissions at receptors in [receptors]
	[receptors] is python list of receptors with x, y, and elevation set
	[y] is Tensor of corresponding true pollution concentrations
	"""
	y_hat = torch.Tensor([model.forward_links(links, r) for r in receptors]) 
	return ERROR_FUNCTION(y_hat, y).item()

def train():
	train_receptors = []
	val_receptors = []

	# 60% Train, 20% Val
	i = 0
	for receptor in ALL_RECEPTORS.values():
		if (i % 5 < 3):
			train_receptors.append(receptor)
		elif (i % 5 == 3):
			val_receptors.append(receptor)
		i += 1

	train_y = torch.Tensor([r.pollution_concentration for r in train_receptors])
	val_y = torch.Tensor([r.pollution_concentration for r in val_receptors])

	model = Model(in_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)

	num_epochs = 10
	optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01)
	for _ in range(num_epochs):
		for i in range(len(train_receptors)):
			optimizer.zero_grad()
			y_hat = model.forward_links(ALL_LINKS, train_receptors[i])
			loss = LOSS_FUNCTION(y_hat, train_y[i])
			loss.backward()
			optimizer.step()
			if ((i + 1) % 100 == 0):
				print('Loss: ' + str(loss.item()))
				print('Train Error: ' + str(get_error(model, ALL_LINKS, train_receptors, train_y)))
				print('Val Error: ' + str(get_error(model, ALL_LINKS, val_receptors, val_y)))

	return model

# TODO: data normalization, meteorological data, use GPU?, hyperparameters, efficient error calc
