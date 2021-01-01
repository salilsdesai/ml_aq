import torch
import pandas as pd
from time import time, strftime, localtime
from random import shuffle
from matplotlib import pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE != 'cuda':
	print('WARNING: Not using GPU')

HIDDEN_SIZE = 8
LOSS_FUNCTION = torch.nn.MSELoss()
ERROR_FUNCTION = torch.nn.MSELoss()
BATCH_SIZE = 1000

# Used to load means and std devs for feature normilization
FEATURES = [
	'distance', 'elevation_difference', 'vmt', 'traffic_speed', 
	'fleet_mix_light', 'fleet_mix_medium', 'fleet_mix_heavy', 
	'fleet_mix_commercial', 'fleet_mix_bus',
]
INPUT_SIZE = len(FEATURES)

def extract_list(filepath, class_name=''):
	df = pd.read_csv(filepath)
	return [type(class_name, (object,), dict(row[1])) for row in df.iterrows()]

def prep_links(links=None):
	if links is None:
		links = extract_list('data/link_data.csv', 'link')
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
	  link.fleet_mix_bus
	] for link in links]]).to(DEVICE).repeat(BATCH_SIZE, 1, 1)
	# If too much memory or need multiple batch sizes, move repeat to make_x (distances.shape[0])
	return (coords, coords_dot, subtract, keep)
  
def prep_receptors(receptors=None):
	if receptors is None:
		receptors = extract_list('data/receptor_data.csv', 'receptor')
	coords = torch.DoubleTensor([[r.x, r.y] for r in receptors]).to(DEVICE)
	coords_dot = (coords * coords).sum(dim=1).unsqueeze(dim=-1)
	subtract = torch.Tensor([[[r.elevation]] for r in receptors]).to(DEVICE)
	return (coords, coords_dot, subtract)

def load_feature_stats(filepath):
	"""
	Load the mean and std dev of each feature from a file
	Used to normalize data
	"""
	f = open(filepath, 'r')
	lines = f.readlines()
	feature_stats_dict = {}
	for i in range(len(lines)):
		splits = lines[i].split(',')
		feature_stats_dict[splits[0]] = (float(splits[1]), float(splits[2]))
	f.close()
	feature_stats = torch.Tensor(2, INPUT_SIZE)
	for i in range(INPUT_SIZE):
		(feature_stats[0][i], feature_stats[1][i]) = feature_stats_dict[FEATURES[i]]
	return feature_stats.to(DEVICE)

FEATURE_STATS = load_feature_stats('data/feature_stats.txt')

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
		return s2

	@staticmethod
	def make_x(links, receptors):
		distances = (links[1] - (2 * (receptors[0] @ links[0])) + receptors[1]).sqrt().unsqueeze(dim=2).type(torch.Tensor).to(DEVICE)
		subtracted = links[2] - receptors[2]
		kept = links[3]
		cat = torch.cat((distances, subtracted, kept), dim=-1)
		normalized = (cat - FEATURE_STATS[0])/FEATURE_STATS[1]
		return normalized

	def forward_batch(self, links, receptors):
		return self.forward(Model.make_x(links, receptors)).sum(dim=1)

	def get_error(self, links, receptors, y):
		"""
		[y] is Tensor of corresponding true pollution concentrations
		"""
		y_hat = self.forward_batch(links, receptors)
		return ERROR_FUNCTION(y_hat, y).item()

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
		train_errors = [] 
		val_errors = []

		def get_stats(loss):
			losses.append(loss / len(train_batches))
			print('Loss: ' + str(losses[-1]))
			train_errors.append(sum([self.get_error(links, receptors, y) for (receptors, y) in train_batches])/len(train_batches))
			print('Train Error: ' + str(train_errors[-1]))
			val_errors.append(sum([self.get_error(links, receptors, y) for (receptors, y) in val_batches])/len(val_batches))
			print('Val Error: ' + str(val_errors[-1]))

		get_stats(sum([LOSS_FUNCTION(self.forward_batch(links, receptors), y).item() for (receptors, y) in train_batches]))

		for curr_epoch in range(num_epochs):
			epoch_loss = 0
			for (receptors, y) in train_batches:
				optimizer.zero_grad()
				y_hat = self.forward_batch(links, receptors)
				loss = LOSS_FUNCTION(y_hat, y)
				epoch_loss += loss.item()
				loss.backward()
				optimizer.step()
			print('---------------------------------------------')
			print('Finished Epoch ' + str(curr_epoch) + ' (' + str(time() - start_time) + ' seconds)')
			get_stats(epoch_loss)
			if save_location is not None and val_errors[-1] == min(val_errors):
				self.save(save_location, optimizer)
				print('Saved Model!')
		
		if make_graphs:
			titles = iter(('Loss', 'Train Error', 'Val Error'))
			for l in (losses, train_errors, val_errors):
				plt.plot(range(len(l)), l)
				plt.title(next(titles))
				plt.show()

if __name__ == "__main__":
	links_list = extract_list('data/link_data.csv', 'link')
	receptors_list = [r for r in extract_list('data/receptor_data.csv', 'receptor') if r.pollution_concentration != 0]

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
				y = torch.Tensor([[r.pollution_concentration] for r in b]).to(DEVICE)
				batches.append((receptors, y))
		return batches

	train_batches = make_batches(lambda i: (i % 5 < 3)) # 60%
	val_batches = make_batches(lambda i: (i % 5 == 3)) # 20%

	model = Model()
	optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001)
	num_epochs = 10
	save_location = 'model_save_' + strftime('%m-%d-%y %H:%M:%S')

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
	(r, y) = val_batches[1]
	fwd = best_model.forward_batch(links, r)
	for i in range(5):
		print((fwd[i].item(), y[i].item()))

	print(best_model.get_error(links, r, y))

# TODO: meteorological data, hyperparameters
