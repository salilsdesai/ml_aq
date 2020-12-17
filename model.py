import torch
import pandas as pd
from math import isnan
from functools import partial

INPUT_SIZE = 9
HIDDEN_SIZE = 8
LOSS_FUNCTION = torch.nn.MSELoss()
ERROR_FUNCTION = torch.nn.MSELoss()

class Network(torch.nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.layer_1 = torch.nn.Linear(in_size, hidden_size)
        self.activ = torch.nn.Sigmoid()
        self.layer_2 = torch.nn.Linear(hidden_size, 1)
         
    def forward(self, x):
        x = self.layer_1(x)
        x = self.activ(x)
        x = self.layer_2(x)
        return x
    
    def forward_batch(self, x):
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

links = extract_list('data/link_data.csv', 'link')
receptors = extract_dict('data/receptor_data.csv', 'receptor')

def pair(receptor, link):
    if isnan(link.fleet_mix_light):
        return None
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

def make_batches(receptor_ids):
    """
    Returns X y = ([X1, X2, ...], [y1, y2, ...])
    """

    X = [None] * len(receptor_ids)
    y = [None] * len(receptor_ids)

    i = 0
    for receptor_id in receptor_ids:
        receptor = receptors[receptor_id]
        pair_with_receptor = partial(pair, receptor)
        X[i] = [p for p in map(pair_with_receptor, links) if p is not None]
        y[i] = [receptor.pollution_concentration]
        i += 1
  
    return (torch.Tensor(X), torch.Tensor(y))

def get_error(model, batches):
    (X, y) = batches
    y_hat = torch.Tensor([model.forward_batch(Xi) for Xi in X]) # TODO: Efficient
    return ERROR_FUNCTION(y_hat, y).item()

train_batches = make_batches(range(1, 1001))
val_batches = make_batches(range(1001, 2001))

train_X_batches, train_y_batches = train_batches
num_epochs = 1
model = Network(in_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001)
for curr_epoch in range(num_epochs):
    for i in range(len(train_X_batches)):
        optimizer.zero_grad()
        y_hat = model.forward_batch(train_X_batches[i])
        loss = LOSS_FUNCTION(y_hat, train_y_batches[i])
        loss.backward()
        optimizer.step()
        if (i % 100 == 0):
            print('Loss: ' + str(loss))
            print('Train Error: ' + str(get_error(model, train_batches)))
            print('Val Error: ' + str(get_error(model, val_batches)))

# TODO: data normalization, meteorological data, use GPU, hyperparameters, efficient error calc
