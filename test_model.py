"""
Test model which demonstrates training on sums of 
target function outputs
"""

import torch
from random import random

TRUE_WEIGHTS = {
    'w1': torch.Tensor([
        [0.1,  0.4],
        [0.2, 0.5],
        [0.3, 0.6],
    ]),
    'b1': torch.Tensor([0.7, 0.8, 0.9]),
    'w2': torch.Tensor([[-6, -6.1, -6.2]]),
    'b2': torch.Tensor([12.4])
}

LOSS_FUNCTION = torch.nn.MSELoss()
ERROR_FUNCTION = torch.nn.MSELoss()

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(2, 3)
        self.layer_2 = torch.nn.Linear(3, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.sigmoid(x)
        x = self.layer_2(x)
        x = self.sigmoid(x)
        return x

def get_true_model():
    q = Model()
    for (param, true) in [
        (q.layer_1.weight, 'w1'),
        (q.layer_1.bias, 'b1'),
        (q.layer_2.weight, 'w2'),
        (q.layer_2.bias, 'b2'),
    ]:
        d = param.data
        t = TRUE_WEIGHTS[true]
        for i in range(d.shape[0]):
            try:
                for j in range(d.shape[1]):
                    d[i][j] = t[i][j]
            except Exception:
                d[i] = t[i]
    return q

def generate_data(path, n):
    t = open(path, 'w')
    m = get_true_model()
    for _ in range(n):
        x = (random() * 2 - 1, random() * 2 - 1)
        t.write(str(x[0]) + ',' + str(x[1]) + ',' + str(m.forward(torch.Tensor(x)).item()) + '\n')
    t.close()

def load_data(path):
    t = open(path, 'r')
    X = []
    y = []
    for l in t:
        splits = l.split(',')
        X.append((float(splits[0]), float(splits[1])))
        y.append([float(splits[2])])
    
    return torch.Tensor(X), torch.Tensor(y)

def load_data_groups(path, group_size):
    t = open(path, 'r')
    X = []
    y = []
    it = iter(t)
    stop_iter = False
    while not stop_iter:
        try:
            splits = [next(it).split(',') for _ in range(group_size)]
            X.append([(float(s[0]), float(s[1])) for s in splits])
            y.append([sum([float(s[2]) for s in splits])])
        except StopIteration:
            stop_iter = True

    return torch.Tensor(X), torch.Tensor(y)

def train(model, num_epochs):
    X, y = load_data('train_data.txt')
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
    for curr_epoch in range(num_epochs):
        for i in range(len(X)):
            optimizer.zero_grad()
            y_hat = model(X[i])
            loss = LOSS_FUNCTION(y_hat, y[i])
            loss.backward()
            optimizer.step()    
        train_outputs = model(X)
        print('Loss after ' + str(curr_epoch + 1) + ': ' + str(LOSS_FUNCTION(train_outputs, y).item()))

def train_on_sums(model, num_epochs, group_size):
    X, y = load_data_groups('train_data.txt', group_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay=0.01)
    for curr_epoch in range(num_epochs):
        running_loss = 0
        for i in range(len(X)):
            optimizer.zero_grad()
            y_hat = torch.sum(model(X[i]), dim=0)
            loss = LOSS_FUNCTION(y_hat, y[i])
            running_loss += loss
            loss.backward()
            optimizer.step()    
        print('Loss after ' + str(curr_epoch + 1) + ': ' + str(running_loss/len(X)))

def get_test_error(model):
    X, y = load_data('test_data.txt')
    y_hat = model(X)
    loss = LOSS_FUNCTION(y_hat, y).item()
    print('Test Loss: ' + str(loss))
    for i in range(10):
        print((X[i][0].item(), X[i][1].item(), y[i].item(), y_hat[i].item()))

if __name__ == "__main__":
    generate_data('train_data.txt', 10000)
    generate_data('test_data.txt', 10000)
    m = Model()
    train_on_sums(m, 50, 5)
    get_test_error(m)