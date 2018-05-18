import torch
import math

from torch import optim
from torch import Tensor
from torch.autograd import Variable
from torch import nn

import build_test as test

import numpy as np

# We generate 1000 points in (0,1) x (0,1) and label them 1 if they are inside
# the circle with radius 1/sqrt(2 * pi), 0 otherwise. This is done for train and
# target
npoints = 1000
train_input, train_target, test_input, test_target = test.generate(npoints)


mean, std = train_input.mean(), train_input.std()

# We normalize the data according to mean and standard deviation
train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)

train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)

def train_model(model, train_input, train_target,n_epochs,eta,mini_batch_size):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=eta)

    for e in range(0, n_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            sum_loss = sum_loss + loss.data.item()
            optimizer.step()

        if (e % 10) == 0:
            train_error = compute_number_errors(model.forward(train_input), train_target) / npoints * 100
            print("Epoch: " + str(e) + " loss_function = {:.02f}".format(sum_loss) + \
                  " train error = {:.02f}".format(train_error))

# compute the number of errors
def compute_number_errors(inputs,outputs):
    indicesy = np.argmax(inputs.detach(),1).float()
    indicesout = np.argmax(outputs,1).float()
    nberrors = np.linalg.norm(indicesy - indicesout,0)
    return nberrors

# create a network equivalent to the network used in test.py
model = nn.Sequential(nn.Linear(2, 25),nn.ReLU(),nn.Linear(25, 25),nn.Tanh(),nn.Linear(25, 2))

# initialize weights in model with normal (0,1)
for p in model.parameters(): p.data.normal_(0, 1)

n_epochs, eta, mini_batch_size = 500, 1e-3, 40
train_model(model, train_input, train_target, n_epochs, eta, mini_batch_size)

# print final errors
print("==================================================")
train_error = compute_number_errors(model.forward(train_input), train_target) / npoints * 100
test_error = compute_number_errors(model.forward(test_input), test_target) / npoints * 100
print("Train error = {:.02f}".format(train_error))
print("Test error = {:.02f}".format(test_error))
