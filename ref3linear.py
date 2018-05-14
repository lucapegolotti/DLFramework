#!/usr/bin/env python

######################################################################

import torch
import math

from torch import optim
from torch import Tensor
from torch.autograd import Variable
from torch import nn

import build_linear_test as test
######################################################################


#train_input, train_target = generate_disc_set(1000)
#test_input, test_target = generate_disc_set(1000)
npoints = 1000
train_input, train_target, test_input, test_target = test.generate(npoints)
#print(train_target.size())

mean, std = train_input.mean(), train_input.std()

train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)

train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)

mini_batch_size = 1000


######################################################################

def train_model(model, train_input, train_target):
	criterion = nn.MSELoss()
	eta = 1e-3
	#optimizer = optim.SGD(model.parameters(), lr=1e-3)
	nb_epochs = 1000

	for e in range(0, nb_epochs):
		sum_loss = 0
		for b in range(0, train_input.size(0), mini_batch_size):
			output = model(train_input.narrow(0, b, mini_batch_size))
			loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
			model.zero_grad()
			loss.backward()
			#sum_loss = sum_loss + loss.data.item()
			for p in model.parameters():
				p.data.sub_(eta * p.grad.data)
		if (e % 100 == 0):
			print('epoch {:d} loss  {:f} sum_loss  {:f}  train_error {:.02f}% test_error {:.02f}%'.format(e,
					loss,
					loss,
					compute_nb_errors(model,
								   train_input,
								   train_target) / train_input.size(
					 0) * 100,
					compute_nb_errors(model,
								   test_input,
								   test_target) / test_input.size(
					 0) * 100
					)
				  )


######################################################################

def compute_nb_errors(model, data_input, data_target):
	nb_data_errors = 0

	for b in range(0, data_input.size(0), mini_batch_size):
		output = model(data_input.narrow(0, b, mini_batch_size))
		_, predicted_classes = torch.max(output.data, 1)
		for k in range(0, mini_batch_size):
			if data_target.data[b + k,predicted_classes[k]] <0.1:
				nb_data_errors = nb_data_errors + 1

	return nb_data_errors


######################################################################



def mymodel():
	return nn.Sequential(
		nn.Linear(2, 25),
		nn.ReLU(),
		nn.Linear(25, 25),
		nn.ReLU(),
		nn.Linear(25, 2)
	)
m=mymodel
model=m()
for p in model.parameters(): p.data.normal_(0, 1)
train_model(model, train_input, train_target)
print('std {:s} train_error {:.02f}% test_error {:.02f}%'.format(
	m.__name__,
	compute_nb_errors(model, train_input, train_target) / train_input.size(0) * 100,
	compute_nb_errors(model, test_input, test_target) / test_input.size(0) * 100
)
)
