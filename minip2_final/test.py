import torch
from torch import nn

import numpy as np

from torch import FloatTensor as FloatTensor
from torch import LongTensor as LongTensor

# import our framework
import framework.modules as M
import framework.criterions as C
import framework.networks as N

import build_test as test

# We generate 1000 points in (0,1) x (0,1) and label them 1 if they are inside
# the circle with radius 1/sqrt(2 * pi), 0 otherwise. This is done for train and
# target
npoints = 1000
train_input, train_target, test_input, test_target = test.generate(npoints)

mean, std = train_input.mean(), train_input.std()

# We normalize the data according to mean and standard deviation
train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)

nsamples = npoints
nfeatures = 2
nchannels = 1
outputs = 2

# Simple net created with our framework
class SimpleNet(N.Sequential):
	def __init__(self,criterion):
		super(SimpleNet, self).__init__(criterion)
		# define modules (Pytorch style)
		self.fc1 = M.Linear(nchannels * nfeatures, 25)
		self.fc2 = M.Linear(25, 25)
		self.fc3 = M.Linear(25, outputs)
		self.nonlinear1 = M.ReLU()
		self.nonlinear2 = M.ReLU()

		# register the module "in order", so that the network is aware of which
		# node communicates with which
		super().registerModules(self.fc1, self.nonlinear1, self.fc2,self.nonlinear2,self.fc3)

	# forward step
	def forward(self, *input):
		x = input[0].view(-1, nchannels * nfeatures)
		x = self.fc1.forward(x)
		x = self.nonlinear1.forward(x)
		x = self.fc2.forward(x)
		x = self.nonlinear2.forward(x)
		x = self.fc3.forward(x)
		return x

# compute the number of errors
def compute_number_errors(inputs,outputs):
	indicesy = np.argmax(inputs,1).float()
	indicesout = np.argmax(outputs,1).float()
	nberrors = np.linalg.norm(indicesy - indicesout,0)
	return nberrors

# train the model using the stochastic gradient descent
def train_model(net,n_epochs,eta,mini_batch_size,train_input, train_target):
	for i in range(n_epochs):
		# at each epoch, we perform a permutation of the train_input and train_target
		perm = torch.randperm(npoints)
		train_input = train_input[perm]
		train_target = train_target[perm]
		loss = 0
		for b in range(0, npoints, mini_batch_size):
			net.resetGradients()
			# this is to avoid problems when npoints % mini_batch_size != 0
			actual_mini_batch_size = min(mini_batch_size,npoints - b)

			# forward and backward pass. The backward computes also the gradient to
			# be used in the updateWeights functions
			output = net.forward(train_input.narrow(0, b, actual_mini_batch_size))
			loss_value = net.backward(output,train_target.narrow(0, b, actual_mini_batch_size))

			loss += loss_value
			net.updateWeights(eta,mini_batch_size)

		if (i % 10) == 0:
			train_error = compute_number_errors(net.forward(train_input), train_target) / npoints * 100
			print("Epoch: " + str(i) + " loss_function = {:.02f}".format(loss) + \
				" train error = {:.02f}".format(train_error))

# declare a loss function and associate it to the simple not
loss = C.LossMSE()
net = SimpleNet(loss)
n_epochs, eta, mini_batch_size = 1000, 1e-3, 40

# train the model according to SGD
train_model(net,n_epochs,eta,mini_batch_size,train_input, train_target)

# print final errors
print("==================================================")
train_error = compute_number_errors(net.forward(train_input), train_target) / npoints * 100
test_error = compute_number_errors(net.forward(test_input), test_target) / npoints * 100
print("Train error = {:.02f}".format(train_error))
print("Test error = {:.02f}".format(test_error))
