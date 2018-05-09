#################### TODO: delete dependencies of pytorch
import torch
from torch import nn

import numpy as np
####################

from torch import FloatTensor as FloatTensor
from torch import LongTensor as LongTensor

import modules as mm
import criterions as C

import build_test as test


npoints = 10000
train_input, train_target, test_input, test_target = test.generate(npoints)

nsamples = npoints
nfeatures = 2
nchannels = 1
outputs = 2

class SimpleNet(mm.Sequential):
    def __init__(self,criterion):
        super(SimpleNet, self).__init__(criterion)
        self.fc1 = mm.Linear(nchannels * nfeatures, outputs)
        self.nonlinear = mm.ReLU()

        super().registerModules(self.fc1)

    def forward(self, *input):
        x = input[0].view(nsamples, nchannels * nfeatures)
        x = self.fc1.forward(x)
        x = self.nonlinear.forward(x)
        return x

def compute_number_errors(inputs,outputs):
    nsamples = inputs.size(0)
    count = 0
    for i in range(nsamples):
        if (inputs[i,0] > inputs[i,1]):
            indexmax_input = 0
        else:
            indexmax_input = 1

        if (outputs[i,0] > outputs[i,1]):
            indexmax_output = 0
        else:
            indexmax_output = 1
        if (indexmax_input == indexmax_output):
            count = count + 1

    return count


loss = C.LossMSE()
net = SimpleNet(loss)

n_epochs = 1000
eta = 0.001
for i in range(n_epochs):
    net.resetGradient()
    output = net.forward(train_input)
    loss_value = net.backwardPass(output,train_target)
    net.updateGradient(eta)

    print("Epoch = " + str(i))
    loss_string = "Loss : {0:.2f}".format(loss_value)
    print(loss_string)
    count = compute_number_errors(train_input,train_target)
    train_string = "Train error : {0:.2f}%".format((nsamples-count)/nsamples*100)
    print(train_string)
