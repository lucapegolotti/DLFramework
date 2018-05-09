#################### TODO: delete dependencies of pytorch
import torch
from torch import nn

import numpy as np
####################

from torch import FloatTensor as FloatTensor
from torch import LongTensor as LongTensor

import modules as mm
import criterions as C

import build_linear_test as test



npoints = 1000
train_input, train_target, test_input, test_target = test.generate(npoints)

nsamples = npoints
nfeatures = 2
nchannels = 1
outputs = 2

class SimpleNet(mm.Sequential):
    def __init__(self,criterion):
        super(SimpleNet, self).__init__(criterion)
        self.fc1 = mm.Linear(nchannels * nfeatures, 25)
        self.fc2 = mm.Linear(25, 25)
        self.fc3 = mm.Linear(25, outputs)
        self.nonlinear1 = mm.ReLU()
        self.nonlinear2 = mm.ReLU()
        self.nonlinear3 = mm.ReLU()


        #super().registerModules(self.fc1,self.nonlinear,self.fc2,self.nonlinear,self.fc3,self.nonlinear)
        super().registerModules(self.fc1, self.nonlinear1, self.fc2,self.nonlinear2,self.fc3,self.nonlinear3)

    def forward(self, *input):
        x = input[0].view(nsamples, nchannels * nfeatures)
        x = self.fc1.forward(x)
        x = self.nonlinear1.forward(x)
        x = self.fc2.forward(x)
        x = self.nonlinear2.forward(x)
        x = self.fc3.forward(x)
        x = self.nonlinear3.forward(x)

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

def train_model(net,n_epochs,eta):
    count = compute_number_errors(train_input,train_target)
    train_string = "Initial train error : {0:.2f}%".format((nsamples-count)/nsamples*100)
    for i in range(n_epochs):
        net.resetGradient()
        output = net.forward(train_input)
        loss_value = net.backwardPass(output,train_target)
        net.updateParameters(eta,nsamples)

        #print("Epoch = " + str(i))
        loss_string = "\tLoss : {0:.2f}".format(loss_value)
        print(loss_string)
        count = compute_number_errors(net.forward(train_input),train_target)
        train_string = "\tTrain error : {0:.2f}%".format((nsamples-count)/nsamples*100)
        print(train_string)
        count = compute_number_errors(net.forward(test_input),test_target)
        train_string = "\tTest error : {0:.2f}%".format((nsamples-count)/nsamples*100)
        print(train_string)


loss = C.LossMSE()
net = SimpleNet(loss)

n_epochs, eta = 500, 1e-2
train_model(net,n_epochs,eta)
