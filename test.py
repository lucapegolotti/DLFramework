#################### TODO: delete dependencies of pytorch
import torch
from torch import nn

import numpy as np
####################

from torch import FloatTensor as FloatTensor
from torch import LongTensor as LongTensor

import modules as mm
import criterions as C

nsamples = 100
nchannels = 5
nfeatures = 50

outputs = 2

class SimpleNet(mm.Sequential):
    def __init__(self,criterion):
        super(SimpleNet, self).__init__(criterion)
        self.fc1 = mm.Linear(nchannels * nfeatures, outputs)
        self.nonlinear = mm.ReLU()

        super().registerModules(self.fc1, self.nonlinear)

    def forward(self, *input):
        x = input[0].view(nsamples, nchannels * nfeatures)
        x = self.nonlinear.forward(self.fc1.forward(x))
        return x

loss = C.LossMSE()
net = SimpleNet(loss)

x = FloatTensor(np.random.normal(0,1,size=(nsamples,nchannels,nfeatures)))
expected = FloatTensor(np.random.normal(0,1,size=(nsamples,outputs)))

output = net.forward(x)
loss_value = net.backwardPass(output,expected)

print(loss_value)
