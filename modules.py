import torch

from torch import FloatTensor as FloatTensor
from torch import LongTensor as LongTensor

import numpy as np
from numpy import random

class Module(object):
    def forward(self,*input):
        raise NotImplementedError

    def backward(self,*gradwrtoutput):
        raise NotImplementedError

    def resetGradient(self):
        raise NotImplementedError

    def param(self):
        return []

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # TODO: fix the initialization of the matrix
        self.weights = FloatTensor(np.random.normal(0,1,size=(out_features,in_features)))
        self.bias = FloatTensor(np.random.normal(0,1,size=(1,out_features)))

        # TODO: are we allowed to use zeros?
        self.weights_grad = torch.zeros(out_features,in_features)
        self.bias_grad = torch.zeros(1,out_features)

    def forward(self,*input):
        # save previous input for backward pass. Attention: this might not be optimal
        self.inputs = input
        return input[0].mm(self.weights.transpose(0,1)) + self.bias

    def backward(self,*gradwrtoutput):
        dl_ds = gradwrtoutput[0]
        self.weights_grad = self.weights_grad + dl_ds.transpose(0,1).mm(self.inputs[0])

        # I am not sure about this step
        self.bias_grad = self.bias_grad + FloatTensor(np.sum(dl_ds.numpy(),axis=0))
        return dl_ds.mm(self.weights)

    def resetGradient(self):
        self.weights_grad.zero_()
        self.bias_grad.zero_()

    def param(self):
        return [(self.weights, self.weights_grad), (self.bias, self.bias_grad)]

class ReLU(Module):
    def forward(self,*input):
        self.input = input
        return FloatTensor(np.maximum(input[0].numpy(),np.zeros(input[0].size())))

    def backard(self,*gradwrtoutput):
        dsigma = FloatTensor(np.heaviside(self.input[0].numpy(),1.0))
        dl_ds = dsigma * self.input[0]
        return dl_ds

    def resetGradient(self):
        return

    def param(self):
        return []

class Sequential(Module):
    def __init__(self,criterion):
        self.modules_list = []
        self.modules_registered = False
        self.criterion = criterion

    def registerModules(self,*modules):
        self.modules_registered = True
        for m in modules:
            self.modules_list.append(m)

    def checkIfModulesAreRegistered(self):
        if (self.modules_registered is False):
            raise RuntimeError('No modules were registered in the Sequential net!')

    # call resetGradient on all the modules
    def resetGradient(self):
        checkIfModulesAreRegistered()

        for m in self.modules_list:
            m.restetGradient()

    def backward(self,*gradwrtoutput):
        checkIfModulesAreRegistered()

        grad = gradwrtoutput[0]

        for m in reversed(self.modules_list):
            grad = m.backard(grad)

        return grad

    def backward(self, output, expected):
        return backward(criterion.grad(output,expected))
