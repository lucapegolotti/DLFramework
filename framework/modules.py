import sys
import os
sys.path.append(os.path.dirname(__file__))

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

    def updateWeights(self,eta):
        raise NotImplementedError

    # for the moment, this is useless
    def param(self):
        return []

class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear,self).__init__()
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

    def updateWeights(self,eta):
        self.weights = self.weights - eta * self.weights_grad
        self.bias = self.bias - eta * self.bias_grad

    def param(self):
        return [(self.weights, self.weights_grad), (self.bias, self.bias_grad)]

class ActivationFunction(Module):
    def __init__(self):
        super(ActivationFunction,self).__init__()

    def resetGradient(self):
        return

    def updateWeights(self,eta):
        return

    def param(self):
        return []

class ReLU(ActivationFunction):
    def __init__(self):
        super(ReLU,self).__init__()

    def forward(self,*input):
        self.input = input
        return FloatTensor(np.maximum(input[0].numpy(),np.zeros(input[0].size())))

    def backward(self,*gradwrtoutput):
        dsigma = FloatTensor(np.heaviside(self.input[0].numpy(),1.0))
        dl_ds = dsigma * gradwrtoutput[0]
        return dl_ds

class Tanh(ActivationFunction):
    def __init__(self):
        super(Tanh,self).__init__()

    def forward(self,*input):
        self.input = input
        return np.tanh(input[0])

    def backward(self,*gradwrtoutput):
        th = np.tanh(self.input[0])
        dsigma = 1 - th*th
        dl_ds = dsigma * gradwrtoutput[0]
        return dl_ds

class Sigmoid(ActivationFunction):
    def __init__(self):
        super(Sigmoid,self).__init__()

    def forward(self,*input):
        self.input = input
        expval = np.exp(input[0])
        return np.divide(expval,1 + expval)

    def backward(self,*gradwrtoutput):
        sigma = self.forward(self.input[0])
        dsigma = np.multiply(sigma,(1 - sigma))
        dl_ds = dsigma * gradwrtoutput[0]
        return dl_ds
