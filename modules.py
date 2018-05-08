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

    def param(self) :
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
        self.weights_grad = self.weights_grad + dl_ds.mm(input[0])

        return self.weights.transpose(0,1).mm(dl_ds)

    def param(self):
        return [(self.weights, self.weights_grad), (self.bias, self.bias_grad)]
