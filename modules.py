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

    def forward(self,*input):
        print(input)
        return input[0].mm(self.weights.transpose(0,1)) + self.bias

    def param(self):
        return [self.weights, self.bias]
