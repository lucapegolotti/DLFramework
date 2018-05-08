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
    def __init__(self, rows, cols):
        super().__init__()
        self.rows = rows
        self.cols = cols
        # TODO: fix the initialization of the matrix
        self.weights = FloatTensor(np.random.uniform(0,1,size=(rows,cols)))
        self.bias = FloatTensor(np.random.uniform(0,1,size=(rows,1)))

    def forward(self,*input):
        print(input)
        return self.weights.mv(input[0]) + self.bias
