import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch

from torch import FloatTensor as FloatTensor
from torch import LongTensor as LongTensor

class Loss(object):
    def function(self,output,expected):
        raise NotImplementedError

    def grad(self,output,expected):
        raise NotImplementedError

class LossMSE(object):
    def function(self,output,expected):
        return torch.mean(torch.pow(expected - output,2))

    def grad(self,output,expected):
        return -2 * (expected - output)