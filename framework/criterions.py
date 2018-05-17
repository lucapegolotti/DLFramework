import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch

from torch import FloatTensor as FloatTensor
from torch import LongTensor as LongTensor

"""
Loss: template for a generic loss function
"""
class Loss(object):
    """
    function: returns the value of the loss function for a specific value of
    output and target. To be implemneted by derived classes
    input:
        - output: the result of the forward pass
        - expected: the target
    """
    def function(self,output,expected):
        raise NotImplementedError

    """
    function: returns the value of the gradient of the loss function for
    a specific value of output and target. To be implemneted by derived classes
    input:
        - output: the result of the forward pass
        - expected: the target
    """
    def grad(self,output,expected):
        raise NotImplementedError

#TODO: check if this is correct
"""
LossMSE: Mean Square Error loss function. This is defined as
    MSE(x,y) = 1/N sum_{i,j} (x_{ij} - y_{ij})^2
"""
class LossMSE(object):
    """
    function: value of the MSE loss function corresponding to the output
    and the expected target
    input:
        - output: the result of the forward pass
        - expect: the target
    output:
        - value of the MSE loss
    """
    def function(self,output,expected):
        return torch.mean(torch.pow(expected - output,2))

    """
    function: value of the gradient of MSE loss function corresponding
    to the output and the expected target
    input:
        - output: the result of the forward pass
        - expect: the target
    output:
        - gradient of the MSE loss
    """
    def grad(self,output,expected):
        return -2 * (expected - output)
