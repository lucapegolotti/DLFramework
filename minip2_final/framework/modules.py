import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch

from torch import FloatTensor as FloatTensor
from torch import LongTensor as LongTensor

import numpy as np
from numpy import random

"""
Module: template of a module (i.e. a member of a network)
"""
class Module(object):
    """
    forward: forward method, to be implemented in derived classes
    input:
        - *input: list of inputs for the current node
    """
    def forward(self,*input):
        raise NotImplementedError

    """
    backward: backward method, to be implemented in derived classes
    input:
        - *gradwrtoutput: list of gradients, provided during backpropagation
          from the modules directly following the current module in the network
    """
    def backward(self,*gradwrtoutput):
        raise NotImplementedError

    """
    resetGradient: set the gradient (if any) to zero. To be implemented in
    derived classes
    """
    def resetGradient(self):
        raise NotImplementedError

    """
    updateWeights: update the weights (if any) according to the precomputed
    gradients. To be implemented in derived classes
    """
    def updateWeights(self,eta):
        raise NotImplementedError

    """
    param: getter method for the parameters and their gradients
    """
    def param(self):
        return []

"""
Linear: linear operation of the form y = x * A^T + b, where:
    - x: input tensor to the module
    - A: weight matrix
    - b: bias vector
    - y: output tensor
"""
class Linear(Module):
    """
    Constructor: initializes weights and bias and their gradients
    input:
        - in_features: number of features of the input tensor. Namely, the
          module expects to receive as input a tensor of size
          nsamples x in_features
        - out_features: number of features of the output tensor, which will
          have dimensions nsamples x out_features
    """
    def __init__(self, in_features, out_features):
        super(Linear,self).__init__()
        self.weights = FloatTensor(np.random.normal(0,1,size=(out_features,in_features)))
        self.bias = FloatTensor(np.random.normal(0,1,size=(1,out_features)))

        self.weights_grad = torch.zeros(out_features,in_features)
        self.bias_grad = torch.zeros(1,out_features)

    """
    forward: forward method computing the operation input * weights^T + bias
    input:
        - *input: list of inputs. Must be composed by a single input
    """
    def forward(self,*input):
        # save previous input for backward pass
        self.inputs = input

        # check input size
        if len(input) > 1:
            raise ValueError("Linear module expects just one input tensor!")

        # check input tensor size
        if input[0].size(1) is not self.weights.size(1):
            raise ValueError("Inconsistent input and weights dimensions in \
                              Linear forward method")

        return input[0].mm(self.weights.transpose(0,1)) + self.bias

    """
    backward: backward method which overrides the one provided by the
    Module super class
    input:
        - *gradwrtoutput: list of gradients, provided during backpropagation
          from the modules directly following the current module in the network.
          Must be composed by a single element
    """
    def backward(self,*gradwrtoutput):

        if not hasattr(self, 'inputs'):
            raise ValueError("Backward must be called after at least one forward call")

        # check gradwrtoutput size
        if len(gradwrtoutput) > 1:
            raise ValueError("Linear module expects just one gradient tensor in \
                              backward call!")

        dl_ds = gradwrtoutput[0]

        if dl_ds.size(0) != self.inputs[0].size(0) and \
           dl_ds.size(1) != self.weights_grad.size(0):
            raise ValueError("Inconsistent gradient dimension in Linear backward \
                              call!")

        self.weights_grad = self.weights_grad + dl_ds.transpose(0,1).mm(self.inputs[0])
        self.bias_grad = self.bias_grad + FloatTensor(np.sum(dl_ds.numpy(),axis=0))

        return dl_ds.mm(self.weights)

    """
    resetGradient: set current gradients to zero
    """
    def resetGradient(self):
        self.weights_grad.zero_()
        self.bias_grad.zero_()

    """
    updateWeights: update weights and bias according to the rule
        x = x - eta * grad(x),
    where x is either the weights tensor or the bias vector
    input:
        - eta: learning rate
    """
    def updateWeights(self,eta):
        self.weights = self.weights - eta * self.weights_grad
        self.bias = self.bias - eta * self.bias_grad

    """
    param: getter method for the parameters and their gradients
    """
    def param(self):
        return [(self.weights, self.weights_grad), (self.bias, self.bias_grad)]

"""
ActivationFunction: blueprint for an activation function. Essentially, this
class implements the method of Module that are not required for modules
that do not have internal parameters (like activation functions)
"""
class ActivationFunction(Module):
    """
    Constructor
    """
    def __init__(self):
        super(ActivationFunction,self).__init__()

    """
    resetGradient: returns nothing
    """
    def resetGradient(self):
        return

    """
    updateWeights: returns nothing
    """
    def updateWeights(self,eta):
        return

    """
    param: returns nothing
    """
    def param(self):
        return []

"""
ReLU: ReLU activation function. This module computes the relu function of
an input vector. The relu function is defined as relu(x) = max(0,x).
"""
class ReLU(ActivationFunction):
    """
    Constructor
    """
    def __init__(self):
        super(ReLU,self).__init__()

    """
    forward: applies relu function to input
    input:
        - *input: list of inputs. Must be composed by a single input
    output:
        - tensor of same size of the input. Each component is computed by
          applying the relu function to each component of the input
    """
    def forward(self,*input):
        self.input = input

        # check input size
        if len(input) > 1:
            raise ValueError("ReLU module expects just one input tensor!")

        return FloatTensor(np.maximum(input[0].numpy(),np.zeros(input[0].size())))

    """
    backward: computes backward pass. Note: the derivative of the relu function
    is relu(x)' = max(0,1) -> heaviside function
    input:
        - *gradwrtoutput: list of gradients, provided during backpropagation
          from the modules directly following the current module in the network.
          Must be composed by a single element
    output:
        - gradient to be used from the previous node in the network during
          a backward pass
    """
    def backward(self,*gradwrtoutput):

        if not hasattr(self, 'input'):
            raise ValueError("Backward must be called after at least one forward call")

        # check gradient size
        if len(gradwrtoutput) > 1:
            raise ValueError("ReLU module expects just one gradient tensor in \
                              backward call!")

        dsigma = FloatTensor(np.heaviside(self.input[0].numpy(),1.0))
        dl_ds = dsigma * gradwrtoutput[0]
        return dl_ds

"""
Tanh: Tanh activation function. This module computes the tanh function of
an input vector. The tanh function is defined as
    tanh(x) = ( exp(x) - exp(-x) ) / ( exp(x) + exp(-x) )
"""
class Tanh(ActivationFunction):
    """
    Constructor
    """
    def __init__(self):
        super(Tanh,self).__init__()

    """
    forward: applies tanh function to all the elements of the input
    input:
        - *input: list of inputs. Must be composed by a single input
    output:
        - tensor of same size of the input. Each component is computed by
          applying the tanh function to each component of the input
    """
    def forward(self,*input):

        # check input size
        if len(input) > 1:
            raise ValueError("Tanh module expects just one input tensor!")

        self.input = input
        return np.tanh(input[0])

    """
    backward: computes backward pass. Note: the derivative of the tanh function
    is tanh(x)' = 1 - tanh(x)^2
    input:
        - *gradwrtoutput: list of gradients, provided during backpropagation
          from the modules directly following the current module in the network.
          Must be composed by a single element
    output:
        - gradient to be used from the previous node in the network during
          a backward pass
    """
    def backward(self,*gradwrtoutput):

        if not hasattr(self, 'input'):
            raise ValueError("Backward must be called after at least one forward call")

        if len(gradwrtoutput) > 1:
            raise ValueError("Tanh module expects just one gradient tensor in \
                              backward call!")

        th = np.tanh(self.input[0])
        dsigma = 1 - th*th
        dl_ds = dsigma * gradwrtoutput[0]
        return dl_ds

"""
Sigmoid: Sigmoid activation function. This module computes the sigmoid function of
an input vector. The sigmoid function is defined as
    sigmoid(x) = exp(x) / ( 1 + exp(x) )
"""
class Sigmoid(ActivationFunction):
    """
    Constructor
    """
    def __init__(self):
        super(Sigmoid,self).__init__()

    """
    forward: applies sigmoid function to all the elements of the input
    input:
        - *input: list of inputs. Must be composed by a single input
    output:
        - tensor of same size of the input. Each component is computed by
          applying the sigmoid function to each component of the input
    """
    def forward(self,*input):

        # check input size
        if len(input) > 1:
            raise ValueError("Sigmoid module expects just one input tensor!")

        self.input = input
        expval = np.exp(input[0])
        return np.divide(expval,1 + expval)

    """
    backward: computes backward pass. Note: the derivative of the sigmoid
    function is sigmoid(x)' = sigmoid(x) * ( 1 - sigmoid(x) )
    input:
        - *gradwrtoutput: list of gradients, provided during backpropagation
          from the modules directly following the current module in the network.
          Must be composed by a single element
    output:
        - gradient to be used from the previous node in the network during
          a backward pass
    """
    def backward(self,*gradwrtoutput):

        if not hasattr(self, 'input'):
            raise ValueError("Backward must be called after at least one forward call")

        if len(gradwrtoutput) > 1:
            raise ValueError("Sigmoid module expects just one gradient tensor in \
                              backward call!")

        sigma = self.forward(self.input[0])
        dsigma = np.multiply(sigma,(1 - sigma))
        dl_ds = dsigma * gradwrtoutput[0]
        return dl_ds
