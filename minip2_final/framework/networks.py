import sys
import os
sys.path.append(os.path.dirname(__file__))

import modules
import criterions

"""
Network: template of a generic neural network
"""
class Network(object):
    """
    Constructor
    input:
        - criterion: loss function to be used during the gradient descent
    """
    def __init__(self,criterion):
        self.criterion = criterion

    """
    forward: compute a single forward pass
    input:
        - *inputs: list of inputs to the network
    """
    def forward(self,*inputs):
        raise NotImplementedError

    """
    __iter__: overloads the default __init__ method. This "abstract" method
    must be implemented by derived classes and should provide a way to loop
    over the elements of the network
    """
    def __iter__(self):
        raise NotImplementedError

    """
    backwardCall: "abstract" method which must be implemented by the derived
    classes. The implementation depends on the topology of the underlying graph
    input:
        - gradloss: gradient with respect to the loss function
    """
    def backwardCall(self,gradloss):
        raise NotImplementedError

    """
    backward: computes a single backward pass
    input:
        - output: output of a fowrard pass
        - expected: target of the forward pass
    """
    def backward(self,output,expected):
        value_loss_grad = self.criterion.grad(output,expected)
        self.backwardCall(value_loss_grad)
        return self.criterion.function(output,expected)

    """
    resetGradients: sets to zero all the gradients of the modules
    """
    def resetGradients(self):
        for m in self:
            m.resetGradient()

    """
    updateWeights: update the weights of modules according to the computed
    gradients
    input:
        - eta: learning rate
        - nsamples: number of samples, which is used to scale the learning
                    rate
    """
    def updateWeights(self,eta,nsamples):
        # scale eta by the number of samples
        eta = eta / nsamples
        for m in self:
            m.updateWeights(eta)

"""
Sequential: template of a sequential neural network. Each network must have
exactly one input node and one output node. All the remaining nodes must
accept exactly one input and must return exactly one outuput.
"""
class Sequential(Network):
    """
    Constructor
    input:
        - criterion: loss function to be used during the gradient descent
    """
    def __init__(self,criterion):
        super(Sequential,self).__init__(criterion)
        self.modules_list = []
        self.modules_registered = False

    """
    registerModules: register all the modules of the network
    input:
        - criterion: loss function to be used during the gradient descent
    """
    def registerModules(self,*modules):
        self.modules_registered = True
        for m in modules:
            self.modules_list.append(m)

    """
    checkIfModulesAreRegistered: sanity check, to verify that the object
    has been initialized correctly
    """
    def checkIfModulesAreRegistered(self):
        if (self.modules_registered is False):
            raise RuntimeError('No modules were registered in the Sequential net! \
                                Call registerModules in the constructor')

    """
    __iter__: overloads the __iter__ method in Network. The modules are
    internally stored as a list, which is simply run over in this method
    """
    def __iter__(self):
        self.checkIfModulesAreRegistered()
        for m in self.modules_list:
            yield m

    """
    backwardCall: overloads the backwardCall in Network.
    input:
        - gradloss: gradient of the loss function
    """
    def backwardCall(self,gradloss):
        self.checkIfModulesAreRegistered()

        grad = gradloss
        for m in reversed(self.modules_list):
            grad = m.backward(grad)

        return grad
