import sys
import os
sys.path.append(os.path.dirname(__file__))

import modules
import criterions

class Network(object):
    def __init__(self,criterion):
        self.criterion = criterion

    def forward(self,*inputs):
        raise NotImplementedError

    # derived classes must implement a way to loop over the modules
    def __iter__(self):
        raise NotImplementedError

    def backwardCall(self,*gradwrtoutput):
        raise NotImplementedError

    def backward(self,output,expected):
        value_loss_grad = self.criterion.grad(output,expected)
        self.backwardCall(value_loss_grad)
        return self.criterion.function(output,expected)

    def resetGradients(self):
        for m in self:
            m.resetGradient()

    def updateParameters(self,eta,nsamples):
        # scale eta by the number of samples
        eta = eta / nsamples
        for m in self:
            m.updateParameters(eta)

class Sequential(Network):
    def __init__(self,criterion):
        super(Sequential,self).__init__(criterion)
        self.modules_list = []
        self.modules_registered = False

    def registerModules(self,*modules):
        self.modules_registered = True
        for m in modules:
            self.modules_list.append(m)

    def checkIfModulesAreRegistered(self):
        if (self.modules_registered is False):
            raise RuntimeError('No modules were registered in the Sequential net! \
                                Call registerModules in the constructor')

    # provide iter method
    def __iter__(self):
        self.checkIfModulesAreRegistered()
        for m in self.modules_list:
            yield m

    def backwardCall(self,*gradwrtoutput):
        self.checkIfModulesAreRegistered()

        grad = gradwrtoutput[0]
        for m in reversed(self.modules_list):
            grad = m.backward(grad)

        return grad
