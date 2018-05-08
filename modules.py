from torch import FloatTensor as floatTensor
from torch import LongTensor as longTensor

class Module(object):
    def forward(self,*input):
        raise NotImplementedError

    def backward(self,*gradwrtoutput):
        raise NotImplementedError

    def param(self) :
        return []
