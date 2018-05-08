#################### TODO: delete dependencies of pytorch
import torch
from torch import nn
import modules as mm

import numpy as np
####################

from torch import FloatTensor as FloatTensor
from torch import LongTensor as LongTensor

dim1 = 100
dim2 = 3
dim3 = 4
x = FloatTensor(np.random.normal(0,1,size=(dim1,dim2)))
linear = mm.Linear(dim2,dim3)

print(linear.forward(x).size())

print(linear.param())
