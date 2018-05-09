import numpy as np
from numpy import random

from torch import FloatTensor as FloatTensor
from torch import LongTensor as LongTensor

import math

def sample(npoints):
    input = np.random.uniform(0,1,size=(npoints,2))
    target = np.zeros(shape=(npoints,2))
    radius_sq = 1/(2 * math.pi)
    for i in range(npoints):
        if (input[i,0] > input[i,1]):
            target[i,0] = 1
        else:
            target[i,1] = 1
    return input, target

def generate(npoints):
    train_input, train_target = sample(npoints)
    test_input, test_target = sample(npoints)

    return FloatTensor(train_input),FloatTensor(train_target), FloatTensor(test_input), FloatTensor(test_target)
