import numpy as np
from numpy import random

from torch import FloatTensor as FloatTensor
from torch import LongTensor as LongTensor

import math

# sample n points in the square (0,1) x (0,1) and label them 1 if they are inside
# the circle with radius 1/sqrt(2*pi), 0 otherwise
def sample(npoints):
    input = np.random.uniform(0,1,size=(npoints,2))
    target = np.zeros(shape=(npoints,2))
    radius_sq = 1/(2 * math.pi)
    for i in range(npoints):
        if (input[i,0] * input[i,0] + input[i,1] * input[i,1] < radius_sq):
            target[i,0] = 1
        else:
            target[i,1] = 1
    return input, target

# generate the datasets
def generate(npoints):
    train_input, train_target = sample(npoints)
    test_input, test_target = sample(npoints)

    return FloatTensor(train_input),FloatTensor(train_target), \
           FloatTensor(test_input), FloatTensor(test_target)
