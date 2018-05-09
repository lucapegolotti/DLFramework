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
        if (input[i,0] * input[i,0] + input[i,1] * input[i,1] < radius_sq):
            target[i,0] = 0.9
            target[i,1] = -0.9
        else:
            target[i,1] = 0.9
            target[i,0] = -0.9
    return input, target

def generate(npoints):
    train_input, train_target = sample(npoints)
    test_input, test_target = sample(npoints)

    print("Test case:")
    npointsinside = (np.sum(train_target,axis=0))
    print_str = "\ttrain dataset has " + str(int(npointsinside[0])) + "/" + str(npoints) + " inside the circle"
    print(print_str)
    npointsinside = (np.sum(test_target,axis=0))
    print_str = "\ttest dataset has " + str(int(npointsinside[0])) + "/" + str(npoints) + " inside the circle"
    print(print_str)

    return FloatTensor(train_input),FloatTensor(train_target), FloatTensor(test_input), FloatTensor(test_target)
