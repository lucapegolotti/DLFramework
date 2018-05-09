import numpy as np
from numpy import random

from torch import FloatTensor as FloatTensor
from torch import LongTensor as LongTensor

import math

def generate_disc_set(nb):
    input = Tensor(nb, 2).uniform_(-1, 1)
    target = input.pow(2).sum(1).sub(2 / math.pi).sign().add(1).div(2).long()
    return input, target


def generate(npoints):
    train_input, train_target = generate_disc_set(npoints)
    test_input, test_target = generate_disc_set(npoints)

    print("Test case:")
    npointsinside = (np.sum(train_target,axis=0))
    print_str = "\ttrain dataset has " + str(int(npointsinside[0])) + "/" + str(npoints) + " inside the circle"
    print(print_str)
    npointsinside = (np.sum(test_target,axis=0))
    print_str = "\ttest dataset has " + str(int(npointsinside[0])) + "/" + str(npoints) + " inside the circle"
    print(print_str)

    return FloatTensor(train_input),FloatTensor(train_target), FloatTensor(test_input), FloatTensor(test_target)
