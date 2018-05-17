import modules as M
import criterions as C
import networks as N

from torch import FloatTensor as FloatTensor
from torch import LongTensor as LongTensor

import numpy as np

import torch

class bcolors:
    header = '\033[95m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    fail = '\033[91m'
    endc = '\033[0m'
    bold = '\033[1m'
    underline = '\033[4m'

class Test(object):
    def __init__(self):
        self.runs = 0
        self.passed = 0
        print("hello")

    def run(self):
        print(bcolors.header + "Running " + self.__class__.__name__ + " tests ..." + bcolors.endc)
        print(bcolors.header + "=================================================" + bcolors.endc)
        for val in dir(self):
            if val[0:4] == "test":
                test_method = getattr(self, val)
                self.runs += 1
                print_string = "\t" + val
                status = test_method()
                if not status:
                    self.passed += 1
                    print(print_string + bcolors.yellow + " passed!" + bcolors.endc)
                else:
                    print(print_string + bcolors.fail  + " failed!" + bcolors.endc)

        if self.runs == self.passed:
            print(bcolors.green + str(self.passed) + "/" + str(self.runs) + " test passed!" + bcolors.endc)
        else:
            print(bcolors.fail + str(self.passed) + "/" + str(self.runs) + " test passed!" + bcolors.endc)

        return self.runs, self.passed

class TestModule(Test):
    def test_forward(self):
        m = M.Module()
        try:
            m.forward([])
        except NotImplementedError:
            return 0

        return 1

    def test_backward(self):
        m = M.Module()
        try:
            m.backward([])
        except NotImplementedError:
            return 0

        return 1

    def test_resetGradient(self):
        m = M.Module()
        try:
            m.resetGradient()
        except NotImplementedError:
            return 0

        return 1

    def test_updateWeights(self):
        m = M.Module()
        try:
            m.updateWeights(0.1)
        except NotImplementedError:
            return 0

        return 1

    def test_params(self):
        m = M.Module()
        if len(m.param()) == 0:
            return 0

        return 1

class TestLinear(Test):
    def test_constructor(self):
        in_features = 4
        out_features = 10

        m = M.Linear(in_features, out_features)

        if m.weights.size() != m.weights_grad.size():
            return 1

        if m.bias.size() != m.bias_grad.size():
            return 1

        if m.weights.size() != (out_features,in_features):
            return 1

        if m.bias.size() != (1,out_features):
            return 1

        return 0

    def test_forwardWrongInputListDimension(self):
        in_features = 4
        out_features = 10

        m = M.Linear(in_features, out_features)
        try:
            m.forward(FloatTensor([[1,2],[3,4]]),FloatTensor([[1,2,3],[3,4,1]]))
        except ValueError:
            return 0

        return 1

    def test_forwardWrongInputDimension(self):
        in_features = 3
        out_features = 10

        m = M.Linear(in_features, out_features)
        try:
            m.forward(FloatTensor([[1,2]]))
        except ValueError:
            return 0

        return 1

    def test_forwardCorrectOutput(self):
        in_features = 3
        out_features = 2

        x = FloatTensor([[1,2,3],[3,2,-1]])
        A = FloatTensor([[1,2,3],[4,5,6]])
        b = FloatTensor([[1,2]])

        expected_output = FloatTensor([[15,34],[5,18]])

        m = M.Linear(in_features, out_features)
        m.weights = A
        m.bias = b
        output = m.forward(x)
        if torch.max(torch.abs(expected_output-output)):
            return 1

        return 0

    def test_backwardBeforeForward(self):
        in_features = 1
        out_features = 10

        m = M.Linear(in_features, out_features)
        try:
            m.backward(FloatTensor([[1]]))
        except ValueError:
            return 0

        return 1

    def test_backwardWrongInputDimension(self):
        in_features = 1
        out_features = 10

        m = M.Linear(in_features, out_features)
        m.forward(FloatTensor([[1]]))
        try:
            m.backward(FloatTensor([[1,2],[3,4]]),FloatTensor([[1,2,3],[3,4,1]]))
        except ValueError:
            return 0

        return 1

    def test_backwardCorrectOutput(self):
        in_features = 3
        out_features = 2

        x = FloatTensor([[1,2,3],[3,2,-1]])
        A = FloatTensor([[1,2,3],[4,5,6]])
        b = FloatTensor([[1,2]])

        expected_output = FloatTensor([[15,34],[5,18]])

        m = M.Linear(in_features, out_features)
        m.weights = A
        m.bias = b
        m.forward(x)

        grad = FloatTensor(([1,4],[-1,-1]))
        output = m.backward(grad)

        expected_output = grad.mm(A)

        if torch.max(torch.abs(expected_output-output)):
            return 1

        return 0

    def test_resetGradient(self):
        in_features = 3
        out_features = 2

        x = FloatTensor([[1,2,3],[3,2,-1]])
        A = FloatTensor([[1,2,3],[4,5,6]])
        b = FloatTensor([[1,2]])

        expected_output = FloatTensor([[15,34],[5,18]])

        m = M.Linear(in_features, out_features)
        m.weights = A
        m.bias = b
        m.forward(x)

        grad = FloatTensor(([1,4],[-1,-1]))
        output = m.backward(grad)

        m.resetGradient()

        if torch.max(torch.abs(m.weights_grad)) != 0 or \
           torch.max(torch.abs(m.bias_grad)) != 0:
           return 1

        return 0

    def test_updateWeights(self):
        in_features = 3
        out_features = 2

        x = FloatTensor([[1,2,3],[3,2,-1]])
        A = FloatTensor([[1,2,3],[4,5,6]])
        b = FloatTensor([[1,2]])

        expected_output = FloatTensor([[15,34],[5,18]])

        m = M.Linear(in_features, out_features)
        m.weights = A
        m.bias = b
        m.forward(x)

        grad = FloatTensor(([1,4],[-1,-1]))
        output = m.backward(grad)

        expected_gradient = grad.transpose(0,1).mm(x)

        eta = 0.1
        m.updateWeights(eta)

        new_value = A - eta * expected_gradient
        if torch.max(torch.abs(m.weights - new_value)) != 0:
            return 1

        new_value = b - eta * FloatTensor(np.sum(grad.numpy(),axis=0))
        if torch.max(torch.abs(m.bias - new_value)) != 0:
            return 1

        return 0

tests = [TestModule(), TestLinear()]

for i in tests:
    i.run()
