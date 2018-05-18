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

def areEqual(tensor1,tensor2,tol=1e-15):
    if torch.max(torch.abs(tensor1 - tensor2)) > tol:
        return 0
    return 1

class Test(object):
    def __init__(self):
        self.runs = 0
        self.passed = 0

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
        if not areEqual(expected_output,output):
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

        if not areEqual(expected_output,output):
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
        if not areEqual(m.weights,new_value):
            return 1

        new_value = b - eta * FloatTensor(np.sum(grad.numpy(),axis=0))
        if not areEqual(m.bias,new_value):
            return 1

        return 0

class TestReLU(Test):
    def test_forwardWrongInputListDimension(self):
        m = M.ReLU()

        try:
            m.forward(FloatTensor([[2]]),FloatTensor([[2]]))
        except ValueError:
            return 0

        return 1

    def test_forwardCorrectOutput(self):
        input = FloatTensor([[1,-3,1],[0,-1,2]])
        expected_output = FloatTensor([[1,0,1],[0,0,2]])

        m = M.ReLU()
        output = m.forward(input)

        if not areEqual(output,expected_output):
            return 1

        return 0

    def test_backwardBeforeForward(self):
        input = FloatTensor([[1,-3,1],[0,-1,2]])

        m = M.ReLU()

        try:
            output = m.backward(input)
        except:
            return 0

        return 1

    def test_backwardCorrectOutput(self):
        input = FloatTensor([[1,-3,1],[0,-1,2]])
        dsigma = FloatTensor([[1,0,1],[1,0,1]])

        m = M.ReLU()
        m.forward(input)
        grad = FloatTensor([[1,2,3],[3,2,1]])

        output = m.backward(grad)
        output_expected = dsigma * grad

        if not areEqual(output,output_expected):
            return 1

        return 0

class TestTanh(Test):
    def test_forwardWrongInputListDimension(self):
        m = M.Tanh()

        try:
            m.forward(FloatTensor([[2]]),FloatTensor([[2]]))
        except ValueError:
            return 0

        return 1

    def test_forwardCorrectOutput(self):
        input = FloatTensor([[1,-3,1],[0,-1,2]])
        expected_output = FloatTensor([[0.761594155955765,-0.995054753686730,0.761594155955765],\
                                       [0,-0.761594155955765,0.964027580075817]])

        m = M.Tanh()
        output = m.forward(input)

        if not areEqual(output,expected_output, tol = 1e-5):
            return 1

        return 0

    def test_backwardBeforeForward(self):
        input = FloatTensor([[1,-3,1],[0,-1,2]])

        m = M.Tanh()

        try:
            output = m.backward(input)
        except:
            return 0

        return 1

    def test_backwardCorrectOutput(self):
        input = FloatTensor([[1,-3,1],[0,-1,2]])

        tanh_value = FloatTensor([[0.761594155955765,-0.995054753686730,0.761594155955765],\
                                  [0,-0.761594155955765,0.964027580075817]])

        dsigma = 1 - tanh_value*tanh_value

        m = M.Tanh()
        m.forward(input)
        grad = FloatTensor([[1,2,3],[3,2,1]])

        output = m.backward(grad)
        output_expected = dsigma * grad

        if not areEqual(output,output_expected):
            return 1

        return 0

class TestSigmoid(Test):
    def test_forwardWrongInputListDimension(self):
        m = M.Sigmoid()

        try:
            m.forward(FloatTensor([[2]]),FloatTensor([[2]]))
        except ValueError:
            return 0

        return 1

    def test_forwardCorrectOutput(self):
        input = FloatTensor([[1,-3,1],[0,-1,2]])
        expected_output = FloatTensor([[0.731058578630005,0.047425873177567,0.731058578630005], \
                                        [0.500000000000000,0.268941421369995,0.880797077977882]])

        m = M.Sigmoid()
        output = m.forward(input)
        if not areEqual(output,expected_output, tol = 1e-5):
            return 1

        return 0

    def test_backwardBeforeForward(self):
        input = FloatTensor([[1,-3,1],[0,-1,2]])

        m = M.Sigmoid()

        try:
            output = m.backward(input)
        except:
            return 0

        return 1

    def test_backwardCorrectOutput(self):
        input = FloatTensor([[1,-3,1],[0,-1,2]])

        sigmoid_value = FloatTensor([[0.731058578630005,0.047425873177567,0.731058578630005], \
                                     [0.500000000000000,0.268941421369995,0.880797077977882]])

        dsigma = sigmoid_value * (1 - sigmoid_value)

        m = M.Sigmoid()
        m.forward(input)
        grad = FloatTensor([[1,2,3],[3,2,1]])

        output = m.backward(grad)
        output_expected = dsigma * grad

        if not areEqual(output,output_expected):
            return 1

        return 0

class TestLoss(Test):
    def test_function(self):
        loss = C.Loss()
        try:
            loss.function()
        except:
            return 0
        return 1

    def test_grad(self):
        loss = C.Loss()
        try:
            loss.gradient()
        except:
            return 0
        return 1

class TestLossMSE(Test):
    def test_function(self):
        loss = C.LossMSE()

        x = FloatTensor([[1,2,3],[-4,2,3]])
        y = FloatTensor([[3,2,0],[-1,2,3]])

        res = loss.function(x,y)

        expected_res = 22/6

        if abs(res-expected_res) > 1e-10:
            return 1

        return 0

    def test_grad(self):
        loss = C.LossMSE()

        x = FloatTensor([[1,2,3],[-4,2,3]])
        y = FloatTensor([[3,2,0],[-1,2,3]])

        output = loss.grad(x,y)

        expected_output = FloatTensor([[-4,0,6],[-6,0,0]])
        if not areEqual(output,expected_output):
            return 1

        return 0

class TestNetwork(Test):
    def test_forward(self):
        loss = C.LossMSE()
        m = N.Network(loss)

        try:
            m.forward()
        except:
            return 0

        return 1

    def test_backward(self):
        loss = C.LossMSE()
        m = N.Network(loss)

        try:
            m.backward()
        except:
            return 0

        return 1

    def test_iter(self):
        loss = C.LossMSE()
        m = N.Network(loss)

        try:
            m.__iter__()
        except:
            return 0

        return 1

    def test_backwardCall(self):
        loss = C.LossMSE()
        m = N.Network(loss)

        try:
            m.backwardCall()
        except:
            return 0

        return 1

    def test_backward(self):
        x = FloatTensor([[1,2,3],[-4,2,3]])
        y = FloatTensor([[3,2,0],[-1,2,3]])

        loss = C.LossMSE()
        m = N.Network(loss)

        try:
            m.backward(x,y)
        except:
            return 0

        return 1

    def test_resetGradients(self):
        loss = C.LossMSE()
        m = N.Network(loss)

        try:
            m.resetGradients(x,y)
        except:
            return 0

        return 1

    def test_updateWeights(self):
        loss = C.LossMSE()
        m = N.Network(loss)

        try:
            m.updateWeights(0.01)
        except:
            return 0

        return 1

class SequentialChild(N.Sequential):
    def __init__(self,criterion):
        super(SequentialChild,self).__init__(criterion)
        self.mod1 = M.Linear(3,2)
        self.mod2 = M.Linear(2,3)

        self.registerModules(self.mod1, self.mod2)

    def forward(self,*inputs):
        x = inputs[0]
        x = self.mod1.forward(x)
        x = self.mod2.forward(x)
        return x

class TestSequential(Test):
    def test_forwardNotImplemented(self):
        loss = C.LossMSE()
        m = N.Sequential(loss)

        try:
            m.forward()
        except:
            return 0

        return 1
    def test_forward(self):
        loss = C.LossMSE()
        m = SequentialChild(loss)

        x = FloatTensor([[1,2,3],[4,3,2]])

        output = m.forward(x)

        if output.size() == (2,3):
            return 0

        return 1

    def test_backward(self):
        loss = C.LossMSE()

        m = SequentialChild(loss)

        x = FloatTensor([[2,1,3],[4,3,2]])
        y = FloatTensor([[1,2,3],[4,-2,2]])

        output = m.forward(x)
        x = output
        output = m.backward(x,y)

        l_value = loss.function(x,y)

        if abs(l_value - output) < 1e-10:
            return 0

        return 1


tests = [TestModule(), TestLinear(), TestReLU(), TestTanh(), TestSigmoid(), \
         TestLoss(), TestLossMSE(), TestNetwork(), TestSequential()]

runs = 0
passed = 0
for i in tests:
    r,p = i.run()
    runs += r
    passed += p

if runs == passed:
    color = bcolors.green
else:
    color = bcolors.fail

print(color + "************************************************" + bcolors.endc)
print(color + "****** Test finished : " + str(passed) + "/" + str(runs) + " test passed! ******" + bcolors.endc)
print(color + "************************************************" + bcolors.endc)
