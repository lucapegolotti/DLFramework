import modules as mm

from torch import FloatTensor as FloatTensor
from torch import LongTensor as LongTensor

x = FloatTensor([1,2])
linear = mm.Linear(2,2)

print(linear.forward(x))
