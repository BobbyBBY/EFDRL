import copy
import torch
import torch.nn as nn
import conv
import torch.optim as optim
import numpy as np
from scipy import *
from math import log

x = [[1,2,3,4],[5,6,7,8]]
x = torch.Tensor(x)
y= [12,-5,14,5]
y = torch.Tensor(y)
y = y.expand(2, 4)
print(y)
z = torch.cat((y, y),0)
print(z)
x = torch.rand(4)
print(x)


