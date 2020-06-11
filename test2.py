import copy
import torch
import torch.nn as nn
import random
import torch.optim as optim
import numpy as np
# from scipy import *
from math import log
import time
import hashlib

# x = [[1,2,3,4],[5,6,7,8]]
# x = torch.Tensor(x)
# list = range(0,16)
# resultList=random.sample(list,16)
# print(resultList)
# y= [12,-5,14,5]
# # y = y.expand(2, 4)
# # print(y)
# # z = torch.cat((y, y),0)
# # print(z)
# # x = torch.rand(4)
# # print(x)

# # print(np.random.rand(16)*50 - 25)
key="123"
md5_str=hashlib.md5(key.encode("utf8")).hexdigest()
seed = int(md5_str[0:10], 16)
torch.manual_seed(seed)
linear_exchange = torch.rand(16)*50 - 25
print(linear_exchange)