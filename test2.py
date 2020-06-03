import copy
import torch
import torch.nn as nn
import random
import torch.optim as optim
import numpy as np
# from scipy import *
from math import log
import time

x = [[1,2,3,4],[5,6,7,8]]
x = torch.Tensor(x)
list = range(0,16)
resultList=random.sample(list,16)
print(resultList)
y= [12,-5,14,5]
# y = y.expand(2, 4)
# print(y)
# z = torch.cat((y, y),0)
# print(z)
# x = torch.rand(4)
# print(x)

# print(np.random.rand(16)*50 - 25)
def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# while True:
#     time.sleep(2)
#     print(get_time())
for i in range(4):
    print("i:{}".format(i))
    for j in range(4):
        print("j:{}".format(j))
        break
    else:
        continue
    break
