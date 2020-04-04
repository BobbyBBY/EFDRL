import random
import json
import torch
import torch.nn as nn
import conv
import numpy as np

from itertools import chain

# ob = [[1,2,3,3,3],[3,3,5,6,7],[3,3,8,9,0],[3,3,8,9,0],[3,3,8,9,0]]
# file = open('maps.txt','w')
# file.write(json.dumps(ob))
# file.close()

# ob = [[[[1,2,3,3,3],[3,3,5,6,7],[3,3,8,9,0],[3,3,8,9,0],[3,3,8,9,0]]]]
# ob = torch.Tensor(ob)
# net = DQN()
# oob = net.forward(ob)
# print(oob.shape)
# a1 = [[[1,2,3]]]
# a2 = [[[1,2,4]]]
# a3 = [[[1,2,5]]]
# a4 = [[[1,2,6]]]
# ob = torch.Tensor(ob)
# a1 = torch.Tensor(a1)
# a2 = torch.Tensor(a2)
# a3 = torch.Tensor(a3)
# a4 = torch.Tensor(a4)
# ob = a1
# # ob = torch.cat((ob,a1),1)
# ob = torch.cat((ob,a2),0)
# ob = torch.cat((ob,a3),0)
# ob = torch.cat((ob,a3),0)
# print(ob)
# print(ob.shape)
# ob = ob[1:4,:]
# print(ob)
# print(ob.shape)
# ob = torch.cat((ob,a4),0)
# print(ob)


# torch.save(net1, 'net1.pkl')  # 方式1：保存整个神经网络
# torch.save(net1.state_dict(), 'net1_parameters.pkl')  # 方式2：保存神经网络的参数    

# net2 = torch.load('net1.pkl')
# net3.load_state_dict(torch.load('net1_parameters.pkl'))  # 提取net1的状态参数，将状态参数给net3

terminals = torch.zeros(3, dtype=torch.bool)
print(terminals)