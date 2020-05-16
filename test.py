import copy
import torch
import torch.nn as nn
import conv
import torch.optim as optim

class DQN_single(nn.Module):
    def __init__(self):
        super(DQN_single, self).__init__()
        # self.f1 = nn.Sequential(
        #     nn.Linear(4, 4),
        #     nn.ReLU()
        # )
        self.f3 = nn.Linear(4, 4)

    def forward(self, x):
        # x = self.f1(x)
        return self.f3(x)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # self.f1 = nn.Sequential(
        #     nn.Linear(4,4),
        #     nn.ReLU()
        # )
        self.f2 =  nn.Linear(4,4)

    def forward(self, x):
        # x = self.f1(x)
        return self.f2(x)

class DQN_single2(nn.Module):
    def __init__(self):
        super(DQN_single2, self).__init__()
        # self.f1 = nn.Sequential(
        #     nn.Linear(4,4),
        #     nn.ReLU()
        # )
        self.f2 =  nn.Linear(4,4)
        # self.f3 = nn.Sequential(
        #     nn.Linear(4,4),
        #     nn.ReLU()
        # )
        self.f4 =  nn.Linear(4,4)

    def forward(self, x):
        # x = self.f1(x)
        x = self.f2(x)
        # x = self.f3(x)
        return self.f4(x)

# 损失函数，平方损失函数
def Square_loss(self, x, y):
    return torch.mean(torch.pow((x - y), 2))
def swap(qvalue,i,j):
    qvalue[i]+=qvalue[j]
    qvalue[j]=qvalue[i]-qvalue[j]
    qvalue[i]-=qvalue[j]
    pass
if torch.cuda.is_available():
    # device = torch.device("cuda")
    device = torch.device("cpu")
    print("CUDA is available.")
else:
    device = torch.device("cpu")
    print("CUDA is not available, fall back to CPU.")

net = DQN_single().to(device)
net2 = DQN_single2().to(device)
mlp = MLP().to(device)
optimizer1 = optim.SGD(net.parameters(), lr=0.001) # 优化器
optimizer2 = optim.SGD(mlp.parameters(), lr=0.001) # 优化器
optimizer3 = optim.SGD(net2.parameters(), lr=0.001) # 优化器
criterion = torch.nn.MSELoss(reduction='mean').to(device)
num=0
while True:
    num += 1
    x = torch.rand(1, 4)[0].to(device)
    # x2 = copy.deepcopy(x)
    x *= 15
    # print(x)
    # result = [x[0]+5,x[1]+10,x[2]+15,x[3]+20]
    # result2 = [x[0]+5,x[1]+10,x[2]+15,x[3]+20]
    result = [x[3],x[2],x[1],x[0]]
    # result2 = [x[3],x[2],x[1],x[0]]
    result = torch.Tensor(result).to(device)
    # result2 = torch.Tensor(result2).to(device)
    targets = net.forward(x)
    # swap(targets1,0,3)
    # swap(targets1,1,2)
    # targets = mlp.forward(targets1)
    # targets2 = net2.forward(x2)
    loss = criterion(result, targets)
    # loss2 = criterion(result2, targets2)
    optimizer1.zero_grad()
    # optimizer2.zero_grad()
    loss.backward()
    optimizer1.step()
    # optimizer2.step()
    # optimizer3.zero_grad()
    # loss2.backward()
    # optimizer3.step()
    if num%200==199 :
        num=0
        print("---------------------------------------------------")
        print(x.cpu().detach().numpy())
        print(loss.item())
        print(targets.cpu().detach().numpy())
        print(result.cpu().detach().numpy())
        print(result.cpu().detach().numpy()-targets.cpu().detach().numpy())
        # print("\n")
        
        print("\n")
