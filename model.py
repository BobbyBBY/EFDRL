import torch
import torch.nn as nn
import conv

class DQN_single(nn.Module):
    def __init__(self, history_length, action_length, state_dim):
        super(DQN_single, self).__init__()
        self.state_dim = state_dim
        self.c1 = nn.Sequential(     
            conv.Conv2d(history_length, 32, state_dim, 1), 
            nn.ReLU()     
        )
        self.f1 = nn.Sequential(
            nn.Linear(32*state_dim*state_dim, 256),
            nn.ReLU()
        )
        self.f2 = nn.Linear(256, action_length)

    def forward(self, x):
        x = self.c1(x)
        x = x.view(-1,self.state_dim*self.state_dim*32)
        x = self.f1(x)
        return self.f2(x)

class DQN_full(nn.Module):
    def __init__(self, history_length, action_length, state_alpha_dim, state_beta_dim):
        super(DQN_full, self).__init__()
        self.state_alpha_dim = state_alpha_dim
        self.state_beta_dim = state_beta_dim
        self.c1 = nn.Sequential(     
            conv.Conv2d(history_length, 32, state_alpha_dim, 1), 
            nn.ReLU()     
        )
        self.c2 = nn.Sequential(     
            conv.Conv2d(history_length, 32, state_beta_dim, 1), 
            nn.ReLU()     
        )
        self.f1 = nn.Sequential(
            nn.Linear(32*state_alpha_dim*state_alpha_dim + 32*state_beta_dim*state_beta_dim, 256),
            nn.ReLU()
        )
        self.f2 =  nn.Linear(256, action_length)

    def forward(self, alpha, beta):
        alpha = self.c1(alpha)
        beta = self.c2(beta)
        alpha = alpha.view(-1,self.state_alpha_dim*self.state_alpha_dim*32)
        beta = beta.view(-1,self.state_beta_dim*self.state_beta_dim*32)
        x = torch.cat((alpha, beta),1)
        x = self.f1(x)
        return self.f2(x)

class MLP(nn.Module):
    def __init__(self, action_length):
        super(MLP, self).__init__()
        # 独占式g(x)需要非relu linear模拟
        # self.f_exc = nn.Linear(action_length, action_length)
        self.f1 = nn.Sequential(
            nn.Linear(2*action_length, 2*action_length),
            nn.ReLU()
        )
        # self.f1 = nn.Linear(2*action_length, 2*action_length)
        self.f2 =  nn.Linear(2*action_length, action_length)

    def forward(self, alpha, beta):
        # x_a = self.f_exc(alpha)
        x = torch.cat((alpha, beta),1)
        x = self.f1(x)
        return self.f2(x)
