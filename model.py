import torch
import torch.nn as nn
import conv

class DQN_single(nn.Module):
    def __init__(self, batch_size,history_length, action_length, state_dim):
        super(DQN_single, self).__init__()
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.c1 = nn.Sequential(     
            conv.Conv2d(history_length, batch_size, state_dim, 1), 
            nn.ReLU()     
        )
        self.f1 = nn.Sequential(
            nn.Linear(batch_size*state_dim*state_dim, 256),
            nn.ReLU()
        )
        self.f2 = nn.Linear(256, action_length)

    def forward(self, x):
        x = self.c1(x)
        x = x.view(-1,self.state_dim*self.state_dim*self.batch_size)
        x = self.f1(x)
        return self.f2(x)

class DQN_full(nn.Module):
    def __init__(self,batch_size, history_length, action_length, state_alpha_dim, state_beta_dim):
        super(DQN_full, self).__init__()
        self.state_alpha_dim = state_alpha_dim
        self.state_beta_dim = state_beta_dim
        self.batch_size = batch_size
        self.c1 = nn.Sequential(     
            conv.Conv2d(history_length, batch_size, state_alpha_dim, 1), 
            nn.ReLU()     
        )
        self.c2 = nn.Sequential(     
            conv.Conv2d(history_length, batch_size, state_beta_dim, 1), 
            nn.ReLU()     
        )
        self.f1 = nn.Sequential(
            nn.Linear(batch_size*state_alpha_dim*state_alpha_dim + batch_size*state_beta_dim*state_beta_dim, 256),
            nn.ReLU()
        )
        self.f2 =  nn.Linear(256, action_length)

    def forward(self, alpha, beta):
        alpha = self.c1(alpha)
        beta = self.c2(beta)
        alpha = alpha.view(-1,self.state_alpha_dim*self.state_alpha_dim*self.batch_size)
        beta = beta.view(-1,self.state_beta_dim*self.state_beta_dim*self.batch_size)
        x = torch.cat((alpha, beta),1)
        x = self.f1(x)
        return self.f2(x)

class MLP(nn.Module):
    def __init__(self, action_length):
        super(MLP, self).__init__()
        self.f1 = nn.Sequential(
            nn.Linear(2*action_length, 2*action_length),
            nn.ReLU()
        )
        self.f2 =  nn.Linear(2*action_length, action_length)

    def forward(self, alpha, beta):
        x = torch.cat((alpha, beta),1)
        x = self.f1(x)
        return self.f2(x)
