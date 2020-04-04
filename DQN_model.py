import torch
import torch.nn as nn
import conv


class DQN(nn.Module):
    def __init__(self, history_length, action_length, state_dim):
        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.c1 = nn.Sequential(     
            conv.Conv2d(history_length, 32, state_dim, 1), 
            nn.ReLU()     
        )
        self.f1 = nn.Sequential(
            nn.Linear(32*state_dim*state_dim, 256),
            nn.ReLU()
        )
        self.f2 =  nn.Linear(256, action_length)

    def forward(self, x):
        x = self.c1(x)
        x = x.view(-1,self.state_dim*self.state_dim*32)
        x = self.f1(x)
        return self.f2(x)