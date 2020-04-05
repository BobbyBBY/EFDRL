import torch
import torch.nn as nn
import torch.optim as optim
import copy
import DQN_model

class FRLDQN(object):

    """docstring for FRLNetwork"""
    def __init__(self,  args):
        self.args = args
        self.gamma = args.gamma
        self.lambda_ = args.lambda_
        self.preset_lambda = args.preset_lambda
        self.add_train_noise = args.add_train_noise
        self.add_predict_noise = args.add_predict_noise
        self.noise_prob = args.noise_prob
        self.stddev = args.stddev
        self.num_actions = args.num_actions
        self.learning_rate = args.learning_rate
        self.hist_len = args.hist_len
        self.state_alpha_dim = args.state_dim + args.image_padding * 2
        self.device = args.device_type
        # self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        self.criterion = self.Square_loss
        self.build_dqn()


    def build_dqn(self):
        # construct DQN-alpha network
        self.alpha_q = DQN_model.DQN(self.hist_len, self.num_actions, self.state_alpha_dim).to(self.device)
        self.alpha_t_q = DQN_model.DQN(self.hist_len, self.num_actions, self.state_alpha_dim).to(self.device)
        self.optimizer = optim.SGD(self.alpha_q.parameters(), lr=self.learning_rate) # 优化器

    # 损失函数，平方损失函数
    def Square_loss(self, x, y):
        return torch.mean(torch.pow((x - y), 2))

    def update_target_network(self):
        self.alpha_t_q=copy.deepcopy(self.alpha_q)

    def train(self, minibatch):
        pre_states_alpha, actions, rewards, post_states_alpha, terminals = minibatch
        # https://www.jianshu.com/p/fae51b5fe000
        # 使用的是同一张Q值表，或者说使用的同一个网络参数，这可能导致选择过高的估计值，从而导致过于乐观的值估计。为了避免这种情况的出现，我们可以对选择和衡量进行解耦，从而就有了双Q学习
        # DoubleDQN
        targets = self.alpha_t_q.forward(post_states_alpha)
        max_postq = torch.max(targets,1)[0]
        tempQ_yi = self.alpha_q.forward(pre_states_alpha)
        tempQ = tempQ_yi.clone()
        for i, action in enumerate(actions):
            if terminals[i]:
                tempQ_yi[i][action] = rewards[i]
            else:
                tempQ_yi[i][action] = rewards[i] + self.gamma * max_postq[i]
        loss = self.criterion(tempQ, tempQ_yi)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def predict(self, state_alpha, predict_net):
        if predict_net == 'alpha':
            qvalue = self.alpha_q.forward(torch.Tensor(state_alpha).to(self.device))
        else:
            print('\n Wrong predict mode! \n')
            raise ValueError
        return qvalue[0]
