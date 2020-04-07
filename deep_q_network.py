import os
import model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from utils import save_pkl, load_pkl

class FRLDQN(object):
    """docstring for FRLNetwork"""
    def __init__(self, args):
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
        self.state_beta_dim = args.state_dim
        self.state_alpha_dim = args.state_dim + args.image_padding * 2
        self.device = args.device_type
        self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        # self.criterion = self.Square_loss
        self.build_dqn()

   
    def build_dqn(self):

        # construct DQN-alpha network
        self.alpha_q = model.DQN_single(self.hist_len, self.num_actions, self.state_alpha_dim).to(self.device)
        self.alpha_t_q = model.DQN_single(self.hist_len, self.num_actions, self.state_alpha_dim).to(self.device)
        self.optimizer_dqn_alpha = optim.SGD(self.alpha_q.parameters(), lr=self.learning_rate) # 优化器

        # construct DQN-beta network
        self.beta_q = model.DQN_single(self.hist_len, self.num_actions, self.state_beta_dim).to(self.device)
        self.beta_t_q = model.DQN_single(self.hist_len, self.num_actions, self.state_beta_dim).to(self.device)
        self.optimizer_dqn_beta = optim.SGD(self.beta_q.parameters(), lr=self.learning_rate)
        
        # construct DQN-full network
        self.full_q = model.DQN_full(self.hist_len, self.num_actions, self.state_alpha_dim, self.state_beta_dim).to(self.device)
        self.full_t_q = model.DQN_full(self.hist_len, self.num_actions, self.state_alpha_dim, self.state_beta_dim).to(self.device)
        self.optimizer_dqn_full = optim.SGD(self.full_q.parameters(), lr=self.learning_rate)

        # construct FRL network
        self.frl_q = model.MLP(self.num_actions).to(self.device)
        self.frl_t_q = model.MLP(self.num_actions).to(self.device)
        self.optimizer_frl = optim.SGD(self.frl_q.parameters(), lr=self.learning_rate)

    # 损失函数，平方损失函数
    def Square_loss(self, x, y):
        return torch.mean(torch.pow((x - y), 2))

    def update_target_network(self):
        if self.args.train_mode == 'single_alpha':
            self.alpha_t_q=copy.deepcopy(self.alpha_q)

        elif self.args.train_mode == 'single_beta':
            self.beta_t_q=copy.deepcopy(self.beta_q)

        elif self.args.train_mode == 'full':
           self.full_t_q=copy.deepcopy(self.full_q)

        else:
            if self.preset_lambda:
                self.alpha_t_q=copy.deepcopy(self.alpha_q)
                self.beta_t_q=copy.deepcopy(self.beta_q)

            else:
                self.frl_t_q=copy.deepcopy(self.frl_q)


    def train(self, minibatch):
        pre_states_alpha, pre_states_beta, actions, rewards, post_states_alpha, post_states_beta, terminals = minibatch
        
        if self.args.train_mode == 'single_alpha':
            targets = self.alpha_t_q.forward(post_states_alpha)
            max_postq = torch.max(targets,1)[0]
            tempQ_yi = self.alpha_q.forward(pre_states_alpha)
            tempQ = tempQ_yi.clone()

        elif self.args.train_mode == 'single_beta':
            targets = self.beta_t_q.forward(post_states_beta)
            max_postq = torch.max(targets,1)[0]
            tempQ_yi = self.beta_q.forward(pre_states_beta)
            tempQ = tempQ_yi.clone()

        elif self.args.train_mode == 'full':
            targets = self.full_t_q.forward(post_states_alpha, post_states_beta)
            max_postq = torch.max(targets,1)[0]
            tempQ_yi = self.full_q.forward(pre_states_alpha, pre_states_beta)
            tempQ = tempQ_yi.clone()
        
        else: # frl
            if self.preset_lambda:
                targets_alpha = self.alpha_t_q.forward(post_states_alpha)
                targets_beta = self.beta_t_q.forward(post_states_beta)
                max_postq = torch.max(self.lambda_ * targets_alpha + (1 - self.lambda_) * targets_beta,1)[0]
                tempQ_yi_alpha = self.alpha_q.forward(pre_states_alpha)
                tempQ_yi_beta = self.beta_q.forward(pre_states_beta)
                tempQ_yi = self.lambda_ * tempQ_yi_alpha + (1 - self.lambda_) * tempQ_yi_beta
                tempQ = tempQ_yi.clone()
            else:
                targets_alpha = self.alpha_t_q.forward(post_states_alpha)
                targets_beta = self.beta_t_q.forward(post_states_beta)
                tempQ_yi_alpha = self.alpha_q.forward(pre_states_alpha)
                tempQ_yi_beta = self.beta_q.forward(pre_states_beta)
                if self.add_train_noise and np.random.rand() <= self.noise_prob:
                    # add Gaussian noise to Q-values with self.noise_prob probility 
                    noise_alpha = np.random.normal(0.0, self.stddev, targets_alpha.shape)
                    noise_beta = np.random.normal(0.0, self.stddev, targets_beta.shape)
                    targets_alpha += noise_alpha
                    targets_beta += noise_beta

                    noise_alpha = np.random.normal(0.0, self.stddev, tempQ_yi_alpha.shape)
                    noise_beta = np.random.normal(0.0, self.stddev, tempQ_yi_beta.shape)
                    tempQ_yi_alpha += noise_alpha
                    tempQ_yi_beta += noise_beta
                targets = self.frl_t_q.forward(targets_alpha, targets_beta)
                max_postq = torch.max(targets,1)[0]
                tempQ_yi = self.frl_q.forward(tempQ_yi_alpha, tempQ_yi_beta)
                tempQ = tempQ_yi.clone()

        for i, action in enumerate(actions):
            if terminals[i]:
                tempQ_yi[i][action] = rewards[i]
            else:
                tempQ_yi[i][action] = rewards[i] + self.gamma * max_postq[i]


        if self.args.train_mode == 'single_alpha':  
            loss = self.criterion(tempQ, tempQ_yi)
            self.optimizer_dqn_alpha.zero_grad()
            loss.backward()
            self.optimizer_dqn_alpha.step()

        elif self.args.train_mode == 'single_beta':  
            loss = self.criterion(tempQ, tempQ_yi)
            self.optimizer_dqn_beta.zero_grad()
            loss.backward()
            self.optimizer_dqn_beta.step()

        elif self.args.train_mode == 'full':
            loss = self.criterion(tempQ, tempQ_yi)
            self.optimizer_dqn_full.zero_grad()
            loss.backward()
            self.optimizer_dqn_full.step()

        elif self.args.train_mode == 'frl_lambda':
            loss = self.criterion(tempQ, tempQ_yi)
            self.optimizer_dqn_alpha.zero_grad()
            self.optimizer_dqn_beta.zero_grad()
            loss.backward()
            self.optimizer_dqn_alpha.step()
            self.optimizer_dqn_beta.step()

        elif self.args.train_mode == 'frl_separate':
            # 这里不确定梯度知否会顺着mlp传到dqn a,b网络
            loss = self.criterion(tempQ, tempQ_yi)
            self.optimizer_dqn_alpha.zero_grad()
            self.optimizer_dqn_beta.zero_grad()
            self.optimizer_frl.zero_grad()
            loss.backward()
            self.optimizer_dqn_alpha.step()
            self.optimizer_dqn_beta.step()
            self.optimizer_frl.step()

        else: 
            print('\n Wrong training mode! \n')
            raise ValueError

        return loss


    def predict(self, state_alpha, state_beta, predict_net):
        
        if predict_net == 'alpha':
            qvalue = self.alpha_q.forward(torch.Tensor(state_alpha).to(self.device))

        elif predict_net == 'beta':
            qvalue = self.beta_q.forward(torch.Tensor(state_beta).to(self.device))
        
        elif predict_net == 'full':
            qvalue = self.full_q.forward(torch.Tensor(state_alpha).to(self.device), torch.Tensor(state_beta).to(self.device))

        elif predict_net == 'both':
            q_alpha = self.alpha_q.forward(torch.Tensor(state_alpha).to(self.device))
            q_beta = self.beta_q.forward(torch.Tensor(state_beta).to(self.device))
            if self.preset_lambda:
                qvalue = self.lambda_ * q_alpha + (1 - self.lambda_) * q_beta
            else:
                if self.add_predict_noise:
                    noise_alpha = np.random.normal(0.0, self.stddev, q_alpha.shape)
                    noise_beta = np.random.normal(0.0, self.stddev, q_beta.shape)
                    q_alpha += noise_alpha
                    q_beta += noise_beta
                qvalue = self.frl_q.forward(q_alpha, q_beta)
        else:
            print('\n Wrong predict mode! \n')
            raise ValueError

        return qvalue[0]


    def save_weights(self, weight_dir, net_name):
        pass
        '''
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        if net_name == 'full':
            print('Saving full network weights ...')
            for name in self.full_w:
                save_pkl(self.full_w[name].eval(), os.path.join(weight_dir, "full_%s.pkl" % name))

        elif net_name == 'beta':
            print('Saving beta network weights ...')
            for name in self.beta_w:
                save_pkl(self.beta_w[name].eval(), os.path.join(weight_dir, "beta_%s.pkl" % name))

        elif net_name == 'alpha':
            print('Saving alpha network weights ...')
            for name in self.alpha_w:
                save_pkl(self.alpha_w[name].eval(), os.path.join(weight_dir, "alpha_%s.pkl" % name))

        else:
            if self.preset_lambda:
                print('Saving frl preset_lambda network weights ...')
                for name in self.beta_w:
                    save_pkl(self.beta_w[name].eval(), os.path.join(weight_dir, "beta_%s.pkl" % name))
                
                for name in self.alpha_w:
                    save_pkl(self.alpha_w[name].eval(), os.path.join(weight_dir, "alpha_%s.pkl" % name))  
            else:
                print('Saving frl mlp network weights ...')
                for name in self.frl_w:
                    save_pkl(self.frl_w[name].eval(), os.path.join(weight_dir, "frl_%s.pkl" % name))
        '''


    def load_weights(self, weight_dir):
        pass
        '''
        print('Loading weights from %s ...' % weight_dir)
        if self.args.train_mode == 'full':
            self.full_w_input, self.full_w_assign_op = self.update_q_network_op(self.full_w, 'load_full_pred_from_pkl')
            for name in self.full_w:
                self.full_w_assign_op[name].eval({self.full_w_input[name]: load_pkl(os.path.join(weight_dir, "full_%s.pkl" % name))})

        elif self.args.train_mode == 'frl_separate':
            self.frl_t_w_input, self.frl_w_assign_op = self.update_q_network_op(self.frl_w, 'load_frl_pred_from_pkl')
            for name in self.frl_w:
                self.frl_w_assign_op[name].eval({self.frl_t_w_input[name]: load_pkl(os.path.join(weight_dir, 'frl_%s.pkl' % name))})
        
        else:
            self.beta_w_input, self.beta_w_assign_op = self.update_q_network_op(self.beta_w, 'load_beta_pred_from_pkl')
            for name in self.beta_w:
                self.beta_w_assign_op[name].eval({self.beta_w_input[name]: load_pkl(os.path.join(weight_dir, "beta_%s.pkl" % name))})

            self.alpha_w_input, self.alpha_w_assign_op = self.update_q_network_op(self.alpha_w, 'load_alpha_pred_from_pkl')
            for name in self.alpha_w:
                self.alpha_w_assign_op[name].eval({self.alpha_w_input[name]: load_pkl(os.path.join(weight_dir, "alpha_%s.pkl" % name))})

        self.update_target_network()
        '''

