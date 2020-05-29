import os
import model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random

class FRLDQN(object):
    """docstring for FRLNetwork"""
    def __init__(self, args):
        self.args = args
        self.test_only = args.test_only
        self.gamma = args.gamma
        self.print_granularity = args.print_granularity
        self.add_noise = args.add_noise
        self.noise_prob = args.noise_prob
        self.stddev = args.stddev
        self.num_actions = args.num_actions
        self.learning_rate = args.learning_rate
        self.hist_len = args.hist_len
        self.state_beta_dim = args.state_dim
        self.state_alpha_dim = args.state_dim + args.image_padding * 2
        self.batch_size = args.batch_size
        self.device = args.device_type
        self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        # self.criterion = self.Square_loss
        self.build_dqn()
        self.exc_p()
        self.same = 0
        self.total = 0
        self.samea = 0
        self.sameb = 0

   
    def build_dqn(self):
        if self.args.train_mode == 'single_alpha':
            # construct DQN-alpha network
            self.alpha_q = model.DQN_single(self.batch_size, self.hist_len, self.num_actions, self.state_alpha_dim).to(self.device)
            self.alpha_t_q = model.DQN_single(self.batch_size, self.hist_len, self.num_actions, self.state_alpha_dim).to(self.device)
            self.optimizer_dqn_alpha = optim.SGD(self.alpha_q.parameters(), lr=self.learning_rate) # 优化器

        elif self.args.train_mode == 'single_alpha':
            # construct DQN-beta network
            self.beta_q = model.DQN_single(self.batch_size, self.hist_len, self.num_actions, self.state_beta_dim).to(self.device)
            self.beta_t_q = model.DQN_single(self.batch_size, self.hist_len, self.num_actions, self.state_beta_dim).to(self.device)
            self.optimizer_dqn_beta = optim.SGD(self.beta_q.parameters(), lr=self.learning_rate)
        
        elif self.args.train_mode == 'full':
            # construct DQN-full network
            self.full_q = model.DQN_full(self.batch_size, self.hist_len, self.num_actions, self.state_alpha_dim, self.state_beta_dim).to(self.device)
            self.full_t_q = model.DQN_full(self.batch_size, self.hist_len, self.num_actions, self.state_alpha_dim, self.state_beta_dim).to(self.device)
            self.optimizer_dqn_full = optim.SGD(self.full_q.parameters(), lr=self.learning_rate)
        else: # frl,fefrl,sfrl
            # construct DQN-alpha network
            self.alpha_q = model.DQN_single(self.batch_size, self.hist_len, self.num_actions, self.state_alpha_dim).to(self.device)
            self.alpha_t_q = model.DQN_single(self.batch_size, self.hist_len, self.num_actions, self.state_alpha_dim).to(self.device)
            self.optimizer_dqn_alpha = optim.SGD(self.alpha_q.parameters(), lr=self.learning_rate) # 优化器

            # construct DQN-beta network
            self.beta_q = model.DQN_single(self.batch_size, self.hist_len, self.num_actions, self.state_beta_dim).to(self.device)
            self.beta_t_q = model.DQN_single(self.batch_size, self.hist_len, self.num_actions, self.state_beta_dim).to(self.device)
            self.optimizer_dqn_beta = optim.SGD(self.beta_q.parameters(), lr=self.learning_rate)

            # construct FRL network
            self.frl_q = model.MLP(self.num_actions).to(self.device)
            self.frl_t_q = model.MLP(self.num_actions).to(self.device)
            self.optimizer_frl = optim.SGD(self.frl_q.parameters(), lr=self.learning_rate)
             
            if self.args.train_mode == 'frl_separate' or self.args.train_mode == 'fefrl':
                self.frl_q_2 = model.MLP(self.num_actions).to(self.device)
                self.optimizer_frl_2 = optim.SGD(self.frl_q_2.parameters(), lr=self.learning_rate)

    # 损失函数，平方损失函数
    def Square_loss(self, x, y):
        return torch.mean(torch.pow((x - y), 2))

    # 准备独占式gx需要的张量
    def exc_p(self):
        linear_exchange = torch.rand(self.num_actions).to(self.device)*50 - 25
        self.change_train = linear_exchange.expand(self.batch_size, self.num_actions)
        self.change_predict = linear_exchange.expand(1, self.num_actions)
        # self.non_linear_exchange=random.sample(range(self.num_actions),self.num_actions)

    #独占式FRL中的g(x)
    def g_t(self,qvalue):
        qvalue+=self.change_train
        return qvalue

    def g_p(self,qvalue):
        qvalue+=self.change_predict
        return qvalue

    def update_target_network(self):
        if self.args.train_mode == 'single_alpha':
            self.alpha_t_q=copy.deepcopy(self.alpha_q)

        elif self.args.train_mode == 'single_beta':
            self.beta_t_q=copy.deepcopy(self.beta_q)

        elif self.args.train_mode == 'full':
            self.full_t_q=copy.deepcopy(self.full_q)

        elif self.args.train_mode == 'frl_separate' or self.args.train_mode == 'fefrl':
            self.alpha_t_q=copy.deepcopy(self.alpha_q)
            self.beta_t_q=copy.deepcopy(self.beta_q)
            self.frl_t_q=copy.deepcopy(self.frl_q)

        else: # sefrl
            self.alpha_t_q=copy.deepcopy(self.alpha_q)
            self.beta_t_q=copy.deepcopy(self.beta_q)
            self.frl_t_q=copy.deepcopy(self.frl_q)

    def train(self, minibatch):
        pre_states_alpha, pre_states_beta, actions, rewards, post_states_alpha, post_states_beta, terminals = minibatch
        
        if self.args.train_mode == 'single_alpha':
            targets = self.alpha_t_q.forward(post_states_alpha)
            max_postq = torch.max(targets,1)[0]
            tempQ_yi = self.alpha_q.forward(pre_states_alpha)
            tempQ = tempQ_yi.clone().detach()

        elif self.args.train_mode == 'single_beta':
            targets = self.beta_t_q.forward(post_states_beta)
            max_postq = torch.max(targets,1)[0]
            tempQ_yi = self.beta_q.forward(pre_states_beta)
            tempQ = tempQ_yi.clone().detach()

        elif self.args.train_mode == 'full':
            targets = self.full_t_q.forward(post_states_alpha, post_states_beta)
            max_postq = torch.max(targets,1)[0]
            tempQ_yi = self.full_q.forward(pre_states_alpha, pre_states_beta)
            tempQ = tempQ_yi.clone().detach()
        
        else: # frl
            targets_alpha = self.alpha_t_q.forward(post_states_alpha)
            targets_beta = self.beta_t_q.forward(post_states_beta)
            tempQ_yi_alpha = self.alpha_q.forward(pre_states_alpha)
            tempQ_yi_beta = self.beta_q.forward(pre_states_beta)
            if self.add_noise and np.random.rand() <= self.noise_prob:
                # add Gaussian noise to Q-values with self.noise_prob probility 
                noise_alpha = torch.normal(0.0, self.stddev, targets_alpha.shape).to(self.device)
                noise_beta = torch.normal(0.0, self.stddev, targets_beta.shape).to(self.device)
                targets_alpha += noise_alpha
                targets_beta += noise_beta

                noise_alpha = torch.normal(0.0, self.stddev, tempQ_yi_alpha.shape).to(self.device)
                noise_beta = torch.normal(0.0, self.stddev, tempQ_yi_beta.shape).to(self.device)
                tempQ_yi_alpha += noise_alpha
                tempQ_yi_beta += noise_beta

            targets = self.frl_t_q.forward(targets_alpha, targets_beta)
            max_postq = torch.max(targets,1)[0]
            tempQ_yi = self.frl_q.forward(tempQ_yi_alpha, tempQ_yi_beta)
            tempQ = tempQ_yi.clone().detach()
            
        for i, action in enumerate(actions):
            if terminals[i]:
                tempQ[i][action] = rewards[i]
            else:
                tempQ[i][action] = rewards[i] + self.gamma * max_postq[i]

        if self.args.train_mode == 'frl_separate':
            tempQ_yi_2 = self.frl_q_2.forward(tempQ_yi_alpha, tempQ_yi_beta)
            tempQ_2 = tempQ.clone().detach()
        elif self.args.train_mode == 'fefrl':
            tempQ_yi_2 = self.g_t(self.frl_q_2.forward(tempQ_yi_alpha, tempQ_yi_beta))
            tempQ_yi_2 = self.g_t(tempQ_yi_2)
            tempQ_2 = tempQ.clone().detach()
        else: 
            pass


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

        elif self.args.train_mode == 'sefrl':
            loss = self.criterion(tempQ, tempQ_yi)
            self.optimizer_dqn_alpha.zero_grad()
            self.optimizer_dqn_beta.zero_grad()
            self.optimizer_frl.zero_grad()
            loss.backward()
            self.optimizer_frl.step()
            self.optimizer_dqn_alpha.step()
            self.optimizer_dqn_beta.step()

        elif self.args.train_mode == 'frl_separate' or self.args.train_mode == 'fefrl':
            loss = self.criterion(tempQ, tempQ_yi)
            self.optimizer_dqn_beta.zero_grad()
            self.optimizer_frl.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer_frl.step()
            self.optimizer_dqn_beta.step()
            loss2 = self.criterion(tempQ_2, tempQ_yi_2)
            self.optimizer_dqn_alpha.zero_grad()
            self.optimizer_frl_2.zero_grad()
            loss2.backward()
            self.optimizer_frl_2.step()
            self.optimizer_dqn_alpha.step()

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

        elif self.test_only and predict_net == 'both1':
            qvalue = self.alpha_q.forward(torch.Tensor(state_alpha).to(self.device))
            
        else: 
            q_alpha = self.alpha_q.forward(torch.Tensor(state_alpha).to(self.device))
            q_beta = self.beta_q.forward(torch.Tensor(state_beta).to(self.device))
            if self.add_noise:
                noise_alpha = torch.normal(0.0, self.stddev, q_alpha.shape).to(self.device)
                noise_beta = torch.normal(0.0, self.stddev, q_beta.shape).to(self.device)
                q_alpha += noise_alpha
                q_beta += noise_beta
            if predict_net == 'both2':
                if self.test_only:
                    qvalue = self.frl_q_2.forward(q_alpha, q_beta)
                else:
                    qvalue = self.frl_q.forward(q_alpha, q_beta)
            elif predict_net == 'both1':
                qvalue = self.frl_q.forward(q_alpha, q_beta)
            else:
                print('\n Wrong predict mode! \n')
                raise ValueError
        return qvalue[0] # qvalue 与 qvalue[0]是一样的


    def save_weights(self, weight_dir, net_name):
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        if net_name == 'full':
            if self.print_granularity == 2:
                print('Saving full network weights ...')      
            torch.save(self.full_q.state_dict(),os.path.join(weight_dir, "full_q_im%s_%s.pth" % (self.args.image_dim,self.args.result_dir_mark)))

        elif net_name == 'beta':
            if self.print_granularity == 2:
                print('Saving beta network weights ...')
            torch.save(self.beta_q.state_dict(),os.path.join(weight_dir, "beta_q_im%s_%s.pth" % (self.args.image_dim,self.args.result_dir_mark)))

        elif net_name == 'alpha':
            if self.print_granularity == 2:
                print('Saving alpha network weights ...')
            torch.save(self.alpha_q.state_dict(),os.path.join(weight_dir, "alpha_q_im%s_%s.pth" % (self.args.image_dim,self.args.result_dir_mark)))
        elif net_name == 'both1':
            if self.print_granularity == 2:
                print('Saving three networks weights in sefrl ...')
            torch.save(self.frl_q.state_dict(),os.path.join(weight_dir, "frl_q_im%s_%s.pth" % (self.args.image_dim,self.args.result_dir_mark)))
            torch.save(self.alpha_q.state_dict(),os.path.join(weight_dir, "alpha_q_f_im%s_%s.pth" % (self.args.image_dim,self.args.result_dir_mark)))
            torch.save(self.beta_q.state_dict(),os.path.join(weight_dir, "beta_q_f_im%s_%s.pth" % (self.args.image_dim,self.args.result_dir_mark)))
        else:# both21 both22
            if self.print_granularity == 2:
                print('Saving four networks weights in frl or fefrl ...')
            torch.save(self.frl_q.state_dict(),os.path.join(weight_dir, "frl_q_im%s_%s.pth" % (self.args.image_dim,self.args.result_dir_mark)))
            torch.save(self.alpha_q.state_dict(),os.path.join(weight_dir, "alpha_q_f_im%s_%s.pth" % (self.args.image_dim,self.args.result_dir_mark)))
            torch.save(self.beta_q.state_dict(),os.path.join(weight_dir, "beta_q_f_im%s_%s.pth" % (self.args.image_dim,self.args.result_dir_mark)))
            torch.save(self.frl_q_2.state_dict(),os.path.join(weight_dir, "frl_q_2_f_im%s_%s.pth" % (self.args.image_dim,self.args.result_dir_mark)))


    def load_weights(self, weight_dir):
        if self.print_granularity == 2:
            print('Loading weights from %s ...' % weight_dir)
        if self.args.train_mode == 'full':
            path = "full_q_im%s_%s.pth" % (self.args.image_dim,self.args.result_dir_mark)
            if os.path.exists(os.path.join(weight_dir, path)):
                self.full_q.load_state_dict(torch.load(os.path.join(weight_dir, path)))

        elif self.args.train_mode == 'sefrl':
            path = "frl_q_im%s_%s.pth" % (self.args.image_dim,self.args.result_dir_mark)
            path2 = "alpha_q_f_im%s_%s.pth" % (self.args.image_dim,self.args.result_dir_mark)
            path3 = "beta_q_f_im%s_%s.pth" % (self.args.image_dim,self.args.result_dir_mark)
            if os.path.exists(os.path.join(weight_dir, path)):
                self.frl_q.load_state_dict(torch.load(os.path.join(weight_dir, path)))
            if os.path.exists(os.path.join(weight_dir, path2)):
                self.alpha_q.load_state_dict(torch.load(os.path.join(weight_dir, path2)))
            if os.path.exists(os.path.join(weight_dir, path3)):
                self.beta_q.load_state_dict(torch.load(os.path.join(weight_dir, path3)))
        elif self.args.train_mode == 'single_alpha':
            path = "alpha_q_im%s_%s.pth" % (self.args.image_dim,self.args.result_dir_mark)
            if os.path.exists(os.path.join(weight_dir, path)):
                self.alpha_q.load_state_dict(torch.load(os.path.join(weight_dir, path)))
        elif self.args.train_mode == 'single_beta':
            path = "beta_q_im%s_%s.pth" % (self.args.image_dim,self.args.result_dir_mark)
            if os.path.exists(os.path.join(weight_dir, path)):
                self.beta_q.load_state_dict(torch.load(os.path.join(weight_dir, path)))
        else: # sefrl,frl
            path = "frl_q_im%s_%s.pth" % (self.args.image_dim,self.args.result_dir_mark)
            path2 = "alpha_q_f_im%s_%s.pth" % (self.args.image_dim,self.args.result_dir_mark)
            path3 = "beta_q_f_im%s_%s.pth" % (self.args.image_dim,self.args.result_dir_mark)
            path4 = "frl_q_2_f_im%s_%s.pth" % (self.args.image_dim,self.args.result_dir_mark)
            if os.path.exists(os.path.join(weight_dir, path)):
                self.frl_q.load_state_dict(torch.load(os.path.join(weight_dir, path)))
            if os.path.exists(os.path.join(weight_dir, path2)):
                self.alpha_q.load_state_dict(torch.load(os.path.join(weight_dir, path2)))
            if os.path.exists(os.path.join(weight_dir, path3)):
                self.beta_q.load_state_dict(torch.load(os.path.join(weight_dir, path3)))
            if os.path.exists(os.path.join(weight_dir, path4)):
                self.frl_q_2.load_state_dict(torch.load(os.path.join(weight_dir, path4)))

        self.update_target_network()

