# coding: utf-8
import numpy as np
import torch
class ReplayMemory(object):
    """docstring for ReplayMemory"""
    def __init__(self, args):
        self.size = args.replay_size
        self.hist_len = args.hist_len
        self.priority = args.priority
        self.state_beta_dim = args.state_dim
        self.state_alpha_dim = args.state_dim + args.image_padding * 2
        self.batch_size = args.batch_size
        self.reward_bound = args.reward_bound
        self.positive_rate = args.positive_rate
        self.count = 0
        self.current = 0
        self.device = args.device_type
        self.actions = np.zeros(self.size, dtype=np.uint8)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.states_alpha = np.zeros([self.size, self.state_alpha_dim, self.state_alpha_dim], dtype=np.uint8)
        self.states_beta = np.zeros([self.size, self.state_beta_dim, self.state_beta_dim], dtype=np.uint8)
        self.terminals = np.zeros(self.size, dtype=np.bool)


    def reset(self):
        print('Reset the replay memory')
        self.actions *= 0
        self.rewards *= 0.0
        self.states_beta *= 0
        self.states_alpha *= 0
        self.terminals *= False
        self.count = 0
        self.current = 0

        
    def add(self, action, reward, state_alpha, state_beta, terminal):
        #assert state.shape == self.dims
        # NB(Nota Bene)! state is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.states_alpha[self.current] = state_alpha[0, -1]
        self.states_beta[self.current] = state_beta[0, -1]
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)  
        self.current = (self.current + 1) % self.size


    def getMinibatch(self):
        pre_states_beta = np.zeros([self.batch_size, self.hist_len, self.state_beta_dim, self.state_beta_dim])
        pre_states_alpha = np.zeros([self.batch_size, self.hist_len, self.state_alpha_dim, self.state_alpha_dim])
        post_states_beta = np.zeros([self.batch_size, self.hist_len, self.state_beta_dim, self.state_beta_dim])
        post_states_alpha = np.zeros([self.batch_size, self.hist_len, self.state_alpha_dim, self.state_alpha_dim])
        if self.priority:
            # 择优回放。经验回放时，选取一组经验同时训练，pos_amount表示该组中，reward>reward_bound的经验元素的个数。
            pos_amount =  int(self.positive_rate*self.batch_size) 

        indices = []
        count_pos = 0
        count_neg = 0
        count_all = 0
        count_ter = 0
        max_circles = 1000 # max times for choosing positive samples or nagative samples
        while len(indices) < self.batch_size:
            # find random index 
            while True:
                # sample one index (ignore states wraping over) 
                index = np.random.randint(self.hist_len+1, self.count)
                # # NB! prestate (last state) can be terminal state!
                # if any(self.terminals[index-self.hist_len: index]) and count_ter < max_circles:
                #     count_ter += 1
                #     continue
                # use prioritized replay trick
                if self.priority:
                    if count_all < max_circles:
                        # if num_pos is already enough but current idx is also pos sample, continue
                        if (count_pos >= pos_amount) and (self.rewards[index] >= self.reward_bound):
                            count_all += 1
                            continue
                        # elif num_nag is already enough but current idx is also nag sample, continue
                        elif (count_neg >= self.batch_size - pos_amount) and (self.rewards[index] < self.reward_bound): 
                            count_all += 1
                            continue
                    if self.rewards[index] >= self.reward_bound:
                        count_pos += 1
                    else:
                        count_neg += 1
                break
            
            # 因为 python3 中取消了 range 函数，而把 xrange 函数重命名为 range
            for i in range(1, self.hist_len + 1):
                if self.terminals[index - i]: # only the last state of the history can be terminal
                    break
                cur_ind = self.hist_len - i
                pre_states_alpha[len(indices)][cur_ind] = self.states_alpha[index - i]
                pre_states_beta[len(indices)][cur_ind] = self.states_beta[index - i]
                post_states_alpha[len(indices)][cur_ind] = self.states_alpha[index - i + 1]
                post_states_beta[len(indices)][cur_ind] = self.states_beta[index - i + 1]
            indices.append(index)

        # copy actions, rewards and terminals with direct slicing
        actions = self.actions[indices]  
        rewards = self.rewards[indices]
        terminals = self.terminals[indices]
        pre_states_alpha = torch.Tensor(pre_states_alpha).to(self.device)
        pre_states_beta = torch.Tensor(pre_states_beta).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        post_states_alpha = torch.Tensor(post_states_alpha).to(self.device)
        post_states_beta = torch.Tensor(post_states_beta).to(self.device)
        terminals = torch.Tensor(terminals).to(self.device)


        return pre_states_alpha, pre_states_beta, actions, rewards, post_states_alpha, post_states_beta, terminals
