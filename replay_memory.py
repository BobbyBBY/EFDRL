# coding: utf-8
# import ipdb
import numpy as np
import random
import torch

class ReplayMemory(object):
    """docstring for ReplayMemory"""
    def __init__(self, args):
        self.size = args.replay_size
        self.hist_len = args.hist_len
        self.priority = args.priority
        self.state_alpha_dim = args.state_dim + args.image_padding * 2
        self.batch_size = args.batch_size
        self.reward_bound = args.reward_bound
        self.positive_rate = args.positive_rate
        self.count = 0
        self.current = 0
        self.device = args.device_type
        self.actions = torch.zeros(self.size, dtype=torch.uint8).to(self.device)
        self.rewards = torch.zeros(self.size, dtype=torch.float32).to(self.device)
        self.states_alpha = torch.zeros([self.size, self.state_alpha_dim, self.state_alpha_dim], dtype=torch.uint8).to(self.device)
        self.terminals = torch.zeros(self.size, dtype=torch.bool).to(self.device)
        

    def reset(self):
        print('Reset the replay memory')
        self.actions *= 0
        self.rewards *= 0.0
        self.states_alpha *= 0
        self.terminals *= False
        self.count = 0
        self.current = 0

        
    def add(self, action, reward, state_alpha, terminal):
        #assert state.shape == self.dims
        # NB(Nota Bene)! state is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.states_alpha[self.current] = torch.Tensor(state_alpha[0, -1]).to(self.device)
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)  
        self.current = (self.current + 1) % self.size


    def getMinibatch(self):
        pre_states_alpha = torch.zeros([self.batch_size, self.hist_len, self.state_alpha_dim, self.state_alpha_dim]).to(self.device)
        post_states_alpha = torch.zeros([self.batch_size, self.hist_len, self.state_alpha_dim, self.state_alpha_dim]).to(self.device)
        if self.priority:
            # 择优回放。经验回放时，选取一组经验同时训练，pos_amount表示该组中，reward>reward_bound的经验元素的个数。
            pos_amount =  int(self.positive_rate*self.batch_size) 

        indices = []
        count_pos = 0
        count_neg = 0
        count_all = 0
        count_ter = 0
        # 这个参数应该放在main中
        max_circles = 1000 # max times for choosing positive samples or nagative samples
        while len(indices) < self.batch_size:
            # find random index 
            while True:
                # sample one index (ignore states wraping over) 
                index = random.randint(self.hist_len+1, self.count-1)
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
            
            for i in range(1, self.hist_len + 1):
                #这里的含义需要思考一下，没太懂
                if self.terminals[index - i]: # only the last state of the history can be terminal
                    break
                cur_ind = self.hist_len - i
                pre_states_alpha[len(indices)][cur_ind] = self.states_alpha[index - i]
                post_states_alpha[len(indices)][cur_ind] = self.states_alpha[index - i + 1]
            indices.append(index)

        # copy actions, rewards and terminals with direct slicing
        actions = self.actions[indices]  
        rewards = self.rewards[indices]
        terminals = self.terminals[indices]

        return pre_states_alpha, actions, rewards, post_states_alpha, terminals
