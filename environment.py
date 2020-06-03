# import ipdb
import numpy as np
import scipy.io as sio


class Environment(object):

    def __init__(self, args):
        self.args = args
        self.hist_len = args.hist_len 
        self.image_dim = args.image_dim 
        self.state_beta_dim = args.state_dim
        self.image_padding = args.image_padding
        self.max_train_doms = args.max_train_doms   
        self.start_valid_dom = args.start_valid_dom  
        self.start_test_dom = args.start_test_dom    
        self.step_reward = args.step_reward
        self.collision_reward = args.collision_reward
        self.terminal_reward = args.terminal_reward
        self.max_steps = args.max_steps
        self.move = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]]) # North, South, West, East
        self.last_train_dom = -1
        self.border_start = self.image_padding + 1  # >= 1 
        self.border_end = self.image_dim + self.image_padding - 2  # <= dim + pad - 2
        self.padded_state_shape = (self.image_dim + self.image_padding*2, self.image_dim + self.image_padding*2)
        self.state_alpha_dim = self.state_beta_dim + self.image_padding * 2
        self.pos_bias = np.array([self.image_padding, self.image_padding])
        self.load_data()


    def load_data(self):
        data = sio.loadmat('data/gridworld_o_%d.mat' % self.image_dim)
        self.images = data['all_images']
        self.states_xy = data['all_states_xy_by_domain']
        self.max_domains = len(self.images) # 8000

    def is_valid_pos(self, xy):
        # 不在边界内
        return not (xy[0] > self.border_end or xy[0] < self.border_start or xy[1] > self.border_end or xy[1] < self.border_start)

    def restart(self, data_flag, init=False):
        if data_flag == 'train':    # training 
            init_dom = 0
            max_dom = self.max_train_doms
            if init:
                self.dom_ind = self.last_train_dom

        elif data_flag == 'valid':  # validation
            init_dom = self.start_valid_dom
            max_dom = self.start_test_dom
            if init:
                self.dom_ind = init_dom - 1
        
        else:   # testing
            init_dom = self.start_test_dom
            max_dom = self.max_domains
            if init:
                self.dom_ind = init_dom - 1
        
        invalid_flag = True
        pd = self.image_padding
        while invalid_flag:
            self.dom_ind += 1
            if self.dom_ind >= max_dom or self.dom_ind < init_dom:  
                self.dom_ind = init_dom
            self.state = np.zeros(self.padded_state_shape, dtype=np.uint8)
            self.state[pd:-pd, pd:-pd] = self.images[self.dom_ind, 0]  # 32 * 32
            self.paths = self.states_xy[self.dom_ind, 0]
            for i in range(len(self.paths)):
                # 由于是随机生成的地图，所以可能越界
                try:
                    self.a_xy = self.paths[i, 0][0] + self.pos_bias   #  alpha的初始位置
                    self.b_xy = self.paths[i, 0][-1] + self.pos_bias  #  beta的初始位置
                    self.min_steps = len(self.paths[i, 0]) / 2  # 计算理论上最短路径
                except:
                    continue
                # 如果a和b的初始位置不与障碍物重合则使用该地图，重合了就抛弃这个地图，while循环下一张地图
                if self.is_valid_pos(self.a_xy) and self.is_valid_pos(self.b_xy):
                    invalid_flag = False
                    break

        self.terminal = False
        self.episode_reward = []
        self.states_alpha = np.zeros([1, self.hist_len, self.state_alpha_dim, self.state_alpha_dim], dtype=np.float32)
        self.states_beta = np.zeros([1, self.hist_len, self.state_beta_dim, self.state_beta_dim], dtype=np.float32)
        self.states_alpha[0, -1] = self.state[self.a_xy[0]-1-pd: self.a_xy[0]+2+pd, self.a_xy[1]-1-pd: self.a_xy[1]+2+pd]
        self.states_beta[0, -1] = self.state[self.b_xy[0]-1: self.b_xy[0]+2, self.b_xy[1]-1: self.b_xy[1]+2]



    def act(self, action, steps):
        act_a, act_b = divmod(action, 4)
        new_a_xy = self.a_xy + self.move[act_a]
        new_b_xy = self.b_xy + self.move[act_b]

        # 0为障碍物，1为空地
        if self.is_valid_pos(new_a_xy) and self.state[new_a_xy[0], new_a_xy[1]] != 0:
            self.a_xy = new_a_xy
            r_a = self.step_reward
        else:
            # 不在边界内或遇到障碍物
            r_a = self.collision_reward
        
        if self.is_valid_pos(new_b_xy) and self.state[new_b_xy[0], new_b_xy[1]] != 0:
            self.b_xy = new_b_xy
            r_b = self.step_reward
        else:
            r_b = self.collision_reward
            
        # 计算激励
        reward = r_a + r_b
        manhattan_distance = abs(sum(self.b_xy - self.a_xy))

        r_ab = self.image_dim / (manhattan_distance + 1.0) # 加一避免除数为0
        reward += r_ab
        if manhattan_distance <= 1:
            reward += self.terminal_reward
        self.episode_reward.append([r_a, r_b, manhattan_distance])

        if manhattan_distance <= 1 or steps >= self.max_steps:
            self.terminal = True
        else:
            self.terminal = False
 
        pd = self.image_padding
        self.states_alpha[0, : -1] = self.states_alpha[0, 1: ]
        self.states_alpha[0, -1] = self.state[self.a_xy[0]-1-pd: self.a_xy[0]+2+pd, self.a_xy[1]-1-pd: self.a_xy[1]+2+pd]
        self.states_beta[0, : -1] = self.states_beta[0, 1: ]
        self.states_beta[0, -1] = self.state[self.b_xy[0]-1: self.b_xy[0]+2, self.b_xy[1]-1: self.b_xy[1]+2]

        return reward


    def getState(self):
        return self.states_alpha, self.states_beta

    def isTerminal(self):
        return self.terminal
        