# coding: utf-8
import random
import torch
import numpy as np
from tqdm import tqdm


class Agent(object):
    """docstring for Agent"""
    def __init__(self, env, mem, dqn, args):
        self.env = env
        self.mem = mem
        self.net = dqn
        self.print_granularity = args.print_granularity
        self.exp_rate_start = args.exploration_rate_start
        self.exp_rate_end = args.exploration_rate_end
        self.exp_decay_steps = args.exploration_decay_steps
        self.exploration_rate_test = args.exploration_rate_test
        self.total_train_steps = args.start_epoch * args.train_episodes * args.image_dim

        self.train_frequency = args.train_frequency
        self.target_steps = args.target_steps
        self.num_actions = args.num_actions
        self.steps = 0

    # epsilon贪心算法中衰减的探索率
    def _explorationRate(self):
        if self.total_train_steps < self.exp_decay_steps:
            return self.exp_rate_start - self.total_train_steps * \
            (self.exp_rate_start - self.exp_rate_end) / self.exp_decay_steps
        else:
            return self.exp_rate_end

    def step(self, exploration_rate, predict_net):
        if np.random.rand() < exploration_rate:
            # 探索
            action = np.random.randint(self.num_actions)
        else:
            # 利用
            state_alpha, state_beta = self.env.getState()
            qvalue = self.net.predict(state_alpha, state_beta, predict_net)
            action = torch.argmax(qvalue).item() # 这里应该调用net里的函数获得下标，降低耦合
            
        # 做出行动 
        self.steps += 1
        reward = self.env.act(action, self.steps)
        state_alpha, state_beta = self.env.getState()
        terminal = self.env.isTerminal()
        
        return action, reward, state_alpha, state_beta, terminal


    def train(self, epoch, train_episodes, outfile, predict_net):
        ep_loss, ep_rewards, details = [], [], []
        min_samples = self.mem.batch_size + self.mem.hist_len
        
        if self.print_granularity == 2:
            print('\n\n Training [%s] predicting [%s] ...' % (self.net.args.train_mode, predict_net))
        self.env.restart(data_flag='train', init=True)
        for episodes in range(train_episodes):
            self.steps = 0
            terminal = False
            while not terminal:
                # 此处state是t+1时的state，即执行act之后的状态
                act, r, s_a, s_b, terminal = self.step(self._explorationRate(), predict_net)
                self.mem.add(act, r, s_a, s_b, terminal)
                # 每target_steps后更新target网络
                if self.target_steps and self.total_train_steps % self.target_steps == 0:
                    self.net.update_target_network()

                # 每train_frequency后训练网络
                if self.mem.count > min_samples and self.total_train_steps % self.train_frequency == 0:
                    # 抽样minibatch
                    minibatch = self.mem.getMinibatch()
                    # 开始训练
                    loss = self.net.train(minibatch)
                    ep_loss.append(loss)

                ep_rewards.append(r)
                self.total_train_steps += 1

            if self.print_granularity == 2:
                print('domain: %d \t min_steps: %d \t max_steps: %d' % (self.env.dom_ind, self.env.min_steps, self.env.max_steps))
            if len(ep_loss) > 0:
                avg_loss = sum(ep_loss) / len(ep_loss)
                max_loss = max(ep_loss)
                min_loss = min(ep_loss)
                if self.print_granularity == 2:
                    print('max_loss: {:>6.6f}\t min_loss: {:>6.6f}\t avg_loss: {:>6.6f}'.format(max_loss, min_loss, avg_loss))
            
            sum_reward = sum(ep_rewards)
            # details.append(self.env.episode_reward)
            if self.print_granularity == 2:
                print('epochs: {}\t episodes: {}\t steps: {}\t sum_reward: {:>6.6f}\n'.format(epoch, episodes, self.steps, sum_reward))

            ep_loss, ep_rewards = [], []
            self.env.restart(data_flag='train')
        self.env.last_train_dom = self.env.dom_ind
        # return details


    def test(self, epoch, test_epidodes, outfile, predict_net, data_flag):
        success = 0.0
        min_steps = 0.0
        real_steps = 0.0
        test_reward = 0.0
        avg_reward = 0.0
        log_step_success = {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}

        if self.print_granularity == 2:
            print('\n\n %s %s net ...' % (data_flag, predict_net))
            outfile.write('\n\n %s %s net ...\n' % (data_flag, predict_net))
        self.steps = 0
        self.env.restart(data_flag=data_flag, init=True)
        # for ep in tqdm(range(test_epidodes)):  # tqdm实现进度条
        for ep in range(test_epidodes):
            terminal = False
            ep_rewards = []
            while not terminal:
                act, r, s_a, s_b, terminal = self.step(self.exploration_rate_test, predict_net)
                ep_rewards.append(r)

            sum_reward = sum(ep_rewards)
            test_reward += sum_reward
            if self.env.episode_reward[-1][-1] <= 1: # distance=0或1意味着alpha与beta相遇
                success += 1.0
                min_steps += self.env.min_steps
                real_steps += self.steps
                for k in log_step_success:
                    if self.steps - self.env.min_steps <= k * self.env.min_steps:
                        log_step_success[k] += 1.0

            self.steps = 0
            self.env.restart(data_flag=data_flag)

        success_rate = success / test_epidodes
        avg_reward = test_reward / test_epidodes
        avg_steps = real_steps / success
        step_diff = (real_steps - min_steps) / min_steps
        for k in log_step_success:
            log_step_success[k] = log_step_success[k] / test_epidodes
        log_step_success[-1] = success_rate

        #输出本次训练的结果
        if self.print_granularity == 2:
            print('\n epochs: {}\t avg_reward: {:.2f}\t avg_steps: {:.2f}\t step_diff: {:.2f}'.format(epoch, avg_reward, avg_steps, step_diff))
            print('episodes: {}\t success_rate: {}\n'.format(test_epidodes, log_step_success)) 
            outfile.write('-----{}-----\n'.format(predict_net))
            outfile.write('\n epochs: {}\t avg_reward: {:.2f}\t avg_steps: {:.2f}\t step_diff: {:.2f}\n'.format(epoch, avg_reward, avg_steps, step_diff))
            outfile.write('episodes: {}\t success_rate: {}\n\n'.format(test_epidodes, log_step_success))   

        return log_step_success, avg_reward, step_diff
