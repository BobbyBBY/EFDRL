import os
# import ipdb
import numpy as np
import tensorflow as tf
from utils import save_pkl, load_pkl

# 似乎没有用到
# from tensorflow.contrib.layers.python.layers import initializers

# 在 Python3 中，reduce() 函数已经被从全局名字空间里移除了，它现在被放置在 functools 模块里
from functools import reduce


class FRLDQN(object):
    """docstring for FRLNetwork"""
    def __init__(self, sess, args):
        self.sess = sess
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
        self.build_dqn()


    def conv2d(self, x, output_dim, kernel_size, stride, initializer, activation_fn=None, padding='VALID', name='conv2d'):
        with tf.compat.v1.variable_scope(name):
            # data_format = 'NHWC'
            stride = [1, stride[0], stride[1], 1]
            kernel_size = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]
            
            w = tf.compat.v1.get_variable('w', kernel_size, tf.float32, initializer=initializer)
            conv = tf.nn.conv2d(x, w, stride, padding)

            b = tf.compat.v1.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.1))
            out = tf.nn.bias_add(conv, b)

        if activation_fn != None:
            out = activation_fn(out)
        return out, w, b


    def max_pooling(self, x, kernel_size, stride, padding='VALID', name='max_pool'):
        with tf.compat.v1.variable_scope(name):
            stride = [1, stride[0], stride[1], 1]
            kernel_size = [1, kernel_size[0], kernel_size[1], 1]
            return tf.nn.max_pool(x, kernel_size, stride, padding)


    def linear(self, x, output_dim, activation_fn=None, name='linear'):
        with tf.compat.v1.variable_scope(name):
            w = tf.compat.v1.get_variable('w', [x.get_shape()[1], output_dim], tf.float32, 
                initializer=tf.truncated_normal_initializer(0, 0.1))
            b = tf.compat.v1.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.1))
            out = tf.nn.bias_add(tf.matmul(x, w), b)

        if activation_fn != None:
            out = activation_fn(out)
        return out, w, b


    def build_dqn(self):
        init = tf.contrib.layers.xavier_initializer_conv2d()

        def build_nn(name, weight, s_t):
            fw = s_t.shape[2] if self.args.autofilter else 3
            with tf.compat.v1.variable_scope(name):
                print('Initializing %s network ...' % name)
                l1, weight['l1_w'], weight['l1_b'] = self.conv2d(s_t, 32, [fw, fw], [1, 1], init, tf.nn.relu, 'SAME', name='l1')
                l1_shape = l1.get_shape().as_list()
                l1_flat = tf.reshape(l1, [-1, reduce(lambda x, y: x * y, l1_shape[1:])])
                l2, weight['l2_w'], weight['l2_b'] = self.linear(l1_flat, 256, tf.nn.relu, name='l2')
                out_layer, weight['q_w'], weight['q_b'] = self.linear(l2, self.num_actions, name='q')

                return out_layer

       
        # construct DQN-alpha network
        self.s_a = tf.compat.v1.placeholder(tf.float32, [None, self.state_alpha_dim, self.state_alpha_dim, self.hist_len], 's_a')
        self.alpha_w, self.alpha_t_w = {}, {}
        self.alpha_q = build_nn('alpha_q', self.alpha_w, self.s_a)
        self.alpha_t_q = build_nn('alpha_t_q', self.alpha_t_w, self.s_a)

        # construct the update q-network operators
        self.alpha_t_w_input, self.alpha_t_w_assign_op = self.update_q_network_op(self.alpha_t_w, 'alpha_update_q_network_op')
            
        with tf.compat.v1.variable_scope('optimizer'):
            print('Initializing optimizer ...')
            self.target_q = tf.compat.v1.placeholder(tf.float32, [None, self.num_actions], 'targets')
            self.delta_alpha = self.target_q - self.alpha_q

            self.loss_alpha = tf.reduce_sum(tf.square(self.delta_alpha), name='loss_alpha')
            self.train_single_alpha = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss_alpha)
           
        tf.compat.v1.global_variables_initializer().run()


    def update_q_network_op(self, t_w, name):
        with tf.compat.v1.variable_scope(name):
            t_w_input = {}
            t_w_assign_op = {}

            for name in t_w:
                t_w_input[name] = tf.compat.v1.placeholder(tf.float32, t_w[name].get_shape().as_list(), name)
                t_w_assign_op[name] = t_w[name].assign(t_w_input[name])

            return t_w_input, t_w_assign_op


    def update_target_network(self):
        if self.args.train_mode == 'single_alpha':
            for name in self.alpha_w:
                self.alpha_t_w_assign_op[name].eval({self.alpha_t_w_input[name]: self.alpha_w[name].eval()})

       


    def train(self, minibatch):
        #ipdb.set_trace()
        pre_states_alpha, pre_states_beta, actions, rewards, post_states_alpha, post_states_beta, terminals = minibatch
        
        if self.args.train_mode == 'single_alpha':
            postq = self.alpha_t_q.eval({self.s_a: post_states_alpha})
            max_postq = np.max(postq, axis=1)
            targets = self.alpha_q.eval({self.s_a: pre_states_alpha})

     

        for i, action in enumerate(actions):
            if terminals[i]:
                targets[i, action] = rewards[i]
            else:
                targets[i, action] = rewards[i] + self.gamma * max_postq[i]


        if self.args.train_mode == 'single_alpha':  
            _, delta, loss = self.sess.run([self.train_single_alpha, 
                                            self.delta_alpha, 
                                            self.loss_alpha
                                         ],
                                         {  self.s_a: pre_states_alpha,
                                            self.target_q: targets
                                         })
        else: 
            print('\n Wrong training mode! \n')
            raise ValueError

        return delta, loss


    def predict(self, state_alpha, predict_net):
        #ipdb.set_trace()
        state_alpha = np.transpose(state_alpha, (0, 2, 3, 1))
        
        if predict_net == 'alpha':
            qvalue = self.alpha_q.eval({self.s_a: state_alpha})
        else:
            print('\n Wrong predict mode! \n')
            raise ValueError

        return qvalue[0]
