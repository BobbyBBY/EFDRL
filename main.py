#coding:utf-8
import time
import argparse
import torch

from agent import Agent
from deep_q_network import FRLDQN
from environment import Environment
from replay_memory import ReplayMemory



def args_init():
    parser = argparse.ArgumentParser()

    envarg = parser.add_argument_group('Environment')
    envarg.add_argument("--image_dim",              type=int,       default=8,    help="")
    envarg.add_argument("--state_dim",              type=int,       default=3,     help="")
    envarg.add_argument("--hist_len",               type=int,       default=2,    help="")
    envarg.add_argument("--max_steps",              type=int,       default=2,     help="")
    envarg.add_argument("--image_padding",          type=int,       default=1,     help="")
    envarg.add_argument("--max_train_doms",         type=int,       default=6400,  help="")
    envarg.add_argument("--start_valid_dom",        type=int,       default=6400,  help="")
    envarg.add_argument("--start_test_dom",         type=int,       default=7200,  help="")
    envarg.add_argument("--automax",                type=int,       default=2,     help="")
    envarg.add_argument("--autolen",                type=int,       default=1,     help="")
    envarg.add_argument("--use_instant_distance",   type=int,       default=1,      help="")
    envarg.add_argument("--step_reward",            type=float,     default=-1.0,   help="")
    envarg.add_argument("--collision_reward",       type=float,     default=-10.0,  help="")
    envarg.add_argument("--terminal_reward",        type=float,     default=+50.0,  help="")

    memarg = parser.add_argument_group('Replay memory')
    memarg.add_argument("--positive_rate",      type=float, default=0.9,    help="")
    memarg.add_argument("--reward_bound",       type=float, default=0.0,    help="")
    memarg.add_argument("--priority",           type=int,   default=1,      help="")
    memarg.add_argument("--replay_size",        type=int,   default=100000, help="")

    netarg = parser.add_argument_group('Deep Q-learning network')
    netarg.add_argument("--autofilter",         type=int,       default=1,      help="")
    netarg.add_argument("--batch_size",         type=int,       default=32,     help="")
    netarg.add_argument("--num_actions",        type=int,       default=16,     help="")
    netarg.add_argument("--learning_rate",      type=float,     default=0.001,  help="")
    netarg.add_argument("--gamma",              type=float,     default=0.9,    help="")
    netarg.add_argument("--lambda_",            type=float,     default=0.5,    help="")
    netarg.add_argument("--preset_lambda",      type=str2bool,  default=False,  help="")
    netarg.add_argument("--add_train_noise",    type=str2bool,  default=False,   help="")
    netarg.add_argument("--add_predict_noise",  type=str2bool,  default=False,   help="")
    netarg.add_argument("--noise_prob",         type=float,     default=0.5,    help="")
    netarg.add_argument("--stddev",             type=float,     default=1.0,    help="")
    netarg.add_argument("--device_type",        type=torch.device,     default=torch.device("cpu"),    help="")

    antarg = parser.add_argument_group('Agent')
    antarg.add_argument("--exploration_rate_start",     type=float, default=1,      help="")
    antarg.add_argument("--exploration_rate_end",       type=float, default=0.1,    help="")
    antarg.add_argument("--exploration_rate_test",      type=float, default=0.0,    help="")
    antarg.add_argument("--exploration_decay_steps",    type=int,   default=1000,   help="")
    antarg.add_argument("--train_frequency",            type=int,   default=1,      help="")
    antarg.add_argument("--target_steps",               type=int,   default=5,      help="")
    
    mainarg = parser.add_argument_group('Main loop')
    mainarg.add_argument("--gpu_fraction",      type=float,     default=0.2,        help="")
    mainarg.add_argument("--epochs",            type=int,       default=200,        help="")
    mainarg.add_argument("--start_epoch",       type=int,       default=0,          help="")
    mainarg.add_argument("--stop_epoch_gap",    type=int,       default=10,         help="")
    mainarg.add_argument("--success_base",      type=int,       default=-1,         help="")
    mainarg.add_argument("--load_weights",      type=str2bool,  default=False,      help="")
    mainarg.add_argument("--save_weights",      type=str2bool,  default=False,       help="")
    mainarg.add_argument("--predict_net",       type=str,       default='alpha',     help="")
    mainarg.add_argument("--result_dir",        type=str,       default="preset_lambda",     help="") #
    mainarg.add_argument("--train_mode",        type=str,       default='single_alpha',     help='')
    mainarg.add_argument("--train_episodes",    type=int,       default=100,        help="") #
    mainarg.add_argument("--valid_episodes",    type=int,       default=800,        help="") #
    mainarg.add_argument("--test_episodes",     type=int,       default=800,        help="") #
    mainarg.add_argument("--test_multi_nets",   type=str2bool,  default=False,      help="") #
    
    args = parser.parse_args()
    if args.load_weights:
        args.exploration_rate_start = args.exploration_rate_end
    if args.autolen:
        lens = {8: 2, 16: 4, 32 :8, 64: 16}
        args.hist_len = lens[args.image_dim] 
    if not args.use_instant_distance:
        args.reward_bound = args.step_reward    # no collisions
    args.result_dir = 'results/{}_{}_im{}_s{}_his{}_{}.txt'.format(
        args.train_mode, args.predict_net, args.image_dim, args.state_dim, args.hist_len, args.result_dir)
    return args

def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def train_single_net(args):
    start = time.time()
    print('Current time is: %s' % get_time())
    print('Starting at train_multi_nets...')
    # 定义是否使用GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # args.device_type = torch.device("cuda")
        args.device_type = torch.device("cpu")
        print("CUDA is available.")
    else:
        args.device_type = torch.device("cpu")
        print("CUDA is not available, fall back to CPU.")

    # Initial environment, replay memory, deep_q_net and agent
    env = Environment(args)
    mem = ReplayMemory(args)
    net = FRLDQN(args)
    agent = Agent(env, mem, net, args)

    best_result = {'valid': {'success_rate': {1: 0., 3: 0., 5: 0., 10: 0., -1 : 0.}, 'avg_reward': 0., 'log_epoch': -1, 'step_diff': -1},
                    'test': {'success_rate': {1: 0., 3: 0., 5: 0., 10: 0., -1 : 0.}, 'avg_reward': 0., 'log_epoch': -1, 'step_diff': -1}
    }

    # 打印参数
    # print('\n Arguments:')
    # for k, v in sorted(args.__dict__.items(), key=lambda x:x[0]):
    #     print('{}: {}'.format(k, v))
    # print('\n')

    try:
        for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
            agent.train(epoch, args.train_episodes, args.predict_net)
            print("test ",epoch)
            rate, reward, diff = agent.test(epoch, args.test_episodes, args.predict_net, 'valid')

            if rate[args.success_base] > best_result['valid']['success_rate'][args.success_base]:
                update_best(best_result, 'valid', epoch, rate, reward, diff)
                print('best_epoch: {}\t best_success: {}\t avg_reward: {}\t step_diff: {}\n'.format(epoch, rate, reward, diff))

                rate, reward, diff = agent.test(epoch, args.test_episodes, args.predict_net, 'test')
                update_best(best_result, 'test', epoch, rate, reward, diff)
                print('\n Test results:\n success_rate: {}\t avg_reward: {}\t step_diff: {}\n'.format(rate, reward, diff))

            if epoch - best_result['valid']['log_epoch'] >= args.stop_epoch_gap:
                print('-----Early stopping, no improvement after %d epochs-----\n' % args.stop_epoch_gap)
                break
    except KeyboardInterrupt:
        print('\n Manually kill the program ... \n')

    print('\n\n Best results:')
    for data_flag, results in best_result.items():
        print('\t{}'.format(data_flag))
        for k, v in results.items():
            print('\t\t{}: {}'.format(k, v))
    end = time.time()          
    print('Current time is: %s' % get_time())
    print('Total time cost: %ds\n' % (end - start))



def update_best(result, data_flag, epoch, rate, reward, diff, net_name=''):
    if net_name:
        result[data_flag][net_name]['success_rate'] = rate
        result[data_flag][net_name]['avg_reward'] = reward
        result[data_flag][net_name]['log_epoch'] = epoch
        result[data_flag][net_name]['step_diff'] = diff
    else:
        result[data_flag]['success_rate'] = rate
        result[data_flag]['avg_reward'] = reward
        result[data_flag]['log_epoch'] = epoch
        result[data_flag]['step_diff'] = diff


if __name__ == '__main__':
    args = args_init()
    train_single_net(args)

