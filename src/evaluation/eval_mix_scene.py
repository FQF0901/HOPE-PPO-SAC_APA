import sys
sys.path.append("..")
sys.path.append(".")
import time
import os
import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# from model.MultiModalPPO_AF import PPO
from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.sac_agent import SACAgent as SAC
from model.agent.parking_agent import ParkingAgent, RsPlanner
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.vehicle import VALID_SPEED
from evaluation.eval_utils import eval
from configs import *


if __name__=="__main__":

    # 设置命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加模型检查点路径参数
    parser.add_argument('ckpt_path', type=str, default=None) # './model/ckpt/HOPE_SAC0.pt'
    # 添加评估轮次参数
    parser.add_argument('--eval_episode', type=int, default=2000)
    # 添加详细输出参数
    parser.add_argument('--verbose', type=bool, default=True)
    # 添加可视化参数
    parser.add_argument('--visualize', type=bool, default=True)
    # 解析命令行参数
    args = parser.parse_args()

    # 获取检查点路径
    checkpoint_path = args.ckpt_path
    print('ckpt path: ',checkpoint_path)
    # 获取详细输出设置
    verbose = args.verbose

    # 根据可视化参数选择环境渲染模式
    if args.visualize:
        raw_env = CarParking(fps=100, verbose=verbose)
    else:
        raw_env = CarParking(fps=100, verbose=verbose, render_mode='rgb_array')
    # 包装环境
    env = CarParkingWrapper(raw_env)

    # 设置相对路径和时间戳，用于保存评估结果
    relative_path = '.'
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = relative_path+'/log/eval/%s/' % timestamp
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 保存配置信息
    configs_file = os.path.join(save_path, 'configs.txt')
    with open(configs_file, 'w') as f:
        f.write(str(checkpoint_path))
    # 根据检查点路径选择代理类型
    Agent_type = PPO if 'ppo' in checkpoint_path.lower() else SAC
    # 初始化TensorBoard写入器
    writer = SummaryWriter(save_path)
    print("You can track the training process by command 'tensorboard --log-dir %s'" % save_path)

    # 设置随机种子以确保可重复性
    seed = SEED
    # env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 设置代理的演员和评论家网络参数
    actor_params = ACTOR_CONFIGS
    critic_params = CRITIC_CONFIGS
    configs = {
        "discrete": False,
        "observation_shape": env.observation_shape,
        "action_dim": env.action_space.shape[0],
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": "gaussian",
        "save_params": False,
        "actor_layers": actor_params,
        "critic_layers": critic_params,
    }
    print('observation_space:',env.observation_space)

    # 初始化强化学习代理
    rl_agent = Agent_type(configs)
    # 如果提供了检查点路径，则加载预训练模型
    if checkpoint_path is not None:
        rl_agent.load(checkpoint_path, params_only=True)
        print('load pre-trained model!')

    # 初始化停车规划器
    step_ratio = env.vehicle.kinetic_model.step_len*env.vehicle.kinetic_model.n_step*VALID_SPEED[1]
    rs_planner = RsPlanner(step_ratio)
    parking_agent = ParkingAgent(rl_agent, rs_planner)

    # 获取评估轮次
    eval_episode = args.eval_episode
    # 根据代理类型决定是否需要选择动作
    choose_action = True if isinstance(rl_agent, PPO) else False
    with torch.no_grad():
        # 在不同难度级别上进行评估
        # 评估极端情况
        env.set_level('Extrem')
        log_path = save_path+'/extreme'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)

        # 评估dragon lake park 情况
        env.set_level('dlp')
        log_path = save_path+'/dlp'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, multi_level=True, post_proc_action=choose_action)
        
        # 评估复杂情况
        env.set_level('Complex')
        log_path = save_path+'/complex'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)
        
        # 评估正常情况
        env.set_level('Normal')
        log_path = save_path+'/normalize'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)

    # 关闭环境
    env.close()