import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 将当前.py路径的父父文件夹路劲加到系统路径
import time
from shutil import copyfile # 文件复制
import argparse # 命令行参数解析

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter   # SummaryWriter 是用于TensorBoard的写入器，用于记录训练过程中的数据
# 以下是 引入自定义模块和类
from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.parking_agent import ParkingAgent, RsPlanner
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.vehicle import VALID_SPEED,Status
from evaluation.eval_utils import eval
from configs import *

'''SceneChoose 类，用于场景选择和记录场景的成功率'''
class SceneChoose():
    def __init__(self) -> None:
        self.scene_types = {0:'Normal', 
                            1:'Complex',
                            2:'Extrem',
                            3:'dlp',
                            }
        self.target_success_rate = np.array([0.95, 0.95, 0.9, 0.99])
        self.success_record = {}    # 用于记录每种场景类型的成功次数列表
        for scene_name in self.scene_types:
            self.success_record[scene_name] = []
        self.scene_record = []  # 用于记录选择的场景类型
        self.history_horizon = 200  # 历史记录的时间窗口大小
        
        
    def choose_case(self,):
        if len(self.scene_record) < self.history_horizon:
            scene_chosen = self._choose_case_uniform()
        else:
            if np.random.random() > 0.5:
                scene_chosen = self._choose_case_worst_perform()    # 根据最近的成功率选择表现最差的场景类型
            else:
                scene_chosen = self._choose_case_uniform()  # 均匀地选择场景类型
        self.scene_record.append(scene_chosen)
        return self.scene_types[scene_chosen]
    
    def update_success_record(self, success:int):
        self.success_record[self.scene_record[-1]].append(success)
    
    '''下面两个私有方法'''
    def _choose_case_uniform(self,):
        case_count = np.zeros(len(self.scene_types))
        for i in range(min(len(self.scene_record), self.history_horizon)):
            scene_id = self.scene_record[-(i+1)]
            case_count[scene_id] += 1
        return np.argmin(case_count)
    
    def _choose_case_worst_perform(self,):
        success_rate = []
        for i in self.success_record.keys():
            idx = int(i)
            recent_success_record = self.success_record[idx][-min(250, len(self.success_record[idx])):]
            success_rate.append(np.sum(recent_success_record)/len(recent_success_record))
        fail_rate = self.target_success_rate - np.array(success_rate)
        fail_rate = np.clip(fail_rate, 0.01, 1)
        fail_rate = fail_rate/np.sum(fail_rate)
        return np.random.choice(np.arange(len(fail_rate)), p=fail_rate)

'''用于选择DLP ( Data Loss Prevention, 数据丢失预防 ）案例'''
# 这个类的目的是根据DLP案例的历史成功率，动态地选择下一个案例，以模拟或预测不同案例的选择概率。
# 通过记录每个案例的成功与失败情况，可以在选择时考虑案例的历史表现，从而影响选择概率的分布。
class DlpCaseChoose():
    def __init__(self) -> None:
        self.dlp_case_num = 248 # 可用的DLP案例的数量，这里设置为 248
        self.case_record = []   # 用于记录选择的案例
        self.case_success_rate = {}
        for i in range(self.dlp_case_num):
            self.case_success_rate[str(i)] = []
        self.horizon = 500  # 历史记录的时间窗口大小，设置为 500
    
    '''choose_case 方法用于选择下一个DLP案例'''
    def choose_case(self,):
        if np.random.random()<0.2 or len(self.case_record)<self.horizon:
            return np.random.randint(0, self.dlp_case_num)
        success_rate = []
        for i in range(self.dlp_case_num):
            idx = str(i)
            if len(self.case_success_rate[idx]) <= 1:
                success_rate.append(0)
            else:
                recent_success_record = self.case_success_rate[idx][-min(10, len(self.case_success_rate[idx])):]
                success_rate.append(np.sum(recent_success_record)/len(recent_success_record))
        fail_rate = 1-np.array(success_rate)
        fail_rate = np.clip(fail_rate, 0.005, 1)
        fail_rate = fail_rate/np.sum(fail_rate)
        return np.random.choice(np.arange(len(fail_rate)), p=fail_rate)
    
    '''用于更新指定案例的成功记录'''
    def update_success_record(self, success:int, case_id:int):
        self.case_success_rate[str(case_id)].append(success)    # 表示成功与否
        self.case_record.append(case_id)
        

if __name__=="__main__":
    
    ''' 1. 解析命令行参数 '''
    # 使用 argparse 模块来解析命令行参数，这些参数可以在运行脚本时指定
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_ckpt', type=str, default=None) # './model/ckpt/PPO.pt'
    parser.add_argument('--img_ckpt', type=str, default='./model/ckpt/autoencoder.pt')  # 图像编码器模型的检查点路径
    parser.add_argument('--train_episode', type=int, default=100000)    # 训练的总episode数
    parser.add_argument('--eval_episode', type=int, default=2000)   # 评估的episode数
    parser.add_argument('--verbose', type=bool, default=True)   # 是否打印详细信息
    parser.add_argument('--visualize', type=bool, default=True) # 是否可视化
    args = parser.parse_args()

    verbose = args.verbose  # TBD, 利于调试信息打印

    ''' 2. 根据参数设置环境 '''
    if args.visualize:
        raw_env = CarParking(fps=100, verbose=verbose,)
    else:
        raw_env = CarParking(fps=100, verbose=verbose, render_mode='rgb_array')
    env = CarParkingWrapper(raw_env)

    ''' 3. 初始化场景选择器 '''
    scene_chooser = SceneChoose()
    dlp_case_chooser = DlpCaseChoose()

    ''' 4. 设置日志和模型保存路径 '''
    # the path to log and save model
    relative_path = '.'
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = relative_path+'/log/exp/ppo_%s/' % timestamp

    if not os.path.exists(save_path):
        os.makedirs(save_path)  # 如果路径不存在，则递归地创建该路径

    writer = SummaryWriter(save_path)   # 创建一个用于写入 TensorBoard 日志的 SummaryWriter 对象
    # configs log
    copyfile('./configs.py', save_path+'configs.txt')   # 将当前目录下的 configs.py 文件复制到 save_path 目录下，并命名为 configs.txt
    print("You can track the training process by command 'tensorboard --log-dir %s'" % save_path)   # 输出一条消息，提示用户可以通过运行 tensorboard 命令并指定 --log-dir 参数来跟踪训练过程中生成的日志文件

    ''' 5. 设置随机种子 '''
    seed = SEED
    # env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    '''6. 配置Agent模型参数 '''
    actor_params = ACTOR_CONFIGS
    critic_params = CRITIC_CONFIGS
    configs = {
        "discrete": False,  # 这里设置为 False 表示动作空间是连续的
        "observation_shape": env.observation_shape, # shape应该理解为 维度 吧
        "action_dim": env.action_space.shape[0],    # 给出了动作空间的维度
        "hidden_size": 64,  # 神经网络中隐藏层的大小
        "activation": "tanh",   # 神经网络中的激活函数类型
        "dist_type": "gaussian",    # 策略网络的输出分布类型。这里选择了高斯分布，通常用于连续动作空间的策略网络
        "save_params": False,
        "actor_layers": actor_params,
        "critic_layers": critic_params,
    }

    ''' 7. 加载预训练模型 '''
    rl_agent = PPO(configs)
    checkpoint_path = args.agent_ckpt
    if checkpoint_path is not None: # 首先检查是否提供了预训练模型的检查点路径 args.agent_ckpt
        # 调用 rl_agent 对象的 load 方法来加载预训练模型
        # 参数 params_only=True 表示仅加载模型的参数而不加载优化器状态或其他附加信息。
        # 这通常用于在训练过程中恢复模型参数以继续训练或用于评估
        rl_agent.load(checkpoint_path, params_only=True)    
        print('load pre-trained model!')

    ''' 8. 加载图像编码器模型 '''
    img_encoder_checkpoint =  args.img_ckpt if USE_IMG else None
    if img_encoder_checkpoint is not None and os.path.exists(img_encoder_checkpoint):
        rl_agent.load_img_encoder(img_encoder_checkpoint, require_grad=UPDATE_IMG_ENCODE)
    else:
        print('not load img encoder')

    ''' 9. 设置规划器和agent '''
    step_ratio = env.vehicle.kinetic_model.step_len*env.vehicle.kinetic_model.n_step*VALID_SPEED[1]
    rs_planner = RsPlanner(step_ratio)
    parking_agent = ParkingAgent(rl_agent, rs_planner)  # 是个 PPO + RS 的 hybrid planner

    ''' 10. 初始化存储变量 '''
    reward_list = []
    reward_per_state_list = []
    reward_info_list = []
    case_id_list = []
    succ_record = []
    best_success_rate = [0, 0, 0, 0]

    ''' 11. train func '''
    for i in range(args.train_episode):
        # 11.1 场景选择器
        scene_chosen = scene_chooser.choose_case()
        if scene_chosen == 'dlp':
            case_id = dlp_case_chooser.choose_case()
        else:
            case_id = None
        
        # 11.2 初始化
        obs = env.reset(case_id, None, scene_chosen)
        parking_agent.reset()
        case_id_list.append(env.map.case_id)
        done = False
        total_reward = 0
        step_num = 0
        reward_info = []
        xy = []

        # 11.3 主循环：一个完整的训练周期
        while not done:
            step_num += 1
            action, log_prob = parking_agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            reward_info.append(list(info['reward_info'].values()))
            total_reward += reward
            reward_per_state_list.append(reward)
            parking_agent.push_memory((obs, action, reward, done, log_prob, next_obs))
            obs = next_obs

            # 11.3.1 当buffer够大时，开始更新 net para
            if len(parking_agent.memory) % parking_agent.configs.batch_size == 0:
                if verbose:
                    print("Updating the agent.")
                actor_loss, critic_loss = parking_agent.update()
                writer.add_scalar("actor_loss", actor_loss, i)
                writer.add_scalar("critic_loss", critic_loss, i)
            
            # 11.3.2 这啥意思？
            if info['path_to_dest'] is not None:
                parking_agent.set_planner_path(info['path_to_dest'])

            # 11.3.3 处理done后的状态
            if done:
                if info['status']==Status.ARRIVED:
                    succ_record.append(1)
                    scene_chooser.update_success_record(1)
                    if scene_chosen == 'dlp':
                        dlp_case_chooser.update_success_record(1, case_id)
                else:
                    succ_record.append(0)
                    scene_chooser.update_success_record(0)
                    if scene_chosen == 'dlp':
                        dlp_case_chooser.update_success_record(0, case_id)

        # 11.4 记录训练统计信息 
        writer.add_scalar("total_reward", total_reward, i)
        writer.add_scalar("avg_reward", np.mean(reward_per_state_list[-1000:]), i)
        writer.add_scalar("action_std0", parking_agent.log_std.detach().cpu().numpy().reshape(-1)[0],i)
        writer.add_scalar("action_std1", parking_agent.log_std.detach().cpu().numpy().reshape(-1)[1],i)
        for type_id in scene_chooser.scene_types:
            writer.add_scalar("success_rate_%s"%scene_chooser.scene_types[type_id],
                np.mean(scene_chooser.success_record[type_id][-100:]), i)
        writer.add_scalar("step_num", step_num, i)
        reward_list.append(total_reward)
        reward_info = np.sum(np.array(reward_info), axis=0)
        reward_info = np.round(reward_info,2)
        reward_info_list.append(list(reward_info))

        # 11.5 输出调试信息 
        if verbose and i%10==0 and i>0:
            print('success rate:',np.sum(succ_record),'/',len(succ_record))
            print(parking_agent.log_std.detach().cpu().numpy().reshape(-1))
            print("episode:%s  average reward:%s"%(i,np.mean(reward_list[-50:])))
            print(np.mean(parking_agent.actor_loss_list[-100:]),np.mean(parking_agent.critic_loss_list[-100:]))
            print('time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward')
            for j in range(10):
                print(case_id_list[-(10-j)],reward_list[-(10-j)],reward_info_list[-(10-j)])
            print("")

        # 11.6 保存最优参数
        # 11.6.1 统计近百次的成功率
        for type_id in scene_chooser.scene_types:
            success_rate_normal = np.mean(scene_chooser.success_record[0][-100:])
            success_rate_complex = np.mean(scene_chooser.success_record[1][-100:])
            success_rate_extreme = np.mean(scene_chooser.success_record[2][-100:])
            success_rate_dlp = np.mean(scene_chooser.success_record[3][-100:])

        # 11.6.2 比较成功率，并保存最有模型
        if success_rate_normal >= best_success_rate[0] and success_rate_complex >= best_success_rate[1] and\
            success_rate_extreme >= best_success_rate[2] and success_rate_dlp >= best_success_rate[3] and i>100:
            raw_best_success_rate = np.array([success_rate_normal, success_rate_complex, success_rate_extreme, success_rate_dlp])
            best_success_rate = list(np.minimum(raw_best_success_rate, scene_chooser.target_success_rate))
            parking_agent.save("%s/PPO_best.pt" % (save_path),params_only=True) # 保存模型和日志 (parking_agent.save 和日志记录)
            f_best_log = open(save_path+'best.txt', 'w')
            f_best_log.write('epoch: %s, success rate: %s %s %s %s'%(i+1, raw_best_success_rate[0],
                                raw_best_success_rate[1], raw_best_success_rate[2], raw_best_success_rate[3]))
            f_best_log.close()

        # 11.6.3 定期保存模型（每隔2000次迭代保存一次，即便不是最优）
        if (i+1) % 2000 == 0:
            parking_agent.save("%s/PPO2_%s.pt" % (save_path, i),params_only=True)

        # 11.6.4 绘制并保存奖励曲线图
        if verbose and i%20==0:
            episodes = [j for j in range(len(reward_list))]
            mean_reward = [np.mean(reward_list[max(0,j-50):j+1]) for j in range(len(reward_list))]
            plt.plot(episodes,reward_list)
            plt.plot(episodes,mean_reward)
            plt.xlabel('episodes')
            plt.ylabel('reward')
            f = plt.gcf()
            f.savefig('%s/reward.png'%save_path)
            f.clear()

    ''' 12. 模型评估 '''
    # evaluation
    eval_episode = args.eval_episode
    choose_action = True
    with torch.no_grad():   # 使用torch.no_grad()上下文管理器,确保在评估过程中不进行梯度计算，以节省内存和提高计算效率
        # eval on dlp
        env.set_level('dlp')
        log_path = save_path+'/dlp'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)
        
        # eval on extreme
        env.set_level('Extrem')
        log_path = save_path+'/extreme'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)
        
        # eval on complex
        env.set_level('Complex')
        log_path = save_path+'/complex'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)
        
        # eval on normalize
        env.set_level('Normal')
        log_path = save_path+'/normalize'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)

    env.close()