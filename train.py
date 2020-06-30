#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#-*- coding: utf-8 -*-

import sys

# relative path
sys.path.append('.\\dqn')

from paddleball import Paddle

import numpy as np
from parl.utils import logger  # 日志打印工具

from model import Model
from algorithm import DQN  # from parl.algorithms import DQN  # parl >= 1.3.1
from agent import Agent

from replay_memory import ReplayMemory

## 策略1 大学习率大数据量
#LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
#MEMORY_SIZE = 20000  # replay memory的大小，越大越占用内存
#MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
#BATCH_SIZE = 32*5  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
#LEARNING_RATE = 0.001*5  # 学习率
#GAMMA = 0.9  # reward 的衰减因子，一般取 0.9 到 0.999 不等
#EPSILON = 1 # 探索率
#EPSILON_DEC = .995

## 策略2 小学习率大衰减因子
#LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
#MEMORY_SIZE = 20000  # replay memory的大小，越大越占用内存
#MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
#BATCH_SIZE = 32  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
#LEARNING_RATE = 0.001  # 学习率
#GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
#EPSILON = 1 # 探索率
#EPSILON_DEC = .995

## 策略3 2000+1000 改了下发球逻辑，现在发球方向随机
#LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
#MEMORY_SIZE = 20000  # replay memory的大小，越大越占用内存
#MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
#BATCH_SIZE = 32  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
#LEARNING_RATE = 0.001  # 学习率
#GAMMA = 0.999  # reward 的衰减因子，一般取 0.9 到 0.999 不等
#EPSILON = 0.5 # 探索率
#EPSILON_DEC = .995


# 训练一个episode
def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        reward, next_obs, done = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            reward, obs, done = env.step(action)
            episode_reward += reward 
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def main():
    env = Paddle()
    
    action_dim = 3 # jump or stay
    obs_shape = (5,)  # ball_num*2+box_num

    rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

    # 根据parl框架构建agent
    model = Model(act_dim=action_dim)
    algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        obs_dim=obs_shape[0],
        act_dim=action_dim,
        e_greed=EPSILON,  # 有一定概率随机选取动作，探索
        e_greed_decrement=EPSILON_DEC)  # 随着训练逐步收敛，探索的程度慢慢降低

    # 加载模型
    episode = 0
#    save_path = './dqn_model_{}.ckpt'.format(episode)
#    agent.restore(save_path)
#    logger.info('{} loaded.'.format(save_path))

    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(env, agent, rpm)

    max_episode = 2000
    save_data = []

    # start train 1     
    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # train part
        for i in range(0, 50):
            total_reward = run_episode(env, agent, rpm)
            episode += 1
            if i%2==0:
                agent.e_greed = max(0.01, agent.e_greed * agent.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低

        # test part
        eval_reward = evaluate(env, agent) 
        logger.info('episode:{}, e_greed:{}, test reward:{}'.format(
                episode, round(agent.e_greed,4), round(eval_reward,2)))
        save_data.append([episode,agent.e_greed,eval_reward]) 

    # 训练结束，保存模型
    save_path = './dqn_model_{}.ckpt'.format(episode)
    agent.save(save_path)
    
    np.save('./dqn_data_{}.npy'.format(episode), np.array(save_data))


if __name__ == '__main__':
    main()
