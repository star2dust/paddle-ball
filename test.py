# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:48:58 2020

@author: Woody
"""

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
import matplotlib.pyplot as plt
from parl.utils import logger  # 日志打印工具

from model import Model
from algorithm import DQN  # from parl.algorithms import DQN  # parl >= 1.3.1
from agent import Agent

LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 32  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.001  # 学习率
GAMMA = 0.95  # reward 的衰减因子，一般取 0.9 到 0.999 不等
EPSILON = 1 # 探索率

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent):
    eval_reward = []
    for i in range(15):
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
        
    # 测试
    env = Paddle()
    
    action_dim = 3 # jump or stay
    obs_shape = (5,)  # ball_num*2+box_num

    # 根据parl框架构建agent
    model = Model(act_dim=action_dim)
    algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        obs_dim=obs_shape[0],
        act_dim=action_dim,
        e_greed=0.1,  # 有一定概率随机选取动作，探索
        e_greed_decrement=1e-6)  # 随着训练逐步收敛，探索的程度慢慢降低

    # 加载模型
    test_model = 2
    save_path = './result/dqn_model_{}.ckpt'.format(test_model)
    agent.restore(save_path)
    logger.info('{} loaded.'.format(save_path))


    # test part
    eval_reward = evaluate(env, agent)  # render=True 查看显示效果
    logger.info('Test reward:{}'.format(round(eval_reward,2)))


if __name__ == '__main__':
    main()
