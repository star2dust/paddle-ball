# DQN to Play Paddle-Ball Game

## 终极复现项目说明

本项目基于使用[PARL](https://github.com/PaddlePaddle/PARL)框架实现算法解决一星环境任务[Orbit](https://github.com/shivaverma/Orbit)中的Paddle-Ball游戏。

- AI Studio昵称：star2dust
- 环境星级：一星
- 环境名称：Paddle
- Github链接：https://github.com/star2dust/paddle-ball
- 展示效果图

## 安装依赖项（requirements）

```shell
# or try: pip install -r requirements.txt
pip install paddlepaddle==1.6.3
pip install parl==1.3.1
pip install gym
pip install atari-py
pip install rlschool==0.3.1
```

## 训练策略（strategy）

本项目训练策略分三步进行。
```python
# 策略1 大学习率大数据量大探索率
BATCH_SIZE = 32*5  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.001*5  # 学习率
EPSILON = 1 # 探索率
EPSILON_DEC = .995
# 策略2 小学习率大衰减因子
BATCH_SIZE = 32  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.001  # 学习率
GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
# 策略3 小探索率大衰减因子（改了下发球逻辑，现在发球方向随机）
GAMMA = 0.999  # reward 的衰减因子，一般取 0.9 到 0.999 不等
EPSILON = 0.5 # 探索率
EPSILON_DEC = .995
```

## 训练效果（result）

效果说明：

- 第一步需要大探索率，不然根本碰不到球，所以尽可能随机性大一点
- 训练一段时间后，白板总是蜷缩在右下角（如图1），发现原环境的发球逻辑有问题，每次必然向右发，于是改了下逻辑，现在发球方向随机
- 为了把原先训练的结果拉回去，把衰减因子调到接近1，考虑动的收益，同时探索率稍微减少，保证能探索到左边（如图2）

图1，15次reward均值0.33：

<img src="result\test_1.gif" alt="test1" width="250" />

图2，15次reward均值7.06：

<img src="result\test_2.gif" alt="test2" width="250" />

代码运行如下：

```python
# train
python .\train.py
# test
python .\test.py
```
