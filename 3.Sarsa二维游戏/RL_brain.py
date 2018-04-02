# _*_ coding:utf-8 _*_
# !/usr/bin/python

import numpy as np
import pandas as pd


# 定义一个基类,除了learn()方法pass,其他方法和之前的Q_learning一模一样
class RL:
    # 初始化
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay   # 奖励衰减
        self.epsilon = e_greedy     # 贪婪度
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)   # 初始 q_table

    # 选行为
    def choose_action(self, observation):
        self.check_state_exist(observation)  # 检测本 state 是否在 q_table 中存在(不存在的状态步骤要添加到Q表中)

        # 选择 action
        if np.random.uniform() < self.epsilon:  # 小于0.9的随机数,选择 Q value 最高的 action
            state_action = self.q_table.ix[observation, :]

            # 同一个 state, 可能会有多个相同的 Q action value, 所以我们乱序一下
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.argmax()

        else:   # 随机选择 action
            action = np.random.choice(self.actions)

        return action

    # 学习更新Q表值,被继承时,此方法要重写
    def learn(self, *args):
        pass

        # self.check_state_exist(s_)  # 检测 q_table 中是否存在 s_
        # q_predict = self.q_table.ix[s, a]
        # if s_ != 'terminal':
        #     q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # 下个 state 不是 终止符
        # else:
        #     q_target = r  # 下个 state 是终止符
        # self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # 更新对应的 state-action 值

    # 检测 state 是否存在,不存在则添加状态
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )








# 定义QLearningTable类
class QLearningTable(RL):   # 继承了父类 RL
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)    # 表示继承关系

    def learn(self, s, a, r, s_):   # learn 的方法在每种类型中有不一样, 需重新定义
        self.check_state_exist(s_)  # 检测 q_table 中是否存在 s_
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # 下个 state 不是 终止符
        else:
            q_target = r  # 下个 state 是终止符
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # 更新对应的 state-action 值







# 定义SarsaTable类
class SarsaTable(RL):   # 继承 RL class

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)    # 表示继承关系

    def learn(self, s, a, r, s_, a_):  # # 多了一个a_(下一次的行动)参数
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            # # Sarsa现实的计算与Q_learning不同, q_target 基于选好的 a_ 而不是 Q(s_) 的最大值
            q_target = r + self.gamma * self.q_table.ix[s_, a_]
        else:
            q_target = r  # 如果 s_ 是终止符
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # 更新 q_table






