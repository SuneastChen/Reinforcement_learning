# _*_ coding:utf-8 _*_
# !/usr/bin/python

import numpy as np
import pandas as pd
import tensorflow as tf
import time


N_STATES = 6   # 1维世界的宽度
ACTIONS = ['left', 'right']     # 探索者的可用动作
EPSILON = 0.9   # 贪婪度 greedy,即多大的概率用Q表决策
ALPHA = 0.1     # 学习率
GAMMA = 0.9    # 奖励递减值,即对未来预见的衰减率
MAX_EPISODES = 13   # 最大回合数,玩的次数
FRESH_TIME = 0.3    # 移动间隔时间,多长时间走一步


# 定义Q表
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table 全 0 初始
        columns=actions,    # columns 对应的是行为名称
    )
    return table
# print(build_q_table(N_STATES,ACTIONS))
# q_table如下:
"""
   left  right
0   0.0    0.0
1   0.0    0.0
2   0.0    0.0
3   0.0    0.0
4   0.0    0.0
5   0.0    0.0
"""



# 在某个 state 地点, 选择行为
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]  # 选出这个 state 的所有 action 值
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # 随机选择动作的情况
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()    # 按Q表选择动作
    return action_name


# 定义环境的奖励:
# 做出行为后, 环境也要给我们的行为一个反馈, 输入上一个 state (S)并做出 action (A),反回出下个 state (S_) 和 所得到的 reward (R).
def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':    # move right,选择往右走,当到终点时,奖励值才+1,其他情况下往右,没有奖励(回合制)
        if S == N_STATES - 2:   # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   # move left,选择往左走,奖励值为0
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R



# 环境的更新(显示出o的行走路线):
def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)



# 强化学习主循环:
def rl():
    q_table = build_q_table(N_STATES, ACTIONS)  # 初始 q table
    for episode in range(MAX_EPISODES):     # 最大回合内循环
        step_counter = 0
        S = 0   # 回合初始位置
        is_terminated = False   # 是否达到了终点
        update_env(S, episode, step_counter)    # 每个回合后,进行环境更新显示
        while not is_terminated:

            A = choose_action(S, q_table)   # 选行为
            S_, R = get_env_feedback(S, A)  # 输入(S,A),返回S_,R
            q_predict = q_table.ix[S, A]    # 在S状态,选择A行为的预测Q表值
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   #  实际的S状态的Q表值=奖励+预见未来的衰减率*下个S_状态的Q表值 (回合没结束时)
            else:
                q_target = R     #  实际的S状态的Q表值=奖励值 (回合结束时)
                is_terminated = True    # 到达了终点,结束回合内的小循环

            q_table.ix[S, A] += ALPHA * (q_target - q_predict)  #  q_table 更新 += 学习率*(实际的Q表值-预测的Q表值)
            S = S_  # 探索者移动到下一个 state

            update_env(S, episode, step_counter+1)  # 环境更新

            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:')
    print(q_table)






















