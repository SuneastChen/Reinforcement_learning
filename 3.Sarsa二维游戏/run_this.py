# _*_ coding:utf-8 _*_
# !/usr/bin/python

from maze_env import Maze
from RL_brain import SarsaTable

def update():
    for episode in range(100):  # 学习 100 回合
        observation = env.reset()  # 初始化 state 的观测值(1,1)
        # # 在这里 --->> RL 大脑根据 state 的观测值挑选 action
        action = RL.choose_action(str(observation))

        while True:
            env.render()  # 更新可视化环境,可见
            # # action = RL.choose_action(str(observation))  # Q_learning在这里行动

            # 探索者在环境中实施这个 action, 并得到环境返回的下一个 state 观测值, reward 和 done (是否是掉下地狱或者升上天堂)
            observation_, reward, done = env.step(action)

            # # 根据下一个 state (obervation_) 选取下一个 action_
            action_ = RL.choose_action(str(observation_))

            # RL 从这个序列 (state, action, reward, state_) 中学习, # # 多了一个action_参数
            RL.learn(str(observation), action, reward, str(observation_), action_)

            observation = observation_  # 将下一个 state 的值传到下一次循环
            # # 下一个action_的值,变成action
            action = action_

            if done:  # 如果掉下地狱或者升上天堂, 这回合就结束了
                break

    print('game over')   # 结束游戏并关闭窗口
    env.destroy()

if __name__ == "__main__":
    # 定义环境 env 和 RL 方式
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))   # # 创建SaraTable类

    # 开始可视化环境 env
    env.after(100, update)
    env.mainloop()
