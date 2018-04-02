# _*_ coding:utf-8 _*_
# !/usr/bin/python

from maze_env import Maze
from RL_brain import DeepQNetwork

# # 与Q_learnging比较不同的地方
def update():

    step = 0  # # 用来控制什么时候学习

    for episode in range(100):  # 学习 100 回合
        observation = env.reset()  # 初始化 state 的观测值'(1,1)'

        while True:
            env.render()  # 更新可视化环境,可见
            action = RL.choose_action(observation)  # RL 大脑根据 state 的观测值挑选 action

            # 探索者在环境中实施这个 action, 并得到环境返回的下一个 state 观测值, reward 和 done (是否是掉下地狱或者升上天堂)
            observation_, reward, done = env.step(action)

            # RL 从这个序列 (state, action, reward, state_) 中学习,DQN需要先存记忆,再学习
            # # RL.learn(str(observation), action, reward, str(observation_))

            # # DQN 存储记忆
            RL.store_transition(observation, action, reward, observation_)

            # # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            observation = observation_  # 将下一个 state 的值传到下一次循环

            if done:  # 如果掉下地狱或者升上天堂, 这回合就结束了
                break

            step += 1   # # 总步数

    print('game over')   # 结束游戏并关闭窗口
    env.destroy()

if __name__ == "__main__":
    # 定义环境 env 和 RL 方式
    env = Maze()

    # # 创建DeepQNetwork类,之前用的是默认参数
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,  # 每 200 步替换一次 target_net 的参数
                      memory_size=2000,  # 记忆上限
                      # output_graph=True   # 是否输出 tensorboard 文件
                      )

    # 开始可视化环境 env
    env.after(100, update)
    env.mainloop()

    RL.plot_cost()  # # 观看神经网络的误差曲线
