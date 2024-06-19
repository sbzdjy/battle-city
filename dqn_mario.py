import retro
import torch
import numpy as np
from collections import deque
import math
import os

from algos.agents.dqn_agent import DQNAgent
from algos.models.dqn_cnn import DQNCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

env = retro.make("SuperMarioBros-Nes", inttype=retro.data.Integrations.ALL)
env.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

print("The size of frame is: ", env.observation_space.shape)
print("No. of Actions: ", env.action_space.n)
env.reset()

possible_actions = {
    # No Operation
    0: [0, 0, 0, 0, 0, 0, 0, 0, 0],
    # Left
    1: [0, 0, 0, 0, 0, 0, 1, 0, 0],
    # Right
    2: [0, 0, 0, 0, 0, 0, 0, 1, 0],
    # Left + b
    3: [0, 1, 0, 0, 0, 0, 1, 0, 0],
    # Right + b
    4: [0, 1, 0, 0, 0, 0, 0, 1, 0],
    # Left + a
    3: [1, 0, 0, 0, 0, 0, 1, 0, 0],
    # Right + a
    4: [1, 0, 0, 0, 0, 0, 0, 1, 0],
    # b
    5: [0, 1, 0, 0, 0, 0, 0, 0, 0],
    # a
    6: [1, 0, 0, 0, 0, 0, 0, 0, 0],
}


def random_play(action_n):
    # 用于随机地与游戏环境互动，以观察智能体的行为和得分
    score = 0
    env.reset()

    for i in range(2000):
        env.render()
        action = possible_actions[np.random.randint(len(possible_actions))]
        state, reward, done, _ = env.step(possible_actions[action_n])
        score += reward
        if done:
            print("Your Score at end of game is: ", score)
            break
    env.reset()
    env.render(close=True)


def stack_frames(frames, state, is_new=False):  # 预处理后的游戏帧堆叠起
    frame = preprocess_frame(state, (1, -1, -1, 1), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames


INPUT_SHAPE = (4, 84, 84)  # 定义了神经网络的输入形状，即输入层的大小。
ACTION_SIZE = len(possible_actions)  # 定义了智能体可以采取的动作的数量
SEED = 0
GAMMA = 0.99  # 折扣因子
BUFFER_SIZE = 100000  # 定义回放缓冲区的大小，存储获得的经验。
BATCH_SIZE = 32  # 定义更新神经网络时使用的经验批量的尺寸
LR = 0.0001  # 定义神经网络学习率，控制优化算法更新权重时大小。
TAU = 1e-3  # 定义软更新参数，用于更新目标网络的参数。
UPDATE_EVERY = 1000  # 定义网络更新的频率，即每多少个时间步更新一次网络
UPDATE_TARGET = 10000  # 定义了开始进行目标网络软更新的阈值，即每积累多少次经验后进行一次目标网络的软更新。
EPS_START = 0.99  # starting value of epsilon
EPS_END = 0.01  # Ending value of epsilon
EPS_DECAY = 500  # Rate by which epsilon to be decayed

agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY,
                 UPDATE_TARGET, DQNCnn)

start_epoch = 0
scores = []
scores_window = deque(maxlen=20)

epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx / EPS_DECAY)


def train(n_episodes=1000):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """

    ind = 0

    for i_episode in range(start_epoch + 1, n_episodes + 1):
        state = stack_frames(None, env.reset(), True)
        score = 0
        eps = epsilon_by_epsiode(i_episode)

        if i_episode % 100 == 0:
            ind = 0

        while True:
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(possible_actions[action])
            env.render()

            score += reward

            next_state = stack_frames(state, next_state, False)
            agent.step(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score

        ind += 1
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.6f}\tInd: {ind}',
              end="")

    return scores


def main():
    env.viewer = None
    # watch an untrained agent
    state = stack_frames(None, env.reset(), True)
    for j in range(10000):
        env.render(close=False)
        action = agent.act(state, eps=0.91)
        next_state, reward, done, _ = env.step(possible_actions[action])
        state = stack_frames(state, next_state, False)
        if done:
            env.reset()
            break
    env.render(close=True)


if __name__ == "__main__":
    # random_play(6)
    scores = train(1000)
    # main()
