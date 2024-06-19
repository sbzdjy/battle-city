import retro
import torch
import numpy as np
from collections import deque
import math
import os
from gym import spaces
from algos.agents.dqn_agent import DQNAgent
from algos.models.dqn_cnn import DQNCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class CustomRetroEnv(retro.RetroEnv):
    def __init__(self, game, state=retro.State.DEFAULT, scenario=None, info=None, use_restricted_actions=retro.Actions.FILTERED,
                 record=False, players=1, inttype=retro.data.Integrations.STABLE, obs_type=retro.Observations.IMAGE):
        super(CustomRetroEnv, self).__init__(game, state, scenario, info, use_restricted_actions, record, players, inttype, obs_type)
        self.game_name = game
        # 获取裁剪信息
        x, y, w, h = self.data.crop_info(player=1)
        # 根据裁剪信息设置观察空间
        self.observation_space = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
    def current_reward(self, player=0):
        """ current_reward(self: retro._retro.GameDataGlue, player: int=0) -> float """
        # 获取游戏变量
        game_over = self.data.get_variable('GameOver')
        lives = self.data.get_variable('Lives')
        score = self.data.get_variable('Score')
        # 计算奖励
        if game_over:
            # 如果游戏结束且生命值大于0，给予惩罚
            reward = -1
        elif lives == 0:
            # 如果生命值为0，给予惩罚
            reward = -0.5
        return reward
    def was_moving(self, prev_action):
        """ 检查智能体在上一步是否执行了移动动作 """
        # 定义哪些动作是移动动作
        moving_actions = {2, 3, 4, 5}  # Up, Down, Left, Right
        return prev_action in moving_actions
    def step(self, action):
        ob, reward, done, info = super(CustomRetroEnv, self).step(action)
        reward = self.current_reward()
        # 检查智能体是否在上一步执行了移动动作
        if hasattr(self, 'prev_action') and not self.was_moving(self.prev_action):
            # 如果智能体没有移动，扣除分数
            reward -= 1  # 假设不移动时扣除的分数
        # 更新上一步的动作
        self.prev_action = action  # 注意这里存储的是动作的索引，而不是动作列表
        return ob, reward, done, info

    def reset(self):
        ob = super(CustomRetroEnv, self).reset()
        return ob

    def render(self, mode='human'):
        return super(CustomRetroEnv, self).render(mode=mode)

    def close(self):
        super(CustomRetroEnv, self).close()

class DQN(nn.Module):
    def __init__(self, h, w, output_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Linear input size is calculated based on the output size of the last convolutional layer
        linear_input_size = self._get_linear_input_size(h, w)

        self.fc1 = nn.Linear(in_features=linear_input_size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=output_size)

    def _get_linear_input_size(self, h, w):
        # Calculate the input size for the linear layer based on the output size of the last convolutional layer
        # You can use a dummy tensor to get the output size of the convolutional layers
        dummy_tensor = torch.randn(1, 4, h, w)
        conv1_output = self.conv1(dummy_tensor)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        return int(np.prod(conv3_output.size()))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 使用您的自定义环境进行训练
env = retro.make("F:/BattleCityAI-main/custom_integrations", inttype=retro.data.Integrations.ALL)
# env.close()  # 关闭之前创建的环境实例
# env = CustomRetroEnv(game='F:/BattleCityAI-main/custom_integrations')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

print("The size of frame is: ", env.observation_space.shape)
print("No. of Actions: ", env.action_space.n)
env.close()  # 关闭之前创建的环境实例
env = CustomRetroEnv(game='F:/BattleCityAI-main/custom_integrations')
env.reset()

possible_actions = {
    # No Operation
    0: [0, 0, 0, 0, 0, 0, 0, 0, 0],
    # a
    1: [1, 0, 0, 0, 0, 0, 0, 0, 0],
    # Up
    2: [0, 0, 0, 0, 1, 0, 0, 0, 0],
    # Down
    3: [0, 0, 0, 0, 0, 1, 0, 0, 0],
    # Left
    4: [0, 0, 0, 0, 0, 0, 1, 0, 0],
    # Right
    5: [0, 0, 0, 0, 0, 0, 0, 1, 0],
}


def random_play():
    # 用于随机地与游戏环境互动，以观察智能体的行为和得分
    score = 0
    env.reset()

    for i in range(2000):
        # env.render()
        action = possible_actions[np.random.randint(len(possible_actions))]
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            print("Your Score at end of game is: ", score)
            break
    env.reset()
    # env.render(close=True)


def stack_frames(frames, state, is_new=False): # 预处理后的游戏帧堆叠起
    frame = preprocess_frame(state, (1, -1, -1, 1), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames


INPUT_SHAPE = (4, 84, 84)  # 定义了神经网络的输入形状，即输入层的大小。
ACTION_SIZE = len(possible_actions)
SEED = 0
GAMMA = 0.85          # 折扣因子
BUFFER_SIZE = 100000   # 定义回放缓冲区的大小，存储获得的经验。
BATCH_SIZE = 128        # 定义更新神经网络时使用的经验批量的尺寸
LR = 0.01         # 定义神经网络学习率，控制优化算法更新权重时大小。
TAU = 1e-3             # 定义软更新参数，用于更新目标网络的参数。
UPDATE_EVERY = 500     # 定义网络更新的频率，即每多少个时间步更新一次网络
UPDATE_TARGET = 10000  # 定义开始进行目标网络软更新的阈值，即每积累多少次经验后进行一次目标网络的软更新。
EPS_START = 0.99     # 初始的探索率 epsilon
EPS_END = 0.01         # 探索率的最终值。
EPS_DECAY = 800         # 探索率随时间衰减的速率。

agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)


start_epoch = 0
scores = []
scores_window = deque(maxlen=20)
# epsilon 值随着时间步数增加而逐渐减小，从 EPS_START 开始，最终减小到 EPS_END。
epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx / EPS_DECAY)


def train(n_episodes=10000):
    ind = 0
    # 检查模型文件是否存在，并尝试加载它
    if os.path.exists(SCRIPT_DIR + "/models/dqnpro"):
        print("Loading previous model...")
        agent.load(SCRIPT_DIR + "/models/dqnpro")
    else:
        print("No previous model found. Starting from scratch.")

    for i_episode in range(start_epoch + 1, n_episodes + 1):
        state = stack_frames(None, env.reset(), True)
        score = 0
        eps = epsilon_by_epsiode(i_episode)
        lives = 3
        previous_score = 0
        while True:
            action = agent.act(state, eps)
            # print(f"Action taken: {action}")  # 打印智能体选择的动作
            next_state, reward, done, info = env.step(possible_actions[action])
            # env.render()
            env.prev_action = 0  # 这里存储的是动作的索引
            score += reward
            next_state = stack_frames(state, next_state, False)
            # 检测得分增加，假设击败敌方坦克或吃掉食物会导致得分增加
            current_score = info['Score']
            #  print(current_score)
            if current_score > previous_score:
                reward = +10000
            else:
                reward = -0.001
            # 更新上一个时间步的得分为当前的得分
            previous_score = current_score
            score += reward
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode {i_episode}, Epsilon: {eps:.6f},Score: {score:.2f}")
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        ind += 1
        if i_episode % 1000 == 0:
            agent.save(SCRIPT_DIR + "/models/dqnpro")
    return scores


def main():
    env.viewer = None
    agent.load(SCRIPT_DIR + "/models/dqnpro")
    # watch an untrained agent
    state = stack_frames(None, env.reset(), True)
    for j in range(10000):
        env.render()
        action = agent.act(state, eps=0.09)
        next_state, reward, done, _ = env.step(possible_actions[action])
        state = stack_frames(state, next_state, False)
        if done:
            env.reset()
            break
    env.render()


if __name__ == "__main__":
    scores = train(10000)
    # main()


