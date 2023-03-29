import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, Tensor
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import time
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 1e-3
hidden = 128
n_episodes = 2000
alpha = 0.999
capacity = 7000
allrewards = []
allloss = []
learn_time=10

def printfig(tlist, title):
    plt.plot(tlist)
    plt.title(title)
    # plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNet, self).__init__()
        self.func1 = nn.Linear(input_size, hidden_size)
        self.func2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = Tensor(x)
        hide = F.relu(self.func1(x))
        out = F.softmax(self.func2(hide), dim=-1)
        return out


def sample(memory):
    return zip(*memory)


def next_action(obs, net):
    prob = net(obs)
    m = Categorical(prob) # 探索机制同时按照动作概率进行采样
    action = m.sample() # 根据采样选择动作
    log_prob = m.log_prob(action)  # 策略梯度函数更新时的log项，之后只需要乘以alpha*reward即可
    return int(action), log_prob


def learn(memory, optimer):
    log_probs, rewards = sample(memory)
    log_probs = torch.stack(log_probs)
    discounts = [alpha ** i for i in range(len(rewards) + 1)]
    Ret = sum([a * b for a, b in zip(discounts, rewards)])
    loss = -Ret * log_probs
    allloss.append(loss.sum().item())
    optimer.zero_grad()
    loss.sum().backward()
    optimer.step()

def push(memory, *transition):
    if len(memory) == capacity:
        memory.pop(0)
    memory.append(transition)


def main():
    start_time = time.process_time()
    memory = []
    env = gym.make("CartPole-v0")
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    net = QNet(o_dim, hidden, a_dim)
    optimer = optim.Adam(net.parameters(), lr=lr)
    for i_episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, log_prob = next_action(obs, net)
            next_obs, reward, done, info = env.step(action)
            push(memory,log_prob, reward)
            '''
            if len(memory) == capacity:
                memory.pop(0)
            memory.append(log_prob)
            memory.append(reward)
            '''
            obs = next_obs
            episode_reward += reward
        learn(memory, optimer)
        memory.clear()
        allrewards.append(episode_reward)
        print(f"Episode: {i_episode}, Reward: {episode_reward}")
    end_time = time.process_time()
    print(f'耗时:{end_time - start_time}s')
    printfig(allloss, "loss")
    printfig(allrewards, "reward")


if __name__ == "__main__":
    main()
