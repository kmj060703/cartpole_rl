import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from dqn import DQN
from replay_buffer import ReplayBuffer
from utils import select_action
import config

env = gym.make(config.ENV_NAME, render_mode="human")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = DQN(state_dim, action_dim)
optimizer = optim.Adam(q_net.parameters(), lr=config.LR)
loss_fn = nn.MSELoss()

memory = ReplayBuffer(config.MEMORY_SIZE)

rewards_history = []
epsilon = config.EPS_START

for ep in range(config.EPISODES):
    state, _ = env.reset()
    total_reward = 0

    while True:
        action = select_action(state, q_net, epsilon, action_dim)
        next_state, reward, done, _, _ = env.step(action)

        memory.push((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if len(memory) >= config.BATCH_SIZE:
            batch = memory.sample(config.BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            q_values = q_net(states).gather(1, actions).squeeze()
            next_q = q_net(next_states).max(1)[0]
            target = rewards + config.GAMMA * next_q * (1 - dones)

            loss = loss_fn(q_values, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    epsilon = max(config.EPS_END, epsilon * config.EPS_DECAY)
    rewards_history.append(total_reward)

    print(f"Episode {ep:3d} | Reward: {total_reward:4.0f} | Epsilon: {epsilon:.3f}")

env.close()

plt.plot(rewards_history)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("CartPole Training Curve")
plt.show()
