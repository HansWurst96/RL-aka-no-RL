import numpy as np
import gym
import quanser_robots

import util

import torch
import torch.nn as nn
import torch.nn.functional as F

env = gym.make('CartpoleSwingShort-v0')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class DQN():
    def __init__(self, input_dim, hidden_dim, output_dim, buffer_capacity, epsilon, discount_factor, learning_rate):
        self.buffer_capacity = buffer_capacity
        self.epsilon = epsilon
        self.gamma = discount_factor
        self.learning_rate = learning_rate

        self.network = Network(input_dim, hidden_dim, output_dim).to(device)
        self.target_network = Network(input_dim, hidden_dim, output_dim).to(device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
    def initialize_replay_buffer(self):
        print("Initializing Replay Buffer...")
        state = env.reset()
        count = 0
        steps = []
        for i in range(self.buffer_capacity):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            self.buffer.add_to_memory((state, action, next_state, reward, done))
            if done:
                state = env.reset()
                steps.append(count)
                count = 0
            else:
                count += 1
                state = next_state
        self.max_steps = np.array(steps).mean()
        print("Initializing Complete")
