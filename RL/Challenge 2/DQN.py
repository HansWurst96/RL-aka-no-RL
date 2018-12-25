import numpy as np
import matplotlib.pyplot as plt
import time
import gym
import quanser_robots

import util

import torch
import torch.nn as nn
import torch.nn.functional as F

env = gym.make('CartpoleSwingShort-v0')
#env = gym.make('Pendulum-v0')
#env = gym.make('MountainCarContinuous-v0')

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
#  GLOBAL CONSTANTS ####################################################################################################
GAMMA = 0.99
EPSILON_INIT = 1
EPSILON_FINAL = 0.1
EPSILON_DECAY = 50000
C = 3000
RANDOM_SAMPLE_SIZE = 100000
BATCH_SIZE = 8

NUM_ACTIONS = 7
########################################################################################################################


class Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, s):
        # x = torch.cat((s, a), dim=1)
        x = s
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class DQN:
    def __init__(self, input_dim, hidden_dim, buffer_capacity, learning_rate):
        self.hidden_dim = hidden_dim
        self.buffer_capacity = buffer_capacity
        self.learning_rate = learning_rate
        self.discrete_actions = self.discretize_action_space()
        self.buffer = util.ReplayMemory(self.buffer_capacity, input_dim, self.discrete_actions.shape[0])

        self.network = Network(input_dim, hidden_dim, self.discrete_actions.shape[0]).to(device)
        self.target_network = Network(input_dim, hidden_dim, self.discrete_actions.shape[0]).to(device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        self.losses = []
        self.rewards = []

    def discretize_action_space(self):
        high = env.action_space.high
        low = -high
        return np.linspace(low, high, NUM_ACTIONS)

    def initialize_replay_buffer(self):
        print("Initializing Replay Buffer...")
        state = env.reset()
        count = 0
        steps = []
        for i in range(self.buffer_capacity):
            action = np.array(np.random.choice(self.discrete_actions)).reshape(1, 1)
            next_state, reward, done, info = env.step(action)
            next_state = np.array(next_state)
            state = np.array(state)
            self.buffer.add_to_memory((np.array(state).reshape(state.shape[0]), action, np.array(next_state).reshape(next_state.shape[0]), reward, done))
            if done:
                state = env.reset()
                steps.append(count)
                count = 0
            else:
                count += 1
                state = next_state
        #self.max_steps = np.array(steps).mean()
        print("Initializing Complete")


    def train(self, epochs):
        epsilon = EPSILON_INIT
        eps_decay = (EPSILON_FINAL/EPSILON_INIT)**(1.0/EPSILON_DECAY)
        update_count = 0
        target_count = 0
        rewards, losses = [], []
        for episode in range(epochs):
            state = env.reset()
            action = np.random.choice(self.discrete_actions)
            done = False
            if episode % 10 == 0:
                print("Starting Episode: ", episode, ("({} %)".format(100 * np.round(float(np.float64(episode/epochs)), 3))))
            while True:
                state = torch.FloatTensor(state).to(device)
                if np.random.uniform() <= epsilon:
                    action = np.random.choice(self.discrete_actions)
                else:
                    action_index = torch.argmax(self.network(state.reshape(state.shape[0]))).detach().data.cpu().numpy()
                    action = np.array(self.discrete_actions[action_index])

                if epsilon >= EPSILON_FINAL:
                    if epsilon <= 1.01 * EPSILON_FINAL:
                        print("Epsilon limit has been reached")
                    epsilon *= eps_decay

                next_state, reward, done, info = env.step(np.array(action).reshape(1, 1))
                next_state = np.array(next_state)
                state = np.array(state)
                if done:
                    break

                self.buffer.add_to_memory((np.array(state).reshape(state.shape[0]), action, np.array(next_state).reshape(next_state.shape[0]), reward, done))

                state = next_state

                state_array, action_array, next_state_array, reward_array, done_array = self.buffer.sample(BATCH_SIZE)
                state_array = torch.FloatTensor(state_array).to(device)
                action_array = torch.FloatTensor(action_array).to(device)
                reward_array = torch.FloatTensor(reward_array).to(device)
                next_state_array = torch.FloatTensor(next_state_array).to(device)
                done_array = torch.FloatTensor(done_array.astype(int)).to(device)
                not_done_array = 1 - done_array

                q_primes = self.target_network(next_state_array).detach().max(1)[0].reshape(BATCH_SIZE, 1)
                q_primes = np.multiply(not_done_array, q_primes).to(device)
                targets = reward_array + GAMMA * q_primes
                q_values = self.network(state_array)
                #print(q_values)
                indices = np.array([np.where(self.discrete_actions == a)[0] for a in action_array])
                q_values = q_values[range(q_values.shape[0]), indices.T].t()
                loss = F.mse_loss(q_values, targets)

                losses.append(loss.detach().data.cpu().numpy())
                rewards.append(reward)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                update_count += 1
                if update_count % C == 0:
                    self.target_network.load_state_dict(self.network.state_dict())
                    target_count += 1
                    if target_count % 5 == 0:
                        print("Target Network updated ({})".format(target_count))
                    update_count = 0
            self.rewards.append(np.mean(rewards))
            self.losses.append(np.mean(losses))

    def visualize(self):
        fig, (loss_plot, reward_plot) = plt.subplots(2, 1, sharex=True)
        loss_plot.plot([x for x in range(len(self.losses))], self.losses, label='Loss')
        loss_plot.set_title(env)
        #loss_plot.ylabel("Loss")
        reward_plot.plot([x for x in range(len(self.rewards))], self.rewards, label='Reward')
        #reward_plot.ylabel("Average reward per timestep")
        plt.text(0, -6, "C: {} \nBatch size: {} \nEpsilon: ({}, {}, {})\nGamma: {}\nNetwork: ({},)"
                 .format(C, BATCH_SIZE, EPSILON_INIT, EPSILON_FINAL, EPSILON_DECAY, GAMMA, self.hidden_dim))
        plt.show()

    def test_policy(self, render=False):
        rewards = []
        for i in range(100):
            state = env.reset()
            done = False
            total_reward = 0
            time_steps = 0
            while not done:
                state = torch.FloatTensor(state).to(device)
                action_index = torch.argmax(self.network(state.reshape(state.shape[0]))).detach().data.cpu().numpy()
                action = np.array(self.discrete_actions[action_index])
                next_state, reward, done, info = env.step(np.array(action).reshape(1, 1))
                next_state = np.array(next_state)
                if render:
                    env.render()
                total_reward += reward
                time_steps += 1
                state = next_state
            rewards.append(total_reward/time_steps)

        random_rewards = util.random_sampling(env)

        plt.plot([i for i in range(len(rewards))], rewards, label='trained')
        plt.plot([i for i in range(len(random_rewards))], random_rewards, label='random')
        plt.title("Training: {}".format(np.round(np.mean(rewards), 4)))
        plt.legend()
        # print(rewards)
        plt.show()

def run():
    dqn = DQN(env.observation_space.shape[0], 100, RANDOM_SAMPLE_SIZE, 1e-3)
    dqn.initialize_replay_buffer()
    start = time.time()  # cpu(15): 143 bs(4), gpu: 896
    s, a, n, r, d = dqn.buffer.sample(8)
    dqn.train(350)
    print("Required time: ", np.round(time.time() - start, 3))
    dqn.visualize()
    dqn.test_policy(True)


run()

