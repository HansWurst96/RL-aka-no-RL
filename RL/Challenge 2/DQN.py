import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import gym
import quanser_robots

import util

import torch
import torch.nn as nn
import torch.nn.functional as F

env_name = 'CartpoleSwingShort-v0'
#env_name = 'Pendulum-v0'

env = gym.make(env_name)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
#  GLOBAL CONSTANTS ####################################################################################################
GAMMA = 0#0.95
EPSILON_INIT = 0#1
EPSILON_FINAL = 0#0.05
EPSILON_DECAY = 0#300000
C = 0#1000
RANDOM_SAMPLE_SIZE = 0#70000
BATCH_SIZE = 0#32

NUM_ACTIONS = 0#7

LEARNING_RATE = 0#1e-5

HIDDEN_NODES = 0# 150
########################################################################################################################

def set_hyperparameters(buffer_size, batch_size, hidden_nodes, lr, c, epsilon, gamma, actions):

    global GAMMA
    GAMMA = gamma
    global HIDDEN_NODES
    HIDDEN_NODES = hidden_nodes
    global RANDOM_SAMPLE_SIZE
    RANDOM_SAMPLE_SIZE = buffer_size
    global BATCH_SIZE
    BATCH_SIZE = batch_size
    global LEARNING_RATE
    LEARNING_RATE = lr
    global C
    C = c
    global EPSILON_INIT
    EPSILON_INIT = epsilon[0]
    global EPSILON_FINAL
    EPSILON_FINAL = epsilon[1]
    global EPSILON_DECAY
    EPSILON_DECAY = epsilon[2]
    global NUM_ACTIONS
    NUM_ACTIONS = actions

class Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self.layer2.weight)

        self.dropout = nn.Dropout(0.5)

    def forward(self, s):
        # x = torch.cat((s, a), dim=1)
        x = s
        x = F.relu(self.layer1(x))
        #x = self.dropout(x)
        x = self.layer2(x)
        return x


class DQN:
    def __init__(self, input_dim, hidden_dim, buffer_capacity, learning_rate):
        self.hidden_dim = hidden_dim
        self.buffer_capacity = buffer_capacity
        self.learning_rate = learning_rate
        self.discrete_actions = self.discretize_action_space()
        self.buffer = util.ReplayMemory(self.buffer_capacity, input_dim, self.discrete_actions.shape[0])
        self.max_steps = 0
        self.start = 0
        self.end = 0

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
        self.max_steps = np.array(steps).mean()
        print("Initializing Complete")


    def train(self, epochs):
        epsilon = EPSILON_INIT
        eps_decay = (EPSILON_FINAL/EPSILON_INIT)**(1.0/EPSILON_DECAY)
        update_count = 0
        target_count = 0
        rewards, losses = [], []
        for episode in range(epochs):
            episode_steps = 0
            state = env.reset()
            action = np.random.choice(self.discrete_actions)
            done = False
            if episode % 10 == 0:
                print("Starting Episode: ", episode, ("({} %)".format(100 * np.round(float(np.float64(episode/epochs)), 3))))
            while episode_steps < 3 * self.max_steps:
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
    def choose_action(self, state):

        state = torch.FloatTensor(state).to(device)
        action_index = torch.argmax(self.network(state.reshape(state.shape[0]))).detach().data.cpu().numpy()
        action = np.array(self.discrete_actions[action_index])
        return action

    def load_model(self, path):
        self.network.load_state_dict(torch.load(path))

    def visualize(self, time, epochs):

        mean_end_reward = np.mean(self.rewards[int(len(self.rewards) * 0.85):])
        date = datetime.datetime.now().strftime("%m-%d %H-%M")
        path = 'C:\\Users\\Jonas\\Desktop\\plots\\DQN\\'

        fig, (loss_plot, reward_plot) = plt.subplots(2, 1, sharex=True)
        loss_plot.plot([x for x in range(len(self.losses))], self.losses, label='Loss')
        loss_plot.set_title(util.shorten_name(env_name) + " Actions: " + str(NUM_ACTIONS) + " Time: " + str(int(time)) + " Peak : " + str(np.round(np.max(self.rewards), 3)) + ' 15%: ' + str(np.round(mean_end_reward, 3)))
        #loss_plot.ylabel("Loss")
        reward_plot.plot([x for x in range(len(self.rewards))], self.rewards, label='Reward')
        arr = np.array(np.copy(self.rewards))
        arr = np.nanmean(
            np.pad(arr.astype(float), (0, ((10 - arr.size % 10) % 10)), mode='constant',
                   constant_values=np.NaN).reshape(-1, 10), axis=1)
        reward_plot.plot([10 * (i + 1) for i in range(len(arr))], arr, label='mean training reward')
        #reward_plot.ylabel("Average reward per timestep")
        plt.text(0.1 * len(self.rewards), np.min(self.rewards),
                 "C: {} \nBatch size: {} \nEpsilon: ({}, {}, {})\nGamma: {}\nNetwork: ({},) \nLearning rate: {} \nBuffer size: {}"
                 .format(C, BATCH_SIZE, EPSILON_INIT, EPSILON_FINAL, EPSILON_DECAY, GAMMA, self.hidden_dim, LEARNING_RATE, RANDOM_SAMPLE_SIZE), bbox=dict(facecolor='white', alpha=0.5))
        full_path = path + str(np.round(mean_end_reward, 2)) + '-' + date + str(int(np.round(time/epochs)))
        plt.savefig(full_path + '.png',bbox_inches='tight')
        torch.save(self.network.state_dict(), full_path)
        #plt.show()

    def test_policy(self, render=False, random=False):
        self.network.eval()
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
        if random:
            random_rewards = util.random_sampling(env)

        plt.plot([i for i in range(len(rewards))], rewards, label='trained')
        plt.plot([i for i in range(len(random_rewards))], random_rewards, label='random')
        plt.title("Training: {}".format(np.round(np.mean(rewards), 4)))
        plt.legend()
        # print(rewards)
        plt.show()

def run(epochs, load=False, path='0'):

    dqn = DQN(env.observation_space.shape[0], HIDDEN_NODES, RANDOM_SAMPLE_SIZE, LEARNING_RATE)
    if not load:
        dqn.initialize_replay_buffer()
        start = time.time()  # cpu(15): 143 bs(4), gpu: 896
        dqn.train(epochs)
    else:
        dqn.load_model(path)
    return dqn



