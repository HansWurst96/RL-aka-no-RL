import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import rl_util as ut
import Networks as networks
from torch.autograd import Variable
import time
import datetime

import gym
import quanser_robots

import torch
import torch.nn as nn
import torch.nn.functional as F

#env_name = 'CartpoleSwingShort-v0'
env_name = 'BallBalancerSim-v0'
#env_name = 'CartpoleStabShort-v0'
#env_name = 'MountainCarContinuous-v0'
#env_name = 'Pendulum-v0'

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
##################################
C = 1
TAU = 0.0001

MULTIPLIER = 1
EPISODE_MULTIPLIER = 10
INTERVAL = 1

EPS_INIT = 1
EPS_FINAL = 1
EPS_DECAY = 0.5
EPOCHS = 700
##################################
#Some arrays for later plotting
critic_losses = []
actor_losses = []
results = []
training_rewards = []
regression_errors = []
episode_lengths = []
x_axes = []

start_time = time.time()
actions = []

# Actor Network according to specifications listed in the DDPG-Paper


    #Initialize Actor and Critic two times, one for the target.
# Initialize random process for Action selection and Replay Buffer
class DDPG(object):
    def __init__(self, env, hidden_dim, buffer_capacity, gamma, lr, batch_size, epochs, epsilon_decay,
                 simple=True, parameter_noise=False, action_noise=True):
        self.env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.buffer_capacity = buffer_capacity
        self.sample_size = buffer_capacity
        self.batch_size = batch_size
        self.epochs = epochs

        self.buffer = ut.ReplayMemory(self.buffer_capacity, state_dim, action_dim)

        self.gamma = gamma
        self.learning_rate = lr
        self.hidden_dim = hidden_dim
        self.random = parameter_noise
        self.action_noise = action_noise

        self.OUP = ut.OrnsteinUhlenbeckProcess(action_dim)
        self.max_steps = self.initialize_replay_buffer()
        if simple:
            self.actor = networks.SimpleActor(state_dim, action_dim, hidden_dim, MULTIPLIER * self.env.action_space.high[0], epsilon_decay, epochs, self.max_steps, random=parameter_noise).to(device)
            self.target_actor = networks.SimpleActor(state_dim, action_dim, hidden_dim, MULTIPLIER * self.env.action_space.high[0], epsilon_decay, epochs, self.max_steps, random=parameter_noise).to(device)
            self.critic = networks.SimpleCritic(state_dim, action_dim, hidden_dim).to(device)
            self.target_critic = networks.SimpleCritic(state_dim, action_dim, hidden_dim).to(device)
        else:
            self.actor = networks.Actor(state_dim, action_dim, MULTIPLIER * self.env.action_space.high[0], hidden_dim, hidden_dim, epsilon_decay, epochs, self.max_steps, random=parameter_noise).to(device)
            self.target_actor = networks.Actor(state_dim, action_dim, MULTIPLIER * self.env.action_space.high[0], hidden_dim, hidden_dim, epsilon_decay, epochs, self.max_steps, random=parameter_noise).to(device)
            self.critic = networks.Critic(state_dim, action_dim, hidden_dim, hidden_dim).to(device)
            self.target_critic = networks.Critic(state_dim, action_dim, hidden_dim, hidden_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2, lr=10 * self.learning_rate)

    def initialize_replay_buffer(self):
        print("Initializing Replay Buffer...")
        if env_name == 'BallBalancerSim-v0':
            state = self.env.reset()
        else:
            state = self.env.reset()
        #print(state)
        count = 0
        steps = []
        for i in range(self.sample_size):
            action = self.env.action_space.sample()
            #action = np.random.randint(-1, 2, size=1)
            #action = np.array(np.random.normal(scale=24))
            next_state, reward, done, info = self.env.step(action)
            #print(" n", next_state)
            self.buffer.add_to_memory((state, action, next_state, reward, done))
            if done:
                if env_name == 'BallBalancerSim-v0':
                    state = self.env.reset()
                else:
                    state = self.env.reset()
                steps.append(count)
                count = 0
            else:
                count += 1
                state = next_state
        max_steps = np.array(steps).mean()
        print("Initializing Complete")
        return max_steps


    def train(self, tau=0.001, cont=True, render=False, random=False, intervals=1):
        start_time = time.time()
        #plt.ion()
        i = 0
        update_count, target_count = 0, 0
        # I have yet to try decreasing the discount factor successively
        eps = EPS_INIT
        plt.show()

        axes = plt.gca()
        axes.set_xlim(0, 100)
        axes.set_ylim(-50, +50)
        line, = axes.plot([], [], 'r-')

        for episode in range(self.epochs):
            i += 1
            print("Episode: ", i, '----------------------------------------------------------')
            if env_name == 'BallBalancerSim-v0':
                state = self.env.reset()
            else:
                state = self.env.reset()
            state = torch.FloatTensor(state).to(device)
            done = False
            training_episode_reward = 0
            regression_error = []
            steps = 0
            max_steps = EPISODE_MULTIPLIER * np.mean(np.array(self.max_steps))
            while not done and steps <= (max_steps):
                if self.action_noise:
                    decay = (EPS_FINAL / EPS_INIT) ** (1.0 / (EPS_DECAY * EPOCHS * self.max_steps))
                    output = (self.actor(state).detach().data.cpu().numpy())
                    action = output + eps * 0.1 * self.add_noise()
                    eps *= decay
                else:
                    action = (self.actor(state).detach().data.cpu().numpy())
                #actions.append(action)
                if steps % 1000 == 0:
                    print("Chosen action: ", action)
                    #print("Steps taken: ", np.round(100 * steps/(EPISODE_MULTIPLIER * self.max_steps), 2), "%")
                next_state, reward, done, info = self.env.step(action)
                next_state = Variable(torch.FloatTensor(next_state)).to(device)
                steps += 1
                if (render and (i%intervals == 0)):
                    self.env.render()
                state = next_state
                training_episode_reward+=reward
                ### Reg_error
                self.buffer.add_to_memory((state, action, next_state, reward, done))
                state_array, action_array, next_state_array, reward_array, done_array = self.buffer.sample(self.batch_size)
                state_array = Variable(torch.FloatTensor(state_array)).to(device)
                action_array = Variable(torch.FloatTensor(action_array)).to(device)
                reward_array = Variable(torch.FloatTensor(reward_array)).to(device)
                next_state_array = Variable(torch.FloatTensor(next_state_array)).to(device)
                done_array = Variable(torch.FloatTensor(done_array.astype(int))).to(device)

                A_Critic = self.target_actor(next_state_array)
                Q_Spr_A = self.target_critic(next_state_array, A_Critic).detach()

                target = reward_array + self.gamma * Q_Spr_A * (1-done_array) #(1 - done-array)
                y = self.critic(state_array, action_array)

                #critic_loss = F.mse_loss(self.critic(state_array, action_array), target)
                #critic_loss = torch.mean(torch.pow(y - target, 2))
                critic_loss = F.mse_loss(y, target)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Compute actor loss
                # this is equivalent to the update rule in the paper
                actor_loss = -1 * torch.mean(self.critic(state_array, self.actor(state_array)))

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if steps % C == 0:
                    self.soft_update()

            actor_losses.append(actor_loss)
            critic_losses.append(regression_error)
            training_rewards.append(training_episode_reward / steps)
            x_axes.append(i)
            ##############################################
            plt.plot(x_axes, training_rewards)
            plt.draw()
            plt.pause(0.1)
            plt.clf()
            ##############################################

            episode_lengths.append(steps)
            self.OUP.reset()
    def update_plot(self, fig, line):
        line.set_xdata(x_axes)
        line.set_ydata(training_rewards)
        plt.draw()
        #fig.canvas.flush_events()

    def add_noise(self):
        noise = self.env.action_space.high * np.random.normal()
        #noise = np.random.choice(np.linspace(-self.env.action_space.high[0], self.env.action_space.high[0], 2 * self.env.action_space.high[0]))
        return noise

    def soft_update(self):
        for target, src in zip(self.target_actor.parameters(), self.actor.parameters()):
            target.data.copy_(target.data * (1.0 - TAU) + src.data * TAU)

        for target, src in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.data.copy_(target.data * (1.0 - TAU) + src.data * TAU)

    def hard_update(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def visualize(self, time, epochs):
        max_reward = np.max(training_rewards)
        mean_end_reward = np.mean(training_rewards[int(len(training_rewards)*0.85):])
        #date = datetime.datetime.now().strftime("%m-%d %H%M")
        date = datetime.datetime.now()
        hour, minute = str(date.hour), str(date.minute)
        path = 'C:\\Users\\Jonas\\Desktop\\plots\\DDPG\\NoiseExperimenting'
        fig, (reward_plot) = plt.subplots(1, 1, sharex=True, figsize=(14, 10))
        #loss_plot.plot([x for x in range(len(actor_losses))], actor_losses, label='Loss')
        reward_plot.set_title(ut.shorten_name(env_name) + "\n" + "Time: " + str(int(time)) + "s Reward: (Peak : " + str(np.round(max_reward, 3)) + " Last 15 %: " + str(np.round(mean_end_reward, 3)) + ")")
        # loss_plot.ylabel("Loss")
        reward_plot.plot([x for x in range(len(training_rewards))], training_rewards, label='Reward', linewidth=0.75, alpha=0.8)

        arr = np.array(np.copy(training_rewards))
        arr = np.nanmean(
        np.pad(arr.astype(float), (0, ((10 - arr.size % 10) % 10)), mode='constant', constant_values=np.NaN).reshape(-1, 10), axis=1)
        arr = np.insert(arr, 0, training_rewards[0])
        reward_plot.plot([10 * (i) for i in range(len(arr))], arr, label='mean training reward', linewidth=1.5)
        plt.text(0.1 * len(training_rewards), (0.5 * (np.max(training_rewards))),
                 "Tau: {} \nBatch size: {} \nGamma: {}\nNetwork: ({},) \nLearning rate: {} \nBuffer size: {} \nMultiplier: {}"
                 .format(TAU, self.batch_size, self.gamma, self.hidden_dim,
                         self.learning_rate, self.buffer_capacity, MULTIPLIER), bbox=dict(facecolor='white', alpha=0.5))
        plt.savefig(path + ut.shorten_name(env_name) + "(" + hour + "h " + minute + "min)" +"1_parameter_noise"+ str(self.random) + str(np.round(mean_end_reward, 3)) + '.png', bbox_inches='tight')
        #plt.show()
        
    def evaluate(self, cont=True):
        for i in range(100):
            state = torch.FloatTensor(self.env.reset()).to(device)
            done = False
            episode_reward, steps = 0, 0
            while not done and steps <= (3 * np.mean(np.array(self.max_steps))):
                if cont:
                    action = self.actor(state).detach().data.cpu().numpy()
                next_state, reward, done, info = self.env.step(action)
                steps += 1
                #env.render()
                next_state = torch.FloatTensor(next_state).to(device)
                episode_reward += reward
                state = next_state
            results.append(episode_reward)
        plt.show()

    def save_model(self, a_path, c_path):
        torch.save(self.actor.state_dict(), a_path)
        torch.save(self.critic.state_dict(), c_path)

    def load_model(self, a_path, c_path):
        self.actor.load_state_dict(torch.load(a_path))
        self.critic.load_state_dict(torch.load(c_path))

    def simulate(self):
        for i in range(100):
            done = False
            state = self.env.reset()
            while not done:
                state = Variable(torch.FloatTensor(state)).to(device)
                action = self.actor(state).data.cpu().numpy()
                state, reward, done, info = self.env.step(action)
                self.env.render()




a_path = "C:\\Users\\Jonas\\Documents\\Uni\\5.Semester\\Reinforcement Learning\\Code\\models\\testing\\BallBalancer_actor_1"
c_path = "C:\\Users\\Jonas\\Documents\\Uni\\5.Semester\\Reinforcement Learning\\Code\\models\\testing\\BallBalancer_critic_1"

def qube_task(epochs, first_hidden, capacity, gamma, batch_size, lr, eps_decay, simple,
              train=True, save=False, simulate=False, evaluate=False, visualize=True, render=False, random=False, i=1, training=True, action_noise=True):
    #a_path = "C:\\Users\\Jonas\\Documents\\Uni\\5.Semester\\Reinforcement Learning\\Code\\models\\CPShort_actor_4"
    #c_path = "C:\\Users\\Jonas\\Documents\\Uni\\5.Semester\\Reinforcement Learning\\Code\\models\\CPShort_critic_4"

    t = time.time()
    environment = gym.make(env_name)
    print("Starting: ", env_name)

    ddpg = DDPG(environment,
                first_hidden, capacity, gamma, lr, batch_size, epochs, eps_decay, simple=simple, parameter_noise=training, action_noise=action_noise)
    if train:
        ddpg.train(render=render, random=random, intervals=i)
        t = np.round(time.time() - t)
    else:
        ddpg.load_model(a_path, c_path)
    if save:
        ddpg.save_model(a_path, c_path)
    if visualize:
        ddpg.visualize(t, epochs)
    if evaluate:
        rewards = []
        for i in range(100):
            state = environment.reset()
            done = False
            episode_reward = 0
            while not done:
                action = environment.action_space.sample()
                next_state, reward, done, info = environment.step(action)
                episode_reward += reward
            rewards.append(episode_reward)
        ddpg.evaluate()
    if simulate:
        ddpg.simulate()
def test_function(actor_model, critic_model, first_hidden, simple=True):
    environment = gym.make(env_name)
    ddpg = DDPG(environment, first_hidden, 10000, 0.99, 1e-4, 10, 10, 10, simple=simple, parameter_noise=False, action_noise=False)
    ddpg.load_model(actor_model, critic_model)
    for i in range(100):
        state = torch.FloatTensor(environment.reset()).to(device)
        done = False
        episode_reward, steps = 0, 0
        while not done:
            action = ddpg.actor(state).detach().data.cpu().numpy()
            if steps % 100 == 0:
                print(action)
            next_state, reward, done, info = environment.step(action)
            environment.render()
            steps += 1
            next_state = torch.FloatTensor(next_state).to(device)
            state = next_state
        environment.close()



#print('save: True')
qube_task(epochs=EPOCHS, first_hidden=100, capacity=50000, batch_size=500, gamma=0.95, lr =1e-5, eps_decay=1.0, simple=False,
          train=True, save=True, evaluate=False, visualize=True, render=False, simulate=False, random=True, i=1, training=False, action_noise=True)

#test_function(a_path, c_path, 50, simple=True)



