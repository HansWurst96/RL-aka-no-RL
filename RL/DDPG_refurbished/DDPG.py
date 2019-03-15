import numpy as np
import matplotlib.pyplot as plt
import util as ut
import Networks as networks

import gym
import quanser_robots

import torch
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cpu")


class DDPG(object):
    def __init__(self,
                 env_name,
                 hidden_dim,
                 buffer_capacity,
                 gamma,
                 lr,
                 batch_size,
                 tau,
                 epochs,
                 epsilon_decay,
                 multiplier=1,
                 simple=True,
                 parameter_noise=False,
                 action_noise=True,
                 gaussian=True,
                 oup=False):
        """
        The main class of the DDPG algorithm, as proposed in

        Lillicrap T. P., Hunt J. J., Pritzel A., Heess N., Erez T., Tassa Y., Silver D., Wierstra D.,
        Continuous Control with Deep Reinforcement Learning, 2015.

        The algorithm consists of two networks for each
        actor and critic. The main difference between our implementation and the one proposed by Lillicrap et al. is
        certainly the possibility to inject parameter noise, as proposed in

        Plappert M., Houthooft R., Dhariwal P., Sidor S., Chen R. Y., Chen X., Asfour T., Abbeel P., Andrychowicz M.,
        Parameter Space Noise for Exploration, 2017

        and the use of a vastly smaller network. It furthermore supports different methods for generating random noise
        to add to the actor's output.
        :param env_name: String, String passed to gym.make() to create the environment
        :param hidden_dim: int, Number of neurons in the hidden layers. Note: This number is used for both actor and
                critic network, as well as for both hidden layers, if the three layered network architecture is chosen.
        :param buffer_capacity: int, Size the replay buffer will be initialized with
        :param gamma: float, Discount factor used in calculating the losses
        :param lr: Learning rate for the actor network. Note: The learning rate for the critic is chosen ten times as
                high, as this is the ratio suggested by Lillicrap et al.
        :param batch_size: int, Number of transitions stored in the replay buffer, that will be sampled in every time
                step
        :param tau: Parameter for performing soft updates
        :param epochs: int, Number of episodes that will be taken during learning
        :param epsilon_decay: float, Value after how many episodes eps_final will be reached (between 0 and 1)
        :param multiplier: float, optional, Coefficient (between 0 and 1) the environments maximum action is multiplied
                by to clip the action space
        :param simple: bool, optional, Determines if a 2 layered network architecture will be used
        :param parameter_noise: bool, optional, Determines if noise is injected into the parameter space
        :param action_noise: bool, optional, Determines if noise is injected into the action space
        :param gaussian: bool, optional, Determines if gaussian noise is being used as action noise
        :param oup: bool, optional, Determines if an Ornstein-Uhlenbeck-Process is used to generate the noise for the
                action space
        """
        self.env_name = env_name
        self.env = gym.make(env_name)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = lr
        self.hidden_dim = hidden_dim

        self.multiplier = multiplier
        self.epochs = epochs

        self.gaussian = gaussian
        self.oup = oup
        self.random = parameter_noise
        self.action_noise = action_noise
        self.OUP = ut.OrnsteinUhlenbeckProcess(action_dim, sigma=self.env.action_space.high)

        self.buffer_capacity = buffer_capacity
        self.sample_size = buffer_capacity
        self.buffer = ut.ReplayMemory(self.buffer_capacity, state_dim, action_dim)

        self.max_steps = self.initialize_replay_buffer()

        self.training_rewards = []  # Array for storing training rewards. Needed for plotting and evaluating
        max_action = self.multiplier * self.env.action_space.high[0]

        if simple:
            self.actor = networks.SimpleActor(state_dim, action_dim, hidden_dim, max_action, epsilon_decay, epochs,
                                              self.max_steps, random=parameter_noise).to(device)
            self.target_actor = networks.SimpleActor(state_dim, action_dim, hidden_dim, max_action, epsilon_decay,
                                                     epochs, self.max_steps, random=parameter_noise).to(device)
            self.critic = networks.SimpleCritic(state_dim, action_dim, hidden_dim).to(device)
            self.target_critic = networks.SimpleCritic(state_dim, action_dim, hidden_dim).to(device)
        else:
            self.actor = networks.Actor(state_dim, action_dim, max_action, hidden_dim, hidden_dim, epsilon_decay,
                                        epochs, self.max_steps, random=parameter_noise).to(device)
            self.target_actor = networks.Actor(state_dim, action_dim, max_action, hidden_dim, hidden_dim, epsilon_decay,
                                               epochs, self.max_steps, random=parameter_noise).to(device)
            self.critic = networks.Critic(state_dim, action_dim, hidden_dim, hidden_dim).to(device)
            self.target_critic = networks.Critic(state_dim, action_dim, hidden_dim, hidden_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=10 * self.learning_rate,
                                                 weight_decay=1e-2)

        self.decay = (0.01 / 1) ** (1.0 / (epsilon_decay * epochs * self.max_steps))  # Coefficient to decrease noise
        # over time

    def initialize_replay_buffer(self):
        """
        Fills the replay buffer with transitions sampled from a uniform distribution. Furthermore it determines the
        average number of steps in an episode. This is needed to decrease the noise level in every step. Of course this
        could be avoided by only decreasing the noise level after every episode.
        :return: Average number of steps per episode
        """
        print("Initializing Replay Buffer...")
        state = self.env.reset()
        count = 0
        steps = []
        for i in range(self.sample_size):
            action = self.env.action_space.sample()
            next_state, reward, done, info = self.env.step(action)
            self.buffer.add_to_memory((state, action, next_state, reward, done))
            if done:
                state = self.env.reset()
                steps.append(count)
                count = 0
            else:
                count += 1
                state = next_state
        max_steps = np.array(steps).mean()
        print("Initializing Complete")
        return max_steps

    def train(self, render=False, liveplot=False):
        """
        Training loop that executes the training steps for as many episodes as specified in the initialization
        :param render: bool, optional, If the environment should be rendered
        :param liveplot: bool, optional, if the reward will be updated and drawn after every episode
        :return: None
        """
        i = 0
        eps = 1
        for episode in range(self.epochs):
            i += 1
            print("Episode: ", i, '----------------------------------------------------------')
            state = self.env.reset()
            state = torch.FloatTensor(state).to(device)
            done = False
            training_episode_reward = 0
            steps = 0
            while not done:
                if self.action_noise:
                    decay = self.decay
                    output = (self.actor(state).detach().data.cpu().numpy())
                    action = output + eps * 0.2 * self.calculate_noise(self.gaussian, self.oup)  # Add noise to output
                    eps *= decay
                else:
                    action = (self.actor(state).detach().data.cpu().numpy())
                next_state, reward, done, info = self.env.step(action)
                next_state = Variable(torch.FloatTensor(next_state)).to(device)
                steps += 1
                if render:
                    self.env.render()
                state = next_state
                training_episode_reward += reward
                self.buffer.add_to_memory((state, action, next_state, reward, done))  # Save transition in replay buffer

                states, actions, next_states, rewards, done_array = self.buffer.sample(self.batch_size)  # Sample batch
                state_array = Variable(torch.FloatTensor(states)).to(device)
                action_array = Variable(torch.FloatTensor(actions)).to(device)
                reward_array = Variable(torch.FloatTensor(rewards)).to(device)
                next_state_array = Variable(torch.FloatTensor(next_states)).to(device)
                done_array = Variable(torch.FloatTensor(done_array.astype(int))).to(device)

                A_Critic = self.target_actor(next_state_array)
                q_value = self.target_critic(next_state_array, A_Critic).detach()

                target = reward_array + self.gamma * q_value * (1 - done_array)
                y = self.critic(state_array, action_array)

                critic_loss = F.mse_loss(y, target)  # Compute critic loss

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

                self.soft_update()  # Update target networks

            self.training_rewards.append(training_episode_reward / steps)

            if liveplot:  # If live plotting is enabled, this part will update the graph in every episode
                plt.plot(self.training_rewards)
                plt.draw()
                plt.pause(0.1)
                plt.clf()

            self.OUP.reset()

    def calculate_noise(self, gaussian=True, oup=False):
        """
        Calculates noise that will be added to the actors output to aid exploration. How this noise is generated is
        determined by the methods parameters. It is possible to obtain noise form a gaussian distribution, an
        Ornstein-Uhlenbeck-Process or a uniform distribution.
        :param gaussian: bool, optional, Determines if noise should be uncorrelated gaussian
        :param oup: bool, optional, Determines whether an  Ornstein-Uhlenbeck-Process will be used for noise generation
        :return: scalar representing the calculated noise
        """
        if gaussian:
            noise = self.env.action_space.high * np.random.normal()
        if oup:
            noise = self.OUP.calculate_noise()
        if (not gaussian) and (not oup):
            noise = np.random.choice(
                np.linspace(-self.env.action_space.high, self.env.action_space.high, 2 * self.env.action_space.high))
        return noise

    def soft_update(self):
        """
        Soft updating the target networks parameters according to the paper referenced in the DDPG init method.
        :return: None
        """
        for target, src in zip(self.target_actor.parameters(), self.actor.parameters()):
            target.data.copy_(target.data * (1.0 - self.tau) + src.data * self.tau)

        for target, src in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.data.copy_(target.data * (1.0 - self.tau) + src.data * self.tau)

    def hard_update(self):
        """
        Hard updating the target networks (not advised to use)
        :return: None
        """
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def visualize(self, path):
        """
        Visualizes the learning curve and saves it as a .png
        :param path: Destination the image should be saved at
        :return: None
        """
        mean_end_reward = np.mean(self.training_rewards[int(len(self.training_rewards) * 0.85):])
        fig, (reward_plot) = plt.subplots(1, 1, figsize=(16, 9))
        reward_plot.set_title(ut.shorten_name(self.env_name))
        reward_plot.set_ylabel('Episode Reward per Timestep')
        reward_plot.set_xlabel('Episode')
        reward_plot.plot(self.training_rewards, label='training reward', linewidth=0.75, alpha=0.8)

        arr = np.array(np.copy(self.training_rewards))
        arr = np.nanmean(
            np.pad(arr.astype(float), (0, ((10 - arr.size % 10) % 10)), mode='constant',
                   constant_values=np.NaN).reshape(-1, 10), axis=1)
        arr = np.insert(arr, 0, self.training_rewards[0])
        reward_plot.plot([10 * i for i in range(len(arr))], arr, label='10 steps average', linewidth=1.5)
        plt.legend()
        plt.text(0, np.min(self.training_rewards),
                 "Tau: {} \nBatch size: {} \nGamma: {}\nNetwork: ({},) \nLearning rate: {} \n"
                 "Buffer size: {} \nMultiplier: {}"
                 .format(self.tau, self.batch_size, self.gamma, self.hidden_dim,
                         self.learning_rate, self.buffer_capacity, self.multiplier),
                 bbox=dict(facecolor='white', alpha=0.4))
        plt.savefig(
            path + "simple_0.7_oup_noise_pen0.75" + str(self.random) + str(np.round(mean_end_reward, 3)) + '.png',
            bbox_inches=None)

    def evaluate(self, episodes=100):
        """
        Evaluates the model on a number of episodes and returns the descriptive statistics
        :param episodes: int, optional, number of episodes the model will be evaluated on
        :return: float, float, mean and standard deviation
        """
        episode_rewards = []
        for i in range(episodes):
            state = torch.FloatTensor(self.env.reset()).to(device)
            done = False
            episode_reward, steps = 0, 0
            while not done:
                action = self.actor(state).detach().data.cpu().numpy()
                next_state, reward, done, info = self.env.step(action)
                steps += 1
                next_state = torch.FloatTensor(next_state).to(device)
                episode_reward += reward
                state = next_state
            episode_rewards.append(episode_reward)
        mean = np.mean(episode_rewards)
        standard_deviation = np.std(episode_rewards)
        return mean, standard_deviation

    def save_model(self, a_path, c_path):
        """
        Saves the current model using pytorch's save function.
        :param a_path: string,  Destination for storing the actor network parameters
        :param c_path: string, Destination for storing the critic network parameters
        :return: None
        """
        torch.save(self.actor.state_dict(), a_path)
        torch.save(self.critic.state_dict(), c_path)

    def load_model(self, a_path, c_path):
        """
        Loads a pretrained model using pytorch's load function. Note: The number of nodes in each layer need to be the
        same in the DDPG object as in that of the previously learned one.
        :param a_path: string, Path to the actor network parameters
        :param c_path: string, Path to the critic network parameters
        :return: None
        """
        self.actor.load_state_dict(torch.load(a_path))
        self.critic.load_state_dict(torch.load(c_path))

    def simulate(self, render):
        """
        Renders the learned model for 25 episodes as a method of visually evaluating the model.
        :param render: bool,  if the environment shall be rendered (should be False for the robots)
        :return: None
        """
        for i in range(25):
            done = False
            state = self.env.reset()
            while not done:
                state = Variable(torch.FloatTensor(state)).to(device)
                action = self.actor(state).data.cpu().numpy()
                state, reward, done, info = self.env.step(action)
                if render:
                    self.env.render()
                    
