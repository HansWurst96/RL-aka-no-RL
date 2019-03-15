import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cpu")

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, max_action, first_hidden, second_hidden, decay, epochs, average_length, random=True):
        """
        Two layered neural network to model the actor of the DDPG algorithm. It furthermore instantiates the gaussian
                noise layer, if needed
        :param input_dim: int, Dimension of the environments state space
        :param output_dim: int, Dimension of the environments action space
        :param max_action: float, Maximum action the actor is allowed to take
        :param first_hidden: int, Number of neurons in the first hidden layer
        :param second_hidden: int, Number of neurons in the second hidden layer
        :param decay: float, Value after how many episodes eps_final will be reached (between 0 and 1)
        :param epochs: int, Number of episodes the DDPG algorithm will train
        :param average_length: float, Average number of steps an episode lasted
        :param random: bool, optional, Binary value whether the gaussian noise layer has to be set
        """
        super(Actor, self).__init__()
        self.random = random
        self.max_action = float(max_action)
        self.layer1 = nn.Linear(input_dim, first_hidden)
        nn.init.xavier_uniform_(self.layer1.weight)
        self.layer2 = nn.Linear(first_hidden, second_hidden)
        nn.init.xavier_uniform_(self.layer2.weight)
        self.layer3 = nn.Linear(second_hidden, output_dim)
        nn.init.uniform_(self.layer3.weight, -0.003, 0.003)
        if self.random:  # creating noise layer
            self.noise = GaussianNoise(self.max_action/5.0, 1, decay, 0.01, epochs, average_length)

    def forward(self, x):
        """
        Forward pass method for the actor
        :param x: State
        :return: Action
        """
        if self.random:
            x = self.noise(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        x = self.max_action * x
        return x

class SimpleActor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, max_action, decay, epochs, average_length, random=False):
        """
        Single layered neural network to model the actor of the DDPG algorithm. It furthermore instantiates the gaussian
                noise layer, if needed
        :param input_dim:  int, Dimension of the environments state space
        :param output_dim: int, Dimension of the environments action space
        :param hidden_dim: int, Number of neurons in the hidden layer
        :param max_action: float, Maximum action the actor is allowed to take
        :param decay: float, Value after how many episodes eps_final will be reached (between 0 and 1)
        :param epochs: int, Number of episodes the DDPG algorithm will train
        :param average_length: float, Average number of steps an episode lasted
        :param random: bool, optional, Binary value whether the gaussian noise layer has to be set
        """
        super(SimpleActor, self).__init__()
        self.max_action = float(max_action)
        self.random = random
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        nn.init.uniform_(self.layer2.weight, -0.003, 0.003)
        if random:
            self.noise = GaussianNoise(self.max_action/5.0, 1, decay, 0.01, epochs, average_length)

    def forward(self, x):
        """
        Forward pass method for the actor
        :param x: State
        :return: Action
        """
        if self.random:
            x = self.noise(x)
        x = F.relu(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.max_action * x
        return x


# Critic Network according to specifications listed in the DDPG-Paper
class Critic(nn.Module):
    def __init__(self, input_dim, action_dim, first_hidden, second_hidden):
        """
        Two layered neural network to learn the Q-function of the DDPG algorithm.
        :param input_dim: int, Dimension of the environments state space
        :param action_dim: int, Dimension of the environments action space
        :param first_hidden: int, Number of neurons in the first hidden layer
        :param second_hidden: int, Number of neurons in the second hidden layer
        """
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(input_dim, first_hidden)
        nn.init.xavier_uniform_(self.layer1.weight)
        self.layer2 = nn.Linear(first_hidden + action_dim, second_hidden)
        nn.init.xavier_uniform_(self.layer2.weight)
        self.layer3 = nn.Linear(second_hidden, 1)
        nn.init.uniform_(self.layer3.weight, -0.003, 0.003)

    def forward(self, state, action):
        """
        Forward pass for the critic
        :param state: State
        :param action: Action
        :return: Scalar Q value
        """
        s = F.relu(self.layer1(state))
        x = torch.cat((s, action), dim=1)
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class SimpleCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        """
        Single layered Neural network to model the critic of the DDPG algorithm. It learns the Q-function.
        :param input_dim: int, Dimension of the environments state space
        :param action_dim: int, Dimension of the environments action space
        :param hidden_dim: int, Number of neurons in the hidden layer
        """
        super(SimpleCritic, self).__init__()
        self.layer1 = nn.Linear(input_dim+action_dim, hidden_dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        self.layer2 = nn.Linear(hidden_dim, 1)
        nn.init.uniform_(self.layer2.weight, -0.003, 0.003)

    def forward(self, state, action):
        """
        Forward pass for the critic
        :param state: State
        :param action: Action
        :return: Scalar Q value
        """
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class GaussianNoise(nn.Module):
    def __init__(self, sigma, eps_init, eps_decay, eps_final, epochs, average_length):
        """
        Neural network for injecting noise into another networks parameter space. It is used in DDPG to aid exploration.
        :param sigma: standard deviation of the gaussian distribution used for noise generation
        :param eps_init: Initial value of coefficient(epsilon) multiplied by generated noise to implement decaying noise
        :param eps_decay: Value after how many episodes eps_final will be reached (between 0 and 1)
        :param eps_final: Final value of epsilon (It will get further reduced, because we never needed constant noise)
        :param epochs: Number of episodes the DDPG algorithm will train
        :param average_length: Average number of steps an episode lasted
        """
        super().__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0).to(device)
        self.eps = eps_init
        self.decay = (eps_final / eps_init) ** (1.0 / (eps_decay * epochs * average_length))
        self.i = 0

    def forward(self, x):
        """
        Calculates gaussian noise
        :param x: Input parameter
        :return: Noise added to input
        """
        if self.sigma != 0:
            self.eps *= self.decay
            sampled_noise = self.eps * Variable(x.data.new(x.size()).normal_(0, self.sigma))
            x = x + sampled_noise
            self.i += 1
        return x
