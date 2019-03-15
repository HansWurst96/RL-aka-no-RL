import numpy as np

class ReplayMemory:
    def __init__(self, max_capacity, state_dim, n_actions):
        """
        Class for creating a replay memory to store transitions in and sample batches of transition from
        :param max_capacity: int, How many transitions can be stored in the buffer
        :param state_dim: int, State dimensions so the replay buffer can store them properly
        :param n_actions: int, Action dimensions for multidimensional action spaces
        """
        self.capacity = max_capacity
        self.index = None
        self.index_list = list(range(max_capacity))

        self.states = np.zeros((max_capacity, state_dim))
        self.next_states = np.zeros((max_capacity, state_dim))
        self.actions = np.random.normal(0, 5, (max_capacity, n_actions))
        self.rewards = np.zeros((max_capacity, 1))
        self.done = np.full((max_capacity, 1), True, dtype=bool)

    def add_to_memory(self, experience):
        """
        Store one transition in the replay buffer
        :param experience:
        :return: None
        """
        if self.index == None or self.index == self.capacity - 1:
            self.index = 0
        else:
            self.index += 1

        state, action, next_state, reward, done = experience
        self.states[self.index] = state
        self.next_states[self.index] = next_state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done

    def sample(self, batch_size):
        """
        Samples a minibatch of transitions according to a uniform distribution.
        :param batch_size: The size of a single minibatch
        :return: one minibatch of transitions
        """
        indices = np.random.randint(0, self.capacity, size=batch_size)
        states = self.states[indices]
        actions = self.actions[indices]
        next_states = self.next_states[indices]
        rewards = self.rewards[indices]
        done = self.done[indices]

        return states, actions, next_states, rewards, done

class OrnsteinUhlenbeckProcess():
    def __init__(self, n_actions, mu=0, theta=0.15, sigma=0.2):
        """
        Random process for generating random numbers that are correlated.
        :param n_actions: Dimension of the environments action space
        :param mu: float, optional,  mean of the underlying gaussian distribution
        :param theta: float, optional, Coefficient to determine correlation
        :param sigma: float, optional, standard deviation of the underlying gaussian distribution
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.n_action = n_actions
        self.X = np.ones(n_actions) * mu

    def reset(self):
        """
        Resets the correlation
        :return: None
        """
        self.X = np.ones(self.n_action) * self.mu

    def calculate_noise(self):
        """
        Calculates the noise depending on the current time step
        :return: Generated noise with shape of action dimension
        """
        dX = self.theta * (self.mu - self.X)
        dX += self.sigma * np.random.randn(self.n_action)
        self.X += dX
        return self.X

def shorten_name(environment_name):
    """
    Shortens an environment name using a dictionary.
    :param environment_name: String passed to gym to make an environment
    :return: string, Shortened name
    """

    shortener = {
        'CartpoleSwingShort-v0': 'SwingShort',
        'CartpoleStabShort-v0': 'StabShort',
        'CartpoleSwingLong-v0': 'SwingLong',
        'CartpoleStabLong-v0': 'StabLong',
        'Pendulum-v0': 'Pendulum',
        'MountainCarContinuous-v0': 'MCarCont'
    }
    return shortener.get(environment_name, environment_name)







