import numpy as np

import gym
import quanser_robots

class ReplayMemory:
    def __init__(self, max_capacity, state_dim, n_actions):
        self.capacity = max_capacity
        self.index = None
        self.index_list = list(range(max_capacity))

        self.S       = np.zeros((max_capacity, state_dim))
        self.S_prime = np.zeros((max_capacity, state_dim))
        self.actions = np.zeros(max_capacity)
        self.rewards = np.zeros((max_capacity, 1))
        self.is_done = np.full((max_capacity, 1), True, dtype=bool)

    def add_to_memory(self, experience):
        '''
        Puts experience into memory buffer
        args:
            :experience: a tuple consisting of (S, A, S_prime, R, is_done)
        '''
        if self.index == None or self.index == self.capacity - 1:
            self.index = 0
        else:
            self.index += 1

        S, A, S_prime, R, is_done = experience
        self.S[self.index]       = S
        self.S_prime[self.index] = S_prime
        self.actions[self.index] = A
        self.rewards[self.index] = R
        self.is_done[self.index] = is_done

    def sample(self, batch_size):
        '''
        Randomly sample from buffer.
        args:
            :batch_size: number of experiences to sample in a minibatch
        '''
        indices = np.random.choice(self.index_list, batch_size)
        S       = self.S[indices]
        A       = self.actions[indices]
        S_prime = self.S_prime[indices]
        R       = self.rewards[indices]
        is_done = self.is_done[indices]

        return (S, A, S_prime, R, is_done)

def random_sampling(env):
    rewards = []
    for i in range(100):
        state = env.reset()
        done = False
        total_reward = 0
        time_steps = 0
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            time_steps += 1
            state = next_state
        rewards.append(total_reward/time_steps)
    return rewards


def shorten_name(environment_name):

    shortener = {
        'CartpoleSwingShort-v0': 'SwingShort',
        'CartpoleStabShort-v0': 'StabShort',
        'CartpoleSwingLong-v0': 'SwingLong',
        'CartpoleStabLong-v0': 'StabLong',
        'Pendulum-v0': 'Pendulum'
    }
    return shortener.get(environment_name, environment_name)

