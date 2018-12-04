import numpy as np
import gym
import quanser_robots

import sklearn

import RegressionProblems as rp

# TODO: Disabling train_test_split?

MAX_ITERATIONS = 5
DISCRETIZATION = 6

class PolicyIteration(object):
    def __init__(self, environment, tolerance, discount_factor):
        # environment specific properties
        self.env_name = environment
        self.env = gym.make(environment)
        self.env.seed(0)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.shape[0]

        # constants
        self.tolerance = tolerance
        self.gamma = discount_factor

        # system- and dynamics model
        self.state_model = rp.fit_state_model(environment)
        self.reward_model = rp.fit_reward_model(environment)

        self.state = self.env.reset()
        self.discretized_states = self.discretize_state_space(DISCRETIZATION)
        self.discretized_actions = self.discretize_action_space(DISCRETIZATION)
        self.value_table = np.zeros(self.discretized_states.shape[1])
        self.policy_table = np.zeros(self.discretize_state_space([1]))

    def info(self, verbose=False):
        if verbose:
            print("Current environment: {} (state space: {}, action space: {})"
                  .format(self.env_name, self.state_space, self.action_space))
        return self.state_space, self.action_space

    def approximate_next_state(self, state, action, tabular=False):
        #state = self.state
        input = np.append(action, state)
        next_state = self.state_model.predict(input.reshape(1, -1))
        next_state, idx = self.find_nearest_state(next_state)
        if not tabular:
            return next_state
        else:
            return idx

    def sample_input(self):
        action = self.env.action_space.sample()
        return np.array(self.state), np.array(action)

    def find_nearest_state(self, state):
        # TODO: Do it for every entry separately ?
        array = self.discretized_states
        difference = np.inf
        idx = array.shape[1] + 1

        for i in range(array.shape[1]):
            dif = np.linalg.norm(array[:,i] - state)
            if dif < difference:
                difference = dif
                idx = i
        return array[:,idx], idx

    def discretize_state_space(self, num):
        high = self.env.observation_space.high
        low = -high
        discrete_states = []
        for i in range(self.state_space):
            interval = np.linspace(low[i], high[i], num)
            if(i==0):
                discrete_states = interval
            else:
                discrete_states = np.vstack((discrete_states,interval))
        return discrete_states

    def discretize_action_space(self, num):
        high = self.env.action_space.high
        low = -high
        discrete_actions = []
        for i in range(self.action_space):
            interval = np.linspace(low[i], high[i], num)
            if(i==0):
                discrete_actions = interval
            else:
                discrete_actions = np.vstack((discrete_actions,interval))
        return discrete_actions

    def bellman_equation(self, state, iteration, actions):
        if iteration >= MAX_ITERATIONS:
            predicted_rewards = [self.reward_model.predict(np.append(a, state).reshape(1, -1)) for a in actions]
            value = np.max(predicted_rewards)
            return value
        predicted_rewards = np.array([self.reward_model.predict(np.append(a, state).reshape(1, -1)) for a in actions])
        v = [self.bellman_equation(self.approximate_next_state(state, a), iteration + 1, actions) for a in actions]
        predicted_values = np.array(v)
        value = np.max(predicted_rewards + self.gamma * predicted_values)
        return value

    def bellman_tabular(self, state, actions):
        table = self.value_table
        action = self.policy_table[state]
        value = self.reward_model.predict(np.append(action, state).reshape(1, -1)) + self.gamma * table[self.approximate_next_state(state, action, tabular=True)]
        return value

    def iterate(self):
        i = 0
        delta = 1
        states = self.discretized_states
        print(states.shape)
        actions = self.discretized_actions
        while delta > self.tolerance:
            delta = 0
            for state in range(states.shape[1]):
                v = self.value_table[state]
                value = self.bellman_tabular(states[:,state], actions)
                self.value_table[state] = value
                abs = np.abs(v - value)
                delta = np.max([delta, abs])

        # Get policy

    def PI(self):
        self.iterate()

        states = self.discretized_states
        policy_stable = True
        for state in range(states.shape[1]):
            old_action = self.policy_table[state]
            self.policy_table[state] = np.max([self.reward_model.predict(np.append(a, state).reshape(1, -1))
                                               + self.gamma * self.value_table[self.approximate_next_state(state, a, tabular=True)] for a in self.discretized_actions])
            if old_action is not self.policy_table[state]:
                policy_stable = False
                break

        if policy_stable is not True:
            self.PI()
        else:
            return self.policy_table, self.value_table

def main(environment, tolerance):
    VI = PolicyIteration(environment, tolerance, 0.9)
    VI.info(verbose=True)
    s, a = VI.sample_input()
    print(VI.discretize_state_space(4))
    actions = VI.discretize_action_space(4)
    #print(VI.bellman_equation(VI.state, 0, actions))
    print(VI.iterate())

    #print(VI.approximate_next_state(s,a))

main('Pendulum-v0', 0.001)

print(gym.make('Pendulum-v0').observation_space.high)