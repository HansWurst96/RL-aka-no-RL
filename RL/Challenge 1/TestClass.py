import numpy as np
from scipy import spatial
import gym
import quanser_robots
import RegressionProblems as reg

########################################################################################################################
# Global Constants for Discretization
NUM_ACTIONS = 3
NUM_STATES_1 = 5  # Theta
NUM_STATES_2 = 5  # Theta dot
########################################################################################################################

class BaseClass():
    def __init__(self, environment, tolerance, discount_factor):
        self.env = gym.make(environment)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.shape[0]
        self.tolerance = tolerance
        self.gamma = discount_factor
        #self.state_model = reg.fit_state_model(environment)
        #self.reward_model = reg.fit_reward_model(environment)
        self.discrete_states = self.discretize_state_space()
        self.discrete_actions = self.discretize_action_space()
        self.value_table = np.zeros(self.discrete_states.shape[0])
        self.policy_table = np.zeros(self.discrete_states.shape[0])

    def discretize_state_space(self):
        s = 0
        discrete_states = []
        for i in np.linspace(-self.env.observation_space.high[0], self.env.observation_space.high[0], NUM_STATES_1):
            for j in np.linspace(-self.env.observation_space.high[1], self.env.observation_space.high[1], NUM_STATES_2):
                state = [i, j]
                if s == 0:
                    discrete_states = state
                    s = 1
                else:
                    discrete_states = np.vstack((discrete_states, state))
        return discrete_states

    def discretize_action_space(self):
        high = self.env.action_space.high
        low = -high
        return np.linspace(low, high, NUM_ACTIONS)

    def predict_reward(self, state, action):
        clf = self.reward_model
        #print("action:", (np.asarray(action).reshape(1, -1)))
        #print((self.env.action_space.sample().shape))
        #x, r, y, z = self.env.step(np.asarray(action).reshape(1, -1))
        query_point = np.append(action, state).reshape(1, -1)
        prediction = clf.predict(query_point)
        #print("Prediction: ", prediction)
        #print("True: ", r)
        return prediction

    def predict_next_state(self, state, action, index):
        # Mittelmäßige Outputs
        clf = self.state_model
        states = self.discrete_states
        query_point = np.append(action, state).reshape(1, -1)
        prediction = clf.predict(query_point)
        idx, nearest_state = self.find_nearest_state(states, prediction)
        if index:
            return idx
        else:
            return nearest_state

    def find_nearest_state_(self, state):
        states = self.discrete_states
        idx = states.shape[0] + 1
        best_theta = -np.inf
        for i in range(states.shape[0]):
            if np.abs(states[i][0] - state[0][0]) < np.abs(best_theta - state[0][0]):
                best_theta = states[i][0]
        best_theta_dt = -np.inf
        idx = -1
        for i in range(states.shape[0]):
            if states[i][0] == best_theta:
                if np.abs(states[i][1] - state[0][1]) < np.abs(best_theta_dt - state[0][1]):
                    idx = i
                    best_theta_dt = states[i][1]
        return idx, states[idx]

    def find_nearest_state(self, state):
        tree = spatial.KDTree(self.discrete_states)
        idx = tree.query(state)[1]
        return idx, self.discrete_states[idx]

    def find_next_action(self, action):
        array = np.asarray(self.discrete_actions)
        idx = (np.abs(array - action)).argmin()
        return array[idx]

    def test_policy(self):
        for i in range(100):
            state = self.env.reset()
            done = False
            while not done:
                state = np.asarray([state])
                idx, state = self.find_nearest_state(state)
                action = np.asarray(self.policy_table[idx]).reshape(1, -1)
                self.env.render()

                #print(self.predict_next_state(state, action, False))
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                #print(state)

class ValueIteration(BaseClass):
    def bellman(self, state):
        value_array = []
        for a in self.discrete_actions:
            next_state, reward, done, info = self.env.step(np.asarray(a).reshape(1, -1))
            idx, next_state = self.find_nearest_state(np.asarray(next_state).reshape(2,))
            value_array.append(reward + self.gamma * self.value_table[idx])
        return np.max(np.array(value_array))

    def iterate(self):
        self.env.reset()
        delta = 1
        policy = []
        while delta > self.tolerance:
            delta = 0
            for s in range(self.discrete_states.shape[0]):
                temp = self.value_table[s]
                value = self.bellman(self.discrete_states[s, :])
                self.value_table[s] = value
                diff = np.abs(temp - value)
                delta = np.max([delta, diff])
                print(delta)
        print("Convergence reached")
        for state in self.discrete_states:
            value_array = []
            for action in self.discrete_actions:
                next_state, reward, done, info = self.env.step(np.asarray(action).reshape(1, -1))
                idx, next_state = self.find_nearest_state(np.asarray(next_state).reshape(2,))
                value_array.append(reward + self.gamma * self.value_table[idx])
            policy.append(np.argmax(np.asarray(value_array)))
        self.policy_table = policy

def main(env, tol, discount):
    VI = ValueIteration(env, tol, discount)
    q = VI.discretize_state_space()[2] + [0.02, 0.1]
    #print(q.shape)
    #print(VI.find_nearest_state(q))
    VI.iterate()
    VI.test_policy()

main('Pendulum-v2', 0.01, 0.95)






