import numpy as np
from scipy import spatial
import gym
import pickle
from profilehooks import timecall
import matplotlib.pyplot as plt
import quanser_robots
import RegressionProblems as reg

########################################################################################################################
# Global Constants for Discretization
NUM_ACTIONS = 5
NUM_STATES_1 = 39  # Theta
NUM_STATES_2 = 20  # Theta dot
########################################################################################################################

class BaseClass():
    def __init__(self, environment, tolerance, discount_factor, load=True):
        self.env = gym.make(environment)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.shape[0]
        self.tolerance = tolerance
        self.gamma = discount_factor
        if not load:
            self.state_model = reg.fit_state_model(environment)
            pickle.dump(self.state_model, open("models\\Pendulum_state_model.sav", 'wb'))
            self.reward_model = reg.fit_reward_model(environment)
            pickle.dump(self.reward_model,  open("models\\Pendulum_reward_model.sav", 'wb'))
        else:
            self.state_model = pickle.load(open("models\\Pendulum_state_model.sav", 'rb'))
            self.reward_model = pickle.load(open("models\\Pendulum_reward_model.sav", 'rb'))
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
        return np.array(discrete_states)

    def discretize_action_space(self):
        high = self.env.action_space.high
        low = -high
        return np.linspace(low, high, NUM_ACTIONS)

    def predict_rewards(self):
        clf = self.reward_model
        states = self.discrete_states
        actions = self.policy_table
        input_data = np.vstack((actions.T, states.T)).T
        return clf.predict(input_data)

    def predict_next_states(self):
        clf = self.state_model
        states = self.discrete_states
        actions = self.policy_table
        input_data = np.vstack((actions.T, states.T)).T
        predictions = clf.predict(input_data)
        return [self.find_nearest_state(prediction) for prediction in predictions]


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
        return idx

    def find_nearest_state(self, state):
        tree = spatial.KDTree(self.discrete_states)
        idx = tree.query(state)[1]
        return idx

    def find_next_action(self, action):
        array = np.asarray(self.discrete_actions)
        idx = (np.abs(array - action)).argmin()
        return array[idx]

    def test_policy(self, render=False, verbose=True):
        #state = np.load("C:\\Users\\Jonas\\Documents\\Uni\\5.Semester\\VIPendulum.npy")
        reward_array = []
        for i in range(100):
            state = self.env.reset()
            done = False
            total_reward = 0
            time_steps = 0
            while not done:
                state = np.asarray([state])
                idx = self.find_nearest_state_(state)
                action = np.asarray(self.policy_table[idx]).reshape(1, -1)
                if render:
                    self.env.render()
                #print(self.predict_next_state(state, action, False))
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward
                time_steps +=1
                state = next_state
            reward_array.append(total_reward/time_steps)

        random_rewards = random_sampling(self.env)
        plt.plot([i for i in range(len(reward_array))], reward_array, label='trained')
        plt.plot([i for i in range(len(random_rewards))], random_rewards, label='random')
        plt.title("Training: {} Random: {}, state space:{} action_space{}"
                  .format(np.round(np.mean(reward_array), 4), np.round(np.mean(random_rewards), 4)
                          , self.discrete_states.shape, self.discrete_actions.shape[0]))
        plt.legend()
        #print(rewards)
        plt.show()

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

class PolicyIteration(BaseClass):
    def policy_evaluation(self):
        sweep = 0
        while True:
            if sweep % 5 == 0 or sweep in range(5):
                print("Sweep number: ", sweep, "(evaluation)")
            delta = 0
            temp = np.copy(self.value_table)
            self.value_table = self.predict_rewards() + self.gamma * self.value_table[self.predict_next_states()]
            delta = np.max(np.abs(temp -self.value_table))
            sweep += 1
            if delta < self.tolerance:
                print("Starting Policy Improvement")
                break

    def policy_improvement(self):
        while True:
            print("Starting policy evaluation")
            self.policy_evaluation()
            policy_stable = True
            for s in range(self.discrete_states.shape[0]):
                state = self.discrete_states[s]
                temp = self.policy_table[s]
                p = np.argmax([self.predict_reward(state, a) + self.gamma *self.value_table[self.predict_next_state(state, a, True)] for a in self.discrete_actions])
                #p = self.find_next_action(p)
                p = self.discrete_actions[p]
                self.policy_table[s] = p
                #print(temp, p)
                if not (temp == p):
                    policy_stable = False
            if policy_stable:
                print("Policy stable")
                #return self.value_table, self.policy_table
                break

    @timecall()
    def policy_iteration(self):
        self.policy_improvement()
        return self.value_table, self.policy_table

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
        idx = self.find_nearest_state( prediction)
        if index:
            return idx
        else:
            return self.discrete_states[idx]


def main(env, tol, discount):
    VI = PolicyIteration(env, tol, discount)
    VI.policy_iteration()
    VI.test_policy()

main('Pendulum-v2', 0.001, 0.95)






