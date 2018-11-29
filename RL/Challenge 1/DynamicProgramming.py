import numpy as np
import gym
import quanser_robots

import sklearn

import RegressionProblems as rp

# TODO: Disabling train_test_split?

MAX_ITERATIONS = 5
DISCRETIZATION = 9

class ValueIteration(object):
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
        #self.discretized_states = self.discretize_state_space(DISCRETIZATION)



        s = 0
        self.discretized_states = []
        for i in np.linspace(-self.env.observation_space.high[0], self.env.observation_space.high[0],13 ):
            for j in np.linspace(-self.env.observation_space.high[1], self.env.observation_space.high[1], 39 ):
                    state = [i, j]
                    if s == 0:
                        self.discretized_states = state
                        s = 1
                    else:
                        self.discretized_states =  np.vstack((self.discretized_states, state))
        print(self.discretized_states)

        self.discretized_actions = [-2,0, 2]
        print(self.discretized_actions)
        self.value_table = np.zeros(self.discretized_states.shape[0])
        self.policy_table = np.zeros(self.discretized_states.shape[0])

        self.debug = 0



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

        #MAYBE
        # TODO: idk, prob correct
        idx = array.shape[1] + 1
        debug1 = state
        debug2 = array[0]


        nearest_theta = -20
        nearest_state = [-20, -50]
        for i in range(array.shape[0]):

            if np.abs(array[i][0] - state[0][0]) < np.abs(nearest_theta - state[0][0]):
                nearest_theta = array[i][0]
        for i in range(array.shape[0]):
            if array[i][0] == nearest_theta:
                if np.abs(array[i][1] - state[0][1]) < np.abs(nearest_state[1] - state[0][1]):
                    #print("val")
                    #print(array[i][1] - state[0][1])
                    #print(nearest_state[1] - state[0][1])
                    nearest_state = array[i]
                    idx = i

        #for i in range(array.shape[0]):
            #dif = np.linalg.norm(array[i,:] - state)
            #if dif < difference:
                #difference = dif
                #idx = i
        return nearest_state, idx

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

    def log_spacing(self, high, n, low=0.001):
        right_side = np.geomspace(low, high, int(np.round(n / 2)))
        left_side = np.flip(-np.copy(right_side))
        left_side = np.append(left_side, 0)
        space = np.concatenate((left_side, right_side))
        return space

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

    def bellman_tabular(self, state, actions, isVI, action):
        table = self.value_table
        #for VI
        if isVI:
            value = np.max([self.reward_model.predict(np.append(a, state).reshape(1, -1)) +
                        self.gamma * table[self.approximate_next_state(state, a, tabular=True)] for a in actions])
        #for PI
        else:
            value = self.reward_model.predict(np.append(action, state).reshape(1, -1)) + self.gamma * table[
                self.approximate_next_state(state, action, tabular=True)]


        return value


    def iterate(self, isVI = True):
        i = 0
        delta = 1
        states = self.discretized_states
        actions = self.discretized_actions
        while delta > self.tolerance:
            delta = 0
            for state in range(states.shape[0]):

                #bad solution
                # TODO:  implement new method since action is not used for VI or find other solution
                action = self.policy_table[state]

                v = self.value_table[state]
                value = self.bellman_tabular(states[state,:], actions, isVI, action)

                self.value_table[state] = value
                abs = np.abs(v - value)
                delta = np.max([delta, abs])
                #print(delta)

        # Get policy

    def PI(self):
        print("END")
        print("_____")
        self.iterate(False)
        print("_____")
        print("START")
        states = self.discretized_states
        actions = self.discretized_actions
        policy_stable = True

        for state in range(states.shape[0]):
            #print(state)
            old_action = self.policy_table[state]
            max_idx = np.argmax([self.reward_model.predict(np.append(a, states[state, :]).reshape(1, -1)) +
                        self.gamma * self.value_table[self.approximate_next_state(states[state, :], a, tabular=True)] for a in actions])



            self.policy_table[state] = actions[max_idx]
            if old_action != self.policy_table[state]:
                policy_stable = False


        if policy_stable is not True:
            self.PI()
        else:
            return

class PolicyIteration(object):
    def __init__(self, environment, tolerance, discount_factor):
        self.env = gym.make(environment)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.shape[0]

        self.tolerance = tolerance
        self.gamma = discount_factor

        self.state_model = rp.fit_state_model(environment)
        self.reward_model = rp.fit_reward_model(environment)

        self.state = self.env.reset()
        self.discrete_states = self.discretize_states(13, 39)
        self.discrete_actions = np.array([-2, 0, 2])
        self.value_table = np.zeros(self.discrete_states.shape[0])
        self.policy_table = np.ones(self.discrete_states.shape[0])

    def discretize_states(self, n_theta, n_theta_dot):
        disc_states = []
        s = 0
        for i in np.linspace(-self.env.observation_space.high[0], self.env.observation_space.high[0], n_theta):
            for j in np.linspace(-self.env.observation_space.high[1], self.env.observation_space.high[1], n_theta_dot):
                    state = [i, j]
                    if s == 0:
                        disc_states = state
                        s = 1
                    else:
                        disc_states = np.vstack((disc_states, state))
        return np.array(disc_states)

    def policy_evaluation(self):
        while True:
            delta = 0
            for s in range(self.discrete_states.shape[0]):
                temp = self.value_table[s]
                action = self.policy_table[s]
                state = self.discrete_states[s]
                value = self.predict_reward(state, action) + self.gamma * self.value_table[self.predict_next_state(state, action, True)]
                self.value_table[s] = value
                delta = np.max([delta, np.abs(temp - value)])
                if delta < self.tolerance:
                    print("Starting policy improvement")
                    self.policy_improvement()
                    break


    def policy_improvement(self):
        policy_stable = True
        for s in range(self.discrete_states.shape[0]):
            state = self.discrete_states[s]
            temp = self.policy_table[s]
            p = np.argmax([self.predict_reward(state, a) + self.gamma * self.predict_next_state(state, a, False) for a in self.discrete_actions])
            self.policy_table[s] = p
            print(temp, p)
            if not (temp == p):
                policy_stable = False
            if policy_stable:
                break
            else:
                print("Starting Policy evaluation...")
                self.policy_evaluation()
            return self.value_table, self.policy_table

    def policy_iteration(self):
        self.policy_evaluation()
        return self.value_table, self.policy_table

    def predict_reward(self, state, action):
        clf = self.reward_model
        print(action)
        x, r, y, z = self.env.step(action)
        query_point = np.append(action, state).reshape(1, -1)
        prediction = clf.predict(query_point)
        print("Prediction: ", prediction)
        print("True: ", r)
        return prediction

    def predict_next_state(self, state, action, index):
        # Mittelmäßige Outputs
        clf = self.state_model
        states = self.discrete_states
        query_point = np.append(action, state).reshape(1, -1)
        prediction = clf.predict(query_point)
        idx = states.shape[0] +1
        best = np.inf
        for i in range(states.shape[0]):
            norm = np.linalg.norm(states[i].reshape(1, -1) - prediction)
            if norm < best:
                best = norm
                idx = i
        if index:
            return idx
        else:
            return states[idx]

    def sample_input(self):
        action = self.env.action_space.sample()
        return np.array(self.state), np.array(action)

def test(env):
    PI = PolicyIteration(env, 0.01, 0.95)
    print(PI.discrete_states[0].shape)
    state, action = PI.sample_input()
    print("Sample: ", state, action)
    print(PI.predict_next_state(state, action, False))
    print(PI.policy_iteration())


test('Pendulum-v2')
def main(environment, tolerance):
    VI = ValueIteration(environment, tolerance, 0.9)
    s, a = VI.sample_input()
    actions = VI.discretize_action_space(25)
    #print(actions)
    #print(VI.bellman_equation(VI.state, 0, actions))
    VI.PI()
    print(VI.policy_table)

    rewards = []
    calc_rewards = []
    for i in range(200):
        obs = VI.env.reset()
        #obs = [0, 0]
        obs = np.asarray([obs])
        done = False
        while not done:
            print("___")

            ns, idx = VI.find_nearest_state(obs)
            action = VI.policy_table[idx]

            obs, reward, done, info = VI.env.step(np.array([action]))

            print(VI.reward_model.predict(np.append(action, VI.discretized_states[idx, :]).reshape(1, -1)))
            print(reward)
            rewards.append(reward)
            calc_rewards.append(VI.reward_model.predict(np.append(action, VI.discretized_states[idx, :]).reshape(1, -1)))
            #obs = [0, 0]
            obs = np.asarray([obs])
            VI.env.render()
        sum = 0
        for i in range(len(rewards)):
            sum += (rewards[i] - calc_rewards[i])**2
        print(sum/len(rewards))
    #print(VI.approximate_next_state(s,a))

#main('Pendulum-v2', 100)

