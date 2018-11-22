import numpy as np
import math
from random import randint
from random import uniform
import gym
from quanser_robots.cartpole.ctrl import SwingupCtrl
import matplotlib.pyplot as plt

class LSPI:
    """epsilon: allowed margin of error such that p' = p
       n: Amount of basis functions (gaussians in this case)
       exploration_rate: chance of exploring instead of exploiting"""
    def __init__(self, environment, cova, exploration_rate = 0.3, discount_factor = 0.95, epsilon = 0.01):
        self.environment = environment

        #we have 5*5*7 gaussian features (see below) as basis functions aswell as the additional constant 1 function
        self.numberOfFeatures = 5*5*7 + 1
        self.m_discount_factor = discount_factor
        self.m_epsilon = epsilon
        self.exploration_rate = exploration_rate

        #discretized action space as i dont know yet how to deal with continous ones
        self.actions = [-2, 0, 2]

        self.numberOfActions = len(self.actions)
        self.n = self.numberOfActions * self.numberOfFeatures
        self.m_B = 1000000*np.identity((self.n))
        self.m_b = np.zeros((self.n, 1))
        self.w = np.zeros((self.n, 1))

        #gaussian features
        self.m_mu = []
        for sinTheta in np.linspace(-1, 1, 5):
            for cosTheta in np.linspace(-1, 1, 5):
                for theta_dot in np.linspace(-6, 6, 7):
                    mu = [sinTheta, cosTheta, theta_dot]
                    self.m_mu.append(mu)
        self.m_mu = np.array(self.m_mu)

        #covariance matrix
        self.cov = np.identity((3))
        self.cov[0][1] = cova
        self.cov[1][0] = cova

        self.invCov = np.linalg.inv(self.cov)

    """returns gaussian function of given state and mean"""
    def Gaussian(self, state, mean, variance = 1):
        exponent = np.dot((state-mean).T, np.dot(self.invCov,(state-mean)))
        return math.exp(-exponent/2)

    """returns best action_id according to policy"""
    def returnBestAction(self, state):
        summed_values = np.zeros((self.numberOfActions, 1))
        for i in range(self.numberOfFeatures):
            #we use the same gaussians for each actions
            if i == 0:
                gaussian = 1
            else:
                mean = self.m_mu[i-1]
                gaussian = self.Gaussian(state, mean)
            for j in range(len(summed_values)):
                summed_values[j] += self.w[i+j*self.numberOfFeatures] * gaussian

        best_action_index = 0
        for i in range(len(summed_values)):
            if(summed_values[i] > summed_values[best_action_index]): best_action_index = i
        return best_action_index

    """transforms five dimensional observation into four dimensional state, probably not needed anymore"""
    def obsToState(self, obs):
        if (obs[1] <= 0):
            theta =  -np.arccos(obs[2])
        else:
            theta = np.arccos(obs[2])

        #x, theta, x_dot, theta_dot
        state = np.array([obs[0], theta, obs[3], obs[4]])
        return state

    """returns update step of LSTDQ-OPT algorithm"""
    def matrixBupdate(self, current_state, previous_state, currentAction_id, previousAction_id):
        basisFunctionColumn = self.getBasisFunctionColumn(previous_state, previousAction_id)
        nextStateBasisFunctionColumn = self.getBasisFunctionColumn(current_state, currentAction_id)

        phi_B_product = np.dot((basisFunctionColumn - self.m_discount_factor * nextStateBasisFunctionColumn).T, self.m_B)

        nominator = np.dot(self.m_B, np.dot(basisFunctionColumn, phi_B_product))
        denominator = 1 + np.dot(phi_B_product, basisFunctionColumn)

        return nominator / denominator

    """returns Basis function column"""
    def getBasisFunctionColumn(self, current_state, currentAction_id):
        basisFunctionColumn = np.zeros((self.n, 1))

        #only the rows corresponding to the current action are not zero
        for i in range(self.numberOfFeatures):
            if i == 0:
                basisFunctionColumn[currentAction_id * self.numberOfFeatures] = 1
            else:
                mean = self.m_mu[i-1]
                basisFunctionColumn[i + currentAction_id * (self.numberOfFeatures)] = self.Gaussian(current_state, mean)
        return basisFunctionColumn

    """returns parameters w"""
    def LSDTQ(self, current_state, previous_state, currentAction_id, previousAction_id, reward):

        self.m_B = self.m_B - self.matrixBupdate(current_state, previous_state, currentAction_id, previousAction_id)
        self.m_b = self.m_b + self.getBasisFunctionColumn(previous_state, previousAction_id) * reward

        return np.dot(self.m_B, self.m_b)

    """returns policy according to the LSPI algorithm"""
    def LSPI_algorithm(self, firstAction_id = 0, training_samples = 1, maxTimeSteps = 1000):
        needed_episodes = 0
        for n in range(training_samples):
            obs = self.environment.reset()
            current_state = obs
            if n == 0:
                currentAction_id = firstAction_id
            else:
                currentAction_id = self.returnBestAction(current_state)

            current_reward = 0

            debug_amountActionWasUsed = np.zeros((self.numberOfActions, 1))
            for t in range(maxTimeSteps):
                needed_episodes +=1
                old_w = self.w
                previous_state = current_state
                previousAction_id = currentAction_id
                obs, reward, done, _ = self.environment.step(np.array([self.actions[previousAction_id]]))
                current_reward += reward
                #self.environment.render()
                current_state = obs

                if done:
                    break

                #exploration
                if(randint(0,9) < self.exploration_rate * 10.):
                    currentAction_id = randint(0,self.numberOfActions - 1)
                else:
                    currentAction_id = self.returnBestAction(current_state)
                    debug_amountActionWasUsed[currentAction_id] +=1


                self.w = self.LSDTQ(current_state, previous_state, currentAction_id, previousAction_id, reward)
                margin = 0
                for k in range(len(self.w)):
                    margin += (self.w[k] - old_w[k]) * (self.w[k] - old_w[k])
                if margin < self.m_epsilon:
                    return self.w




def apply(env):
    cov_test_values = np.linspace(-0.8, -0.8, 1)
    rewards = []
    for j in range(len(cov_test_values)):
        print(j)
        average_reward = 0
        testObject = LSPI(env, cov_test_values[j])
        testObject.LSPI_algorithm()
        for n in range(100):
            print(n)
            obs = testObject.environment.reset()
            current_state = obs
            currentAction_id = testObject.returnBestAction(current_state)

            done = False
            while not done:
                obs, reward, done, _ = testObject.environment.step(np.array([testObject.actions[currentAction_id]]))
                average_reward += reward
                #testObject.environment.render()
                current_state = obs
                currentAction_id = testObject.returnBestAction(current_state)

        average_reward = average_reward / 500
        rewards.append(average_reward)
    print(rewards)
    plt.scatter(cov_test_values, rewards)
    plt.show()
    print("done")

def main():
    env = gym.make('Pendulum-v0')  # Use "Cartpole-v0" for the simulation
    env.reset()
    ctrl = SwingupCtrl(long=False)  # Use long=True if you are using the long pole
    env.step(np.array([0.]))
    env.close()

    apply(env)




main()



