import numpy as np
import math
from random import randint
from random import uniform
import gym
from quanser_robots.cartpole.ctrl import SwingupCtrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing.dummy import Pool as ThreadPool

class LSPI:
    """epsilon: allowed margin of error such that p' = p
       n: Amount of basis functions
       exploration_rate: chance of exploring instead of exploiting"""
    def __init__(self, environment, noF, bandwidth, exploration_rate = 1, discount_factor = 0.95, epsilon = 0.0000001):
        self.environment = environment
        self.numberOfFeatures = noF
        self.m_discount_factor = discount_factor
        self.m_epsilon = epsilon
        self.exploration_rate = exploration_rate
        self.exploration_decy_rate = 0.9999

        

        #discretized action space as i dont know yet how to deal with continous ones
        self.actions = [-1, 1]

        self.numberOfActions = len(self.actions)
        self.n = self.numberOfActions * self.numberOfFeatures
        #im not sure if the first B matrix is the first or the second one since the paper states that B is the inverse of A
        #self.m_B = np.zeros((n, n)) + 0.01 * np.ones((n,n))
        self.m_B = 10000*np.identity((self.n))
        self.m_b = np.zeros((self.n, 1))
        self.w = np.zeros((self.n, 1))


        self.allFeatures = self.get_feature_fun(5, self.numberOfFeatures, bandwidth)

    """returns gaussian function of given state and mean"""
    def Gaussian(self, state, mean, variance = 1):
        exponent = 0
        exponent += np.dot((state - mean).T , np.dot(self.invCov , (state - mean)))
        if exponent < -1000:
            print("exp")
            return 1
        return math.exp(-exponent/2)

    def get_feature_fun(self, state_dim, feature_dim, bandwidth):
        freq = np.random.randn(feature_dim, state_dim) * np.sqrt(2 / bandwidth)
        shift = np.random.uniform(-np.pi, np.pi, feature_dim)
        return lambda x: np.sin(x @ freq.T + shift)

    """returns best action_id according to policy"""
    def returnBestAction(self, state):
        summed_values = np.zeros((self.numberOfActions, 1))
        basis = self.allFeatures(np.transpose(state))
        for i in range(self.numberOfFeatures):
            for j in range(len(summed_values)):
                summed_values[j] += self.w[i+j*self.numberOfFeatures] * basis[i]

        best_action_index = 0
        for i in range(len(summed_values)):
            if(summed_values[i] > summed_values[best_action_index]): best_action_index = i
        return best_action_index

    """transforms five dimensional observation into four dimensional state"""
    def obsToState(self, obs):
        if (obs[1] <= 0):
            theta =  -np.arccos(obs[2])
        else:
            theta = np.arccos(obs[2])

        #x, theta, x_dot, theta_dot
        state = np.array([obs[0], theta, obs[3], obs[4]])
        return state

    """returns update step of LSTDQ-OPT algorithm"""
    def matrixBupdate(self, current_state, currentAction_id, bfc):
        basisFunctionColumn = bfc
        nextStateBasisFunctionColumn = self.getBasisFunctionColumn(current_state, currentAction_id)

        phi_B_product = np.dot((basisFunctionColumn - self.m_discount_factor * nextStateBasisFunctionColumn).T,
                               self.m_B)

        nominator = np.dot(self.m_B, np.dot(basisFunctionColumn, phi_B_product))
        denominator = 1 + np.dot(phi_B_product, basisFunctionColumn)

        return nominator / denominator

    """returns Basis function column"""
    def getBasisFunctionColumn(self, current_state, currentAction_id):
        basisFunctionColumn = np.zeros((self.n, 1))
        # only the rows corresponding to the current action are not zero
        basis = self.allFeatures(np.transpose(current_state))
        for i in range(self.numberOfFeatures):
            basisFunctionColumn[i + currentAction_id * (self.numberOfFeatures)] = basis[i]
        return basisFunctionColumn

    """returns parameters w"""
    def LSDTQ(self, current_state, previous_state, currentAction_id, previousAction_id, reward, current_n):

        bfc = self.getBasisFunctionColumn(previous_state, previousAction_id)
        self.m_B = self.m_B - self.matrixBupdate(current_state, currentAction_id, bfc)
        self.m_b = self.m_b +  bfc * reward
        if current_n % 10 == 0:
            return np.dot(self.m_B, self.m_b)

    """returns policy according to the LSPI algorithm"""
    def LSPI_algorithm(self, firstAction_id = 0, training_samples = 200, maxTimeSteps = 2000):
        doneActions = 0
        while doneActions < 100:


            obs = self.environment.reset()
            current_state =obs
            currentAction_id = randint(0, self.numberOfActions - 1)
            for t in range(maxTimeSteps):
                if doneActions % 10000 == 0:
                    print(doneActions)
                previous_state = current_state
                previousAction_id = currentAction_id

                obs, reward, done, _ = self.environment.step(np.array([self.actions[previousAction_id]]))

                doneActions += 1

                #self.environment.render()
                current_state = obs

                if done:
                    break

                currentAction_id = randint(0, self.numberOfActions - 1)

                self.LSDTQ(current_state, previous_state, currentAction_id, previousAction_id, reward, 4)






            #x_axis.append(doneActions)
            #y_axis.append(current_reward)
        self.w = np.dot(self.m_B, self.m_b)
        #plt.scatter(x_axis, y_axis)
        #plt.show()




    def apply(self):
        print("start")
        allReward = 0
        for n in range(50):
            obs = self.environment.reset()
            current_state = obs
            currentAction_id = self.returnBestAction(current_state)
            for t in range(1000):
                obs, reward, done, _ = self.environment.step(np.array([self.actions[currentAction_id]]))
                allReward += reward
                if done:
                    break
                self.environment.render()
                current_state = obs
                currentAction_id = self.returnBestAction(current_state)
        return allReward / 50

env = gym.make('CartpoleStabShort-v0')  # Use "Cartpole-v0" for the simulation
env.reset()
def main():

    ctrl = SwingupCtrl(long=False)  # Use long=True if you are using the long pole
    env.step(np.array([0.]))
    env.close()
    ls = LSPI(env, 260, 5.2)
    ls.LSPI_algorithm()
    ls.apply()
    #xd2 = LSPI(env, 200, 5.2)
    #xd2.LSPI_algorithm()
    #for i in range(len(data)):
        #data_sample = data[i]
        #xd2.LSDTQ(data_sample[0], data_sample[1], data_sample[2], data_sample[3], data_sample[4], 1)
    #xd2.w = np.dot(xd2.m_B, xd2.m_b)


    #val1 = xd.apply()
    #val2 = xd2.apply()

    #print(val1)
    #print(val2)


def findBest(input):
    z_axis = []
    noF = input
    print(input)
    averageReward = 0
    for x in range(1):
        print(x)
        xd = LSPI(env, noF, 0.25)
        xd.LSPI_algorithm()
        averageReward += xd.apply()
    averageReward = averageReward / 1
    z_axis.append(averageReward)
    return z_axis




main()



