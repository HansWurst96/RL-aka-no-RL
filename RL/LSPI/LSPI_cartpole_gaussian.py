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
    def __init__(self, environment, exploration_rate = 0.3, discount_factor = 0.95, epsilon = 0.01):
        self.environment = environment

        self.numberOfFeatures = 3*3*3*3*9 + 1
        self.m_discount_factor = discount_factor
        self.m_epsilon = epsilon
        self.exploration_rate = exploration_rate

        #discretized action space as i dont know yet how to deal with continous ones
        self.actions = [-24, 0, 24]

        self.numberOfActions = len(self.actions)
        self.n = self.numberOfActions * self.numberOfFeatures
        #im not sure if the first B matrix is the first or the second one since the paper states that B is the inverse of A
        #self.m_B = np.zeros((n, n)) + 0.01 * np.ones((n,n))
        self.m_B = 100000*np.identity((self.n))
        self.m_b = np.zeros((self.n, 1))
        self.w = np.zeros((self.n, 1))


        #4*4*4*4 gaussian features
        self.m_mu = []
        for x in np.linspace(-0.25, 0.25, 3):
                for sin_th in np.linspace(-1.0, 1, 3):
                    for cos_th in np.linspace(-1, 1, 3):
                        for x_dot in np.linspace(-1.0, 1.0, 3):
                            for theta_dot in [-8, -4, -2, -1 , 0 ,1, 2, 4, 8]:
                                mu = [x, sin_th, cos_th, x_dot, theta_dot]
                                self.m_mu.append(mu)
        self.m_mu = np.array(self.m_mu)
        self.cov = np.identity((5))

        #betw. sinth and costh
        self.cov[1][2] = -0.8
        self.cov[2][1] = -0.8

        #betw xdot and thetadot
        self.cov[3][4] = 0.4
        self.cov[4][3] = 0.4

        #betw. x and x dot
        self.cov[0][3] = 0.0
        self.cov[3][0] = 0.0

        self.invCov = np.linalg.inv(self.cov)
        print(self.invCov)
        print(np.linalg.eigvals(self.cov))
        print(np.linalg.eigvals(self.invCov))

    """returns gaussian function of given state and mean"""
    def Gaussian(self, state, mean, variance = 1):
        exponent = 0
        exponent += np.dot((state - mean).T , np.dot(self.invCov , (state - mean)))
        if exponent < -1000:
            print("exp")
            return 1
        return math.exp(-exponent/2)

    """returns best action_id according to policy"""
    def returnBestAction(self, state):
        summed_values = np.zeros((self.numberOfActions, 1))
        for i in range(self.numberOfFeatures):
            #we use the same gaussians for each of the 8 actions
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
    """(one of) the bug(s)/errors should occur here, as the update matrix only seems to have the same entries which shouldnt be the case"""
    """Also the entries are very small"""
    def matrixBupdate(self, current_state, previous_state, currentAction_id, previousAction_id):
        basisFunctionColumn = self.getBasisFunctionColumn(previous_state, previousAction_id)
        nextStateBasisFunctionColumn = self.getBasisFunctionColumn(current_state, currentAction_id)

        debug1 = basisFunctionColumn - self.m_discount_factor * nextStateBasisFunctionColumn

        phi_B_product = np.dot((basisFunctionColumn - self.m_discount_factor * nextStateBasisFunctionColumn).T,
                               self.m_B)

        debug2 = np.dot(basisFunctionColumn, phi_B_product)
        nominator = np.dot(self.m_B, np.dot(basisFunctionColumn, phi_B_product))
        denominator = 1 + np.dot(phi_B_product, basisFunctionColumn)

        return nominator / denominator

    """returns Basis function column"""
    def getBasisFunctionColumn(self, current_state, currentAction_id):
        basisFunctionColumn = np.zeros((self.n, 1))

        # only the rows corresponding to the current action are not zero
        #currentAction_id * self.magicNumber, (currentAction_id + 1) * self.magicNumber
        for i in range(self.numberOfFeatures):
            if i == 0:
                basisFunctionColumn[currentAction_id * self.numberOfFeatures] = 1
            else:
                mean = self.m_mu[i-1]
                basisFunctionColumn[i + currentAction_id * (self.numberOfFeatures)] = self.Gaussian(current_state, mean)
        #for i in range(self.n):
         #   basisFunctionColumn[self.n -1 - i] = basisFunctionColumn[self.n - i - 2]
        #basisFunctionColumn[currentAction_id * self.magicNumber] = 1
        return basisFunctionColumn

    """returns parameters w"""
    def LSDTQ(self, current_state, previous_state, currentAction_id, previousAction_id, reward):

        self.m_B = self.m_B - self.matrixBupdate(current_state, previous_state, currentAction_id, previousAction_id)
        self.m_b = self.m_b + self.getBasisFunctionColumn(previous_state, previousAction_id) * reward

        return np.dot(self.m_B, self.m_b)

    #returns policy according to the LSPI algorithm
    def LSPI_algorithm(self, firstAction_id = 0, training_samples = 100, maxTimeSteps = 1000):
        for n in range(training_samples):
            obs = self.environment.reset()
            current_state = obs
            if n == 0:
                currentAction_id = firstAction_id
            else:
                currentAction_id = self.returnBestAction(current_state)
            print("N: ")
            print(n)
            old_w = self.w
            current_reward = 0

            debug_amountActionWasUsed = np.zeros((self.numberOfActions, 1))
            for t in range(maxTimeSteps):
                print(t)
                previous_state = current_state
                previousAction_id = currentAction_id
                obs, reward, done, _ = self.environment.step(np.array([self.actions[previousAction_id]]))
                # since sometimes done is an array and sometimes its not (and i dont know why this is yet) this has to be checked

                #print(done)
                current_reward += reward
                #self.environment.render()
                current_state = obs

                if (current_state[0] < -0.35 or current_state[0] > 0.35):
                        break
                #elif done:
                    #break

                #exploration
                if(randint(0,9) < self.exploration_rate * 10.):
                    currentAction_id = randint(0,self.numberOfActions - 1)
                else:
                    currentAction_id = self.returnBestAction(current_state)
                    debug_amountActionWasUsed[currentAction_id] +=1
                    #print("best action")
                    #print(currentAction_id)

                self.w = self.LSDTQ(current_state, previous_state, currentAction_id, previousAction_id, reward)
                distance = 0


                for k in range(len(self.w)):
                   distance += (self.w[k] - old_w[k]) * (self.w[k] - old_w[k])


                if distance < self.m_epsilon:
                    return self.w
            print(current_reward / t)
    def apply(self):
        print("APPLY STARTS HERE")
        for n in range(10000):
            obs = self.environment.reset()
            current_state = obs

            currentAction_id = self.returnBestAction(current_state)
            for t in range(300):
                obs, reward, done, _ = self.environment.step(np.array([self.actions[currentAction_id]]))
                # since sometimes done is an array and sometimes its not (and i dont know why this is yet) this has to be checked
                self.environment.render()
                current_state = obs
                currentAction_id = self.returnBestAction(current_state)


def main():
    env = gym.make('CartpoleSwingShort-v0')  # Use "Cartpole-v0" for the simulation
    env.reset()
    ctrl = SwingupCtrl(long=False)  # Use long=True if you are using the long pole
    env.step(np.array([0.]))
    env.close()

    xd =LSPI(env)
    w = xd.LSPI_algorithm()

    xd.apply()



main()



