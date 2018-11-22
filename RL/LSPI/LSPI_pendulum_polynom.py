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
    def __init__(self, environment, exploration_rate = 0.3, discount_factor = 0.95, epsilon = 0.001):
        self.environment = environment
        #195
        self.numberOfFeatures = 95 + 1
        self.m_discount_factor = discount_factor
        self.m_epsilon = epsilon
        self.exploration_rate = exploration_rate

        self.actions = [-2, 0, 2]
        self.amountOfActions = len(self.actions)
        self.n = self.numberOfFeatures * self.amountOfActions
        #im not sure if the first B matrix is the first or the second one since the paper states that B is the inverse of A
        #self.m_B = np.zeros((n, n)) + 0.01 * np.ones((n,n))
        self.m_B = 1000000*np.identity((self.n))
        self.m_b = np.zeros((self.n, 1))
        self.w = np.zeros((self.n, 1))

        #discretized action space as i dont know yet how to deal with continous ones

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
        summed_values = np.zeros((self.amountOfActions, 1))
        for i in range(len(summed_values)):
            basis = self.getBasisFunctionColumn(state, i)
            for j in range(self.numberOfFeatures):
                summed_values[i] += self.w[i*self.numberOfFeatures + j] * basis[i*self.numberOfFeatures + j]

        best_action_index = 0
        for i in range(len(summed_values)):
            if(summed_values[i] > summed_values[best_action_index]): best_action_index = i
        return best_action_index

    """transforms five dimensional observation into four dimensional state"""
    def obsToState(self, obs):
        if (obs[1] <= 0):
            theta = 2 * np.pi - np.arccos(obs[2])
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
        i = currentAction_id*self.numberOfFeatures
        basisFunctionColumn[i] = 1
        i += 1
        for l in range(len(current_state)):
            for m in range(len(current_state)):
                for k in range(1, 3, 1):
                    for j in range(1, 3, 1):
                        if l == m:
                            continue
                        basisFunctionColumn[i] = (current_state[l] ** k) * (current_state[m] ** j)
                        i += 1

        for l in range(len(current_state)):
            for k in range(1, 4, 1):
                basisFunctionColumn[i] = current_state[l] ** k
        """for l in range(len(current_state)):
            for m in range(len(current_state)):
                for n in range(len(current_state)):
                    for x in range(1, 4, 1):
                        for y in range(1, 4, 1):
                            for z in range(1 , 4, 1):
                                if (l == m and m == n) or (x + y + z) > 4 :
                                    continue
                                basisFunctionColumn[i] = (current_state[l] ** x) * (current_state[m] ** y) *(current_state[n] ** z)
                                i += 1"""


        """basisFunctionColumn[i+1] = x #x
        basisFunctionColumn[i+2] = x*x
        basisFunctionColumn[i+3] = theta
        basisFunctionColumn[i+4] = theta * theta
        basisFunctionColumn[i+5] = xdot
        basisFunctionColumn[i+6] = xdot * xdot
        basisFunctionColumn[i+7] = thetadot
        basisFunctionColumn[i+8] = thetadot*thetadot

        basisFunctionColumn[i+9] = x*xdot
        basisFunctionColumn[i+10] =x*x*xdot
        basisFunctionColumn[i+11] = x*xdot*xdot

        basisFunctionColumn[i+12] = theta*xdot
        basisFunctionColumn[i+13] = theta*theta*xdot
        basisFunctionColumn[i+14] = theta*xdot*xdot #

        basisFunctionColumn[i+15] = xdot *thetadot
        basisFunctionColumn[i+16] = xdot*xdot*thetadot
        basisFunctionColumn[i+17] = xdot * thetadot*thetadot

        basisFunctionColumn[i+18] =  theta* thetadot
        basisFunctionColumn[i+19] = theta*theta*thetadot
        basisFunctionColumn[i+20] = theta*thetadot*thetadot

        basisFunctionColumn[i + 21] = theta*thetadot*xdot
        basisFunctionColumn[i + 22] = theta*thetadot*xdot*xdot
        basisFunctionColumn[i + 23] = theta*thetadot*xdot*thetadot
        basisFunctionColumn[i + 24] = theta*thetadot*xdot*theta

        basisFunctionColumn[i + 25] = theta*theta*theta*thetadot
        basisFunctionColumn[i + 26] = theta*thetadot*thetadot*thetadot
        basisFunctionColumn[i + 27] = theta*theta *thetadot*thetadot

        basisFunctionColumn[i+28] = xdot *thetadot*thetadot*thetadot
        basisFunctionColumn[i+29] = xdot*xdot*thetadot*thetadot
        basisFunctionColumn[i+30] = xdot * xdot*xdot*thetadot"""

        return basisFunctionColumn

    """returns parameters w"""
    def LSDTQ(self, current_state, previous_state, currentAction_id, previousAction_id, reward):

        self.m_B = self.m_B - self.matrixBupdate(current_state, previous_state, currentAction_id, previousAction_id)
        self.m_b = self.m_b + self.getBasisFunctionColumn(previous_state, previousAction_id) * reward

        return np.dot(self.m_B, self.m_b)

    #returns policy according to the LSPI algorithm
    def LSPI_algorithm(self, firstAction_id = 0, training_samples = 100, maxTimeSteps = 4000):
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
            for t in range(maxTimeSteps):
                #print(t)
                previous_state = current_state
                previousAction_id = currentAction_id
                obs, reward, done, _ = self.environment.step(np.array([self.actions[previousAction_id]]))
                # since sometimes done is an array and sometimes its not (and i dont know why this is yet) this has to be checked


                current_reward += reward
                #self.environment.render()
                current_state = obs

                #if (current_state[0] < -0.4 or current_state[0] > 0.4):
                        #break
                if done:
                    break

                #exploration
                if(randint(0,9) < self.exploration_rate * 10.):
                    currentAction_id = randint(0,self.amountOfActions -1)
                else:
                    currentAction_id = self.returnBestAction(current_state)
                    #print("best action")
                    #print(currentAction_id)

                self.w = self.LSDTQ(current_state, previous_state, currentAction_id, previousAction_id, reward)
                distance = 0


                for k in range(len(self.w)):
                   distance += (self.w[k] - old_w[k]) * (self.w[k] - old_w[k])
                #print(distance)
                if distance < self.m_epsilon:
                    return self.w

    def apply(self):
        print("APPLY STARTS HERE")
        for n in range(10000):
            obs = self.environment.reset()
            current_state = obs

            currentAction_id = self.returnBestAction(current_state)
            for t in range(1000):
                obs, reward, done, _ = self.environment.step(np.array([self.actions[currentAction_id]]))
                # since sometimes done is an array and sometimes its not (and i dont know why this is yet) this has to be checked
                self.environment.render()
                current_state = obs
                currentAction_id = self.returnBestAction(current_state)


def main():
    env = gym.make('Pendulum-v0')  # Use "Cartpole-v0" for the simulation
    env.reset()
    ctrl = SwingupCtrl(long=False)  # Use long=True if you are using the long pole
    env.step(np.array([0.]))
    env.close()

    xd =LSPI(env)
    w = xd.LSPI_algorithm()

    xd.apply()



main()



