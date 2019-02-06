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
    def __init__(self, environment, noF, bandwidth, loaded = False, discount_factor = 0.99, epsilon = 0.0000001):
        self.environment = environment
        self.numberOfFeatures = noF
        self.m_discount_factor = discount_factor
        self.m_epsilon = epsilon

        self.bandwidth = bandwidth
        
        self.actions = [-2, 2]
        #discretized action space as i dont know yet how to deal with continous ones

        self.numberOfActions = len(self.actions)
        self.n = self.numberOfActions * self.numberOfFeatures
        #im not sure if the first B matrix is the first or the second one since the paper states that B is the inverse of A
        #self.m_B = np.zeros((n, n)) + 0.01 * np.ones((n,n))
        self.m_B = 100000*np.identity((self.n))
        self.m_b = np.zeros((self.n, 1))
        self.w = np.zeros((self.n, 1))

        self.reuse_B = np.identity((self.n)) * 0
        self.reuse_b = np.zeros((self.n, 1))

        self.allBFC = []
        self.state_dim = 5
        self.freq = []
        self.shift = []
        print("k")
        if not loaded:
            self.allFeatures = self.get_feature_fun(self.state_dim, self.numberOfFeatures, bandwidth, loaded)

    """returns gaussian function of given state and mean"""
    def Gaussian(self, state, mean, variance = 1):
        exponent = 0
        exponent += np.dot((state - mean).T , np.dot(self.invCov , (state - mean)))
        if exponent < -1000:
            print("exp")
            return 1
        return math.exp(-exponent/2)

    def get_feature_fun(self, state_dim, feature_dim, bandwidth, loaded):
        if not loaded:
            freq = np.random.randn(feature_dim, state_dim) * np.sqrt(2 / bandwidth)
            print(freq)
            shift = np.random.uniform(-np.pi, np.pi, feature_dim)
            self.freq = freq
            self.shift = shift
        if loaded:
            freq = self.freq
            shift = self.shift
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
    def matrixBupdate(self, i):

        basisFunctionColumn = self.allBFC[i]

        nextStateBasisFunctionColumn = self.allBFC[i+1]

        phi_B_product = np.dot((basisFunctionColumn - self.m_discount_factor * nextStateBasisFunctionColumn).T,
                               self.m_B)

        nominator = np.dot(self.m_B, np.dot(basisFunctionColumn, phi_B_product))
        denominator = 1 + np.dot(phi_B_product, basisFunctionColumn)
        return nominator / denominator


    """returns Basis function column"""
    def getBasisFunctionColumn(self, allPrev, allPrevID, allCurr, allCurrID):
        for i in range(len(allPrev)):
            basisFunctionColumn = np.zeros((self.n, 1))
            current_state = allPrev[i]
            currentAction_id = allPrevID[i]
            # only the rows corresponding to the current action are not zero
            basis = self.allFeatures(np.transpose(current_state))
            for i in range(self.numberOfFeatures):
                basisFunctionColumn[i + currentAction_id * (self.numberOfFeatures)] = basis[i]

            self.allBFC.append(basisFunctionColumn)

        basisFunctionColumn = np.zeros((self.n, 1))
        current_state = allCurr[len(allCurr) - 1]
        currentAction_id = allCurrID[len(allCurrID) - 1]
        # only the rows corresponding to the current action are not zero
        basis = self.allFeatures(np.transpose(current_state))
        for i in range(self.numberOfFeatures):
            basisFunctionColumn[i + currentAction_id * (self.numberOfFeatures)] = basis[i]

        self.allBFC.append(basisFunctionColumn)

    def reuse(self):
        self.m_B = self.m_B - self.reuse_B
        self.m_b = self.m_b - self.reuse_b
        self.w = np.dot(self.m_B, self.m_b)

    """returns parameters w"""
    def LSDTQ(self, data):

        allPrev = data[:, 1]
        allPrevID = data[:, 3]

        allCur = data[:, 0]
        allCurrID = data[:, 2]
        self.getBasisFunctionColumn(allPrev, allPrevID, allCur, allCurrID)
        for i in range(len(data)):
            if i % 1000 == 0:
                print(i)
            dt = data[i]
            reward = dt[4]
            bfc = self.allBFC[i]
            update_B = self.matrixBupdate(i)
            update_b = bfc * reward

            self.m_B = self.m_B - update_B
            self.m_b = self.m_b +  update_b

            #self.reuse_B += update_B
            #self.reuse_b += update_b

    """returns policy according to the LSPI algorithm"""
    def LSPI_algorithm(self, firstAction_id = 0, training_samples = 200, maxTimeSteps = 2000):
        doneActions = 0
        data = []
        while doneActions < 50000:


            obs = self.environment.reset()
            current_state =obs
            currentAction_id = randint(0, self.numberOfActions - 1)
            for t in range(maxTimeSteps):
                previous_state = current_state
                previousAction_id = currentAction_id

                obs, reward, done, _ = self.environment.step(np.array([self.actions[previousAction_id]]))

                doneActions += 1

                #self.environment.render()
                current_state = obs

                if done:
                    break

                currentAction_id = randint(0, self.numberOfActions - 1)
                dt = [current_state, previous_state, currentAction_id, previousAction_id, reward]
                data.append(dt)




        data = np.array(data)
        return data


    def load(self):
        file = open("cartpoleStabparams.txt", "r")
        for i in range(len(self.w)):
            x = file.readline()
            self.w[i] = float(x)
        file.close()


        file = open("cartpoleFourierFreq.txt", "r")
        for i in range(self.numberOfFeatures * self.state_dim):
            if i == 0:
                dt = [0,0,0,0,0]
                index = 0
            if i == (self.numberOfFeatures * self.state_dim -1):
                self.freq.append(dt)
            if i % 5 == 0 and i > 0:
                self.freq.append(dt)
                dt = [0,0,0,0,0]
                index = 0

            x = file.readline()
            dt[index] =float(x)
            index = index +1

        file.close()
        self.freq = np.array(self.freq)


        file = open("cartpoleFourierShift.txt", "r")
        for i in range(self.numberOfFeatures):
            x = file.readline()
            self.shift.append(float(x))
        file.close()
        self.shift = np.array(self.shift)

        self.allFeatures = self.get_feature_fun(self.state_dim, self.numberOfFeatures, self.bandwidth, True)

    def save(self):
        file = open("cartpoleParams.txt", "w")
        for i in range(len(self.w)):
            file.write(str(self.w[i][0]) + "\n")
        file.close()

        file = open("cartpoleFourierFreq.txt", "w")
        for i in range(len(self.freq)):
            for j in range(self.state_dim):
                file.write(str(self.freq[i][j]) + "\n")
        file.close()

        file = open("cartpoleFourierShift.txt", "w")
        for i in range(len(self.shift)):
            file.write(str(self.shift[i])+"\n")
        file.close()

    def learn(self):
        data = self.LSPI_algorithm()
        self.LSDTQ(data)
        self.w = np.dot(self.m_B, self.m_b)

    def learn_online(self):
        doneActions = 0
        data = []

        for i in range(100):
            obs = self.environment.reset()
            current_state =obs
            currentAction_id = self.returnBestAction(current_state)
            while not done:
                previous_state = current_state
                previousAction_id = currentAction_id
                obs, reward, done, _ = self.environment.step(np.array([self.actions[previousAction_id]]))
                doneActions += 1
                current_state = obs

                if done:
                    break

                currentAction_id =  self.returnBestAction(current_state)
                dt = [current_state, previous_state, currentAction_id, previousAction_id, reward]
                data.append(dt)

                if doneActions % 10 == 0:
                    doneActions = 0
                    data = np.array(data)
                    self.LSDTQ(data)
                    data = []
                    self.w = np.dot(self.m_B, self.m_b)






    def apply(self):
        print("start")
        allReward = 0
        doneActions = 0
        for n in range(50):
            obs = self.environment.reset()
            current_state = obs
            currentAction_id = self.returnBestAction(current_state)
            for t in range(1000):
                obs, reward, done, _ = self.environment.step(np.array([self.actions[currentAction_id]]))
                doneActions += 1
                allReward += reward
                if done:
                    break
                #self.environment.render()
                current_state = obs
                currentAction_id = self.returnBestAction(current_state)
        print(self.bandwidth)
        print(self.numberOfFeatures)
        print(allReward /doneActions)
        print(" ")
        return allReward / 50

env = gym.make('CartpoleStabShort-v0')  # Use "Cartpole-v0" for the simulation
env.reset()
def main():

    ctrl = SwingupCtrl(long=False)  # Use long=True if you are using the long pole
    env.step(np.array([0.]))
    env.close()
    nof = [ 200, 250, 275, 290, 305, 320,360]
    bandwidth = [6, 6, 6.5, 6.5, 7, 7]

    x = []
    y1 = []
    y2 = []
    y3 = []
    xd = LSPI(env, 300, 0.1, False)
    #xd.load()
    #xd.apply()
    xd.learn()
    xd.save()
    xd.apply()

    #for i in range(len(nof)):
        #print(i)

        #xd = LSPI(env, nof[i], bandwidth[0], False)
        #xd.learn()
        #valy = xd.apply()
        #y1.append(valy)

       # xd = LSPI(env, nof[i], bandwidth[1], False)
        #xd.learn()
        #valy = xd.apply()
        #y1.append(valy)

        #xd = LSPI(env, nof[i], bandwidth[2], False)
        #xd.learn()
        #valy = xd.apply()
        #y2.append(valy)

        #xd = LSPI(env, nof[i], bandwidth[3], False)
        #xd.learn()
        #valy = xd.apply()
        #y2.append(valy)

        #xd = LSPI(env, nof[i], bandwidth[4], False)
        #xd.learn()
        #valy = xd.apply()
        #y3.append(valy)

        #xd = LSPI(env, nof[i], bandwidth[5], False)
        #xd.learn()
        #valy = xd.apply()
        #y3.append(valy)

        #x.append(nof[i])


    #print(y1)
    #print(y2)
    #print(y3)
    #plt.scatter(x, y1)
    #plt.show()

    #plt.scatter(x, y2)
    #plt.show()

    #plt.scatter(x, y3)
    #plt.show()
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



