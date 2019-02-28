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
    def __init__(self, environment, noF, bandwidth, exploration_rate = 1, discount_factor = 0.99, epsilon = 0.0000001, loaded = False):
        self.environment = environment
        self.numberOfFeatures = noF
        self.m_discount_factor = discount_factor
        self.m_epsilon = epsilon
        self.exploration_rate = exploration_rate
        self.exploration_decy_rate = 0.9999

        self.lastDecision = 1
        self.currentDecision = 1
        self.lastAction = 0
        self.currentAction = 0
        self.resolution = 100
        self.delta_min = 1
        self.delta_max = self.resolution
        self.delta = self.delta_min
        self.maxA = 6
        self.minA = -6
        self.preComp = (self.maxA - self.minA) / (self.resolution - 1)
        #discretized action space as i dont know yet how to deal with continous ones
        self.actions = [-1, 1]

        self.numberOfActions = len(self.actions)
        self.n = self.numberOfActions * self.numberOfFeatures
        #im not sure if the first B matrix is the first or the second one since the paper states that B is the inverse of A
        #self.m_B = np.zeros((n, n)) + 0.01 * np.ones((n,n))
        self.m_B = 100000*np.identity((self.n))
        self.m_b = np.zeros((self.n, 1))
        self.w = np.zeros((self.n, 1))

        self.bandwidth = bandwidth
        self.allBFC = []
        self.OnlineBFC = []
        self.state_dim = 6
        self.freq = []
        self.shift = []

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


    def contAction(self):
        if self.lastDecision + self.currentDecision == 0:
            self.delta = self.delta - 1
        else:
            self.delta = self.delta + 1

        if self.delta > self.delta_max:
            self.delta = self.delta_max
        elif self.delta < self.delta_min:
            self.delta = self.delta_min

        self.currentAction = self.lastAction + (self.currentDecision * self.delta * self.preComp)

        if self.currentAction > self.maxA:
            self.currentAction = self.maxA
        elif self.currentAction < self.minA:
            self.currentAction = self.minA

        self.lastAction = self.currentAction
        self.lastDecision = self.currentDecision

        return self.currentAction

    def reset(self):
        self.lastDecision = 1
        self.currentDecision = 1
        self.lastAction = 0
        self.currentAction = 0
        self.delta = self.delta_min
        
    """transforms five dimensional observation into four dimensional state"""
    def obsToState(self, obs, act):
        if (obs[1] <= 0):
            theta =  -np.arccos(obs[2])
        else:
            theta = np.arccos(obs[2])

        #x, theta, x_dot, theta_dot
        state = np.array([obs[0], obs[1], obs[2], obs[3], obs[4], act])
        return state

    """returns update step of LSTDQ-OPT algorithm"""
    def matrixBupdate(self, i, online):
        if not online:
            bfc_array = self.allBFC
        else:
            bfc_array = self.OnlineBFC

        basisFunctionColumn = bfc_array[i]

        nextStateBasisFunctionColumn = bfc_array[i+1]

        phi_B_product = np.dot((basisFunctionColumn - self.m_discount_factor * nextStateBasisFunctionColumn).T,
                               self.m_B)

        nominator = np.dot(np.dot(self.m_B, basisFunctionColumn), phi_B_product)
        denominator = 1 + np.dot(phi_B_product, basisFunctionColumn)
        return nominator / denominator

    """returns Basis function column"""
    def getBasisFunctionColumn(self, allPrev, allPrevID, allCurr, allCurrID, online):
        for i in range(len(allPrev)):
            basisFunctionColumn = np.zeros((self.n, 1))
            current_state = allPrev[i]
            currentAction_id = allPrevID[i]
            # only the rows corresponding to the current action are not zero
            basis = self.allFeatures(np.transpose(current_state))
            for j in range(self.numberOfFeatures):
                basisFunctionColumn[j + currentAction_id * (self.numberOfFeatures)] = basis[j]

            if not online:
                self.allBFC.append(basisFunctionColumn)
            else:
                self.OnlineBFC.append(basisFunctionColumn)

        basisFunctionColumn = np.zeros((self.n, 1))
        current_state = allCurr[len(allCurr) - 1]
        currentAction_id = allCurrID[len(allCurrID) - 1]
        # only the rows corresponding to the current action are not zero
        basis = self.allFeatures(np.transpose(current_state))
        for i in range(self.numberOfFeatures):
            basisFunctionColumn[i + currentAction_id * (self.numberOfFeatures)] = basis[i]

        if not online:
            self.allBFC.append(basisFunctionColumn)
        else:
            self.OnlineBFC.append(basisFunctionColumn)

    def collectData(self, training_samples = 3000, maxTimeSteps = 1000):
        doneActions = 0
        allData = []
        while doneActions < training_samples:
            self.reset()
            obs = self.environment.reset()
            current_state = self.obsToState(obs, self.environment.action_space.sample())
            while doneActions < training_samples:
                prev_state = current_state
                self.currentAction= self.environment.action_space.sample()
                obs, reward, done, _ = self.environment.step(np.array(self.currentAction))
                current_state = self.obsToState(obs, self.currentAction)
                doneActions += 1

                if self.currentAction > 0:
                    dec = 1
                else:
                    dec = 0
                data = [current_state, prev_state, dec, reward]
                allData.append(data)
                if done:
                    break
        return allData


    """returns parameters w"""
    def LSDTQ(self, data, online = False):

        allPrev = data[:, 1]
        allPrevID = data[:, 3]

        allCur = data[:, 0]
        allCurrID = data[:, 2]
        self.getBasisFunctionColumn(allPrev, allPrevID, allCur, allCurrID, online)
        if not online:
            bfc_array = self.allBFC
        else:
            bfc_array = self.OnlineBFC
        for i in range(len(data)):
            #if i % 1000 == 0:
                #print(i)
            dt = data[i]
            reward = dt[4]
            bfc = bfc_array[i]

            update_B = self.matrixBupdate(i, online)
            update_b = bfc * reward

            self.m_B = self.m_B - update_B
            self.m_b = self.m_b +  update_b


        self.OnlineBFC = []

    def load(self):
        file = open("cartpoleParams.txt", "r")
        for i in range(len(self.w)):
            x = file.readline()
            self.w[i] = float(x)
        file.close()


        file = open("cartpoleFourierFreq.txt", "r")
        for i in range(self.numberOfFeatures * self.state_dim):
            if i == 0:
                dt = [0,0,0,0,0,0]
                index = 0
            if i == (self.numberOfFeatures * self.state_dim -1):
                self.freq.append(dt)
            if i % 6 == 0 and i > 0:
                self.freq.append(dt)
                dt = [0,0,0,0,0, 0]
                index = 0

            x = file.readline()
            print(x)
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


    """returns policy according to the LSPI algorithm"""

    def LSPI_algorithm(self, firstAction_id=0, training_samples=200, maxTimeSteps=2000):
        doneActions = 0
        data = []
        while doneActions < 3000:
            self.reset()
            obs = self.environment.reset()
            current_state = obs
            currentAction_id = randint(0, self.numberOfActions - 1)

            self.currentDecision = self.actions[currentAction_id]
            act = self.contAction()

            f = 0
            for t in range(maxTimeSteps):
                if f == 0:
                    f = 1
                    current_state = self.obsToState(obs, act)

                previous_state = current_state
                previousAction_id = currentAction_id


                prev_act = act

                obs, reward, done, _ = self.environment.step(np.array(prev_act))
                doneActions += 1

                # self.environment.render()


                if done:
                    break

                currentAction_id = randint(0, self.numberOfActions - 1)
                self.currentDecision = self.actions[currentAction_id]
                act = self.contAction()
                current_state = self.obsToState(obs, act)

                dt = [current_state, previous_state, currentAction_id, previousAction_id, reward]
                data.append(dt)

        data = np.array(data)
        return data

    def learn(self):
        data = self.LSPI_algorithm()
        self.LSDTQ(data)
        self.w = np.dot(self.m_B, self.m_b)



    def apply(self):
        allReward = 0
        doneActions = 0
        for n in range(50):
            self.reset()
            obs = self.environment.reset()
            current_state = self.obsToState(obs, self.lastAction)
            currentAction_id = self.returnBestAction(current_state)
            self.currentDecision = self.actions[currentAction_id]
            act = self.contAction()
            for t in range(1000):
                obs, reward, done, _ = self.environment.step(np.array(act))
                doneActions += 1
                allReward += reward
                if done:
                    break
                self.environment.render()
                current_state = self.obsToState(obs, self.lastAction)
                currentAction_id = self.returnBestAction(current_state)
                self.currentDecision = self.actions[currentAction_id]
                act = self.contAction()
                #print(act)
        return allReward / doneActions

env = gym.make('CartpoleStabShort-v0')  # Use "Cartpole-v0" for the simulation
env.reset()
def main():

    ctrl = SwingupCtrl(long=False)  # Use long=True if you are using the long pole
    env.step(np.array([0.]))
    env.close()
    #old bw 5.2
    xd = LSPI(env, 320, 3, loaded = True)
    #xd.learn()
    #xd.save()
    print("end")
    xd.load()
    xd.apply()
    x = []
    y = []
    max = 0
    for i in range(30):
        i+=1
        print(i)
        x.append(i*3000)
        xd.learn()
        val = xd.apply()
        y.append(val)
        if val > max:
            max = val
            xd.save()
    plt.plot(x, y)
    plt.show()
    #xd.LSPI_algorithm()
    #xd.learn()
    #xd.apply()
    #print("pog")
    ##xd2 = LSPI(env, 200, 5.2)
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



