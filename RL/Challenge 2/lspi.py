import numpy as np
import math
from random import randint

class LSPI:
    """epsilon: allowed margin of error such that p' = p
       n: Amount of basis functions
       exploration_rate: chance of exploring instead of exploiting"""

    def __init__(self, environment, noF, bandwidth, discount_factor=0.95):
        self.environment = environment
        self.numberOfFeatures = noF
        self.m_discount_factor = discount_factor


        self.actions = [-24, 24]
        # discretized action space as i dont know yet how to deal with continous ones

        self.numberOfActions = len(self.actions)
        self.n = self.numberOfActions * self.numberOfFeatures
        # im not sure if the first B matrix is the first or the second one since the paper states that B is the inverse of A
        # self.m_B = np.zeros((n, n)) + 0.01 * np.ones((n,n))
        self.m_B = 100000 * np.identity((self.n))
        self.m_b = np.zeros((self.n, 1))
        self.w = np.zeros((self.n, 1))

        self.allBFC = []
        self.allFeatures = self.get_feature_fun(5, self.numberOfFeatures, bandwidth)

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
                summed_values[j] += self.w[i + j * self.numberOfFeatures] * basis[i]

        best_action_index = 0
        for i in range(len(summed_values)):
            if (summed_values[i] > summed_values[best_action_index]): best_action_index = i
        return np.array(self.actions[best_action_index])

    """returns update step of LSTDQ-OPT algorithm"""

    def matrixBupdate(self, i):

        basisFunctionColumn = self.allBFC[i]
        nextStateBasisFunctionColumn = self.allBFC[i + 1]
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
            self.m_B = self.m_B - self.matrixBupdate(i)
            self.m_b = self.m_b + bfc * reward



    def LSPI_algorithm(self, maxTimeSteps=2000):
        doneActions = 0
        data = []
        while doneActions < 50000:

            obs = self.environment.reset()
            current_state = obs
            currentAction_id = randint(0, self.numberOfActions - 1)
            for t in range(maxTimeSteps):
                previous_state = current_state
                previousAction_id = currentAction_id

                obs, reward, done, _ = self.environment.step(np.array([self.actions[previousAction_id]]))

                doneActions += 1

                # self.environment.render()
                current_state = obs

                if done:
                    break

                currentAction_id = randint(0, self.numberOfActions - 1)
                dt = [current_state, previous_state, currentAction_id, previousAction_id, reward]
                data.append(dt)

        data = np.array(data)
        return data

    def learn(self):
        data = self.LSPI_algorithm()
        self.LSDTQ(data)
        self.w = np.dot(self.m_B, self.m_b)
