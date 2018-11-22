import gym
import numpy as np
from profilehooks import timecall

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor

#env = gym.make('Pendulum-v0')

def generate_data(train_size, environment='Pendulum-v0'):

    env = gym.make(environment)

    state = env.reset()

    rewards = []
    actions = []
    states = []
    next_states = []
    for i in range(train_size):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        states.append(state)
        next_states.append(next_state)
        if done:
            state = env.reset()
        else:
            state = next_state
    #reward_data = np.vstack((actions.T, states.T))

    return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)

def run_task_one(train_size, model, param):
    states, actions, rewards, next_states = generate_data(train_size)
    data = np.vstack((actions.T, states.T))
    #data = np.vstack((states.T, actions.T))
    data = data.T
    labels = rewards
    X_train, X_test, y_train, y_test = train_test_split(data, labels)

    if model == 'svr':
        clf = SVR(gamma='scale')
    if model == 'krr':
        clf = KernelRidge(alpha=param, kernel='polynomial', degree=8)
    clf.fit(X_train, y_train)

    #predictions = clf.predict(X_test)

    error = cross_validated_error(clf, X_train, y_train, 10)

    return np.round(error.mean(), 5), np.round(error.std(), 5)
    #print("Training error: {}, Test error: {}".format(training_error, test_error))


def average_error(size, model, degree=8, verbose=False):
    error, std = 0, 0
    errors, stds = [], []
    for i in range(size):
        mean, s = run_task_two(2000, model, degree)
        error += mean
        std += s
        errors.append(mean)
        stds.append(s)
        if verbose:
            print("Averaging {} % complete".format(np.round(100 * i/size, 1)))
    if verbose:
        plt.plot([i for i in range(size)], errors, label="mean cv error")
        plt.plot([i for i in range(size)], stds, label='standard deviation')
        plt.legend()
        plt.title('Polynomial Kernel degree: {}, mean error: {}, standard dev: {}'
              .format(9, np.round(error/size,6), np.round(std/size, 6)))
        plt.show()
    return error/size, std/size

def cross_validated_error(model, X_train, y_train, n_folds):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf))
    return rmse

# All values for alpha yield a pretty high RMSE of around 4
@timecall()
def improve_alpha(low, high, n, samples):
    alphas = np.linspace(low, high, n)
    errors, stds = [], []
    it = 0
    for alpha in alphas:
        mean, s = run_task_one(samples, 'krr', alpha)
        errors.append(mean)
        stds.append(s)
        it += 1
        print("{} % finished (Iteration {}/{}...)".format(np.round(100 * it/n, 1), it, n))
    plt.plot(alphas, errors, label='cv error')
    plt.plot(alphas, stds, label='standard deviation')
    plt.title('Optimize alpha KRR  ({} data samples, degree: 8)'.format(samples))
    plt.legend()
    plt.show()

def optimize_degree(low, high, n, samples):
    degrees = np.linspace(low, high, n)

    errors, stds = [], []
    it = 0
    for degree in degrees:
        mean, s = run_task_two(samples, 'krr', degree)
        errors.append(mean)
        stds.append(s)
        it += 1
        print("{} % finished (Iteration {}/{}...)".format(np.round(100 * it / n, 1), it, n))
    plt.plot(degrees, errors, label='cv error')
    plt.plot(degrees, stds, label='standard deviation')
    plt.title('Optimize degree KRR (state approximation)  ({} data samples)'.format(samples))
    plt.legend()
    plt.show()

@timecall()
def sample_size_sampling(verbose=False):
    sample_sizes = np.linspace(500, 10000, 90)
    errors = []
    i = 0
    for sample_size in sample_sizes:
        mean, std = run_task_two(int(np.round(sample_size, 0)), 'rfr', 9)
        errors.append(mean)
        i += 1
        if verbose:
            print("Sampling {} % complete".format(np.round(100 * i / 100, 1)))
    plt.plot(sample_sizes, errors, label='cv error')
    plt.title('RFR error for different sample sizes')
    plt.legend()
    plt.show()


def run_task_two(size, model, param):
    states, actions, rewards, next_states = generate_data(size)
    data = np.vstack((actions.T, states.T))
    data = data.T
    labels = next_states

    X_train, X_test, y_train, y_test = train_test_split(data, labels)

    if model == 'svr':
        clf = SVR(gamma='scale')
    if model == 'krr':
        clf = KernelRidge(alpha=0.6, kernel='polynomial', degree=param)
    if model == 'rfr':
        clf = RandomForestRegressor(n_estimators=50)
    clf.fit(X_train, y_train)

    error = cross_validated_error(clf, X_train, y_train, 10)

    return np.round(error.mean(), 5), np.round(error.std(), 5)

#print(average_error(200, 'svr', True))
#print(sample_size_sampling(verbose=True))
#print(average_error(500, 'krr', True))
#improve_alpha(0.001, 1, 200, 2000)
#optimize_degree(1,10,10,3000)
# TODO: Create a nice table of the dataset in pandas including labels
# TODO: Use other methods: e.g. XGBoost, RandomForestRegression
# TODO: (Use a Neural Network)
# TODO: Use Regression Mixins

# EXTERNAL METHODS:
def fit_state_model(environment):
    if environment == 'Pendulum-v0':
        states, actions, rewards, next_states = generate_data(10000, environment)
        #clf = KernelRidge(alpha=0.5, kernel='polynomial', degree=9)
        clf = RandomForestRegressor(n_estimators=30)
    else:
        return -1
    data = np.vstack((actions.T, states.T))
    data = data.T
    labels = next_states
    X_train, X_test, y_train, y_test = train_test_split(data, labels)

    clf.fit(X_train, y_train)
    return clf

def fit_reward_model(environment):
    if environment == 'Pendulum-v0':
        states, actions, rewards, next_states = generate_data(10000, environment)
        #clf = KernelRidge(alpha=0.5, kernel='polynomial', degree=8)
        clf = RandomForestRegressor(n_estimators=30)
    else:
        return -1
    data = np.vstack((actions.T, states.T))
    data = data.T
    labels = rewards
    X_train, X_test, y_train, y_test = train_test_split(data, labels)

    clf.fit(X_train, y_train)
    return clf
