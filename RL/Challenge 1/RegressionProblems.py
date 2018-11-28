import gym
import quanser_robots.pendulum
import numpy as np
from profilehooks import timecall

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, BayesianRidge, SGDRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

#env = gym.make('Pendulum-v0')

def generate_data(train_size, environment='Pendulum-v2'):

    env = gym.make(environment)

    state = env.reset()

    rewards, next_states = [], []
    actions, states = [], []
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
    if model == 'rfr':
        clf = RandomForestRegressor(n_estimators=30)
    if model == 'gpr':
        clf = GaussianProcessRegressor()
    if model == 'gbr':
        clf = GradientBoostingRegressor()
    if model == 'sgd':
        clf = SGDRegressor
    if model == 'mlp':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        clf = MLPRegressor(solver='lbfgs', max_iter=200)
    clf.fit(X_train, y_train)

    #predictions = clf.predict(X_test)

    error = cross_validated_error(clf, X_train, y_train, 10)

    return np.round(error.mean(), 5), np.round(error.std(), 5)
    #print("Training error: {}, Test error: {}".format(training_error, test_error))


def average_error(size, model, degree=30, samples=6000, verbose=False):
    error, std = 0, 0
    errors, stds = [], []
    for i in range(size):
        mean, s = run_task_one(samples, model, degree)
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
        plt.title('model: {} n_samples: {}, mean error: {}, standard dev: {}'
              .format(model.upper(), 6000, np.round(error/size,6), np.round(std/size, 6)))
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
        mean, s = run_task_two(samples, 'rfr', alpha)
        errors.append(mean)
        stds.append(s)
        it += 1
        print("{} % finished (Iteration {}/{}...)".format(np.round(100 * it/n, 1), it, n))
    plt.plot(alphas, errors, label='cv error')
    plt.plot(alphas, stds, label='standard deviation')
    plt.title('Optimize alpha KRR  ({} data samples, degree: 8)'.format(samples))
    plt.legend()
    plt.show()

def optimize(model, low, high, n, samples):
    return

@timecall()
def sample_size_sampling(verbose=False):
    sample_sizes = np.linspace(500, 10000, 70)
    errors = []
    i = 0
    for sample_size in sample_sizes:
        mean, std = run_task_two(int(np.round(sample_size, 0)), 'mlp', 9)
        errors.append(mean)
        i += 1
        if verbose:
            print("Sampling {} % complete".format(np.round(100 * i / 100, 1)))
    plt.plot(sample_sizes, errors, label='cv error')
    plt.title('Elastic Net error for different sample sizes')
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
        clf = RandomForestRegressor(n_estimators=30)
    if model == 'gpr':
        clf = GaussianProcessRegressor()
    if model == 'gbr':
        clf = GradientBoostingRegressor()
    if model == 'mlp':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        clf = MLPRegressor(solver='lbfgs', max_iter=300)
    clf.fit(X_train, y_train)

    error = cross_validated_error(clf, X_train, y_train, 10)
    return np.round(error.mean(), 5), np.round(error.std(), 5)

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.empty((X.shape[0], X.shape[1]-1))
        for model in self.models_:
            predictions = np.add(predictions, model.predict(X))
        predictions = predictions / len(self.models_)
        return predictions

def test_averaging_models(samples, environment, learn_rewards=False):
    states, actions, rewards, next_states = generate_data(samples, environment)
    data = np.vstack((actions.T, states.T))
    data = data.T
    if learn_rewards:
        labels = rewards
    else:
        labels = next_states

    X_train, X_test, y_train, y_test = train_test_split(data, labels)

    clf_RFR = RandomForestRegressor(n_estimators=30)
    clf_GPR = GaussianProcessRegressor()
    clf_GBR = GradientBoostingRegressor()
    clf_MIX = AveragingModels(models=(clf_GPR, clf_RFR))

    clf_MIX.fit(X_train, y_train)
    clf_RFR.fit(X_train, y_train)
    clf_GPR.fit(X_train, y_train)
    clf_GBR.fit(X_train, y_train)

    MIX_error = cross_validated_error(clf_MIX, X_train, y_train, 10).mean()
    #RFR_error = cross_validated_error(clf_RFR, X_train, y_train, 10).mean()
    #GPR_error = cross_validated_error(clf_GPR, X_train, y_train, 10).mean()
    #SVR_error = cross_validated_error(clf_SVR, X_train, y_train, 20)

    print("Mixin error: {} \n RFR error: {} \n GPR error: {} \n"
          .format(MIX_error, 0, 0))

#test_averaging_models(5000, 'Pendulum-v0')
print(average_error(10,'mlp', verbose=True))
#print(sample_size_sampling(verbose=True))
#print(average_error(10, 'rfr', verbose=True))
#improve_alpha(10, 100, 90, 3000)
#optimize_degree(1,10,10,3000)
# TODO: Create a nice table of the dataset in pandas including labels
# TODO: Use other methods: e.g. XGBoost, RandomForestRegression
# TODO: (Use a Neural Network)
# TODO: Use Regression Mixins

# EXTERNAL METHODS:
def fit_state_model(environment):
    if environment == 'Pendulum-v2':
        states, actions, rewards, next_states = generate_data(10000, environment)
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
    if environment == 'Pendulum-v2':
        states, actions, rewards, next_states = generate_data(10000, environment)
        clf = MLPRegressor(solver='lbfgs', max_iter=300)
    else:
        return -1
    data = np.vstack((actions.T, states.T))
    data = data.T
    labels = rewards
    X_train, X_test, y_train, y_test = train_test_split(data, labels)

    clf.fit(X_train, y_train)
    return clf
