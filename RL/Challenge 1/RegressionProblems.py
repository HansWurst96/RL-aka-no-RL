import gym
import quanser_robots.pendulum
import numpy as np
#from profilehooks import timecall

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import sklearn.tree as tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
import  sklearn.neighbors as neigh
from sklearn.kernel_ridge import KernelRidge
import sklearn.gaussian_process.kernels as kernels
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
import sklearn.linear_model as lm
from sklearn.linear_model import ElasticNet, BayesianRidge, SGDRegressor, PassiveAggressiveRegressor, LassoLars, LogisticRegression, LinearRegression, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

#env = gym.make('Pendulum-v0')

def generate_data(train_size, environment='Pendulum-v2', gaussian=False):

    env = gym.make(environment)

    state = env.reset()

    rewards, next_states = [], []
    actions, states = [], []
    if gaussian:
        right_side = np.geomspace(0.001, env.action_space.high, int(np.round(train_size/2)))
        left_side = np.flip(-np.copy(right_side))
        left_side = np.append(left_side, 0)
        action_array = np.concatenate((left_side, right_side))
        for i in action_array:
            next_state, reward, done, info = env.step(np.asarray(i).reshape(1, -1))
            next_state = next_state.reshape((next_state.shape[0],))
            #print("gaussian", "ns", next_state.shape, "r", reward,"a", i,"s", state)
            rewards.append(reward)
            actions.append(i)
            states.append(state)
            next_states.append(next_state)
            if done:
                state = env.reset()
            else:
                state = next_state

    if not gaussian:
        for i in range(train_size):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            #print("not gaussian", next_state.shape, reward, action, state)
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
    if model == 'etr':
        clf = ExtraTreesRegressor(min_samples_split=3, criterion='mse', n_estimators=100, n_jobs=-1)
    if model == 'sgd':
        clf = SGDRegressor
    if model == 'log':
        clf = LogisticRegression()
    if model == 'mlp':
        scaler = StandardScaler()
        scaler.fit(data)
        X_train = scaler.transform(data)
        X_test = scaler.transform(X_test)
        clf = MLPRegressor(hidden_layer_sizes=(50,50,50), solver='lbfgs', max_iter=500, learning_rate='adaptive', alpha=0.05,activation='tanh')
    clf.fit(X_train, y_train)

    #predictions = clf.predict(X_test)

    error = cross_validated_error(clf, X_train, y_train, 10)

    return np.round(error.mean(), 5), np.round(error.std(), 5)
    #print("Training error: {}, Test error: {}".format(training_error, test_error))


def average_error(size, model, degree=(1), samples=10000, verbose=False):
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
              .format(model.upper(), 6000, np.round(error/size, 6), np.round(std/size, 6)))
        plt.show()
    return error/size, std/size

def cross_validated_error(model, X_train, y_train, n_folds):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf))
    return rmse

# All values for alpha yield a pretty high RMSE of around 4
#@timecall()

def get_error(model, samples, iterations, environnment='Pendulum-v2', is_rewards=False, verbose=False):
    errors, stds = [], []
    for i in range(iterations):
        states, actions, rewards, next_states = generate_data(samples, environment=environnment)
        data = np.vstack((actions.T, states.T))
        data = data.T
        if is_rewards:
            labels = rewards
        else:
            labels = next_states
        X_train, X_test, y_train, y_test = train_test_split(data, labels)
        model.fit(X_train, y_train)
        error = cross_validated_error(model, X_train, y_train, 5)
        errors.append(np.round(np.array(error).mean(), 4))
        stds.append(np.round(np.array(error).std(), 4))
        if verbose:
            print("Averaging {} % complete".format(np.round(100 * (i+1)/iterations, 1)))
    if verbose:
        plt.plot([i for i in range(iterations)], errors, label="mean cv error")
        plt.plot([i for i in range(iterations)], stds, label='standard deviation')
        #plt.text(0.3, 0.1, model)
        plt.legend()
        plt.title('n_samples: {}, mean error: {}, standard dev: {}'
              .format(samples, np.round(np.array(errors).mean(), 5), np.round(np.array(stds).mean(), 5)))
        plt.show()

def grid_search(model, grid, samples, env='Pendulum-v2', is_rewards=False):
    states, actions, rewards, next_states = generate_data(samples, environment=env)
    data = np.vstack((actions.T, states.T))
    # data = np.vstack((states.T, actions.T))
    data = data.T
    if is_rewards:
        labels = rewards
    else:
        labels = states
    X_train, X_test, y_train, y_test = train_test_split(data, labels)
    clf = GridSearchCV(model, grid, n_jobs=-1, cv=3)
    clf.fit(X_train, y_train)
    return clf.best_params_

mlp_grid = {
    'hidden_layer_sizes': [(50,50,50), (50,), (100,), (150,), (200,)],
    'activation': ['tanh', 'relu'],
    'max_iter': [200, 300, 400, 500],
    'alpha': [0.0001, 0.0002, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}
rfr_grid = {
    'n_estimators': [int(np.round(i)) for i in np.linspace(5, 100, 10)],
    'criterion': ['mse', 'mae'],
    'min_samples_split': [int(np.round(i)) for i in np.linspace(2, 5, 6)],
    'min_samples_leaf': [int(np.round(i)) for i in np.linspace(1, 5, 5)],
}
gpr_grid = {
    'normalize_y': [True, False],
    'alpha': np.linspace(1e-13, 1e-6, 10),
    'n_restarts_optimizer': [0, 1, 2, 3],
}
bayesian_grid = {
    'n_iter': np.linspace(200, 600, 15, dtype=int),
    'tol': [1e-4, 1e-3, 1e-2],
    'alpha_1': np.linspace(1e-8, 1e-4, 6),
    'alpha_2': np.linspace(1e-8, 1e-4, 6),
    'lambda_1': np.linspace(1e-8, 1e-4, 6),
    'lambda_2': np.linspace(1e-8, 1e-4, 6),
}
# SECTION: Results
########################################################################################################################
# Qube-states RFR {'min_samples_leaf': 1, 'n_estimators': 100, 'min_samples_split': 2, 'criterion': 'mse'} 10k: 0.75
# Qube-states MLP {'hidden_layer_sizes': (50,), 'activation': 'relu', 'alpha': 0.0002, 'max_iter': 500, 'learning_rate': 'adaptive'} 12k: 0.58
# Qube-rewards GPR {'alpha': 1e-06, 'normalize_y': True, 'n_restarts_optimizer': 0} 10k:


#get_error(GaussianProcessRegressor(normalize_y=True, alpha=1e-6), 10000, 5, 'Qube-v0',is_rewards=True, verbose=True)
#print(grid_search(GaussianProcessRegressor(), gpr_grid, 3000, 'Qube-v0', is_rewards=True))
########################################################################################################################

#@timecall()
def sample_size_sampling(verbose=False):
    sample_sizes = np.linspace(500, 10000, 50)
    errors = []
    i = 0
    for sample_size in sample_sizes:
        mean, std = run_task_two(int(np.round(sample_size, 0)), 'gpr', 9)
        errors.append(mean)
        i += 1
        if verbose:
            print("Sampling {} % complete".format(np.round(100 * i / 60, 1)))
    plt.plot(sample_sizes, errors, label='cv error')
    plt.title('GPR (states) error for different sample sizes')
    plt.legend()
    plt.show()


def run_task_two(size, model, param):
    states, actions, rewards, next_states = generate_data(size)
    data = np.vstack((states.T, actions.T))
    data = data.T
    labels = next_states

    X_train, X_test, y_train, y_test = train_test_split(data, labels)

    if model == 'svr':
        clf = SVR(gamma='scale')
    if model == 'krr':
        clf = KernelRidge(alpha=0.1, kernel='rbf')
    if model == 'rfr':
        clf = RandomForestRegressor(n_estimators=70)
    if model == 'etr':
        clf = ExtraTreesRegressor(min_samples_split=3, criterion='mse', n_estimators=100, n_jobs=-1)
    if model == 'gpr':
        clf = GaussianProcessRegressor(normalize_y=True, alpha=1e-12)
    if model == 'gbr':
        clf = GradientBoostingRegressor()
    if model == 'log':
        clf = LogisticRegression()
    if model == 'lr':
        #clf = tree.DecisionTreeRegressor('mae')
        clf = ExtraTreesRegressor(n_estimators=70, criterion='mae')
    if model == 'mlp':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        #clf = MLPRegressor(hidden_layer_sizes=[int(np.round(i)) for i in param], solver='lbfgs', max_iter=300)
        clf = MLPRegressor(hidden_layer_sizes=(50,100), learning_rate='adaptive', learning_rate_init=0.0001, solver='adam', max_iter=500)
    clf.fit(X_train, y_train)

    error = cross_validated_error(clf, data, labels, 10)
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
    clf_DTR = tree.DecisionTreeRegressor()
    clf_ETR = ExtraTreesRegressor(n_estimators=30, criterion='mae')
    clf_MIX = AveragingModels(models=(clf_DTR, clf_RFR, clf_ETR))

    #clf_test = RandomForestRegressor(n_estimators=30)

    clf_MIX.fit(X_train, y_train)
    #clf_test.fit(X_train, y_train)
    #clf_RFR.fit(X_train, y_train)
    #clf_GPR.fit(X_train, y_train)
    #clf_GBR.fit(X_train, y_train)

    MIX_error = cross_validated_error(clf_MIX, X_train, y_train, 10).mean()
    #test_error = cross_validated_error(clf_test, X_train, y_train, 10).mean()
    #RFR_error = cross_validated_error(clf_RFR, X_train, y_train, 10).mean()
    #GPR_error = cross_validated_error(clf_GPR, X_train, y_train, 10).mean()
    #SVR_error = cross_validated_error(clf_SVR, X_train, y_train, 20)

    print("Mixin error: {} \n RFR error: {} \n"
          .format(MIX_error, 9))

def test(samples, model, environment='Pendulum-v2', gaussian=False):
    states, actions, rewards, next_states = generate_data(samples, environment, gaussian)
    data = np.vstack((states.T, actions.T))
    data = data.T
    labels = next_states

    X_train, X_test, y_train, y_test = train_test_split(data, labels)
    model.fit(X_train, y_train)
    error = cross_validated_error(model, data, labels, 10)
    return np.round(error.mean(), 5), np.round(error.std(), 5)

#print("Non gaussian: ", test(10000,MLPRegressor(max_iter=5000), gaussian=False))
#print("Gaussian: ", test(10000,MLPRegressor(max_iter=5000, solver='lbfgs'), gaussian=False))

#print(sample_size_sampling(verbose=True))
#print(average_error(10, 'mlp', verbose=True))
#improve_alpha(3,5 , 40, 6000)
#optimize_degree(1,10,10,3000)
#test_averaging_models(8000, 'Pendulum-v2')
# TODO: Create a nice table of the dataset in pandas including labels
# TODO: Use Regression Mixins

# EXTERNAL METHODS:
def fit_state_model(environment):
    if environment == 'Pendulum-v2':
        states, actions, rewards, next_states = generate_data(10000, environment)
        clf = RandomForestRegressor(n_estimators=100, min_samples_split=2, criterion='mae', min_samples_leaf=1)
    if environment == 'Qube-v0':
        states, actions, rewards, next_states = generate_data(10000, environment)
        clf = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(50,), learning_rate='adaptive', max_iter=300)
    else:
        return -1
    data = np.vstack((actions.T, states.T))
    data = data.T
    labels = next_states
    X_train, X_test, y_train, y_test = train_test_split(data, labels)

    clf.fit(data, labels)
    return clf

def fit_reward_model(environment):
    if environment == 'Pendulum-v2':
        states, actions, rewards, next_states = generate_data(10000, environment)
        clf = GaussianProcessRegressor(normalize_y=True, alpha=1e-14)
        #clf = MLPRegressor(solver='lbfgs', max_iter=500,
        #                   hidden_layer_sizes=(50, 50, 50), learning_rate='adaptive', alpha=0.05, activation='tanh')
    if environment == 'Qube-v0':
        states, actions, rewards, next_states = generate_data(10000, environment)
        clf = GaussianProcessRegressor(normalize_y=True, alpha=1e-6)
    else:
        return -1
    data = np.vstack((actions.T, states.T))
    data = data.T
    labels = rewards
    X_train, X_test, y_train, y_test = train_test_split(data, labels)

    clf.fit(data, labels)
    return clf



