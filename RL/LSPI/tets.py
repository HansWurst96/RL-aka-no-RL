import gym
import numpy as np
env = gym.make('CartPole-v0')
current_obs = env.reset()
for _ in range(1000):
    env.render()
    old_obs = current_obs
    obs,bla, blub, xd = env.step(env.action_space.sample()) # take a random action
    current_obs = obs
    debug = np.zeros((5, 1))
    for i in range(len(current_obs)):
        debug[i] = current_obs[i] - old_obs[i]
    print(np.linalg.norm(debug))
