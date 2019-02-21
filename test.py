import gym
import gym_tidal_turbine

import numpy as np

env = gym.make('TidalTurbine-v0')

for _ in range(10):
    obs = env.reset()
    for _ in range(200):
        a = env.action_space.sample()
        obs, r, done, _ = env.step(np.array([action, action]))
        print(obs)
        if done:
            break
