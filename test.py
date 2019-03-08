import gym
import gym_tidal_turbine

import numpy as np

env = gym.make('TidalTurbine-v0')

for _ in range(1):
    obs = env.reset()
    for _ in range(20):
        a = env.action_space.sample()
        # print(a)
        obs, r, done, _ = env.step(a)
        print(obs, r, obs.shape)
        if done:
            break
