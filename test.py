import gym
import gym_tidal_turbine

import numpy as np

env = gym.make('TidalTurbine-v0')

for _ in range(1):
    obs = env.reset()
    for _ in range(20):
        a = env.action_space.sample()
        obs, r, done, _ = env.step([0])
        print(obs)
        if done:
            break
