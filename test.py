import gym
import gym_tidal_turbine

import numpy as np

env = gym.make('TidalTurbine-v0')
obs = env.reset()

k = 0.5 * 1014.0 * np.pi * 0.6 ** 5 * (0.41 / 6 ** 3)

for _ in range(100):
    # Controller here
    Tm_opt = k * obs[0] ** 2
    # print(Tm_opt)
    a = np.array([Tm_opt])
    
    obs, r, done, _ = env.step(a)
    print(obs, Tm_opt)
    if done:
        print('turbine stopped')
        break
