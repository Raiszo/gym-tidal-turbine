import gym
import gym_tidal_turbine

import numpy as np

env = gym.make('WindTurbine-v2')
obs = env.reset()

done = False
# while not done:
for _ in range(40*20):
    # Tm_opt = k * obs[0] ** 2
    a = np.array([0.0])
    
    obs, r, done, _ = env.step(a)
    # print(obs)
    # print(obs, Tm_opt)

# print(obs)
env.render()
