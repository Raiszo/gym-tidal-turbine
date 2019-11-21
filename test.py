import gym
import gym_tidal_turbine

import numpy as np

env = gym.make('WindTurbine-v1')
obs = env.reset()

done = False
# while not done:
for _ in range(10):
    # Tm_opt = k * obs[0] ** 2
    a = np.array([0.0 * 0.05])
    
    obs, r, done, _ = env.step(a)
    # print(obs, Tm_opt)

# env.render()
