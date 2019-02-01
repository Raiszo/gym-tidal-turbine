import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class WindTurbine(gym.Env):
    T = 0.05

    
    def __init__(self):
        """
        The data is from a 5MW wind turbine for offshore systema development
        """
        self.beta = 0.0
        self.alpha_0 = None

        self.area_blade = 12445.3
        self.radius = 63
        self.k_constants = [
            None,
            0.73,
            151.0,
            0.58,
            2e-3,
            2.14,
            13.2,
            18.4
        ]

        self.vel_w = None
        self.w_t = None
        

        self.dt = WintTurbine.T

        high = np.array([
        ])
        
        self.action_space = spaces.Box(
            low = np.array([]),
            high = np.array([]),
            dtype = np.float32
        )
        self.observation_space = spaces.Box(
            -high,
            high,
            dtype=np.float32
        )

        self.seed()
        self.viewer = None
        self.state = None

    @property
    def Cp(self):
        k = self.k_constants
        beta = self.beta
        
        lamb_i = 1/(self.lamb + 8e-2*beta) - 35e-3/(1 + beta**3)
        lamb_i = 1 / lamb_i

        cp = k[1] * (k[2]/lamb_i  - k[3]*beta - k[4]*beta^k[5]- k[6]) * np.exp(-k[7]/lamb_i)

        return cp

    def _get_obs(self):
        pass

    def step(self, action):
        pass

    def _apply_action(self, u):
        pass

    def reset(self):
        pass
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer: self.viewer.close()
