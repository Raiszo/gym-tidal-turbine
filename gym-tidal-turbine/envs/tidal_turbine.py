import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class TidalTurbine(gym.Env):
    T = 0.05

    
    def __init__(self):
        # Air/Water density
        self.rho = 1014.0
        # Blade pitch angle
        self.beta = 0.0

        self.area_blade = 12.0
        self.radius = 1.2 # Rotod diameter
        
        # Combined inertia of turbine + rotor
        self.J = 1 # [kg-m2]
        # Viscous friction of the rotor
        self.B = 0 # []

        # Wind velocity
        self.vel_w = None
        self.w_t = None
        # Mechanical angular speed of the turbine
        self.w_m = 0.0

        self.minU = - np.finfo(np.float32).max
        self.maxU = np.finfo(np.float32).max
        

        self.dt = WintTurbine.T

        high = np.array([
            np.finfo(np.float32).max, # w_m
            np.finfo(np.float32).max, # w_m_dot
        ])
        
        self.action_space = spaces.Box(
            low = np.array([ self.minU, self.minU ]),
            high = np.array([ self.maxU, self.maxU ]),
            dtype = np.float32
        )
        self.observation_space = spaces.Box(
            -high,
            high,
            dtype=np.float32
        )

        self.__state = None

        self.reset()

        self.seed()
        self.viewer = None

    @property
    def Cp(self):
        k = self.k_constants
        beta = self.beta

        # Tip Speed Ration, TSR
        lamb = self.radius * self.w_m
        lamb_i = 1/(lamb + 8e-2*beta) - 35e-3/(1 + beta**3)
        lamb_i = 1 / lamb_i

        cp = k[1] * (k[2]/lamb_i  - k[3]*beta - k[4]*beta^k[5]- k[6]) * np.exp(-k[7]/lamb_i)

        return cp

    @property
    def P_m(self):
        return 0.5 * self.Cp * self.rho * self.area_blade * self.v_w**3

    @property
    def T_m(self):
        return self.P_m / self.w_m

    @property
    def w_m(self):
        return self.__state[0]

    def _get_obs(self):
        pass

    def step(self, action):
        pass

    def _apply_action(self, u):
        sdot = np.array([
            w_m_dot,
            (u - self.T_m - self.friction*self.w_m) / self.J
        ])

        self.state = sdot * self.dt + self.state

        return 0

    def reset(self):
        self.__state = np.zeros(2)
        self.w_m = 0
        self.vel_w = 0
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer: self.viewer.close()
