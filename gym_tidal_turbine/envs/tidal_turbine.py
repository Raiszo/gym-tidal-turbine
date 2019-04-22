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

        self.radius = 0.6 # Rotor radius
        self.area_blade = np.pi * self.radius ** 2
        
        # Combined inertia of turbine + rotor
        self.J = 1 # [kg-m2]
        # Viscous friction of the rotor
        self.B = 0 # []

        # Velocity of the wind/water
        # self.vel_w = None
        self.w_t = None

        self.minU = 0
        self.maxU = 100
        

        self.dt = 0.05

        high = np.array([
            np.finfo(np.float32).max, # w_m
            np.finfo(np.float32).max, # w_m_dot
        ])
        
        self.action_space = spaces.Box(
            low = np.array([ self.minU ]),
            high = np.array([ self.maxU ]),
            dtype = np.float32
        )
        # self.observation_space = spaces.Box(
        #     low = -high,
        #     high = high,
        #     dtype=np.float32
        # )

        self.__state__ = None

        self.reset()

        self.seed()
        self.viewer = None

    @property
    def tsr(self):
        """
        Also known as Lambda
        """
        return self.radius * self.w_m / self.v_w

    @property
    def w_m(self):
        return self.__state__[0]

    @property
    def Cp(self):
        return 0.095 + 0.0975 * self.tsr - 0.0075 * self.tsr ** 2

    @property
    def Pm(self):
        return 0.5 * self.Cp * self.rho * self.area_blade * self.v_w**3

    @property
    def Tm(self):
        return self.Pm / (self.w_m + 1e-3 )

    def step(self, action):
        # w_m, w_m_dot = self.__state__

        # sdot = np.array([
        #     w_m_dot,
        #     (self.Tm - action[0] - self.B*self.w_m) / self.J
        # ])
        # print(sdot)

        self.__state__ = (self.Tm - action - self.B*self.w_m) / self.J * self.dt + self.__state__

        # clamp it to only positive w_m
        done = self.__state__[0] <= 0.1

        return self.obs, 0, done, {}

    def reset(self):
        self.v_w = 0.81 # Mean current velocity
        initial_tsr = 4.5
        
        w_m_0 = initial_tsr * self.v_w / self.radius
        # self.__state__ = np.array([ w_m_0, 0 ])
        self.__state__ = np.array([ w_m_0 ])

        return self.obs

    @property
    def obs(self):
        return np.concatenate((
            self.__state__,
            np.array([ self.Tm, self.tsr ])
        ))
    

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer: self.viewer.close()
