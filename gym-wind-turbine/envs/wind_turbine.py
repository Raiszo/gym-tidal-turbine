import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class WindTurbine(gym.Env):
    T = 0.05
    
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : 1/T
    }

    def __init__(self):
        self.gravity = 9.81 #: [m/s2] acceleration
        self.mass = 0.5 #: [kg] mass
        self.Ixx = 0.00232
        self.arm_length = 0.1 # [m]
        self.arm_width = 0.02 # [m]
        self.height = 0.02 # [m]

        # max and min force for each motor
        self.maxF = 2 * self.mass * self.gravity
        self.minF = 0
        self.maxAngle = np.pi / 2
        self.dt = Drone.T

        high = np.array([
            1.0,
            1.0,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            # 1.0,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
        ])
        
        self.action_space = spaces.Box(
            low = np.array([self.minF, self.minF]),
            high = np.array([self.maxF, self.maxF]),
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

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer: self.viewer.close()
