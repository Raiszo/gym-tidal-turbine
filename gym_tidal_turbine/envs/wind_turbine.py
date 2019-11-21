import gym
from gym import spaces
from ccblade.ccblade import CCAirfoil, CCBlade

import numpy as np
from pkg_resources import resource_filename
from os import path, makedirs

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D

from datetime import datetime


class WindTurbine(gym.Env):
    def __init__(self, env_settings):
        self.dt = env_settings['timestep']
        self.t_max = env_settings['duration']

        # Set Observation limits
        self.obs_space_limits = np.array([
            [0, 30],            # wind speed [m/s]
            [-10, 100000],      # aero power [kW]
            [0, 15],           # rotor speed omega [rpm]
            [-5000, 5000],      # aero torque [kNm]
        ])
        self.observation_space = spaces.Box(
            low=self.obs_space_limits[:, 0],
            high=self.obs_space_limits[:, 1]
        )

        # Set Action limits
        self.ac_space_limits = np.array([
            [-25.0, 25.0]       # gen torque rate [kNm/s]
        ]) * self.dt
        self.action_space = spaces.Box(
            low=self.ac_space_limits[:1],
            high=self.ac_space_limits[-1:],
        )


        self.Uinf = 8.0        # wind speed in the inf [m/s]
        self.nrel_5mw_drivetrain_param = {
            'N_gear': 97.0,              # 97:1 gear box ratio
            'I_rotor': 38759228.0,       # kg*m^2 rotor inertia
            'I_gen': 534.116             # kg*m^2 generator inertia
        }
        self.rotor = self._initialise_nrel_5mw()

    def reset(self):
        self.t = 0.0
        self.i = 0
        self.omega = 0.0
        self.next_omega = self.omega # rotor rotation speed [rpm/s]
        self.t_gen = 0.0
        self.pitch = 0.0

        x_t = np.arange(0, self.t_max + self.dt, self.dt)
        self.plot_vars = {
            'x_t': x_t,
            'P_aero': np.full(x_t.shape, np.nan),
            'omega': np.full(x_t.shape, np.nan),
            'T_aero': np.full(x_t.shape, np.nan),
            'T_gen': np.full(x_t.shape, np.nan),
            'rewards': np.full((x_t.shape[0], 3), np.nan),
        }

        obs = np.array([
            self.Uinf,
            0.0,
            self.omega,
            0.0,
        ])
        self.prev_observation = obs

        return obs

    def aero_evaluate(self, Uinf, omega, pitch):
        """
        Evaluates the wind and velocity conditions to get Power and Torque
        More info: http://wisdem.github.io/CCBlade/generated/ccblade.CCBlade.evaluate.html

        Uinf: wind velocity [m/s]
        omega: rotor speed [rad/s]
        pitch: blade pitch [deg]

        Returns
        p_aero: Aerodynamic power [kW]
        t_aero: Aerodynamic torque [kNm]
        """
        P, T, Q, _ = self.rotor.evaluate([Uinf], [self.omega], [self.pitch])

        return P[0]/1e3, Q[0]/1e3

    def step(self, action):
        # if self.action_space.contains(action):
        action = np.clip(action, self.ac_space_limits[:, 0], self.ac_space_limits[:, 1])
        self.t_gen += action[0]
        print('torque gen', self.t_gen)

        self.omega = self.next_omega
        omega_rad_s = self.omega * np.pi / 30
        

        p_aero, t_aero = self.aero_evaluate(self.Uinf, self.omega, self.pitch)
        print('+++')
        print('omega, p_aero, t_aero')
        print(self.omega, p_aero, t_aero)
        print('+++')
        # Gonna scale down the aerodynamic torq compare it with the generator torque
        t_aero = t_aero / self.nrel_5mw_drivetrain_param['N_gear']
        observation = np.array([self.Uinf, p_aero, self.omega, t_aero])
        print(observation)
        
        done = not self.observation_space.contains(observation)


        reward, rew_ctrl, rewards = 0.0, 0.0, np.array([0.0, 0.0, 0.0])

        P = p_aero
        (_, prev_P, _, _) = self.prev_observation
        P_weight = 1.0
        T_weight = 0.0
        ctrl_weight = 0.1

        
        rewards = np.array([
            1.0 * (P - prev_P), # From increasing the power
            - 0.1 * np.square(action).sum(), # Cost for control input
            0.05                             # alive bonus
        ])

        self.plot_vars['x_t'][self.i] = self.t
        self.plot_vars['P_aero'][self.i] = observation[1]
        self.plot_vars['omega'][self.i] = observation[2]
        self.plot_vars['T_aero'][self.i ] = observation[3]
        self.plot_vars['T_gen'][self.i] = self.t_gen
        self.plot_vars['rewards'][self.i, :] = rewards

        diff_omega = self._diff_omega(t_aero, self. t_gen, self.nrel_5mw_drivetrain_param) \
            * self.dt * 30/np.pi

        print('diff', diff_omega)

        self.next_omega += diff_omega
        self.t += self.dt
        self.i += 1
        self.prev_observation = observation


        return observation, rewards.sum(), done, {}
        
    def _diff_omega(self, t_aero, t_gen, drivetrain_param):
        """
        I_total = I_rotor + N_gear^2 * I_gen
        
        The differential equation:
        I_total * omega_dot = T_aero - N_gear * T_gen
        omega_dot = (T_aero - N_gear * T_gen) / I_total
        omega_dot = (T_aero - N_gear * T_gen) / (I_rotor + N_gear^2 * I_gen)
        :return: omega_dot [rad/s^2]
        """
        I_rotor = drivetrain_param['I_rotor']
        I_gen = drivetrain_param['I_gen']
        N_gear = drivetrain_param['N_gear']

        I_total = I_rotor + N_gear ** 2 * I_gen
        return (t_aero - N_gear * t_gen) / I_total

    def render(self, mode='human', close=False):
        self.plot_and_save()

    def _get_timestamp(self):
        return datetime.now().strftime('%Y%m%d%H%M%S')

    def _initialise_nrel_5mw(self):
        """
        Initialise NREL 5MW 
        
        Load NREL 5MW CCAirfoil data
        Based on CCBlade/test/test_ccblade.py
        
        :return: 
        """
        # geometry
        Rhub = 1.5
        Rtip = 63.0

        r = np.array(
            [2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
             28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
             56.1667, 58.9000, 61.6333])
        chord = np.array(
            [3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
             3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419])
        theta = np.array(
            [13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
             6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106])
        B = 3  # number of blades

        # atmosphere
        rho = 1.225
        mu = 1.81206e-5

        afinit = CCAirfoil.initFromAerodynFile  # just for shorthand
        basepath = resource_filename('gym_tidal_turbine',
                              path.join('data', '5MW_AFFiles'))

        # load all airfoils
        airfoil_types = [0] * 8
        airfoil_types[0] = afinit(path.join(basepath, 'Cylinder1.dat'))
        airfoil_types[1] = afinit(path.join(basepath, 'Cylinder2.dat'))
        airfoil_types[2] = afinit(path.join(basepath, 'DU40_A17.dat'))
        airfoil_types[3] = afinit(path.join(basepath, 'DU35_A17.dat'))
        airfoil_types[4] = afinit(path.join(basepath, 'DU30_A17.dat'))
        airfoil_types[5] = afinit(path.join(basepath, 'DU25_A17.dat'))
        airfoil_types[6] = afinit(path.join(basepath, 'DU21_A17.dat'))
        airfoil_types[7] = afinit(path.join(basepath, 'NACA64_A17.dat'))

        # place at appropriate radial stations
        af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

        af = [0] * len(r)
        for i in range(len(r)):
            af[i] = airfoil_types[af_idx[i]]

        tilt = -5.0
        precone = 2.5
        yaw = 0.0

        # create CCBlade object
        rotor = CCBlade(r, chord, theta, af, Rhub, Rtip, B, rho, mu,
                        precone, tilt, yaw, shearExp=0.2, hubHt=90.0)

        return rotor    

    def plot_and_save(self):
        time = self._get_timestamp()
        rout_dir = path.join('gwt_output',
                             'render_{}'.format(time))
        rout_filename = '{}.png'.format(time)
        rout_path = path.join(rout_dir, rout_filename)

        # Create render output directory
        try:
            makedirs(rout_dir)
        except OSError:
            if not path.isdir(rout_dir):
                raise

        # Plot
        fig, (ax_P,
              ax_omega,
              ax_torq,
              ax_reward) = plt.subplots(4, figsize=(8, 12), sharex='all', tight_layout=True,
                                        gridspec_kw={'height_ratios': [ 1, 1, 2, 2]})
        

        fig.suptitle('gym-wind-turbine')

        # Handy variables
        x_t = self.plot_vars['x_t']
        pvars = self.plot_vars
        

        ax_P.set_ylabel('Power [kW]')
        line_P = Line2D(x_t, pvars['P_aero'], color='black')
        ax_P.add_line(line_P)
        ax_P.set_xlim(0, self.t_max)
        ax_P.set_ylim(0, 2100)
        ax_P.grid(linestyle='--', linewidth=0.5)

        ax_omega.set_ylabel('Rotor speed [rpm]')
        line_omega = Line2D(x_t, pvars['omega'], color='black')
        ax_omega.add_line(line_omega)
        ax_omega.set_xlim(0, self.t_max)
        ax_omega.set_ylim(0, 15)
        ax_omega.grid(linestyle='--', linewidth=0.5)

        ax_torq.set_ylabel('Torque [kNm]')
        # print(self.y_gen_torq, self.y_aero_torq)
        line_gen_torq = Line2D(x_t, pvars['T_gen'], color='blue')
        line_aero_torq = Line2D(x_t, pvars['T_aero'], color='red')
        ax_torq.add_line(line_gen_torq)
        ax_torq.add_line(line_aero_torq)
        ax_torq.set_xlim(0, self.t_max)
        # ax_torq.set_ylim(0.606, 47.403)
        ax_torq.set_ylim(-30.0, 70.0)
        ax_torq.grid(linestyle='--', linewidth=0.5)
        ax_torq.legend((line_gen_torq, line_aero_torq),
                       ('Gen. Torq', 'Aero. Torq'),
                       loc='upper right', shadow=True)

        ax_reward.set_ylabel('Reward [units]')
        # print(x_t.shape, self.y_rewards[:, 0].shape)
        line_reward_0 = Line2D(x_t, pvars['rewards'][:, 0], color='green')
        line_reward_1 = Line2D(x_t, pvars['rewards'][:, 1], color='blue')
        line_reward_2 = Line2D(x_t, pvars['rewards'][:, 2], color='red')
        ax_reward.add_line(line_reward_0)
        ax_reward.add_line(line_reward_1)
        ax_reward.add_line(line_reward_2)
        ax_reward.set_xlim(0, self.t_max)
        #ax_reward.set_ylim(-200, 5600)
        ax_reward.set_ylim(-20, 20)
        ax_reward.grid(linestyle='--', linewidth=0.5)
        ax_reward.set_xlabel('Time [s]')
        ax_reward.legend((line_reward_0, line_reward_1, line_reward_2),
                         ('Power', 'Control', 'Alive'),
                         loc='upper right', shadow=True)

        # logger.info("Saving figure: {}".format(rout_path))
        plt.savefig(rout_path, dpi=72)
        # plt.close(fig)
        # plt.show()
