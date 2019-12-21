import gym
from gym import spaces

import numpy as np
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
            [-10, np.finfo(np.float32).max], # aero power [kW]
            [0, 50],            # rotor speed omega [rad/s]
            [-5000, np.finfo(np.float32).max], # aero torque [kNm]
        ])
        self.observation_space = spaces.Box(
            low=self.obs_space_limits[:, 0],
            high=self.obs_space_limits[:, 1]
        )

        # Set Action limits
        self.ac_space_limits = np.array([
            [-25.0, 25.0]       # gen torque rate [kNm/s]
        ]) * self.dt
        # self.action_space = spaces.Box(
        #     low=self.ac_space_limits[:1],
        #     high=self.ac_space_limits[-1:],
        #     # shape=(1,)
        # )
        self.action_space = spaces.Box(
            low=-15.0,
            high=15.0,
            shape=(1,)
        )

        self.Uinf = 11.0        # wind speed in the inf [m/s]
        self.drivetrain_param = {
            'N_gear': 104.494,    # 97:1 gear box ratio
            'I_rotor': 4456761.0, # [kg*m^2] rotor inertia
            'I_gen': 123.0,       # [kg*m^2] generator inertia
            'K_rotor': 45.52,     # [] rotor
            'K_gen': 0.4,         # [] generator
        }

        rho = 1.25              # [kg/m^3] air density
        R = 38.5                # [m] rotor radius
        self.rotor = self._initialize_rotor(rho, R)

    def reset(self):
        self.t = 0.0
        self.i = 0
        self.omega = 1.0
        self.next_omega = self.omega # rotor rotation speed [rad/s]
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

    def step(self, action):
        # if self.action_space.contains(action):
        # print(action)
        # action = np.clip(action, self.ac_space_limits[:, 0], self.ac_space_limits[:, 1])
        # print(action/100)
        self.t_gen += action[0] * 10
        self.t_gen = self.t_gen * (self.t_gen > 0)
        # print('torque gen', self.t_gen)

        self.omega = self.next_omega
        omega_rpm = self.omega * 30/np.pi
        

        p_aero, t_aero = self.rotor(self.Uinf, self.omega, self.pitch)
        # print('+++')
        # print('omega, p_aero, t_aero')
        # print(self.omega, p_aero, t_aero)
        # print('+++')
        observation = np.array([self.Uinf, p_aero/1e3, self.omega * 30/np.pi, t_aero/1e3])
        # Gonna scale down the aerodynamic torq compare it with the generator torque, later
        # print(observation)
        
        done = not self.observation_space.contains(observation)


        reward, rew_ctrl, rewards = 0.0, 0.0, np.array([0.0, 0.0, 0.0])

        P = p_aero/1e3
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
        self.plot_vars['T_gen'][self.i] = self.t_gen / 1e3
        self.plot_vars['rewards'][self.i, :] = rewards

        diff_omega = self._diff_omega(t_aero, self.t_gen, self.omega, self.drivetrain_param)
        # print(diff_omega)

        # print('diff', diff_omega)

        self.next_omega += diff_omega * self.dt
        # clamp this to positive, values to not overflow the C_p equation in the exponential
        # and do not let omega be zero :v
        self.next_omega = self.next_omega if (self.next_omega > 5e-1) else 5e-1
        self.t += self.dt
        self.i += 1
        self.prev_observation = observation


        return observation, rewards.sum(), done, {}
        
    def _diff_omega(self, t_aero, t_gen, omega, drivetrain_param):
        """
        I_total = I_rotor + N_gear^2 * I_gen
        K_total = K_rotor + N_gear^2 * K_gen
        
        The differential equation:
        I_total * omega_dot = T_aero - K_total * omega - N_gear * T_gen
        omega_dot = (T_aero - K_total * omega - N_gear * T_gen) / I_total
        :return: omega_dot [rad/s^2]
        """
        I_rotor = drivetrain_param['I_rotor']
        I_gen = drivetrain_param['I_gen']
        K_rotor = drivetrain_param['K_rotor']
        K_gen = drivetrain_param['K_gen']
        N_gear = drivetrain_param['N_gear']

        I_total = I_rotor + N_gear ** 2 * I_gen
        K_total = K_rotor + N_gear ** 2 * K_gen
        # print(t_aero, K_total * omega, N_gear * t_gen)
        # return (t_aero - K_total * omega - N_gear * t_gen) / I_total
        # print(t_gen, t_aero, N_gear * t_gen, (t_aero - N_gear * t_gen) / I_total)
        return (t_aero - N_gear * t_gen) / I_total

    def render(self, mode='human', close=False):
        self.plot_and_save()

    def _get_timestamp(self):
        return datetime.now().strftime('%Y%m%d%H%M%S')

    def _initialize_rotor(self, rho, R):
        relu = lambda x: x * (x > 0)
        def rotor(Uinf, omega, pitch):
            """
            Uinf can't be zero
            """
            lam = R * omega / Uinf # (4)
            m_inv = 1/(lam + 0.08*pitch) - 0.035/(pitch**3 + 1) # (3)
            C_p = 0.22 * (116*m_inv - 0.4*pitch - 5)*np.exp(-12.5*m_inv) # (2)
            # print(lam, m_inv, C_p)
            C_p = relu(C_p)
            P = 0.5 * rho * R**2 * C_p * Uinf**3
            Q = P/omega

            return P, Q

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
        ax_omega.set_ylim(0, 50)
        ax_omega.grid(linestyle='--', linewidth=0.5)

        ax_torq.set_ylabel('Torque [kNm]')
        # print(self.y_gen_torq, self.y_aero_torq)
        line_gen_torq = Line2D(x_t, pvars['T_gen'] * self.drivetrain_param["N_gear"], color='blue')
        line_aero_torq = Line2D(x_t, pvars['T_aero'], color='red')
        ax_torq.add_line(line_gen_torq)
        ax_torq.add_line(line_aero_torq)
        ax_torq.set_xlim(0, self.t_max)
        # ax_torq.set_ylim(0.606, 47.403)
        ax_torq.set_ylim(-1.0, 500.0)
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
        ax_reward.set_ylim(-10, 10)
        ax_reward.grid(linestyle='--', linewidth=0.5)
        ax_reward.set_xlabel('Time [s]')
        ax_reward.legend((line_reward_0, line_reward_1, line_reward_2),
                         ('Power', 'Control', 'Alive'),
                         loc='upper right', shadow=True)

        # logger.info("Saving figure: {}".format(rout_path))
        plt.savefig(rout_path, dpi=72)
        # plt.close(fig)
        # plt.show()
