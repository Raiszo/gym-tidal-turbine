from typing import Tuple

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


def plot_power(ax: Axes, t, y, t_max=None):
    ax.set_ylabel('Power [kW]')
    line_P = Line2D(t, y, color='blue')
    ax.add_line(line_P)
    if t_max:
        ax.set_xlim(0, t_max)
    ax.set_ylim(0, 2100)
    ax.grid(linestyle='--', linewidth=0.5)

def plot_omega(self, ax, t, y, t_max=None):
    ax.set_ylabel('Rotor speed [rpm]')
    line_omega = Line2D(t, y, color='blue')
    ax.add_line(line_omega)
    if t_max:
        ax.set_xlim(0, t_max)
    ax.set_ylim(0, 50)
    ax.grid(linestyle='--', linewidth=0.5)

def plot_torq(self, ax, t, y: Tuple[np.ndarray, np.ndarray], t_max=None):
    """
    x: x_t
    y: [ t_gen, t_aero ]
    """

    ax.set_ylabel('Torque [kNm]')
    line_gen_torq = Line2D(t, y[0], color='blue')
    line_aero_torq = Line2D(t, y[1], color='red')
    ax.add_line(line_gen_torq)
    ax.add_line(line_aero_torq)
    if t_max:
        ax.set_xlim(0, t_max)
    # ax.set_ylim(0.606, 47.403)
    ax.set_ylim(-1.0, 500.0)
    ax.grid(linestyle='--', linewidth=0.5)
    ax.legend((line_gen_torq, line_aero_torq),
              ('Gen. Torq', 'Aero. Torq'),
              loc='upper right', shadow=True)

def show_plots(states, actions, rewards, T: float, t_max: int):
    x_t = np.arange(0, t_max+T, T)
    y_t = states[:,1]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plot_power(ax, x_t, y_t, t_max)
    fig.savefig('power.png')
    

    # fg, ax = plt.subplots(figsize=[6.4, 4.8]*2)
    # fg, ax = plt.subplots()
    # plot_power(ax, x_t, states[:,1], t_max)
    # fg.suptitle('Power')
    # ax.set_xlabel('Time [s]')
    # fg.imshow()
