from argparse import ArgumentParser
from pathlib import Path
import math
import gym
import gym_tidal_turbine
import numpy as np
import tensorflow as tf
from rl_agents.ppo import GaussianSample, get_env, get_env_step, TFStep

from examples.plot_fns import show_plots

@tf.function
def run_episode(env_step: TFStep, initial_state: tf.Tensor, actor: tf.keras.Model, max_steps: int) -> float:
    initial_state_shape = initial_state.shape
    state_o = initial_state

    states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    reward_sum = 0.0

    for t in tf.range(1, max_steps + 1):
        state_no = tf.expand_dims(state_o, 0)
        action_na = actor(state_no).mean()


        action_a = tf.squeeze(action_na, axis=[0])
        state, reward, done = env_step(action_a)

        state.set_shape(initial_state_shape)

        # reward_sum += reward.astype(np.float32).item()

        # save values to TensorArray s
        states = states.write(t, state_o)
        actions = actions.write(t, action_a)
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    states = states.stack()
    actions = actions.stack()
    rewards = rewards.stack()

    return states, actions, rewards


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('actor_dir', help='path to actor tf saved model', type=Path)
    args = parser.parse_args()

    actor_dir = args.actor_dir

    env = get_env('WindTurbine-v2')

    initial_state = tf.constant(env.reset(), dtype=tf.float32)

    env_step = get_env_step(env)
    custom_objects={'GaussianSample': GaussianSample}
    t_max = 5 * 60
    T = 1/20
    with tf.keras.utils.custom_object_scope(custom_objects):
        actor = tf.keras.models.load_model(actor_dir)
        states, actions, rewards = run_episode(env_step, initial_state, actor, int(t_max / T))

        show_plots(states, actions, rewards, T, t_max)
