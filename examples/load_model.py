from argparse import ArgumentParser
import gym
import gym_tidal_turbine
import numpy as np
import tensorflow as tf
from rl_agents.ppo import GaussianSample, get_env, get_env_step, TFStep
from pathlib import Path
import os

@tf.function
def run_episode(env_step: TFStep, initial_state: tf.Tensor, actor: tf.keras.Model, max_steps: tf.constant) -> float:
    initial_state_shape = initial_state.shape
    state = initial_state

    reward_sum = 0.0

    for i in range(1, max_steps + 1):
        state = tf.expand_dims(state, 0)
        action_na = actor(state).mean()

        state, reward, done = env_step(tf.squeeze(action_na, axis=[0]))
        state.set_shape(initial_state_shape)
        # print(done)

        # reward_sum += reward.astype(np.float32).item()

        if tf.cast(done, tf.bool):
            break

    return reward_sum



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('actor_dir', help='path to actor tf saved model', type=Path)
    args = parser.parse_args()

    actor_dir = args.actor_dir

    env = get_env('WindTurbine-v2')
    initial_state = tf.constant(env.reset(), dtype=tf.float32)

    env_step = get_env_step(env)
    custom_objects={'GaussianSample': GaussianSample}
    with tf.keras.utils.custom_object_scope(custom_objects):
        actor = tf.keras.models.load_model(actor_dir)
        reward_sum = run_episode(env_step, initial_state, actor, tf.constant(5 * 60 * 20, dtype=tf.int32))
