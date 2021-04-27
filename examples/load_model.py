import tensorflow as tf
from rl_agents.ppo import get_model, get_env

"""
Overview
- [] get windturbine environment
- [] get model, should use manually serialized architecture values if possible
- [] rollout loop, better to rely on tf by using get_env_step from ppo
- [] plot from tf-arrays, convert them to numpy and to matplotlib
"""

env = get_env('WindTurbine-v2')

# only valid for continuous action and observation spaces
# these values should be manually serialized
obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
actor, critic = get_model(obs_dim, act_dim, actor_output_activation='tanh')

