import gym
import gym_tidal_turbine
from rl_agents.ppo import run_experiment as run_ppo_experiment


if __name__ == '__main__':
    # Pendulum-v0 is considered solved if average reward is >= 180 over 100
    # consecutive trials
    # some benchmarks here: https://github.com/gouxiangchen/ac-ppo
    base_dir = 'experiments'

    ####
    # Experiment parameters
    ####
    # base dir is experiments/trials
    run_ppo_experiment(
        environment='WindTurbine-v2',
        n_iterations=600, iteration_size=8192,
        n_epochs=10, minibatch_size=128,
        gamma=0.99,
        actor_lr=1e-4,
        critic_lr=1e-4,
        actor_output_activation='tanh',
        base_dir=base_dir,
    )
