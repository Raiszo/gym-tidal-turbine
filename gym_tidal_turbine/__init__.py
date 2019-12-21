from gym.envs.registration import register

register(
    id='TidalTurbine-v0',
    entry_point='gym_tidal_turbine.envs:TidalTurbine',
    max_episode_steps=500
)

register(
    id='WindTurbine-v0',
    entry_point='gym_tidal_turbine.envs:WindTurbine_v0',
    max_episode_steps=int(30.0/(1.0/20)),  # 30s -> 600 steps
    kwargs={
        'env_settings': {
            'timestep': 1.0/20.0,
            'duration': 30.0,
        }
    }
)

register(
    id='WindTurbine-v1',
    entry_point='gym_tidal_turbine.envs:WindTurbine_v1',
    max_episode_steps=int(1 * 60.0/(1.0/20)),  # 30s -> 600 steps
    kwargs={
        'env_settings': {
            'timestep': 1.0/20.0,
            'duration': 1 * 60.0,
        }
    }
)

register(
    id='WindTurbine-v2',
    entry_point='gym_tidal_turbine.envs:WindTurbine_v2',
    max_episode_steps=int(12 * 60.0/(1.0/20)),  # 30s -> 600 steps
    kwargs={
        'env_settings': {
            'timestep': 1.0/20.0,
            'duration': 12 * 60.0,
        }
    }
)
