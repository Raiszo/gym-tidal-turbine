from gym.envs.registration import register

register(
    id='TidalTurbine-v0',
    entry_point='gym_tidal_turbine.envs:TidalTurbine',
    max_episode_steps=500
)

register(
    id='WindTurbine-v0',
    entry_point='gym_tidal_turbine.envs:WindTurbine',
    max_episode_steps=int(30.0/(1.0/20)),  # 30s -> 600 steps
    kwargs={
        'env_settings': {
            'timestep': 1.0/20.0,
            'duration': 30.0,
            'wind': {
                'mode': 'constant',
                'speed': 8.0,
            }
        }
    }
)
