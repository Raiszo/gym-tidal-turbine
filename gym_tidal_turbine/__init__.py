from gym.envs.registration import register

register(
    id='TidalTurbine-v0',
    entry_point='gym_tidal_turbine.envs:TidalTurbine',
    max_episode_steps=500
)
