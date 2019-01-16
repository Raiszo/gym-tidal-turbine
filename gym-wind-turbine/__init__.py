from gym.envs.registration import register

register(
    id='WindTurbine-v0',
    entry_point='gym_pendrogone.envs:Drone_zero',
    timestep_limit=500
)
