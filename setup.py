from setuptools import setup

setup(
    name='gym_tidal_turbine',
    version='0.0.1',
    install_requires=[
        'gym',
        'numpy',
        'rl_agents @ git+https://github.com/Raiszo/rl-agents.git@master#egg=rl_agents',
        # 'CCBlade==1.1.1',
        # 'matplotlib==2.0.0'     # keep this version for now
    ]
)
