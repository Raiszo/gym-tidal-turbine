from setuptools import setup

setup(
    name='gym_tidal_turbine',
    version='0.0.1',
    install_requires=[
        'gym',
        'numpy==1.22.0',
        # 'rl_agents @ git+https://github.com/Raiszo/rl-agents.git@master#egg=rl_agents',
        'matplotlib',
        'tensorflow==2.7.2',
        # 'CCBlade==1.1.1',
    ]
)
