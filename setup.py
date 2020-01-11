from setuptools import setup

setup(
    name='gym_tidal_turbine',
    version='0.0.1',
    install_requires=[
        'gym',
        'numpy',
        # 'CCBlade==1.1.1',
        'matplotlib==2.0.0'     # keep this version for now
    ]
)
