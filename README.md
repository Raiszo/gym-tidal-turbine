# gym-tidal-turbine
GYM environment for RL testing

### How to test
#### test manual controllers, no RL

better to use python3
Be sure to have virtualenv installed, refer to this [manual](https://virtualenv.pypa.io/en/latest/installation/)

```bash
git clone https://github.com/Raiszo/gym-tidal-turbine.git
cd gym-tidal-turbine
python3 -m virtualenv .venv
source .venv/bin/activate

pip install -e .

python test.py
```

### Notes
- when w_m is below 0.1, the environment is terminated
