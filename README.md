## Installation

All essential libraries with corresponding versions are listed in [`requirements.txt`](requirements.txt).

## Test

### random_sampling

```bash
```python
import sys
import time
import gymnasium as gym
import panda_mujoco_gym

if __name__ == "__main__":
    env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human")

    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        time.sleep(0.2)

    env.close()

```
### PPO algorithm for training task pick and place

```bash
```python

import gymnasium as gym
from stable_baselines3 import PPO
import panda_mujoco_gym

env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human")
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()

```

## Citation

If you use this repo in your work, please cite:

```
@misc{xu2023opensource,
      title={Open-Source Reinforcement Learning Environments Implemented in MuJoCo with Franka Manipulator}, 
      author={Zichun Xu and Yuntao Li and Xiaohang Yang and Zhiyuan Zhao and Lei Zhuang and Jingdong Zhao},
      year={2023},
      eprint={2312.13788},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```