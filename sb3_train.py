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
