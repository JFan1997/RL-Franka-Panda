import gymnasium as gym
from stable_baselines3 import PPO
import panda_mujoco_gym
import time

env = gym.make("FrankaSlideSparse-v0", render_mode="rgb_array")
model = PPO("MultiInputPolicy", env, verbose=1)
start_time = time.time()
model.learn(total_timesteps=1000000)
model.save('model/PPO_slide.zip')
end_time = time.time()
with open('ppo_slide_train_time.txt', 'w') as opener:
    opener.write('spend_tine:{}'.format(end_time - start_time))

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
