import time

import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="rgb_array")
model = PPO("MultiInputPolicy", env, verbose=1)
start_time = time.time()
model.learn(total_timesteps=1000000)
model.save('model/PPO_pick_and_place.zip')
end_time = time.time()
with open('ppo_pick_and_place_train_time.txt', 'w') as opener:
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
