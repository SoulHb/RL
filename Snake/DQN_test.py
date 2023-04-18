import os
import time
import gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3 import DQN
from SnakeEnv import SnakeEnv

video_folder = "videos/"
video_length = 100

model_path = 'D:\RL\models\model.zip'
env = SnakeEnv()
model = DQN.load(model_path, env)
for i in range(50):
    done = False
    obs = env.reset()
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)