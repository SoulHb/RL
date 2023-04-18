import os
import time
from stable_baselines3 import DQN
from SnakeEnv import SnakeEnv

TIMESTEPS = 1000000

models_dir = f"../models/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
   os.makedirs(models_dir)

if not os.path.exists(logdir):
   os.makedirs(logdir)

env = SnakeEnv()
env.reset()

model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN")
model.save(f"{models_dir}/model")
