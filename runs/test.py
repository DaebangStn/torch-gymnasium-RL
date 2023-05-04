import os

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
import torch

import envs
import agents
from utils.find_project_base_dir import find_project_base_dir

import logging.config
from utils.save_frames_as_gif import save_frames_as_gif

num_run = 1

render_as_gif = True
#environment_id = 'CartPoleReward-v1'
environment_id = 'CartPoleDist-v1'

model_filename = '2023-05-03-22-27-49-CartPoleReward-v1.pth'
model_saving_dir = 'out/saved-models/cartpole'
gif_saving_dir = 'out/gifs'

random_seed = 20
default_max_steps = 1500

if __name__ == '__main__':
    project_base_dir = find_project_base_dir()

    logging.config.fileConfig(os.path.join(project_base_dir, 'config.ini'),
                              defaults={'logfilename': os.path.join(project_base_dir, 'logs', 'test.log')
                              .replace('\\', '\\\\')})
    logger = logging.getLogger('TEST')

    if render_as_gif:
        render_mode = "rgb_array"
    else:
        render_mode = "human"

    env = gym.make(environment_id, render_mode=render_mode, theta_threshold=90)
    env = TimeLimit(env, max_episode_steps=default_max_steps)
    logger.info(f'loading environments: {env.unwrapped.spec.id}')

    env.reset(seed=random_seed)
    np.random.seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f'using device: {device}')

    model = agents.dqn.DQN(env, device, logger).to(device)
    model_saving_path = os.path.join(project_base_dir, model_saving_dir, model_filename)
    model.load(model_saving_path)
    logger.info(f'testing start. running {num_run} times. model with {model_saving_path}')

    if render_as_gif:
        frames = []
    else:
        frames = None

    model.test_with_episodes(num_run, frames=frames)

    gif_saving_path = os.path.join(project_base_dir, gif_saving_dir, os.path.splitext(model_filename)[0] + '.gif')
    if render_as_gif:
        save_frames_as_gif(os.path.join(project_base_dir, gif_saving_path), frames, logger)

    logger.error('test ended')

