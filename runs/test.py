import os

import gymnasium as gym
import numpy as np
import torch

import envs
import agents
from utils.find_project_base_dir import find_project_base_dir

import ast
from discord_logging.handler import DiscordHandler
import python_telegram_logger
import logging.config
from utils.save_frames_as_gif import save_frames_as_gif

num_run = 1

render_as_gif = True
environment_id = 'CartPoleReward-v1'
model_saving_path = 'saved-models\\cartpole\\2023-05-02-11-06-39-CartPoleReward-v1.pth'
git_saving_path = 'logs\\2023-05-02-11-06-39-CartPoleReward-v1.pth.gif'
random_seed = 21

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

    env = gym.make(environment_id, render_mode=render_mode)
    logger.info(f'loading environments: {env.unwrapped.spec.id}')

    env.reset(seed=random_seed)
    np.random.seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'using device: {device}')

    model = agents.dqn.DQN(env, device, logger).to(device)
    model.load(os.path.join(project_base_dir, model_saving_path))
    logger.info(f'testing start. running {num_run} times. model with {model_saving_path}')

    if render_as_gif:
        frames = []
    else:
        frames = None

    model.test_with_episodes(num_run, frames=frames)
    save_frames_as_gif(os.path.join(project_base_dir, git_saving_path), frames)

    logger.error('test ended')

