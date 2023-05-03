import os.path

import gymnasium as gym
import numpy as np
import torch

import envs
import agents
from utils.find_project_base_dir import find_project_base_dir

from discord_logging.handler import DiscordHandler
import python_telegram_logger
import logging.config


hyper_params = \
    agents.utils.HyperParams(lr=0.001, gamma=0.99, epsilon_begin=0.9, epsilon_end=0.05,
                             epsilon_decay=0.995, batch_size=128, num_episodes=10,
                             target_update=5, render=False, tau=0.005, max_steps=1000)

environment_id = 'CartPoleReward-v1'
model_saving_path = 'saved-models/cartpole'
random_seed = 21

if __name__ == '__main__':
    project_base_dir = find_project_base_dir()

    logging.config.fileConfig(os.path.join(project_base_dir, 'config.ini'),
                              defaults={'logfilename': os.path.join(project_base_dir, 'logs', 'train.log')
                              .replace('\\', '\\\\')})
    logger = logging.getLogger("TRAIN")

    render_mode = None
    if hyper_params.render:
        render_mode = 'human'

    env = gym.make(environment_id, render_mode=render_mode)

    logger.info(f'loading environments: {env.unwrapped.spec.id}')

    env.reset(seed=random_seed)
    np.random.seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'using device: {device}')

    model = agents.dqn.DQN(env, device, logger).to(device)
    logger.info(f'training start: {hyper_params}')
    model.train_with_episodes(hyper_params)

    # save model in the root/saved_models/cartpole
    model.save(project_base_dir + '/' + model_saving_path)

    logger.error('training ended')
