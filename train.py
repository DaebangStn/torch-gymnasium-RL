import gymnasium as gym
import numpy as np
import torch

import envs
import agents

import logging
import logging.config
import json


if __name__ == '__main__':
    hyper_params = \
        agents.utils.HyperParams(lr=0.001, gamma=0.99, epsilon_begin=0.9, epsilon_end=0.05,
                                 epsilon_decay=0.995, batch_size=128, num_episodes=1000,
                                 target_update=5, render=False, tau=0.005, max_steps=1000)

    with open('logger.json', 'rt') as f:
        config = json.load(f)

    logging.config.dictConfig(config)
    logger = logging.getLogger('Train')

    render_mode = None
    if hyper_params.render:
        render_mode = 'human'

    env = gym.make('CartPoleReward-v1', render_mode=render_mode)

    logger.info(f'loading environments: {env.unwrapped.spec.id}')

    env.reset(seed=21)
    np.random.seed(21)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'using device: {device}')

    logger_model = logging.getLogger('DQN')

    model = agents.dqn.DQN(env, device, logger_model).to(device)
    logger.info(f'training start: {hyper_params}')
    model.train_with_episodes(hyper_params)
