import os.path

import gymnasium as gym
from gymnasium.wrappers import TimeLimit, AtariPreprocessing, FlattenObservation
import numpy as np
import torch

import envs
import agents
from utils.find_project_base_dir import find_project_base_dir
from plots.plot_list import plot_list_with_moving_mean as plmm

import logging.config

default_hyper_params = \
    agents.utils.HyperParams(lr=0.001, gamma=0.99, epsilon_begin=0.9, epsilon_end=0.05,
                             epsilon_decay=0.995, batch_size=128, num_episodes=2000,
                             target_update=5, render=False, tau=0.005)

default_environment_id = 'Breakout-v4'

default_model_saving_path = 'out/saved-models/breakout'
default_plot_saving_path = 'out/plots/braekout'

default_is_model_saved = False
default_is_plot_saved = False

default_random_seed = 21
default_max_steps = 1000


def train(hyper_params=default_hyper_params, environment_id=default_environment_id,
          model_saving_path=default_model_saving_path, plot_saving_path=default_plot_saving_path,
          is_model_saved=default_is_model_saved, is_plot_saved=default_is_plot_saved, random_seed=default_random_seed):

    project_base_dir = find_project_base_dir()

    logging.config.fileConfig(os.path.join(project_base_dir, 'config.ini'),
                              defaults={'logfilename': os.path.join(project_base_dir, 'logs', 'train.log')
                              .replace('\\', '\\\\')})
    logger = logging.getLogger("TRAIN")

    render_mode = None
    if hyper_params.render:
        render_mode = 'human'

    env = gym.make(environment_id, render_mode=render_mode, obs_type='rgb')
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, grayscale_newaxis=True, scale_obs=True,
                             frame_skip=1)
    env = FlattenObservation(env)
    env = TimeLimit(env, max_episode_steps=default_max_steps)

    logger.info(f'loading environments: {env.unwrapped.spec.id}')

    env.reset(seed=random_seed)
    np.random.seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f'using device: {device}')

    model = agents.dqn.DQN(env, device, logger).to(device)
    logger.info(f'training start: {hyper_params}')
    rewards_list, steps_list = model.train_with_episodes(hyper_params)

    # save model in the root/saved_models/cartpole
    if is_model_saved:
        model_saving_path_abs = os.path.join(project_base_dir, model_saving_path)
        model.save(model_saving_path_abs)

    if is_plot_saved:
        logger.info(f'saving plots to {plot_saving_path}')
        plot_saving_path_abs = os.path.join(project_base_dir, plot_saving_path)
        plmm(plot_saving_path_abs, rewards_list, ylabel='rewards', title='Rewards while Training')
        plmm(plot_saving_path_abs, steps_list, ylabel='steps', title='Steps while Training')

    logger.error('training ended')

    return rewards_list, steps_list


if __name__ == '__main__':
    train()
