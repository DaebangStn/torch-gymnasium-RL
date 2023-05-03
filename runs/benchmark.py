import logging
import logging.config
import os
import pandas as pd

from utils.find_project_base_dir import find_project_base_dir
from plots.plot_list import plot_moving_mean as plmm
from agents.utils import HyperParams
from runs.train import train

default_environment_id = 'CartPoleReward-v1'
# default_environment_id = 'CartPole-v1'

default_model_saving_path = 'out/saved-models/cartpole'
default_plot_saving_path = 'out/plots/cartpole'

default_is_model_saved = True
default_is_plot_saved = True

default_random_seed = 21
default_max_steps = 500

p_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
#p_list = [0.001, 0.002]


# benchmark function that runs the train() for the different parameters
def benchmark():
    project_base_dir = find_project_base_dir()

    logging.config.fileConfig(os.path.join(project_base_dir, 'config.ini'),
                              defaults={'logfilename': os.path.join(project_base_dir, 'logs', 'benchmark.log')
                              .replace('\\', '\\\\')})
    logger = logging.getLogger("BENCH")

    logger.info(f'benchmark start. parameters: {p_list}')

    df = pd.DataFrame(columns=['parameter', 'rewards', 'steps'])

    for p in p_list:
        logger.info(f'training for parameter: {p} started')

        hyper_param = HyperParams(lr=p, gamma=0.99, epsilon_begin=1, epsilon_end=0.05,
                                  epsilon_decay=0.995, batch_size=128, num_episodes=300,
                                  target_update=5, render=False, tau=0.005)

        rewards_list, steps_list = \
            train(hyper_param, environment_id=default_environment_id, model_saving_path=default_model_saving_path,
                  plot_saving_path=default_plot_saving_path, is_model_saved=default_is_model_saved,
                  is_plot_saved=False, random_seed=default_random_seed)

        df = pd.concat([pd.DataFrame({'parameter': p, 'rewards': [rewards_list], 'steps': [steps_list]}), df],
                       ignore_index=True)

    df.attrs['parameter'] = 'lr'
    df.attrs['environment'] = default_environment_id

    logger.info(f'saving plots to {default_plot_saving_path}')
    plot_saving_path_abs = os.path.join(project_base_dir, default_plot_saving_path)

    plmm(plot_saving_path_abs, df[['parameter', 'rewards']], ylabel='rewards', title='Bench: Rewards while Training')
    plmm(plot_saving_path_abs, df[['parameter', 'steps']], ylabel='steps', title='Bench: Steps while Training')

    logger.error(f'benchmark finished')


if __name__ == '__main__':
    benchmark()
