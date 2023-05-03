import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os
from pandas import DataFrame as df


def plot_list_with_moving_mean(path, data, title='Result', ylabel='value', show=True, window_size=30):
    if not show:
        matplotlib.use('Agg')
    else:
        matplotlib.use('TkAgg')

    fig, ax = plt.subplots()

    if type(data) is list:
        ax.plot(data, label='Raw data')
        moving_mean_data = moving_mean(data, window_size)
        ax.plot(np.arange(window_size - 1, len(data)), moving_mean_data, label=f'Moving mean (w: {window_size})')
    elif type(data) is df:
        parameter_name = data.attrs['parameter']

        for col in data.columns:
            if col == 'parameter':
                continue

            for row in data[col].index:
                list_for_plot = data.loc[row, col]
                parameter = data.loc[row, 'parameter']
                ax.plot(list_for_plot, label=f'Raw data. {parameter_name}= {parameter}')
                moving_mean_data = moving_mean(list_for_plot, window_size)
                ax.plot(np.arange(window_size - 1, len(list_for_plot)), moving_mean_data,
                        label=f'Moving mean (w: {window_size}). {parameter_name}= {parameter}')

    ax.set_xlabel('Episode')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 6})
    plt.tight_layout(pad=0.5)
    plt.show()

    filename = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '-' + title.replace(' ', '_').replace(':', '_') + '.jpg'
    fig.savefig(os.path.join(path, filename))

    plt.close(fig)


def plot_moving_mean(path, data, title='Result with Moving mean', ylabel='value', show=True, window_size=30):
    if not show:
        matplotlib.use('Agg')
    else:
        matplotlib.use('TkAgg')

    fig, ax = plt.subplots()

    if type(data) is list:
        moving_mean_data = moving_mean(data, window_size)
        ax.plot(np.arange(window_size - 1, len(data)), moving_mean_data, label=f'w: {window_size}')
    elif type(data) is df:
        parameter_name = data.attrs['parameter']

        for col in data.columns:
            if col == 'parameter':
                continue

            for row in data[col].index:
                list_for_plot = data.loc[row, col]
                parameter = data.loc[row, 'parameter']
                moving_mean_data = moving_mean(list_for_plot, window_size)
                ax.plot(np.arange(window_size - 1, len(list_for_plot)), moving_mean_data,
                        label=f'(w: {window_size}). {parameter_name}= {parameter}')

    ax.set_xlabel('Episode')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 6})
    plt.tight_layout(pad=0.5)
    plt.show()

    filename = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '-' + title.replace(' ', '_').replace(':', '_') + '.jpg'
    fig.savefig(os.path.join(path, filename))

    plt.close(fig)


def moving_mean(data, window_size=10):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size
