import os

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

from srcsep.utils import (
    configsdir,
    parse_input_args,
    read_config,
    plotsdir,
    query_experiments,
    collect_results,
)

sns.set_style("whitegrid")
font = {'family': 'serif', 'style': 'normal', 'size': 10}
matplotlib.rc('font', **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")

# Random seed.
SEED = 12
np.random.seed(SEED)

CONFIG_FILE = 'toy_example.json'


def plot_result(**kwargs):

    fig = plt.figure(figsize=(7, 1.5))
    plt.plot(np.arange(args.window_size),
             kwargs['x_true'][0, 0, :],
             color="#000000",
             lw=.6,
             alpha=0.7)
    ax = plt.gca()
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.xlim([0, args.window_size - 1])
    plt.ylim([-0.17, 0.17])
    plt.ylabel("True")
    plt.savefig(os.path.join(plotsdir(kwargs['args'].experiment),
                             "x_true.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 1.5))
    plt.plot(np.arange(args.window_size),
             kwargs['x_obs'][0, 0, :],
             color="#000000",
             lw=.6,
             alpha=0.7)
    ax = plt.gca()
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.xlim([0, args.window_size - 1])
    plt.ylim([-0.17, 0.17])
    plt.ylabel("Observed")
    plt.savefig(os.path.join(plotsdir(kwargs['args'].experiment), "x_obs.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 1.5))
    plt.plot(np.arange(args.window_size),
             kwargs['x_hat_with_reg'][0, 0, :],
             color="#000000",
             lw=.6,
             alpha=0.7)
    ax = plt.gca()
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.xlim([0, args.window_size - 1])
    plt.ylim([-0.17, 0.17])
    plt.ylabel("Predicted")
    plt.savefig(os.path.join(plotsdir(kwargs['args'].experiment),
                             "x_hat_with_reg.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 1.5))
    plt.plot(np.arange(args.window_size),
             kwargs['x_true'][0, 0, :] - kwargs['x_hat_with_reg'][0, 0, :],
             color="#000000",
             lw=.6,
             alpha=0.7)
    ax = plt.gca()
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.xlim([0, args.window_size - 1])
    plt.ylim([-0.17, 0.17])
    plt.ylabel("Error")
    plt.savefig(os.path.join(plotsdir(kwargs['args'].experiment), "error.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 1.5))
    plt.plot(np.arange(args.window_size),
             kwargs['x_obs'][0, 0, :] - kwargs['x_true'][0, 0, :],
             color="#000000",
             lw=.6,
             alpha=0.7)
    ax = plt.gca()
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.xlim([0, args.window_size - 1])
    plt.ylim([-0.17, 0.17])
    plt.ylabel("Added signal")
    plt.savefig(os.path.join(plotsdir(kwargs['args'].experiment), "noise.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)


if __name__ == '__main__':
    # Command line arguments.
    args = read_config(os.path.join(configsdir(), CONFIG_FILE))
    args = parse_input_args(args)

    args.q = ([int(j) for j in args.q.replace(' ', '').split(',')], )
    args.j = ([int(j) for j in args.j.replace(' ', '').split(',')], )

    experiment_args = query_experiments(CONFIG_FILE, **vars(args))
    experiment_results = collect_results(
        experiment_args, ['x_obs', 'x_true', 'x_hat_with_reg'])[0]
    plot_result(**experiment_results)
