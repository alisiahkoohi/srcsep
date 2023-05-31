import itertools
import os
import subprocess

import h5py
import numpy as np

from srcsep.utils import (checkpointsdir, configsdir, make_experiment_name,
                          parse_input_args, plotsdir, read_config)


def make_complete_args(config_file, **kwargs):
    """Make the arguments for the query."""

    args = read_config(os.path.join(configsdir(), config_file))
    args = parse_input_args(args)

    args.q = [int(j) for j in args.q.replace(' ', '').split(',')]
    args.j = [int(j) for j in args.j.replace(' ', '').split(',')]

    # Create args from the kwargs.
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


def query_experiments(config_file, **kwargs):
    """Make the arguments for the query."""
    args_list = []
    key_list = []
    # Create args from the kwargs.
    for key, value in kwargs.items():
        if (not isinstance(value, list)) and (not isinstance(value, tuple)):
            value = (value, )
        args_list.append(value)
        key_list.append(key)

    list_args = []
    for args in itertools.product(*args_list):
        list_args.append(dict(zip(key_list, args)))

    experiment_args = []
    for kwargs in list_args:
        args = make_complete_args(config_file, **kwargs)
        args.experiment = make_experiment_name(args)
        experiment_args.append(args)

    return experiment_args


def collect_results(experiment_args, keys):
    """Collect the results from the experiments."""
    results = []
    for args in experiment_args:
        h5_path = os.path.join(checkpointsdir(args.experiment),
                               'reconstruction.h5')
        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as f:
                results.append({key: f[key][...] for key in keys})
                results[-1]['args'] = args

    return results


def plot_results(experiment_args):
    """Collect the results from the experiments."""
    for args in experiment_args:
        h5_path = os.path.join(checkpointsdir(args.experiment),
                               'reconstruction.h5')
        if os.path.exists(h5_path):

            file = h5py.File(h5_path, 'r')
            x_hat_with_reg, x_obs = file['x_hat_with_reg'][...], file['x_obs'][
                ...]
            if x_hat_with_reg.ndim == 4:
                x_hat_with_reg = np.mean(x_hat_with_reg, axis=0)
            file.close()
            plot_deglitching(args, 'deglitching', x_obs, x_hat_with_reg)
