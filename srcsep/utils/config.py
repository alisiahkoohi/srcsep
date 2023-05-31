import argparse
import json
import torch


def read_config(filename):
    """Read input variables and values from a json file."""
    with open(filename) as f:
        configs = json.load(f)
    return configs


def write_config(args, filename):
    "Write command line arguments into a json file."
    with open(filename, 'w') as f:
        json.dump(args, f)


def parse_input_args(args):
    "Use variables in args to create command line input parser."
    parser = argparse.ArgumentParser(description='')
    for key, value in args.items():
        parser.add_argument('--' + key, default=value, type=type(value))
    return parser.parse_args()


def make_experiment_name(args):
    """Make experiment name based on input arguments"""
    experiment_name = args.experiment_name + '_'
    for key, value in vars(args).items():
        if key not in [
                'cuda', 'phase', 'experiment_name', 'experiment', 'gpu_id',
                'nchunks'
        ]:
            experiment_name += key + '-{}_'.format(value)
    return experiment_name[:-1].replace(' ', '').replace('[', '').replace(
        ']', '').replace(',', '-')


def make_h5_file_name(args):
    """Make HDF5 file name based on input arguments"""
    filename = args.experiment_name + '_'
    for key, value in vars(args).items():
        if key not in [
                'cuda', 'phase', 'experiment_name', 'experiment', 'gpu_id',
                'nchunks'
        ]:
            filename += key + '-{}_'.format(value)
    filename = filename[:-1].replace(' ', '').replace('[', '').replace(
        ']', '').replace(',', '-') + '.h5'
    return filename
