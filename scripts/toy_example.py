import os
import numpy as np
import torch

from srcsep.frontend import load_data, format_np, generate, format_tensor
from srcsep.utils import (configsdir, parse_input_args, read_config,
                          make_experiment_name, checkpointsdir, save_exp_to_h5,
                          exp_glitch, square_glitch, datadir)

# Random seed.
SEED = 12
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

CONFIG_FILE = 'toy_example.json'
TURBULENCE_PATH = datadir(os.path.join('turbulence', 'heliumjet_synthesized'))


def optimize(args):
    # Background noise signal (to be reconstructed).
    x_true = load_data(process_name=args.process_name,
                       R=1,
                       T=args.window_size + 1,
                       H=0.5,
                       lam=0.2)[0, 0, :]
    x_true = format_np(np.diff(x_true))

    if args.type == 'square_glitch':
        noise = format_np(square_glitch(args.window_size, np.std(x_true)))
    elif args.type == 'exp_glitch':
        noise = format_np(exp_glitch(args.window_size))
    elif args.type == 'turbulence':
        noise = np.load(os.path.join(TURBULENCE_PATH, '15393898.npy'),
                        'r')[..., :args.window_size]
        noise = noise * 15 * np.linalg.norm(x_true) / np.linalg.norm(noise)
        noise -= np.mean(noise)

    # Assemble the glitched signal (to be deglitched).
    x_obs = x_true + noise

    # realizations of background noise
    x_dataset = load_data(process_name=args.process_name,
                          R=args.R + 1,
                          T=args.window_size + 1,
                          H=0.5,
                          lam=0.2)[1:, ...]
    x_dataset = np.diff(x_dataset)

    # Access to a representative (unsupervised) dataset of data snippets.
    deglitching_params = {
        'nks': format_tensor(x_dataset),
        'x_init': format_tensor(x_obs),
        'indep_loss_w': 1.0,
        'x_loss_w': 1.0,
        'fixed_ts': None,
        'cuda': args.cuda
    }
    x_hat_with_reg = generate(x_obs,
                              x0=x_obs,
                              J=args.j,
                              Q=args.q,
                              wav_type=args.wavelet,
                              it=args.max_itr,
                              tol_optim=args.tol_optim,
                              deglitching_params=deglitching_params,
                              cuda=args.cuda,
                              nchunks=args.nchunks,
                              gpus=[args.gpu_id],
                              exp_name='{}_R-{}_type-{}'.format(
                                  args.experiment_name, args.R, args.type))

    # Save the results.
    save_exp_to_h5(os.path.join(checkpointsdir(args.experiment),
                                'reconstruction.h5'),
                   args,
                   x_obs=x_obs,
                   x_true=x_true,
                   noise=noise,
                   x_dataset=x_dataset,
                   x_hat_with_reg=x_hat_with_reg)


if __name__ == '__main__':
    # Command line arguments.
    cmd_args = read_config(os.path.join(configsdir(), CONFIG_FILE))
    cmd_args = parse_input_args(cmd_args)
    cmd_args.q = [int(j) for j in cmd_args.q.replace(' ', '').split(',')]
    cmd_args.j = [int(j) for j in cmd_args.j.replace(' ', '').split(',')]
    cmd_args.experiment = make_experiment_name(cmd_args)

    optimize(cmd_args)
