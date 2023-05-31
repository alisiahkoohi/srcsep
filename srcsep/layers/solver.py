""" Manage generation algorithm. """
from typing import *
from termcolor import colored
from time import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, grad

from .described_tensor import DescribedTensor


class Solver(nn.Module):
    """ A class that contains all information necessary for generation. """
    def __init__(self,
                 model: nn.Module, loss: nn.Module,
                 xf: Optional[np.ndarray] = None,
                 Rxf: Optional[DescribedTensor] = None,
                 x0: Optional[np.ndarray] = None,
                 fixed_ts: Optional[np.ndarray] = None,
                 cuda: bool = False) -> None:
        super(Solver, self).__init__()

        self.model = model
        self.loss = loss

        self.B = xf.shape[0]
        self.N = xf.shape[1]
        self.nchunks = 1
        self.is_cuda = cuda
        self.x0 = torch.DoubleTensor(x0)

        # time indices at which to put the gradient to zero
        self.fixed_ts = fixed_ts

        self.counter = 0
        self.res = None, None

        # for debug only
        self.grad_stored = []

        if cuda:
            self.cuda()
            if Rxf is not None:
                Rxf = Rxf.cuda()

        self.Rxf = Rxf

        # compute initial loss
        Rx0 = self.model(self.format(x0, requires_grad=False))
        self.loss0 = self.loss(Rx0, self.Rxf, None, None).detach().cpu().numpy()

    def format(self, x: np.ndarray, requires_grad: Optional[bool] = True) -> torch.tensor:
        """ Transforms x into a compatible format for the embedding. """
        x = torch.tensor(x.reshape(self.B, self.N, -1)).unsqueeze(-2).unsqueeze(-2)
        if self.is_cuda:
            x = x.cuda()
        x = Variable(x, requires_grad=requires_grad)
        return x

    def joint(self, x: np.ndarray) -> Tuple[torch.tensor, torch.tensor]:
        """ Computes the loss on current vector. """

        # format x and set gradient to 0
        x_torch = self.format(x)

        res_max = {c_type: 0.0 for c_type in self.model.module.c_types}
        res_max_pct = {c_type: 0.0 for c_type in self.model.module.c_types}

        # chunk gradient computation if necessary
        # for deglitching it is not equivalent to computing the gradient once
        if self.model.nchunks > 1:

            total_loss = 0.0
            total_grad = np.zeros(x_torch.shape, dtype=np.float64)

            for i_chunk in range(self.model.nchunks):
                # clear gradient
                if x_torch.grad is not None:
                    x_torch.grad.x.zero_()

                Rxt = self.model(x_torch, i_chunk)

                loss = self.loss(Rxt, self.Rxf, None, None)
                res_max = {c_type: max(res_max[c_type], self.loss.max_gap[c_type] if c_type in self.loss.max_gap else 0.0)
                           for c_type in self.model.module.c_types}
                res_mean_pct = {c_type: max(res_max_pct[c_type],
                                            self.loss.mean_gap_pct[c_type] if c_type in self.loss.mean_gap_pct else 0.0)
                                for c_type in self.model.module.c_types}
                res_max_pct = {c_type: max(res_max_pct[c_type],
                                           self.loss.max_gap_pct[c_type] if c_type in self.loss.max_gap_pct else 0.0)
                               for c_type in self.model.module.c_types}

                # compute gradient
                grad_x, = grad([loss], [x_torch], retain_graph=True)

                # move to numpy
                loss = loss.detach().cpu().numpy()
                grad_x = grad_x.contiguous().detach().cpu().numpy()

                total_loss += loss / self.model.nchunks
                total_grad += grad_x

            if self.fixed_ts is not None:
                total_grad[..., self.fixed_ts] = 0.0

            self.res = total_loss, total_grad.ravel(), res_max, res_mean_pct, res_max_pct

            return total_loss, total_grad.ravel()

        else:
            # clear gradient
            if x_torch.grad is not None:
                x_torch.grad.x.zero_()

            # compute moments
            Rxt = self.model(x_torch)

            loss = self.loss(Rxt, self.Rxf, None, None)
            res_max = {c_type: max(res_max[c_type], self.loss.max_gap[c_type] if c_type in self.loss.max_gap else 0.0)
                       for c_type in self.model.module.c_types}
            res_mean_pct = {c_type: max(res_max_pct[c_type], self.loss.mean_gap_pct[c_type] if c_type in self.loss.mean_gap_pct else 0.0)
                            for c_type in self.model.module.c_types}
            res_max_pct = {c_type: max(res_max_pct[c_type], self.loss.max_gap_pct[c_type] if c_type in self.loss.max_gap_pct else 0.0)
                           for c_type in self.model.module.c_types}

            # compute gradient
            grad_x, = grad([loss], [x_torch], retain_graph=True)

            # move to numpy
            grad_x = grad_x.contiguous().detach().cpu().numpy()
            loss = loss.detach().cpu().numpy()

            if self.fixed_ts is not None:
                grad_x[..., self.fixed_ts] = 0.0

            self.res = loss, grad_x.ravel(), res_max, res_mean_pct, res_max_pct

            return loss, grad_x.ravel()


class SmallEnoughException(Exception):
    pass


class CheckConvCriterion:
    """ A callback function given to the optimizer. """
    def __init__(self,
                 solver: Solver,
                 tol: float,
                 max_wait: Optional[int] = 100,
                 save_data_evolution_p: Optional[bool] = False):
        self.solver = solver
        self.tol = tol
        self.result = None
        self.next_milestone = None
        self.counter = 0
        self.err = None
        self.max_gap = None
        self.gerr = None
        self.tic = time()

        self.max_wait, self.wait = max_wait, 0
        self.save_data_evolution_p = save_data_evolution_p

        self.logs_loss = []
        self.logs_grad = []
        self.logs_x = []

    def __call__(self, xk: np.ndarray) -> None:
        err, grad_xk, max_gap, mean_gap_pct, max_gap_pct = self.solver.res

        gerr = np.max(np.abs(grad_xk))
        err, gerr = float(err), float(gerr)
        self.err = err
        self.max_gap = max_gap
        self.mean_gap_pct = mean_gap_pct
        self.max_gap_pct = max_gap_pct
        self.gerr = gerr
        self.counter += 1
        self.solver.counter += 1

        self.logs_loss.append(err)
        self.logs_grad.append(gerr)

        if self.next_milestone is None:
            self.next_milestone = 10 ** (np.floor(np.log10(gerr)))

        info_already_printed_p = False
        if self.save_data_evolution_p and not np.log2(self.counter) % 1:
            self.logs_x.append(xk)
            self.print_info_line('SAVED X')
            info_already_printed_p = True

        if np.sqrt(err) <= self.tol:
            self.result = xk
            raise SmallEnoughException()
        elif gerr <= self.next_milestone or self.wait >= self.max_wait:
            if not info_already_printed_p:
                self.print_info_line()
            if gerr <= self.next_milestone:
                self.next_milestone /= 10
            self.wait = 0
        else:
            self.wait += 1

    def print_info_line(self, msg: Optional[str] = '') -> None:
        delta_t = time() - self.tic

        def cap(pct):
            return pct if pct < 1e3 else np.inf

        print(colored(
            f"{self.counter:6}it in {self.hms_string(delta_t)} ( {self.counter / delta_t:.2f}it/s )"
            + " .... "
            + f"err {np.sqrt(self.err):.2E} -- max {max(self.max_gap.values()):.2E}"
            + f" -- maxpct {cap(max(self.max_gap_pct.values())):.3%} -- gerr {self.gerr:.2E}",
            'cyan'))
        print(colored(
            "".join([f"\n -- {c_type:<15} max {value:.2e} -- meanpct {cap(self.mean_gap_pct[c_type]):.2%} "
                     + f"-- maxpct {cap(self.max_gap_pct[c_type]):.1%}, "
                     for c_type, value in self.max_gap.items()])
            + msg,
            'green'))

    @staticmethod
    def hms_string(sec_elapsed: float) -> str:
        """ Format  """
        h = int(sec_elapsed / (60 * 60))
        m = int((sec_elapsed % (60 * 60)) / 60)
        s = sec_elapsed % 60.
        return f"{h}:{m:>02}:{s:>05.2f}"
