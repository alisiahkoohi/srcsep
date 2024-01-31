""" Loss functions on scattering moments. """
from typing import *
import numpy as np
import torch.nn as nn
import torch

from srcsep.layers.described_tensor import DescribedTensor


class MSELossScat(nn.Module):
    """ Implements l2 norm on the scattering coefficients or scattering covariances. """
    def __init__(self):
        super(MSELossScat, self).__init__()
        self.max_gap, self.mean_gap_pct, self.max_gap_pct = {}, {}, {}  # tracking

    def compute_gap(self, input: Optional[DescribedTensor], target: DescribedTensor, weights):
        if input is None:
            gap = torch.zeros_like(target.y) - target.y
        else:
            gap = input.y - target.y
        gap = gap[:, :, 0]

        gap = gap if weights is None else weights.unsqueeze(-1) * gap

        for c_type in np.unique(target.descri['c_type']):
            # discard very small coefficients
            mask_ctype = target.descri.where(c_type=c_type)
            mask_small = (torch.abs(target.y[:, :, 0].mean(0)) < 0.01).cpu().numpy()
            mask = mask_ctype & ~mask_small

            if mask.sum() == 0:
                self.max_gap[c_type] = self.mean_gap_pct[c_type] = self.max_gap_pct[c_type] = 0.0
                continue

            self.max_gap[c_type] = torch.max(torch.abs(gap[:, mask])).item()

            mean_gap_pct = torch.abs(gap[:, mask]).mean()
            mean_gap_pct /= torch.abs(target.select(mask)[:, :, 0]).mean()
            self.mean_gap_pct[c_type] = mean_gap_pct.item()

            max_gap_pct = torch.max(torch.abs(gap[:, mask] / target.select(mask)[:, :, 0]))
            self.max_gap_pct[c_type] = max_gap_pct.item()

        return gap

    def forward(self, input, target, weights_gap, weights_l2):
        """ Computes l2 norm. """
        gap = self.compute_gap(input, target, weights_gap)
        if weights_l2 is None:
            loss = torch.abs(gap).pow(2.0).mean()
        else:
            loss = (weights_l2 * torch.abs(gap).pow(2.0)).sum()
        return loss


class DeglitchingLoss(nn.Module):
    """ Computes Ave_i |phi(nt) - phi(ni)|^2 + |phi(x-nt,nt)|^2 """
    def __init__(self, phi_x, phi_nks, std_nks, std_x_nks, std_cross, x_loss_w=1.0, indep_loss_w=1.0):
        super(DeglitchingLoss, self).__init__()
        self.phi_x, self.phi_nks = phi_x, phi_nks  # fixed representations used in the loss
        self.std_nks, self.std_x_nks, self.std_cross = std_nks, std_x_nks, std_cross  # stds used to weight l2 norm

        self.max_gap, self.mean_gap_pct, self.max_gap_pct = {}, {}, {}  # tracking

        self.indep_loss_w = indep_loss_w
        self.x_loss_w = x_loss_w

    def compute_gap(self, input: Optional[DescribedTensor], target: DescribedTensor, weights):
        if input is None:
            gap = torch.zeros_like(target.y) - target.y
        else:
            gap = input.y - target.y
        gap = gap[:, :, 0]

        gap = gap if weights is None else weights.unsqueeze(-1) * gap

        for c_type in np.unique(target.descri['c_type']):
            # discard very small coefficients
            mask_ctype = target.descri.where(c_type=c_type)
            mask_small = (torch.abs(target.y[:, :, 0].mean(0)) < 0.01).cpu().numpy()
            mask = mask_ctype & ~mask_small

            if mask.sum() == 0:
                self.max_gap[c_type] = self.mean_gap_pct[c_type] = self.max_gap_pct[c_type] = 0.0
                continue

            self.max_gap[c_type] = torch.max(torch.abs(gap[:, mask])).item()

            mean_gap_pct = torch.abs(gap[:, mask]).mean()
            mean_gap_pct /= torch.abs(target.select(mask)[:, :, 0]).mean()
            self.mean_gap_pct[c_type] = mean_gap_pct.item()

            max_gap_pct = torch.max(torch.abs(gap[:, mask] / target.select(mask)[:, :, 0]))
            self.max_gap_pct[c_type] = max_gap_pct.item()

        return gap

    @staticmethod
    def mse(x):
        return torch.abs(x).pow(2.0).mean()

    def forward(self, input, target, weights_gap, weights_l2):
        # loss term Ave_k |phi(nt) - phi(nk)|^2
        phi_nt = input.reduce(b=0, deglitch_loss_term=0)
        gap1 = self.compute_gap(phi_nt, self.phi_nks, weights_gap)
        # gap1 = self.compute_gap(phi_nt, self.phi_nks.mean_batch(), weights_gap)  # an alternative
        gap1 /= self.std_nks[None, :]
        loss1 = self.mse(gap1)

        # loss term Ave_k |phi(x) - phi(x-nk+nt)|^2
        phi_x_nks = input.reduce(deglitch_loss_term=1)
        gap2 = self.phi_x.y[:,:,0] - phi_x_nks.y[:,:,0]
        # gap2 = self.phi_x.y[:,:,0] - phi_x_nks.mean_batch().y[:,:,0]  # an alternative
        gap2 /= self.std_x_nks[None, :]
        loss2 = self.mse(gap2)

        # independence loss |phi(x-nt, nk)|^2 + |phi(nk, x-nt)|^2
        c_types = ['spars','ps','phaseenv','envelope']  # the coefficient type on which imposing independence
        cross12 = input.select(nl=0, nr=1,c_type=c_types,deglitch_loss_term=2)[:,:,0] / self.std_cross[0][None,:]
        cross21 = input.select(nl=1, nr=0,c_type=c_types,deglitch_loss_term=2)[:,:,0] / self.std_cross[1][None,:]
        loss3 = 0.5 * (self.mse(cross12) + self.mse(cross21))

        #print('data loss:', loss1.item(), 'prior loss:', loss2.item(), 'independence loss:', loss3.item())

        return loss1 + self.x_loss_w * loss2 + self.indep_loss_w * loss3
