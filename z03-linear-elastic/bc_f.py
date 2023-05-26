#!/usr/bin/env python3

"""
boundary condition and rhs force data
"""

import torch
import numpy as np
import config
from config import sf_nd_nb

nloc = config.nloc
dev = config.dev
nonods = config.nonods


def bc_f(ndim, bc, u, x_all, prob: str):
    if prob == 'linear-elastic':
        if ndim == 2:
            # apply boundary conditions (4 Dirichlet bcs)
            for inod in bc[0]:
                u[int(inod / nloc), inod % nloc, :] = 0.
            for inod in bc[1]:
                u[int(inod / nloc), inod % nloc, :] = 0.
            for inod in bc[2]:
                u[int(inod / nloc), inod % nloc, :] = 0.
            for inod in bc[3]:
                u[int(inod / nloc), inod % nloc, :] = 0.
                # x_inod = x_ref_in[inod//10, 0, inod%10]
                # u[:,inod]= torch.sin(torch.pi*x_inod)

            mu = config.mu
            # takes in coordinates numpy array (nonods, ndim)
            # output body force: torch tensor (nele*nloc, ndim)
            mu = mu.cpu().numpy()
            f = np.zeros((nonods, ndim), dtype=np.float64)
            f[:, 0] += -2.0 * mu * np.power(np.pi, 3) * \
                    np.cos(np.pi * x_all[:, 1]) * np.sin(np.pi * x_all[:, 1]) \
                    * (2 * np.cos(2 * np.pi * x_all[:, 0]) - 1)
            f[:, 1] += 2.0 * mu * np.power(np.pi, 3) * \
                    np.cos(np.pi * x_all[:, 0]) * np.sin(np.pi * x_all[:, 0]) \
                    * (2 * np.cos(2 * np.pi * x_all[:, 1]) - 1)
            f = torch.tensor(f, device=dev, dtype=torch.float64)
            fNorm = torch.linalg.norm(f.view(-1), dim=0)

        else:
            for bci in bc:
                for inod in bci:
                    x_inod = sf_nd_nb.x_ref_in[inod // nloc, :, inod % nloc]
                    # u[inod//nloc, inod%nloc, :] = 0.
                    u[inod // nloc, inod % nloc, :] = torch.exp(-x_inod[0] - x_inod[1] - x_inod[2])
            f = torch.zeros((nonods, ndim), device=dev, dtype=torch.float64)
            mu = config.mu
            lam = config.lam
            x = torch.tensor(x_all[:, 0], device=dev)
            y = torch.tensor(x_all[:, 1], device=dev)
            z = torch.tensor(x_all[:, 2], device=dev)
            # f[:, 0] = mu * (torch.pi ** 2 * torch.sin(torch.pi * (x + 1)) * torch.sin(torch.pi * (y + 1)) * torch.sin(
            #     torch.pi * (z + 1)) - torch.pi ** 2 * torch.cos(torch.pi * (x + 1)) * torch.cos(torch.pi * (y + 1)) * torch.sin(
            #     torch.pi * (z + 1))) + mu * (
            #                       torch.pi ** 2 * torch.sin(torch.pi * (x + 1)) * torch.sin(torch.pi * (y + 1)) * torch.sin(
            #                   torch.pi * (z + 1)) - torch.pi ** 2 * torch.cos(torch.pi * (x + 1)) * torch.cos(
            #                   torch.pi * (z + 1)) * torch.sin(torch.pi * (y + 1))) - lam * (
            #                       torch.pi ** 2 * torch.cos(torch.pi * (x + 1)) * torch.cos(torch.pi * (y + 1)) * torch.sin(
            #                   torch.pi * (z + 1)) - torch.pi ** 2 * torch.sin(torch.pi * (x + 1)) * torch.sin(
            #                   torch.pi * (y + 1)) * torch.sin(torch.pi * (z + 1)) + torch.pi ** 2 * torch.cos(
            #                   torch.pi * (x + 1)) * torch.cos(torch.pi * (z + 1)) * torch.sin(
            #                   torch.pi * (y + 1))) + 2 * mu * torch.pi ** 2 * torch.sin(torch.pi * (x + 1)) * torch.sin(
            #     torch.pi * (y + 1)) * torch.sin(torch.pi * (z + 1))
            # f[:, 1] = mu * (torch.pi ** 2 * torch.sin(torch.pi * (x + 1)) * torch.sin(torch.pi * (y + 1)) * torch.sin(
            #     torch.pi * (z + 1)) - torch.pi ** 2 * torch.cos(torch.pi * (x + 1)) * torch.cos(torch.pi * (y + 1)) * torch.sin(
            #     torch.pi * (z + 1))) + mu * (
            #                       torch.pi ** 2 * torch.sin(torch.pi * (x + 1)) * torch.sin(torch.pi * (y + 1)) * torch.sin(
            #                   torch.pi * (z + 1)) - torch.pi ** 2 * torch.cos(torch.pi * (y + 1)) * torch.cos(
            #                   torch.pi * (z + 1)) * torch.sin(torch.pi * (x + 1))) - lam * (
            #                       torch.pi ** 2 * torch.cos(torch.pi * (x + 1)) * torch.cos(torch.pi * (y + 1)) * torch.sin(
            #                   torch.pi * (z + 1)) - torch.pi ** 2 * torch.sin(torch.pi * (x + 1)) * torch.sin(
            #                   torch.pi * (y + 1)) * torch.sin(torch.pi * (z + 1)) + torch.pi ** 2 * torch.cos(
            #                   torch.pi * (y + 1)) * torch.cos(torch.pi * (z + 1)) * torch.sin(
            #                   torch.pi * (x + 1))) + 2 * mu * torch.pi ** 2 * torch.sin(torch.pi * (x + 1)) * torch.sin(
            #     torch.pi * (y + 1)) * torch.sin(torch.pi * (z + 1))
            # f[:, 2] = mu * (torch.pi ** 2 * torch.sin(torch.pi * (x + 1)) * torch.sin(torch.pi * (y + 1)) * torch.sin(
            #     torch.pi * (z + 1)) - torch.pi ** 2 * torch.cos(torch.pi * (x + 1)) * torch.cos(torch.pi * (z + 1)) * torch.sin(
            #     torch.pi * (y + 1))) + mu * (
            #                       torch.pi ** 2 * torch.sin(torch.pi * (x + 1)) * torch.sin(torch.pi * (y + 1)) * torch.sin(
            #                   torch.pi * (z + 1)) - torch.pi ** 2 * torch.cos(torch.pi * (y + 1)) * torch.cos(
            #                   torch.pi * (z + 1)) * torch.sin(torch.pi * (x + 1))) - lam * (
            #                       torch.pi ** 2 * torch.cos(torch.pi * (x + 1)) * torch.cos(torch.pi * (z + 1)) * torch.sin(
            #                   torch.pi * (y + 1)) - torch.pi ** 2 * torch.sin(torch.pi * (x + 1)) * torch.sin(
            #                   torch.pi * (y + 1)) * torch.sin(torch.pi * (z + 1)) + torch.pi ** 2 * torch.cos(
            #                   torch.pi * (y + 1)) * torch.cos(torch.pi * (z + 1)) * torch.sin(
            #                   torch.pi * (x + 1))) + 2 * mu * torch.pi ** 2 * torch.sin(torch.pi * (x + 1)) * torch.sin(
            #     torch.pi * (y + 1)) * torch.sin(torch.pi * (z + 1))
            f[:, 0] = - 3 * lam * torch.exp(- x - y - z) - 6 * mu * torch.exp(- x - y - z)
            f[:, 1] = - 3 * lam * torch.exp(- x - y - z) - 6 * mu * torch.exp(- x - y - z)
            f[:, 2] = - 3 * lam * torch.exp(- x - y - z) - 6 * mu * torch.exp(- x - y - z)
            fNorm = torch.linalg.norm(f.view(-1), dim=0)
            del x, y, z
    if prob == 'hyper-elastic':
        ...
    return u, f, fNorm
