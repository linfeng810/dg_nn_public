#!/usr/bin/env python3

"""
boundary condition and rhs force data
"""

import torch
import numpy as np
import config
from config import sf_nd_nb

# nloc = config.nloc
dev = config.dev
# nonods = config.nonods


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
                    u[inod // nloc, inod % nloc, :] = torch.exp(x_inod[0] + x_inod[1] + x_inod[2])
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
            f[:, 0] = - 3 * lam * torch.exp(x + y + z) - 6 * mu * torch.exp(x + y + z)
            f[:, 1] = - 3 * lam * torch.exp(x + y + z) - 6 * mu * torch.exp(x + y + z)
            f[:, 2] = - 3 * lam * torch.exp(x + y + z) - 6 * mu * torch.exp(x + y + z)
            fNorm = torch.linalg.norm(f.view(-1), dim=0)
            del x, y, z
    elif prob == 'hyper-elastic':
        if ndim == 2:
            raise Exception('2D hyper-elastic problem is not defined in bc_f.py')
        else:
            alpha = 0.1
            gamma = 0.1
            lam = config.lam
            mu = config.mu
            for bci in bc:
                for inod in bci:
                    x_inod = sf_nd_nb.x_ref_in[inod // nloc, :, inod % nloc]
                    # problem 1 Abbas 2018
                    theta = alpha * torch.sin(torch.pi * x_inod[1])  # alpha sin(pi Y)
                    g = gamma * torch.sin(torch.pi * x_inod[0])  # gamma sin(pi X)
                    h = 0  # 0

                    u[inod // nloc, inod % nloc, 0] = (1. / lam + alpha) * x_inod[0] + theta
                    u[inod // nloc, inod % nloc, 1] = -(1. / lam + (alpha + gamma + alpha * gamma) /
                                                        (1 + alpha + gamma + alpha * gamma)) * x_inod[1]
                    u[inod // nloc, inod % nloc, 2] = (1. / lam + alpha) * x_inod[2] + g + h
                    # # problem 2 Simple shear
                    # u[inod // nloc, inod % nloc, 0] = 0.1 * x_inod[1]
                    # u[inod // nloc, inod % nloc, 1] = 0
                    # u[inod // nloc, inod % nloc, 2] = 0
                    # # problem 3 exponential expansion
                    # u[inod // nloc, inod % nloc, 0] = torch.exp(x_inod[0] + x_inod[1] + x_inod[2])
                    # # problem 4 Eyck 2008 (Indentation)
                    # u[inod // nloc, inod % nloc, 0] = \
                    #     -1.75 * x_inod[1] * (1 - x_inod[1]) * x_inod[2] * (1 - x_inod[2]) \
                    #     * torch.cos(torch.pi * x_inod[0])
                    # u[inod // nloc, inod % nloc, 1] = \
                    #     -1.75 * x_inod[0] * (1 - x_inod[0]) * x_inod[2] * (1 - x_inod[2]) \
                    #     * torch.cos(torch.pi * x_inod[1])
                    # u[inod // nloc, inod % nloc, 2] = \
                    #     -0.12 * x_inod[2] ** 2 * (1 - torch.cos(2 * torch.pi * x_inod[0])) \
                    #     * (1 - torch.cos(2 * torch.pi * x_inod[1])) + 0.15 * x_inod[2]
            x = torch.tensor(x_all[:, 0], device=dev)
            y = torch.tensor(x_all[:, 1], device=dev)
            z = torch.tensor(x_all[:, 2], device=dev)
            f = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)
            # problem 1 Abbas 2018
            f[:, 0] = mu * alpha * torch.pi ** 2 * torch.sin(torch.pi * y)
            f[:, 1] = 0.
            f[:, 2] = mu * gamma * torch.pi ** 2 * torch.sin(torch.pi * x)
            # # problem 2 Simple shear
            # f *= 0
            # # problem 3 exponential expansion
            # for idim in range(ndim):
            #     from torch import exp, log
            #     f[:, idim] = -(3*exp(x + y + z)*(lam + 2*mu + 9*mu*exp(2*x + 2*y + 2*z) - lam*log(3*exp(x + y + z) + 1) + 6*mu*exp(x + y + z)))/(3*exp(x + y + z) + 1)^2
            #     ((3*lam*log(3*exp(x + y + z) + 1) - 9*lam)*exp(x + y + z)
            #                   - 18*lam*exp(2*x + 2*y + 2*z)
            #                   - 27*lam*exp(3*x + 3*y + 3*z)) / \
            #                  (6*exp(x + y + z) + 9*exp(2*x + 2*y + 2*z) + 1)
            # # problem 4 Eyck 2008 (Indentation)
            # ...
            fNorm = torch.linalg.norm(f.view(-1), dim=0)
            del x, y, z
    else:
        raise Exception('the problem ', prob, ' is not defined in bc_f.py')
    return u, f, fNorm


def vel_bc_f(ndim, bc, u, x_all, prob: str):
    nloc = sf_nd_nb.vel_func_space.element.nloc
    nonods = sf_nd_nb.vel_func_space.nonods
    if prob == 'stokes':
        if ndim == 2:
            # apply boundary conditions (4 Dirichlet bcs)
            for bci in bc:
                for inod in range(bci.shape[0]):
                    if not bci[inod]:  # this is not a boundary node
                        continue
                    x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                    # u[inod//nloc, inod%nloc, :] = 0.
                    u[inod // nloc, inod % nloc, 0] = -torch.exp(x_inod[0]) * \
                                                      (x_inod[1] * torch.cos(x_inod[1]) + torch.sin(x_inod[1]))
                    u[inod // nloc, inod % nloc, 1] = torch.exp(x_inod[0]) * x_inod[1] * torch.sin(x_inod[1])

            f = np.zeros((nonods, ndim), dtype=np.float64)

            f = torch.tensor(f, device=dev, dtype=torch.float64)
            fNorm = torch.linalg.norm(f.view(-1), dim=0)

        else:  # ndim == 3
            for bci in bc:
                for inod in range(bci.shape[0]):
                    if not bci[inod]:  # this is not a boundary node
                        continue
                    x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                    # u[inod//nloc, inod%nloc, :] = 0.
                    u[inod // nloc, inod % nloc, 0] = -2./3. * torch.sin(x_inod[0])**3
                    u[inod // nloc, inod % nloc, 1] = torch.sin(x_inod[0])**2 * \
                                                      (x_inod[1] * torch.cos(x_inod[0])
                                                       - x_inod[2] * torch.sin(x_inod[0]))
                    u[inod // nloc, inod % nloc, 2] = torch.sin(x_inod[0])**2 * \
                                                      (x_inod[2] * torch.cos(x_inod[0])
                                                       + x_inod[1] * torch.sin(x_inod[0]))
            f = torch.zeros((nonods, ndim), device=dev, dtype=torch.float64)

            x = torch.tensor(x_all[:, 0], device=dev)
            y = torch.tensor(x_all[:, 1], device=dev)
            z = torch.tensor(x_all[:, 2], device=dev)

            # sin pressure
            f[:, 0] = torch.cos(x) - 2*torch.sin(x) + 6*torch.cos(x)**2 * torch.sin(x)
            f[:, 1] = 7*y*torch.cos(x) - 9*y*torch.cos(x)**3 \
                      - 3*z*torch.sin(x) + 9*z*torch.cos(x)**2*torch.sin(x)
            f[:, 2] = 7*z*torch.cos(x) - 9*z*torch.cos(x)**3 \
                      + 3*y*torch.sin(x) - 9*y*torch.cos(x)**2*torch.sin(x)

            # # linear pressure
            # from torch import sin, cos
            # f[:, 0] = 4 * cos(x) ** 2 * sin(x) - 2 * sin(x) ** 3 + 1
            # f[:, 1] = 3 * sin(x) ** 2 * (y * cos(x) - z * sin(x)) \
            #           - 2 * cos(x) ** 2 * (y * cos(x) - z * sin(x)) \
            #           + 4 * cos(x) * sin(x) * (z * cos(x) + y * sin(x))
            # f[:, 2] = 3 * sin(x) ** 2 * (z * cos(x) + y * sin(x)) \
            #           - 2 * cos(x) ** 2 * (z * cos(x) + y * sin(x)) \
            #           - 4 * cos(x) * sin(x) * (y * cos(x) - z * sin(x))

            fNorm = torch.linalg.norm(f.view(-1), dim=0)
            del x, y, z
    else:
        raise Exception('the problem '+prob+' is not defined in bc_f.py')
    return u, f, fNorm


def ana_soln(problem):
    """get analytical solution"""
    from volume_mf_st import slicing_x_i
    ndim = config.ndim
    if problem == 'stokes' and ndim == 3:
        u_nloc = sf_nd_nb.vel_func_space.element.nloc
        p_nloc = sf_nd_nb.pre_func_space.element.nloc
        nele = config.nele
        u_nonods = u_nloc * nele
        p_nonods = p_nloc * nele
        u_ana = torch.zeros(u_nonods*ndim + p_nonods, device=dev, dtype=torch.float64)
        u, p = slicing_x_i(u_ana)
        for inod in range(u_nonods):
            x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // u_nloc, :, inod % u_nloc]
            # u[inod//u_nloc, inod%u_nloc, :] = 0.
            u[inod // u_nloc, inod % u_nloc, 0] = -2. / 3. * torch.sin(x_inod[0]) ** 3
            u[inod // u_nloc, inod % u_nloc, 1] = torch.sin(x_inod[0]) ** 2 * \
                                                  (x_inod[1] * torch.cos(x_inod[0])
                                                   - x_inod[2] * torch.sin(x_inod[0]))
            u[inod // u_nloc, inod % u_nloc, 2] = torch.sin(x_inod[0]) ** 2 * \
                                                  (x_inod[2] * torch.cos(x_inod[0])
                                                   + x_inod[1] * torch.sin(x_inod[0]))
        for inod in range(p_nonods):
            x_inod = sf_nd_nb.pre_func_space.x_ref_in[inod // p_nloc, :, inod % p_nloc]
            p[inod // p_nloc, inod % p_nloc] = torch.sin(x_inod[0]) - 1. + np.cos(1.)
    else:
        raise Exception('problem analytical solution not detined for '+problem)
    return u_ana
