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

            mu = config.mu_s
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
            mu = config.mu_s
            lam = config.lam_s
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
            lam = config.lam_s
            mu = config.mu_s
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


def fsi_bc(ndim, bc_node_list, x_all, prob: str, t=None):
    nloc = sf_nd_nb.vel_func_space.element.nloc
    nonods = sf_nd_nb.vel_func_space.nonods
    u_bc = [torch.zeros(config.nele, nloc, ndim, device=dev, dtype=torch.float64) for _ in bc_node_list]
    f = torch.zeros((nonods, ndim), device=dev, dtype=torch.float64)
    if prob == 'stokes':
        if ndim == 2:
            # apply boundary conditions (4 Dirichlet bcs)
            # for ibc in range(1):
            #     bci = bc_node_list[ibc]
            #     for inod in range(bci.shape[0]):
            #         if not bci[inod]:  # this is not a boundary node
            #             continue
            #         x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
            #         # u[inod//nloc, inod%nloc, :] = 0.
            #         u_bc[ibc][inod // nloc, inod % nloc, 0] = \
            #             -torch.exp(x_inod[0]) * \
            #             (x_inod[1] * torch.cos(x_inod[1]) + torch.sin(x_inod[1]))
            #         u_bc[ibc][inod // nloc, inod % nloc, 1] = \
            #             torch.exp(x_inod[0]) * x_inod[1] * torch.sin(x_inod[1])
            # for ibc in range(1, len(bc_node_list)):  # neumann boundary
            #     pass  # let's assume there's no neumann boundary

            # poiseuille flow
            ibc = 0
            bci = bc_node_list[0]
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                u_bc[ibc][inod // nloc, inod % nloc, 0] = 1 - x_inod[1] **2
                u_bc[ibc][inod // nloc, inod % nloc, 1] = 0.
            ibc = 1
            bci = bc_node_list[1]
            for inod in range(bci.shape[0]):
                if not bci[inod]:
                    continue
                u_bc[ibc][inod // nloc, inod % nloc, :] = 0.  # 0 stress out-flow condition

            f = np.zeros((nonods, ndim), dtype=np.float64)

            f = torch.tensor(f, device=dev, dtype=torch.float64)
            fNorm = torch.linalg.norm(f.view(-1), dim=0)

        else:  # ndim == 3
            for ibc in range(1):
                bci = bc_node_list[ibc]
                for inod in range(bci.shape[0]):
                    if not bci[inod]:  # this is not a boundary node
                        continue
                    x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                    # u[inod//nloc, inod%nloc, :] = 0.
                    u_bc[ibc][inod // nloc, inod % nloc, 0] = -2./3. * torch.sin(x_inod[0])**3
                    u_bc[ibc][inod // nloc, inod % nloc, 1] = \
                        torch.sin(x_inod[0])**2 * (x_inod[1] * torch.cos(x_inod[0]) - x_inod[2] * torch.sin(x_inod[0]))
                    u_bc[ibc][inod // nloc, inod % nloc, 2] = torch.sin(x_inod[0])**2 * \
                                                      (x_inod[2] * torch.cos(x_inod[0])
                                                       + x_inod[1] * torch.sin(x_inod[0]))
            # neumann bc (at z=1 plane) the rest are neumann bc... what a raw assumption
            for ibc in range(1, len(bc_node_list)):
                print('=== in bc_f : has neumann bc ===')
                bci = bc_node_list[ibc]
                for inod in range(bci.shape[0]):
                    if not bci[inod]:
                        continue
                    x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod%nloc]
                    x = x_inod[0]
                    y = x_inod[1]
                    z = x_inod[2]
                    # unsymmetry stress formulation
                    u_bc[ibc][inod // nloc, inod % nloc, 0] = 0
                    u_bc[ibc][inod // nloc, inod % nloc, 1] = - torch.sin(x) ** 3
                    u_bc[ibc][inod // nloc, inod % nloc, 2] = torch.cos(x) * torch.sin(x) ** 2 - torch.sin(x)

                    # symmetry stress formulation
                    # u_bc[ibc][inod // nloc, inod % nloc, 0] = \
                    #     torch.sin(x) * (3 * torch.cos(x) ** 2 + 3 * y * torch.cos(x) * torch.sin(x) - 1)
                    # u_bc[ibc][inod // nloc, inod % nloc, 1] = 0
                    # u_bc[ibc][inod // nloc, inod % nloc, 2] = torch.sin(x) * (torch.sin(2 * x) - 1)
            f = torch.zeros((nonods, ndim), device=dev, dtype=torch.float64)

            x = torch.tensor(x_all[:, 0], device=dev)
            y = torch.tensor(x_all[:, 1], device=dev)
            z = torch.tensor(x_all[:, 2], device=dev)

            # sin pressure
            # unsymmetric stress formulation
            f[:, 0] = torch.cos(x) - 2 * torch.sin(x) + 6 * torch.cos(x) ** 2 * torch.sin(x)
            f[:, 1] = 7 * y * torch.cos(x) - 9 * y * torch.cos(x) ** 3 - 3 * z * torch.sin(x) + 9 * z * torch.cos(x) ** 2 * torch.sin(x)
            f[:, 2] = 7 * z * torch.cos(x) - 9 * z * torch.cos(x) ** 3 + 3 * y * torch.sin(x) - 9 * y * torch.cos(x) ** 2 * torch.sin(x)

            # # symmetryc tress formulation
            # f[:, 0] = torch.cos(x) - 2*torch.sin(x) + 6*torch.cos(x)**2 * torch.sin(x)
            # f[:, 1] = 7*y*torch.cos(x) - 9*y*torch.cos(x)**3 \
            #           - 3*z*torch.sin(x) + 9*z*torch.cos(x)**2*torch.sin(x)
            # f[:, 2] = 7*z*torch.cos(x) - 9*z*torch.cos(x)**3 \
            #           + 3*y*torch.sin(x) - 9*y*torch.cos(x)**2*torch.sin(x)

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
            # del x, y, z
    elif prob == 'poiseuille' and ndim == 2:
        # poiseuille flow
        for ibc in range(1):
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                u_bc[ibc][inod // nloc, inod % nloc, 0] = x_inod[1] - x_inod[1] ** 2
                u_bc[ibc][inod // nloc, inod % nloc, 1] = 0.
        for ibc in range(1, len(bc_node_list)):
            print('=== in bc_f : has neumann bc ===')
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:
                    continue
                u_bc[ibc][inod // nloc, inod % nloc, :] = 0.  # 0 stress out-flow condition

        f = np.zeros((nonods, ndim), dtype=np.float64)

        f = torch.tensor(f, device=dev, dtype=torch.float64)
        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    elif prob == 'poiseuille' and ndim == 3:
        # poiseuille flow
        for ibc in range(1):
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                # u[inod//nloc, inod%nloc, :] = 0.
                u_bc[ibc][inod // nloc, inod % nloc, 0] = 1 - x_inod[1]**2
                u_bc[ibc][inod // nloc, inod % nloc, 1] = 0
                u_bc[ibc][inod // nloc, inod % nloc, 2] = 0
        # neumann bc (at z=1 plane) the rest are neumann bc... what a raw assumption
        for ibc in range(1, len(bc_node_list)-1):
            print('=== in bc_f : has neumann bc ===')
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod%nloc]
                x = x_inod[0]
                y = x_inod[1]
                z = x_inod[2]
                u_bc[ibc][inod // nloc, inod % nloc, :] = 0  # outflow no stress bc
        # symmetry bc
        for ibc in range(2, len(bc_node_list)):
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                z = x_inod[2]
                # symmetry is too hard... I'll just use dirichlet
                u_bc[ibc][inod // nloc, inod % nloc, 0] = 1 - x_inod[1] ** 2
                u_bc[ibc][inod // nloc, inod % nloc, 1] = 0
                u_bc[ibc][inod // nloc, inod % nloc, 2] = 0
        f = torch.zeros((nonods, ndim), device=dev, dtype=torch.float64)
        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    elif prob == 'ns' and ndim == 3:
        t = torch.tensor(0, device=dev, dtype=torch.float64)
        for ibc in range(1):
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                # t = 0
                x = x_inod[0]
                y = x_inod[1]
                z = x_inod[2]
                # u[inod//u_nloc, inod%u_nloc, :] = 0.
                u_bc[0][inod // nloc, inod % nloc, 0] = (- torch.exp(z - t) * torch.cos(x + y)
                                                         - torch.exp(x - t) * torch.sin(y + z))
                u_bc[0][inod // nloc, inod % nloc, 1] = (- torch.exp(x - t) * torch.cos(y + z)
                                                         - torch.exp(y - t) * torch.sin(x + z))
                u_bc[0][inod // nloc, inod % nloc, 2] = (- torch.exp(y - t) * torch.cos(x + z)
                                                         - torch.exp(z - t) * torch.sin(x + y))
        # neumann bc (at z=1 plane) the rest are neumann bc... what a raw assumption
        for ibc in range(1, len(bc_node_list)):
            print('=== in bc_f : has neumann bc ===')
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                # t = torch.tensor(0)
                x = x_inod[0]
                y = x_inod[1]
                z = x_inod[2]
                # unsymmetry stress formulation
                u_bc[ibc][inod // nloc, inod % nloc, 0] = - torch.cos(y + 1) * torch.exp(x - t) - torch.exp(1 - t) * torch.cos(x + y)
                u_bc[ibc][inod // nloc, inod % nloc, 1] = torch.sin(y + 1) * torch.exp(x - t) - torch.cos(x + 1) * torch.exp(y - t)
                u_bc[ibc][inod // nloc, inod % nloc, 2] = \
                    torch.sin(x + 1) * torch.exp(y - t) - torch.exp(1 - t) * torch.sin(x + y) + torch.exp(-2 * t) * (
                            torch.exp(2 * x) / 2 + torch.exp(2 * y) / 2 + np.exp(2) / 2 + torch.cos(
                        x + 1) * torch.exp(y + 1) * torch.sin(x + y) + torch.cos(
                        y + 1) * torch.sin(x + 1) * torch.exp(x + y) + torch.sin(y + 1) * torch.exp(x + 1) * torch.cos(
                        x + y))

        f = torch.zeros((nonods, ndim), device=dev, dtype=torch.float64)
        # t = 0
        x = torch.tensor(x_all[:, 0], device=dev)
        y = torch.tensor(x_all[:, 1], device=dev)
        z = torch.tensor(x_all[:, 2], device=dev)

        # sin pressure
        # unsymmetric stress formulation
        f[:, 0] = - torch.exp(z - t)*torch.cos(x + y) - torch.exp(x - t)*torch.sin(y + z)
        f[:, 1] = - torch.exp(x - t)*torch.cos(y + z) - torch.exp(y - t)*torch.sin(x + z)
        f[:, 2] = - torch.exp(y - t)*torch.cos(x + z) - torch.exp(z - t)*torch.sin(x + y)
        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    elif prob == 'ns' and ndim == 2:
        Re = torch.tensor(1/config.mu_f, device=dev, dtype=torch.float64)
        for ibc in range(1):
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                u_bc[0][inod // nloc, inod % nloc, 0] = (x**2*y*(y**2 - 2)*(x - 2)**2)/4
                u_bc[0][inod // nloc, inod % nloc, 1] = -(x*y**2*(y**2 - 4)*(x**2 - 3*x + 2))/4
        # neumann bc (at x=1 plane) the rest are neumann bc... what a raw assumption
        for ibc in range(1, len(bc_node_list)):
            print('=== in bc_f : has neumann bc ===')
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                # unsymmetry stress formulation
                u_bc[ibc][inod // nloc, inod % nloc, 0] = \
                    (y ** 2 * (y ** 4 - 2 * y ** 2 + 8)) / 128 - (8 * y) / (5 * Re) + 8 / (5 * Re) - 1408 / 33075
                u_bc[ibc][inod // nloc, inod % nloc, 1] = (y**2*(y**2 - 4))/(4*Re)
        f = torch.zeros((nonods, ndim), device=dev, dtype=torch.float64)
        # t = 0
        x = torch.tensor(x_all[:, 0], device=dev)
        y = torch.tensor(x_all[:, 1], device=dev)

        # unsymmetric stress formulation
        f[:, 0] = \
            (x * y * (12 * x ** 3 - 45 * x ** 2 + 20 * x * y ** 2 - 30 * y ** 2 + 60)) / (5 * Re) - (
                        y * (30 * x * (y ** 2 - 2) - 10 * x ** 2 * y ** 2 + 15 * x ** 3 - 3 * x ** 4 - 20 * y ** 2 + 40)) / (
                        5 * Re) - (3 * x ** 2 * y * (x - 2) ** 2) / (2 * Re) - (
                        x ** 3 * y ** 2 * (x - 2) ** 4 * (y ** 4 - 2 * y ** 2 + 8)) / 32 - (
                        x ** 4 * y ** 2 * (x - 2) ** 3 * (y ** 4 - 2 * y ** 2 + 8)) / 32 - (
                        (y * (y ** 2 - 2) * (x - 2) ** 2) / 2 + (x ** 2 * y * (y ** 2 - 2)) / 2 + x * y * (2 * x - 4) * (
                            y ** 2 - 2)) / Re - (x * y ** 2 * (y ** 2 - 4) * (
                        (x ** 2 * y ** 2 * (x - 2) ** 2) / 2 + (x ** 2 * (y ** 2 - 2) * (x - 2) ** 2) / 4) * (
                                                            x ** 2 - 3 * x + 2)) / 4 + (x ** 2 * y * (y ** 2 - 2) * (
                        (x ** 2 * y * (2 * x - 4) * (y ** 2 - 2)) / 4 + (x * y * (y ** 2 - 2) * (x - 2) ** 2) / 2) * (
                                                                                                   x - 2) ** 2) / 4
        f[:, 1] = \
            ((x * (y ** 2 - 4) * (x ** 2 - 3 * x + 2)) / 2 + (5 * x * y ** 2 * (x ** 2 - 3 * x + 2)) / 2) / Re + (
                    (y ** 2 * (2 * x - 3) * (y ** 2 - 4)) / 2 + (x * y ** 2 * (y ** 2 - 4)) / 2) / Re - (
                    x * (30 * x * (y ** 2 - 2) - 10 * x ** 2 * y ** 2 + 15 * x ** 3 - 3 * x ** 4 - 20 * y ** 2 + 40)) / (
                    5 * Re) + (x * y * (20 * y * x ** 2 - 60 * y * x + 40 * y)) / (5 * Re) - (
                    x ** 4 * y * (x - 2) ** 4 * (y ** 4 - 2 * y ** 2 + 8)) / 64 + (
                    x ** 4 * y ** 2 * (- 4 * y ** 3 + 4 * y) * (x - 2) ** 4) / 128 - (
                    x ** 2 * y * (y ** 2 - 2) * (x - 2) ** 2 * ((y ** 2 * (y ** 2 - 4) * (x ** 2 - 3 * x + 2)) / 4 + (
                        x * y ** 2 * (2 * x - 3) * (y ** 2 - 4)) / 4)) / 4 + (x * y ** 2 * (y ** 2 - 4) * (
                    (x * y ** 3 * (x ** 2 - 3 * x + 2)) / 2 + (x * y * (y ** 2 - 4) * (x ** 2 - 3 * x + 2)) / 2) * (
                                                                                        x ** 2 - 3 * x + 2)) / 4
        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    elif prob == 'kovasznay' and ndim == 2:
        Re = torch.tensor(1/config.mu_f, device=dev, dtype=torch.float64)
        for ibc in range(1):
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                u_bc[0][inod // nloc, inod % nloc, 0] = 1 - torch.exp(
                    (Re / 2 - torch.sqrt(Re ** 2 / 4 + 4 * np.pi ** 2)) * x) * torch.cos(2 * np.pi * y)
                u_bc[0][inod // nloc, inod % nloc, 1] = \
                    (Re / 2 - torch.sqrt(Re ** 2 / 4 + 4 * np.pi ** 2)) / (2 * np.pi) * torch.exp(
                          (Re / 2 - torch.sqrt(Re ** 2 / 4 + 4 * np.pi ** 2)) * x) \
                      * torch.sin(2 * np.pi * y)
        # neumann bc (at x=1 plane) the rest are neumann bc... what a raw assumption
        for ibc in range(1, len(bc_node_list)):
            print('=== in bc_f : has neumann bc ===')
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                # unsymmetry stress formulation
                u_bc[ibc][inod // nloc, inod % nloc, 0] = (
                        torch.exp(Re - 2 * (Re ** 2 / 4 + 4 * torch.pi ** 2) ** (1 / 2)) / 2 - (
                        torch.exp(Re / 2 - (Re ** 2 / 4 + 4 * torch.pi ** 2) ** (1 / 2)) * torch.cos(
                    2 * torch.pi * y) * (
                                Re / 2 - (Re ** 2 / 4 + 4 * torch.pi ** 2) ** (1 / 2))) / Re)
                u_bc[ibc][inod // nloc, inod % nloc, 1] = \
                    (torch.exp(Re / 2 - (Re ** 2 / 4 + 4 * torch.pi ** 2) ** (1 / 2)) * torch.sin(2 * torch.pi * y) * (
                            Re / 2 - (Re ** 2 / 4 + 4 * torch.pi ** 2) ** (1 / 2)) * (
                             Re / 4 - (Re ** 2 / 4 + 4 * torch.pi ** 2) ** (1 / 2) / 2)) / (Re * torch.pi)

        f = torch.zeros((nonods, ndim), device=dev, dtype=torch.float64)

        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    elif prob == 'ldc' and ndim == 2:  # lid-driven cavity
        for ibc in range(1):
            bci = bc_node_list[ibc]
            u_m = 1.  # max velocity @ boundary
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                if torch.abs(y) < 1e-10 or torch.abs(x) < 1e-10 or torch.abs(x-1) < 1e-10:
                    u_bc[0][inod // nloc, inod % nloc, 0] = 0
                else:
                    u_bc[0][inod // nloc, inod % nloc, 0] = (1 - (2*x - 1)**4) * u_m
                    # if config.isTransient:  # ramp up
                    #     u_bc[0][inod // nloc, inod % nloc, 0] *= np.min([1.0, t])
                u_bc[0][inod // nloc, inod % nloc, 1] = 0
        # neumann bc (at x=1 plane) the rest are neumann bc... what a raw assumption
        for ibc in range(1, len(bc_node_list)):
            print('=== in bc_f : has neumann bc === but please dont have this for ldc problem')
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                # unsymmetry stress formulation
                u_bc[ibc][inod // nloc, inod % nloc, 0] = 0
                u_bc[ibc][inod // nloc, inod % nloc, 1] = 0

        f = torch.zeros((nonods, ndim), device=dev, dtype=torch.float64)

        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    elif prob == 'tgv' and ndim == 2:  # taylor-green vortex
        if t is None:
            raise ValueError('should provide time when getting bc condition for taylor-green vortex problem.')
        Re = torch.tensor(1/config.mu_f, device=dev, dtype=torch.float64)
        for ibc in range(1):
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                u_bc[0][inod // nloc, inod % nloc, 0] = \
                    -torch.exp(-(2 * torch.pi ** 2 * t) / Re) * torch.cos(torch.pi * x) * torch.sin(torch.pi * y)
                u_bc[0][inod // nloc, inod % nloc, 1] = \
                    torch.exp(-(2 * torch.pi ** 2 * t) / Re) * torch.cos(torch.pi * y) * torch.sin(torch.pi * x)
        # neumann bc (at x=1 plane) the rest are neumann bc... what a raw assumption
        for ibc in range(1, len(bc_node_list)):
            print('=== in bc_f : has neumann bc === but please try to avoid having this for tgv problem')
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                # unsymmetry stress formulation
                u_bc[ibc][inod // nloc, inod % nloc, 0] = \
                    torch.exp(-(4 * torch.pi ** 2 * t) / Re) * (np.cos(2 * torch.pi) / 4 + torch.cos(2 * torch.pi * y) / 4) + (
                                torch.pi * torch.exp(-(2 * torch.pi ** 2 * t) / Re) * torch.sin(torch.pi * y) * np.sin(torch.pi)) / Re
                u_bc[ibc][inod // nloc, inod % nloc, 1] = \
                    (torch.pi * torch.exp(-(2 * torch.pi ** 2 * t) / Re) * torch.cos(torch.pi * y) * np.cos(torch.pi)) / Re

        f = torch.zeros((nonods, ndim), device=dev, dtype=torch.float64)

        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    elif prob == 'tgv10' and ndim == 2:  # taylor-green vortex
        if t is None:
            raise ValueError('should provide time when getting bc condition for taylor-green vortex problem.')
        Re = torch.tensor(1/config.mu_f, device=dev, dtype=torch.float64)
        u_m = 10.
        for ibc in range(1):
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                u_bc[0][inod // nloc, inod % nloc, 0] = \
                    -torch.exp(-(2 * torch.pi ** 2 * t) / Re) * torch.cos(torch.pi * x) * torch.sin(torch.pi * y) * u_m
                u_bc[0][inod // nloc, inod % nloc, 1] = \
                    torch.exp(-(2 * torch.pi ** 2 * t) / Re) * torch.cos(torch.pi * y) * torch.sin(torch.pi * x) * u_m
        # neumann bc (at x=1 plane) the rest are neumann bc... what a raw assumption
        for ibc in range(1, len(bc_node_list)):
            print('=== in bc_f : has neumann bc === but please try to avoid having this for tgv problem')
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                # unsymmetry stress formulation
                u_bc[ibc][inod // nloc, inod % nloc, 0] = \
                    u_m **2 * torch.exp(-(4 * torch.pi ** 2 * t) / Re) * (np.cos(2 * torch.pi) / 4 + torch.cos(2 * torch.pi * y) / 4) + (
                                u_m * torch.pi * torch.exp(-(2 * torch.pi ** 2 * t) / Re) * torch.sin(torch.pi * y) * np.sin(torch.pi)) / Re
                u_bc[ibc][inod // nloc, inod % nloc, 1] = \
                    u_m * (torch.pi * torch.exp(-(2 * torch.pi ** 2 * t) / Re) * torch.cos(torch.pi * y) * np.cos(torch.pi)) / Re

        f = torch.zeros((nonods, ndim), device=dev, dtype=torch.float64)

        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    elif prob == 'bfs' and ndim == 2:
        for ibc in range(1):
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                u_bc[0][inod // nloc, inod % nloc, 0] = 0
                u_bc[0][inod // nloc, inod % nloc, 1] = 0
                if torch.abs(x + 2) < 1e-10:  # this is inlet
                    u_bc[0][inod // nloc, inod % nloc, 0] = 1 - y ** 2
        # neumann bc (at x=1 plane) the rest are neumann bc... what a raw assumption
        for ibc in range(1, len(bc_node_list)):
            print('=== in bc_f : has neumann bc ===')
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                # unsymmetry stress formulation
                u_bc[ibc][inod // nloc, inod % nloc, 0] = 0
                u_bc[ibc][inod // nloc, inod % nloc, 1] = 0

        f = torch.zeros((nonods, ndim), device=dev, dtype=torch.float64)

        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    elif prob == 'nozzle' and ndim == 3:
        for ibc in range(1):
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                z = x_inod[2]
                if torch.abs(z + 0.11114916161319763) < 1e-6:
                    u_bc[0][inod // nloc, inod % nloc, 2] = 1
                # u_bc[0][inod // nloc, inod % nloc, 2] = ibc + 1

        f = torch.zeros((nonods, ndim), device=dev, dtype=torch.float64)

        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    elif prob == 'fpc' and ndim == 2:  # flow-past cylinder
        if t is None:
            raise ValueError('should provide time when getting bc condition for flow-past cylinder problem.')
        u_m = 1.5  # max velocity
        T = 8  # half period
        for ibc in range(1):
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                if torch.abs(x) < 1e-8:  # this is x=0 inlet boundary
                    u_bc[0][inod // nloc, inod % nloc, 0] = \
                        u_m * 4 * y * (0.41 - y) / 0.41**2
                        # * torch.sin(
                        #     torch.tensor(torch.pi * t / T, device=dev, dtype=torch.float64))
                else:
                    u_bc[0][inod // nloc, inod % nloc, 0] = 0
                u_bc[0][inod // nloc, inod % nloc, 1] = 0
        # neumann bc (at x=1 plane) the rest are neumann bc... what a raw assumption
        for ibc in range(1, len(bc_node_list)):
            print('=== in bc_f : has neumann bc ===')
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                # unsymmetry stress formulation
                u_bc[ibc][inod // nloc, inod % nloc, 0] = 0
                u_bc[ibc][inod // nloc, inod % nloc, 1] = 0

        f = torch.zeros((nonods, ndim), device=dev, dtype=torch.float64)

        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    elif prob == 'fsi-test' and ndim == 3:
        t = torch.tensor(t, device=dev, dtype=torch.float64)
        for ibc in range(len(bc_node_list)):
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                # t = 0
                x = x_inod[0]
                y = x_inod[1]
                z = x_inod[2]
                # u[inod//u_nloc, inod%u_nloc, :] = 0.
                u_bc[ibc][inod // nloc, inod % nloc, 0] = ibc+1
                u_bc[ibc][inod // nloc, inod % nloc, 1] = ibc+1
                u_bc[ibc][inod // nloc, inod % nloc, 2] = ibc+1

        f = torch.zeros((nonods, ndim), device=dev, dtype=torch.float64)
        # t = 0
        x = torch.tensor(x_all[:, 0], device=dev)
        y = torch.tensor(x_all[:, 1], device=dev)
        z = torch.tensor(x_all[:, 2], device=dev)

        # sin pressure
        # unsymmetric stress formulation
        f[:, 0] = - torch.exp(z - t)*torch.cos(x + y) - torch.exp(x - t)*torch.sin(y + z)
        f[:, 1] = - torch.exp(x - t)*torch.cos(y + z) - torch.exp(y - t)*torch.sin(x + z)
        f[:, 2] = - torch.exp(y - t)*torch.cos(x + z) - torch.exp(z - t)*torch.sin(x + y)
        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    elif prob == 'fsi-test' and ndim == 2:
        Re = torch.tensor(1/config.mu_f, device=dev, dtype=torch.float64)
        for ibc in range(len(bc_node_list)):
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                u_bc[ibc][inod // nloc, inod % nloc, 0] = ibc+1
                u_bc[ibc][inod // nloc, inod % nloc, 1] = ibc+1

        f = torch.zeros((nonods, ndim), device=dev, dtype=torch.float64)

        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    elif prob == 'fsi-poiseuille' and ndim == 2:
        # poiseuille flow but with a flexible wall with thickness 0.25
        # steady problem
        for ibc in range(1):  # fluid dirichlet bc
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.disp_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                if torch.abs(x) < 1e-8:  # this is x=0 inlet boundary
                    u_bc[ibc][inod // nloc, inod % nloc, 0] = \
                        y - y**2  # -y^2 + y
                    u_bc[ibc][inod // nloc, inod % nloc, 1] = 0
                else:  # this is wall, do nothing
                    pass
        # neumann bc are all zero
        # solid diri bc is also zero
        f = torch.zeros((nonods, ndim), device=dev, dtype=torch.float64)
        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    elif prob == 'turek' and ndim == 2:
        # turek benchmark
        # 3 kinds of BC: fluid diri; fluid neu; solid diri
        for ibc in range(1):
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.disp_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                if torch.abs(x) < 1e-8:  # this is x=0 inlet boundary
                    u_bc[ibc][inod // nloc, inod % nloc, 0] = \
                        1.5 * 2 * 4 * y * (0.41 - y) / 0.41**2
                    u_bc[ibc][inod // nloc, inod % nloc, 1] = 0
                else:  # this is wall, do nothing
                    pass
            if t < 2.0:
                u_bc[ibc] *= (1 - np.cos(np.pi * t / 2.0)) / 2.0
        # neumann bc are all zero
        # solid diri bc is also zero
        f = torch.zeros((nonods, ndim), device=dev, dtype=torch.float64)
        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    elif prob == 'seeweed' and ndim == 2:
        # seeweed in channel flow
        for ibc in range(1):
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.disp_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                if torch.abs(x) < 1e-8:  # this is x=0 inlet boundary
                    u_bc[ibc][inod // nloc, inod % nloc, 0] = \
                        1.5 * 1 * 4 * y * (0.5 - y) / 0.5**2
                    u_bc[ibc][inod // nloc, inod % nloc, 1] = 0
                else:  # this is wall, do nothing
                    pass
            if t < 2.0:
                u_bc[ibc] *= (1 - np.cos(np.pi * t / 2.0)) / 2.0
        # neumann bc are all zero
        # solid diri bc is also zero
        f = torch.zeros((nonods, ndim), device=dev, dtype=torch.float64)
        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    elif prob == 'hyper-elastic' and ndim == 3:
        # analytical solution from Abbas 2018
        alpha = 0.1
        gamma = 0.1
        lam = config.lam_s
        mu = config.mu_s
        for ibc in range(2, 3):
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.disp_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                z = x_inod[2]
                theta = alpha * torch.sin(torch.pi * x_inod[1])  # alpha sin(pi Y)
                g = gamma * torch.sin(torch.pi * x_inod[0])  # gamma sin(pi X)
                h = 0  # 0

                u_bc[ibc][inod // nloc, inod % nloc, 0] = (1. / lam + alpha) * x_inod[0] + theta
                u_bc[ibc][inod // nloc, inod % nloc, 1] = -(1. / lam + (alpha + gamma + alpha * gamma) /
                                                    (1 + alpha + gamma + alpha * gamma)) * x_inod[1]
                u_bc[ibc][inod // nloc, inod % nloc, 2] = (1. / lam + alpha) * x_inod[2] + g + h
        x = torch.tensor(x_all[:, 0], device=dev)
        y = torch.tensor(x_all[:, 1], device=dev)
        z = torch.tensor(x_all[:, 2], device=dev)
        f[:, 0] = mu * alpha * torch.pi ** 2 * torch.sin(torch.pi * y)
        f[:, 1] = 0.
        f[:, 2] = mu * gamma * torch.pi ** 2 * torch.sin(torch.pi * x)
        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    elif prob == 'he_test' and ndim == 3:
        # uni-axial strentch test
        for ibc in range(3, 4):
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.disp_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                z = x_inod[2]
                if torch.abs(x-1) < 1e-8:  # x = 1 plane to apply force
                    u_bc[ibc][inod // nloc, inod % nloc, 0] = 1.
                    # u_bc[ibc][inod // nloc, inod % nloc, 1] = -(1. / lam + (alpha + gamma + alpha * gamma) /
                    #                                             (1 + alpha + gamma + alpha * gamma)) * x_inod[1]
                    # u_bc[ibc][inod // nloc, inod % nloc, 2] = (1. / lam + alpha) * x_inod[2] + g + h
        x = torch.tensor(x_all[:, 0], device=dev)
        y = torch.tensor(x_all[:, 1], device=dev)
        z = torch.tensor(x_all[:, 2], device=dev)
        # f[:, 0] = mu * alpha * torch.pi ** 2 * torch.sin(torch.pi * y)
        # f[:, 1] = 0.
        # f[:, 2] = mu * gamma * torch.pi ** 2 * torch.sin(torch.pi * x)
        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    elif prob == 'he_test_translation' and ndim == 3:
        # "rigid-body" translation test
        for ibc in range(2, 3):
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.disp_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                z = x_inod[2]

                u_bc[ibc][inod // nloc, inod % nloc, 0] = np.sin(np.pi / 2 * t)
                u_bc[ibc][inod // nloc, inod % nloc, 1] = np.sin(np.pi / 2 * t)
                u_bc[ibc][inod // nloc, inod % nloc, 2] = np.sin(np.pi / 2 * t)
        x = torch.tensor(x_all[:, 0], device=dev)
        y = torch.tensor(x_all[:, 1], device=dev)
        z = torch.tensor(x_all[:, 2], device=dev)
        # f[:, 0] = mu * alpha * torch.pi ** 2 * torch.sin(torch.pi * y)
        # f[:, 1] = 0.
        # f[:, 2] = mu * gamma * torch.pi ** 2 * torch.sin(torch.pi * x)
        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    elif prob == 'he_test' and ndim == 2:
        # uni-axial strentch test
        # for ibc in range(3, 4):
        #     bci = bc_node_list[ibc]
        #     for inod in range(bci.shape[0]):
        #         if not bci[inod]:  # this is not a boundary node
        #             continue
        #         x_inod = sf_nd_nb.disp_func_space.x_ref_in[inod // nloc, :, inod % nloc]
        #         x = x_inod[0]
        #         y = x_inod[1]
        #         if torch.abs(y - 1) < 1e-8:  # y = 1 plane to apply force
        #             u_bc[ibc][inod // nloc, inod % nloc, 1] = 0.  # -1000. * (1 + 0.001 * y) * (1 + 0.001 * x)
        #         if torch.abs(x - 1) < 1e-8:  # x = 1 plane to apply force
        #             u_bc[ibc][inod // nloc, inod % nloc, 1] = 0.  # -1000. * (1 + 0.001 * y) * (1 + 0.001 * x)
        #         if torch.abs(x) < 1e-8:  # x = 0 plane
        #             u_bc[ibc][inod // nloc, inod % nloc, 1] = 0.  # 1000. * (1 + 0.001 * y) * (1 + 0.001 * x)
        #             # u_bc[ibc][inod // nloc, inod % nloc, 1] = -(1. / lam + (alpha + gamma + alpha * gamma) /
        #             #                                             (1 + alpha + gamma + alpha * gamma)) * x_inod[1]
        #             # u_bc[ibc][inod // nloc, inod % nloc, 2] = (1. / lam + alpha) * x_inod[2] + g + h
        x = torch.tensor(x_all[:, 0], device=dev)
        y = torch.tensor(x_all[:, 1], device=dev)
        f[:, 0] = 9.81
        # f[:, 0] = mu * alpha * torch.pi ** 2 * torch.sin(torch.pi * y)
        # f[:, 1] = 0.
        # f[:, 2] = mu * gamma * torch.pi ** 2 * torch.sin(torch.pi * x)
        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    elif prob == 'ls1' and ndim == 2:
        # channel with flexible wall (vertical, like a seeweed), from La Spina 2020 section 4.4
        for ibc in range(0, 1):
            bci = bc_node_list[ibc]
            for inod in range(bci.shape[0]):
                if not bci[inod]:  # this is not a boundary node
                    continue
                x_inod = sf_nd_nb.disp_func_space.x_ref_in[inod // nloc, :, inod % nloc]
                x = x_inod[0]
                y = x_inod[1]
                if torch.abs(x - 0) < 1e-8:  # x = 1 fluid inlet
                    u_bc[ibc][inod // nloc, inod % nloc, 0] = (1 - 4 * y ** 2) * 0.06067 * 956
            if t < 10:
                u_bc[ibc] *= (1 - np.cos(np.pi * t / 10.0)) / 2.0
        fNorm = torch.linalg.norm(f.view(-1), dim=0)
    else:
        raise Exception('the problem ' + prob + ' is not defined in fsi_bc_f')
    return u_bc, f, fNorm


def ana_soln(problem, t=None):
    """get analytical solution"""
    from volume_mf_st import slicing_x_i
    ndim = config.ndim
    if problem == 'stokes' and ndim == 3:
        u_nloc = sf_nd_nb.vel_func_space.element.nloc
        p_nloc = sf_nd_nb.pre_func_space.element.nloc
        nele = config.nele
        u_nonods = u_nloc * nele
        p_nonods = p_nloc * nele
        u_ana = torch.zeros(u_nonods * ndim + u_nonods, device=dev, dtype=torch.float64)
        u, p = (u_ana[0:u_nonods*ndim].view(nele, u_nloc, ndim),
                u_ana[u_nonods*ndim:u_nonods*ndim+u_nonods].view(nele, u_nloc))
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
        for inod in range(u_nonods):
            x_inod = sf_nd_nb.pre_func_space.x_ref_in[inod // u_nloc, :, inod % u_nloc]
            p[inod // u_nloc, inod % u_nloc] = torch.sin(x_inod[0])  # - 1. + np.cos(1.)

        # # poiseuille flow
        # for inod in range(u_nonods):
        #     x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // u_nloc, :, inod % u_nloc]
        #     # u[inod//u_nloc, inod%u_nloc, :] = 0.
        #     u[inod // u_nloc, inod % u_nloc, 0] = 1 - x_inod[1]**2
        #     u[inod // u_nloc, inod % u_nloc, 1] = 0
        #     u[inod // u_nloc, inod % u_nloc, 2] = 0
        # for inod in range(p_nonods):
        #     x_inod = sf_nd_nb.pre_func_space.x_ref_in[inod // p_nloc, :, inod % p_nloc]
        #     p[inod // p_nloc, inod % p_nloc] = -2 * x_inod[0] + 2
    elif problem == 'stokes' and ndim == 2:
        # poiseuille flow
        u_nloc = sf_nd_nb.vel_func_space.element.nloc
        p_nloc = sf_nd_nb.pre_func_space.element.nloc
        nele = config.nele
        u_nonods = u_nloc * nele
        p_nonods = p_nloc * nele
        u_ana = torch.zeros(u_nonods * ndim + u_nonods, device=dev, dtype=torch.float64)
        u, p = (u_ana[0:u_nonods*ndim].view(nele, u_nloc, ndim),
                u_ana[u_nonods*ndim:u_nonods*ndim+u_nonods].view(nele, u_nloc))
        for inod in range(u_nonods):
            x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // u_nloc, :, inod % u_nloc]
            # u[inod//u_nloc, inod%u_nloc, :] = 0.
            u[inod // u_nloc, inod % u_nloc, 0] = 1 - x_inod[1]**2
            u[inod // u_nloc, inod % u_nloc, 1] = 0
        for inod in range(u_nonods):
            x_inod = sf_nd_nb.pre_func_space.x_ref_in[inod // u_nloc, :, inod % u_nloc]
            p[inod // u_nloc, inod % u_nloc] = -2 * x_inod[0] + 2
    elif problem == 'poiseuille' and ndim == 3:
        u_nloc = sf_nd_nb.vel_func_space.element.nloc
        p_nloc = sf_nd_nb.pre_func_space.element.nloc
        nele = config.nele
        u_nonods = u_nloc * nele
        p_nonods = p_nloc * nele
        u_ana = torch.zeros(u_nonods * ndim + u_nonods, device=dev, dtype=torch.float64)
        u, p = (u_ana[0:u_nonods * ndim].view(nele, u_nloc, ndim),
                u_ana[u_nonods * ndim:u_nonods * ndim + u_nonods].view(nele, u_nloc))
        # t = 0  # steady sln
        t = torch.tensor(0, device=dev, dtype=torch.float64)
        for inod in range(u_nonods):
            x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // u_nloc, :, inod % u_nloc]
            x = x_inod[0]
            y = x_inod[1]
            z = x_inod[2]
            # u[inod//u_nloc, inod%u_nloc, :] = 0.
            u[inod // u_nloc, inod % u_nloc, 0] = 1 - x_inod[1] ** 2
            u[inod // u_nloc, inod % u_nloc, 1] = 0
            u[inod // u_nloc, inod % u_nloc, 2] = 0
        for inod in range(u_nonods):
            x_inod = sf_nd_nb.pre_func_space.x_ref_in[inod // u_nloc, :, inod % u_nloc]
            x = x_inod[0]
            y = x_inod[1]
            z = x_inod[2]
            p[inod // u_nloc, inod % u_nloc] = -2*x+2
    elif problem == 'poiseuille' and ndim == 2:
        # poiseuille flow
        u_nloc = sf_nd_nb.vel_func_space.element.nloc
        p_nloc = sf_nd_nb.pre_func_space.element.nloc
        nele = config.nele
        u_nonods = u_nloc * nele
        p_nonods = p_nloc * nele
        u_ana = torch.zeros(u_nonods * ndim + u_nonods, device=dev, dtype=torch.float64)
        u, p = (u_ana[0:u_nonods * ndim].view(nele, u_nloc, ndim),
                u_ana[u_nonods * ndim:u_nonods * ndim + u_nonods].view(nele, u_nloc))
        for inod in range(u_nonods):
            x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // u_nloc, :, inod % u_nloc]
            # u[inod//u_nloc, inod%u_nloc, :] = 0.
            u[inod // u_nloc, inod % u_nloc, 0] = x_inod[1] - x_inod[1] ** 2  # -y^2 + y
            u[inod // u_nloc, inod % u_nloc, 1] = 0
        for inod in range(u_nonods):
            x_inod = sf_nd_nb.pre_func_space.x_ref_in[inod // u_nloc, :, inod % u_nloc]
            p[inod // u_nloc, inod % u_nloc] = -2 * x_inod[0] + 2 - 1  # -1 to make ave = 0
    elif problem == 'ns' and ndim == 3:
        # Beltrami flow
        u_nloc = sf_nd_nb.vel_func_space.element.nloc
        p_nloc = sf_nd_nb.pre_func_space.element.nloc
        nele = config.nele
        u_nonods = u_nloc * nele
        p_nonods = p_nloc * nele
        u_ana = torch.zeros(u_nonods * ndim + u_nonods, device=dev, dtype=torch.float64)
        u, p = (u_ana[0:u_nonods*ndim].view(nele, u_nloc, ndim),
                u_ana[u_nonods*ndim:u_nonods*ndim+u_nonods].view(nele, u_nloc))
        # t = 0  # steady sln
        t = torch.tensor(0, device=dev, dtype=torch.float64)
        for inod in range(u_nonods):
            x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // u_nloc, :, inod % u_nloc]
            x = x_inod[0]
            y = x_inod[1]
            z = x_inod[2]
            # u[inod//u_nloc, inod%u_nloc, :] = 0.
            u[inod // u_nloc, inod % u_nloc, 0] = (- torch.exp(z - t)*torch.cos(x + y)
                                                   - torch.exp(x - t)*torch.sin(y + z))
            u[inod // u_nloc, inod % u_nloc, 1] = (- torch.exp(x - t)*torch.cos(y + z)
                                                   - torch.exp(y - t)*torch.sin(x + z))
            u[inod // u_nloc, inod % u_nloc, 2] = (- torch.exp(y - t)*torch.cos(x + z)
                                                   - torch.exp(z - t)*torch.sin(x + y))
        for inod in range(u_nonods):
            x_inod = sf_nd_nb.pre_func_space.x_ref_in[inod // u_nloc, :, inod % u_nloc]
            x = x_inod[0]
            y = x_inod[1]
            z = x_inod[2]
            p[inod // u_nloc, inod % u_nloc] = -torch.exp(-2*t)*(torch.exp(2*x)/2 + torch.exp(2*y)/2 + torch.exp(2*z)/2
                                                                 + torch.exp(x + y)*torch.cos(y + z)*torch.sin(x + z)
                                                                 + torch.exp(x + z)*torch.cos(x + y)*torch.sin(y + z)
                                                                 + torch.exp(y + z)*torch.cos(x + z)*torch.sin(x + y))
                                                                 # - 7.639581710561036)  # make ave pre be 0
    elif problem == 'ns' and ndim == 2:
        # from Farrell 2019
        u_nloc = sf_nd_nb.vel_func_space.element.nloc
        p_nloc = sf_nd_nb.pre_func_space.element.nloc
        nele = config.nele
        u_nonods = u_nloc * nele
        p_nonods = p_nloc * nele
        u_ana = torch.zeros(u_nonods * ndim + u_nonods, device=dev, dtype=torch.float64)
        u, p = (u_ana[0:u_nonods*ndim].view(nele, u_nloc, ndim),
                u_ana[u_nonods*ndim:u_nonods*ndim+u_nonods].view(nele, u_nloc))
        # t = 0  # steady sln
        Re = torch.tensor(1./config.mu_f, device=dev, dtype=torch.float64)
        for inod in range(u_nonods):
            x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // u_nloc, :, inod % u_nloc]
            x = x_inod[0]
            y = x_inod[1]

            u[inod // u_nloc, inod % u_nloc, 0] = (x**2*y*(y**2 - 2)*(x - 2)**2)/4
            u[inod // u_nloc, inod % u_nloc, 1] = -(x*y**2*(y**2 - 4)*(x**2 - 3*x + 2))/4
        for inod in range(u_nonods):
            x_inod = sf_nd_nb.pre_func_space.x_ref_in[inod // u_nloc, :, inod % u_nloc]
            x = x_inod[0]
            y = x_inod[1]

            p[inod // u_nloc, inod % u_nloc] = \
                1408 / 33075 - (x * y * (
                            30 * x * (y ** 2 - 2) - 10 * x ** 2 * y ** 2 + 15 * x ** 3 - 3 * x ** 4 - 20 * y ** 2 + 40)) / (
                            5 * Re) - (x ** 4 * y ** 2 * (x - 2) ** 4 * (y ** 4 - 2 * y ** 2 + 8)) / 128 - 8 / (5 * Re)
    elif problem == 'kovasznay' and ndim == 2:
        # from Farrell 2019
        u_nloc = sf_nd_nb.vel_func_space.element.nloc
        p_nloc = sf_nd_nb.pre_func_space.element.nloc
        nele = config.nele
        u_nonods = u_nloc * nele
        p_nonods = p_nloc * nele
        u_ana = torch.zeros(u_nonods * ndim + u_nonods, device=dev, dtype=torch.float64)
        u, p = (u_ana[0:u_nonods*ndim].view(nele, u_nloc, ndim),
                u_ana[u_nonods*ndim:u_nonods*ndim+u_nonods].view(nele, u_nloc))
        # t = 0  # steady sln
        Re = torch.tensor(1/config.mu_f, device=dev, dtype=torch.float64)
        for inod in range(u_nonods):
            x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // u_nloc, :, inod % u_nloc]
            x = x_inod[0]
            y = x_inod[1]

            u[inod // u_nloc, inod % u_nloc, 0] = 1 - torch.exp(
                    (Re / 2 - torch.sqrt(Re ** 2 / 4 + 4 * np.pi ** 2)) * x) * torch.cos(2 * np.pi * y)
            u[inod // u_nloc, inod % u_nloc, 1] = \
                    (Re / 2 - torch.sqrt(Re ** 2 / 4 + 4 * np.pi ** 2)) / (2 * np.pi) * torch.exp(
                          (Re / 2 - torch.sqrt(Re ** 2 / 4 + 4 * np.pi ** 2)) * x) \
                      * torch.sin(2 * np.pi * y)
        for inod in range(u_nonods):
            x_inod = sf_nd_nb.pre_func_space.x_ref_in[inod // u_nloc, :, inod % u_nloc]
            x = x_inod[0]
            y = x_inod[1]
            lamb = Re / 2 - torch.sqrt(Re ** 2 / 4 + 4 * np.pi ** 2)
            p[inod // u_nloc, inod % u_nloc] = - torch.exp(2 * lamb * x)/2
    elif problem == 'ldc' or problem == 'bfs' or problem == 'fpc' and ndim == 2:
        print("====WARNING====:no analytical soln exists for %s problem..." % problem)
        u_nloc = sf_nd_nb.vel_func_space.element.nloc
        p_nloc = sf_nd_nb.pre_func_space.element.nloc
        nele = config.nele
        u_nonods = u_nloc * nele
        p_nonods = p_nloc * nele
        u_ana = torch.zeros(u_nonods * ndim + u_nonods, device=dev, dtype=torch.float64)
    elif problem == 'tgv' and ndim == 2:
        # taylor green vortex
        u_nloc = sf_nd_nb.vel_func_space.element.nloc
        p_nloc = sf_nd_nb.pre_func_space.element.nloc
        nele = config.nele
        u_nonods = u_nloc * nele
        p_nonods = p_nloc * nele
        u_ana = torch.zeros(u_nonods * ndim + u_nonods, device=dev, dtype=torch.float64)
        u, p = (u_ana[0:u_nonods*ndim].view(nele, u_nloc, ndim),
                u_ana[u_nonods*ndim:u_nonods*ndim+u_nonods].view(nele, u_nloc))
        if t is None:
            raise ValueError('should provide t when using taylor-green vortex analytical soln')
        Re = torch.tensor(1/config.mu_f, device=dev, dtype=torch.float64)
        for inod in range(u_nonods):
            x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // u_nloc, :, inod % u_nloc]
            x = x_inod[0]
            y = x_inod[1]

            u[inod // u_nloc, inod % u_nloc, 0] = \
                -torch.exp(-(2 * torch.pi ** 2 * t) / Re) * torch.cos(torch.pi * x) * torch.sin(torch.pi * y)
            u[inod // u_nloc, inod % u_nloc, 1] = \
                torch.exp(-(2 * torch.pi ** 2 * t) / Re) * torch.cos(torch.pi * y) * torch.sin(torch.pi * x)
        for inod in range(u_nonods):
            x_inod = sf_nd_nb.pre_func_space.x_ref_in[inod // u_nloc, :, inod % u_nloc]
            x = x_inod[0]
            y = x_inod[1]
            p[inod // u_nloc, inod % u_nloc] = \
                -torch.exp(-(4 * torch.pi ** 2 * t) / Re) * (torch.cos(2 * torch.pi * x) / 4 + torch.cos(
                    2 * torch.pi * y) / 4)
    elif problem == 'tgv10' and ndim == 2:
        # taylor green vortex
        u_nloc = sf_nd_nb.vel_func_space.element.nloc
        p_nloc = sf_nd_nb.pre_func_space.element.nloc
        nele = config.nele
        u_nonods = u_nloc * nele
        p_nonods = p_nloc * nele
        u_ana = torch.zeros(u_nonods * ndim + u_nonods, device=dev, dtype=torch.float64)
        u, p = (u_ana[0:u_nonods*ndim].view(nele, u_nloc, ndim),
                u_ana[u_nonods*ndim:u_nonods*ndim+u_nonods].view(nele, u_nloc))
        if t is None:
            raise ValueError('should provide t when using taylor-green vortex analytical soln')
        Re = torch.tensor(1/config.mu_f, device=dev, dtype=torch.float64)
        u_m = 10.
        for inod in range(u_nonods):
            x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // u_nloc, :, inod % u_nloc]
            x = x_inod[0]
            y = x_inod[1]

            u[inod // u_nloc, inod % u_nloc, 0] = \
                -torch.exp(-(2 * torch.pi ** 2 * t) / Re) * torch.cos(torch.pi * x) * torch.sin(torch.pi * y) * u_m
            u[inod // u_nloc, inod % u_nloc, 1] = \
                torch.exp(-(2 * torch.pi ** 2 * t) / Re) * torch.cos(torch.pi * y) * torch.sin(torch.pi * x) * u_m
        for inod in range(u_nonods):
            x_inod = sf_nd_nb.pre_func_space.x_ref_in[inod // u_nloc, :, inod % u_nloc]
            x = x_inod[0]
            y = x_inod[1]
            p[inod // u_nloc, inod % u_nloc] = \
                -torch.exp(-(4 * torch.pi ** 2 * t) / Re) * (torch.cos(2 * torch.pi * x) / 4 + torch.cos(
                    2 * torch.pi * y) / 4) * u_m**2
    else:
        raise Exception('problem analytical solution not defined for '+problem)
    return u_ana


def he_ana_sln(prob, t=None):
    """
    get analytical solution for hyper-elastic problems
    (this is used to set first few timesteps; or to compute error)
    """
    ndim = config.ndim
    if prob == 'hyper-elastic' and ndim == 3:
        # abbas 2018
        alpha = 0.1
        gamma = 0.1
        lam = config.lam_s
        mu = config.mu_s
        nloc = sf_nd_nb.disp_func_space.element.nloc
        u_ana = torch.zeros(config.nele * nloc * ndim, device=dev, dtype=torch.float64)
        for inod in range(config.nele_f * nloc, config.nele * nloc):
            x_inod = sf_nd_nb.vel_func_space.x_ref_in[inod // nloc, :, inod % nloc]
            theta = alpha * torch.sin(torch.pi * x_inod[1])  # alpha sin(pi Y)
            g = gamma * torch.sin(torch.pi * x_inod[0])  # gamma sin(pi X)
            h = 0  # 0

            u_ana[inod // nloc, inod % nloc, 0] = (1. / lam + alpha) * x_inod[0] + theta
            u_ana[inod // nloc, inod % nloc, 1] = -(1. / lam + (alpha + gamma + alpha * gamma) /
                                                (1 + alpha + gamma + alpha * gamma)) * x_inod[1]
            u_ana[inod // nloc, inod % nloc, 2] = (1. / lam + alpha) * x_inod[2] + g + h
    else:
        raise Exception('the problem ' + prob + ' is not defined in he_bc_f')
    return u_ana
