import torch
import numpy as np
import config
from config import sf_nd_nb


def get_pg_tau(u, w, v, vx, q, detwei, batch_in: int):
    """
    get petrov-galerkin tau term.

    tau = rho * min(tau_max, tau_1, tau_2}. for details, see document.

    input:
    u: field that undergoes advection-diffusion (this is velocity in NS momentum equation)
        shape (batch_in, u_nloc, ndim)
    w: advection velocity
        shape (batch_in, u_nloc, ndim)
    v, vx: velocity shape function and its derivative
        shape (u_nloc, ngi)
        shape (batch_in, ndim, u_nloc, ngi)
    q: one-order lower space shape function
        shape (p_nloc, ngi)  # usually we set pressure space one-order lower that vel space, so we simply use p_nloc
    detwei: determinant x weight, sum this can get element volume then get h
        shape (batch_in, ngi)
    batch_in: number of elements in this batch

    output:
    tau: tau_pg, to be used in the PG diffusion term
        shape (batch_in, ndim, ngi)
    """
    epsilon = 1e-8  # safeguard to avoid divide by zero
    gamma = 2 ** (sf_nd_nb.vel_func_space.element.ele_order - 2)
    alpha_1 = 0.125 * gamma
    alpha_2 = 2 * gamma
    h = torch.sum(detwei, dim=-1)
    if config.ndim == 2:
        h = torch.sqrt(h)
    else:
        h = torch.pow(h, 1./3.)

    # first get residual (difference of gradient in high and low order representation)
    gradu = torch.einsum(
        'bjng,bni->bijg',
        vx,
        u
    )  # du_i / dx_j
    gradu_low = torch.einsum(
        'mn,bjng,bni->bijg',
        sf_nd_nb.projection_one_order_lower,  # (p_nloc, u_nloc)
        vx,  # (batch_in, ndim, u_nloc, ngi)
        u,  # (batch_in, u_nloc, ngi)
    )  # low-order representation of grad u
    w_on_quad = torch.einsum(
        'bmj,mg->bjg',
        w,  # (batch_in, u_nloc, ndim)
        v,  # (u_nloc, ngi)
    )
    r = torch.einsum(
        'bjg,bijg->big',
        w_on_quad,  # (batch_in, ndim, ngi)
        gradu - gradu_low,  # (batch_in, ndim, ndim, ngi)
    )  # residual of each component on each quadrature points

    # tau_1
    tau_1 = alpha_1 * torch.einsum(
        'big,b,big->big',  # (batch_in, ndim, ngi)
        torch.abs(r),  # residual (batch_in, ndim, ngi)
        h,  # ele size (batch_in)
        torch.reciprocal(epsilon + 1./config.ndim * torch.sum(torch.abs(gradu), dim=-2)),
            # (batch_in, ndim, ngi)
    )

    # u*
    u_star = torch.einsum(
        'bkg,bikg,big,bijg->bijg',
        w_on_quad,  # (batch_in, ndim, ngi)
        gradu,  # (batch_in, ndim, ndim, ngi)
        torch.reciprocal(epsilon + torch.sum(torch.square(gradu), dim=-2)),  # (batch_in, ndim, ngi)
        gradu,  # (batch_in, ndim, ndim, ngi)
    )  # takes different value for each u component (i), u_star itself has ndim components (j)

    # tau_2
    tau_2 = alpha_2 * torch.einsum(
        'big,b,big->big',
        torch.square(r),  # (batch_in, ndim, ngi)
        h,  # (batch_in)
        torch.reciprocal(epsilon + 1. / config.ndim * torch.sum(torch.abs(u_star), dim=-2) *
                         torch.sum(torch.square(gradu), dim=-2))  # (batch_in, ndim, ngi)
    )

    # tau_max
    # assuming always transient
    tau_max = h ** 2 / sf_nd_nb.dt
    tau_max = tau_max.unsqueeze(-1).unsqueeze(-1).expand(tau_1.shape)

    # eventually, tau
    tau = torch.minimum(tau_1, torch.minimum(tau_2, tau_max)) * config.rho

    return tau


def get_projection_one_order_lower(k, ndim):
    """
    return the projection matrix from order k to order k-1
    """
    p = None
    if ndim == 3:
        if k == 3:
            p = torch.tensor([
                [11 / 56, 9 / 112, 9 / 112, 9 / 112, 45 / 56, -45 / 112, 45 / 56, -45 / 112, 9 / 112, 9 / 112, 45 / 56,
                 -45 / 112, 9 / 112, 9 / 112, 9 / 112, 9 / 112, 9 / 112, -45 / 112, -45 / 112, -45 / 112],
                [9 / 112, 11 / 56, 9 / 112, 9 / 112, 9 / 112, 9 / 112, -45 / 112, 45 / 56, 45 / 56, -45 / 112, 9 / 112,
                 9 / 112, 45 / 56, -45 / 112, 9 / 112, 9 / 112, -45 / 112, -45 / 112, 9 / 112, -45 / 112],
                [9 / 112, 9 / 112, 11 / 56, 9 / 112, -45 / 112, 45 / 56, 9 / 112, 9 / 112, -45 / 112, 45 / 56, 9 / 112,
                 9 / 112, 9 / 112, 9 / 112, 45 / 56, -45 / 112, -45 / 112, -45 / 112, -45 / 112, 9 / 112],
                [9 / 112, 9 / 112, 9 / 112, 11 / 56, 9 / 112, 9 / 112, 9 / 112, 9 / 112, 9 / 112, 9 / 112, -45 / 112,
                 45 / 56, -45 / 112, 45 / 56, -45 / 112, 45 / 56, -45 / 112, 9 / 112, -45 / 112, -45 / 112],
                [9 / 112, -1 / 448, -1 / 448, 9 / 112, -9 / 56, 9 / 448, -9 / 56, 9 / 448, 153 / 448, 153 / 448,
                 9 / 112, 9 / 112, 9 / 448, -9 / 56, 9 / 448, -9 / 56, 99 / 224, 99 / 224, -9 / 56, -9 / 56],
                [-1 / 448, 9 / 112, -1 / 448, 9 / 112, 153 / 448, 153 / 448, 9 / 448, -9 / 56, -9 / 56, 9 / 448,
                 9 / 448, -9 / 56, 9 / 112, 9 / 112, 9 / 448, -9 / 56, -9 / 56, 99 / 224, 99 / 224, -9 / 56],
                [-1 / 448, -1 / 448, 9 / 112, 9 / 112, 9 / 448, -9 / 56, 153 / 448, 153 / 448, 9 / 448, -9 / 56,
                 9 / 448, -9 / 56, 9 / 448, -9 / 56, 9 / 112, 9 / 112, -9 / 56, 99 / 224, -9 / 56, 99 / 224],
                [-1 / 448, 9 / 112, 9 / 112, -1 / 448, 9 / 448, -9 / 56, 9 / 448, -9 / 56, 9 / 112, 9 / 112,
                 153 / 448, 153 / 448, -9 / 56, 9 / 448, -9 / 56, 9 / 448, -9 / 56, -9 / 56, 99 / 224, 99 / 224],
                [9 / 112, -1 / 448, 9 / 112, -1 / 448, 9 / 112, 9 / 112, -9 / 56, 9 / 448, 9 / 448, -9 / 56, -9 / 56,
                 9 / 448, 153 / 448, 153 / 448, -9 / 56, 9 / 448, 99 / 224, -9 / 56, -9 / 56, 99 / 224],
                [9 / 112, 9 / 112, -1 / 448, -1 / 448, -9 / 56, 9 / 448, 9 / 112, 9 / 112, -9 / 56, 9 / 448, -9 / 56,
                 9 / 448, -9 / 56, 9 / 448, 153 / 448, 153 / 448, 99 / 224, -9 / 56, 99 / 224, -9 / 56],
            ], device=config.dev, dtype=torch.float64)
        elif k == 2:
            p = torch.tensor([
                [1 / 5, -2 / 15, -2 / 15, -2 / 15, -2 / 15, 8 / 15, 8 / 15, 8 / 15, -2 / 15, -2 / 15],
                [-2 / 15, 1 / 5, -2 / 15, -2 / 15, 8 / 15, -2 / 15, 8 / 15, -2 / 15, 8 / 15, -2 / 15],
                [-2 / 15, -2 / 15, 1 / 5, -2 / 15, 8 / 15, 8 / 15, -2 / 15, -2 / 15, -2 / 15, 8 / 15],
                [-2 / 15, -2 / 15, -2 / 15, 1 / 5, -2 / 15, -2 / 15, -2 / 15, 8 / 15, 8 / 15, 8 / 15],
            ], device=config.dev, dtype=torch.float64)
        else:
            raise ValueError('cannot find projection from %d- to %d- order element' % (k, k-1))
    elif ndim == 2:
        if k == 3:
            p = torch.tensor([
                [17 / 35, 9 / 70, 9 / 70, 27 / 35, -18 / 35, 9 / 70, 9 / 70, -18 / 35, 27 / 35, -18 / 35],
                [9 / 70, 17 / 35, 9 / 70, -18 / 35, 27 / 35, 27 / 35, -18 / 35, 9 / 70, 9 / 70, -18 / 35],
                [9 / 70, 9 / 70, 17 / 35, 9 / 70, 9 / 70, -18 / 35, 27 / 35, 27 / 35, -18 / 35, -18 / 35],
                [-1 / 70, -1 / 70, 9 / 70, 9 / 20, 9 / 20, -9 / 280, -27 / 140, -27 / 140, -9 / 280, 9 / 20],
                [9 / 70, -1 / 70, -1 / 70, -27 / 140, -9 / 280, 9 / 20, 9 / 20, -9 / 280, -27 / 140, 9 / 20],
                [-1 / 70, 9 / 70, -1 / 70, -9 / 280, -27 / 140, -27 / 140, -9 / 280, 9 / 20, 9 / 20, 9 / 20],
            ], device=config.dev, dtype=torch.float64)
        elif k == 2:
            p = torch.tensor([
                [2 / 5, -1 / 5, -1 / 5, 3 / 5, -1 / 5, 3 / 5],
                [-1 / 5, 2 / 5, -1 / 5, 3 / 5, 3 / 5, -1 / 5],
                [-1 / 5, -1 / 5, 2 / 5, -1 / 5, 3 / 5, 3 / 5],
            ], device=config.dev, dtype=torch.float64)
        else:
            raise ValueError('cannot find projection from %d- to %d- order element' % (k, k-1))
    return p
