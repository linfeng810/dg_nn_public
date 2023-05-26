#!/usr/bin/env python3

"""
all integrations for hyper-elastic
matric-free implementation
"""


import torch
from torch import Tensor
import numpy as np
import config
from config import sf_nd_nb
if config.ndim == 2:
    from shape_function import get_det_nlx as get_det_nlx
    from shape_function import sdet_snlx as sdet_snlx
else:
    from shape_function import get_det_nlx_3d as get_det_nlx
    from shape_function import sdet_snlx_3d as sdet_snlx


dev = config.dev
nele = config.nele
mesh = config.mesh
nonods = config.nonods
ngi = config.ngi
ndim = config.ndim
nloc = config.nloc
nface = config.ndim+1
sngi = config.sngi
cijkl = config.cijkl
lam = config.lam
mu = config.mu
# note on indices:
#  b: batch_in,
#  n: nloc,
#  g: ngi, or sngi, gaussian points
#  i,j,k,l: dimension of tensors,
#  i,j: can also refer to iloc, jloc :-(


def _calc_F(nlx, u, batch_in: int):
    # compute deformation gradient
    F = torch.einsum('bni,bjng->bijg', u.view(batch_in, nloc, ndim), nlx)
    return F


def _calc_C(F):
    """
    compute right Cauchy-Green tensor
    C = F^T F
    """
    C = torch.einsum('bikg,bijg->bkjg', F, F)
    return C


def _calc_CC(C, F):
    """
    compute elasticity tensor $\mathbb C$
    $\mathbb C = \partial S / \partial C$
    """
    batch_in = C.shape[0]
    invC = torch.linalg.inv(C)  # C^{-1}
    invCinvC = torch.einsum('bijg,bklg->bijklg', invC, invC)  # C^{-1} \otimes C^{-1}
    J = torch.linalg.det(F)  # this is J = det F, or J^2 = det C
    mu_m_lam_lnJ = mu - lam * torch.log(J)
    CC = torch.zeros(batch_in, ndim, ndim, ndim, ndim, device=dev, dtype=torch.float64)
    CC += lam * invCinvC
    CC += torch.einsum('bg,bikjlg->bijklg', mu_m_lam_lnJ, invCinvC)
    CC += torch.einsum('bg,biljkg->bijklg', mu_m_lam_lnJ, invCinvC)
    return CC


def _calc_S(C, F):
    # compute PK2 tensor
    batch_in = C.shape[0]
    invC = torch.linalg.inv(C)  # C^{-1}
    lnJ = torch.log(torch.linalg.det(F))  # ln J = ln det F
    S = torch.zeros(batch_in, ndim, ndim, device=dev, dtype=torch.float64)
    S += mu * (torch.eye(ndim) - invC)
    S += lam * torch.einsum('bg,bijg->bijg', lnJ, invC)
    return S


def calc_P(nlx, u, batch_in: int):
    """
    compute PK1 tensor from given displacement
    """
    F = _calc_F(nlx, u, batch_in)
    C = _calc_C(F)
    S = _calc_S(C, F)
    P = torch.einsum('bijg,bjkg->bikg', F, S)
    return P


def calc_AA(nlx, u, batch_in: int):
    """
    compute elasticity tensor \mathbb A
    at given intermediate state (displacement)
    \mathbb A = \partial P / \partial F
              = delta S + F F C
    """
    F = _calc_F(nlx, u, batch_in)
    C = _calc_C(F)
    S = _calc_S(C, F)
    CC = _calc_CC(C, F)
    AA = torch.zeros(batch_in, ndim, ndim, ndim, ndim, device=dev, dtype=torch.float64)
    AA += torch.einsum('biIg,bkKg,bIJKLg->biJkLg', F, F, CC)
    AA += torch.einsum('ik,bJLg->biJkLg', torch.eye(ndim), S)
    return AA


def calc_RAR_mf_color():
    ...


def get_residual_and_smooth_once():
    ...


def get_residual_only():
    ...


def get_rhs(u, u_bc, f, u_n=0):
    """
    get right-hand side at the start of each newton step
    Note that in the input lists,
    u_n is field value at last *time* step (if transient).
    """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0,nnn+1)/nnn*nele, dtype=int)
    idx_in = torch.zeros(nele, dtype=torch.bool)
    idx_in_f = torch.zeros(nele * nface, dtype=torch.bool, device=dev)

    rhs = torch.zeros(nele, nloc, ndim, device=dev, dtype=torch.float64)
    u_bc = u_bc.view(nele, nloc, ndim)
    f = f.view(nele, nloc, ndim)

    alnmt = sf_nd_nb.alnmt

    for i in range(nnn):
        idx_in *= 0
        # volume integral
        idx_in[brk_pnt[i]:brk_pnt[i+1]] = True
        batch_in = torch.sum(idx_in)
        n = sf_nd_nb.n
        nx, detwei = get_det_nlx(
            nlx=sf_nd_nb.nlx,
            x_loc=sf_nd_nb.x_ref_in[idx_in],
            weight=sf_nd_nb.weight
        )
        K = torch.einsum('ig,jg,bg->bij', n, n, detwei)  # Ni Nj, this is batch_in x nloc x nloc, same for 3 direction
        rhs[idx_in, ...] += torch.einsum('bij,bjd->bid', K, f[idx_in, ...])
        # surface integral
        idx_in_f *= 0
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        F_b = torch.where(torch.logical_and(alnmt < 0, idx_in_f))[0]  # boundary face
        E_F_b = torch.floor_divide(F_b, nface)
        f_b = torch.remainder(F_b, nface)
        if ndim == 3:  # in 3D one element might have multiple boundary faces
            for iface in range(nface):
                idx_face = f_b == iface
                rhs = _s_rhs_fb(rhs, f_b[idx_face], E_F_b[idx_face],
                                u, u_bc)
        else:  # in 2D we requrie in the mesh, each element can have at most 1 boundary face
            raise Exception('2D hyper-elasticity is not implemented yet...')
            rhs = _s_rhs_fb(rhs, f_b, E_F_b,
                            u, u_bc)
    return rhs


def _s_rhs_fb(rhs, f_b, E_F_b, u, u_bc):
    """
    add surface integral contribution to equation right-hand side
    """
    batch_in = f_b.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    if batch_in < 1:  # nothing to do here.
        return rhs
    # get face shape function
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.snlx,
        x_loc=sf_nd_nb.x_ref_in[E_F_b],
        sweight=sf_nd_nb.sweight
    )
    sn = sf_nd_nb.sn[f_b, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_b, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_b, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_b, ...]  # (batch_in, ndim)
    mu_e = config.eta_e / torch.sum(sdetwei, -1)
    # get elasticity tensor at face quadrature points
    # (batch_in, ndim, ndim, ndim, ndim, sngi)
    AA = calc_AA(nlx=snx, u=u[E_F_b, ...], batch_in=batch_in)
    rhs = rhs.view(nele, nloc, ndim)
    rhs[E_F_b, ...] += torch.einsum(
        'bni,bng,bj,bijklg,blmg->bmk',  # could easily by wrong...
        u_bc[E_F_b, ...],  # (batch_in, nloc, ndim)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        snx,  # batch_in, ndim, nloc, sngi)
    )
    rhs[E_F_b, ...] += torch.einsum(
        'b,bng,bj,bijklg,bmg,bl,bmk->bni',  # again could easily be wrong...
        mu_e,  # (batch_in)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        u_bc[E_F_b, ...]  # (batch_in, nloc, ndim)
    )
    return rhs


def update_nonlinear_residual(u, rhs):
    """
    update non-linear residual at current non-linear step.
    L_N(u^n, v) = L(v) - a(u^n, v)
    Note that in the input lists, u is current non-linear step
    value (i.e. u^n in the above equation).
    """
    ...


def k_mf_one_batch():
    ...


def k_nl_one_batch():
    """
    get volume integral terms in non-linear residual
    """
    ...


def s_mf_one_batch():
    ...


def _s_fi():
    ...


def _s_fb():
    ...

