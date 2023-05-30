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
# we sould be able to reuse all multigrid functions in linear-elasticity in hyper-elasticity
import multigrid_linearelastic as mg_le
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
# ngi = config.ngi
ndim = config.ndim
nloc = config.nloc
nface = config.ndim+1
# sngi = config.sngi
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
    # here we assum nx shape is (batch_in, ndim, nloc, ngi)
    F = torch.einsum('bni,bjng->bgij', u.view(batch_in, nloc, ndim), nlx)
    F += torch.eye(ndim, device=dev, dtype=torch.float64)
    # output shape is (batch_in, ngi, ndim, ndim)
    return F


def _calc_C(F):
    """
    compute right Cauchy-Green tensor
    C = F^T F
    """
    C = torch.einsum('bgik,bgij->bgkj', F, F)
    # output shape is (batch_in, ngi, ndim, ndim)
    return C


def _calc_CC(C, F):
    """
    compute elasticity tensor $\mathbb C$
    $\mathbb C = \partial S / \partial C$
    """
    batch_in = C.shape[0]
    ngi = C.shape[1]
    invC = torch.linalg.inv(C)  # C^{-1}
    invCinvC = torch.einsum('bgij,bgkl->bgijkl', invC, invC)  # C^{-1} \otimes C^{-1}
    J = torch.linalg.det(F)  # this is J = det F, or J^2 = det C
    if torch.any(J <= 0):
        raise Exception(f'determinant of some element is negative! ',
                        f'consider larger relaxation... or there is a bug :/')
    mu_m_lam_lnJ = mu - lam * torch.log(J)
    CC = torch.zeros(batch_in, ngi, ndim, ndim, ndim, ndim, device=dev, dtype=torch.float64)
    CC += lam * invCinvC
    CC += torch.einsum('bg,bgikjl->bgijkl', mu_m_lam_lnJ, invCinvC)
    CC += torch.einsum('bg,bgiljk->bgijkl', mu_m_lam_lnJ, invCinvC)
    # output shape is (batch_in, ngi, ndim, ndim, ndim, ndim)
    return CC


def _calc_S(C, F):
    # compute PK2 tensor
    batch_in = C.shape[0]
    ngi = C.shape[1]
    invC = torch.linalg.inv(C)  # C^{-1}
    lnJ = torch.log(torch.linalg.det(F))  # ln J = ln det F
    S = torch.zeros(batch_in, ngi, ndim, ndim, device=dev, dtype=torch.float64)
    S += mu * (torch.eye(ndim, device=dev, dtype=torch.float64) - invC)
    S += lam * torch.einsum('bg,bgij->bgij', lnJ, invC)
    # output shape is (batch_in, ngi, ndim, ndim)
    return S


def calc_P(nlx, u, batch_in: int):
    """
    compute PK1 tensor from given displacement
    """
    F = _calc_F(nlx, u, batch_in)
    C = _calc_C(F)
    S = _calc_S(C, F)
    P = torch.einsum('bgij,bgjk->bikg', F, S)
    # output shape is (batch_in, ndim, ndim, ngi)
    return P


def calc_AA(nlx, u, batch_in: int):
    """
    compute elasticity tensor \mathbb A
    at given intermediate state (displacement)
    \mathbb A = \partial P / \partial F
              = delta S + F F C
    """
    ngi = nlx.shape[-1]
    F = _calc_F(nlx, u, batch_in)
    C = _calc_C(F)
    S = _calc_S(C, F)
    CC = _calc_CC(C, F)
    AA = torch.zeros(batch_in, ndim, ndim, ndim, ndim, ngi, device=dev, dtype=torch.float64)
    AA += torch.einsum('bgiI,bgkK,bgIJKL->biJkLg', F, F, CC)
    AA += torch.einsum('ik,bgJL->biJkLg', torch.eye(ndim, device=dev, dtype=torch.float64), S)
    # output shape is (batch_in, ndim, ndim, ndim, ndim, ngi)
    return AA


def calc_RAR_mf_color(
        I_fc, I_cf,
        whichc, ncolor,
        fina, cola, ncola,
        u  # this is the displacement field at current non-linear step,
        # it's to be used to get lhs.
):
    """
    get operator on P1CG grid, i.e. RAR
    where R is prolongator/restrictor,
    via coloring method.
    NOTE that for non-linear eq.,
    lhs operator is determined by u at current non-linear step,
    lhs vector is the color vector.
    Do not mix use them.
    """
    import time
    start_time = time.time()
    cg_nonods = sf_nd_nb.cg_nonods
    p1dg_nonods = config.p1dg_nonods
    value = torch.zeros(ncola, ndim, ndim, device=dev, dtype=torch.float64)  # NNZ entry values
    dummy = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)  # dummy variable of same length as PnDG
    Rm = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)
    ARm = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)
    RARm = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
    mask = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)  # color vec
    for color in range(1, ncolor + 1):
        print('color: ', color)
        for jdim in range(ndim):
            mask *= 0
            mask[:, jdim] += torch.tensor((whichc == color),
                                          device=dev,
                                          dtype=torch.float64)  # 1 if true; 0 if false
            Rm *= 0
            for idim in range(ndim):
                # Rm[:, idim] += torch.mv(I_fc, mask[:, idim].view(-1))  # (p1dg_nonods, ndim)
                Rm[:, idim] += mg_le.p1dg_to_p3dg_prolongator(
                    torch.mv(I_fc, mask[:, idim].view(-1))
                )  # (p3dg_nonods, ndim)
            ARm *= 0
            ARm = get_residual_only(ARm, u,
                                    Rm, dummy)
            ARm *= -1.  # (p3dg_nonods, ndim)
            # RARm = multi_grid.p3dg_to_p1dg_restrictor(ARm)  # (p1dg_nonods, )
            RARm *= 0
            for idim in range(ndim):
                RARm[:, idim] += torch.mv(I_cf, mg_le.p3dg_to_p1dg_restrictor(ARm[:, idim]))  # (cg_nonods, ndim)
            for idim in range(ndim):
                # add to value
                for i in range(RARm.shape[0]):
                    for count in range(fina[i], fina[i + 1]):
                        j = cola[count]
                        value[count, idim, jdim] += RARm[i, idim] * mask[j, jdim]
        print('finishing (another) one color, time comsumed: ', time.time() - start_time)
    return value


def get_residual_and_smooth_once(
        r0, u_i, du_i, u_rhs
):
    """
    update residual, then do one (block-) Jacobi smooth.
    will output r0 and updated u_i
    in case one want to get how large the linear residual is
    """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele, dtype=int)
    # add precalculated rhs to residual
    r0 += u_rhs
    for i in range(nnn):
        # volume integral
        idx_in = torch.zeros(nele, device=dev, dtype=bool)
        idx_in[brk_pnt[i]:brk_pnt[i + 1]] = True
        batch_in = int(torch.sum(idx_in))
        # dummy diagA and bdiagA
        diagA = torch.zeros(batch_in, nloc, ndim, device=dev, dtype=torch.float64)
        bdiagA = torch.zeros(batch_in, nloc, ndim, nloc, ndim, device=dev, dtype=torch.float64)
        r0, diagA, bdiagA = _k_res_one_batch(r0, u_i, du_i,
                                             diagA, bdiagA,
                                             idx_in)
        # surface integral
        idx_in_f = torch.zeros(nele * nface, dtype=bool, device=dev)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0, diagA, bdiagA = _s_res_one_batch(r0, u_i, du_i,
                                             diagA, bdiagA,
                                             idx_in_f, brk_pnt[i])
        # smooth once
        if config.blk_solver == 'direct':
            bdiagA = torch.inverse(bdiagA.view(batch_in, nloc * ndim, nloc * ndim))
            du_i = du_i.view(nele, nloc * ndim)
            du_i[idx_in, :] += config.jac_wei * torch.einsum('...ij,...j->...i',
                                                             bdiagA,
                                                             r0.view(nele, nloc * ndim)[idx_in, :])
        if config.blk_solver == 'jacobi':
            new_b = torch.einsum('...ij,...j->...i',
                                 bdiagA.view(batch_in, nloc * ndim, nloc * ndim),
                                 du_i.view(nele, nloc * ndim)[idx_in, :]) \
                    + config.jac_wei * r0.view(nele, nloc * ndim)[idx_in, :]
            new_b = new_b.view(-1)
            diagA = diagA.view(-1)
            du_i = du_i.view(nele, nloc * ndim)
            du_i_partial = du_i[idx_in, :]
            for its in range(3):
                du_i_partial += ((new_b - torch.einsum('...ij,...j->...i',
                                                       bdiagA.view(batch_in, nloc * ndim, nloc * ndim),
                                                       du_i_partial).view(-1))
                                / diagA).view(-1, nloc * ndim)
            du_i[idx_in, :] = du_i_partial.view(-1, nloc * ndim)
    r0 = r0.view(nele * nloc, ndim)
    du_i = du_i.view(nele * nloc, ndim)
    return r0, du_i


def get_residual_only(
        r0, u_i, du_i, u_rhs
):
    """
    update residual
    r0 = u_rhs - (K+S)*u_i
    """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele, dtype=int)
    r0 = r0.view(nonods, ndim)
    u_rhs = u_rhs.view(nonods, ndim)
    # add pre-computed right hand side to residual
    r0 += u_rhs
    for i in range(nnn):
        # volume integral
        idx_in = torch.zeros(nele, dtype=torch.bool)
        idx_in[brk_pnt[i]:brk_pnt[i + 1]] = True
        batch_in = torch.sum(idx_in)
        # dummy diagA and bdiagA
        diagA = torch.zeros(batch_in, nloc, ndim, device=dev, dtype=torch.float64)
        bdiagA = torch.zeros(batch_in, nloc, ndim, nloc, ndim, device=dev, dtype=torch.float64)
        r0, diagA, bdiagA = _k_res_one_batch(r0, u_i, du_i,
                                             diagA, bdiagA,
                                             idx_in)
        # surface integral
        idx_in_f = torch.zeros(nele * nface, dtype=torch.bool, device=dev)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0, diagA, bdiagA = _s_res_one_batch(r0, u_i, du_i,
                                             diagA, bdiagA,
                                             idx_in_f, brk_pnt[i])
    r0 = r0.view(nele * nloc, ndim)
    return r0


def _k_res_one_batch(
        r0, u_i, du_i,
        diagA, bdiagA,
        idx_in
):
    batch_in = diagA.shape[0]
    # change view
    r0 = r0.view(-1, nloc, ndim)
    u_i = u_i.view(-1, nloc, ndim)
    du_i = du_i.view(-1, nloc, ndim)
    diagA = diagA.view(-1, nloc, ndim)
    bdiagA = bdiagA.view(-1, nloc, ndim, nloc, ndim)
    # get shape function and derivatives
    n = sf_nd_nb.n
    nx, detwei = get_det_nlx(
        nlx=sf_nd_nb.nlx,
        x_loc=sf_nd_nb.x_ref_in[idx_in],
        weight=sf_nd_nb.weight
    )
    AA = calc_AA(nlx=nx, u=u_i[idx_in, ...], batch_in=batch_in)

    K = torch.zeros(nele, nloc, ndim, nloc, ndim, device=dev, dtype=torch.float64)
    # (\nabla v)_ij A (\nabla \delta u)_kl
    K += torch.einsum('bjmg,bijklg,blng,bg->bmink', nx, AA, nx, detwei)
    if config.isTransient:
        # ni nj
        for idim in range(ndim):
            K[:, :, idim, :, idim] += torch.einsum('mg,ng,bg->bmn', n, n, detwei) / config.dt
    # update residual
    r0[idx_in, ...] -= torch.einsum('bminj,bnj->bmi', K, du_i[idx_in, ...])
    # get diagonal
    diagA += torch.diagonal(K.view(batch_in, nloc * ndim, nloc * ndim), dim1=1, dim2=2).view(batch_in, nloc, ndim)
    bdiagA += K
    return r0, diagA, bdiagA


def _s_res_one_batch(
        r, u_i, du_i,
        diagA, bdiagA,
        idx_in_f: Tensor,
        batch_start_idx
):
    # get essential data
    nbf = sf_nd_nb.nbf
    alnmt = sf_nd_nb.alnmt

    u_i = u_i.view(nele, nloc, ndim)
    du_i = du_i.view(nele, nloc, ndim)
    r = r.view(nele, nloc, ndim)

    # first lets separate nbf to get two list of F_i and F_b
    F_i = torch.where(torch.logical_and(alnmt >= 0, idx_in_f))[0]  # interior face
    F_b = torch.where(torch.logical_and(alnmt < 0, idx_in_f))[0]  # boundary face
    F_inb = nbf[F_i]  # neighbour list of interior face
    F_inb = F_inb.type(torch.int64)

    # create two lists of which element f_i / f_b is in
    E_F_i = torch.floor_divide(F_i, nface)
    E_F_b = torch.floor_divide(F_b, nface)
    E_F_inb = torch.floor_divide(F_inb, nface)

    # local face number
    f_i = torch.remainder(F_i, nface)
    f_b = torch.remainder(F_b, nface)
    f_inb = torch.remainder(F_inb, nface)

    # for interior faces, update residual
    # r <= r - S*u_i
    # let's hope that each element has only one boundary face.
    for iface in range(nface):
        for nb_gi_aln in range(nface - 1):
            idx_iface = (f_i == iface) & (sf_nd_nb.alnmt[F_i] == nb_gi_aln)
            if idx_iface.sum() < 1:
                # there is nothing to do here, go on
                continue
            r, diagA, bdiagA = _s_res_fi(
                r, f_i[idx_iface], E_F_i[idx_iface],
                f_inb[idx_iface], E_F_inb[idx_iface],
                u_i, du_i,
                diagA, bdiagA, batch_start_idx,
                nb_gi_aln)

    # update residual for boundary faces
    # r <= r + S*u_bc - S*u_i
    if ndim == 3:
        for iface in range(nface):
            idx_iface = f_b == iface
            r, diagA, bdiagA = _s_res_fb(
                r, f_b[idx_iface], E_F_b[idx_iface],
                u_i, du_i,
                diagA, bdiagA, batch_start_idx)
    else:
        raise Exception('2D hyper-elasticity not implemented!')
        r, diagA, bdiagA = _S_fb(
            r, f_b, E_F_b,
            u_i, du_i,
            diagA, bdiagA, batch_start_idx)
    return r, diagA, bdiagA


def _s_res_fi(
        r, f_i, E_F_i,
        f_inb, E_F_inb,
        u_i, du_i,
        diagA, bdiagA, batch_start_idx,
        nb_gi_aln
):
    batch_in = f_i.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    # shape function on this side
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.snlx,
        x_loc=sf_nd_nb.x_ref_in[E_F_i],
        sweight=sf_nd_nb.sweight
    )
    sn = sf_nd_nb.sn[f_i, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_i, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_i, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_i, ...]  # (batch_in, ndim)

    # shape function on the other side
    snx_nb, _, snormal_nb = sdet_snlx(
        snlx=sf_nd_nb.snlx,
        x_loc=sf_nd_nb.x_ref_in[E_F_inb],
        sweight=sf_nd_nb.sweight
    )
    # change gaussian points order on other side
    nb_aln = sf_nd_nb.gi_align[nb_gi_aln, :]
    snx_nb = snx_nb[..., nb_aln]
    # get faces we want
    sn_nb = sf_nd_nb.sn[f_inb, ...]  # (batch_in, nloc, sngi)
    snx_nb = snx_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim, nloc, sngi)
    snormal_nb = snormal_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim)

    mu_e = config.eta_e / torch.sum(sdetwei, -1)
    u_ith = u_i[E_F_i, ...]  # u^n on this side (th)
    u_inb = u_i[E_F_inb, ...]  # u^n on the other side (neighbour)

    S = torch.zeros(batch_in, nloc, ndim, nloc, ndim, device=dev, dtype=torch.float64)
    AA_th = calc_AA(nlx=snx, u=u_ith, batch_in=batch_in)
    AA_nb = calc_AA(nlx=snx_nb, u=u_inb, batch_in=batch_in)
    # this side
    # [vi nj] {A \nabla u_kl}
    S += torch.einsum(
        'bmg,bj,bijklg,blng,bg->bmink',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        AA_th,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (-0.5)
    # [ui nj] {A \nabla v_kl}
    S += torch.einsum(
        'bng,bj,bijklg,blmg,bg->bmkni',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        AA_th,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (-0.5)
    # penalty term
    # \mu_e [v_i n_j]{A}
    S += torch.einsum(
        'b,bmg,bj,bijklg,bng,bl,bg->bmink',
        mu_e,  # (batch_in)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        0.5 * (AA_th + AA_nb),  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        sdetwei,  # (batch_in, sngi)
    )
    # update residual
    r[E_F_i, ...] -= torch.einsum('bminj,bnj->bmi', S, du_i[E_F_i, ...])
    # put diagonal of S into diagS
    diagA[E_F_i-batch_start_idx, :, :] += torch.diagonal(S.view(batch_in, nloc*ndim, nloc*ndim),
                                                         dim1=1, dim2=2).view(batch_in, nloc, ndim)
    bdiagA[E_F_i-batch_start_idx, ...] += S

    # other side
    S *= 0
    # [vi nj] {A \nabla u_kl}
    S += torch.einsum(
        'bmg,bj,bijklg,blng,bg->bmink',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        AA_nb,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        snx_nb,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (-0.5)
    # [ui nj] {A \nabla v_kl}
    S += torch.einsum(
        'bng,bj,bijklg,blmg,bg->bmkni',
        sn_nb,  # (batch_in, nloc, sngi)
        snormal_nb,  # (batch_in, ndim)
        AA_th,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (-0.5)
    # penalty term
    # \mu_e [v_i n_j]{A}
    S += torch.einsum(
        'b,bmg,bj,bijklg,bng,bl,bg->bmink',
        mu_e,  # (batch_in)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        0.5 * (AA_th + AA_nb),  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        sn_nb,  # (batch_in, nloc, sngi)
        snormal_nb,  # (batch_in, ndim)
        sdetwei,  # (batch_in, sngi)
    )
    # update residual
    r[E_F_i, ...] -= torch.einsum('bminj,bnj->bmi', S, du_i[E_F_inb, ...])

    return r, diagA, bdiagA


def _s_res_fb(
        r, f_b, E_F_b,
        u_i, du_i,
        diagA, bdiagA, batch_start_idx
):
    batch_in = f_b.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    if batch_in < 1:  # nothing to do here.
        return r
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
    AA = calc_AA(nlx=snx, u=u_i[E_F_b, ...], batch_in=batch_in)

    # boundary terms from last 3 terms in eq 60b
    # only one side
    S = torch.zeros(batch_in, nloc, ndim, nloc, ndim,
                    device=dev, dtype=torch.float64)
    # [v_i n_j] {A \nabla u_kl}
    S -= torch.einsum(
        'bmg,bj,bijklg,blng,bg->bmink',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    )
    # [u_i n_j] {A \nabla v_kl}
    S -= torch.einsum(
        'bng,bj,bijklg,blmg,bg->bmkni',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    )
    # \gamma_e [v_i n_j] {A} [u_k n_l]
    S += torch.einsum(
        'b,bmg,bj,bijklg,bng,bl,bg->bmink',
        mu_e,  # (batch_in)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        sdetwei,  # (batch_in, sngi)
    )
    # update residual
    r[E_F_b, ...] -= torch.einsum('bminj,bnj->bmi', S, du_i[E_F_b, ...])
    # get diagonal
    diagA[E_F_b - batch_start_idx, :, :] += torch.diagonal(S.view(batch_in, nloc * ndim, nloc * ndim),
                                                           dim1=-2, dim2=-1).view(batch_in, nloc, ndim)
    bdiagA[E_F_b - batch_start_idx, ...] += S
    return r, diagA, bdiagA


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

    for i in range(nnn):
        idx_in *= False
        # volume integral
        idx_in[brk_pnt[i]:brk_pnt[i+1]] = True
        rhs = _k_rhs_one_batch(rhs, u, u_n, f, idx_in)
        # surface integral
        idx_in_f *= False
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        rhs = _s_rhs_one_batch(rhs, u, u_bc, idx_in_f)
    rhs = rhs.view(-1, ndim)
    return rhs


def _k_rhs_one_batch(rhs, u, u_n, f, idx_in):
    batch_in = int(torch.sum(idx_in))
    n = sf_nd_nb.n
    nx, detwei = get_det_nlx(
        nlx=sf_nd_nb.nlx,
        x_loc=sf_nd_nb.x_ref_in[idx_in],
        weight=sf_nd_nb.weight
    )
    K = torch.zeros(nele, nloc, nloc, device=dev, dtype=torch.float64)
    K += torch.einsum('ig,jg,bg->bij', n, n, detwei)  # Ni Nj, this is batch_in x nloc x nloc, same for 3 direction
    # f v
    rhs[idx_in, ...] += torch.einsum('bij,bjd->bid', K, f[idx_in, ...])
    if config.isTransient:
        # u^t v / dt
        rhs[idx_in, ...] += torch.einsum('bij,bjd->bid', K, u_n.view(nele, nloc, ndim)[idx_in, ...]) / config.dt
    K *= 0
    # Nxi Nj P
    P = calc_P(nx, u.view(nele, nloc, ndim)[idx_in, ...], batch_in)  # PK1 stress evaluated at current state u
    rhs[idx_in, ...] -= torch.einsum(
        'bijg,bjmg,bg->bmi',  # i,j is idim and jdim; m, n is mloc and nloc
        P,  # (batch_in, ndim, ndim, ngi)
        # n,  # (nloc, ngi)
        nx,  # (batch_in, ndim, nloc, ngi)
        detwei,  # (batch_in, ngi)
    )
    return rhs


def _s_rhs_one_batch(rhs, u, u_bc, idx_in_f):
    nbf = sf_nd_nb.nbf
    alnmt = sf_nd_nb.alnmt

    # first lets separate nbf to get two list of F_i and F_b
    F_i = torch.where(torch.logical_and(alnmt >= 0, idx_in_f))[0]  # interior face
    F_b = torch.where(torch.logical_and(alnmt < 0, idx_in_f))[0]  # boundary face
    F_inb = nbf[F_i]  # neighbour list of interior face
    F_inb = F_inb.type(torch.int64)

    # create two lists of which element f_i / f_b is in
    E_F_i = torch.floor_divide(F_i, nface)
    E_F_b = torch.floor_divide(F_b, nface)
    E_F_inb = torch.floor_divide(F_inb, nface)

    # local face number
    f_i = torch.remainder(F_i, nface)
    f_b = torch.remainder(F_b, nface)
    f_inb = torch.remainder(F_inb, nface)

    # interior face term
    for iface in range(nface):
        for nb_gi_aln in range(nface-1):
            idx_iface = (f_i == iface) & (sf_nd_nb.alnmt[F_i] == nb_gi_aln)
            if idx_iface.sum() < 1:
                # there is nothing to do here, go on
                continue
            rhs = _s_rhs_fi(
                rhs, f_i[idx_iface], E_F_i[idx_iface],
                f_inb[idx_iface], E_F_inb[idx_iface],
                u,
                nb_gi_aln)

    # boundary term
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


def _s_rhs_fi(rhs,
              f_i, E_F_i,
              f_inb, E_F_inb,
              u,
              nb_gi_aln):
    batch_in = f_i.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    # shape function on this side
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.snlx,
        x_loc=sf_nd_nb.x_ref_in[E_F_i],
        sweight=sf_nd_nb.sweight
    )
    sn = sf_nd_nb.sn[f_i, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_i, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_i, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_i, ...]  # (batch_in, ndim)

    # shape function on the other side
    snx_nb, _, snormal_nb = sdet_snlx(
        snlx=sf_nd_nb.snlx,
        x_loc=sf_nd_nb.x_ref_in[E_F_inb],
        sweight=sf_nd_nb.sweight
    )
    # change gaussian points order
    nb_aln = sf_nd_nb.gi_align[nb_gi_aln, :]
    snx_nb = snx_nb[..., nb_aln]
    sn_nb = sn[..., nb_aln]
    # fetch faces we want
    sn_nb = sf_nd_nb.sn[f_inb, ...]  # (batch_in, nloc, sngi)
    snx_nb = snx_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim, nloc, sngi)
    snormal_nb = snormal_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim)

    mu_e = config.eta_e / torch.sum(sdetwei, -1)
    u_i = u[E_F_i, ...]  # u^n on this side
    u_inb = u[E_F_inb, ...]  # u^n on the other side

    # [vi nj]{P^n_kl} term
    P = calc_P(nlx=snx, u=u_i, batch_in=batch_in)
    P_nb = calc_P(nlx=snx_nb, u=u_inb, batch_in=batch_in)
    P *= 0.5
    P_nb *= 0.5
    P += P_nb  # this is {P^n} = 1/2 (P^1 + P^2)  average on both sides
    # this side + other side
    rhs[E_F_i, ...] += torch.einsum(
        'bmg,bj,bijg,bg->bmi',  # i,j is idim/jdim; m, n is mloc/nloc
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        # sn,  # (batch_in, nloc, sngi)
        P,  # (batch_in, ndim, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
    )
    del P, P_nb

    # [ui nj] {A (\nabla v)_kl] term
    AA = calc_AA(nlx=snx, u=u_i, batch_in=batch_in)
    # this side + other side
    rhs[E_F_i, ...] -= torch.einsum(
        'bnijg,bijklg,blmg,bg->bmk',  # i/j : idim/jdim; m/n: mloc/nloc
        (
            torch.einsum('bni,bng,bj->bnijg', u_i, sn, snormal)
            + torch.einsum('bni,bng,bj->bnijg', u_inb, sn_nb, snormal_nb)
        ),  # (batch_in, nloc, ndim, ndim, sngi)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei  # (batch_in, sngi)
    ) * 0.5

    # \gamma_e [vi nj] A [uk nl]
    # this A is 1/2(A_this + A_nb)
    AA += calc_AA(nlx=snx_nb, u=u_inb, batch_in=batch_in)
    AA *= 0.5
    rhs[E_F_i, ...] -= torch.einsum(
        'b,bmg,bj,bijklg,bnklg,bg->bmi',
        mu_e,  # (batch_in)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        (
            torch.einsum('bnk,bng,bl->bnklg', u_i, sn, snormal)
            + torch.einsum('bnk,bng,bl->bnklg', u_inb, sn_nb, snormal_nb)
        ),  # (batch_in, nloc, ndim, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
    )
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
    # u_Di nj A \delta v_kl
    rhs[E_F_b, ...] -= torch.einsum(
        'bni,bng,bj,bijklg,blmg,bg->bmk',  # could easily by wrong...
        u_bc[E_F_b, ...],  # (batch_in, nloc, ndim)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch, sngi)
    )
    # gamma_e v_i n_j A u_Dk n_l
    rhs[E_F_b, ...] += torch.einsum(
        'b,bmg,bj,bijklg,bng,bl,bnk,bg->bmi',  # again could easily be wrong...
        mu_e,  # (batch_in)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        u_bc[E_F_b, ...],  # (batch_in, nloc, ndim)
        sdetwei,  # (batch_in, sngi
    )
    # add boundary contribution from lhs. (last 3 terms in eq 60c)
    # u_i n_j A \nabla v_kl
    rhs[E_F_b, ...] += torch.einsum(
        'bni,bng,bj,bijklg,blmg,bg->bmk',  # could easily by wrong...
        u[E_F_b, ...],  # (batch_in, nloc, ndim)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch, sngi)
    )
    # \gamma_e v_i n_j A u_k n_l
    rhs[E_F_b, ...] -= torch.einsum(
        'b,bmg,bj,bijklg,bng,bl,bnk,bg->bmi',  # again could easily be wrong...
        mu_e,  # (batch_in)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        u[E_F_b, ...],  # (batch_in, nloc, ndim)
        sdetwei,  # (batch_in, sngi
    )
    del AA  # no longer need
    P = calc_P(nlx=snx, u=u[E_F_b, ...], batch_in=batch_in)
    # [v_i n_j] {P_ij}
    rhs[E_F_b, ...] += torch.einsum(
        'bmg,bj,bijg,bg->bmi',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        P,  # (batch_in, ndim, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
    )
    return rhs

