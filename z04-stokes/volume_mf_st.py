#!/usr/bin/env python3

"""
all integrations for hyper-elastic
matric-free implementation
"""

import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm
import config
from config import sf_nd_nb
# import materials
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
ndim = config.ndim
nface = config.ndim+1


def calc_RAR_mf_color(
        I_dc, I_cd,
        whichc, ncolor,
        fina, cola, ncola,
):
    """
    get operator on P1CG grid, i.e. RAR
    where R is prolongator/restrictor
    via coloring method.

    """
    import time
    start_time = time.time()
    cg_nonods = sf_nd_nb.vel_func_space.cg_nonods
    # p1dg_nonods = sf_nd_nb.vel_func_space.p1dg_nonods
    u_nonods = sf_nd_nb.vel_func_space.nonods
    p_nonods = sf_nd_nb.pre_func_space.nonods
    # u_nloc = sf_nd_nb.vel_func_space.element.nloc
    # p_nloc = sf_nd_nb.pre_func_space.element.nloc

    value = torch.zeros(ncola, ndim + 1, ndim + 1, device=dev, dtype=torch.float64)  # NNZ entry values
    u_dummy = torch.zeros(u_nonods, ndim, device=dev, dtype=torch.float64)  # dummy variable of same length as PnDG
    p_dummy = torch.zeros(p_nonods, device=dev, dtype=torch.float64)  # dummy variable of same length as PnDG
    dummy = [u_dummy, p_dummy]  # this is a dummy rhs == 0.

    Rm_u = torch.zeros(u_nonods, ndim, device=dev, dtype=torch.float64)
    Rm_p = torch.zeros(p_nonods, device=dev, dtype=torch.float64)
    Rm = [Rm_u, Rm_p]
    ARm_u = torch.zeros(u_nonods, ndim, device=dev, dtype=torch.float64)
    ARm_p = torch.zeros(p_nonods, device=dev, dtype=torch.float64)
    ARm = [ARm_u, ARm_p]
    RARm = torch.zeros(cg_nonods, ndim + 1, device=dev, dtype=torch.float64)
    mask = torch.zeros(cg_nonods, ndim + 1, device=dev, dtype=torch.float64)  # color vec
    for color in tqdm(range(1, ncolor + 1)):
        # print('color: ', color)
        for jdim in range(ndim + 1):
            mask *= 0
            mask[:, jdim] += torch.tensor((whichc == color),
                                          device=dev,
                                          dtype=torch.float64)  # 1 if true; 0 if false
            Rm_u *= 0
            Rm_p *= 0
            for idim in range(ndim):
                # Rm[:, idim] += torch.mv(I_dc, mask[:, idim].view(-1))  # (p1dg_nonods, ndim)
                Rm_u[:, idim] += mg_le.vel_p1dg_to_pndg_prolongator(
                    torch.mv(I_dc, mask[:, idim].view(-1))
                )  # (p3dg_nonods, ndim)
            Rm_p += mg_le.pre_p1dg_to_pndg_prolongator(
                torch.mv(I_dc, mask[:, -1].view(-1))
            )
            ARm_u *= 0
            ARm_p *= 0
            ARm = get_residual_only(r0=ARm, x_i=Rm, x_rhs=dummy)
            ARm_u *= -1.  # (p3dg_nonods, ndim)
            ARm_p *= -1.
            # RARm = multi_grid.p3dg_to_p1dg_restrictor(ARm)  # (p1dg_nonods, )
            RARm *= 0
            for idim in range(ndim):
                RARm[:, idim] += torch.mv(I_cd, mg_le.vel_pndg_to_p1dg_restrictor(ARm_u[:, idim]))  # (cg_nonods, ndim)
            RARm[:, -1] += torch.mv(I_cd, mg_le.pre_pndg_to_p1dg_restrictor(ARm_p))
            for idim in range(ndim+1):
                # add to value
                for i in range(RARm.shape[0]):
                    for count in range(fina[i], fina[i + 1]):
                        j = cola[count]
                        value[count, idim, jdim] += RARm[i, idim] * mask[j, jdim]
        # print('finishing (another) one color, time comsumed: ', time.time() - start_time)
    return value


def get_residual_and_smooth_once(r0, x_i, x_rhs):
    """update residual, then do one (block-) Jacobi smooth.
    Block matrix structure:
    [ K    G ]
    [ G^T  S ]
    where S is a diagonal matrix with entries = mu / dx^2 (dx is element diameter)
    """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele, dtype=int)
    # retrive func space / element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    r0[0] = r0[0].view(nele * u_nloc, ndim)
    r0[1] = r0[1].view(-1)
    x_i[0] = x_i[0].view(nele * u_nloc, ndim)
    x_i[1] = x_i[1].view(-1)
    x_rhs[0] = x_rhs[0].view(nele * u_nloc, ndim)
    x_rhs[1] = x_rhs[1].view(-1)
    # add precalculated rhs to residual
    for i in range(2):
        r0[i] += x_rhs[i]
    for i in range(nnn):
        # volume integral
        idx_in = torch.zeros(nele, device=dev, dtype=bool)  # element indices in this batch
        idx_in[brk_pnt[i]:brk_pnt[i + 1]] = True
        batch_in = int(torch.sum(idx_in))
        # dummy diagA and bdiagA
        diagK = torch.zeros(batch_in, u_nloc, ndim, device=dev, dtype=torch.float64)
        bdiagK = torch.zeros(batch_in, u_nloc, ndim, u_nloc, ndim, device=dev, dtype=torch.float64)
        diagS = torch.zeros(batch_in, p_nloc, p_nloc, device=dev, dtype=torch.float64)
        r0, diagK, bdiagK, diagS = _k_res_one_batch(r0, x_i,
                                                    diagK, bdiagK, diagS,
                                                    idx_in)
        # surface integral
        idx_in_f = torch.zeros(nele * nface, dtype=bool, device=dev)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0, diagK, bdiagK = _s_res_one_batch(r0, x_i,
                                             diagK, bdiagK,
                                             idx_in_f, brk_pnt[i])
        # smooth once
        if config.blk_solver == 'direct':
            bdiagK = torch.inverse(bdiagK.view(batch_in, u_nloc * ndim, u_nloc * ndim))
            x_i[0] = x_i[0].view(nele, u_nloc * ndim)
            x_i[0][idx_in, :] += config.jac_wei * torch.einsum('...ij,...j->...i',
                                                               bdiagK,
                                                               r0[0].view(nele, u_nloc * ndim)[idx_in, :])
            x_i[1] = x_i[1].view(nele, p_nloc)
            diagS = torch.inverse(diagS.view(batch_in, p_nloc, p_nloc))
            x_i[1][idx_in, :] += config.jac_wei * torch.einsum('bij,bj->bi',
                                                               diagS,
                                                               r0[1].view(nele, p_nloc)[idx_in, :])
        if config.blk_solver == 'jacobi':
            raise NotImplemented('Jacobi iteration for block diagonal matrix is not implemented!')
    r0[0] = r0[0].view(nele * u_nloc, ndim)
    r0[1] = r0[1].view(-1)
    x_i[0] = x_i[0].view(nele * u_nloc, ndim)
    x_i[1] = x_i[1].view(-1)
    return r0, x_i


def get_residual_only(r0, x_i, x_rhs):
    """update residual,
        Block matrix structure:
        [ K    G ]
        [ G^T  S ]
        where S is a block diagonal mass matrix with blocks
        \int q_i mu / dx^2 q_j (dx is element diameter)
        """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele, dtype=int)
    # retrive func space / element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    r0[0] = r0[0].view(nele * u_nloc, ndim)
    r0[1] = r0[1].view(nele, p_nloc)
    x_rhs[0] = x_rhs[0].view(nele * u_nloc, ndim)
    x_rhs[1] = x_rhs[1].view(nele, p_nloc)
    # add precalculated rhs to residual
    for i in range(2):
        r0[i] += x_rhs[i]
    for i in range(nnn):
        # volume integral
        idx_in = torch.zeros(nele, device=dev, dtype=bool)  # element indices in this batch
        idx_in[brk_pnt[i]:brk_pnt[i + 1]] = True
        batch_in = int(torch.sum(idx_in))
        # dummy diagA and bdiagA
        diagK = torch.zeros(batch_in, u_nloc, ndim, device=dev, dtype=torch.float64)
        bdiagK = torch.zeros(batch_in, u_nloc, ndim, u_nloc, ndim, device=dev, dtype=torch.float64)
        diagS = torch.zeros(batch_in, p_nloc, p_nloc, device=dev, dtype=torch.float64)
        r0, diagK, bdiagK, diagS = _k_res_one_batch(r0, x_i,
                                                    diagK, bdiagK, diagS,
                                                    idx_in)
        # surface integral
        idx_in_f = torch.zeros(nele * nface, dtype=bool, device=dev)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0, diagK, bdiagK = _s_res_one_batch(r0, x_i,
                                             diagK, bdiagK,
                                             idx_in_f, brk_pnt[i])
    r0[0] = r0[0].view(nele * u_nloc, ndim)
    r0[1] = r0[1].view(-1)
    return r0


def _k_res_one_batch(
        r0, x_i,
        diagK, bdiagK, diagS,
        idx_in
):
    """this contains volume integral part of the residual update
    let velocity shape function be N, pressure be Q
    i.e.
    Nx_i Nx_j in K
    Nx_i Q_j in G
    Q_i Nx_j in G^T
    mu/|e|^2 in S
    """
    batch_in = diagK.shape[0]
    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    r0[0] = r0[0].view(-1, u_nloc, ndim)
    r0[1] = r0[1].view(-1, p_nloc)
    x_i[0] = x_i[0].view(-1, u_nloc, ndim)
    x_i[1] = x_i[1].view(-1, p_nloc)
    diagK = diagK.view(-1, u_nloc, ndim)
    bdiagK = bdiagK.view(-1, u_nloc, ndim, u_nloc, ndim)
    diagS = diagS.view(-1, p_nloc, p_nloc)

    # get shape function and derivatives
    n = sf_nd_nb.vel_func_space.element.n
    nx, ndetwei = get_det_nlx(
        nlx=sf_nd_nb.vel_func_space.element.nlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[idx_in],
        weight=sf_nd_nb.vel_func_space.element.weight,
        nloc=u_nloc,
        ngi=sf_nd_nb.vel_func_space.element.ngi
    )
    q = sf_nd_nb.pre_func_space.element.n
    _, qdetwei = get_det_nlx(
        nlx=sf_nd_nb.pre_func_space.element.nlx,
        x_loc=sf_nd_nb.pre_func_space.x_ref_in[idx_in],
        weight=sf_nd_nb.pre_func_space.element.weight,
        nloc=p_nloc,
        ngi=sf_nd_nb.pre_func_space.element.ngi
    )

    # local K
    K = torch.zeros(batch_in, u_nloc, ndim, u_nloc, ndim, device=dev, dtype=torch.float64)
    # Nx_i Nx_j
    K += torch.einsum('bimg,bing,bg,jk->bmjnk', nx, nx, ndetwei,
                      torch.eye(ndim, device=dev, dtype=torch.float64)
                      ) \
        * config.mu
        # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim)\
    if config.isTransient:
        # ni nj
        for idim in range(ndim):
            K[:, :, idim, :, idim] += torch.einsum('mg,ng,bg->bmn', n, n, ndetwei) \
                                      * config.rho / config.dt
    # update residual of velocity block K
    r0[0][idx_in, ...] -= torch.einsum('bminj,bnj->bmi', K, x_i[0][idx_in, ...])
    # get diagonal of velocity block K
    diagK += torch.diagonal(K.view(batch_in, u_nloc * ndim, u_nloc * ndim)
                            , dim1=1, dim2=2).view(batch_in, u_nloc, ndim)
    bdiagK += K

    # local G
    G = torch.zeros(batch_in, u_nloc, ndim, p_nloc, device=dev, dtype=torch.float64)
    # Nx_i Q_j
    G += torch.einsum('bimg,ng,bg->bmin', nx, q, ndetwei) * (-1.)
    # update velocity residual of pressure gradient G * p
    r0[0][idx_in, ...] -= torch.einsum('bmin,bn->bmi', G, x_i[1][idx_in, ...])
    # update pressure residual of velocity divergence G^T * u
    r0[1][idx_in, ...] -= torch.einsum('bmjn,bmj->bn', G, x_i[0][idx_in, ...])

    # local S
    S = torch.zeros(batch_in, p_nloc, p_nloc, device=dev, dtype=torch.float64)
    # mu / dx^2 p q, here we use dx =
    # (in 2D) sqrt(area of element)
    # (in 3D) (volume of element)^(1/3)
    # so dx^2 = (volume of element)^(2/ndim)
    dx2 = 1. / torch.sum(ndetwei, dim=-1)**(2/ndim)
    S += torch.einsum(
        'b,mg,ng,bg->bmn',
        dx2,  # (batch_in)
        q,  # (p_nloc, ngi)
        q,  # (p_nloc, ngi)
        qdetwei,  # (batch_in, ngi)
    ) * config.mu
    # update pressure residual
    r0[1][idx_in] -= torch.einsum(
        'bmn,bn->bm',
        S,
        x_i[1][idx_in, ...]
    )
    # save to diagonal
    diagS += S
    return r0, diagK, bdiagK, diagS


def _s_res_one_batch(
        r0, x_i,
        diagK, bdiagK,
        idx_in_f,
        batch_start_idx
):
    # get essential data
    nbf = sf_nd_nb.vel_func_space.nbf
    alnmt = sf_nd_nb.vel_func_space.alnmt

    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    r0[0] = r0[0].view(-1, u_nloc, ndim)
    r0[1] = r0[1].view(-1, p_nloc)
    x_i[0] = x_i[0].view(-1, u_nloc, ndim)
    x_i[1] = x_i[1].view(-1, p_nloc)

    # separate nbf to get internal face list and boundary face list
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

    # for interior faces
    for iface in range(nface):
        for nb_gi_aln in range(nface - 1):
            idx_iface = (f_i == iface) & (sf_nd_nb.vel_func_space.alnmt[F_i] == nb_gi_aln)
            if idx_iface.sum() < 1:
                # there is nothing to do here, go on
                continue
            r0, diagK, bdiagK = _s_res_fi(
                r0, f_i[idx_iface], E_F_i[idx_iface],
                f_inb[idx_iface], E_F_inb[idx_iface],
                x_i,
                diagK, bdiagK, batch_start_idx,
                nb_gi_aln)

    # update residual for boundary faces
    # r <= r + S*u_bc - S*u_i
    if ndim == 3:
        for iface in range(nface):
            idx_iface = f_b == iface
            r0, diagK, bdiagK = _s_res_fb(
                r0, f_b[idx_iface], E_F_b[idx_iface],
                x_i,
                diagK, bdiagK, batch_start_idx)
    else:
        raise Exception('2D hyper-elasticity not implemented!')
    return r0, diagK, bdiagK


def _s_res_fi(
        r, f_i, E_F_i,
        f_inb, E_F_inb,
        x_i,
        diagK, bdiagK, batch_start_idx,
        nb_gi_aln
):
    """internal faces"""
    batch_in = f_i.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)

    # get element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    # shape function on this side
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_i],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi
    )
    sn = sf_nd_nb.vel_func_space.element.sn[f_i, ...]  # (batch_in, nloc, sngi)
    sq = sf_nd_nb.pre_func_space.element.sn[f_i, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_i, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_i, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_i, ...]  # (batch_in, ndim)

    # shape function on the other side
    snx_nb, _, snormal_nb = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_inb],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi
    )
    # get faces we want
    sn_nb = sf_nd_nb.vel_func_space.element.sn[f_inb, ...]  # (batch_in, nloc, sngi)
    sq_nb = sf_nd_nb.pre_func_space.element.sn[f_inb, ...]  # (batch_in, nloc, sngi)
    snx_nb = snx_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim, nloc, sngi)
    snormal_nb = snormal_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim)
    # change gaussian points order on other side
    nb_aln = sf_nd_nb.vel_func_space.element.gi_align[nb_gi_aln, :]  # nb_aln for velocity element
    snx_nb = snx_nb[..., nb_aln]
    # don't forget to change gaussian points order on sn_nb!
    sn_nb = sn_nb[..., nb_aln]
    nb_aln = sf_nd_nb.pre_func_space.element.gi_align[nb_gi_aln, :]  # nb_aln for pressure element
    sq_nb = sq_nb[..., nb_aln]

    gamma_e = config.eta_e / torch.sum(sdetwei, -1)
    u_ith = x_i[0][E_F_i, ...]
    u_inb = x_i[0][E_F_inb, ...]
    dp_ith = x_i[1][E_F_i, ...]
    dp_inb = x_i[1][E_F_inb, ...]

    # K block
    K = torch.zeros(batch_in, u_nloc, ndim, u_nloc, ndim, device=dev, dtype=torch.float64)
    # this side
    # [v_i n_j] {du_i / dx_j}  consistent term
    K += torch.einsum(
        'bmg,bj,bjng,bg,kl->bmknl',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        torch.eye(ndim, device=dev, dtype=torch.float64),  # (ndim, ndim)
    ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim)
    # {dv_i / dx_j} [u_i n_j]  symmetry term
    K += torch.einsum(
        'bjmg,bng,bj,bg,kl->bmknl',
        snx,  # (batch_in, ndim, nloc, sngi)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        sdetwei,  # (batch_in, sngi)
        torch.eye(ndim, device=dev, dtype=torch.float64),  # (ndim, ndim)
    ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim) \
    # \gamma_e * [v_i][u_i]  penalty term
    K += torch.einsum(
        'bmg,bng,bg,b,ij->bminj',
        sn,  # (batch_in, nloc, sngi)
        sn,  # (batch_in, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        gamma_e,  # (batch_in)
        torch.eye(ndim, device=dev, dtype=torch.float64),
    )
    K *= config.mu
    # update residual
    r[0][E_F_i, ...] -= torch.einsum('bminj,bnj->bmi', K, u_ith)
    # put diagonal into diagK and bdiagK
    diagK[E_F_i-batch_start_idx, :, :] += torch.diagonal(K.view(batch_in, u_nloc*ndim, u_nloc*ndim),
                                                         dim1=1, dim2=2).view(batch_in, u_nloc, ndim)
    bdiagK[E_F_i-batch_start_idx, :, :] += K

    # other side
    K *= 0
    # [v_i n_j] {du_i / dx_j}  consistent term
    K += torch.einsum(
        'bmg,bj,bjng,bg,kl->bmknl',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        snx_nb,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        torch.eye(ndim, device=dev, dtype=torch.float64)
    ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim) \
    # {dv_i / dx_j} [u_i n_j]  symmetry term
    K += torch.einsum(
        'bjmg,bng,bj,bg,kl->bmknl',
        snx,  # (batch_in, ndim, nloc, sngi)
        sn_nb,  # (batch_in, nloc, sngi)
        snormal_nb,  # (batch_in, ndim)
        sdetwei,  # (batch_in, sngi)
        torch.eye(ndim, device=dev, dtype=torch.float64)
    ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim) \
    # \gamma_e * [v_i][u_i]  penalty term
    K += torch.einsum(
        'bmg,bng,bg,b,ij->bminj',
        sn,  # (batch_in, nloc, sngi)
        sn_nb,  # (batch_in, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        gamma_e,  # (batch_in)
        torch.eye(ndim, device=dev, dtype=torch.float64),
    ) * (-1.)  # because n2 \cdot n1 = -1
    K *= config.mu
    # update residual
    r[0][E_F_i, ...] -= torch.einsum('bminj,bnj->bmi', K, u_inb)

    # G block
    del K
    G = torch.zeros(batch_in, u_nloc, ndim, p_nloc, device=dev, dtype=torch.float64)
    # this side
    # [v_i n_i] {p}
    G += torch.einsum(
        'bmg,bi,bng,bg->bmin',
        sn,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim)
        sq,  # (batch_in, p_nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (0.5)
    # update velocity residual from pressure gradient
    r[0][E_F_i, ...] -= torch.einsum('bmin,bn->bmi', G, dp_ith)
    # update pressure residual from velocity divergence
    r[1][E_F_i, ...] -= torch.einsum('bnjm,bnj->bm', G, u_ith)

    # other side
    G *= 0
    # {p} [v_i n_i]
    G += torch.einsum(
        'bmg,bi,bng,bg->bmin',
        sn,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim)
        sq_nb,  # (batch_in, p_nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (0.5)
    # update velocity residual from pressure gradient
    r[0][E_F_i, ...] -= torch.einsum('bmin,bn->bmi', G, dp_inb)
    # G^T
    G *= 0
    # {q} [u_j n_j]
    G += torch.einsum(
        'bmg,bng,bj,bg->bnjm',
        sq,  # (batch_in, p_nloc, sngi)
        sn_nb,  # (batch_in, u_nloc, sngi)
        snormal_nb,  # (batch_in, ndim)
        sdetwei,  # (batch_in, sngi)
    ) * (0.5)
    # update pressure residual from velocity divergence
    r[1][E_F_i, ...] -= torch.einsum('bnjm,bnj->bm', G, u_inb)
    # this concludes surface integral on interior faces.
    return r, diagK, bdiagK


def _s_res_fb(
        r, f_b, E_F_b,
        x_i,
        diagK, bdiagK, batch_start_idx
):
    """boundary faces"""
    batch_in = f_b.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    if batch_in < 1:  # nothing to do here.
        return r

    # get element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    # shape function
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_b],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi
    )
    sn = sf_nd_nb.vel_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    sq = sf_nd_nb.pre_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_b, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_b, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_b, ...]  # (batch_in, ndim)
    gamma_e = config.eta_e / torch.sum(sdetwei, -1)

    u_ith = x_i[0][E_F_b, ...]
    dp_ith = x_i[1][E_F_b, ...]

    # block K
    K = torch.zeros(batch_in, u_nloc, ndim, u_nloc, ndim,
                    device=dev, dtype=torch.float64)
    # [vi nj] {du_i / dx_j}  consistent term
    K -= torch.einsum(
        'bmg,bj,bjng,bg,kl->bmknl',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        torch.eye(ndim, device=dev, dtype=torch.float64)
    )  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim)
    # {dv_i / dx_j} [ui nj]  symmetry term
    K -= torch.einsum(
        'bjmg,bng,bj,bg,kl->bmknl',
        snx,  # (batch_in, ndim, nloc, sngi)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        sdetwei,  # (batch_in, sngi)
        torch.eye(ndim, device=dev, dtype=torch.float64)
    )  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim)
    # \gamma_e [v_i] [u_i]  penalty term
    K += torch.einsum(
        'bmg,bng,bg,b,ij->bminj',
        sn,  # (batch_in, nloc, sngi)
        sn,  # (batch_in, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        gamma_e,  # (batch_in)
        torch.eye(ndim, device=dev, dtype=torch.float64)
    )
    K *= config.mu
    # update residual
    r[0][E_F_b, ...] -= torch.einsum('bminj,bnj->bmi', K, u_ith)
    # put in diagonal
    diagK[E_F_b - batch_start_idx, :, :] += torch.diagonal(K.view(batch_in, u_nloc * ndim, u_nloc * ndim),
                                                           dim1=-2, dim2=-1).view(batch_in, u_nloc, ndim)
    bdiagK[E_F_b - batch_start_idx, ...] += K

    # block G
    del K
    G = torch.zeros(batch_in, u_nloc, ndim, p_nloc,
                    device=dev, dtype=torch.float64)
    # [v_i n_i] {p}
    G += torch.einsum(
        'bmg,bi,bng,bg->bmin',
        sn,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim)
        sq,  # (batch_in, p_nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    )
    # update velocity residual from pressure gradient
    r[0][E_F_b, :, :] -= torch.einsum('bmin,bn->bmi', G, dp_ith)

    # block G^T
    # update pressure residual from velocity divergence
    r[1][E_F_b, :] -= torch.einsum('bnjm,bnj->bm', G, u_ith)

    return r, diagK, bdiagK


def get_rhs(x_rhs, p_i, u_bc, f, u_n=0):
    """get right-hand side"""
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele, dtype=int)
    idx_in = torch.zeros(nele, dtype=torch.bool)
    idx_in_f = torch.zeros(nele * nface, dtype=torch.bool, device=dev)

    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    x_rhs[0] = x_rhs[0].view(nele, u_nloc, ndim)
    x_rhs[1] = x_rhs[1].view(nele, p_nloc)
    # x_i[0] = x_i[0].view(nele, u_nloc, ndim)  # this is u
    # x_i[1] = x_i[1].view(nele, p_nloc)  # this is dp
    p_i = p_i.view(nele, p_nloc)
    u_bc = u_bc.view(nele, u_nloc, ndim)
    f = f.view(nele, u_nloc, ndim)

    for i in range(nnn):
        idx_in *= False
        # volume integral
        idx_in[brk_pnt[i]:brk_pnt[i+1]] = True
        x_rhs = _k_rhs_one_batch(x_rhs, p_i, u_n, f, idx_in)
        # surface integral
        idx_in_f *= False
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        x_rhs = _s_rhs_one_batch(x_rhs, p_i, u_bc, idx_in_f)

    return x_rhs


def _k_rhs_one_batch(
        rhs, p_i, u_n, f, idx_in
):
    batch_in = int(torch.sum(idx_in))
    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    # rhs[0] = rhs[0].view(-1, u_nloc, ndim)
    # rhs[1] = rhs[1].view(-1, p_nloc)
    # p_i = p_i.view(-1, p_nloc)
    # f = f.view(-1, u_nloc, ndim)

    # get shape functions
    n = sf_nd_nb.vel_func_space.element.n
    nx, ndetwei = get_det_nlx(
        nlx=sf_nd_nb.vel_func_space.element.nlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[idx_in],
        weight=sf_nd_nb.vel_func_space.element.weight,
        nloc=u_nloc,
        ngi=sf_nd_nb.vel_func_space.element.ngi
    )
    q = sf_nd_nb.pre_func_space.element.n
    _, qdetwei = get_det_nlx(
        nlx=sf_nd_nb.pre_func_space.element.nlx,
        x_loc=sf_nd_nb.pre_func_space.x_ref_in[idx_in],
        weight=sf_nd_nb.pre_func_space.element.weight,
        nloc=p_nloc,
        ngi=sf_nd_nb.pre_func_space.element.ngi
    )

    # f . v contribution to vel rhs
    rhs[0][idx_in, ...] += torch.einsum(
        'mg,ng,bg,ij,bnj->bmi',
        n,  # (u_nloc, ngi)
        n,  # (u_nloc, ngi)
        ndetwei,  # (batch_in, ngi)
        torch.eye(ndim, device=dev, dtype=torch.float64),  # (ndim, ndim)
        f[idx_in, ...],  # (batch_in, u_nloc, ndim)
    )

    # if transient, add rho/dt * u_n to vel rhs
    if config.isTransient:
        u_n = u_n.view(-1, u_nloc, ndim)
        rhs[0][idx_in, ...] += torch.einsum(
            'bmg,bng,bg,ij,bnj->bmi',
            n,
            n,
            ndetwei,
            torch.eye(ndim, device=dev, dtype=torch.float64),  # (ndim, ndim)
            u_n[idx_in, ...],
        ) * config.rho / config.dt

    # p \nabla.v contribution to vel rhs
    rhs[0][idx_in, ...] += torch.einsum(
        'bimg,ng,bg,bn->bmi',
        nx,  # (batch_in, ndim, u_nloc, ngi)
        q,  # (p_nloc, ngi)
        ndetwei,  # (batch_in, ngi)
        p_i[idx_in, ...],  # (batch_in, p_nloc)
    )

    return rhs


def _s_rhs_one_batch(
        rhs, p_i, u_bc, idx_in_f
):
    # get essential data
    nbf = sf_nd_nb.vel_func_space.nbf
    alnmt = sf_nd_nb.vel_func_space.alnmt

    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    # rhs[0] = rhs[0].view(-1, u_nloc, ndim)
    # rhs[1] = rhs[1].view(-1, p_nloc)
    # u_bc = u_bc.view(-1, u_nloc, ndim)
    # p_i = p_i.view(-1, p_nloc)

    # separate nbf to get internal face list and boundary face list
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

    # for interior faces
    for iface in range(nface):
        for nb_gi_aln in range(nface - 1):
            idx_iface = (f_i == iface) & (sf_nd_nb.vel_func_space.alnmt[F_i] == nb_gi_aln)
            if idx_iface.sum() < 1:
                # there is nothing to do here, go on
                continue
            rhs = _s_rhs_fi(
                rhs, f_i[idx_iface], E_F_i[idx_iface],
                f_inb[idx_iface], E_F_inb[idx_iface],
                p_i,
                nb_gi_aln)

    # update residual for boundary faces
    # r <= r + S*u_bc - S*u_i
    if ndim == 3:
        for iface in range(nface):
            idx_iface = f_b == iface
            rhs = _s_rhs_fb(
                rhs, f_b[idx_iface], E_F_b[idx_iface],
                p_i, u_bc)
    else:
        raise Exception('2D stokes not implemented!')

    return rhs


def _s_rhs_fi(
        rhs, f_i, E_F_i,
        f_inb, E_F_inb,
        p_i,
        nb_gi_aln
):
    batch_in = f_i.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)

    # shape function on this side
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_i],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi
    )
    sn = sf_nd_nb.vel_func_space.element.sn[f_i, ...]  # (batch_in, nloc, sngi)
    sq = sf_nd_nb.pre_func_space.element.sn[f_i, ...]  # (batch_in, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_i, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_i, ...]  # (batch_in, ndim)

    # get faces we want
    sq_nb = sf_nd_nb.pre_func_space.element.sn[f_inb, ...]  # (batch_in, nloc, sngi)
    # change gaussian points order on other side
    nb_aln = sf_nd_nb.pre_func_space.element.gi_align[nb_gi_aln, :]  # nb_aln for pressure element
    # don't forget to change gaussian points order on sn_nb!
    sq_nb = sq_nb[..., nb_aln]

    # this side {p} [v_i n_i]  (Gradient of p)
    rhs[0][E_F_i, ...] -= torch.einsum(
        'bmg,bi,bng,bg,bn->bmi',
        sn,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim)
        sq,  # (batch_in, p_nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        p_i[E_F_i, ...],  # (batch_in, p_nloc)
    ) * (0.5)

    # other side {p} [v_i n_i]  (Gradient of p)
    rhs[0][E_F_i, ...] -= torch.einsum(
        'bmg,bi,bng,bg,bn->bmi',
        sn,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim)
        sq_nb,  # (batch_in, p_nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        p_i[E_F_inb, ...],  # (batch_in, p_nloc)
    ) * (0.5)

    return rhs


def _s_rhs_fb(
        rhs, f_b, E_F_b,
        p_i, u_bc
):
    """
    contains contribution of
    1. vel dirichlet BC to velocity rhs
    2. vel dirichlet BC to pressure rhs
    3. TODO: stress neumann BC to velocity rhs
    4. gradient of p on boundary to velocith rhs
    """
    batch_in = f_b.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    if batch_in < 1:  # nothing to do here.
        return rhs

    # get element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    # shape function
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_b],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi
    )
    sn = sf_nd_nb.vel_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    sq = sf_nd_nb.pre_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_b, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_b, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_b, ...]  # (batch_in, ndim)
    gamma_e = config.eta_e / torch.sum(sdetwei, -1)

    u_bc_th = u_bc[E_F_b, ...]
    p_i_th = p_i[E_F_b, ...]

    # 1.1 {dv_i / dx_j} [u_Di n_j]
    rhs[0][E_F_b, ...] -= torch.einsum(
        'bjmg,bng,bj,bg,bni->bmi',
        snx,  # (batch_in, ndim, u_nloc, sngi)
        sn,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim)
        sdetwei,  # (batch_in, sngi)
        u_bc_th,  # (batch_in, u_nloc, ndim)
    ) * config.mu

    # 1.2 \gamma_e [u_Di] [v_i]
    rhs[0][E_F_b, ...] += torch.einsum(
        'b,bmg,bng,bg,bni->bmi',
        gamma_e,  # (batch_in)
        sn,  # (batch_in, u_nloc, sngi)
        sn,
        sdetwei,  # (batch_in, sngi)
        u_bc_th,  # (batch_in, u_nloc, ndim)
    ) * config.mu

    # 2. {q} [u_Di n_i]
    rhs[1][E_F_b, ...] += torch.einsum(
        'bmg,bng,bi,bni->bm',
        sq,  # (batch_in, p_nloc, sngi)
        sn,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim)
        u_bc_th,  # (batch_in, u_nloc, ndim)
    )

    # 3. TODO: Neumann BC

    # 4. grad p: {p} [v_i n_i]
    rhs[0][E_F_b, ...] -= torch.einsum(
        'bmg,bi,bng,bg,bn->bmi',
        sn,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim)
        sq,  # (batch_in, p_nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        p_i_th,  # (batch_in, p_nloc)
    )

    return rhs


def get_r0_l2_norm(r0):
    """return l2 norm of residual for stokes problem
    r0 is a list of velocity residual and pressure residual
    """
    return torch.linalg.norm(torch.cat((r0[0].view(-1), r0[1].view(-1)), 0))


def update_rhs(x_rhs, p_i):
    """update right-hand side due to pressure correction
    i.e.
    add -G*dp to rhs
    """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele, dtype=int)
    idx_in = torch.zeros(nele, dtype=torch.bool)
    idx_in_f = torch.zeros(nele * nface, dtype=torch.bool, device=dev)

    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    x_rhs[0] = x_rhs[0].view(nele, u_nloc, ndim)
    x_rhs[1] = x_rhs[1].view(nele, p_nloc)
    # x_i[0] = x_i[0].view(nele, u_nloc, ndim)  # this is u
    # x_i[1] = x_i[1].view(nele, p_nloc)  # this is dp
    p_i = p_i.view(nele, p_nloc)

    for i in range(nnn):
        idx_in *= False
        # volume integral
        idx_in[brk_pnt[i]:brk_pnt[i+1]] = True
        x_rhs = _k_update_rhs_one_batch(x_rhs, p_i, idx_in)
        # surface integral
        idx_in_f *= False
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        x_rhs = _s_update_rhs_one_batch(x_rhs, p_i, idx_in_f)

    return x_rhs


def _k_update_rhs_one_batch(
        rhs, p_i, idx_in
):
    batch_in = int(torch.sum(idx_in))
    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    # rhs[0] = rhs[0].view(-1, u_nloc, ndim)
    # rhs[1] = rhs[1].view(-1, p_nloc)
    # p_i = p_i.view(-1, p_nloc)
    # f = f.view(-1, u_nloc, ndim)

    # get shape functions
    n = sf_nd_nb.vel_func_space.element.n
    nx, ndetwei = get_det_nlx(
        nlx=sf_nd_nb.vel_func_space.element.nlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[idx_in],
        weight=sf_nd_nb.vel_func_space.element.weight,
        nloc=u_nloc,
        ngi=sf_nd_nb.vel_func_space.element.ngi
    )
    q = sf_nd_nb.pre_func_space.element.n
    _, qdetwei = get_det_nlx(
        nlx=sf_nd_nb.pre_func_space.element.nlx,
        x_loc=sf_nd_nb.pre_func_space.x_ref_in[idx_in],
        weight=sf_nd_nb.pre_func_space.element.weight,
        nloc=p_nloc,
        ngi=sf_nd_nb.pre_func_space.element.ngi
    )

    # p \nabla.v contribution to vel rhs
    rhs[0][idx_in, ...] += torch.einsum(
        'bimg,ng,bg,bn->bmi',
        nx,  # (batch_in, ndim, u_nloc, ngi)
        q,  # (p_nloc, ngi)
        ndetwei,  # (batch_in, ngi)
        p_i[idx_in, ...],  # (batch_in, p_nloc)
    )

    return rhs


def _s_update_rhs_one_batch(
        rhs, p_i, idx_in_f
):
    # get essential data
    nbf = sf_nd_nb.vel_func_space.nbf
    alnmt = sf_nd_nb.vel_func_space.alnmt

    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    # rhs[0] = rhs[0].view(-1, u_nloc, ndim)
    # rhs[1] = rhs[1].view(-1, p_nloc)
    # u_bc = u_bc.view(-1, u_nloc, ndim)
    # p_i = p_i.view(-1, p_nloc)

    # separate nbf to get internal face list and boundary face list
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

    # for interior faces
    for iface in range(nface):
        for nb_gi_aln in range(nface - 1):
            idx_iface = (f_i == iface) & (sf_nd_nb.vel_func_space.alnmt[F_i] == nb_gi_aln)
            if idx_iface.sum() < 1:
                # there is nothing to do here, go on
                continue
            rhs = _s_update_rhs_fi(
                rhs, f_i[idx_iface], E_F_i[idx_iface],
                f_inb[idx_iface], E_F_inb[idx_iface],
                p_i,
                nb_gi_aln)

    # update residual for boundary faces
    if ndim == 3:
        for iface in range(nface):
            idx_iface = f_b == iface
            rhs = _s_update_rhs_fb(
                rhs, f_b[idx_iface], E_F_b[idx_iface],
                p_i)
    else:
        raise Exception('2D stokes not implemented!')

    return rhs


def _s_update_rhs_fi(
        rhs, f_i, E_F_i,
        f_inb, E_F_inb,
        p_i,
        nb_gi_aln
):
    batch_in = f_i.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)

    # shape function on this side
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_i],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi
    )
    sn = sf_nd_nb.vel_func_space.element.sn[f_i, ...]  # (batch_in, nloc, sngi)
    sq = sf_nd_nb.pre_func_space.element.sn[f_i, ...]  # (batch_in, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_i, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_i, ...]  # (batch_in, ndim)

    # get faces we want
    sq_nb = sf_nd_nb.pre_func_space.element.sn[f_inb, ...]  # (batch_in, nloc, sngi)
    # change gaussian points order on other side
    nb_aln = sf_nd_nb.pre_func_space.element.gi_align[nb_gi_aln, :]  # nb_aln for pressure element
    # don't forget to change gaussian points order on sn_nb!
    sq_nb = sq_nb[..., nb_aln]

    # this side {p} [v_i n_i]  (Gradient of p)
    rhs[0][E_F_i, ...] -= torch.einsum(
        'bmg,bi,bng,bg,bn->bmi',
        sn,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim)
        sq,  # (batch_in, p_nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        p_i[E_F_i, ...],  # (batch_in, p_nloc)
    ) * (0.5)

    # other side {p} [v_i n_i]  (Gradient of p)
    rhs[0][E_F_i, ...] -= torch.einsum(
        'bmg,bi,bng,bg,bn->bmi',
        sn,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim)
        sq_nb,  # (batch_in, p_nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        p_i[E_F_inb, ...],  # (batch_in, p_nloc)
    ) * (0.5)

    return rhs


def _s_update_rhs_fb(
        rhs, f_b, E_F_b,
        p_i
):
    """
    contains contribution of
    . gradient of p on boundary to velocith rhs
    """
    batch_in = f_b.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    if batch_in < 1:  # nothing to do here.
        return rhs

    # get element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    # shape function
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_b],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi
    )
    sn = sf_nd_nb.vel_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    sq = sf_nd_nb.pre_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_b, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_b, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_b, ...]  # (batch_in, ndim)
    gamma_e = config.eta_e / torch.sum(sdetwei, -1)

    p_i_th = p_i[E_F_b, ...]

    # 4. grad p: {p} [v_i n_i]
    rhs[0][E_F_b, ...] -= torch.einsum(
        'bmg,bi,bng,bg,bn->bmi',
        sn,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim)
        sq,  # (batch_in, p_nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        p_i_th,  # (batch_in, p_nloc)
    )

    return rhs
