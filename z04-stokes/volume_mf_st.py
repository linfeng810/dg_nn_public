#!/usr/bin/env python3

"""
all integrations for hyper-elastic
matric-free implementation
"""

import torch
from torch import Tensor
import scipy as sp
import numpy as np
from types import NoneType
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
        I_fc, I_cf,
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
    p1dg_nonods = sf_nd_nb.vel_func_space.p1dg_nonods
    nonods = sf_nd_nb.vel_func_space.nonods
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
                Rm[:, idim] += mg_le.vel_p1dg_to_pndg_prolongator(
                    torch.mv(I_fc, mask[:, idim].view(-1))
                )  # (p3dg_nonods, ndim)
            ARm *= 0
            ARm = get_residual_only(r0=ARm,
                                    x_i=Rm,
                                    x_rhs=dummy,
                                    include_p=False)
            ARm *= -1.  # (p3dg_nonods, ndim)
            # RARm = multi_grid.p3dg_to_p1dg_restrictor(ARm)  # (p1dg_nonods, )
            RARm *= 0
            for idim in range(ndim):
                RARm[:, idim] += torch.mv(I_cf, mg_le.vel_pndg_to_p1dg_restrictor(ARm[:, idim]))  # (cg_nonods, ndim)
            for idim in range(ndim):
                # add to value
                for i in range(RARm.shape[0]):
                    for count in range(fina[i], fina[i + 1]):
                        j = cola[count]
                        value[count, idim, jdim] += RARm[i, idim] * mask[j, jdim]
        print('finishing (another) one color, time comsumed: ', time.time() - start_time)
    return value


def get_residual_and_smooth_once(r0_in, x_i_in, x_rhs, include_p=True):
    """update residual, then do one (block-) Jacobi smooth.
    Block matrix structure:
    [ K    G ]
    [ G^T  S ]
    where S is pressure penalty term

    from now on we will assume r0, x_i, x_rhs all in
    """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele, dtype=int)
    # retrive func space / element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    # r0[0] = r0[0].view(nele * u_nloc, ndim)
    # r0[1] = r0[1].view(-1)
    # x_i[0] = x_i[0].view(nele * u_nloc, ndim)
    # x_i[1] = x_i[1].view(-1)
    # x_rhs[0] = x_rhs[0].view(nele * u_nloc, ndim)
    # x_rhs[1] = x_rhs[1].view(-1)
    # add precalculated rhs to residual
    # for i in range(2):
    #     r0[i] += x_rhs[i]
    r0_in += x_rhs
    for i in range(nnn):
        # volume integral
        idx_in = torch.zeros(nele, device=dev, dtype=bool)  # element indices in this batch
        idx_in[brk_pnt[i]:brk_pnt[i + 1]] = True
        batch_in = int(torch.sum(idx_in))
        # dummy diagA and bdiagA
        diagK = torch.zeros(batch_in, u_nloc, ndim, device=dev, dtype=torch.float64)
        bdiagK = torch.zeros(batch_in, u_nloc, ndim, u_nloc, ndim, device=dev, dtype=torch.float64)
        diagS = torch.zeros(batch_in, p_nloc, p_nloc, device=dev, dtype=torch.float64)
        r0_in, diagK, bdiagK, diagS = _k_res_one_batch(
            r0_in, x_i_in,
            diagK, bdiagK, diagS,
            idx_in, include_p)
        # surface integral
        idx_in_f = torch.zeros(nele * nface, dtype=bool, device=dev)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0_in, diagK, bdiagK, diagS = _s_res_one_batch(
            r0_in, x_i_in,
            diagK, bdiagK, diagS,
            idx_in_f, brk_pnt[i], include_p)
        # smooth once
        if config.blk_solver == 'direct':
            bdiagK = torch.inverse(bdiagK.view(batch_in, u_nloc * ndim, u_nloc * ndim))
            r0 = slicing_x_i(r0_in, include_p)
            x_i = slicing_x_i(x_i_in, include_p)
            # x_i[0] = x_i[0].view(nele, u_nloc * ndim)
            x_i[0][idx_in, :] += config.jac_wei * torch.einsum('...ij,...j->...i',
                                                               bdiagK,
                                                               r0[0].view(nele, u_nloc * ndim)[idx_in, :]
                                                               ).view(nele, u_nloc, ndim)
            if not include_p:
                return r0_in, x_i_in
            # x_i[1] = x_i[1].view(nele, p_nloc)
            # TODO: actually I think we will never reach here.
            diagS = torch.inverse(diagS.view(batch_in, p_nloc, p_nloc))
            x_i[1][idx_in, :] += config.jac_wei * torch.einsum('bij,bj->bi',
                                                               diagS,
                                                               r0[1].view(nele, p_nloc)[idx_in, :])
        if config.blk_solver == 'jacobi':
            raise NotImplemented('Jacobi iteration for block diagonal matrix is not implemented!')
    # r0[0] = r0[0].view(nele * u_nloc, ndim)
    # r0[1] = r0[1].view(-1)
    # x_i[0] = x_i[0].view(nele * u_nloc, ndim)
    # x_i[1] = x_i[1].view(-1)
    return r0_in, x_i_in


def get_residual_only(r0, x_i, x_rhs, include_p=True):
    """update residual,
        Block matrix structure:
        [ K    G ]
        [ G^T  S ]
        K is velocity block, G is gradient,
        G^T is divergence, S is block for pressure penalty term.

        from now on, we will only assume x_i is a 1-D array that has
        the following storage:
        nele * u_nloc * ndim + nele * p_nloc

        by slicing x_i we can get u_i and p_i

        so is x_rhs and r0
        """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele, dtype=int)
    # retrive func space / element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    # add precalculated rhs to residual
    r0 += x_rhs
    for i in range(nnn):
        # volume integral
        idx_in = torch.zeros(nele, device=dev, dtype=bool)  # element indices in this batch
        idx_in[brk_pnt[i]:brk_pnt[i + 1]] = True
        batch_in = int(torch.sum(idx_in))
        # dummy diagA and bdiagA
        diagK = torch.zeros(batch_in, u_nloc, ndim, device=dev, dtype=torch.float64)
        bdiagK = torch.zeros(batch_in, u_nloc, ndim, u_nloc, ndim, device=dev, dtype=torch.float64)
        # TODO: we might omit outputting diagS since we're only doing jacobi iter on block K
        diagS = torch.zeros(batch_in, p_nloc, p_nloc, device=dev, dtype=torch.float64)
        r0, diagK, bdiagK, diagS = _k_res_one_batch(r0, x_i,
                                                    diagK, bdiagK, diagS,
                                                    idx_in, include_p)
        # surface integral
        idx_in_f = torch.zeros(nele * nface, dtype=bool, device=dev)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0, diagK, bdiagK, diagS = _s_res_one_batch(
            r0, x_i,
            diagK, bdiagK, diagS,
            idx_in_f, brk_pnt[i], include_p)
    # if include_p and config.hasNullSpace:
    #     r0[-1] -= x_i[-1]  # this effectively add 1 to the last diagonal entry in pressure block to remove null space.
        # print(r0.shape, x_i.shape)
    # r0[0] = r0[0].view(nele * u_nloc, ndim)
    # r0[1] = r0[1].view(-1)
    return r0


def _k_res_one_batch(
        r0, x_i,
        diagK, bdiagK, diagS,
        idx_in, include_p
):
    """this contains volume integral part of the residual update
    let velocity shape function be N, pressure be Q
    i.e.
    Nx_i Nx_j in K
    Nx_i Q_j in G
    Q_i Nx_j in G^T
    ~mu/|e|^2 in S~ NO S in this block-diagonal preconditioning
    """
    batch_in = diagK.shape[0]
    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    # r0[0] = r0[0].view(-1, u_nloc, ndim)
    # r0[1] = r0[1].view(-1, p_nloc)
    # x_i[0] = x_i[0].view(-1, u_nloc, ndim)
    # x_i[1] = x_i[1].view(-1, p_nloc)
    r0_u, r0_p = slicing_x_i(r0, include_p)
    x_i_u, x_i_p = slicing_x_i(x_i, include_p)
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
    if include_p:
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
    r0_u[idx_in, ...] -= torch.einsum('bminj,bnj->bmi', K, x_i_u[idx_in, ...])
    # get diagonal of velocity block K
    diagK += torch.diagonal(K.view(batch_in, u_nloc * ndim, u_nloc * ndim)
                            , dim1=1, dim2=2).view(batch_in, u_nloc, ndim)
    bdiagK[idx_in, ...] += K

    if include_p:
        # local G
        G = torch.zeros(batch_in, u_nloc, ndim, p_nloc, device=dev, dtype=torch.float64)
        # Nx_i Q_j
        G += torch.einsum('bimg,ng,bg->bmin', nx, q, ndetwei) * (-1.)
        # update velocity residual of pressure gradient G * p
        r0_u[idx_in, ...] -= torch.einsum('bmin,bn->bmi', G, x_i_p[idx_in, ...])
        # update pressure residual of velocity divergence G^T * u
        r0_p[idx_in, ...] -= torch.einsum('bmjn,bmj->bn', G, x_i_u[idx_in, ...])

    # NO S in this block-diagonal preconditioner.
    # local S
    # S = torch.zeros(batch_in, p_nloc, p_nloc, device=dev, dtype=torch.float64)
    # # mu / dx^2 p q, here we use dx =
    # # (in 2D) sqrt(area of element)
    # # (in 3D) (volume of element)^(1/3)
    # # so dx^2 = (volume of element)^(2/ndim)
    # dx2 = 1. / torch.sum(ndetwei, dim=-1)**(2/ndim)
    # S += torch.einsum(
    #     'b,mg,ng,bg->bmn',
    #     dx2,  # (batch_in)
    #     q,  # (p_nloc, ngi)
    #     q,  # (p_nloc, ngi)
    #     qdetwei,  # (batch_in, ngi)
    # ) * config.mu
    # # update pressure residual
    # r0[1][idx_in] -= torch.einsum(
    #     'bmn,bn->bm',
    #     S,
    #     x_i[1][idx_in, ...]
    # )
    # # save to diagonal
    # diagS += S
    return r0, diagK, bdiagK, diagS


def _s_res_one_batch(
        r0, x_i,
        diagK, bdiagK, diagS,
        idx_in_f,
        batch_start_idx,
        include_p
):
    # get essential data
    nbf = sf_nd_nb.vel_func_space.nbf
    alnmt = sf_nd_nb.vel_func_space.alnmt

    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    # r0[0] = r0[0].view(-1, u_nloc, ndim)
    # r0[1] = r0[1].view(-1, p_nloc)
    # x_i[0] = x_i[0].view(-1, u_nloc, ndim)
    # x_i[1] = x_i[1].view(-1, p_nloc)

    # separate nbf to get internal face list and boundary face list
    F_i = torch.where(torch.logical_and(alnmt >= 0, idx_in_f))[0]  # interior face
    # for boundary faces, only include dirichlet boundary faces
    F_b = torch.where(torch.logical_and(
        torch.logical_and(alnmt < 0, sf_nd_nb.vel_func_space.glb_bcface_type == 0),
        idx_in_f))[0]  # boundary face
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
            r0, diagK, bdiagK, diagS = _s_res_fi(
                r0, f_i[idx_iface], E_F_i[idx_iface],
                f_inb[idx_iface], E_F_inb[idx_iface],
                x_i,
                diagK, bdiagK, diagS, batch_start_idx,
                nb_gi_aln, include_p)

    # update residual for boundary faces
    # r <= r + S*u_bc - S*u_i
    if True:  # ndim == 3:  # TODO: we could omit this iface loop in 2D. we'll make do with it at the moment.
        for iface in range(nface):
            idx_iface = f_b == iface
            r0, diagK, bdiagK = _s_res_fb(
                r0, f_b[idx_iface], E_F_b[idx_iface],
                x_i,
                diagK, bdiagK, batch_start_idx, include_p)
    else:
        raise Exception('2D hyper-elasticity not implemented!')
    return r0, diagK, bdiagK, diagS


def _s_res_fi(
        r_in, f_i, E_F_i,
        f_inb, E_F_inb,
        x_i_in,
        diagK, bdiagK, diagS, batch_start_idx,
        nb_gi_aln, include_p
):
    """internal faces"""
    batch_in = f_i.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)

    # get element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    x_i = slicing_x_i(x_i_in, include_p)
    r = slicing_x_i(r_in, include_p)

    # shape function on this side
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_i],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi
    )
    sn = sf_nd_nb.vel_func_space.element.sn[f_i, ...]  # (batch_in, nloc, sngi)
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
    snx_nb = snx_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim, nloc, sngi)
    snormal_nb = snormal_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim)
    # change gaussian points order on other side
    nb_aln = sf_nd_nb.vel_func_space.element.gi_align[nb_gi_aln, :]  # nb_aln for velocity element
    snx_nb = snx_nb[..., nb_aln]
    # don't forget to change gaussian points order on sn_nb!
    sn_nb = sn_nb[..., nb_aln]
    nb_aln = sf_nd_nb.pre_func_space.element.gi_align[nb_gi_aln, :]  # nb_aln for pressure element


    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = config.eta_e / h
    u_ith = x_i[0][E_F_i, ...]
    u_inb = x_i[0][E_F_inb, ...]
    if include_p:
        p_ith = x_i[1][E_F_i, ...]
        p_inb = x_i[1][E_F_inb, ...]

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

    if not include_p:  # no need to go to G, G^T and S
        return r_in, diagK, bdiagK, diagS
    # include p blocks (G, G^T, S)
    sq = sf_nd_nb.pre_func_space.element.sn[f_i, ...]  # (batch_in, nloc, sngi)
    sq_nb = sf_nd_nb.pre_func_space.element.sn[f_inb, ...]  # (batch_in, nloc, sngi)
    sq_nb = sq_nb[..., nb_aln]
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
    r[0][E_F_i, ...] -= torch.einsum('bmin,bn->bmi', G, p_ith)
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
    r[0][E_F_i, ...] -= torch.einsum('bmin,bn->bmi', G, p_inb)
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

    if config.is_pressure_stablise:
        del G
        S = torch.zeros((batch_in, p_nloc, p_nloc), device=dev, dtype=torch.float64)
        # S: pressure penalty term: h[q][p]
        # this side
        S += torch.einsum(
            'b,bmg,bng,bg->bmn',
            h,  # (batch_in)
            sq,  # (batch_in, p_nloc, sngi)
            sq,
            sdetwei,  # (batch_in, sngi)
        )
        r[1][E_F_i, ...] -= torch.einsum('bmn,bn->bm', S, p_ith)
        diagS[E_F_i, ...] += S
        # other side
        S *= 0
        S += torch.einsum(
            'b,bmg,bng,bg->bmn',
            h,  # (batch_in)
            sq,  # (batch_in, p_nloc, sngi)
            sq_nb,  # (batch_in, p_nloc, sngi)
            sdetwei,  # (batch_in, sngi)
        ) * (-1.)  # due to n1 \cdot n2 = -1
        r[1][E_F_i, ...] -= torch.einsum('bmn,bn->bm', S, p_inb)
    # this concludes surface integral on interior faces.
    return r_in, diagK, bdiagK, diagS


def _s_res_fb(
        r_in, f_b, E_F_b,
        x_i_in,
        diagK, bdiagK, batch_start_idx,
        include_p
):
    """boundary faces"""
    batch_in = f_b.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    if batch_in < 1:  # nothing to do here.
        return r_in, diagK, bdiagK

    # get element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    r = slicing_x_i(r_in, include_p)
    x_i = slicing_x_i(x_i_in, include_p)

    # shape function
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_b],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi
    )
    sn = sf_nd_nb.vel_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    if include_p:
        sq = sf_nd_nb.pre_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_b, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_b, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_b, ...]  # (batch_in, ndim)
    if ndim == 3:
        gamma_e = config.eta_e / torch.sqrt(torch.sum(sdetwei, -1))
    else:
        gamma_e = config.eta_e / torch.sum(sdetwei, -1)

    u_ith = x_i[0][E_F_b, ...]
    if include_p:
        p_ith = x_i[1][E_F_b, ...]

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

    if not include_p:
        return r_in, diagK, bdiagK
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
    r[0][E_F_b, :, :] -= torch.einsum('bmin,bn->bmi', G, p_ith)

    # block G^T
    # update pressure residual from velocity divergence
    r[1][E_F_b, :] -= torch.einsum('bnjm,bnj->bm', G, u_ith)

    return r_in, diagK, bdiagK


def get_rhs(x_rhs, u_bc, f, u_n=0):
    """get right-hand side"""
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele, dtype=int)
    idx_in = torch.zeros(nele, dtype=torch.bool)
    idx_in_f = torch.zeros(nele * nface, dtype=torch.bool, device=dev)

    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    # x_rhs[0] = x_rhs[0].view(nele, u_nloc, ndim)
    # x_rhs[1] = x_rhs[1].view(nele, p_nloc)
    # x_i[0] = x_i[0].view(nele, u_nloc, ndim)  # this is u
    # x_i[1] = x_i[1].view(nele, p_nloc)  # this is dp
    # p_i = p_i.view(nele, p_nloc)
    for u_bci in u_bc:
        u_bci = u_bci.view(nele, u_nloc, ndim)
    f = f.view(nele, u_nloc, ndim)

    for i in range(nnn):
        idx_in *= False
        # volume integral
        idx_in[brk_pnt[i]:brk_pnt[i+1]] = True
        x_rhs = _k_rhs_one_batch(x_rhs, u_n, f, idx_in)
        # surface integral
        idx_in_f *= False
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        x_rhs = _s_rhs_one_batch(x_rhs, u_bc, idx_in_f)

    return x_rhs


def _k_rhs_one_batch(
        rhs_in, u_n, f, idx_in
):
    batch_in = int(torch.sum(idx_in))
    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    # rhs[0] = rhs[0].view(-1, u_nloc, ndim)
    # rhs[1] = rhs[1].view(-1, p_nloc)
    # p_i = p_i.view(-1, p_nloc)
    # f = f.view(-1, u_nloc, ndim)
    rhs = slicing_x_i(rhs_in)

    # get shape functions
    n = sf_nd_nb.vel_func_space.element.n
    nx, ndetwei = get_det_nlx(
        nlx=sf_nd_nb.vel_func_space.element.nlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[idx_in],
        weight=sf_nd_nb.vel_func_space.element.weight,
        nloc=u_nloc,
        ngi=sf_nd_nb.vel_func_space.element.ngi
    )
    # q = sf_nd_nb.pre_func_space.element.n
    # _, qdetwei = get_det_nlx(
    #     nlx=sf_nd_nb.pre_func_space.element.nlx,
    #     x_loc=sf_nd_nb.pre_func_space.x_ref_in[idx_in],
    #     weight=sf_nd_nb.pre_func_space.element.weight,
    #     nloc=p_nloc,
    #     ngi=sf_nd_nb.pre_func_space.element.ngi
    # )

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

    # # p \nabla.v contribution to vel rhs
    # rhs[0][idx_in, ...] += torch.einsum(
    #     'bimg,ng,bg,bn->bmi',
    #     nx,  # (batch_in, ndim, u_nloc, ngi)
    #     q,  # (p_nloc, ngi)
    #     ndetwei,  # (batch_in, ngi)
    #     p_i[idx_in, ...],  # (batch_in, p_nloc)
    # )

    return rhs_in


def _s_rhs_one_batch(
        rhs, u_bc, idx_in_f
):
    # get essential data
    nbf = sf_nd_nb.vel_func_space.nbf
    alnmt = sf_nd_nb.vel_func_space.alnmt

    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    # separate nbf to get internal face list and boundary face list
    F_b_d = torch.where(torch.logical_and(
        torch.logical_and(alnmt < 0, sf_nd_nb.vel_func_space.glb_bcface_type == 0),
        idx_in_f))[0]  # boundary face
    F_b_n = torch.where(torch.logical_and(
        torch.logical_and(alnmt < 0, sf_nd_nb.vel_func_space.glb_bcface_type == 1),
        idx_in_f))[0]  # boundary face

    E_F_b_d = torch.floor_divide(F_b_d, nface)
    E_F_b_n = torch.floor_divide(F_b_n, nface)

    # local face number
    f_b_d = torch.remainder(F_b_d, nface)
    f_b_n = torch.remainder(F_b_n, nface)

    # # for interior faces
    # for iface in range(nface):
    #     for nb_gi_aln in range(nface - 1):
    #         idx_iface = (f_i == iface) & (sf_nd_nb.vel_func_space.alnmt[F_i] == nb_gi_aln)
    #         if idx_iface.sum() < 1:
    #             # there is nothing to do here, go on
    #             continue
    #         rhs = _s_rhs_fi(
    #             rhs, f_i[idx_iface], E_F_i[idx_iface],
    #             f_inb[idx_iface], E_F_inb[idx_iface],
    #             p_i,
    #             nb_gi_aln)
    hasNeuBC = False
    if len(u_bc) > 1: hasNeuBC = True
    # update residual for boundary faces
    # r <= r + S*u_bc - S*u_i
    if True:  # ndim == 3:  # this is slower for 2D, but we'll make do with it.
        for iface in range(nface):
            idx_iface_d = f_b_d == iface
            idx_iface_n = f_b_n == iface
            rhs = _s_rhs_fb(
                rhs,
                f_b_d[idx_iface_d], E_F_b_d[idx_iface_d],
                u_bc[0])
            if not hasNeuBC: continue
            rhs = _s_rhs_fb_neumann(
                rhs,
                f_b_n[idx_iface_n], E_F_b_n[idx_iface_n],
                u_bc[1])
    else:
        raise Exception('2D stokes not implemented!')

    return rhs


def _s_rhs_fi(
        rhs, f_i, E_F_i,
        f_inb, E_F_inb,
        p_i,
        nb_gi_aln
):
    """this is of no use for block-diagonal preconditioner"""
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
        rhs_in, f_b, E_F_b,
        u_bc
):
    """
    contains contribution of
    1. vel dirichlet BC to velocity rhs
    2. vel dirichlet BC to pressure rhs
    ~3. : stress neumann BC to velocity rhs~
    ~4. gradient of p on boundary to velocith rhs~
    """
    batch_in = f_b.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    if batch_in < 1:  # nothing to do here.
        return rhs_in

    # get element parameters
    # u_nloc = sf_nd_nb.vel_func_space.element.nloc
    # p_nloc = sf_nd_nb.pre_func_space.element.nloc
    rhs = slicing_x_i(rhs_in)

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
    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = config.eta_e / h
    # print('gamma_e', gamma_e)
    u_bc_th = u_bc[E_F_b, ...]
    # p_i_th = p_i[E_F_b, ...]

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
        'bmg,bng,bi,bg,bni->bm',
        sq,  # (batch_in, p_nloc, sngi)
        sn,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim)
        sdetwei,  # (batch_in, sngi)
        u_bc_th,  # (batch_in, u_nloc, ndim)
    )

    # ~3. : Neumann BC~

    # # 4. grad p: {p} [v_i n_i]
    # rhs[0][E_F_b, ...] -= torch.einsum(
    #     'bmg,bi,bng,bg,bn->bmi',
    #     sn,  # (batch_in, u_nloc, sngi)
    #     snormal,  # (batch_in, ndim)
    #     sq,  # (batch_in, p_nloc, sngi)
    #     sdetwei,  # (batch_in, sngi)
    #     p_i_th,  # (batch_in, p_nloc)
    # )

    return rhs_in


def _s_rhs_fb_neumann(
        rhs_in, f_b, E_F_b,
        u_bc
):
    """
    contains contribution of
    3. stress neumann BC to velocity rhs
    """
    batch_in = f_b.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    if batch_in < 1:  # nothing to do here.
        return rhs_in

    # get element parameters
    # u_nloc = sf_nd_nb.vel_func_space.element.nloc
    # p_nloc = sf_nd_nb.pre_func_space.element.nloc
    rhs = slicing_x_i(rhs_in)

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
    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = config.eta_e / h
    # print('gamma_e', gamma_e)
    u_bc_th = u_bc[E_F_b, ...]

    # 3. TODO: Neumann BC
    rhs[0][E_F_b, ...] += torch.einsum(
        'bmg,bng,bg,bni->bmi',
        sn,   # (batch_in, nloc, sngi)
        sn,   # (batch_in, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        u_bc_th,  # (batch_in, u_nloc, ndim)
    )

    return rhs_in


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


def slicing_x_i(x_i, include_p=True):
    """slicing a one-dimention array
    to get u and p"""
    if include_p:
        u_nloc = sf_nd_nb.vel_func_space.element.nloc
        p_nloc = sf_nd_nb.pre_func_space.element.nloc
        u = x_i[0:nele*u_nloc*ndim].view(nele, u_nloc, ndim)
        p = x_i[nele*u_nloc*ndim:nele*u_nloc*ndim+nele*p_nloc].view(nele, p_nloc)
    else:
        u_nloc = sf_nd_nb.vel_func_space.element.nloc
        u = x_i[0:nele*u_nloc*ndim].view(nele, u_nloc, ndim)
        p = 0  # put in 0 to ... do nothing...
    return u, p


def get_l2_error(x_i_in, x_ana):
    """given numerical solution and analytical solution (exact up to interpolation error),
    calculate l2 error"""
    if not type(x_i_in) == torch.Tensor:
        x_i = torch.tensor(x_i_in, device=dev, dtype=torch.float64)
    else:
        x_i = x_i_in
    u_i, p_i = slicing_x_i(x_i)
    u_ana, p_ana = slicing_x_i(x_ana)
    n = sf_nd_nb.vel_func_space.element.n
    q = sf_nd_nb.pre_func_space.element.n
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    _, ndetwei = get_det_nlx(
        nlx=sf_nd_nb.vel_func_space.element.nlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in,
        weight=sf_nd_nb.vel_func_space.element.weight,
        nloc=u_nloc,
        ngi=sf_nd_nb.vel_func_space.element.ngi
    )
    u_i_gi = torch.einsum('ng,bni->bnig', n, u_i)
    u_ana_gi = torch.einsum('ng,bni->bnig', n, u_ana)
    u_l2 = torch.einsum(
        'bnig,bg->bni',
        (u_i_gi - u_ana_gi)**2,
        ndetwei,
    )
    u_l2 = torch.sum(torch.sum(torch.sum(u_l2)))
    u_l2 = torch.sqrt(u_l2).cpu().numpy()
    p_i_gi = torch.einsum('ng,bn->bng', q, p_i)
    p_ana_gi = torch.einsum('ng,bn->bng', q, p_ana)
    p_l2 = torch.einsum(
        'bng,bg->bn',
        (p_i_gi - p_ana_gi)**2,
        ndetwei,
    )
    p_l2 = torch.sum(torch.sum(p_l2))
    p_l2 = torch.sqrt(p_l2).cpu().numpy()

    u_linf = torch.max(torch.max(torch.max(torch.abs(u_i - u_ana))))
    p_linf = torch.max(torch.max(torch.abs(p_i - p_ana)))
    return u_l2, p_l2, u_linf, p_linf


def pre_blk_precon(x_p):
    """given vector x_p, apply pressure block's preconditioner:
    x_p <- Q^-1 x_p
    here Q is the mass matrix of pressure space
    """
    # x_p should be of dimension (nele, p_nloc) or (p_nonods)
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    x_p = x_p.view(nele, p_nloc)
    q = sf_nd_nb.pre_func_space.element.n
    _, qdetwei = get_det_nlx(
        nlx=sf_nd_nb.pre_func_space.element.nlx,
        x_loc=sf_nd_nb.pre_func_space.x_ref_in,
        weight=sf_nd_nb.pre_func_space.element.weight,
        nloc=p_nloc,
        ngi=sf_nd_nb.pre_func_space.element.ngi
    )
    Q = torch.einsum(
        'mg,ng,bg->bmn',
        q,
        q,
        qdetwei
    )
    Q = torch.linalg.inv(Q)
    x_temp = torch.einsum('bmn,bn->bm', Q, x_p)
    x_p *= 0
    x_p += x_temp
    return x_p.view(-1)


def vel_blk_precon(x_u, x_rhs):
    """
    given vector x_u, apply velocity block's preconditioner:
    x_u <- K^-1 x_rhs
    here K^-1 is the approximation of the inverse of K,
    The approximation is computed by a multi-grid cycle.

    This operation equals to _multigrid_one_cycle.
    """
    nonods = sf_nd_nb.vel_func_space.nonods
    nloc = sf_nd_nb.vel_func_space.element.nloc
    x_u = x_u.view(nonods, ndim)
    x_rhs = x_rhs.view(nonods, ndim)
    cg_nonods = sf_nd_nb.vel_func_space.cg_nonods
    r0 = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)

    # pre smooth
    for its1 in range(config.pre_smooth_its):
        r0 *= 0
        r0, x_u = get_residual_and_smooth_once(r0, x_u, x_rhs=x_rhs, include_p=False)

    # get residual on PnDG
    r0 *= 0
    r0 = get_residual_only(r0, x_u, x_rhs=x_rhs, include_p=False)

    # restrict residual
    if not config.is_pmg:
        r1 = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
        for idim in range(ndim):
            r1[:, idim] += torch.mv(sf_nd_nb.I_cd, mg_le.vel_pndg_to_p1dg_restrictor(r0[:, idim]))
    else:
        raise NotImplemented('PMG not implemented!')

    # smooth/solve on P1CG grid
    if not config.is_sfc:  # two-grid method
        e_i = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
        e_direct = sp.sparse.linalg.spsolve(
            sf_nd_nb.RARmat,
            r1.contiguous().view(-1).cpu().numpy()
        )
        e_i += torch.tensor(e_direct, device=dev, dtype=torch.float64).view(cg_nonods, ndim)
    else:  # multi-grid on space-filling curve generated grids
        ncurve = 1  # always use 1 sfc
        N = len(sf_nd_nb.sfc_data.space_filling_curve_numbering)
        inverse_numbering = np.zeros((N, ncurve), dtype=int)
        inverse_numbering[:, 0] = np.argsort(sf_nd_nb.sfc_data.space_filling_curve_numbering[:, 0])
        r1_sfc = r1[inverse_numbering[:, 0], :].view(cg_nonods, ndim)

        # go to SFC coarse grid levels and do 1 mg cycles there
        e_i = mg_le.mg_on_P1CG(
            r1_sfc.view(cg_nonods, ndim),
            sf_nd_nb.sfc_data.variables_sfc,
            sf_nd_nb.sfc_data.nlevel,
            sf_nd_nb.sfc_data.nodes_per_level
        )
        # reverse to original order
        e_i = e_i[sf_nd_nb.sfc_data.space_filling_curve_numbering[:, 0] - 1, :].view(cg_nonods, ndim)

    # prolongate residual
    if not config.is_pmg:
        e_i0 = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)
        for idim in range(ndim):
            e_i0[:, idim] += mg_le.vel_p1dg_to_pndg_prolongator(torch.mv(sf_nd_nb.I_dc, e_i[:, idim]))
    else:
        raise Exception('pmg not implemented')

    # correct find grid solution
    x_u += e_i0

    # post smooth
    for its1 in range(config.pre_smooth_its):
        r0 *= 0
        r0, x_u = get_residual_and_smooth_once(r0, x_u, x_rhs=x_rhs, include_p=False)
    # print('x_rhs norm: ', torch.linalg.norm(x_rhs.view(-1)), 'r0 norm: ', torch.linalg.norm(r0.view(-1)))
    return x_u.view(nele, nloc, ndim)


def vel_blk_precon_direct_inv(x_rhs):
    """
    as a test,
    use direct inverse of K as velocity block preconditioner.
    i.e. lhs mat is
    A = [ K,   G, ]
        [ G^t, S  ]
        preconditioner is
    M = [ K,   0, ]
        [ 0,   Q  ]

    in this subroutine we apply K^-1 to x_rhs
    """
    # we will use stokes_assemble to get K
    import stokes_assemble
    u_nonods = sf_nd_nb.vel_func_space.nonods
    p_nonods = sf_nd_nb.pre_func_space.nonods
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    if type(sf_nd_nb.Kmatinv) is NoneType:
        dummy = torch.zeros(u_nonods*ndim, device=dev, dtype=torch.float64)
        Amat_sp, _ = stokes_assemble.assemble(u_bc_in=[dummy, dummy],
                                              f=dummy)
        Kmat = Amat_sp.todense()[0:u_nonods*ndim, 0:u_nonods*ndim]
        Kmatinv = np.linalg.inv(Kmat)
        sf_nd_nb.set_data(Kmatinv=Kmatinv)
    else:
        Kmatinv = sf_nd_nb.Kmatinv
    x_in = x_rhs.view(-1).cpu().numpy()
    x_out = np.matmul(Kmatinv, x_in)
    x_rhs *= 0
    x_rhs += torch.tensor(x_out, device=dev, dtype=torch.float64).view(nele, u_nloc, ndim)
    return x_rhs
