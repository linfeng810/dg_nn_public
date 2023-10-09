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
import volume_mf_st
from config import sf_nd_nb
import materials
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
nele_f = config.nele_f
nele_s = config.nele_s
ndim = config.ndim
nface = config.ndim+1
cijkl = config.cijkl
lam = config.lam
mu = config.mu
# note on indices:
#  b: batch_in,
#  n: nloc,
#  g: ngi, or sngi, gaussian points
#  i,j,k,l: dimension of tensors,
#  i,j: can also refer to iloc, jloc :-(
material = materials.NeoHookean(sf_nd_nb.disp_func_space.element.nloc,
                                ndim, dev, mu, lam)
# material = materials.LinearElastic(nloc, ndim, dev, mu, lam)


def calc_RAR_mf_color(
        I_fc, I_cf,
        whichc, ncolor,
        fina, cola, ncola,
        x_k  # this is the displacement field at current non-linear step,
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
    cg_nonods = sf_nd_nb.sparse_s.cg_nonods
    p1dg_nonods = sf_nd_nb.sparse_s.p1dg_nonods
    total_no_dofs = (sf_nd_nb.vel_func_space.nonods
                     + sf_nd_nb.pre_func_space.nonods
                     + sf_nd_nb.disp_func_space.nonods)
    value = torch.zeros(ncola, ndim, ndim, device=dev, dtype=torch.float64)  # NNZ entry values

    dummy = torch.zeros(total_no_dofs, device=dev, dtype=torch.float64)  # dummy variable of same length as PnDG
    Rm = torch.zeros(total_no_dofs, device=dev, dtype=torch.float64)
    ARm = torch.zeros(total_no_dofs, device=dev, dtype=torch.float64)
    dummy_dict = volume_mf_st.slicing_x_i(dummy)
    Rm_dict = volume_mf_st.slicing_x_i(Rm)
    ARm_dict = volume_mf_st.slicing_x_i(ARm)

    RARm = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
    mask = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)  # color vec
    for color in tqdm(range(1, ncolor + 1)):
        # print('color: ', color)
        for jdim in range(ndim):
            mask *= 0
            mask[:, jdim] += torch.tensor((whichc == color),
                                          device=dev,
                                          dtype=torch.float64)  # 1 if true; 0 if false
            Rm *= 0
            for idim in range(ndim):
                # Rm[:, idim] += torch.mv(I_fc, mask[:, idim].view(-1))  # (p1dg_nonods, ndim)
                Rm['disp'][nele_f:nele, :, idim] += mg_le.vel_p1dg_to_pndg_prolongator(
                    torch.mv(I_fc, mask[:, idim].view(-1))
                )  # (p3dg_nonods, ndim)
            ARm *= 0
            ARm = get_residual_only(ARm, x_k,
                                    Rm, dummy,
                                    include_s_blk=True,
                                    include_itf=False)
            ARm *= -1.  # (p3dg_nonods, ndim)
            # RARm = multi_grid.p3dg_to_p1dg_restrictor(ARm)  # (p1dg_nonods, )
            RARm *= 0
            for idim in range(ndim):
                RARm[:, idim] += torch.mv(
                    I_cf,
                    mg_le.vel_pndg_to_p1dg_restrictor(ARm[nele_f:nele, :, idim])
                )  # (cg_nonods, ndim)
            for idim in range(ndim):
                # add to value
                for i in range(RARm.shape[0]):
                    for count in range(fina[i], fina[i + 1]):
                        j = cola[count]
                        value[count, idim, jdim] += RARm[i, idim] * mask[j, jdim]
        # print('finishing (another) one color, time comsumed: ', time.time() - start_time)
    return value


def get_residual_and_smooth_once(
        r0, x_k, x_i, x_rhs,
        include_s_blk=True,
        include_itf=True,
):
    """
    update residual, then do one (block-) Jacobi smooth.
    will output r0 and updated u_i
    in case one want to get how large the linear residual is
    """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele_s + nele_f, dtype=int)
    
    nloc = sf_nd_nb.disp_func_space.element.nloc
    r0_dict = volume_mf_st.slicing_x_i(r0)
    x_rhs_dict = volume_mf_st.slicing_x_i(x_rhs)
    # add precalculated rhs to residual
    r0_dict['disp'] += x_rhs_dict['disp']
    for i in range(nnn):
        # volume integral
        idx_in = torch.zeros(nele, device=dev, dtype=torch.bool)
        idx_in[brk_pnt[i]:brk_pnt[i + 1]] = True
        batch_in = int(torch.sum(idx_in))
        # dummy diagA and bdiagA
        diagA = torch.zeros(batch_in, nloc, ndim, device=dev, dtype=torch.float64)
        bdiagA = torch.zeros(batch_in, nloc, ndim, nloc, ndim, device=dev, dtype=torch.float64)
        if include_s_blk:
            r0, diagA, bdiagA = _k_res_one_batch(r0, x_k, x_i,
                                                 diagA, bdiagA,
                                                 idx_in)
        # surface integral
        idx_in_f = torch.zeros(nele * nface, dtype=torch.bool, device=dev)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0, diagA, bdiagA = _s_res_one_batch(r0, x_k, x_i,
                                             diagA, bdiagA,
                                             idx_in_f, brk_pnt[i],
                                             include_s_blk, include_itf)
        # smooth once
        if config.blk_solver == 'direct':
            bdiagA = torch.inverse(bdiagA.view(batch_in, nloc * ndim, nloc * ndim))
            x_i_dict = volume_mf_st.slicing_x_i(x_i)
            x_i['disp'][idx_in, :] += config.jac_wei * torch.einsum(
                '...ij,...j->...i',
                bdiagA,
                r0_dict['disp'].view(nele, nloc * ndim)[idx_in, :]
            )
        if config.blk_solver == 'jacobi':
            raise Exception('blk solver not tested for hyper elastic!')
            # new_b = torch.einsum('...ij,...j->...i',
            #                      bdiagA.view(batch_in, nloc * ndim, nloc * ndim),
            #                      du_i.view(nele, nloc * ndim)[idx_in, :]) \
            #         + config.jac_wei * r0.view(nele, nloc * ndim)[idx_in, :]
            # new_b = new_b.view(-1)
            # diagA = diagA.view(-1)
            # du_i = du_i.view(nele, nloc * ndim)
            # du_i_partial = du_i[idx_in, :]
            # for its in range(3):
            #     du_i_partial += ((new_b - torch.einsum('...ij,...j->...i',
            #                                            bdiagA.view(batch_in, nloc * ndim, nloc * ndim),
            #                                            du_i_partial).view(-1))
            #                     / diagA).view(-1, nloc * ndim)
            # du_i[idx_in, :] = du_i_partial.view(-1, nloc * ndim)
    # r0 = r0.view(nele * nloc, ndim)
    # du_i = du_i.view(nele * nloc, ndim)
    return r0, du_i


def get_residual_only(
        r0, x_k, x_i, x_rhs,
        include_s_blk=True,
        include_itf=True,
):
    """
    update residual for solid
    r0 = x_rhs - A*x_i

    where A = [I_SF,u  I_SF,p  S]
    x_i = [u p d]^T
    """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele_s + nele_f, dtype=int)

    nloc = sf_nd_nb.disp_func_space.element.nloc
    r0_dict = volume_mf_st.slicing_x_i(r0)
    x_rhs_dict = volume_mf_st.slicing_x_i(x_rhs)
    # add pre-computed right hand side to residual
    r0_dict['disp'] += x_rhs_dict['disp']
    for i in range(nnn):
        # volume integral
        idx_in = torch.zeros(nele, device=dev, dtype=torch.bool)
        idx_in[brk_pnt[i]:brk_pnt[i + 1]] = True
        batch_in = int(torch.sum(idx_in))
        # dummy diagA and bdiagA
        diagA = torch.zeros(batch_in, nloc, ndim, device=dev, dtype=torch.float64)
        bdiagA = torch.zeros(batch_in, nloc, ndim, nloc, ndim, device=dev, dtype=torch.float64)
        if include_s_blk:
            r0, diagA, bdiagA = _k_res_one_batch(r0, x_k, x_i,
                                                 diagA, bdiagA,
                                                 idx_in)
        # surface integral
        idx_in_f = torch.zeros(nele * nface, dtype=torch.bool, device=dev)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0, diagA, bdiagA = _s_res_one_batch(r0, x_k, x_i,
                                             diagA, bdiagA,
                                             idx_in_f, brk_pnt[i],
                                             include_s_blk, include_itf)
    # r0 = r0.view(nele * nloc, ndim)
    return r0


def _k_res_one_batch(
        r0_in, x_k, x_i,
        diagA, bdiagA,
        idx_in
):
    batch_in = diagA.shape[0]
    nloc = sf_nd_nb.disp_func_space.element.nloc
    # change view
    r0_dict = volume_mf_st.slicing_x_i(r0_in)
    x_k_dict = volume_mf_st.slicing_x_i(x_k)
    x_i_dict = volume_mf_st.slicing_x_i(x_k)
    diagA = diagA.view(-1, nloc, ndim)
    bdiagA = bdiagA.view(-1, nloc, ndim, nloc, ndim)
    # get shape function and derivatives
    n = sf_nd_nb.disp_func_space.element.n
    nx, detwei = get_det_nlx(
        nlx=sf_nd_nb.disp_func_space.element.nlx,
        x_loc=sf_nd_nb.disp_func_space.x_ref_in[idx_in],
        weight=sf_nd_nb.disp_func_space.element.weight,
        nloc=nloc,
        ngi=sf_nd_nb.disp_func_space.element.ngi
    )
    AA = material.calc_AA(nx=nx, u=x_k_dict['disp'][idx_in, ...], batch_in=batch_in)

    K = torch.zeros(nele, nloc, ndim, nloc, ndim, device=dev, dtype=torch.float64)
    # (\nabla v)_ij A (\nabla \delta u)_kl
    K += torch.einsum('bjmg,bijklg,blng,bg->bmink', nx, AA, nx, detwei)
    if config.isTransient:
        # ni nj
        K += (torch.einsum('mg,ng,bg,ij->bminj', n, n, detwei, torch.eye(ndim, device=dev, dtype=torch.float64))
              / config.dt
              * config.rho_s * sf_nd_nb.bdfscm.beta[0])
    # update residual
    r0_dict['disp'][idx_in, ...] -= torch.einsum('bminj,bnj->bmi', K, x_i_dict['disp'][idx_in, ...])
    # get diagonal
    diagA += torch.diagonal(K.view(batch_in, nloc * ndim, nloc * ndim), dim1=1, dim2=2).view(batch_in, nloc, ndim)
    bdiagA += K
    return r0_in, diagA, bdiagA


def _s_res_one_batch(
        r_in, x_k, x_i,
        diagA, bdiagA,
        idx_in_f: Tensor,
        batch_start_idx,
        include_s_blk: bool,
        include_itf: bool,
):
    # get essential data
    nbf = sf_nd_nb.disp_func_space.nbf
    alnmt = sf_nd_nb.disp_func_space.alnmt
    glb_bcface_type = sf_nd_nb.disp_func_space.glb_bcface_type

    # u_i = u_i.view(nele, nloc, ndim)
    # du_i = du_i.view(nele, nloc, ndim)
    # r = r.view(nele, nloc, ndim)

    # separate nbf to get internal face list and boundary face list
    F_i = torch.where(torch.logical_and(glb_bcface_type < 0,
                                        idx_in_f))[0]  # interior face of solid subdomain
    F_inb = nbf[F_i]  # neighbour list of interior face
    F_inb = F_inb.type(torch.int64)
    F_b = torch.where(torch.logical_and(
        sf_nd_nb.disp_func_space.glb_bcface_type == 2,
        idx_in_f))[0]  # dirichlet boundary face
    # F_b_n = torch.where(torch.logical_and(
    #     torch.logical_and(alnmt < 0, sf_nd_nb.disp_func_space.glb_bcface_type == 3),
    #     idx_in_f))[0]  # neumann boundary face
    F_itf = torch.where(torch.logical_and(
        sf_nd_nb.disp_func_space.glb_bcface_type == 5,
        idx_in_f))[0]  # boundary face  (interface)
    F_itf_nb = nbf[F_itf].type(torch.int64)  # neighbour list of interface face

    # create two lists of which element f_i / f_b is in
    E_F_i = torch.floor_divide(F_i, nface)
    E_F_inb = torch.floor_divide(F_inb, nface)
    E_F_b = torch.floor_divide(F_b, nface)
    # E_F_b_n = torch.floor_divide(F_b_n, nface)
    E_F_itf = torch.floor_divide(F_itf, nface)
    E_F_itf_nb = torch.floor_divide(F_itf_nb, nface)

    # local face number
    f_b = torch.remainder(F_b, nface)
    # f_b_n = torch.remainder(F_b_n, nface)
    f_i = torch.remainder(F_i, nface)
    f_inb = torch.remainder(F_inb, nface)
    f_itf = torch.remainder(F_itf, nface)
    f_itf_nb = torch.remainder(F_itf_nb, nface)

    # for interior faces, update residual
    # r <= r - S*u_i
    # let's hope that each element has only one boundary face.
    for iface in range(nface):
        for nb_gi_aln in range(nface - 1):
            if include_s_blk:
                idx_iface = (f_i == iface) & (sf_nd_nb.alnmt[F_i] == nb_gi_aln)
                if idx_iface.sum() < 1:
                    # there is nothing to do here, go on
                    continue
                r_in, diagA, bdiagA = _s_res_fi(
                    r_in, f_i[idx_iface], E_F_i[idx_iface],
                    f_inb[idx_iface], E_F_inb[idx_iface],
                    x_k, x_i,
                    diagA, bdiagA, batch_start_idx,
                    nb_gi_aln)

    # update residual for boundary faces
    # r <= r + S*u_bc - S*u_i
    if ndim == 3:
        for iface in range(nface):
            if include_s_blk:
                idx_iface = f_b == iface
                r_in, diagA, bdiagA = _s_res_fb(
                    r_in, f_b[idx_iface], E_F_b[idx_iface],
                    x_k, x_i,
                    diagA, bdiagA, batch_start_idx)
            if include_itf:
                idx_iface = f_itf == iface
                r_in = _s_res_fitf(
                    r_in, x_k, x_i,
                    f_itf[idx_iface], E_F_itf[idx_iface],
                    f_itf_nb[idx_iface], E_F_itf_nb[idx_iface],
                )
    else:
        raise Exception('2D hyper-elasticity not implemented!')
        r, diagA, bdiagA = _S_fb(
            r, f_b, E_F_b,
            u_i, du_i,
            diagA, bdiagA, batch_start_idx)
    return r_in, diagA, bdiagA


def _s_res_fi(
        r_in, f_i, E_F_i,
        f_inb, E_F_inb,
        x_k, x_i,
        diagA, bdiagA, batch_start_idx,
        nb_gi_aln
):
    batch_in = f_i.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    nloc = sf_nd_nb.disp_func_space.element.nloc

    # change view
    r_dict = volume_mf_st.slicing_x_i(r_in)
    x_k_dict = volume_mf_st.slicing_x_i(x_k)
    x_i_dict = volume_mf_st.slicing_x_i(x_i)
    # shape function on this side
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.disp_func_space.element.snlx,
        x_loc=sf_nd_nb.disp_func_space.x_ref_in[E_F_i],
        sweight=sf_nd_nb.disp_func_space.element.sweight,
        nloc=sf_nd_nb.disp_func_space.element.nloc,
        sngi=sf_nd_nb.disp_func_space.element.sngi,
        sn=sf_nd_nb.disp_func_space.element.sn,
    )
    sn = sf_nd_nb.disp_func_space.sn[f_i, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_i, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_i, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_i, ...]  # (batch_in, ndim, sngi)

    # shape function on the other side
    snx_nb, _, snormal_nb = sdet_snlx(
        snlx=sf_nd_nb.disp_func_space.element.snlx,
        x_loc=sf_nd_nb.disp_func_space.x_ref_in[E_F_inb],
        sweight=sf_nd_nb.disp_func_space.element.sweight,
        nloc=sf_nd_nb.disp_func_space.element.nloc,
        sngi=sf_nd_nb.disp_func_space.element.sngi,
        sn=sf_nd_nb.disp_func_space.element.sn,
    )
    # change gaussian points order on other side
    nb_aln = sf_nd_nb.disp_func_space.element.gi_align[nb_gi_aln, :]
    snx_nb = snx_nb[..., nb_aln]
    # get faces we want
    sn_nb = sf_nd_nb.disp_func_space.sn[f_inb, ...]  # (batch_in, nloc, sngi)
    snx_nb = snx_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim, nloc, sngi)
    snormal_nb = snormal_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim, sngi)
    # don't forget to change gaussian points order on sn_nb!
    sn_nb = sn_nb[..., nb_aln]
    snormal_nb = snormal_nb[..., nb_aln]

    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = config.eta_e / h

    u_ith = x_k_dict['disp'][E_F_i, ...]  # u^n on this side (th)
    u_inb = x_k_dict['disp'][E_F_inb, ...]  # u^n on the other side (neighbour)

    S = torch.zeros(batch_in, nloc, ndim, nloc, ndim, device=dev, dtype=torch.float64)
    # AA_th = material.calc_AA(nx=snx, u=u_ith, batch_in=batch_in)
    # AA_nb = material.calc_AA(nx=snx_nb, u=u_inb, batch_in=batch_in)
    # use AA_ave instead.
    AA = material.calc_AA_ave(nx=snx, u=u_ith, nx_nb=snx_nb, u_nb=u_inb, batch_in=batch_in)
    # this side
    # [vi nj] {A \nabla u_kl}
    S += torch.einsum(
        'bmg,bjg,bijklg,blng,bg->bmink',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        # AA_th,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        AA,  # use AA_ave, i.e. A({F})
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (-0.5)
    # [ui nj] {A \nabla v_kl}
    S += torch.einsum(
        'bng,bjg,bijklg,blmg,bg->bmkni',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        # AA_th,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        AA,  # use AA_ave, i.e. A({F})
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (-0.5)
    # penalty term
    # \gamma_e [v_i n_j]{A}
    S += torch.einsum(
        'b,bmg,bjg,bijklg,bng,blg,bg->bmink',
        gamma_e,  # (batch_in)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        # 0.5 * (AA_th + AA_nb),  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        AA,  # use AA_ave = A({F}), {F} is average F on the face
        # config.cijkl,
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
    )
    # # use same penalty as in linear elastic
    # # gamma_e [u_i] [v_i]
    # S += torch.einsum(
    #     'bmg,bng,bg,b,ij->bminj',
    #     sn,  # (batch_in, nloc, sngi)
    #     sn,  # (batch_in, nloc, sngi)
    #     sdetwei,  # (batch_in, sngi)
    #     gamma_e,  # (batch_in)
    #     torch.eye(3, device=dev, dtype=torch.float64),
    # )
    # update residual
    r_dict['disp'][E_F_i, ...] -= torch.einsum('bminj,bnj->bmi', S, x_i['disp'][E_F_i, ...])
    # put diagonal of S into diagS
    diagA[E_F_i-batch_start_idx, :, :] += torch.diagonal(S.view(batch_in, nloc*ndim, nloc*ndim),
                                                         dim1=1, dim2=2).view(batch_in, nloc, ndim)
    bdiagA[E_F_i-batch_start_idx, ...] += S

    # other side
    S *= 0
    # [vi nj] {A \nabla u_kl}
    S += torch.einsum(
        'bmg,bjg,bijklg,blng,bg->bmink',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        # AA_nb,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        AA,  # use AA_ave, i.e. A({F})
        snx_nb,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (-0.5)
    # [ui nj] {A \nabla v_kl}
    S += torch.einsum(
        'bng,bjg,bijklg,blmg,bg->bmkni',
        sn_nb,  # (batch_in, nloc, sngi)
        snormal_nb,  # (batch_in, ndim, sngi)
        # AA_th,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        AA,  # use AA_ave, i.e. A({F})
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (-0.5)
    # penalty term
    # \gamma_e [v_i n_j]{A}
    S += torch.einsum(  # FIXME: here it should be S-= ... ? Please make sure.
        'b,bmg,bjg,bijklg,bng,blg,bg->bmink',
        gamma_e,  # (batch_in)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        # 0.5 * (AA_th + AA_nb),  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        AA,  # use AA_ave, i.e. A({F})
        # config.cijkl,
        sn_nb,  # (batch_in, nloc, sngi)
        snormal_nb,  # (batch_in, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
    )
    # # use same penalty as in linear elastic
    # # gamma_e [u_i] [v_i]
    # S -= torch.einsum(
    #     'bmg,bng,bg,b,ij->bminj',
    #     sn,  # (batch_in, nloc, sngi)
    #     sn_nb,  # (batch_in, nloc, sngi)
    #     sdetwei,  # (batch_in, sngi)
    #     gamma_e,  # (batch_in)
    #     torch.eye(3, device=dev, dtype=torch.float64),
    # )
    # update residual
    r_dict['disp'][E_F_i, ...] -= torch.einsum('bminj,bnj->bmi', S, x_i['disp'][E_F_inb, ...])

    return r_in, diagA, bdiagA


def _s_res_fb(
        r_in, f_b, E_F_b,
        x_k, x_i,
        diagA, bdiagA, batch_start_idx
):
    batch_in = f_b.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    if batch_in < 1:  # nothing to do here.
        return r_in
    nloc = sf_nd_nb.disp_func_space.element.nloc
    x_k_dict = volume_mf_st.slicing_x_i(x_k)
    x_i_dict = volume_mf_st.slicing_x_i(x_i)
    r_dict = volume_mf_st.slicing_x_i(r_in)
    # get face shape function
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.disp_func_space.element.snlx,
        x_loc=sf_nd_nb.disp_func_space.x_ref_in[E_F_b],
        sweight=sf_nd_nb.disp_func_space.element.sweight,
        nloc=sf_nd_nb.disp_func_space.element.nloc,
        sngi=sf_nd_nb.disp_func_space.element.sngi,
        sn=sf_nd_nb.disp_func_space.element.sn,
    )
    sn = sf_nd_nb.disp_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_b, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_b, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_b, ...]  # (batch_in, ndim, sngi)
    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = config.eta_e / h
    # get elasticity tensor at face quadrature points
    # (batch_in, ndim, ndim, ndim, ndim, sngi)
    AA = material.calc_AA(nx=snx, u=x_k_dict['disp'][E_F_b, ...], batch_in=batch_in)

    # boundary terms from last 3 terms in eq 60b
    # only one side
    S = torch.zeros(batch_in, nloc, ndim, nloc, ndim,
                    device=dev, dtype=torch.float64)
    # [v_i n_j] {A \nabla u_kl}
    S -= torch.einsum(
        'bmg,bjg,bijklg,blng,bg->bmink',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    )
    # [u_i n_j] {A \nabla v_kl}
    S -= torch.einsum(
        'bng,bjg,bijklg,blmg,bg->bmkni',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    )
    # penalty term
    # \gamma_e [v_i n_j] {A} [u_k n_l]
    S += torch.einsum(
        'b,bmg,bjg,bijklg,bng,blg,bg->bmink',
        gamma_e,  # (batch_in)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        # config.cijkl,
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
    )
    # # use same penalty as in linear elastic
    # # gamma_e [u_i] [v_i]
    # S += torch.einsum(
    #     'bmg,bng,bg,b,ij->bminj',
    #     sn,  # (batch_in, nloc, sngi)
    #     sn,  # (batch_in, nloc, sngi)
    #     sdetwei,  # (batch_in, sngi)
    #     gamma_e,  # (batch_in)
    #     torch.eye(3, device=dev, dtype=torch.float64)
    # )
    # update residual
    r_dict['disp'][E_F_b, ...] -= torch.einsum('bminj,bnj->bmi', S, x_i['disp'][E_F_b, ...])
    # get diagonal
    diagA[E_F_b - batch_start_idx, :, :] += torch.diagonal(S.view(batch_in, nloc * ndim, nloc * ndim),
                                                           dim1=-2, dim2=-1).view(batch_in, nloc, ndim)
    bdiagA[E_F_b - batch_start_idx, ...] += S
    return r_in, diagA, bdiagA


def _s_res_fitf(
        r_in, x_k, x_i,
        f_itf, E_F_itf,
        f_itf_nb, E_F_itf_nb,
):
    # now compute interface contribution and add to r0['disp']
    batch_in = f_itf.shape[0]
    if batch_in < 1:  # nothing to do here.
        return r_in
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)

    r_dict = volume_mf_st.slicing_x_i(r_in)
    x_i_dict = volume_mf_st.slicing_x_i(x_i)
    x_k_dict = volume_mf_st.slicing_x_i(x_k)
    # shape function
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.disp_func_space.element.snlx,
        x_loc=sf_nd_nb.disp_func_space.x_ref_in[E_F_itf],
        sweight=sf_nd_nb.disp_func_space.element.sweight,
        nloc=sf_nd_nb.disp_func_space.element.nloc,
        sngi=sf_nd_nb.disp_func_space.element.sngi,
        sn=sf_nd_nb.disp_func_space.element.sn,
    )
    sn = sf_nd_nb.disp_func_space.element.sn[f_itf, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_itf, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_itf, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_itf, ...]  # (batch_in, ndim, sngi)
    # neighbour face shape function
    # sn_nb = sf_nd_nb.vel_func_space.element.sn[f_itf_nb, ...]
    sq_nb = sf_nd_nb.pre_func_space.element.sn[f_itf_nb, ...]  # (batch_in, nloc, sngi)
    snx_nb, _, _ = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_itf],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi,
        sn=sf_nd_nb.vel_func_space.element.sn,
    )
    snx_nb = snx_nb[dummy_idx, f_itf_nb, ...]  # (batch_in, ndim, nloc, sngi)
    for nb_gi_aln in range(ndim):  # 'ndim' alignnment of GI points on neighbour faces
        idx = sf_nd_nb.vel_func_space.alnmt[E_F_itf * nface + f_itf] == nb_gi_aln
        nb_aln = sf_nd_nb.vel_func_space.element.gi_align[nb_gi_aln, :]
        # sn_nb[idx] = sn_nb[idx][..., nb_aln]
        sq_nb[idx] = sq_nb[idx][..., nb_aln]
        snx_nb[idx] = snx_nb[idx][..., nb_aln]

    d_s_th = x_k_dict['disp'][E_F_itf, ...]  # displacement on solid side (batch_in, u_nloc, ndim)
    u_f_nb = x_i_dict['vel'][E_F_itf_nb, ...]  # (batch_in, u_nloc, ndim)
    p_f_nb = x_i_dict['pre'][E_F_itf_nb, ...]  # (batch_in, p_nloc, ndim)
    Ieye = torch.eye(ndim, device=dev, dtype=torch.float64)
    # Cauchy stress on fluid side  (batch_in, ndim, ndim, sngi)
    sigma = torch.einsum('bing,bnj,ij->bijg', snx_nb, u_f_nb, Ieye) * config.mu \
            - torch.einsum('bng,bn,ij->bijg', sq_nb, p_f_nb, Ieye)
            # + torch.einsum('bing,bnj->bijg', snx_nb, u_f_nb) * config.mu
    # Deformation gradient on face F (batch_in, ndim, ndim, sngi)
    F = torch.einsum('bni,bjng->bgij', d_s_th, snx)
    # determinant on face quadrature (batch_in, sngi)
    detF = torch.linalg.det(F)
    # inv F
    invF = torch.linalg.inv(F)
    # PK1 stress tensor on face quadrature
    PK1 = torch.einsum('bg,bijg,bgIj->biIg', detF, sigma, invF)
    # interface term
    r_dict['disp'][E_F_itf, ...] -= torch.einsum(
        'bijg,bmg,bjg,bg->bmi',
        PK1,  # (batch_in, ndim, ndim, sngi)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
    )
    return r_in


def get_rhs(rhs_in, u, u_bc, f, u_n=0,
            is_get_nonlin_res=True,
            r0_dict=None, x_i_dict=None):
    """
    get right-hand side at the start of each newton step
    Note that in the input lists,
    u_n is field value at last *time* step (if transient).
    """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele_s + nele_f, dtype=int)
    idx_in = torch.zeros(nele, dtype=torch.bool)
    idx_in_f = torch.zeros(nele * nface, dtype=torch.bool, device=dev)

    # change view
    nloc = sf_nd_nb.disp_func_space.element.nloc
    rhs = volume_mf_st.slicing_x_i(rhs_in)
    for u_bci in u_bc:
        u_bci = u_bci.view(nele, nloc, ndim)
    f = f.view(nele, nloc, ndim)

    for i in range(nnn):
        idx_in *= False
        # volume integral
        idx_in[brk_pnt[i]:brk_pnt[i+1]] = True
        rhs = _k_rhs_one_batch(rhs, u, u_n, f, idx_in)
        # surface integral
        idx_in_f *= False
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        rhs = _s_rhs_one_batch(rhs, u, u_bc, idx_in_f,
                               is_get_nonlin_res, r0_dict, x_i_dict)

    return rhs_in, r0_dict


def _k_rhs_one_batch(rhs, u, u_n, f, idx_in):
    batch_in = int(torch.sum(idx_in))
    nloc = sf_nd_nb.disp_func_space.element.nloc
    n = sf_nd_nb.disp_func_space.element.n
    nx, detwei = get_det_nlx(
        nlx=sf_nd_nb.disp_func_space.element.nlx,
        x_loc=sf_nd_nb.disp_func_space.x_ref_in[idx_in],
        weight=sf_nd_nb.disp_func_space.element.weight,
        nloc=nloc,
        ngi=sf_nd_nb.disp_func_space.element.ngi
    )

    # f v
    rhs['disp'][idx_in, ...] += torch.einsum(
        'mg,ng,bg,ij,bnj->bmi',
        n,  # (u_nloc, ngi)
        n,  # (u_nloc, ngi)
        detwei,  # (batch_in, ngi)
        torch.eye(ndim, device=dev, dtype=torch.float64),  # (ndim, ndim)
        f[idx_in, ...],  # (batch_in, u_nloc, ndim)
    )
    if config.isTransient:
        u_n = u_n.view(-1, nloc, ndim)
        rhs['disp'][idx_in, ...] += torch.einsum(
            'mg,ng,bg,ij,bnj->bmi',
            n,
            n,
            detwei,
            torch.eye(ndim, device=dev, dtype=torch.float64),  # (ndim, ndim)
            u_n[idx_in, ...],
        ) * config.rho_s / sf_nd_nb.dt

    # Nxi Nj P
    P = material.calc_P(nx=nx,
                        u=u.view(nele, nloc, ndim)[idx_in, ...],
                        batch_in=batch_in)  # PK1 stress evaluated at current state u
    rhs['disp'][idx_in, ...] -= torch.einsum(
        'bijg,bjmg,bg->bmi',  # i,j is idim and jdim; m, n is mloc and nloc
        P,  # (batch_in, ndim, ndim, ngi)
        # n,  # (nloc, ngi)
        nx,  # (batch_in, ndim, nloc, ngi)
        detwei,  # (batch_in, ngi)
    )
    return rhs


def _s_rhs_one_batch(
        rhs, u, u_bc, idx_in_f,
        is_get_nonlin_res, r0, x_i_dict
):
    nbf = sf_nd_nb.disp_func_space.nbf
    alnmt = sf_nd_nb.disp_func_space.alnmt
    glb_bcface_type = sf_nd_nb.disp_func_space.glb_bcface_type

    nloc = sf_nd_nb.disp_func_space.element.nloc

    # separate nbf to get internal face list and boundary face list
    F_i = torch.where(torch.logical_and(glb_bcface_type < 0,
                                        idx_in_f))[0]  # interior face of solid subdomain
    F_inb = nbf[F_i]  # neighbour list of interior face
    F_inb = F_inb.type(torch.int64)
    F_b_d = torch.where(torch.logical_and(
        sf_nd_nb.disp_func_space.glb_bcface_type == 2,
        idx_in_f))[0]  # dirichlet boundary face
    F_b_n = torch.where(torch.logical_and(
        sf_nd_nb.disp_func_space.glb_bcface_type == 3,
        idx_in_f))[0]  # neumann boundary face
    F_itf = torch.where(torch.logical_and(
        sf_nd_nb.disp_func_space.glb_bcface_type == 5,
        idx_in_f))[0]  # boundary face  (interface)
    F_itf_nb = nbf[F_itf].type(torch.int64)  # neighbour list of interface face

    # create two lists of which element f_i / f_b is in
    E_F_i = torch.floor_divide(F_i, nface)
    E_F_inb = torch.floor_divide(F_inb, nface)
    E_F_b_d = torch.floor_divide(F_b_d, nface)
    E_F_b_n = torch.floor_divide(F_b_n, nface)
    E_F_itf = torch.floor_divide(F_itf, nface)
    E_F_itf_nb = torch.floor_divide(F_itf_nb, nface)

    # local face number
    f_b_d = torch.remainder(F_b_d, nface)
    f_b_n = torch.remainder(F_b_n, nface)
    f_i = torch.remainder(F_i, nface)
    f_inb = torch.remainder(F_inb, nface)
    f_itf = torch.remainder(F_itf, nface)
    f_itf_nb = torch.remainder(F_itf_nb, nface)

    # interior face term
    for iface in range(nface):
        for nb_gi_aln in range(nface-1):
            idx_iface = (f_i == iface) & (alnmt[F_i] == nb_gi_aln)
            if idx_iface.sum() < 1:
                # there is nothing to do here, go on
                continue
            rhs = _s_rhs_fi(
                rhs, f_i[idx_iface], E_F_i[idx_iface],
                f_inb[idx_iface], E_F_inb[idx_iface],
                u,
                nb_gi_aln)
            r0['disp'][E_F_b_d[idx_iface], ...] += rhs['disp'][E_F_b_d[idx_iface], ...]

    # boundary term
    # if ndim == 3:  # in 3D one element might have multiple boundary faces
    if True:  # in 2D we don't need to split nface cause no element has >1 boundary faces. but we'll leave it here.
        for iface in range(nface):
            idx_iface = f_b_d == iface
            rhs = _s_rhs_fb(rhs, f_b_d[idx_iface], E_F_b_d[idx_iface],
                            u, u_bc[2])
            r0['disp'][E_F_b_d[idx_iface], ...] += rhs['disp'][E_F_b_d[idx_iface], ...]
            # neumann bc
            idx_iface = f_b_n == iface
            if torch.sum(idx_iface) == 0:
                continue
            rhs = _s_rhs_fb_neumann(rhs, f_b_n[idx_iface], E_F_b_n[idx_iface], u_bc[3])
            r0['disp'][E_F_b_d[idx_iface], ...] += rhs['disp'][E_F_b_d[idx_iface], ...]
            # interface
            if not is_get_nonlin_res:
                continue
            idx_iface = f_itf == iface
            if torch.sum(idx_iface) == 0:
                continue
            r0 = _s_rhs_f_itf(r0, rhs, f_itf[idx_iface], E_F_itf[idx_iface],
                              f_itf_nb[idx_iface], E_F_itf_nb[idx_iface],
                              x_i_dict)
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
    nloc = sf_nd_nb.disp_func_space.element.nloc
    # shape function on this side
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.disp_func_space.element.snlx,
        x_loc=sf_nd_nb.disp_func_space.x_ref_in[E_F_i],
        sweight=sf_nd_nb.disp_func_space.element.sweight,
        nloc=sf_nd_nb.disp_func_space.element.nloc,
        sngi=sf_nd_nb.disp_func_space.element.sngi,
        sn=sf_nd_nb.disp_func_space.element.sn,
    )
    sn = sf_nd_nb.disp_func_space.element.sn[f_i, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_i, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_i, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_i, ...]  # (batch_in, ndim, sngi)

    # shape function on the other side
    snx_nb, _, snormal_nb = sdet_snlx(
        snlx=sf_nd_nb.disp_func_space.element.snlx,
        x_loc=sf_nd_nb.disp_func_space.x_ref_in[E_F_inb],
        sweight=sf_nd_nb.disp_func_space.element.sweight,
        nloc=sf_nd_nb.disp_func_space.element.nloc,
        sngi=sf_nd_nb.disp_func_space.element.sngi,
        sn=sf_nd_nb.disp_func_space.element.sn,
    )
    # change gaussian points order
    nb_aln = sf_nd_nb.gi_align[nb_gi_aln, :]
    snx_nb = snx_nb[..., nb_aln]
    # fetch faces we want
    sn_nb = sf_nd_nb.disp_func_space.element.sn[f_inb, ...]  # (batch_in, nloc, sngi)
    snx_nb = snx_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim, nloc, sngi)
    snormal_nb = snormal_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim, sngi)
    # don't forget to change gaussian points order on sn_nb!
    sn_nb = sn_nb[..., nb_aln]
    snormal_nb = snormal_nb[..., nb_aln]

    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = config.eta_e / h
    u_i = u[E_F_i, ...]  # u^n on this side
    u_inb = u[E_F_inb, ...]  # u^n on the other side

    # [vi nj]{P^n_kl} term
    # P = material.calc_P(nx=snx, u=u_i, batch_in=batch_in)
    # P_nb = material.calc_P(nx=snx_nb, u=u_inb, batch_in=batch_in)
    # P *= 0.5
    # P_nb *= 0.5
    # P += P_nb  # this is {P^n} = 1/2 (P^1 + P^2)  average on both sides
    # new average P = P({F})
    P = material.calc_P_ave(nx=snx, u=u_i, nx_nb=snx_nb, u_nb=u_inb, batch_in=batch_in)
    # this side + other side
    rhs['disp'][E_F_i, ...] += torch.einsum(
        'bmg,bjg,bijg,bg->bmi',  # i,j is idim/jdim; m, n is mloc/nloc
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        # sn,  # (batch_in, nloc, sngi)
        P,  # (batch_in, ndim, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
    )
    # del P, P_nb
    del P

    # [ui nj] {A (\nabla v)_kl] term
    # AA = material.calc_AA(nx=snx, u=u_i, batch_in=batch_in)
    # use new AA = A({F})
    AA = material.calc_AA_ave(nx=snx, u=u_i, nx_nb=snx_nb, u_nb=u_inb, batch_in=batch_in)
    # this side + other side
    rhs['disp'][E_F_i, ...] += torch.einsum(
        'bnijg,bijklg,blmg,bg->bmk',  # i/j : idim/jdim; m/n: mloc/nloc
        (
            torch.einsum('bni,bng,bjg->bnijg', u_i, sn, snormal)
            + torch.einsum('bni,bng,bjg->bnijg', u_inb, sn_nb, snormal_nb)
        ),  # (batch_in, nloc, ndim, ndim, sngi)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei  # (batch_in, sngi)
    ) * 0.5

    # penalty term
    # \gamma_e [vi nj] A [uk nl]
    # # this A is 1/2(A_this + A_nb)
    # AA += material.calc_AA(nx=snx_nb, u=u_inb, batch_in=batch_in)
    # AA *= 0.5
    rhs['disp'][E_F_i, ...] -= torch.einsum(
        'b,bmg,bj,bijklg,bnklg,bg->bmi',
        gamma_e,  # (batch_in)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        # config.cijkl,
        (
            torch.einsum('bnk,bng,blg->bnklg', u_i, sn, snormal)
            + torch.einsum('bnk,bng,blg->bnklg', u_inb, sn_nb, snormal_nb)
        ),  # (batch_in, nloc, ndim, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
    )
    # # use same penalty as in linear elastic
    # # gamma_e [u_i] [v_i]
    # rhs[E_F_i, ...] -= torch.einsum(
    #     'bmg,bg,b,bnig->bmi',
    #     sn,  # (batch_in, nloc, sngi)
    #     sdetwei,  # (batch_in, sngi)
    #     gamma_e,  # (batch_in)
    #     (
    #         torch.einsum('bng,bni->bnig', sn, u_i)
    #         - torch.einsum('bng,bni->bnig', sn_nb, u_inb)
    #     )  # (batch_in, nloc, ndim, sngi)
    # )
    return rhs


def _s_rhs_fb(rhs, f_b, E_F_b, u, u_bc):
    """
    add surface integral contribution to equation right-hand side
    this is only for dirichlet boundary in solid subdomain
    """
    batch_in = f_b.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    if batch_in < 1:  # nothing to do here.
        return rhs
    nloc = sf_nd_nb.disp_func_space.element.nloc
    # get face shape function
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.disp_func_space.element.snlx,
        x_loc=sf_nd_nb.disp_func_space.x_ref_in[E_F_b],
        sweight=sf_nd_nb.disp_func_space.element.sweight,
        nloc=sf_nd_nb.disp_func_space.element.nloc,
        sngi=sf_nd_nb.disp_func_space.element.sngi,
        sn=sf_nd_nb.disp_func_space.element.sn,
    )
    sn = sf_nd_nb.disp_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_b, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_b, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_b, ...]  # (batch_in, ndim, sngi)
    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = config.eta_e / h
    # get elasticity tensor at face quadrature points
    # (batch_in, ndim, ndim, ndim, ndim, sngi)
    AA = material.calc_AA(nx=snx, u=u[E_F_b, ...], batch_in=batch_in)
    rhs = rhs.view(nele, nloc, ndim)
    # u_Di nj A \delta v_kl
    rhs['disp'][E_F_b, ...] -= torch.einsum(
        'bni,bng,bjg,bijklg,blmg,bg->bmk',  # could easily by wrong...
        u_bc[E_F_b, ...],  # (batch_in, nloc, ndim)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch, sngi)
    )
    # penalty term
    # gamma_e v_i n_j A u_Dk n_l
    rhs['disp'][E_F_b, ...] += torch.einsum(
        'b,bmg,bjg,bijklg,bng,blg,bnk,bg->bmi',  # again could easily be wrong...
        gamma_e,  # (batch_in)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        # config.cijkl,
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        u_bc[E_F_b, ...],  # (batch_in, nloc, ndim)  # TODO: could be u_bc - u_i or sth like that to avoid 2 einsums
        sdetwei,  # (batch_in, sngi
    )
    # # use same penalty as in linear elastic
    # # gamma_e [u_Di] [v_i]
    # rhs[E_F_b, ...] += torch.einsum(
    #     'bmg,bng,bg,b,bni->bmi',
    #     sn,  # (batch_in, nloc, sngi)
    #     sn,  # (batch_in, nloc, sngi)
    #     sdetwei,  # (batch_in, sngi)
    #     gamma_e,  # (batch_in)
    #     u_bc[E_F_b, ...] - u[E_F_b, ...],  # (batch_in, nloc, ndim)
    # )
    # add boundary contribution from lhs. (last 3 terms in eq 60c)
    # u_i n_j A \nabla v_kl
    rhs['disp'][E_F_b, ...] += torch.einsum(
        'bni,bng,bjg,bijklg,blmg,bg->bmk',  # could easily by wrong...
        u[E_F_b, ...],  # (batch_in, nloc, ndim)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch, sngi)
    )
    # penalty term
    # \gamma_e v_i n_j A u_k n_l
    rhs['disp'][E_F_b, ...] -= torch.einsum(
        'b,bmg,bjg,bijklg,bng,blg,bnk,bg->bmi',  # again could easily be wrong...
        gamma_e,  # (batch_in)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        # config.cijkl,
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        u[E_F_b, ...],  # (batch_in, nloc, ndim)
        sdetwei,  # (batch_in, sngi
    )
    del AA  # no longer need
    P = material.calc_P(nx=snx, u=u[E_F_b, ...], batch_in=batch_in)
    # [v_i n_j] {P_ij}
    rhs['disp'][E_F_b, ...] += torch.einsum(
        'bmg,bjg,bijg,bg->bmi',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        P,  # (batch_in, ndim, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
    )
    return rhs


def _s_rhs_fb_neumann(
        rhs, f_b, E_F_b,
        u_bc
):
    """
    put neumann boundary to solid subdomain rhs
    """
    batch_in = f_b.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    if batch_in < 1:  # nothing to do here.
        return rhs

    # shape function
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.disp_func_space.element.snlx,
        x_loc=sf_nd_nb.disp_func_space.x_ref_in[E_F_b],
        sweight=sf_nd_nb.disp_func_space.element.sweight,
        nloc=sf_nd_nb.disp_func_space.element.nloc,
        sngi=sf_nd_nb.disp_func_space.element.sngi,
        sn=sf_nd_nb.disp_func_space.element.sn,
    )
    sn = sf_nd_nb.disp_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    sq = sf_nd_nb.pre_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_b, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_b, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_b, ...]  # (batch_in, ndim, sngi)

    # print('gamma_e', gamma_e)
    u_bc_th = u_bc[E_F_b, ...]

    # 3. TODO: Neumann BC
    rhs['disp'][E_F_b, ...] += torch.einsum(
        'bmg,bng,bg,bni->bmi',
        sn,  # (batch_in, nloc, sngi)
        sn,  # (batch_in, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        u_bc_th,  # (batch_in, u_nloc, ndim)
    )
    return rhs


def _s_rhs_f_itf(
        r0, rhs, f_itf, E_F_itf,
        f_itf_nb, E_F_itf_nb,
        x_i_dict
):
    """
    given rhs of solid subdomain, subtracting interface contribution
    to get non-linear residual in solid subdomain

    input:
    r0: residual (already set to zero in solid subdomain)
    x_i_dict: a dict of current field values (vel, pre, disp)
    x_rhs: right-hand side at this non-linear step
    """

    # now compute interface contribution and add to r0['disp']
    batch_in = f_itf.shape[0]
    if batch_in < 1:  # nothing to do here.
        return r0
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)

    # shape function
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.disp_func_space.element.snlx,
        x_loc=sf_nd_nb.disp_func_space.x_ref_in[E_F_itf],
        sweight=sf_nd_nb.disp_func_space.element.sweight,
        nloc=sf_nd_nb.disp_func_space.element.nloc,
        sngi=sf_nd_nb.disp_func_space.element.sngi,
        sn=sf_nd_nb.disp_func_space.element.sn,
    )
    sn = sf_nd_nb.disp_func_space.element.sn[f_itf, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_itf, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_itf, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_itf, ...]  # (batch_in, ndim, sngi)
    # neighbour face shape function
    # sn_nb = sf_nd_nb.vel_func_space.element.sn[f_itf_nb, ...]
    sq_nb = sf_nd_nb.pre_func_space.element.sn[f_itf_nb, ...]  # (batch_in, nloc, sngi)
    snx_nb, _, _ = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_itf],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi,
        sn=sf_nd_nb.vel_func_space.element.sn,
    )
    snx_nb = snx_nb[dummy_idx, f_itf_nb, ...]  # (batch_in, ndim, nloc, sngi)
    for nb_gi_aln in range(ndim):  # 'ndim' alignnment of GI points on neighbour faces
        idx = sf_nd_nb.vel_func_space.alnmt[E_F_itf * nface + f_itf] == nb_gi_aln
        nb_aln = sf_nd_nb.vel_func_space.element.gi_align[nb_gi_aln, :]
        # sn_nb[idx] = sn_nb[idx][..., nb_aln]
        sq_nb[idx] = sq_nb[idx][..., nb_aln]
        snx_nb[idx] = snx_nb[idx][..., nb_aln]

    d_s_th = x_i_dict['disp'][E_F_itf, ...]  # displacement on solid side (batch_in, u_nloc, ndim)
    u_f_nb = x_i_dict['vel'][E_F_itf_nb, ...]  # (batch_in, u_nloc, ndim)
    p_f_nb = x_i_dict['pre'][E_F_itf_nb, ...]  # (batch_in, p_nloc, ndim)
    Ieye = torch.eye(ndim, device=dev, dtype=torch.float64)
    # Cauchy stress on fluid side  (batch_in, ndim, ndim, sngi)
    sigma = torch.einsum('bing,bnj,ij->bijg', snx_nb, u_f_nb, Ieye) * config.mu \
            - torch.einsum('bng,bn,ij->bijg', sq_nb, p_f_nb, Ieye)
            # + torch.einsum('bing,bnj->bijg', snx_nb, u_f_nb) * config.mu
    # Deformation gradient on face F (batch_in, ndim, ndim, sngi)
    F = torch.einsum('bni,bjng->bgij', d_s_th, snx)
    # determinant on face quadrature (batch_in, sngi)
    detF = torch.linalg.det(F)
    # inv F
    invF = torch.linalg.inv(F)
    # PK1 stress tensor on face quadrature
    PK1 = torch.einsum('bg,bijg,bgIj->biIg', detF, sigma, invF)
    # interface term
    r0['disp'][E_F_itf, ...] -= torch.einsum(
        'bijg,bmg,bjg,bg->bmi',
        PK1,  # (batch_in, ndim, ndim, sngi)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
    )

    return r0


def get_RAR_and_sfc_data_Sp(
        x_k
):
    """
    get RAR for solid block to do MG.
    """
    I_fc = sf_nd_nb.sparse_s.I_fc
    I_cf = sf_nd_nb.sparse_s.I_cf
    whichc = sf_nd_nb.sparse_s.whichc
    ncolor = sf_nd_nb.sparse_s.ncolor
    fina = sf_nd_nb.sparse_s.fina
    cola = sf_nd_nb.sparse_s.cola
    ncola = sf_nd_nb.sparse_s.ncola
    cg_nonods = sf_nd_nb.sparse_s.cg_nonods

    # prepare for MG on SFC-coarse grids
    RARvalues = calc_RAR_mf_color(
        I_fc, I_cf,
        whichc, ncolor,
        fina, cola, ncola,
        x_k
    )
    from scipy.sparse import bsr_matrix

    if not config.is_sfc:
        RAR = bsr_matrix((RARvalues.cpu().numpy(), cola, fina), shape=(ndim * cg_nonods, ndim * cg_nonods))
        sf_nd_nb.set_data(RARmat=RAR.tocsr())
    # np.savetxt('RAR.txt', RAR.toarray(), delimiter=',')
    RARvalues = torch.permute(RARvalues, (1, 2, 0)).contiguous()  # (ndim,ndim,ncola)
    # get SFC, coarse grid and operators on coarse grid. Store them to save computational time?
    space_filling_curve_numbering, variables_sfc, nlevel, nodes_per_level = \
        mg_le.mg_on_P1CG_prep(fina, cola, RARvalues)
    sf_nd_nb.sfc_data_S.set_data(
        space_filling_curve_numbering=space_filling_curve_numbering,
        variables_sfc=variables_sfc,
        nlevel=nlevel,
        nodes_per_level=nodes_per_level
    )


# def all kinds of preconditioning operations...
