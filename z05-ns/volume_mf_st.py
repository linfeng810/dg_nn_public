#!/usr/bin/env python3

"""
all integrations for navier-stokes
matric-free implementation
"""

import torch
from torch import Tensor
import scipy as sp
import numpy as np
from types import NoneType
from tqdm import tqdm
import config
import ns_assemble
import petrov_galerkin
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
import pressure_matrix
import multigrid_linearelastic

dev = config.dev
nele = config.nele
nele_f = config.nele_f
nele_s = config.nele_s
mesh = config.mesh
ndim = config.ndim
nface = config.ndim+1


def calc_RAR_mf_color(
        I_fc, I_cf,
        whichc, ncolor,
        fina, cola, ncola,
        include_adv, x_k, u_bc
):
    """
    get operator on P1CG grid, i.e. RAR
    where R is prolongator/restrictor
    via coloring method.

    """
    import time
    start_time = time.time()
    cg_nonods = sf_nd_nb.sparse_f.cg_nonods
    p1dg_nonods = sf_nd_nb.sparse_f.p1dg_nonods
    nonods = sf_nd_nb.vel_func_space.nonods
    total_no_dofs = nonods * ndim * 2 + sf_nd_nb.pre_func_space.nonods
    value = torch.zeros(ncola, ndim, ndim, device=dev, dtype=torch.float64)  # NNZ entry values
    dummy = torch.zeros(total_no_dofs, device=dev, dtype=torch.float64)  # dummy variable of same length as PnDG
    Rm = torch.zeros_like(dummy, device=dev, dtype=torch.float64)
    ARm = torch.zeros_like(dummy, device=dev, dtype=torch.float64)
    dummy_dict = slicing_x_i(dummy)
    Rm_dict = slicing_x_i(Rm)
    ARm_dict = slicing_x_i(ARm)
    RARm = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
    mask = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)  # color vec
    for color in tqdm(range(1, ncolor + 1), disable=config.disabletqdm):
        # print('color: ', color)
        for jdim in range(ndim):
            mask *= 0
            mask[:, jdim] += torch.tensor((whichc == color),
                                          device=dev,
                                          dtype=torch.float64)  # 1 if true; 0 if false
            Rm *= 0
            for idim in range(ndim):
                # Rm[:, idim] += torch.mv(I_fc, mask[:, idim].view(-1))  # (p1dg_nonods, ndim)
                Rm_dict['vel'][0:nele_f, :, idim] += \
                    (mg_le.vel_p1dg_to_pndg_prolongator(
                        torch.mv(I_fc, mask[:, idim].view(-1))
                )).view(nele_f, -1)  # (p3dg_nonods, ndim)
            ARm *= 0
            ARm = get_residual_only(r0=ARm,
                                    x_i=Rm,
                                    x_rhs=dummy,
                                    include_adv=include_adv,
                                    x_k=x_k,
                                    u_bc=u_bc,
                                    a00=True, a01=False, a10=False, a11=False,
                                    use_fict_dt_in_vel_precond=sf_nd_nb.use_fict_dt_in_vel_precond,
                                    include_itf=False)
            ARm *= -1.  # (p3dg_nonods, ndim)
            # RARm = multi_grid.p3dg_to_p1dg_restrictor(ARm)  # (p1dg_nonods, )
            RARm *= 0
            for idim in range(ndim):
                RARm[:, idim] += torch.mv(
                    I_cf,
                    mg_le.vel_pndg_to_p1dg_restrictor(ARm_dict['vel'][0:nele_f, :, idim])
                )  # (cg_nonods, ndim)
            # add to value
            for i in range(RARm.shape[0]):
                count = np.arange(fina[i], fina[i+1])
                j = cola[fina[i]:fina[i+1]]
                for idim in range(ndim):
                    # for count in range(fina[i], fina[i + 1]):
                    #     j = cola[count]
                    #     value[count, idim, jdim] += RARm[i, idim] * mask[j, jdim]
                    value[count, idim, jdim] += RARm[i, idim] * mask[j, jdim]
        # print('finishing (another) one color, time comsumed: ', time.time() - start_time)
    return value


def get_residual_and_smooth_once(
        r0_in, x_i_in, x_rhs,
        include_adv=False, x_k=None, u_bc=None,
        a00=True, a01=False, a10=False, a11=False,
        use_fict_dt_in_vel_precond=False,
        include_itf=True
):
    """update residual, then do one (block-) Jacobi smooth.
    Block matrix structure:
        [ K    G    I_uS]
        [ G^T  S    I_pS]
        K is velocity block, G is gradient,
        G^T is divergence, S is block for pressure penalty term.
        I's are interface block. Solid's contribution to fluid at interface.
    """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele_f, dtype=int)
    # retrive func space / element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    r0_dict = slicing_x_i(r0_in)
    x_rhs_dict = slicing_x_i(x_rhs)

    # from now on we will assume always include advection term...
    # we either have real advection velocity or advection due to mesh velocity
    x_k_dict = slicing_x_i(x_k)
    if include_adv:
        u_bc = u_bc.view(nele, u_nloc, ndim)  # only diri bc is passed in here.

    # add precalculated rhs to residual
    if type(x_rhs) is int:
        r0_in += x_rhs  # FIXME: what does this mean...?
    else:
        r0_dict['vel'] += x_rhs_dict['vel']
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
        r0_in, diagK, bdiagK, diagS = _k_res_one_batch(
            r0_in, x_i_in,
            diagK, bdiagK, diagS,
            idx_in,
            include_adv, x_k,
            a00, a01, a10, a11,
            use_fict_dt_in_vel_precond,
        )
        # surface integral
        idx_in_f = torch.zeros(nele * nface, dtype=bool, device=dev)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0_in, diagK, bdiagK, diagS = _s_res_one_batch(
            r0_in, x_i_in,
            diagK, bdiagK, diagS,
            idx_in_f, brk_pnt[i],
            include_adv, x_k, u_bc,
            a00, a01, a10, a11,
            include_itf
        )
        # smooth once
        if config.blk_solver == 'direct':
            bdiagK = torch.inverse(bdiagK.view(batch_in, u_nloc * ndim, u_nloc * ndim))
            x_i_dict = slicing_x_i(x_i_in)

            x_i_dict['vel'][idx_in, :] += config.jac_wei * torch.einsum(
                '...ij,...j->...i',
                bdiagK,
                r0_dict['vel'].view(nele, u_nloc * ndim)[idx_in, :]
                ).view(batch_in, u_nloc, ndim)
            if not a01:
                return r0_in, x_i_in
            # x_i[1] = x_i[1].view(nele, p_nloc)
            # TODO: actually I think we will never reach here.
            diagS = torch.inverse(diagS.view(batch_in, p_nloc, p_nloc))
            x_i_dict['pre'][idx_in, :] += config.jac_wei * torch.einsum(
                'bij,bj->bi',
                diagS,
                r0_dict['pre'].view(nele, p_nloc)[idx_in, :])
        if config.blk_solver == 'jacobi':
            raise NotImplemented('Jacobi iteration for block diagonal matrix is not implemented!')
    # r0[0] = r0[0].view(nele * u_nloc, ndim)
    # r0[1] = r0[1].view(-1)
    # x_i[0] = x_i[0].view(nele * u_nloc, ndim)
    # x_i[1] = x_i[1].view(-1)
    return r0_in, x_i_in


def get_residual_only(r0, x_i, x_rhs,
                      include_adv=False, x_k=None, u_bc=None,
                      a00=True, a01=True, a10=True, a11=False,
                      use_fict_dt_in_vel_precond=False,
                      include_itf=True):
    """update residual,
        Block matrix structure:
        [ K    G    I_uS]
        [ G^T  S    I_pS]
        K is velocity block, G is gradient,
        G^T is divergence, S is block for pressure penalty term.
        I's are interface block. Solid's contribution to fluid at interface.

        use a00, a01, a10, a11 to control which block(s) is activated
        during residual update. for example, we may only want to get
        velocity block a00, then set a01, a10, a11 be False.
        For another example, we may only want to use G, then
        set a00, a10, a11 be false.

        r0, x_i, x_rhs, x_k are vector that contain all dofs (vel, pre, disp)
        by slicing_xi them we get a dict of the fields

        include_itf controls if we want to include the interface terms (or
        only include the interface terms when block preconditioning)

        u_bc is only *FLUID DIRICHLET BC* not the full list of u_bc
        """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele_f, dtype=int)
    # retrive func space / element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    r0_dict = slicing_x_i(r0)

    # from now on we will assume always include advection term...
    # we either have real advection velocity or advection due to mesh velocity
    # x_k_dict = slicing_x_i(x_k)
    if include_adv:
        u_bc = u_bc.view(nele, u_nloc, ndim)  # only diri bc is passed in here.

    # add precalculated rhs to residual
    if type(x_rhs) is int:
        r0 += x_rhs  # FIXME: what does this mean...?
    else:
        x_rhs_dict = slicing_x_i(x_rhs)
        r0_dict['vel'] += x_rhs_dict['vel']
        r0_dict['pre'] += x_rhs_dict['pre']
    for i in range(nnn):
        # volume integral
        idx_in = torch.zeros(nele, device=dev, dtype=torch.bool)  # element indices in this batch
        idx_in[brk_pnt[i]:brk_pnt[i + 1]] = True
        batch_in = int(torch.sum(idx_in))
        # dummy diagA and bdiagA
        diagK = torch.zeros(batch_in, u_nloc, ndim, device=dev, dtype=torch.float64)
        bdiagK = torch.zeros(batch_in, u_nloc, ndim, u_nloc, ndim, device=dev, dtype=torch.float64)
        # TODO: we might omit outputting diagS since we're only doing jacobi iter on block K
        diagS = torch.zeros(batch_in, p_nloc, p_nloc, device=dev, dtype=torch.float64)
        r0, diagK, bdiagK, diagS = _k_res_one_batch(r0, x_i,
                                                    diagK, bdiagK, diagS,
                                                    idx_in,
                                                    include_adv, x_k,
                                                    a00, a01, a10, a11,
                                                    use_fict_dt_in_vel_precond,)
        # surface integral
        idx_in_f = torch.zeros(nele * nface, dtype=torch.bool, device=dev)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0, diagK, bdiagK, diagS = _s_res_one_batch(
            r0, x_i,
            diagK, bdiagK, diagS,
            idx_in_f, brk_pnt[i],
            include_adv, x_k, u_bc,
            a00, a01, a10, a11,
            include_itf
        )
    if a00 and a10 and config.hasNullSpace:
        x_i_dict = slicing_x_i(x_i)
        r0_dict['pre'][1] -= x_i_dict['pre'][1]
        # this effectively add 1 to the last diagonal entry in pressure block to remove null space.
        # print(r0.shape, x_i.shape)
    # r0[0] = r0[0].view(nele * u_nloc, ndim)
    # r0[1] = r0[1].view(-1)
    return r0


def _k_res_one_batch(
        r0, x_i,
        diagK, bdiagK, diagS,
        idx_in,
        include_adv,
        u_n,
        a00, a01, a10, a11,
        use_fict_dt_in_vel_precond=False,
):
    """this contains volume integral part of the residual update
    let velocity shape function be N, pressure be Q
    i.e.
    Nx_i Nx_j in K
    Nx_i Q_j in G
    Q_i Nx_j in G^T
    ~mu/|e|^2 in S~ NO S in this block-diagonal preconditioning
    """
    if not (a00 or a01 or a10 or a11):
        # nothing to do here.
        return r0, diagK, bdiagK, diagS
    batch_in = diagK.shape[0]
    if batch_in == 0:
        return r0, diagK, bdiagK, diagS
    include_p = a01 or a10
    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    # if a00:
    #     r0_u, r0_p = slicing_x_i(r0, include_p)
    #     x_i_u, x_i_p = slicing_x_i(x_i, include_p)
    # elif not a00 and a01:
    #     r0_u = r0.view(batch_in, u_nloc, ndim)
    #     x_i_p = x_i.view(batch_in, p_nloc)
    # elif not a00 and a10:
    #     r0_p = r0.view(batch_in, p_nloc)
    #     x_i_u = x_i.view(batch_in, u_nloc, ndim)
    r0_dict = slicing_x_i(r0)
    x_i_dict = slicing_x_i(x_i)

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
    # if include_p:
    #     _, qdetwei = get_det_nlx(
    #         nlx=sf_nd_nb.pre_func_space.element.nlx,
    #         x_loc=sf_nd_nb.pre_func_space.x_ref_in[idx_in],
    #         weight=sf_nd_nb.pre_func_space.element.weight,
    #         nloc=p_nloc,
    #         ngi=sf_nd_nb.pre_func_space.element.ngi
    #     )

    # local K
    K = torch.zeros(batch_in, u_nloc, ndim, u_nloc, ndim, device=dev, dtype=torch.float64)
    if a00:
        # Nx_i Nx_j
        K += torch.einsum('bimg,bing,bg,jk->bmjnk', nx, nx, ndetwei,
                          torch.eye(ndim, device=dev, dtype=torch.float64)
                          ) \
            * config.mu_f
        if sf_nd_nb.isTransient or use_fict_dt_in_vel_precond:
            # ni nj
            for idim in range(ndim):
                K[:, :, idim, :, idim] += torch.einsum('mg,ng,bg->bmn', n, n, ndetwei) \
                                          * config.rho_f / sf_nd_nb.dt * sf_nd_nb.bdfscm.gamma
        # elif use_fict_dt_in_vel_precond:
        #     for idim in range(ndim):
        #         K[:, :, idim, :, idim] += torch.einsum('mg,ng,bg->bmn', n, n, ndetwei) * sf_nd_nb.fict_mass_coeff
        if include_adv:
            x_k_dict = slicing_x_i(u_n)  # non-linear vel/pre/disp
            K += torch.einsum(
                'lg,bli,bing,mg,bg,jk->bmjnk',
                n,
                x_k_dict['vel'][idx_in, ...] - sf_nd_nb.u_m[idx_in, ...],  # u_k - u_m
                nx,
                n,
                ndetwei,
                torch.eye(ndim, device=dev, dtype=torch.float64)
            ) * config.rho_f
            if sf_nd_nb.isPetrovGalerkin:  # TODO: this is not updated for moving mesh
                # get tau for petrov galerkin, but only update tau at the start of one nonlinear step
                tau = petrov_galerkin.get_pg_tau(
                        u=u_n[idx_in, ...],
                        w=u_n[idx_in, ...],
                        v=n,
                        vx=nx,
                        cell_volume=sf_nd_nb.vel_func_space.cell_volume[idx_in],
                        batch_in=batch_in
                    )  # (batch_in, ndim, ngi)
                K += torch.einsum(
                    'bjng,big,bjmg,bg,ik->bmink',
                    nx,  # (batch_in, ndim, u_nloc, ngi)
                    tau,  # (batch_in, ndim, ngi)
                    nx,  # (batch_in, ndim, u_nloc, ngi)
                    ndetwei,  # (batch_in, ngi)
                    torch.eye(ndim, device=dev, dtype=torch.float64)
                )
            elif config.isGradDivStab:
                # get tau for grad-div stabilisation
                u_ave = sf_nd_nb.u_ave[idx_in, :]
                u_ave_abs = torch.sqrt(torch.sum(torch.square(u_ave), dim=-1))  # (batch_in)
                h = torch.pow(sf_nd_nb.vel_func_space.cell_volume[idx_in], 1./config.ndim)  # characteristic element len
                tau = config.zeta * u_ave_abs * h / (sf_nd_nb.vel_func_space.element.ele_order + 1)  # (batch_in)
                K += torch.einsum(
                    'bing,b,bjmg,bg->bminj',
                    nx,  # (batch_in, ndim, u_nloc, ngi)
                    tau,  # (batch_in)
                    nx,  # (batch_in, ndim, u_nloc, ngi)
                    ndetwei,  # (batch_in, ngi)
                    # torch.eye(ndim, device=dev, dtype=torch.float64)
                )
        # update residual of velocity block K
        r0_dict['vel'][idx_in, ...] -= torch.einsum('bminj,bnj->bmi', K, x_i_dict['vel'][idx_in, ...])
        # get diagonal of velocity block K
        diagK += torch.diagonal(K.view(batch_in, u_nloc * ndim, u_nloc * ndim)
                                , dim1=1, dim2=2).view(batch_in, u_nloc, ndim)
        bdiagK[idx_in[0:nele_f], ...] += K

    if include_p:
        # local G
        G = torch.zeros(batch_in, u_nloc, ndim, p_nloc, device=dev, dtype=torch.float64)
        # Nx_i Q_j
        G += torch.einsum('bimg,ng,bg->bmin', nx, q, ndetwei) * (-1.)
        if a01:
            # update velocity residual of pressure gradient G * p
            r0_dict['vel'][idx_in, ...] -= torch.einsum('bmin,bn->bmi', G, x_i_dict['pre'][idx_in, ...])
        if a10:
            # update pressure residual of velocity divergence G^T * u
            r0_dict['pre'][idx_in, ...] -= torch.einsum('bmjn,bmj->bn', G, x_i_dict['vel'][idx_in, ...])

    return r0, diagK, bdiagK, diagS


def _s_res_one_batch(
        r0, x_i,
        diagK, bdiagK, diagS,
        idx_in_f,
        batch_start_idx,
        include_adv, u_n, u_bc,
        a00, a01, a10, a11,
        include_itf=True,
):
    # get essential data
    nbf = sf_nd_nb.vel_func_space.nbf
    alnmt = sf_nd_nb.vel_func_space.alnmt
    glb_bcface_type = sf_nd_nb.vel_func_space.glb_bcface_type

    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    # separate nbf to get internal face list and boundary face list
    F_i = torch.where(torch.logical_and(glb_bcface_type < 0,
                                        idx_in_f))[0]  # interior face of fluid subdomain
    F_inb = nbf[F_i]  # neighbour list of interior face
    F_inb = F_inb.type(torch.int64)
    F_b = torch.where(torch.logical_and(
        torch.logical_or(sf_nd_nb.vel_func_space.glb_bcface_type == 0,
                         sf_nd_nb.vel_func_space.glb_bcface_type == 4),
        idx_in_f))[0]  # boundary face (dirichlet and interface this side)
    F_itf = torch.where(torch.logical_and(
        sf_nd_nb.vel_func_space.glb_bcface_type == 4,
        idx_in_f))[0]  # boundary face  # (interface)
    F_itf_nb = nbf[F_itf].type(torch.int64)  # neighbour list of interface face

    # create two lists of which element f_i / f_b is in
    E_F_i = torch.floor_divide(F_i, nface)
    E_F_inb = torch.floor_divide(F_inb, nface)
    E_F_b = torch.floor_divide(F_b, nface)
    E_F_itf = torch.floor_divide(F_itf, nface)
    E_F_itf_nb = torch.floor_divide(F_itf_nb, nface)

    # local face number
    f_b = torch.remainder(F_b, nface)
    f_i = torch.remainder(F_i, nface)
    f_inb = torch.remainder(F_inb, nface)
    f_itf = torch.remainder(F_itf, nface)
    f_itf_nb = torch.remainder(F_itf_nb, nface)

    if a00 or a01 or a10 or a11:
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
                    nb_gi_aln,
                    include_adv, u_n,
                    a00, a01, a10, a11
                )

        # update residual for boundary faces
        # r <= r + S*u_bc - S*u_i
        if True:  # ndim == 3:  # TODO: we could omit this iface loop in 2D. we'll make do with it at the moment.
            for iface in range(nface):
                idx_iface = f_b == iface
                r0, diagK, bdiagK = _s_res_fb(
                    r0, f_b[idx_iface], E_F_b[idx_iface],
                    x_i,
                    diagK, bdiagK, batch_start_idx,
                    include_adv, u_bc, u_n,
                    a00, a01, a10, a11
                )
                if include_itf:
                    r0 = _s_res_itf(
                        r0, x_i,
                        f_itf, E_F_itf,
                        f_itf_nb, E_F_itf_nb,
                        include_adv=include_adv,
                        x_k_in=u_n,
                    )  # include interface term
        else:
            raise Exception('2D hyper-elasticity not implemented!')
    elif include_itf:
        # only include interface term from other side (I_FS block)
        for iface in range(nface):  # TODO: WE COULD omit this loop for 2D since no element'll have 2 bc face
            r0 = _s_res_itf(
                r0, x_i,
                f_itf, E_F_itf,
                f_itf_nb, E_F_itf_nb,
                include_adv=include_adv,
                x_k_in=u_n,
            )  # include interface term
    return r0, diagK, bdiagK, diagS


def _s_res_fi(
        r_in, f_i, E_F_i,
        f_inb, E_F_inb,
        x_i_in,
        diagK, bdiagK, diagS, batch_start_idx,
        nb_gi_aln,
        include_adv, u_n,
        a00, a01, a10, a11
):
    """internal faces"""
    batch_in = f_i.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    include_p = a01 or a10
    # get element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    # if a00:
    #     r_u, r_p = slicing_x_i(r_in, include_p)
    #     x_i_u, x_i_p = slicing_x_i(x_i_in, include_p)
    # elif not a00 and a01:
    #     r_u = r_in.view(nele, u_nloc, ndim)
    #     x_i_p = x_i_in.view(nele, p_nloc)
    # elif not a00 and a10:
    #     r_p = r_in.view(nele, p_nloc)
    #     x_i_u = x_i_in.view(nele, u_nloc, ndim)
    r_dict = slicing_x_i(r_in)
    x_i_dict = slicing_x_i(x_i_in)

    # shape function on this side
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_i],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi,
        sn=sf_nd_nb.vel_func_space.element.sn,
    )
    sn = sf_nd_nb.vel_func_space.element.sn[f_i, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_i, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_i, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_i, ...]  # (batch_in, ndim, sngi)

    # shape function on the other side
    snx_nb, _, snormal_nb = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_inb],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi,
        sn=sf_nd_nb.vel_func_space.element.sn,
    )
    # get faces we want
    sn_nb = sf_nd_nb.vel_func_space.element.sn[f_inb, ...]  # (batch_in, nloc, sngi)
    snx_nb = snx_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim, nloc, sngi)
    snormal_nb = snormal_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim, sngi)
    # change gaussian points order on other side
    nb_aln = sf_nd_nb.vel_func_space.element.gi_align[nb_gi_aln, :]  # nb_aln for velocity element
    snx_nb = snx_nb[..., nb_aln]
    snormal_nb = snormal_nb[..., nb_aln]
    # don't forget to change gaussian points order on sn_nb!
    sn_nb = sn_nb[..., nb_aln]
    nb_aln = sf_nd_nb.pre_func_space.element.gi_align[nb_gi_aln, :]  # nb_aln for pressure element

    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = config.eta_e / h
    if a01:
        p_ith = x_i_dict['pre'][E_F_i, ...]
        p_inb = x_i_dict['pre'][E_F_inb, ...]
    if a00 or a10:
        u_ith = x_i_dict['vel'][E_F_i, ...]
        u_inb = x_i_dict['vel'][E_F_inb, ...]

    if a00:
        # K block
        K = torch.zeros(batch_in, u_nloc, ndim, u_nloc, ndim, device=dev, dtype=torch.float64)
        # this side
        # [v_i n_j] {du_i / dx_j}  consistent term
        K += torch.einsum(
            'bmg,bjg,bjng,bg,kl->bmknl',
            sn,  # (batch_in, nloc, sngi)
            snormal,  # (batch_in, ndim, sngi)
            snx,  # (batch_in, ndim, nloc, sngi)
            sdetwei,  # (batch_in, sngi)
            torch.eye(ndim, device=dev, dtype=torch.float64),  # (ndim, ndim)
        ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim)
        # {dv_i / dx_j} [u_i n_j]  symmetry term
        K += torch.einsum(
            'bjmg,bng,bjg,bg,kl->bmknl',
            snx,  # (batch_in, ndim, nloc, sngi)
            sn,  # (batch_in, nloc, sngi)
            snormal,  # (batch_in, ndim, sngi)
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
        K *= config.mu_f
        if include_adv:
            x_k_dict = slicing_x_i(u_n)  # this is non-linear vel/pre/displacement
            # Edge stabilisation (vel gradient penalty term)
            if sf_nd_nb.isES:
                u_ave = 0.5 * (sf_nd_nb.u_ave[E_F_i] + sf_nd_nb.u_ave[E_F_inb])  # (batch_in)
                # \gamma_e h^2 [v_i / x_j] [u_i / x_j]
                K += torch.einsum(
                    'b,bjmg,bjng,bg,kl->bmknl',
                    config.gammaES * h**2 * u_ave,  # (batch_in)
                    snx,  # (batch_in, ndim, nloc, sngi)
                    snx,  # (batch_in, ndim, nloc, sngi)
                    sdetwei,  # (batch_in, sngi)
                    torch.eye(ndim, device=dev, dtype=torch.float64),
                ) * (.5)
            # get upwind vel
            wknk_ave = torch.einsum(
                'bmg,bmi,big->bg',
                sn,  # (batch_in, u_nloc, sngi)
                x_k_dict['vel'][E_F_i, :, :] - sf_nd_nb.u_m[E_F_i, :, :],  # u_k - u_m (batch_in, u_nloc, sngi)
                snormal,  # (batch_in, ndim, sngi)
            ) * 0.5 + torch.einsum(
                'bmg,bmi,big->bg',
                sn_nb,  # (batch_in, u_nloc, sngi)
                x_k_dict['vel'][E_F_inb, :, :] - sf_nd_nb.u_m[E_F_inb, :, :],  # (batch_in, u_nloc, sngi)
                snormal,  # (batch_in, ndim, sngi)
            ) * 0.5
            wknk_upwd = 0.5 * (wknk_ave - torch.abs(wknk_ave)) * config.rho_f
            K += -torch.einsum(
                'bg,bmg,bng,bg,ij->bminj',
                wknk_upwd,  # (batch_in, sngi)
                sn,  # (batch_in, u_nloc, sngi)
                sn,  # (batch_in, u_nloc, sngi)
                sdetwei,  # (bathc_in, sngi)
                torch.eye(ndim, device=dev, dtype=torch.float64)
            )
            if sf_nd_nb.isPetrovGalerkinFace:
                # get petrov galerkin diffusivity, but only update if this is the 1st non-linear step
                tau_pg = petrov_galerkin.get_pg_tau_on_face(
                    u=u_n[E_F_i, :, :],
                    w=u_n[E_F_i, :, :],
                    v=sn,
                    vx=snx,
                    cell_volume=sf_nd_nb.vel_func_space.cell_volume[E_F_i],
                    batch_in=batch_in
                )
                tau_pg_nb = petrov_galerkin.get_pg_tau_on_face(
                    u=u_n[E_F_inb, :, :],
                    w=u_n[E_F_inb, :, :],
                    v=sn_nb,
                    vx=snx_nb,
                    cell_volume=sf_nd_nb.vel_func_space.cell_volume[E_F_inb],
                    batch_in=batch_in
                )
                tau_pg *= 0.5
                tau_pg_nb *= 0.5
                tau_pg += tau_pg_nb
                # this side
                # [v_i n_j] {du_i / dx_j tau_pg_i}  consistent term
                K += torch.einsum(
                    'bmg,bjg,bjng,blg,bg,kl->bmknl',
                    sn,  # (batch_in, nloc, sngi)
                    snormal,  # (batch_in, ndim, sngi)
                    snx,  # (batch_in, ndim, nloc, sngi)
                    tau_pg,  # (batch_in, ndim, sngi)
                    sdetwei,  # (batch_in, sngi)
                    torch.eye(ndim, device=dev, dtype=torch.float64),  # (ndim, ndim)
                ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim)
                # {dv_i / dx_j} [u_i n_j tau_pg_i]  symmetry term
                K += torch.einsum(
                    'bjmg,bng,bjg,blg,bg,kl->bmknl',
                    snx,  # (batch_in, ndim, nloc, sngi)
                    sn,  # (batch_in, nloc, sngi)
                    snormal,  # (batch_in, ndim, sngi)
                    tau_pg,  # (batch_in, ndim, sngi)
                    sdetwei,  # (batch_in, sngi)
                    torch.eye(ndim, device=dev, dtype=torch.float64),  # (ndim, ndim)
                ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim) \
                # \gamma_e * [v_i][u_i]  penalty term
                K += torch.einsum(
                    'bmg,bng,bjg,bg,b,ij->bminj',
                    sn,  # (batch_in, nloc, sngi)
                    sn,  # (batch_in, nloc, sngi)
                    tau_pg,  # (batch_in, ndim, sngi)
                    sdetwei,  # (batch_in, sngi)
                    gamma_e,  # (batch_in)
                    torch.eye(ndim, device=dev, dtype=torch.float64),
                )
            # elif config.isGradDivStab:
            #     u_ave_th = sf_nd_nb.u_ave[E_F_i, :]
            #     u_ave_nb = sf_nd_nb.u_ave[E_F_inb, :]
            #     u_ave_abs_th = torch.sqrt(torch.sum(torch.square(u_ave_th), dim=-1))  # (batch_in)
            #     u_ave_abs_nb = torch.sqrt(torch.sum(torch.square(u_ave_nb), dim=-1))  # (batch_in)
            #     tau = 0.5 * config.zeta * (u_ave_abs_th + u_ave_abs_nb)  # (batch_in)
            #     K += torch.einsum(
            #         'bmg,bi,bng,bj,b,bg->bminj',
            #         sn,  # (batch_in, nloc, sngi)
            #         snormal,  # (batch_in, ndim)
            #         sn,
            #         snormal,
            #         tau,
            #         sdetwei,  # (batch_in, sngi)
            #     )
        # update residual
        r_dict['vel'][E_F_i, ...] -= torch.einsum('bminj,bnj->bmi', K, u_ith)
        # put diagonal into diagK and bdiagK
        diagK[E_F_i-batch_start_idx, :, :] += torch.diagonal(K.view(batch_in, u_nloc*ndim, u_nloc*ndim),
                                                             dim1=1, dim2=2).view(batch_in, u_nloc, ndim)
        bdiagK[E_F_i-batch_start_idx, :, :] += K

        # other side
        K *= 0
        # [v_i n_j] {du_i / dx_j}  consistent term
        K += torch.einsum(
            'bmg,bjg,bjng,bg,kl->bmknl',
            sn,  # (batch_in, nloc, sngi)
            snormal,  # (batch_in, ndim, sngi)
            snx_nb,  # (batch_in, ndim, nloc, sngi)
            sdetwei,  # (batch_in, sngi)
            torch.eye(ndim, device=dev, dtype=torch.float64)
        ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim) \
        # {dv_i / dx_j} [u_i n_j]  symmetry term
        K += torch.einsum(
            'bjmg,bng,bjg,bg,kl->bmknl',
            snx,  # (batch_in, ndim, nloc, sngi)
            sn_nb,  # (batch_in, nloc, sngi)
            snormal_nb,  # (batch_in, ndim, sngi)
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
        K *= config.mu_f
        if include_adv:
            # Edge stabilisation (vel gradient penalty term)
            if sf_nd_nb.isES:
                # \gamma_e h^2 [v_i / x_j] [u_i / x_j]
                K += torch.einsum(
                    'b,bjmg,bjng,bg,kl->bmknl',
                    config.gammaES * h ** 2 * u_ave,  # (batch_in)
                    snx,  # (batch_in, ndim, nloc, sngi)
                    snx_nb,  # (batch_in, ndim, nloc, sngi)
                    sdetwei,  # (batch_in, sngi)
                    torch.eye(ndim, device=dev, dtype=torch.float64),
                ) * (-.5)
            K += -torch.einsum(
                'bg,bmg,bng,bg,ij->bminj',
                wknk_upwd,  # (batch_in, sngi)
                sn,  # (batch_in, u_nloc, sngi)
                sn_nb,  # (batch_in, u_nloc, sngi)
                sdetwei,  # (bathc_in, sngi)
                torch.eye(ndim, device=dev, dtype=torch.float64)
            ) * (-1.)
            if sf_nd_nb.isPetrovGalerkinFace:
                # other side
                # [v_i n_j] {du_i / dx_j tau_pg_i}  consistent term
                K += torch.einsum(
                    'bmg,bjg,bjng,blg,bg,kl->bmknl',
                    sn,  # (batch_in, nloc, sngi)
                    snormal,  # (batch_in, ndim, sngi)
                    snx_nb,  # (batch_in, ndim, nloc, sngi)
                    tau_pg,  # (batch_in, ndim, sngi)
                    sdetwei,  # (batch_in, sngi)
                    torch.eye(ndim, device=dev, dtype=torch.float64)
                ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim) \
                # {dv_i / dx_j} [u_i n_j tau_pg_i]  symmetry term
                K += torch.einsum(
                    'bjmg,bng,bjg,blg,bg,kl->bmknl',
                    snx,  # (batch_in, ndim, nloc, sngi)
                    sn_nb,  # (batch_in, nloc, sngi)
                    snormal_nb,  # (batch_in, ndim, sngi)
                    tau_pg,  # (batch_in, ndim, sngi)
                    sdetwei,  # (batch_in, sngi)
                    torch.eye(ndim, device=dev, dtype=torch.float64)
                ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim) \
                # \gamma_e * [v_i][u_i tau_pg_i]  penalty term
                K += torch.einsum(
                    'bmg,bng,bjg,bg,b,ij->bminj',
                    sn,  # (batch_in, nloc, sngi)
                    sn_nb,  # (batch_in, nloc, sngi)
                    tau_pg,  # (batch_in, ndim, sngi)
                    sdetwei,  # (batch_in, sngi)
                    gamma_e,  # (batch_in)
                    torch.eye(ndim, device=dev, dtype=torch.float64),
                ) * (-1.)  # because n2 \cdot n1 = -1
            # elif config.isGradDivStab:
            #     K += torch.einsum(
            #         'bmg,bi,bng,bj,b,bg->bminj',
            #         sn,  # (batch_in, nloc, sngi)
            #         snormal,  # (batch_in, ndim)
            #         sn_nb,
            #         snormal_nb,
            #         tau,
            #         sdetwei,  # (batch_in, sngi)
            #     )
        # update residual
        r_dict['vel'][E_F_i, ...] -= torch.einsum('bminj,bnj->bmi', K, u_inb)
        del K

    if not include_p:  # no need to go to G, G^T and S
        return r_in, diagK, bdiagK, diagS
    # include p blocks (G, G^T, S)
    sq = sf_nd_nb.pre_func_space.element.sn[f_i, ...]  # (batch_in, nloc, sngi)
    sq_nb = sf_nd_nb.pre_func_space.element.sn[f_inb, ...]  # (batch_in, nloc, sngi)
    sq_nb = sq_nb[..., nb_aln]
    # G block
    G = torch.zeros(batch_in, u_nloc, ndim, p_nloc, device=dev, dtype=torch.float64)
    # this side
    # [v_i n_i] {p}
    G += torch.einsum(
        'bmg,big,bng,bg->bmin',
        sn,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        sq,  # (batch_in, p_nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (0.5)
    if a01:
        # update velocity residual from pressure gradient
        r_dict['vel'][E_F_i, ...] -= torch.einsum('bmin,bn->bmi', G, p_ith)
    if a10:
        # update pressure residual from velocity divergence
        r_dict['pre'][E_F_i, ...] -= torch.einsum('bnjm,bnj->bm', G, u_ith)

    # other side
    if a01:
        G *= 0
        # {p} [v_i n_i]
        G += torch.einsum(
            'bmg,big,bng,bg->bmin',
            sn,  # (batch_in, u_nloc, sngi)
            snormal,  # (batch_in, ndim, sngi)
            sq_nb,  # (batch_in, p_nloc, sngi)
            sdetwei,  # (batch_in, sngi)
        ) * (0.5)
        # update velocity residual from pressure gradient
        r_dict['vel'][E_F_i, ...] -= torch.einsum('bmin,bn->bmi', G, p_inb)
    if a10:
        # G^T
        G *= 0
        # {q} [u_j n_j]
        G += torch.einsum(
            'bmg,bng,bjg,bg->bnjm',
            sq,  # (batch_in, p_nloc, sngi)
            sn_nb,  # (batch_in, u_nloc, sngi)
            snormal_nb,  # (batch_in, ndim, sngi)
            sdetwei,  # (batch_in, sngi)
        ) * (0.5)
        # update pressure residual from velocity divergence
        r_dict['pre'][E_F_i, ...] -= torch.einsum('bnjm,bnj->bm', G, u_inb)

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
        r_dict['pre'][E_F_i, ...] -= torch.einsum('bmn,bn->bm', S, p_ith)
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
        r_dict['pre'][E_F_i, ...] -= torch.einsum('bmn,bn->bm', S, p_inb)
    # this concludes surface integral on interior faces.
    return r_in, diagK, bdiagK, diagS


def _s_res_fb(
        r_in, f_b, E_F_b,
        x_i_in,
        diagK, bdiagK, batch_start_idx,
        include_adv, u_bc, u_n,
        a00, a01, a10, a11
):
    """boundary faces and interface on fluid side"""
    batch_in = f_b.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    if batch_in < 1:  # nothing to do here.
        return r_in, diagK, bdiagK
    include_p = a01 or a10
    # get element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    # if a00:
    #     r_u, r_p = slicing_x_i(r_in, include_p)
    #     x_i_u, x_i_p = slicing_x_i(x_i_in, include_p)
    # elif not a00 and a01:
    #     r_u = r_in.view(nele, u_nloc, ndim)
    #     x_i_p = x_i_in.view(nele, p_nloc)
    # elif not a00 and a10:
    #     r_p = r_in.view(nele, p_nloc)
    #     x_i_u = x_i_in.view(nele, u_nloc, ndim)
    r_dict = slicing_x_i(r_in)
    x_i_dict = slicing_x_i(x_i_in)
    # x_k_dict = slicing_x_i(u_n)  # this is non-linear vel/pre/displacement

    # shape function
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_b],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi,
        sn=sf_nd_nb.vel_func_space.element.sn,
    )
    sn = sf_nd_nb.vel_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    if include_p:
        sq = sf_nd_nb.pre_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_b, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_b, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_b, ...]  # (batch_in, ndim, sngi)
    if ndim == 3:
        gamma_e = config.eta_e / torch.sqrt(torch.sum(sdetwei, -1))
    else:
        gamma_e = config.eta_e / torch.sum(sdetwei, -1)

    if a01:
        p_ith = x_i_dict['pre'][E_F_b, ...]
    if a00 or a10:
        u_ith = x_i_dict['vel'][E_F_b, ...]

    if a00:
        # block K
        K = torch.zeros(batch_in, u_nloc, ndim, u_nloc, ndim,
                        device=dev, dtype=torch.float64)
        # [vi nj] {du_i / dx_j}  consistent term
        K -= torch.einsum(
            'bmg,bjg,bjng,bg,kl->bmknl',
            sn,  # (batch_in, nloc, sngi)
            snormal,  # (batch_in, ndim, sngi)
            snx,  # (batch_in, ndim, nloc, sngi)
            sdetwei,  # (batch_in, sngi)
            torch.eye(ndim, device=dev, dtype=torch.float64)
        )  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim)
        # {dv_i / dx_j} [ui nj]  symmetry term
        K -= torch.einsum(
            'bjmg,bng,bjg,bg,kl->bmknl',
            snx,  # (batch_in, ndim, nloc, sngi)
            sn,  # (batch_in, nloc, sngi)
            snormal,  # (batch_in, ndim, sngi)
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
        K *= config.mu_f
        if include_adv:
            # get upwind velocity
            # note that here u_bc will always be 0 on interface
            # so we add no advection term to interface here.
            # if we want to add advection term, we should do it in
            # _s_res_itf
            wknk_ave = torch.einsum(
                'bmg,bmi,big->bg',
                sn,  # (batch_in, u_nloc, sngi)
                u_bc[E_F_b, ...],  # (batch_in, u_nloc, ndim)
                snormal,  # (batch_in, ndim, sngi)
            )
            wknk_upwd = 0.5 * (wknk_ave - torch.abs(wknk_ave)) * config.rho_f
            K += -torch.einsum(
                'bg,bmg,bng,bg,ij->bminj',
                wknk_upwd,
                sn,  # (batch_in, u_nloc, sngi)
                sn,  # (batch_in, u_nloc, sngi)
                sdetwei,  # (batch_in, sngi)
                torch.eye(ndim, device=dev, dtype=torch.float64)
            )
            if sf_nd_nb.isPetrovGalerkinFace:
                # get tau_pg
                tau_pg = petrov_galerkin.get_pg_tau_on_face(
                    u=u_n[E_F_b, ...],
                    w=u_n[E_F_b, ...],
                    v=sn,
                    vx=snx,
                    cell_volume=sf_nd_nb.vel_func_space.cell_volume[E_F_b],
                    batch_in=batch_in,
                )
                # [vi nj] {du_i / dx_j}  consistent term
                K -= torch.einsum(
                    'bmg,bjg,bjng,blg,bg,kl->bmknl',
                    sn,  # (batch_in, nloc, sngi)
                    snormal,  # (batch_in, ndim, sngi)
                    snx,  # (batch_in, ndim, nloc, sngi)
                    tau_pg,  # (batch_in, ndim, sngi)
                    sdetwei,  # (batch_in, sngi)
                    torch.eye(ndim, device=dev, dtype=torch.float64)
                )  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim)
                # {dv_i / dx_j} [ui nj]  symmetry term
                K -= torch.einsum(
                    'bjmg,bng,bjg,blg,bg,kl->bmknl',
                    snx,  # (batch_in, ndim, nloc, sngi)
                    sn,  # (batch_in, nloc, sngi)
                    snormal,  # (batch_in, ndim, sngi)
                    tau_pg,  # (batch_in, ndim, sngi)
                    sdetwei,  # (batch_in, sngi)
                    torch.eye(ndim, device=dev, dtype=torch.float64)
                )  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim)
                # \gamma_e [v_i] [u_i]  penalty term
                K += torch.einsum(
                    'bmg,bng,bjg,bg,b,ij->bminj',
                    sn,  # (batch_in, nloc, sngi)
                    sn,  # (batch_in, nloc, sngi)
                    tau_pg,  # (batch_in, ndim, sngi)
                    sdetwei,  # (batch_in, sngi)
                    gamma_e,  # (batch_in)
                    torch.eye(ndim, device=dev, dtype=torch.float64)
                )
        # update residual
        r_dict['vel'][E_F_b, ...] -= torch.einsum('bminj,bnj->bmi', K, u_ith)
        # put in diagonal
        diagK[E_F_b - batch_start_idx, :, :] += torch.diagonal(K.view(batch_in, u_nloc * ndim, u_nloc * ndim),
                                                               dim1=-2, dim2=-1).view(batch_in, u_nloc, ndim)
        bdiagK[E_F_b - batch_start_idx, ...] += K
        del K

    if not include_p:
        return r_in, diagK, bdiagK
    # block G
    G = torch.zeros(batch_in, u_nloc, ndim, p_nloc,
                    device=dev, dtype=torch.float64)
    # [v_i n_i] {p}
    G += torch.einsum(
        'bmg,big,bng,bg->bmin',
        sn,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        sq,  # (batch_in, p_nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    )
    if a01:
        # update velocity residual from pressure gradient
        r_dict['vel'][E_F_b, :, :] -= torch.einsum('bmin,bn->bmi', G, p_ith)
    if a10:
        # block G^T
        # update pressure residual from velocity divergence
        r_dict['pre'][E_F_b, :] -= torch.einsum('bnjm,bnj->bm', G, u_ith)

    return r_in, diagK, bdiagK


def _s_res_itf(
        r_in, x_i_in,
        f_itf, E_F_itf,
        f_itf_nb, E_F_itf_nb,
        include_adv,
        x_k_in,
):
    """
    add contribution from solid to interface
    """
    batch_in = f_itf.shape[0]
    if batch_in < 1:  # nothing to do here.
        return r_in
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)

    r_dict = slicing_x_i(r_in)
    x_i_dict = slicing_x_i(x_i_in)

    # shape function
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_itf],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi,
        sn=sf_nd_nb.vel_func_space.element.sn,
    )
    sn = sf_nd_nb.vel_func_space.element.sn[f_itf, ...]  # (batch_in, nloc, sngi)
    sq = sf_nd_nb.pre_func_space.element.sn[f_itf, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_itf, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_itf, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_itf, ...]  # (batch_in, ndim, sngi)
    # neighbour face shape function
    sn_nb = sf_nd_nb.vel_func_space.element.sn[f_itf_nb, ...]
    # sn_nb = torch.zeros_like(sn, device=dev, dtype=torch.float64)
    for nb_gi_aln in range(ndim):  # 'ndim' alignnment of GI points on neighbour faces
        idx = sf_nd_nb.vel_func_space.alnmt[E_F_itf * nface + f_itf] == nb_gi_aln
        nb_aln = sf_nd_nb.vel_func_space.element.gi_align[nb_gi_aln, :]
        sn_nb[idx] = sn_nb[idx][..., nb_aln]
    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = config.eta_e / h
    u_s_nb = x_i_dict['disp'][E_F_itf_nb, ...] / config.dt * sf_nd_nb.bdfscm.gamma

    # {dv_i / dx_j} [u_Si n_Sj]
    r_dict['vel'][E_F_itf, ...] += torch.einsum(
        'bjmg,bng,bjg,bg,bni->bmi',
        snx,  # (batch_in, ndim, u_nloc, sngi)
        sn_nb,  # (batch_in, u_nloc, sngi)
        -snormal,  # (batch_in, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
        u_s_nb,  # (batch_in, u_nloc, ndim)
    ) * config.mu_f

    # \gamma_e [u_Di] [v_i]
    r_dict['vel'][E_F_itf, ...] += torch.einsum(
        'b,bmg,bng,bg,bni->bmi',
        gamma_e,  # (batch_in)
        sn,  # (batch_in, u_nloc, sngi)
        sn_nb,
        sdetwei,  # (batch_in, sngi)
        u_s_nb,  # (batch_in, u_nloc, ndim)
    ) * config.mu_f

    # 2. {q} [u_Di n_i]
    r_dict['pre'][E_F_itf, ...] -= torch.einsum(
        'bmg,bng,big,bg,bni->bm',
        sq,  # (batch_in, p_nloc, sngi)
        sn_nb,  # (batch_in, u_nloc, sngi)
        -snormal,  # (batch_in, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
        u_s_nb,  # (batch_in, u_nloc, ndim)
    )

    if False:  # include_adv:
        x_k_dict = slicing_x_i(x_k_in)
        u_f_k_th = x_k_dict['vel'][E_F_itf, ...]
        u_f_th = x_i_dict['vel'][E_F_itf, ...]
        # get upwind velocity
        # add advection term on interface
        wknk_ave = torch.einsum(
            'bmg,bmi,big->bg',
            sn,  # (batch_in, u_nloc, sngi)
            u_f_k_th - sf_nd_nb.u_m[E_F_itf, ...],  # (batch_in, u_nloc, ndim)
            snormal,  # (batch_in, ndim, sngi)
        )
        wknk_upwd = 0.5 * (wknk_ave - torch.abs(wknk_ave)) * config.rho_f
        r_dict['vel'][E_F_itf, ...] += torch.einsum(
            'bg,bmg,bng,bg,bnj->bmj',
            wknk_upwd,
            sn,  # (batch_in, u_nloc, sngi)
            sn,  # (batch_in, u_nloc, sngi)
            sdetwei,  # (batch_in, sngi)
            u_f_th,  # (batch_in, u_nloc, ndim)
        )
        r_dict['vel'][E_F_itf, ...] += torch.einsum(
            'bg,bmg,bng,bg,bnj->bmj',
            wknk_upwd,
            sn,  # (batch_in, u_nloc, sngi)
            sn_nb,  # (batch_in, u_nloc, sngi)
            sdetwei,  # (batch_in, sngi)
            u_s_nb,  # (batch_in, u_nloc, ndim)
        )
    return r_in


def get_rhs(x_rhs, u_bc, f, include_adv, u_n=0, isAdvExp=False, u_k=0,
            d_n=0):
    """get right-hand side"""
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele_f, dtype=int)
    idx_in = torch.zeros(nele, dtype=torch.bool)
    idx_in_f = torch.zeros(nele * nface, dtype=torch.bool, device=dev)

    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    for u_bci in u_bc:
        u_bci = u_bci.view(nele, u_nloc, ndim)
    f = f.view(nele, u_nloc, ndim)

    for i in range(nnn):
        idx_in *= False
        # volume integral
        idx_in[brk_pnt[i]:brk_pnt[i+1]] = True
        x_rhs = _k_rhs_one_batch(x_rhs, u_n, f, idx_in, isAdvExp, u_k)
        # surface integral
        idx_in_f *= False
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        x_rhs = _s_rhs_one_batch(x_rhs, u_bc, idx_in_f, include_adv, isAdvExp, u_k,
                                 d_n)

    return x_rhs


def _k_rhs_one_batch(
        rhs_in, u_n, f, idx_in, isAdvExp, u_k
):
    batch_in = int(torch.sum(idx_in))  # only fluid subdomain
    if batch_in < 1:  # nothing to do here.
        return rhs_in
    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    rhs = slicing_x_i(rhs_in)  # now rhs is a dict of "vel" "pre" and "disp"

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
    rhs['vel'][idx_in, ...] += torch.einsum(
        'mg,ng,bg,ij,bnj->bmi',
        n,  # (u_nloc, ngi)
        n,  # (u_nloc, ngi)
        ndetwei,  # (batch_in, ngi)
        torch.eye(ndim, device=dev, dtype=torch.float64),  # (ndim, ndim)
        f[idx_in, ...],  # (batch_in, u_nloc, ndim)
    )

    # if transient, add rho/dt * u_n to vel rhs
    if sf_nd_nb.isTransient:
        u_n = u_n.view(-1, u_nloc, ndim)
        rhs['vel'][idx_in, ...] += torch.einsum(
            'mg,ng,bg,ij,bnj->bmi',
            n,
            n,
            ndetwei,
            torch.eye(ndim, device=dev, dtype=torch.float64),  # (ndim, ndim)
            u_n[idx_in, ...],
        ) * config.rho_f / sf_nd_nb.dt

    # # p \nabla.v contribution to vel rhs
    # rhs[0][idx_in, ...] += torch.einsum(
    #     'bimg,ng,bg,bn->bmi',
    #     nx,  # (batch_in, ndim, u_nloc, ngi)
    #     q,  # (p_nloc, ngi)
    #     ndetwei,  # (batch_in, ngi)
    #     p_i[idx_in, ...],  # (batch_in, p_nloc)
    # )

    if isAdvExp:
        u_th = u_k.view(nele, u_nloc, ndim)[idx_in, ...]
        rhs[0][idx_in, ...] -= torch.einsum(
            'lg,bli,bing,mg,bg,bnk->bmk',
            n,  # (u_nloc, ngi)
            u_th,  # (batch_in, u_nloc, ndim)
            nx,  # (batch_in, ndim, u_nloc, ngi)
            n,  # (u_nloc, ngi)
            ndetwei,  # (batch_in, ngi)
            u_th,  # (batch_in, u_nloc, ndim)
        ) * config.rho_f

    return rhs_in


def _s_rhs_one_batch(
        rhs, u_bc, idx_in_f,
        include_adv,
        isAdvExp, u_k,
        d_n
):
    # get essential data
    nbf = sf_nd_nb.vel_func_space.nbf
    alnmt = sf_nd_nb.vel_func_space.alnmt
    glb_bcface_type = sf_nd_nb.vel_func_space.glb_bcface_type

    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    d_n = d_n.view(nele, u_nloc, ndim)

    # separate nbf to get internal face list and boundary face list
    F_i = torch.where(torch.logical_and(glb_bcface_type < 0,
                                        idx_in_f))[0]  # interior face of fluid subdomain
    F_inb = nbf[F_i]  # neighbour list of interior face
    F_inb = F_inb.type(torch.int64)
    F_b_d = torch.where(torch.logical_and(
        sf_nd_nb.vel_func_space.glb_bcface_type == 0,
        idx_in_f))[0]  # boundary face
    F_b_n = torch.where(torch.logical_and(
        sf_nd_nb.vel_func_space.glb_bcface_type == 1,
        idx_in_f))[0]  # boundary face
    F_itf = torch.where(torch.logical_and(
        sf_nd_nb.vel_func_space.glb_bcface_type == 4,
        idx_in_f))[0]  # boundary face  # (interface)
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

    # for interior faces
    if isAdvExp:
        for iface in range(nface):
            for nb_gi_aln in range(nface - 1):
                idx_iface = (f_i == iface) & (sf_nd_nb.vel_func_space.alnmt[F_i] == nb_gi_aln)
                if idx_iface.sum() < 1:
                    # there is nothing to do here, go on
                    continue
                rhs = _s_rhs_fi(
                    rhs, f_i[idx_iface], E_F_i[idx_iface],
                    f_inb[idx_iface], E_F_inb[idx_iface],
                    u_k,
                    nb_gi_aln)
    # hasNeuBC = False
    # if len(u_bc) > 1: hasNeuBC = True
    # update residual for boundary faces
    # r <= r + S*u_bc - S*u_i
    if True:  # ndim == 3:  # this is slower for 2D, but we'll make do with it.
        for iface in range(nface):
            idx_iface_d = f_b_d == iface
            idx_iface_n = f_b_n == iface
            idx_iface_itf = f_itf == iface
            rhs = _s_rhs_fb(
                rhs,
                f_b_d[idx_iface_d], E_F_b_d[idx_iface_d],
                u_bc[0], u_k,
                include_adv,
                isAdvExp
            )
            rhs = _s_rhs_fitf(
                rhs,
                f_itf[idx_iface_itf], E_F_itf[idx_iface_itf],
                f_itf_nb[idx_iface_itf], E_F_itf_nb[idx_iface_itf],
                include_adv,
                u_bc, u_k,
                d_n
            )  # rhs terms from interface
            # neumann bc
            if idx_iface_n.sum() > 0:
                rhs = _s_rhs_fb_neumann(
                    rhs,
                    f_b_n[idx_iface_n], E_F_b_n[idx_iface_n],
                    u_bc[1])
    else:
        raise Exception('2D stokes not implemented!')

    return rhs


def _s_rhs_fi(
        rhs_in, f_i, E_F_i,
        f_inb, E_F_inb,
        u_k,
        nb_gi_aln
):
    """
    update explicit advection term
    """
    batch_in = f_i.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    # get element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc

    rhs = slicing_x_i(rhs_in)

    # shape function on this side
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_i],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi,
        sn=sf_nd_nb.vel_func_space.element.sn,
    )
    sn = sf_nd_nb.vel_func_space.element.sn[f_i, ...]  # (batch_in, nloc, sngi)
    # snx = snx[dummy_idx, f_i, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_i, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_i, ...]  # (batch_in, ndim, sngi)

    # shape function on the other side
    snx_nb, _, snormal_nb = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_inb],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi,
        sn=sf_nd_nb.vel_func_space.element.sn,
    )
    # get faces we want
    sn_nb = sf_nd_nb.vel_func_space.element.sn[f_inb, ...]  # (batch_in, nloc, sngi)
    # snx_nb = snx_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim, nloc, sngi)
    # snormal_nb = snormal_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim)
    # change gaussian points order on other side
    nb_aln = sf_nd_nb.vel_func_space.element.gi_align[nb_gi_aln, :]  # nb_aln for velocity element
    # snx_nb = snx_nb[..., nb_aln]
    # don't forget to change gaussian points order on sn_nb!
    sn_nb = sn_nb[..., nb_aln]
    # nb_aln = sf_nd_nb.pre_func_space.element.gi_align[nb_gi_aln, :]  # nb_aln for pressure element

    # h = torch.sum(sdetwei, -1)
    # if ndim == 3:
    #     h = torch.sqrt(h)
    # gamma_e = config.eta_e / h

    u_k = u_k.view(nele, u_nloc, ndim)
    u_k_th = u_k[E_F_i, :, :]
    u_k_nb = u_k[E_F_inb, :, :]

    # get upwind vel
    wknk_ave = torch.einsum(
        'bmg,bmi,big->bg',
        sn,  # (batch_in, u_nloc, sngi)
        u_k_th,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
    ) * 0.5 + torch.einsum(
        'bmg,bmi,big->bg',
        sn_nb,  # (batch_in, u_nloc, sngi)
        u_k_nb,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
    ) * 0.5
    wknk_upwd = 0.5 * (wknk_ave - torch.abs(wknk_ave)) * config.rho_f

    # this side
    rhs[0][E_F_i, ...] -= -torch.einsum(
        'bg,bmg,bng,bg,bni->bmi',
        wknk_upwd,  # (batch_in, sngi)
        sn,  # (batch_in, u_nloc, sngi)
        sn,  # (batch_in, u_nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        u_k_th,  # (batch_in, u_nloc, ndim)
    )
    # other side
    rhs[0][E_F_i, ...] -= -torch.einsum(
        'bg,bmg,bng,bg,bni->bmi',
        wknk_upwd,  # (batch_in, sngi)
        sn,  # (batch_in, u_nloc, sngi)
        sn_nb,  # (batch_in, u_nloc, sngi)
        sdetwei,  # (bathc_in, sngi)
        u_k_nb,  # (batch_in, u_nloc, ndim)
    ) * (-1.)

    return rhs_in


def _s_rhs_fb(
        rhs_in, f_b, E_F_b,
        u_bc, u_n,
        include_adv,
        isAdvExp
):
    """
    contains contribution of
    1. vel dirichlet BC to velocity rhs
    2. vel dirichlet BC to pressure rhs

    if isAdvExp:
    3. if we treat advection explicitly, we include bc contri of advection term
    here
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
        sngi=sf_nd_nb.vel_func_space.element.sngi,
        sn=sf_nd_nb.vel_func_space.element.sn,
    )
    sn = sf_nd_nb.vel_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    sq = sf_nd_nb.pre_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_b, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_b, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_b, ...]  # (batch_in, ndim, sngi)
    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = config.eta_e / h
    # print('gamma_e', gamma_e)
    u_bc_th = u_bc[E_F_b, ...]
    # p_i_th = p_i[E_F_b, ...]

    # 1.1 {dv_i / dx_j} [u_Di n_j]
    rhs['vel'][E_F_b, ...] -= torch.einsum(
        'bjmg,bng,bjg,bg,bni->bmi',
        snx,  # (batch_in, ndim, u_nloc, sngi)
        sn,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
        u_bc_th,  # (batch_in, u_nloc, ndim)
    ) * config.mu_f

    # 1.2 \gamma_e [u_Di] [v_i]
    rhs['vel'][E_F_b, ...] += torch.einsum(
        'b,bmg,bng,bg,bni->bmi',
        gamma_e,  # (batch_in)
        sn,  # (batch_in, u_nloc, sngi)
        sn,
        sdetwei,  # (batch_in, sngi)
        u_bc_th,  # (batch_in, u_nloc, ndim)
    ) * config.mu_f

    # 2. {q} [u_Di n_i]
    rhs['pre'][E_F_b, ...] += torch.einsum(
        'bmg,bng,big,bg,bni->bm',
        sq,  # (batch_in, p_nloc, sngi)
        sn,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
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

    if include_adv:
        # bc contribution of advection term (in Navier-Stokes)
        # get upwind vel
        wknk_ave = torch.einsum(
            'bmg,bmi,big->bg',
            sn,  # (batch_in, u_nloc, sngi)
            u_bc_th,  # (batch_in, u_nloc, ndim)
            snormal,  # (batch_in, ndim, sngi)
        )
        wknk_upwd = 0.5 * (wknk_ave - torch.abs(wknk_ave)) * config.rho_f
        rhs['vel'][E_F_b, ...] += -torch.einsum(
            'bg,bmg,bng,bg,ij,bnj->bmi',
            wknk_upwd,
            sn,  # (batch_in, u_nloc, sngi)
            sn,  # (batch_in, u_nloc, sngi)
            sdetwei,  # (batch_in, sngi)
            torch.eye(ndim, device=dev, dtype=torch.float64),
            u_bc_th,  # (batch_in, u_nloc, ndim)
        )
        if sf_nd_nb.isPetrovGalerkinFace:
            # get tau_pg
            tau_pg = petrov_galerkin.get_pg_tau_on_face(
                u=u_n[E_F_b, ...],
                w=u_n[E_F_b, ...],
                v=sn,
                vx=snx,
                cell_volume=sf_nd_nb.vel_func_space.cell_volume[E_F_b],
                batch_in=batch_in,
            ) * config.rho_f
            # 1.1 {dv_i / dx_j} [u_Di n_j tau_pg_i]
            rhs['vel'][E_F_b, ...] -= torch.einsum(
                'bjmg,bng,bjg,bg,bni,big->bmi',
                snx,  # (batch_in, ndim, u_nloc, sngi)
                sn,  # (batch_in, u_nloc, sngi)
                snormal,  # (batch_in, ndim, sngi)
                sdetwei,  # (batch_in, sngi)
                u_bc_th,  # (batch_in, u_nloc, ndim)
                tau_pg,  # (batch_in, ndim, sngi)
            )

            # 1.2 \gamma_e [u_Di tau_pg_i] [v_i]
            rhs['vel'][E_F_b, ...] += torch.einsum(
                'b,bmg,bng,bg,bni,big->bmi',
                gamma_e,  # (batch_in)
                sn,  # (batch_in, u_nloc, sngi)
                sn,
                sdetwei,  # (batch_in, sngi)
                u_bc_th,  # (batch_in, u_nloc, ndim)
                tau_pg,  # (batch_in, ndim, sngi)
            )

    return rhs_in


def _s_rhs_fitf(
    rhs_in,
    f_itf, E_F_itf,
    f_itf_nb, E_F_itf_nb,
    include_adv,
    u_bc, u_k,
    d_n
):
    """
    interface terms that goes to rhs
    """
    batch_in = f_itf.shape[0]
    if batch_in < 1:  # nothing to do here.
        return rhs_in
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    rhs = slicing_x_i(rhs_in)

    # shape function
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_itf],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi,
        sn=sf_nd_nb.vel_func_space.element.sn,
    )
    sn = sf_nd_nb.vel_func_space.element.sn[f_itf, ...]  # (batch_in, nloc, sngi)
    sq = sf_nd_nb.pre_func_space.element.sn[f_itf, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_itf, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_itf, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_itf, ...]  # (batch_in, ndim, sngi)
    # neighbour face shape function
    sn_nb = sf_nd_nb.vel_func_space.element.sn[f_itf_nb, ...]
    # sn_nb = torch.zeros_like(sn, device=dev, dtype=torch.float64)
    for nb_gi_aln in range(ndim):  # 'ndim' alignnment of GI points on neighbour faces
        idx = sf_nd_nb.vel_func_space.alnmt[E_F_itf * nface + f_itf] == nb_gi_aln
        nb_aln = sf_nd_nb.vel_func_space.element.gi_align[nb_gi_aln, :]
        sn_nb[idx] = sn_nb[idx][..., nb_aln]
    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = config.eta_e / h
    u_s_nb = d_n[E_F_itf_nb, ...] / sf_nd_nb.dt

    # {dv_i / dx_j} [u_Si n_Sj]
    rhs['vel'][E_F_itf, ...] -= torch.einsum(
        'bjmg,bng,bjg,bg,bni->bmi',
        snx,  # (batch_in, ndim, u_nloc, sngi)
        sn_nb,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
        u_s_nb,  # (batch_in, u_nloc, ndim)
    ) * config.mu_f

    # \gamma_e [u_Di] [v_i]
    rhs['vel'][E_F_itf, ...] += torch.einsum(
        'b,bmg,bng,bg,bni->bmi',
        gamma_e,  # (batch_in)
        sn,  # (batch_in, u_nloc, sngi)
        sn_nb,
        sdetwei,  # (batch_in, sngi)
        u_s_nb,  # (batch_in, u_nloc, ndim)
    ) * config.mu_f

    # 2. {q} [u_Di n_i]
    rhs['pre'][E_F_itf, ...] += torch.einsum(
        'bmg,bng,big,bg,bni->bm',
        sq,  # (batch_in, p_nloc, sngi)
        sn_nb,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
        u_s_nb,  # (batch_in, u_nloc, ndim)
    )

    if False:  # include_adv:
        # add advection term on interface
        # get upwind velocity
        u_f_k_th = u_k[E_F_itf, ...]
        wknk_ave = torch.einsum(
            'bmg,bmi,big->bg',
            sn,  # (batch_in, u_nloc, sngi)
            u_f_k_th - sf_nd_nb.u_m[E_F_itf, ...],  # (batch_in, u_nloc, ndim)
            snormal,  # (batch_in, ndim, sngi)
        )
        wknk_upwd = 0.5 * (wknk_ave - torch.abs(wknk_ave)) * config.rho_f
        rhs['vel'][E_F_itf, ...] += torch.einsum(
            'bg,bmg,bng,bg,bnj->bmj',
            wknk_upwd,
            sn,  # (batch_in, u_nloc, sngi)
            sn_nb,  # (batch_in, u_nloc, sngi)
            sdetwei,  # (batch_in, sngi)
            u_s_nb,  # (batch_in, u_nloc, ndim)
        )

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
        sngi=sf_nd_nb.vel_func_space.element.sngi,
        sn=sf_nd_nb.vel_func_space.element.sn,
    )
    sn = sf_nd_nb.vel_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    # sq = sf_nd_nb.pre_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    # snx = snx[dummy_idx, f_b, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_b, ...]  # (batch_in, sngi)
    # snormal = snormal[dummy_idx, f_b, ...]  # (batch_in, ndim, sngi)
    # h = torch.sum(sdetwei, -1)
    # if ndim == 3:
    #     h = torch.sqrt(h)
    # gamma_e = config.eta_e / h
    # print('gamma_e', gamma_e)
    u_bc_th = u_bc[E_F_b, ...]

    # 3. TODO: Neumann BC
    rhs['vel'][E_F_b, ...] += torch.einsum(
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
    r0_dict = slicing_x_i(r0)
    return torch.linalg.norm(
        torch.cat((r0_dict['vel'][0:nele_f, ...].view(-1),
                   r0_dict['pre'][0:nele_f, ...].view(-1),
                   r0_dict['disp'][nele_f:nele, ...].view(-1)),
                  0))


def slicing_x_i(x_i, include_p=True, isFSI=config.isFSI):
    """slicing a one-dimention array
    to get u and p"""
    if not isFSI:
        if include_p:
            u_nloc = sf_nd_nb.vel_func_space.element.nloc
            p_nloc = sf_nd_nb.pre_func_space.element.nloc
            u = x_i[0:nele*u_nloc*ndim].view(nele, u_nloc, ndim)
            p = x_i[nele*u_nloc*ndim:nele*u_nloc*ndim+nele*p_nloc].view(nele, p_nloc)
        else:
            u_nloc = sf_nd_nb.vel_func_space.element.nloc
            u = x_i[0:nele*u_nloc*ndim].view(nele, u_nloc, ndim)
            p = 0  # put in 0 to ... do nothing...
        d = 0  # put 0 to displacement, do nothing...
        return u, p
    else:
        if not include_p:
            raise ValueError('its not expected to not include pressure in FSI when doing slicing...')
        u_nloc = sf_nd_nb.vel_func_space.element.nloc
        p_nloc = sf_nd_nb.pre_func_space.element.nloc
        u = x_i[0:nele*u_nloc*ndim].view(nele, u_nloc, ndim)
        d = x_i[nele*u_nloc*ndim:
                nele*u_nloc*ndim*2].view(nele, u_nloc, ndim)
        p = x_i[nele*u_nloc*ndim*2:
                nele*u_nloc*ndim*2+nele*p_nloc].view(nele, p_nloc)
    return {'vel': u, 'pre': p, 'disp': d, 'all': x_i}


def get_l2_error(x_i_in, x_ana):
    """given numerical solution and analytical solution (exact up to interpolation error),
    calculate l2 error"""
    if not type(x_i_in) == torch.Tensor:
        x_i = torch.tensor(x_i_in, device=dev, dtype=torch.float64)
    else:
        x_i = x_i_in
    u_i, p_i = slicing_x_i(x_i)['vel'], slicing_x_i(x_i)['pre']
    # first project pressure to velocity space
    prolongator = sf_nd_nb.pre_func_space.get_pndg_prolongator(
        p=sf_nd_nb.pre_func_space.element.ele_order,
        gi_order=sf_nd_nb.pre_func_space.element.gi_order,
        ndim=sf_nd_nb.pre_func_space.element.ndim,
        dev=config.dev,
    )
    p_i = torch.einsum(
        'mn,bn->bm', prolongator, p_i
    )
    u_nonods = sf_nd_nb.vel_func_space.element.nloc * nele
    p_nonods = sf_nd_nb.pre_func_space.element.nloc * nele
    u_ana, p_ana = (x_ana[0:u_nonods*ndim].view(nele, -1, ndim),
                    x_ana[u_nonods*ndim:u_nonods*ndim+u_nonods].view(nele, -1))
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
    p_i_gi = torch.einsum('ng,bn->bng', n, p_i)
    p_ana_gi = torch.einsum('ng,bn->bng', n, p_ana)
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
        nlx=sf_nd_nb.pre_func_space.x_element.nlx,
        x_loc=sf_nd_nb.pre_func_space.x_ref_in,
        weight=sf_nd_nb.pre_func_space.element.weight,
        nloc=p_nloc,
        ngi=sf_nd_nb.pre_func_space.element.ngi,
        real_nlx=sf_nd_nb.pre_func_space.element.nlx
    )
    Q = torch.einsum(
        'mg,ng,bg->bmn',
        q,
        q,
        qdetwei
    ) / config.mu_f  # 1/nu Q as pressure preconditioner
    Q = torch.linalg.inv(Q)
    x_temp = torch.einsum('bmn,bn->bm', Q, x_p)
    x_p *= 0
    x_p += x_temp
    return x_p.view(-1)


def vel_precond_invK_mg(x_i, x_rhs, include_adv, x_k=None, u_bc=None):
    """
    given vector x_u, apply velocity block's preconditioner:
    x_u <- K^-1 x_rhs
    here K^-1 is the approximation of the inverse of K,
    The approximation is computed by a multi-grid cycle.

    This operation equals to _multigrid_one_cycle.
    """
    u_nloc = sf_nd_nb.disp_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    total_no_dofs = nele * u_nloc * ndim * 2 + nele * p_nloc

    r0 = torch.zeros(total_no_dofs, device=dev, dtype=torch.float64)
    # x_u = x_u.view(nonods, ndim)
    # x_rhs = x_rhs.view(nonods, ndim)
    r0_dict = slicing_x_i(r0)
    x_i_dict = slicing_x_i(x_i)
    x_k_dict = slicing_x_i(x_k)
    cg_nonods = sf_nd_nb.sparse_f.cg_nonods

    # pre smooth
    for its1 in range(config.pre_smooth_its):
        r0 *= 0
        r0, x_i = get_residual_and_smooth_once(
            r0, x_i, x_rhs=x_rhs,
            include_adv=include_adv, x_k=x_k, u_bc=u_bc,
            a00=True, a01=False, a10=False, a11=False,
            use_fict_dt_in_vel_precond=sf_nd_nb.use_fict_dt_in_vel_precond,
            include_itf=False
        )

    # get residual on PnDG
    r0 *= 0
    r0 = get_residual_only(r0, x_i, x_rhs=x_rhs,
                           include_adv=include_adv, x_k=x_k, u_bc=u_bc,
                           a00=True, a01=False, a10=False, a11=False,
                           use_fict_dt_in_vel_precond=sf_nd_nb.use_fict_dt_in_vel_precond,
                           include_itf=False
                           )

    # restrict residual
    if not config.is_pmg:
        r1 = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
        for idim in range(ndim):
            r1[:, idim] += torch.mv(
                sf_nd_nb.sparse_f.I_cf,
                mg_le.vel_pndg_to_p1dg_restrictor(r0_dict['vel'][0:nele_f, :, idim]).view(-1)
            )
    else:
        raise NotImplemented('PMG not implemented!')

    # smooth/solve on P1CG grid
    if config.mg_opt_F == 1:  # two-grid method
        # direct solve on P1CG
        e_i = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
        e_direct = sp.sparse.linalg.spsolve(
            sf_nd_nb.RARmat_F,
            r1.contiguous().view(-1).cpu().numpy()
        )
        e_i += torch.tensor(e_direct, device=dev, dtype=torch.float64).view(cg_nonods, ndim)
    elif config.mg_opt_F == 3:  # AMG on P1CG
        e_i = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
        e_direct = sf_nd_nb.RARmat_F.solve(
            r1.contiguous().view(-1).cpu().numpy(),
            maxiter=1,
            tol=1e-10)
        e_direct = np.reshape(e_direct, (cg_nonods, ndim))
        e_i += torch.tensor(e_direct, device=dev, dtype=torch.float64)
    else:  # multi-grid on space-filling curve generated grids
        ncurve = 1  # always use 1 sfc
        N = len(sf_nd_nb.sfc_data_F.space_filling_curve_numbering)
        inverse_numbering = np.zeros((N, ncurve), dtype=int)
        inverse_numbering[:, 0] = np.argsort(sf_nd_nb.sfc_data_F.space_filling_curve_numbering[:, 0])
        r1_sfc = r1[inverse_numbering[:, 0], :].view(cg_nonods, ndim)

        # go to SFC coarse grid levels and do 1 mg cycles there
        e_i = mg_le.mg_on_P1CG(
            r1_sfc.view(cg_nonods, ndim),
            sf_nd_nb.sfc_data_F.variables_sfc,
            sf_nd_nb.sfc_data_F.nlevel,
            sf_nd_nb.sfc_data_F.nodes_per_level,
            cg_nonods
        )
        # reverse to original order
        e_i = e_i[sf_nd_nb.sfc_data_F.space_filling_curve_numbering[:, 0] - 1, :].view(cg_nonods, ndim)

    # prolongate residual
    if not config.is_pmg:
        e_i0 = torch.zeros(nele_f * u_nloc, ndim, device=dev, dtype=torch.float64)
        for idim in range(ndim):
            e_i0[:, idim] += mg_le.vel_p1dg_to_pndg_prolongator(torch.mv(
                sf_nd_nb.sparse_f.I_fc,
                e_i[:, idim]))
    else:
        raise Exception('pmg not implemented')

    # correct find grid solution
    x_i_dict['vel'][0:nele_f, ...] += e_i0.view(nele_f, u_nloc, ndim)
    # x_u += e_i0

    # post smooth
    for its1 in range(config.pre_smooth_its):
        r0 *= 0
        r0, x_i = get_residual_and_smooth_once(
            r0, x_i, x_rhs=x_rhs,
            include_adv=include_adv, x_k=x_k, u_bc=u_bc,
            a00=True, a01=False, a10=False, a11=False,
            use_fict_dt_in_vel_precond=sf_nd_nb.use_fict_dt_in_vel_precond,
            include_itf=False
        )
    # print('x_rhs norm: ', torch.linalg.norm(x_rhs.view(-1)), 'r0 norm: ', torch.linalg.norm(r0.view(-1)))
    return x_i


def vel_precond_invK_direct(x_rhs, u_n, u_bc):
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
    # we will use ns_assemble to get K
    import ns_assemble
    u_nonods = sf_nd_nb.vel_func_space.nonods
    p_nonods = sf_nd_nb.pre_func_space.nonods
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    if type(sf_nd_nb.Kmatinv) is NoneType:
        dummy = torch.zeros(u_nonods*ndim, device=dev, dtype=torch.float64)
        # Amat_sp, _ = ns_assemble.assemble(u_bc_in=[dummy, dummy],
        #                                   f=dummy)
        dummy_u_bc = [dummy for _ in sf_nd_nb.vel_func_space.bc_node_list]
        if sf_nd_nb.indices_st is None:
            indices_st = []
            values_st = []
            rhs_np, indices_st, values_st = ns_assemble.assemble(
                u_bc_in=dummy_u_bc,
                f=dummy,
                indices=indices_st,
                values=values_st,
                use_fict_dt_in_vel_precond=sf_nd_nb.use_fict_dt_in_vel_precond
            )
            sf_nd_nb.set_data(
                indices_st=indices_st,
                values_st=values_st
            )
        else:
            indices_st = sf_nd_nb.indices_st
            values_st = sf_nd_nb.values_st
        indices_ns = []
        values_ns = []
        indices_ns += indices_st
        values_ns += values_st
        if config.include_adv:
            rhs_c, indices_ns, values_ns = ns_assemble.assemble_adv(
                u_n,
                [u_bc],
                indices_ns,
                values_ns)
        Fmat_sp = ns_assemble.assemble_csr_mat(indices_ns, values_ns)
        # rhs_all = rhs_np + rhs_c
        Fmat = Fmat_sp.todense()[0:u_nonods*ndim, 0:u_nonods*ndim]
        Fmatinv = np.linalg.inv(Fmat)
        sf_nd_nb.set_data(Kmatinv=Fmatinv)
    else:
        Fmatinv = sf_nd_nb.Kmatinv
    x_in = x_rhs.view(-1).cpu().numpy()
    x_out = np.matmul(Fmatinv, x_in)
    x_rhs *= 0
    x_rhs += torch.tensor(x_out, device=dev, dtype=torch.float64).view(x_rhs.shape)
    return x_rhs


def pre_precond_all(x_p, include_adv, u_n, u_bc):
    """
    apply pressure preconditioner
    x_p <- (- Q^-1 F_p L_p^-1) x_p

    note that this function will apply F_p^-1 and store
    the results back to x_p at same memory location.

    so it's safe to pass in and pass out the same
    tensor. (otherwise we usually need to create an
    x_temp outside)
    """
    if include_adv:
    # if True:
        # apply L_p^-1
        x_p_temp = torch.zeros_like(x_p, device=dev, dtype=torch.float64)
        x_p_temp = pressure_matrix.pre_precond_invLp(x_p_temp, x_p)
        x_p *= 0
        x_p += x_p_temp.view(x_p.shape)
        # if sf_nd_nb.Lpmatinv is None:
        #     print('we re going to find the direct inverse of pressure Laplacian')
        #     p_nonods = sf_nd_nb.pre_func_space.nonods
        #     # get Lp inv
        #     idx_lp = []
        #     val_lp = []
        #     idx_lp, val_lp = ns_assemble.pressure_laplacian_assemble(idx_lp, val_lp)
        #     Lpmat = ns_assemble.assemble_csr_mat(idx_lp, val_lp, (p_nonods, p_nonods))
        #     Lpmatinv = np.linalg.inv(Lpmat.todense())
        #     sf_nd_nb.Lpmatinv = Lpmatinv
        # else:
        #     Lpmatinv = sf_nd_nb.Lpmatinv
        # x_in = x_p.view(-1).cpu().numpy()
        # x_out = np.matmul(Lpmatinv, x_in)
        # x_p *= 0
        # x_p += torch.tensor(x_out, device=dev, dtype=torch.float64).view(x_p.shape)

        # apply -F_p  <- thisis the negative sign!
        x_p = pressure_matrix.pre_precond_Fp(x_p, u_n, u_bc)

    # apply Q^-1
    x_p = pressure_matrix.pre_precond_invQ(x_p)

    # return x_p


def vel_precond_all(x_i, x_k, u_bc, include_adv=True):
    """
    apply velocity preconditioner
    w_u <- K_u^-1 (x_u - G x_p)

    input u_bc is fluid dirichlet boundary condition (not the full u_bc list)
    """
    u_nonods = sf_nd_nb.vel_func_space.nonods
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    total_no_dof = nele * (u_nloc * ndim * 2 + p_nloc)
    x_temp = torch.zeros(total_no_dof, device=dev, dtype=torch.float64)
    x_temp_dict = slicing_x_i(x_temp)
    x_i_dict = slicing_x_i(x_i)
    x_k_dict = slicing_x_i(x_k)

    # x_u - G x_p
    x_temp_dict['vel'] += x_i_dict['vel']
    x_temp = get_residual_only(
        r0=x_temp,
        x_i=x_i,
        x_rhs=0,
        a00=False,
        a01=True,
        a10=False,
        a11=False,
        include_itf=False,
    )
    # move x_temp to x_u
    x_i_dict['vel'] *= 0
    x_i_dict['vel'] += x_temp_dict['vel']

    # left multiply K_u^-1
    x_temp *= 0
    x_temp = vel_precond_invK_mg(
        x_i=x_temp,
        x_rhs=x_i,
        include_adv=include_adv,
        x_k=x_k,
        u_bc=u_bc,
    )
    # move x_temp to x_u
    x_i_dict['vel'] *= 0
    x_i_dict['vel'] += x_temp_dict['vel']

    return x_i


# def backward_GS_precond_all(x_u, x_p, u_n, u_bc):
#     """
#     do an additional step for backward GS
#     """
#     p_nonods = sf_nd_nb.pre_func_space.nonods
#     x_temp = torch.zeros(p_nonods, device=dev, dtype=torch.float64)
#     # x_p - G^T x_u
#     x_temp = get_residual_only(
#         r0=x_temp,
#         x_i=x_u,
#         x_rhs=x_p,
#         a00=False,
#         a01=False,
#         a10=True,
#         a11=False,
#     )
#     # move x_temp to x_p
#     x_p *= 0
#     x_p += x_temp.view(x_p.shape)
#     x_temp *= 0
#
#     # apply P_s^-1
#     # apply L_p^-1
#     x_temp = pressure_matrix.pre_precond_invLp(x_temp, x_p)
#     x_p *= 0
#     x_p += x_temp.view(x_p.shape)
#     # apply -F_p  <- thisis the negative sign!
#     x_p = pressure_matrix.pre_precond_Fp(x_p, u_n, u_bc)
#     # apply Q^-1
#     x_p = pressure_matrix.pre_precond_invQ(x_p)
#     return x_p


# def vel_precond_all_only_Mass(x_u, x_p):
#     """
#     apply velocity preconditioner
#     w_u <- M_u^-1 (x_u - G x_p)
#     """
#     u_nonods = sf_nd_nb.vel_func_space.nonods
#     x_temp = torch.zeros(u_nonods, ndim, device=dev, dtype=torch.float64)
#     # x_u - G x_p
#     x_temp = get_residual_only(
#         r0=x_temp,
#         x_i=x_p,
#         x_rhs=x_u,
#         a00=False,
#         a01=True,
#         a10=False,
#         a11=False,
#     )
#
#     # left multiply M^-1 (here M is velocity space mass matrix)
#     u_nloc = sf_nd_nb.vel_func_space.element.nloc
#     x_temp = x_temp.view(nele, u_nloc, ndim)
#     v = sf_nd_nb.vel_func_space.element.n
#     _, vdetwei = get_det_nlx(
#         nlx=sf_nd_nb.vel_func_space.element.nlx,
#         x_loc=sf_nd_nb.vel_func_space.x_ref_in,
#         weight=sf_nd_nb.vel_func_space.element.weight,
#         nloc=u_nloc,
#         ngi=sf_nd_nb.vel_func_space.element.ngi
#     )
#     M = torch.einsum(
#         'mg,ng,bg->bmn',
#         v,
#         v,
#         vdetwei
#     ) / config.dt * sf_nd_nb.bdfscm.gamma  # velocity mass matrix /dt * gamma (BDF coeff)
#     M = torch.linalg.inv(M)
#     x_temp = torch.einsum('bmn,bni->bmi', M, x_temp)
#     # move x_temp to x_u
#     x_u *= 0
#     x_u += x_temp.view(x_u.shape)
#     x_temp *= 0
#
#     return x_u


# def pre_precond_all_only_Lp(x_p, include_adv):
#     """
#     apply pressure preconditioner
#     x_p <- (gamma/dt L_p)^-1 x_p
#     """
#     if include_adv:
#         # apply L_p^-1
#         x_p_temp = torch.zeros_like(x_p, device=dev, dtype=torch.float64)
#         x_p_temp = pressure_matrix.pre_precond_invLp(x_p_temp, x_p) / sf_nd_nb.bdfscm.gamma * config.dt
#         x_p *= 0
#         x_p += x_p_temp.view(x_p.shape)
#
#     return x_p


def get_RAR_and_sfc_data_Fp(
        x_k, u_bc
):
    """
    get RAR and coarse grid operators for velocity block F
    """
    I_fc = sf_nd_nb.sparse_f.I_fc
    I_cf = sf_nd_nb.sparse_f.I_cf
    whichc = sf_nd_nb.sparse_f.whichc
    ncolor = sf_nd_nb.sparse_f.ncolor
    fina = sf_nd_nb.sparse_f.fina
    cola = sf_nd_nb.sparse_f.cola
    ncola = sf_nd_nb.sparse_f.ncola
    cg_nonods = sf_nd_nb.sparse_f.cg_nonods
    if cg_nonods == 0:
        return None
    print('=== get RAR and sfc data for fluid velocity blk ===')
    # prepare for MG on SFC-coarse grids
    if config.use_fict_dt_in_vel_precond:  # change mass matrix magnitude in vel blk preconditioner
        sf_nd_nb.dt = sf_nd_nb.fict_dt
    RARvalues = calc_RAR_mf_color(
        I_fc, I_cf,
        whichc, ncolor,
        fina, cola, ncola,
        include_adv=config.include_adv,
        x_k=x_k,
        u_bc=u_bc[0]
    )
    if config.use_fict_dt_in_vel_precond:  # change back to real timestep
        sf_nd_nb.dt = config.dt
    from scipy.sparse import bsr_matrix

    if config.mg_opt_F == 1:
        RAR = bsr_matrix((RARvalues.cpu().numpy(), cola, fina),
                         shape=((ndim) * cg_nonods, (ndim) * cg_nonods))
        sf_nd_nb.set_data(RARmat_F=RAR.tocsr())
    elif config.mg_opt_F == 3:  # use pyAMG to smooth on P1CG
        RAR = bsr_matrix((RARvalues.cpu().numpy(), cola, fina),
                         shape=((ndim) * cg_nonods, (ndim) * cg_nonods))
        # RAR_ml = pyamg.ruge_stuben_solver(RAR.tocsr())
        RAR_ml = config.pyAMGsmoother(RAR.tocsr())
        sf_nd_nb.set_data(RARmat_F=RAR_ml)
    else:
        # np.savetxt('RAR.txt', RAR.toarray(), delimiter=',')
        RARvalues = torch.permute(RARvalues, (1, 2, 0)).contiguous()  # (ndim, ndim, ncola)
        # get SFC, coarse grid and operators on coarse grid. Store them to save computational time?
        space_filling_curve_numbering, variables_sfc, nlevel, nodes_per_level = \
            multigrid_linearelastic.mg_on_P1CG_prep(fina, cola, RARvalues, sparse_in=sf_nd_nb.sparse_f)
        sf_nd_nb.sfc_data_F.set_data(
            space_filling_curve_numbering=space_filling_curve_numbering,
            variables_sfc=variables_sfc,
            nlevel=nlevel,
            nodes_per_level=nodes_per_level
        )
