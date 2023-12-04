"""matrix-free integral for mesh velocity
or mesh displacement"""

import torch
import numpy as np
import scipy as sp
import pyamg
import config
from config import sf_nd_nb
import multigrid_linearelastic as mg
from tqdm import tqdm

if config.ndim == 2:
    from shape_function import get_det_nlx as get_det_nlx
    from shape_function import sdet_snlx as sdet_snlx
else:
    from shape_function import get_det_nlx_3d as get_det_nlx
    from shape_function import sdet_snlx_3d as sdet_snlx

nele = config.nele
nele_f = config.nele_f
nele_s = config.nele_s
ndim = config.ndim
dev = config.dev
nface = ndim + 1


def solve_for_diff(
        x_i, f, u_bc, alpha_x_i,
        t=None,
):
    """
    input: current non-linear step solution

    from the solid displacement at interface,
    solve a diffusion equation for the mesh
    displacement in the fluid subdomain

    then do a projection to make it C1 continuous

    x_i contains all displacement field (fluid + solid)
    we will separate them into x_i fluid subdomain to solve;
    and x_i solid subdomain as boundary condition
    """
    shape_in = x_i.shape
    x_i = _solve_diffusion(x_i, f, u_bc, alpha_x_i)
    # x_i = _project_to_make_continuous(x_i)
    # x_i = _make_0_bc_strongly_enforced(x_i)
    x_i = x_i.view(shape_in)
    return x_i


def _solve_diffusion(
        x_i, f, u_bc, alpha_x_i
):
    x_i = x_i.view(nele, -1)
    x_rhs = torch.zeros_like(x_i, device=dev, dtype=torch.float64)
    # get RAR and SFC data
    _get_RAR_and_sfc_data_Um()
    # get rhs
    x_rhs = _get_rhs(x_rhs, f, u_bc, alpha_x_i)

    # solve with left-preconditionerd GMRES
    x_i, its = _gmres_mg_solver(
        x_i, x_rhs, tol=config.tol
    )

    return x_i


def _get_rhs(
        x_rhs, f, u_bc, alpha_x_i
):
    """
    get rhs of the diffusion equation (interface
    displacement on solid side as a dirichlet bc)

    input:

    x_i has length (nele * u_nloc * ndim)
    i.e. it contains solid displacement

    x_rhs has length (nele_f * u_nloc * ndim)
    i.e. only in fluid displacement
    """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele_f, dtype=int)
    idx_in = torch.zeros(nele, dtype=torch.bool)
    idx_in_f = torch.zeros(nele * nface, dtype=torch.bool, device=dev)

    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    for bc in u_bc:
        bc = bc.view(nele, u_nloc)

    x_rhs_inshape = x_rhs.shape
    x_rhs = x_rhs.view(nele, u_nloc)

    for i in range(nnn):
        idx_in *= False
        # volume integral
        idx_in[brk_pnt[i]:brk_pnt[i + 1]] = True
        x_rhs = _k_rhs_one_batch(x_rhs, alpha_x_i, f, idx_in)
        # surface integral
        idx_in_f *= False
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        x_rhs = _s_rhs_one_batch(x_rhs, u_bc, idx_in_f)

    x_rhs = x_rhs.view(x_rhs_inshape)
    return x_rhs


def _k_rhs_one_batch(x_rhs, alpha_x_i, f, idx_in):
    """get the volume integral part of rhs"""
    batch_in = int(torch.sum(idx_in))  # only fluid subdomain
    if batch_in < 1:  # nothing to do here.
        return x_rhs
    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    f = f.view(nele, u_nloc)

    # get shape functions
    n = sf_nd_nb.vel_func_space.element.n
    nx, ndetwei = get_det_nlx(
        nlx=sf_nd_nb.vel_func_space.element.nlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[idx_in],
        weight=sf_nd_nb.vel_func_space.element.weight,
        nloc=u_nloc,
        ngi=sf_nd_nb.vel_func_space.element.ngi
    )

    # f . v contribution to vel rhs
    x_rhs[idx_in, ...] += torch.einsum(
        'mg,ng,bg,bn->bm',
        n,  # (u_nloc, ngi)
        n,  # (u_nloc, ngi)
        ndetwei,  # (batch_in, ngi)
        f[idx_in, ...],  # (batch_in, u_nloc, ndim)
    )

    # if transient, add rho/dt * u_n to vel rhs
    if sf_nd_nb.isTransient:
        alpha_x_i = alpha_x_i.view(nele, u_nloc)
        x_rhs[idx_in, ...] += torch.einsum(
            'mg,ng,bg,bn->bm',
            n,
            n,
            ndetwei,
            alpha_x_i[idx_in, ...],
        ) / sf_nd_nb.dt
    return x_rhs


def _s_rhs_one_batch(
        x_rhs, u_bc, idx_in_f
):
    # get essential data
    nbf = sf_nd_nb.vel_func_space.nbf
    glb_bcface_type = sf_nd_nb.vel_func_space.glb_bcface_type

    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc

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

    # create two lists of which element f_i / f_b is in
    E_F_i = torch.floor_divide(F_i, nface)
    E_F_inb = torch.floor_divide(F_inb, nface)
    E_F_b_d = torch.floor_divide(F_b_d, nface)
    E_F_b_n = torch.floor_divide(F_b_n, nface)

    # local face number
    f_b_d = torch.remainder(F_b_d, nface)
    f_b_n = torch.remainder(F_b_n, nface)
    f_i = torch.remainder(F_i, nface)
    f_inb = torch.remainder(F_inb, nface)

    # nothing to do in interior face, pass...

    # boundary face
    for iface in range(nface):
        idx_iface_d = f_b_d == iface
        idx_iface_n = f_b_n == iface
        # dirichlet boundary
        x_rhs = _s_rhs_fd(
            x_rhs,
            f_b_d[idx_iface_d], E_F_b_d[idx_iface_d],
            u_bc[0]
        )  # rhs terms from interface
        if idx_iface_n.sum() > 0:
            # neumann boundary
            x_rhs = _s_rhs_fb(
                x_rhs,
                f_b_n[idx_iface_n], E_F_b_n[idx_iface_n],
                u_bc[1]
            )
    return x_rhs


def _s_rhs_fd(
        rhs,
        f_b, E_F_b,
        u_bc
):
    batch_in = f_b.shape[0]
    if batch_in < 1:  # nothing to do here.
        return rhs
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)

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
    snx = snx[dummy_idx, f_b, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_b, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_b, ...]  # (batch_in, ndim, sngi)
    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = config.eta_e / h

    u_bc_th = u_bc[E_F_b, ...]  # (batch_in, u_nloc, ndim)

    # {dv_i / dx_j} [u_Si n_Sj]
    rhs[E_F_b, ...] -= torch.einsum(
        'bjmg,bng,bjg,bg,bn->bm',
        snx,  # (batch_in, ndim, u_nloc, sngi)
        sn,  # (batch_in, u_nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
        u_bc_th,  # (batch_in, u_nloc, ndim)
    ) * config.mu_f

    # \gamma_e [u_Di] [v_i]
    rhs[E_F_b, ...] += torch.einsum(
        'b,bmg,bng,bg,bn->bm',
        gamma_e,  # (batch_in)
        sn,  # (batch_in, u_nloc, sngi)
        sn,
        sdetwei,  # (batch_in, sngi)
        u_bc_th,  # (batch_in, u_nloc, ndim)
    ) * config.mu_f

    return rhs


def _s_rhs_fb(
        x_rhs,
        f_b_n, E_F_b_n,
        u_bc
):
    """contributions to rhs of neumann boundary condition"""
    batch_in = f_b_n.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    if batch_in < 1:  # nothing to do here.
        return x_rhs

    # shape function
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_b_n],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi,
        sn=sf_nd_nb.vel_func_space.element.sn,
    )
    sn = sf_nd_nb.vel_func_space.element.sn[f_b_n, ...]  # (batch_in, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_b_n, ...]  # (batch_in, sngi)
    u_bc_th = u_bc[E_F_b_n, ...]

    # 3. TODO: Neumann BC
    x_rhs[E_F_b_n, ...] += torch.einsum(
        'bmg,bng,bg,bn->bm',
        sn,  # (batch_in, nloc, sngi)
        sn,  # (batch_in, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        u_bc_th,  # (batch_in, u_nloc, ndim)
    )
    return x_rhs


def get_residual_or_smooth(
        r0, x_i, x_rhs,
        do_smooth=False,
):
    """
    matrix free, integral, for terms in the diffusion equation
    """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele_f, dtype=int)
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    r0 = r0.view(nele, u_nloc)
    if type(x_rhs) is int:
        r0 += x_rhs
    else:
        x_rhs = x_rhs.view(nele, u_nloc)
        r0 += x_rhs
    for i in range(nnn):
        # volume integral
        idx_in = torch.zeros(nele, device=dev, dtype=torch.bool)  # element indices in this batch
        idx_in[brk_pnt[i]:brk_pnt[i + 1]] = True
        batch_in = int(torch.sum(idx_in))
        # dummy diagA and bdiagA
        diagK = torch.zeros(batch_in, u_nloc, device=dev, dtype=torch.float64)
        bdiagK = torch.zeros(batch_in, u_nloc, u_nloc, device=dev, dtype=torch.float64)
        r0, diagK, bdiagK = _k_res_one_batch(
            r0, x_i,
            diagK, bdiagK,
            idx_in
        )
        # surface integral
        idx_in_f = torch.zeros(nele * nface, dtype=torch.bool, device=dev)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0, diagK, bdiagK = _s_res_one_batch(
            r0, x_i,
            diagK, bdiagK,
            idx_in_f, brk_pnt[i]
        )
        if do_smooth:
            # smooth once
            bdiagK = torch.inverse(bdiagK.view(batch_in, u_nloc, u_nloc))
            x_i = x_i.view(nele, u_nloc)
            x_i[idx_in, :] += config.jac_wei * torch.einsum(
                '...ij,...j->...i',
                bdiagK,
                r0.view(nele, u_nloc)[idx_in, :]
            ).view(batch_in, u_nloc)
    r0 = r0.view(-1)
    x_i = x_i.view(-1)
    return r0, x_i


def _k_res_one_batch(
        r0, x_i,
        diagK, bdiagK,
        idx_in,
):
    batch_in = diagK.shape[0]
    u_nloc = sf_nd_nb.vel_func_space.element.nloc

    r0 = r0.view(nele, u_nloc)
    x_i = x_i.view(nele, u_nloc)
    diagK = diagK.view(-1, u_nloc)
    bdiagK = bdiagK.view(-1, u_nloc, u_nloc)

    # get shape function and derivatives
    n = sf_nd_nb.vel_func_space.element.n
    nx, ndetwei = get_det_nlx(
        nlx=sf_nd_nb.vel_func_space.element.nlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[idx_in],
        weight=sf_nd_nb.vel_func_space.element.weight,
        nloc=u_nloc,
        ngi=sf_nd_nb.vel_func_space.element.ngi
    )

    K = torch.zeros(batch_in, u_nloc, u_nloc, device=dev, dtype=torch.float64)
    K += torch.einsum(
        'bimg,bing,bg->bmn', nx, nx, ndetwei,
    ) * config.mu_f
    r0[idx_in, ...] -= torch.einsum(
        'bmn,bn->bm',
        K, x_i[idx_in, ...]
    )
    # get diagonal of velocity block K
    diagK += torch.diagonal(K.view(batch_in, u_nloc, u_nloc)
                            , dim1=1, dim2=2).view(batch_in, u_nloc)
    bdiagK[idx_in[0:nele_f], ...] += K

    return r0, diagK, bdiagK


def _s_res_one_batch(
        r0, x_i,
        diagK, bdiagK,
        idx_in_f,
        batch_start_idx
):
    """
    surface integral (left hand side)
    """
    nbf = sf_nd_nb.vel_func_space.nbf
    glb_bcface_type = sf_nd_nb.vel_func_space.glb_bcface_type

    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc

    # separate nbf to get internal face list and boundary face list
    F_i = torch.where(torch.logical_and(glb_bcface_type < 0,
                                        idx_in_f))[0]  # interior face of fluid subdomain
    F_inb = nbf[F_i]  # neighbour list of interior face
    F_inb = F_inb.type(torch.int64)
    F_b = torch.where(torch.logical_and(
        sf_nd_nb.vel_func_space.glb_bcface_type == 0,
        idx_in_f))[0]  # dirichlet boundary faces

    # create two lists of which element f_i / f_b is in
    E_F_i = torch.floor_divide(F_i, nface)
    E_F_inb = torch.floor_divide(F_inb, nface)
    E_F_b = torch.floor_divide(F_b, nface)

    # local face number
    f_b = torch.remainder(F_b, nface)
    f_i = torch.remainder(F_i, nface)
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
                nb_gi_aln,
            )
    # r0, diagK, bdiagK = _s_res_fi_all_face(
    #     r0, diagK, bdiagK,
    #     f_i, E_F_i,
    #     f_inb, E_F_inb,
    #     x_i,
    # )
    # boundary faces (dirichlet)
    for iface in range(nface):
        idx_iface = f_b == iface
        r0, diagK, bdiagK = _s_res_fb(
            r0, f_b[idx_iface], E_F_b[idx_iface],
            x_i,
            diagK, bdiagK, batch_start_idx,
        )
    return r0, diagK, bdiagK


def _s_res_fi(
        r0, f_i, E_F_i,
        f_inb, E_F_inb,
        x_i,
        diagK, bdiagK, batch_start_idx,
        nb_gi_aln,
):
    """internal faces"""
    batch_in = f_i.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    # get element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    r0 = r0.view(nele, u_nloc)
    x_i = x_i.view(nele, u_nloc)

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

    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = config.eta_e / h

    u_ith = x_i[E_F_i, ...]
    u_inb = x_i[E_F_inb, ...]

    # K block
    K = torch.zeros(batch_in, u_nloc, u_nloc, device=dev, dtype=torch.float64)
    # this side
    # [v_i n_j] {du_i / dx_j}  consistent term
    K += torch.einsum(
        'bmg,bjg,bjng,bg->bmn',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim)
    # {dv_i / dx_j} [u_i n_j]  symmetry term
    K += torch.einsum(
        'bjmg,bng,bjg,bg->bmn',
        snx,  # (batch_in, ndim, nloc, sngi)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim) \
    # \gamma_e * [v_i][u_i]  penalty term
    K += torch.einsum(
        'bmg,bng,bg,b->bmn',
        sn,  # (batch_in, nloc, sngi)
        sn,  # (batch_in, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        gamma_e,  # (batch_in)
    )
    K *= config.mu_f

    # update residual
    r0[E_F_i, ...] -= torch.einsum('bmn,bn->bm', K, u_ith)
    # put diagonal into diagK and bdiagK
    diagK[E_F_i - batch_start_idx, ...] += torch.diagonal(K.view(batch_in, u_nloc, u_nloc),
                                                          dim1=1, dim2=2).view(batch_in, u_nloc)
    bdiagK[E_F_i - batch_start_idx, ...] += K

    # other side
    K *= 0
    # [v_i n_j] {du_i / dx_j}  consistent term
    K += torch.einsum(
        'bmg,bjg,bjng,bg->bmn',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        snx_nb,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim) \
    # {dv_i / dx_j} [u_i n_j]  symmetry term
    K += torch.einsum(
        'bjmg,bng,bjg,bg->bmn',
        snx,  # (batch_in, ndim, nloc, sngi)
        sn_nb,  # (batch_in, nloc, sngi)
        snormal_nb,  # (batch_in, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim) \
    # \gamma_e * [v_i][u_i]  penalty term
    K += torch.einsum(
        'bmg,bng,bg,b->bmn',
        sn,  # (batch_in, nloc, sngi)
        sn_nb,  # (batch_in, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        gamma_e,  # (batch_in)
    ) * (-1.)  # because n2 \cdot n1 = -1
    K *= config.mu_f

    # update residual
    r0[E_F_i, ...] -= torch.einsum('bmn,bn->bm', K, u_inb)
    return r0, diagK, bdiagK


def _s_res_fi_all_face(
        r0, diagK, bdiagK,
        f_i, E_F_i,
        f_inb, E_F_inb,
        x_i,
):
    """internal faces"""
    batch_in = f_i.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    # get element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    r0 = r0.view(nele, u_nloc)
    x_i = x_i.view(nele, u_nloc)

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
    snx = snx[E_F_i, f_i, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[E_F_i, f_i, ...]  # (batch_in, sngi)
    snormal = snormal[E_F_i, f_i, ...]  # (batch_in, ndim, sngi)

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
    snx_nb = snx_nb[E_F_inb, f_inb, ...]  # (batch_in, ndim, nloc, sngi)
    snormal_nb = snormal_nb[E_F_inb, f_inb, ...]  # (batch_in, ndim, sngi)
    # change gaussian points order on other side
    # ===== old ======
    # nb_aln = sf_nd_nb.vel_func_space.element.gi_align[nb_gi_aln, :]  # nb_aln for velocity element
    # snx_nb = snx_nb[..., nb_aln]
    # snormal_nb = snormal_nb[..., nb_aln]
    # # don't forget to change gaussian points order on sn_nb!
    # sn_nb = sn_nb[..., nb_aln]
    # ===== new ======
    for nb_gi_aln in range(ndim):  # 'ndim' alignnment of GI points on neighbour faces
        idx = sf_nd_nb.vel_func_space.alnmt[E_F_i * nface + f_i] == nb_gi_aln
        nb_aln = sf_nd_nb.vel_func_space.element.gi_align[nb_gi_aln, :]
        snx_nb[idx, ...] = snx_nb[idx][..., nb_aln]
        snormal_nb[idx, ...] = snormal_nb[idx][..., nb_aln]
        sn_nb[idx, ...] = sn_nb[idx][..., nb_aln]

    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = config.eta_e / h

    u_ith = x_i[E_F_i, ...]
    u_inb = x_i[E_F_inb, ...]

    # # K block
    # K = torch.zeros(batch_in, u_nloc, u_nloc, device=dev, dtype=torch.float64)
    # # this side
    # # [v_i n_j] {du_i / dx_j}  consistent term
    # K += torch.einsum(
    #     'bmg,bjg,bjng,bg->bmn',
    #     sn,  # (batch_in, nloc, sngi)
    #     snormal,  # (batch_in, ndim, sngi)
    #     snx,  # (batch_in, ndim, nloc, sngi)
    #     sdetwei,  # (batch_in, sngi)
    # ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim)
    # # {dv_i / dx_j} [u_i n_j]  symmetry term
    # K += torch.einsum(
    #     'bjmg,bng,bjg,bg->bmn',
    #     snx,  # (batch_in, ndim, nloc, sngi)
    #     sn,  # (batch_in, nloc, sngi)
    #     snormal,  # (batch_in, ndim, sngi)
    #     sdetwei,  # (batch_in, sngi)
    # ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim) \
    # # \gamma_e * [v_i][u_i]  penalty term
    # K += torch.einsum(
    #     'bmg,bng,bg,b->bmn',
    #     sn,  # (batch_in, nloc, sngi)
    #     sn,  # (batch_in, nloc, sngi)
    #     sdetwei,  # (batch_in, sngi)
    #     gamma_e,  # (batch_in)
    # )
    # K *= config.mu_f

    # assume we don't need to get block diagonal
    # K block
    Au = torch.zeros(batch_in, u_nloc, device=dev, dtype=torch.float64)
    # this side
    # [v_i n_j] {du_i / dx_j}  consistent term
    Au += torch.einsum(
        'bmg,bjg,bjng,bg,bn->bm',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        u_ith,  # (batch_in, u_nloc)
    ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim)
    # {dv_i / dx_j} [u_i n_j]  symmetry term
    Au += torch.einsum(
        'bjmg,bng,bjg,bg,bn->bm',
        snx,  # (batch_in, ndim, nloc, sngi)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
        u_ith,  # (batch_in, u_nloc)
    ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim) \
    # \gamma_e * [v_i][u_i]  penalty term
    Au += torch.einsum(
        'bmg,bng,bg,b,bn->bm',
        sn,  # (batch_in, nloc, sngi)
        sn,  # (batch_in, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        gamma_e,  # (batch_in)
        u_ith,  # (batch_in, u_nloc)
    )
    Au *= config.mu_f

    # update residual
    # Au = torch.einsum('bmn,bn->bm', K, u_ith)
    # # put diagonal into diagK and bdiagK
    # diagS = torch.diagonal(K.view(batch_in, u_nloc, u_nloc),
    #                        dim1=1, dim2=2).view(batch_in, u_nloc)
    # put them to r0, diagK and bdiagK (SCATTER)
    for iface in range(nface):
        idx_iface = (f_i == iface)
        r0[E_F_i[idx_iface], ...] -= Au[idx_iface, ...]
        # diagK[E_F_i[idx_iface], ...] += diagS[idx_iface, ...]
        # bdiagK[E_F_i[idx_iface], ...] += K[idx_iface, ...]

    # other side
    Au *= 0
    # [v_i n_j] {du_i / dx_j}  consistent term
    Au += torch.einsum(
        'bmg,bjg,bjng,bg,bn->bm',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        snx_nb,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        u_inb,  # (batch_in, u_nloc)
    ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim) \
    # {dv_i / dx_j} [u_i n_j]  symmetry term
    Au += torch.einsum(
        'bjmg,bng,bjg,bg,bn->bm',
        snx,  # (batch_in, ndim, nloc, sngi)
        sn_nb,  # (batch_in, nloc, sngi)
        snormal_nb,  # (batch_in, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
        u_inb,  # (batch_in, u_nloc)
    ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim) \
    # \gamma_e * [v_i][u_i]  penalty term
    Au += torch.einsum(
        'bmg,bng,bg,b,bn->bm',
        sn,  # (batch_in, nloc, sngi)
        sn_nb,  # (batch_in, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        gamma_e,  # (batch_in)
        u_inb,  # (batch_in, u_nloc)
    ) * (-1.)  # because n2 \cdot n1 = -1
    Au *= config.mu_f

    # update residual
    # scatter
    for iface in range(nface):
        idx_iface = (f_i == iface)
        r0[E_F_inb[idx_iface], ...] -= Au[idx_iface, ...]
    return r0, diagK, bdiagK


def _s_res_fb(
        r0, f_b, E_F_b,
        x_i,
        diagK, bdiagK,
        batch_start_idx,
):
    """boundary faces"""
    batch_in = f_b.shape[0]
    if batch_in < 1:  # nothing to do here.
        return r0, diagK, bdiagK
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    # get element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    x_i = x_i.view(nele, u_nloc)
    r0 = r0.view(nele, u_nloc)
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
    snx = snx[dummy_idx, f_b, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_b, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_b, ...]  # (batch_in, ndim, sngi)
    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = config.eta_e / h

    u_ith = x_i[E_F_b, ...]
    # block K
    K = torch.zeros(batch_in, u_nloc, u_nloc,
                    device=dev, dtype=torch.float64)
    # [vi nj] {du_i / dx_j}  consistent term
    K -= torch.einsum(
        'bmg,bjg,bjng,bg->bmn',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    )  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim)
    # {dv_i / dx_j} [ui nj]  symmetry term
    K -= torch.einsum(
        'bjmg,bng,bjg,bg->bmn',
        snx,  # (batch_in, ndim, nloc, sngi)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
    )  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim)
    # \gamma_e [v_i] [u_i]  penalty term
    K += torch.einsum(
        'bmg,bng,bg,b->bmn',
        sn,  # (batch_in, nloc, sngi)
        sn,  # (batch_in, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        gamma_e,  # (batch_in)
    )
    K *= config.mu_f

    # update residual
    r0[E_F_b, ...] -= torch.einsum('bmn,bn->bm', K, u_ith)
    # put in diagonal
    diagK[E_F_b - batch_start_idx, ...] += torch.diagonal(K.view(batch_in, u_nloc, u_nloc),
                                                          dim1=-2, dim2=-1).view(batch_in, u_nloc)
    bdiagK[E_F_b - batch_start_idx, ...] += K
    return r0, diagK, bdiagK


def _get_RAR_and_sfc_data_Um():
    """
    get RAR and coarser grid operator for mesh displacement
    """
    print('=== get RAR and sfc_data for diffusion ===')
    I_fc = sf_nd_nb.sparse_f.I_fc
    I_cf = sf_nd_nb.sparse_f.I_cf
    whichc = sf_nd_nb.sparse_f.whichc
    ncolor = sf_nd_nb.sparse_f.ncolor
    fina = sf_nd_nb.sparse_f.fina
    cola = sf_nd_nb.sparse_f.cola
    ncola = sf_nd_nb.sparse_f.ncola
    cg_nonods = sf_nd_nb.sparse_f.cg_nonods

    RARvalues = _calc_RAR_mf_color(
        I_fc, I_cf,
        whichc, ncolor,
        sf_nd_nb.sparse_f.spIdx_for_color,
        sf_nd_nb.sparse_f.colIdx_for_color,
        fina, cola, ncola,
    )
    from scipy.sparse import csr_matrix

    if not config.is_sfc:
        RAR = csr_matrix((RARvalues.cpu().numpy(), cola, fina),
                         shape=(cg_nonods, cg_nonods))
        if not config.is_amg:  # directly solve on P1CG
            sf_nd_nb.set_data(RARmat_Um=RAR.tocsr())
        else:  # use pyamg to smooth on P1CG, setup AMG here
            # RAR_ml = pyamg.ruge_stuben_solver(RAR.tocsr())
            RAR_ml = pyamg.smoothed_aggregation_solver(RAR.tocsr())
            sf_nd_nb.set_data(RARmat_Um=RAR_ml)
    else:
        # RARvalues = torch.permute(RARvalues, (1, 2, 0)).contiguous()  # (ndim, ndim, ncola)
        # get SFC, coarse grid and operators on coarse grid. Store them to save computational time?
        space_filling_curve_numbering, variables_sfc, nlevel, nodes_per_level = \
            mg.mg_on_P1CG_prep(fina, cola, RARvalues, sparse_in=sf_nd_nb.sparse_f)
        sf_nd_nb.sfc_data_Um.set_data(
            space_filling_curve_numbering=space_filling_curve_numbering,
            variables_sfc=variables_sfc,
            nlevel=nlevel,
            nodes_per_level=nodes_per_level
        )

    return 0


def _calc_RAR_mf_color(
        I_fc, I_cf,
        whichc, ncolor,
        spIdx_for_color, colIdx_for_color,
        fina, cola, ncola,
):
    """
    get operator on P1CG grid, i.e. RAR
    where R is prolongator/restrictor
    via coloring method.
    """
    cg_nonods = sf_nd_nb.sparse_f.cg_nonods
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    nonods = nele * u_nloc

    value = torch.zeros(ncola, device=dev, dtype=torch.float64)  # NNZ entry values
    dummy = torch.zeros(nonods, device=dev, dtype=torch.float64)  # dummy variable of same length as PnDG
    Rm = torch.zeros_like(dummy, device=dev, dtype=torch.float64)
    ARm = torch.zeros_like(dummy, device=dev, dtype=torch.float64)
    RARm = torch.zeros(cg_nonods, device=dev, dtype=torch.float64)
    mask = torch.zeros(cg_nonods, device=dev, dtype=torch.float64)  # color vec
    for color in tqdm(range(1, ncolor + 1), disable=config.disabletqdm):
        # print('color: ', color)

        mask *= 0
        mask += torch.tensor((whichc == color),
                             device=dev,
                             dtype=torch.float64)  # 1 if true; 0 if false
        Rm *= 0
        Rm += \
            (mg.vel_p1dg_to_pndg_prolongator(
                torch.mv(I_fc, mask)
            ))  # (p3dg_nonods)
        ARm *= 0
        ARm, _ = get_residual_or_smooth(
            r0=ARm,
            x_i=Rm,
            x_rhs=dummy,
            do_smooth=False,
        )
        ARm *= -1.  # (p3dg_nonods)
        RARm *= 0
        RARm += torch.mv(
            I_cf,
            mg.vel_pndg_to_p1dg_restrictor(ARm)
        )  # (cg_nonods)

        # add to value
        # for i in range(RARm.shape[0]):
        #     # for count in range(fina[i], fina[i + 1]):
        #     #     j = cola[count]
        #     #     value[count] += RARm[i] * mask[j]
        #     count = np.arange(fina[i], fina[i+1])
        #     j = cola[count]
        #     value[count] += RARm[i] * mask[j]
        value[spIdx_for_color[color - 1]] += \
            RARm[colIdx_for_color[color - 1]]
        # print('finishing (another) one color, time comsumed: ', time.time() - start_time)
    return value


def _gmres_mg_solver(
        x_i, x_rhs, tol
):
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    total_nonods = nele * u_nloc
    real_nonods = nele_f * u_nloc

    m = config.gmres_m  # TODO: maybe we can use smaller number for this
    v_m = torch.zeros(m + 1, real_nonods, device=dev, dtype=torch.float64)
    v_m_j = torch.zeros(total_nonods, device=dev, dtype=torch.float64)
    h_m = torch.zeros(m + 1, m, device=dev, dtype=torch.float64)
    r0 = torch.zeros(total_nonods, device=dev, dtype=torch.float64)

    x_dummy = torch.zeros_like(r0, device=dev, dtype=torch.float64)

    r0l2 = 1.
    sf_nd_nb.its = 0
    e_1 = torch.zeros(m + 1, device=dev, dtype=torch.float64)
    e_1[0] += 1

    while r0l2 > tol and sf_nd_nb.its < config.gmres_its:  # TODO: maybe we can use smaller number for this
        h_m *= 0
        v_m *= 0
        r0 *= 0
        # get residual
        r0, _ = get_residual_or_smooth(
            r0, x_i, x_rhs,
            do_smooth=False)
        # apply left preconditioner
        x_dummy *= 0
        x_dummy = _um_left_precond(x_dummy, r0)
        r0 *= 0
        r0 += x_dummy.view(r0.shape)

        beta = torch.linalg.norm(r0.view(-1))
        r0 = r0.view(nele, u_nloc)
        v_m[0, :] += r0[0:nele_f, :].view(-1) / beta
        w = r0  # this should place w in the same memory as r0 so that we don't take two nonods memory space
        for j in tqdm(range(0, m), disable=config.disabletqdm):
            w *= 0
            # w = A * v_m[j]
            v_m_j *= 0
            v_m_j = v_m_j.view(nele, u_nloc)
            v_m_j[0:nele_f, :] += v_m[j, :].view(nele_f, u_nloc)
            w, _ = get_residual_or_smooth(
                r0=w,
                x_i=v_m_j,
                x_rhs=0,
                do_smooth=False)
            w *= -1.  # providing rhs=0, b-Ax is -Ax
            # apply preconditioner
            x_dummy *= 0
            x_dummy = _um_left_precond(x_dummy, w)
            w *= 0
            w += x_dummy.view(w.shape)
            w = w.view(nele, u_nloc)
            for i in range(0, j + 1):
                h_m[i, j] = torch.linalg.vecdot(w[0:nele_f, :].view(-1),
                                                v_m[i, :])
                w[0:nele_f, :] -= (h_m[i, j] * v_m[i, :]).view(nele_f, u_nloc)

            h_m[j + 1, j] = torch.linalg.norm(w[0:nele_f, :].view(-1))
            v_m[j + 1, :] += w[0:nele_f, :].view(-1) / h_m[j + 1, j]
            sf_nd_nb.its += 1
        # solve least-square problem
        q, r = torch.linalg.qr(h_m, mode='complete')  # h_m: (m+1)xm, q: (m+1)x(m+1), r: (m+1)xm
        e_1[0] = 0
        e_1[0] += beta
        y_m = torch.linalg.solve(r[0:m, 0:m], q[0:m + 1, 0:m].T @ e_1)  # y_m: m
        # update c_i and get residual
        dx_i = torch.einsum('ji,j->i', v_m[0:m, :], y_m)
        x_i = x_i.view(nele, u_nloc)
        x_i[0:nele_f, :] += dx_i.view(nele_f, u_nloc)

        # r0l2 = torch.linalg.norm(q[:, m:m+1].T @ e_1)
        r0 *= 0
        # get residual
        r0, _ = get_residual_or_smooth(
            r0, x_i, x_rhs,
            do_smooth=False)
        r0 = r0.view(nele, u_nloc)
        r0l2 = torch.linalg.norm(r0[0:nele_f, :].view(-1))
        print('its=', sf_nd_nb.its, 'fine grid rel residual l2 norm=', r0l2.cpu().numpy())
    return x_i, sf_nd_nb.its


def _um_left_precond(x_i, x_rhs):
    """
    do one multi-grid V cycle as left preconditioner
    """
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    total_no_dofs = nele * u_nloc
    real_no_dofs = nele_f * u_nloc

    r0 = torch.zeros(nele, u_nloc, device=dev, dtype=torch.float64)

    cg_nonods = sf_nd_nb.sparse_f.cg_nonods

    # pre smooth
    for its1 in range(config.pre_smooth_its):
        r0 *= 0
        r0, x_i = get_residual_or_smooth(
            r0, x_i, x_rhs,
            do_smooth=True
        )
    # get residual on PnDG
    r0 *= 0
    r0, _ = get_residual_or_smooth(
        r0, x_i, x_rhs,
        do_smooth=False
    )

    if not config.is_pmg:  # PnDG to P1CG
        r1 = torch.zeros(cg_nonods, device=dev, dtype=torch.float64)
        r0 = r0.view(nele, u_nloc)
        r1 += torch.mv(
            sf_nd_nb.sparse_f.I_cf,
            mg.vel_pndg_to_p1dg_restrictor(r0[0:nele_f, :]).view(-1)
            )
    else:  # PnDG down one order each time, eventually go to P1CG
        raise NotImplementedError('pmg not implemented!')

    if not config.is_sfc:  # two-grid method
        if not config.is_amg:  # directly solve on P1CG
            e_i = torch.zeros(cg_nonods, device=dev, dtype=torch.float64)
            e_direct = sp.sparse.linalg.spsolve(
                sf_nd_nb.RARmat_Um,
                r1.contiguous().view(-1).cpu().numpy())
            e_direct = np.reshape(e_direct, (cg_nonods))
            e_i += torch.tensor(e_direct, device=dev, dtype=torch.float64)
        else:  # use pyamg to smooth on P1CG
            e_i = torch.zeros(cg_nonods, device=dev, dtype=torch.float64)
            e_direct = sf_nd_nb.RARmat_Um.solve(
                r1.contiguous().view(-1).cpu().numpy(),
                maxiter=1,
                tol=1e-10)
            # e_direct = np.reshape(e_direct, (cg_nonods))
            e_i += torch.tensor(e_direct, device=dev, dtype=torch.float64)
    else:  # multi-grid method
        ncurve = 1  # always use 1 sfc
        N = len(sf_nd_nb.sfc_data_Um.space_filling_curve_numbering)
        inverse_numbering = np.zeros((N, ncurve), dtype=int)
        inverse_numbering[:, 0] = np.argsort(sf_nd_nb.sfc_data_Um.space_filling_curve_numbering[:, 0])
        r1_sfc = r1[inverse_numbering[:, 0]].view(cg_nonods)

        # go to SFC coarse grid levels and do 1 mg cycles there
        e_i = mg.mg_on_P1CG(
            r1_sfc.view(cg_nonods),
            sf_nd_nb.sfc_data_Um.variables_sfc,
            sf_nd_nb.sfc_data_Um.nlevel,
            sf_nd_nb.sfc_data_Um.nodes_per_level,
            cg_nonods
        )
        # reverse to original order
        e_i = e_i[sf_nd_nb.sfc_data_Um.space_filling_curve_numbering[:, 0] - 1].view(cg_nonods)
    if not config.is_pmg:  # from P1CG to P3DG
        # prolongate error to fine grid
        e_i0 = torch.zeros(nele_f * u_nloc, device=dev, dtype=torch.float64)
        e_i0 += mg.vel_p1dg_to_pndg_prolongator(torch.mv(
            sf_nd_nb.sparse_f.I_fc,
            e_i))
    else:  # from P1CG to P1DG, then go one order up each time while also do post smoothing
        raise Exception('pmg not implemented')
    # correct fine grid solution
    x_i = x_i.view(nele, u_nloc)
    x_i[0:nele_f, ...] += e_i0.view(nele_f, u_nloc)

    # post smooth
    for its1 in range(config.pre_smooth_its):
        r0 *= 0
        r0, x_i = get_residual_or_smooth(
            r0, x_i, x_rhs,
            do_smooth=True
        )
    # r0l2 = torch.linalg.norm(r0.view(-1), dim=0) / r0_init  # fNorm

    return x_i


def get_l2_error(x_num, x_ana):
    """ get l2 norm and inf norm lf error"""
    n = sf_nd_nb.vel_func_space.element.n
    _, detwei = get_det_nlx(
        nlx=sf_nd_nb.vel_func_space.element.nlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in,
        weight=sf_nd_nb.vel_func_space.element.weight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        ngi=sf_nd_nb.vel_func_space.element.ngi
    )
    u_i_gi = torch.einsum('ng,bn->bng', n, x_num.view(nele, -1))
    u_ana_gi = torch.einsum('ng,bn->bng', n, x_ana.view(nele, -1))
    u_l2 = torch.einsum(
        'bng,bg->bn',
        (u_i_gi - u_ana_gi) ** 2,
        detwei,
    )
    u_l2 = torch.sum(torch.sum(u_l2, dim=1), dim=0)
    u_l2 = torch.sqrt(u_l2).cpu().numpy()

    u_linf = torch.max((torch.abs(x_num.view(-1) - x_ana.view(-1)))
                       ).cpu().numpy()
    return u_l2, u_linf
