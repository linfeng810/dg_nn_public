"""matrix-free integral for mesh velocity
or mesh displacement"""

import torch
import numpy as np
import scipy as sp
import pyamg
import config
import function_space
from config import sf_nd_nb
import multigrid_linearelastic as mg
from tqdm import tqdm
import sa_amg
from typing import List, Tuple

if config.ndim == 2:
    from shape_function import get_det_nlx as get_det_nlx
    from shape_function import sdet_snlx as sdet_snlx
else:
    from shape_function import get_det_nlx_3d as get_det_nlx
    from shape_function import sdet_snlx_3d as sdet_snlx

from torch.profiler import profile, record_function, ProfilerActivity

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

    if False:  # test integration speed
        import time
        r0 = torch.zeros_like(x_i, device=dev, dtype=torch.float64)
        starttime = time.time()
        for i in tqdm(range(1000)):
            r0, x_i = get_residual_or_smooth(
                r0, x_i, x_rhs,
                do_smooth=False,
            )
        endtime = time.time()
        print('time for 1000 integration: ', endtime - starttime)
        exit(0)

    if False:  # test integration speed
        # import time
        from torch.profiler import profile, record_function, ProfilerActivity
        r0 = torch.zeros_like(x_i, device=dev, dtype=torch.float64)
        # starttime = time.time()
        # for ii in tqdm(range(20)):
        with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/ef191bb_d8D8_cpu'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True) as prof:
            for i in range(6):  # do smooth 5 times
                r0, x_i = get_residual_or_smooth(
                    r0, x_i, x_rhs,
                    do_smooth=False,
                )
                prof.step()

        # endtime = time.time()
        # print('time for 1000 integration: ', endtime - starttime)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # prof.export_chrome_trace("trace.json")
        # Print aggregated stats
        print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=5))

        exit(0)

    if False:  # use torch.utils.benchmark to time get_residual_or_smooth
        import torch.utils.benchmark as benchmark
        r0 = torch.zeros_like(x_i, device=dev, dtype=torch.float64)
        get_residual_or_smooth(
            r0, x_i, x_rhs,
            do_smooth=True,
        )
        t0 = benchmark.Timer(
            stmt='get_residual_or_smooth(r0, x_i, x_rhs, do_smooth)',
            setup='from volume_mf_diff import get_residual_or_smooth',
            globals={'r0': r0, 'x_i': x_i, 'x_rhs': x_rhs, 'do_smooth': False},
        )
        print(t0.timeit(10))
        exit(0)

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
    if sf_nd_nb.vel_func_space.jac_v.shape[0] == nele:
        j = sf_nd_nb.vel_func_space.jac_v[idx_in]
    else:
        j = sf_nd_nb.vel_func_space.jac_v  # will pass an empty tensor [] and get_det_nlx will handle it.
    nx, ndetwei = get_det_nlx(
        nlx=sf_nd_nb.vel_func_space.element.nlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[idx_in],
        weight=sf_nd_nb.vel_func_space.element.weight,
        nloc=u_nloc,
        ngi=sf_nd_nb.vel_func_space.element.ngi,
        real_nlx=None,
        j=j,
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
    if sf_nd_nb.vel_func_space.jac_s.shape[0] != 0:
        j = sf_nd_nb.vel_func_space.jac_s[:, :, E_F_b, :, :]
    else:
        j = sf_nd_nb.vel_func_space.jac_s  # will pass an empty tensor [] and sdet_snlx will handle it.
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_b],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi,
        sn=sf_nd_nb.vel_func_space.element.sn,
        real_snlx=None,
        is_get_f_det_normal=True,
        j=j,
        drst_duv=sf_nd_nb.vel_func_space.drst_duv,
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
    if sf_nd_nb.vel_func_space.jac_s.shape[0] != 0:
        j = sf_nd_nb.vel_func_space.jac_s[:, :, E_F_b_n, :, :]
    else:
        j = sf_nd_nb.vel_func_space.jac_s  # will pass an empty tensor [] and sdet_snlx will handle it.
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in[E_F_b_n],
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi,
        sn=sf_nd_nb.vel_func_space.element.sn,
        real_snlx=None,
        is_get_f_det_normal=True,
        j=j,
        drst_duv=sf_nd_nb.vel_func_space.drst_duv,
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
    if do_smooth, preconputed diagK and bdiagK will be used.

    inside this function, we no longer generate diagK and bdiagK.
    Please seek get_bdiag_diag.py:get_bdiag_diag for that.
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
        diagK = sf_nd_nb.diagK
        bdiagK = sf_nd_nb.bdiagK
        _k_res_one_batch(
            r0, x_i,
            sf_nd_nb.vel_func_space,
            config.mu_f, nele_f,
        )
        # surface integral
        idx_in_f = torch.zeros(nele * nface, dtype=torch.bool, device=dev)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True

        r0 = _s_res_one_batch(
            r0, x_i,
            # diagK, bdiagK,
            idx_in_f, brk_pnt[i],
            sf_nd_nb.vel_func_space,
            config.mu_f, config.eta_e
        )

        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # # prof.export_chrome_trace("trace.json")
        # # Print aggregated stats
        # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=5))
        #
        # exit(0)
        if do_smooth:
            if config.blk_solver == 'direct':
                # smooth once
                bdiagK = torch.inverse(bdiagK[batch_in, :, :])
                x_i = x_i.view(nele, u_nloc)
                x_i[idx_in, :] += config.jac_wei * torch.einsum(
                    '...ij,...j->...i',
                    bdiagK,
                    r0.view(nele, u_nloc)[idx_in, :]
                ).view(batch_in, u_nloc)
            else:  # use point Jabocian to approximate inv block diagonal
                # getting diagonal of K
                # diagK = diagK[batch_in, :]
                if False:  # test get_bdiag_diag
                    from get_bdiag_diag import get_bdiag_diag
                    diagK1, bdiagK1 = get_bdiag_diag()
                    print('norm diagK: ', torch.linalg.norm(diagK1 - diagK),
                          'norm bdiagK:', torch.linalg.norm(bdiagK1 - bdiagK))
                    exit(0)

                c_i = torch.zeros_like(x_i, device=dev, dtype=torch.float64)  # correction to x_i
                c_i = c_i.view(nele, u_nloc)
                for it in range(3):
                    # we need proper relaxing coefficient here as well! 1.0 simply doesn't work.
                    c_i += config.jac_wei * (r0 - torch.bmm(bdiagK, c_i.view(nele, u_nloc, 1)).squeeze()) / diagK
                    # print(torch.norm(r0 - torch.bmm(bdiagK, c_i.view(nele, u_nloc, 1)).squeeze()))
                x_i += config.jac_wei * c_i.view(x_i.shape)
    r0 = r0.view(-1)
    x_i = x_i.view(-1)
    return r0, x_i


# @torch.jit.optimize_for_inference
@torch.jit.script
def _k_res_one_batch(
        r0, x_i,
        # idx_in,  # those without explicit type declaration default to torch.Tensor
        func_space: function_space.FuncSpaceTS,
        mu_f: float, nele_f: int,
) -> None:  # Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u_nloc = func_space.element.nloc
    nele = func_space.nele
    dev = func_space.dev
    ngi = func_space.element.ngi
    ndim = func_space.element.ndim

    r0 = r0.view(nele, u_nloc)
    x_i = x_i.view(nele, u_nloc)
    # with torch.profiler.record_function("GETTING VOLUME SF"):
    # get shape function and derivatives
    # n = sf_nd_nb.vel_func_space.element.n
    nx, ndetwei = get_det_nlx(
        nlx=func_space.element.nlx,
        x_loc=func_space.x_ref_in,
        weight=func_space.element.weight,
        nloc=u_nloc,
        ngi=func_space.element.ngi,
        real_nlx=None,
        j=func_space.jac_v,
    )
    # torch.cuda.synchronize()

    nx_u = torch.mul(
        nx,  # (nele, ndim, u_nloc, ngi)
        x_i.view(nele, 1, u_nloc, 1)
    ).sum(2)  # (nele, ndim, ngi)
    nx_u_nx = torch.mul(
        nx_u.view(nele, ndim, 1, ngi),  # (nele, ndim, ngi)
        nx  # (nele, ndim, u_nloc, ngi)
    ).sum(1)  # (nele, u_nloc, ngi)
    r0 -= torch.bmm(
        nx_u_nx,  # (nele, u_nloc, ngi)
        ndetwei.view(nele, ngi, 1)  # (nele, ngi, 1)
    ).squeeze() * mu_f


# @torch.jit.optimize_for_inference
# @torch.jit.script
def _s_res_fi(
        r0, f_i, E_F_i,
        f_inb, E_F_inb,
        x_i,
        diagK, bdiagK, batch_start_idx: int,
        nb_gi_aln: int,  # index of neighbour face alignment (3 options in 3D; 2 in 2D)
        func_space: function_space.FuncSpaceTS,
        mu_f: float, eta_e: float,
) -> None:  # Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """internal faces"""
    batch_in = f_i.shape[0]
    dev = func_space.dev
    nele = func_space.nele
    ndim = func_space.element.ndim
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    # get element parameters
    u_nloc = func_space.element.nloc
    sngi = func_space.element.sngi
    r0 = r0.view(nele, u_nloc)
    x_i = x_i.view(nele, u_nloc)

    # shape function on this side
    if func_space.jac_s.shape[0] != 0:
        j = func_space.jac_s[:, :, E_F_i, :, :]
    else:
        j = func_space.jac_s  # will pass an empty tensor [] and sdet_snlx will handle it.
    snx, sdetwei, snormal = sdet_snlx(
        snlx=func_space.element.snlx,
        x_loc=func_space.x_ref_in[E_F_i],
        sweight=func_space.element.sweight,
        nloc=func_space.element.nloc,
        sngi=func_space.element.sngi,
        sn=func_space.element.sn,
        real_snlx=None,
        is_get_f_det_normal=True,
        j=j,
        drst_duv=func_space.drst_duv,
    )
    sn = func_space.element.sn[f_i, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_i, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_i, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_i, ...]  # (batch_in, ndim, sngi)

    # shape function on the other side
    if func_space.jac_s.shape[0] != 0:
        j = func_space.jac_s[:, :, E_F_inb, :, :]
    else:
        j = func_space.jac_s  # will pass an empty tensor [] and sdet_snlx will handle it.
    snx_nb, _, _ = sdet_snlx(
        snlx=func_space.element.snlx,
        x_loc=func_space.x_ref_in[E_F_inb],
        sweight=func_space.element.sweight,
        nloc=func_space.element.nloc,
        sngi=func_space.element.sngi,
        sn=func_space.element.sn,
        real_snlx=None,
        is_get_f_det_normal=False,  # don't need to get snormal and sdetwei for neighbouring ele because
        # its either the same as this ele or the opposite of this ele
        j=j,
        drst_duv=func_space.drst_duv,
    )
    # get faces we want
    sn_nb = func_space.element.sn[f_inb, ...]  # (batch_in, nloc, sngi)
    snx_nb = snx_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim, nloc, sngi)
    # snormal_nb = snormal_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim, sngi)
    # change gaussian points order on other side
    nb_aln = func_space.element.gi_align[nb_gi_aln, :]  # nb_aln for velocity element
    snx_nb = snx_nb[:, :, :, nb_aln]
    # snormal_nb = snormal_nb[..., nb_aln]
    # don't forget to change gaussian points order on sn_nb!
    sn_nb = sn_nb[:, :, nb_aln]

    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = eta_e / h

    u_ith = x_i[E_F_i, ...]
    u_inb = x_i[E_F_inb, ...]

    # K block
    K = torch.zeros(batch_in, u_nloc, u_nloc, device=dev, dtype=torch.float64)
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
    # K = torch.zeros_like(K, device=dev, dtype=torch.float64)
    # unfold einsum
    snx_snormal = torch.bmm(
        snx.permute(0, 3, 2, 1).reshape(-1, u_nloc, ndim),
        snormal.permute(0, 2, 1).reshape(-1, ndim, 1)
    )
    snx_snormal_sn = torch.bmm(
        sn.permute(0, 2, 1).reshape(-1, u_nloc, 1),
        snx_snormal.view(-1, 1, u_nloc),
    ).view(batch_in, sngi, u_nloc, u_nloc)
    snx_snormal_sn_sdetwei = (snx_snormal_sn
                              * sdetwei.view(batch_in, sngi, 1, 1)).sum(dim=1)
    K += snx_snormal_sn_sdetwei * (-0.5)
    K += snx_snormal_sn_sdetwei.transpose(1, 2) * (-0.5)
    sn_sn = (
        sn[f_i[0]].permute(1, 0).reshape(-1, u_nloc, 1) *
        sn[f_i[0]].permute(1, 0).reshape(-1, 1, u_nloc)
    ).view(sngi, u_nloc, u_nloc)
    sn_sn_sdetwei = (sn_sn.view(1, sngi, u_nloc, u_nloc)
                     * sdetwei.view(batch_in, sngi, 1, 1)).sum(dim=1)
    K += sn_sn_sdetwei * gamma_e.unsqueeze(1).unsqueeze(2)
    # print(torch.linalg.norm(K - K1))  # 1.2437e-16

    # # what if we put quadrature point dim as 2nd dimension?
    # # it seems g is more like a batch dimension than a compute dimension (contraction, mul etc)
    # sn1 = sn.permute(0, 2, 1).contiguous()
    # snx1 = snx.permute(0, 3, 2, 1).contiguous()
    # snormal1 = snormal.permute(0, 2, 1).contiguous()
    # if False and f_i[0] == 2 and nb_gi_aln == 1:
    #     print("batch in ", batch_in)
    #     # import time
    #     # starttime = time.time()
    #     start_event = torch.cuda.Event(enable_timing=True)
    #     end_event = torch.cuda.Event(enable_timing=True)
    #     start_event.record()
    #
    #     # Run some things here
    #
    #     iisteps = 100
    #     for ii in range(iisteps):
    #         # unfold einsum
    #         snx_snormal = torch.bmm(
    #             snx.permute(0, 3, 2, 1).reshape(-1, u_nloc, ndim),
    #             snormal.permute(0, 2, 1).reshape(-1, ndim, 1)
    #         )
    #         snx_snormal_sn = torch.bmm(
    #             sn.permute(0, 2, 1).reshape(-1, u_nloc, 1),
    #             snx_snormal.view(-1, 1, u_nloc),
    #         ).view(batch_in, sngi, u_nloc, u_nloc)
    #         snx_snormal_sn_sdetwei = (snx_snormal_sn
    #                                   * sdetwei.view(batch_in, sngi, 1, 1)).sum(dim=1)
    #         K1 += snx_snormal_sn_sdetwei * (-0.5)
    #         K1 += snx_snormal_sn_sdetwei.transpose(1, 2) * (-0.5)
    #         sn_sn = torch.bmm(
    #             sn.permute(0, 2, 1).reshape(-1, u_nloc, 1),
    #             sn.permute(0, 2, 1).reshape(-1, 1, u_nloc),
    #         ).view(batch_in, sngi, u_nloc, u_nloc)
    #         sn_sn_sdetwei = (sn_sn
    #                          * sdetwei.view(batch_in, sngi, 1, 1)).sum(dim=1)
    #         K1 += sn_sn_sdetwei * gamma_e.unsqueeze(1).unsqueeze(2)
    #     # endtime = time.time()
    #     end_event.record()
    #     torch.cuda.synchronize()  # Wait for the events to be recorded!
    #     elapsed_time_ms = start_event.elapsed_time(end_event)
    #     print('time for %d integration: %f ms' % (iisteps, elapsed_time_ms))
    #     exit(0)
    # if False and f_i[0] == 2 and nb_gi_aln == 1:
    #     print('going to profile...')
    #     with profile(activities=[
    #         ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #             schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
    #             on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/_K_no_einsum'),
    #             record_shapes=True,
    #             profile_memory=True,
    #             with_stack=True) as prof:
    #         with record_function("model_inference"):
    #             iisteps = 1000
    #             for ii in range(iisteps):
    #                 prof.step()
    #                 # unfold einsum
    #                 snx_snormal = torch.bmm(
    #                     snx.permute(0, 3, 2, 1).reshape(-1, u_nloc, ndim),
    #                     snormal.permute(0, 2, 1).reshape(-1, ndim, 1)
    #                 )
    #                 snx_snormal_sn = torch.bmm(
    #                     sn.permute(0, 2, 1).reshape(-1, u_nloc, 1),
    #                     snx_snormal.view(-1, 1, u_nloc),
    #                 ).view(batch_in, sngi, u_nloc, u_nloc)
    #                 snx_snormal_sn_sdetwei = (snx_snormal_sn
    #                                           * sdetwei.view(batch_in, sngi, 1, 1)).sum(dim=1)
    #                 K1 += snx_snormal_sn_sdetwei * (-0.5)
    #                 K1 += snx_snormal_sn_sdetwei.transpose(1, 2) * (-0.5)
    #                 sn_sn = torch.bmm(
    #                     sn.permute(0, 2, 1).reshape(-1, u_nloc, 1),
    #                     sn.permute(0, 2, 1).reshape(-1, 1, u_nloc),
    #                 ).view(batch_in, sngi, u_nloc, u_nloc)
    #                 sn_sn_sdetwei = (sn_sn
    #                                  * sdetwei.view(batch_in, sngi, 1, 1)).sum(dim=1)
    #                 K1 += sn_sn_sdetwei * gamma_e.unsqueeze(1).unsqueeze(2)
    #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    #     # prof.export_chrome_trace("trace.json")
    #     # Print aggregated stats
    #     print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=5))
    #
    #     exit(0)
    K *= mu_f

    # update residual
    r0[E_F_i, ...] -= torch.einsum('bmn,bn->bm', K, u_ith)
    # put diagonal into diagK and bdiagK
    # diagK[E_F_i - batch_start_idx, ...] += torch.diagonal(K.view(batch_in, u_nloc, u_nloc),
    #                                                       dim1=1, dim2=2).view(batch_in, u_nloc)
    bdiagK[E_F_i - batch_start_idx, ...] += K

    # other side
    K *= 0
    # # [v_i n_j] {du_i / dx_j}  consistent term
    # K += torch.einsum(
    #     'bmg,bjg,bjng,bg->bmn',
    #     sn,  # (batch_in, nloc, sngi)
    #     snormal,  # (batch_in, ndim, sngi)
    #     snx_nb,  # (batch_in, ndim, nloc, sngi)
    #     sdetwei,  # (batch_in, sngi)
    # ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim) \
    # # {dv_i / dx_j} [u_i n_j]  symmetry term
    # K += torch.einsum(
    #     'bjmg,bng,bjg,bg->bmn',
    #     snx,  # (batch_in, ndim, nloc, sngi)
    #     sn_nb,  # (batch_in, nloc, sngi)
    #     -snormal,  # (batch_in, ndim, sngi)
    #     sdetwei,  # (batch_in, sngi)
    # ) * (-0.5)  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim) \
    # # \gamma_e * [v_i][u_i]  penalty term
    # K += torch.einsum(
    #     'bmg,bng,bg,b->bmn',
    #     sn,  # (batch_in, nloc, sngi)
    #     sn_nb,  # (batch_in, nloc, sngi)
    #     sdetwei,  # (batch_in, sngi)
    #     gamma_e,  # (batch_in)
    # ) * (-1.)  # because n2 \cdot n1 = -1

    # # K1 = torch.zeros_like(K, device=dev, dtype=torch.float64)
    # unfold einsum
    snx_snormal = torch.bmm(
        snx_nb.permute(0, 3, 2, 1).reshape(-1, u_nloc, ndim),
        snormal.permute(0, 2, 1).reshape(-1, ndim, 1)
    )  # (b*g, n)
    snx_snormal_sn = torch.bmm(
        sn.permute(0, 2, 1).reshape(-1, u_nloc, 1),
        snx_snormal.view(-1, 1, u_nloc),
    ).view(batch_in, sngi, u_nloc, u_nloc)
    snx_snormal_sn_sdetwei = (snx_snormal_sn
                              * sdetwei.view(batch_in, sngi, 1, 1)).sum(dim=1)
    K += snx_snormal_sn_sdetwei * (-0.5)
    snx_snormal = torch.bmm(
        snx.permute(0, 3, 2, 1).reshape(-1, u_nloc, ndim),
        -snormal.permute(0, 2, 1).reshape(-1, ndim, 1)  # should be snormal_nb, but it equals to -snormal
    )  # (b*g, m)
    snx_snormal_sn = (
        sn_nb.permute(0, 2, 1).reshape(-1, 1, u_nloc)  # (b*g, 1, n)
        * snx_snormal.view(-1, u_nloc, 1)  # (b*g, m, 1)
    ).view(batch_in, sngi, u_nloc, u_nloc)  # (b, g, m, n)
    snx_snormal_sn_sdetwei = (snx_snormal_sn
                              * sdetwei.view(batch_in, sngi, 1, 1)).sum(dim=1)
    K += snx_snormal_sn_sdetwei * (-0.5)
    sn_sn = (
        sn.permute(0, 2, 1).reshape(-1, u_nloc, 1) *
        sn_nb.permute(0, 2, 1).reshape(-1, 1, u_nloc)
    ).view(batch_in, sngi, u_nloc, u_nloc)
    sn_sn_sdetwei = (sn_sn
                     * sdetwei.view(batch_in, sngi, 1, 1)).sum(dim=1)
    K += sn_sn_sdetwei * gamma_e.unsqueeze(1).unsqueeze(2) * (-1.)
    K *= mu_f
    # print(torch.linalg.norm(K - K1))

    # update residual
    r0[E_F_i, ...] -= torch.einsum('bmn,bn->bm', K, u_inb)
    # return r0, diagK, bdiagK


@torch.jit.script
def _s_res_fi_all_face(
        r0,
        f_i, E_F_i,
        f_inb, E_F_inb,
        x_i,
        func_space: function_space.FuncSpaceTS,
        mu_f: float, eta_e: float,
) -> None:
    """all internal faces computed in 1 batch
    no matter local face index or neighbour face gi pnts alignment
    will deal those when putting into global matrix/residual
    at the end of this function"""
    batch_in = f_i.shape[0]
    dev = func_space.dev
    nele = func_space.nele
    ndim = func_space.element.ndim
    nface = ndim + 1
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    # get element parameters
    u_nloc = func_space.element.nloc
    r0 = r0.view(nele, u_nloc)
    x_i = x_i.view(nele, u_nloc)

    # with torch.profiler.record_function("GETTING FI SURFACE SF"):
    # shape function on this side
    snx, sdetwei, snormal = sdet_snlx(
        snlx=func_space.element.snlx,
        x_loc=func_space.x_ref_in,
        sweight=func_space.element.sweight,
        nloc=func_space.element.nloc,
        sngi=func_space.element.sngi,
        sn=func_space.element.sn,
        real_snlx=None,
        is_get_f_det_normal=True,
        j=func_space.jac_s,
        drst_duv=func_space.drst_duv,
    )
        # torch.cuda.synchronize()
    # with torch.profiler.record_function("FI internal"):
    Au = torch.zeros(nele, nface, u_nloc, device=dev, dtype=torch.float64)
    # this side
    # without fetching faces, we will do for all faces,
    # and use a flag to make bc face be 0.

    # shape:
    # sn (nface, nloc, sngi)
    # snx (nele, nface, ndim, nloc, sngi)
    # sdetwei (nele, nface, sngi)
    # snormal (nele, nface, ndim, sngi)
    sn = func_space.element.sn
    sngi = func_space.element.sngi
    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = eta_e / h  # (nele, nface)

    snx_u = torch.mul(
        snx,  # (nele, nface, ndim, nloc, sngi)
        x_i.view(nele, 1, 1, u_nloc, 1)  # (nele, 1, 1, u_nloc, 1)
    ).sum(3)  # (nele, nface, ndim, sngi)
    snx_u_snormal = torch.mul(
        snx_u,  # (nele, nface, ndim, sngi)
        snormal  # (nele, nface, ndim, sngi)
    ).sum(2)  # (nele, nface, sngi)
    snx_u_snormal_sn = torch.mul(
        snx_u_snormal.view(nele, nface, 1, sngi),  # (nele, nface, 1, sngi)
        sn.view(1, nface, u_nloc, sngi)  # (1, nface, nloc, sngi)
    )
    snx_u_snormal_sn_sdetwei = torch.mul(
        snx_u_snormal_sn,  # (nele, nface, nloc, sngi)
        sdetwei.view(nele, nface, 1, sngi),  # (nele, nface, sngi)
    ).sum(3)  # (nele, nface, nloc)
    Au += snx_u_snormal_sn_sdetwei * (-0.5)  # consistent term

    sn_u = torch.mul(
        sn.view(1, nface, u_nloc, sngi),  # (1, nface, nloc, sngi)
        x_i.view(nele, 1, u_nloc, 1)  # (nele, 1, nloc, 1)
    ).sum(2)  # (nele, nface, sngi)
    sn_u_snormal = torch.mul(
        sn_u.view(nele, nface, 1, sngi),
        snormal.view(nele, nface, ndim, sngi)
    )
    sn_u_snormal_snx = torch.mul(
        sn_u_snormal.view(nele, nface, ndim, 1, sngi),  # (nele, nface, ndim, 1, sngi)
        snx,  # (nele, nface, ndim, nloc, sngi)
    ).sum(2)  # (nele, nface, nloc, sngi)
    sn_u_snormal_snx_sdetwei = torch.mul(
        sn_u_snormal_snx,  # (nele, nface, nloc, sngi)
        sdetwei.view(nele, nface, 1, sngi),  # (nele, nface, sngi)
    ).sum(3)
    Au += sn_u_snormal_snx_sdetwei * (-0.5)  # symmetry term

    sn_u_sn = sn_u.view(nele, nface, 1, sngi) * sn.view(1, nface, u_nloc, sngi)
    sn_u_sn_sdetwei = (sn_u_sn * sdetwei.view(nele, nface, 1, sngi)).sum(3)
    Au += sn_u_sn_sdetwei * gamma_e.view(nele, nface, 1)  # penalty term

    # snx_snormal = torch.bmm(
    #     snx.permute(0, 1, 4, 3, 2).reshape(-1, u_nloc, ndim),
    #     snormal.permute(0, 1, 3, 2).reshape(-1, ndim, 1)
    # )  # (nele * nface * sngi, u_nloc, 1)
    # if False:
    #     snx_snormal_sn = torch.einsum(
    #         'fng,bfgm->bfmng',
    #         sn.view(nface, u_nloc, sngi),
    #         snx_snormal.view(nele, nface, sngi, u_nloc)
    #     )
    #     snx_snormal_sn_sdetwei = torch.einsum(
    #         'bfmng,bfg->bfmn',
    #         snx_snormal_sn,  # (nele, nface, nloc, nloc, sngi)
    #         sdetwei  # .view(nele, nface, sngi)
    #     )
    # else:  # let's switch multiply sequence to save memory
    #     snx_snormal_sdetwei = torch.mul(
    #         snx_snormal.view(nele, nface, sngi, u_nloc),
    #         sdetwei.view(nele, nface, sngi, 1)
    #     )
    #     snx_snormal_sn_sdetwei = torch.einsum(
    #         'bfgm,fng->bfmn',
    #         snx_snormal_sdetwei,  # (nele, nface, sngi, nloc)
    #         sn.view(nface, u_nloc, sngi),
    #     )
    # K += snx_snormal_sn_sdetwei * (-0.5)  # consistent term
    # K += snx_snormal_sn_sdetwei.transpose(2, 3) * (-0.5)  # symmetry term
    # sn_sn = torch.einsum(
    #     'fmg,fng->fmng',
    #     sn, sn,  # (nface, nloc, sngi)
    # )
    # sn_sn_sdetwei = torch.einsum(
    #     'fmng,bfg->bfmn',  # (nele, nface, nloc, nloc)
    #     sn_sn, sdetwei  # (nele, nface, sngi)
    # )
    # K += sn_sn_sdetwei * gamma_e.unsqueeze(2).unsqueeze(3)  # penalty term

    Au *= mu_f
    # K *= mu_f

    # set boundary face to 0
    # K = K.view(nele * nface, u_nloc, u_nloc)
    # K *= (func_space.glb_bcface_type < 0).view(-1, 1, 1).to(torch.float64)
    Au = Au.view(nele * nface, u_nloc)
    Au *= (func_space.glb_bcface_type < 0).view(-1, 1).to(torch.float64)

    # put them to r0, diagK and bdiagK (SCATTER)
    # K = K.view(nele, nface, u_nloc, u_nloc)
    Au = Au.view(nele, nface, u_nloc)
    # x_i = x_i.view(nele, u_nloc)
    for iface in range(nface):
        r0 -= Au[:, iface, :]
    # torch.cuda.synchronize()
    # with torch.profiler.record_function("FI external"):
    # other side (we don't need K anymore)
    # we can get residual only. no contribution to diagK or bdiagK
    Au = torch.zeros(batch_in, u_nloc, device=dev, dtype=torch.float64)
    u_inb = x_i[E_F_inb, ...]

    # get faces we want
    sn_th = func_space.element.sn[f_i, ...]  # (batch_in, nloc, sngi)
    sn_nb = func_space.element.sn[f_inb, ...]  # (batch_in, nloc, sngi)
    snx_nb = snx[E_F_inb, f_inb, ...]  # (batch_in, ndim, nloc, sngi)
    # snormal_nb = snormal_nb[E_F_inb, f_inb, ...]  # (batch_in, ndim, sngi)
    snormal_th = snormal[E_F_i, f_i, ...]  # (batch_in, ndim, sngi)
    sdetwei_th = sdetwei[E_F_i, f_i, ...]  # (batch_in, sngi)
    # change gaussian points order on other side
    # ===== new ======
    for nb_gi_aln in range(ndim):  # 'ndim' alignnment of GI points on neighbour faces
        idx = func_space.alnmt[E_F_i * nface + f_i] == nb_gi_aln
        nb_aln = func_space.element.gi_align[nb_gi_aln, :]
        snx_nb[idx, :, :, :] = snx_nb[idx][:, :, :, nb_aln]
        # snormal_nb[idx, ...] = snormal_nb[idx][..., nb_aln]
        sn_nb[idx, :, :] = sn_nb[idx][:, :, nb_aln]
    if False:
        # *consistent term
        snx_ui = torch.einsum(
            'bing,bn->big',
            snx_nb,  # (batch_in, ndim, nloc, sngi)
            u_inb,  # (batch_in, nloc)
        )
        snx_ui_snormal = torch.einsum(
            'big,big->bg',
            snx_ui,  # (batch_in, ndim, sngi)
            snormal_th,  # (batch_in, ndim, sngi)
        )
        snx_ui_snormal_sn = snx_ui_snormal.view(batch_in, 1, sngi) * sn_th
        Au -= torch.einsum(
            'bmg,bg->bm',
            snx_ui_snormal_sn,  # (batch_in, nloc, sngi)
            sdetwei_th,  # (batch_in, sngi)
        ) * (-0.5)
        # *symmetry term
        sn_ui = torch.einsum(
            'bng,bn->bg',
            sn_nb,  # (batch_in, nloc, sngi)
            u_inb,  # (batch_in, nloc)
        )  # this will be reused later in penalty term
        sn_ui_snormal = sn_ui.view(batch_in, 1, sngi) * snormal_th * (-1.)  # mul -1 to be snormal_nb
        sn_ui_snormal_snx = torch.einsum(
            'big,bimg->bmg',
            sn_ui_snormal,  # (batch_in, ndim, sngi)
            snx[E_F_i, f_i, ...],  # (batch_in, ndim, nloc, sngi)
        )
        Au -= torch.einsum(
            'bmg,bg->bm',
            sn_ui_snormal_snx,
            sdetwei_th,
        ) * (-0.5)
        # *penalty term
        sn_ui_sn = sn_ui.view(batch_in, 1, sngi) * sn_th
        Au -= torch.einsum(
            'bmg,bg->bm',
            sn_ui_sn,  # (batch_in, nloc, sngi)
            sdetwei_th,  # (batch_in, sngi)
        ) * (-gamma_e[E_F_i, f_i].view(batch_in, 1))
    else:  # replace einsum with bmm, mul, sum etc.
        # consistent term
        snx_ui = torch.mul(
            snx_nb,
            u_inb.view(batch_in, 1, u_nloc, 1)
        ).sum(2)  # (batch_in, ndim, sngi)
        snx_ui_snormal = torch.mul(
            snx_ui,
            snormal_th,
        ).sum(1)  # (batch_in, sngi)
        snx_ui_snormal_sn = torch.mul(
            snx_ui_snormal.view(batch_in, 1, sngi),
            sn_th,
        )  # (batch_in, nloc, sngi)
        Au += torch.bmm(
            snx_ui_snormal_sn,
            sdetwei_th.view(batch_in, sngi, 1),
        ).squeeze() * (-0.5)
        # symmetry term
        sn_ui = torch.bmm(
            u_inb.view(batch_in, 1, u_nloc),
            sn_nb  # (batch_in, nloc, sngi)
        )
        sn_ui_snormal = sn_ui.view(batch_in, 1, sngi) * snormal_th * (-1.)  # mul -1 to be snormal_nb
        sn_ui_snormal_snx = torch.mul(
            sn_ui_snormal.view(batch_in, ndim, 1, sngi),  # (batch_in, ndim, sngi)
            snx[E_F_i, f_i, ...],  # (batch_in, ndim, nloc, sngi)
        ).sum(1)  # (batch_in, nloc, sngi)
        Au += torch.bmm(
            sn_ui_snormal_snx,
            sdetwei_th.view(batch_in, sngi, 1),
        ).squeeze() * (-0.5)
        # penalty term
        sn_ui_sn = sn_ui.view(batch_in, 1, sngi) * sn_th
        Au += torch.bmm(
            sn_ui_sn,
            sdetwei_th.view(batch_in, sngi, 1),
        ).squeeze() * (-gamma_e[E_F_i, f_i].view(batch_in, 1))

    # update residual
    # scatter
    for iface in range(nface):
        idx_iface = (f_i == iface)
        r0[E_F_i[idx_iface], ...] -= Au[idx_iface, ...]
    # torch.cuda.synchronize()
    # return r0, diagK, bdiagK


# @torch.jit.optimize_for_inference
# @torch.jit.script
def _s_res_fb(
        r0, f_b, E_F_b,
        x_i,
        diagK, bdiagK,
        batch_start_idx: int,
        func_space: function_space.FuncSpaceTS,
        mu_f: float, eta_e: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """boundary faces"""
    batch_in = f_b.shape[0]
    if batch_in < 1:  # nothing to do here.
        return None  # r0, diagK, bdiagK
    dev = func_space.dev
    nele = func_space.nele
    ndim = func_space.element.ndim
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    # get element parameters
    u_nloc = func_space.element.nloc
    x_i = x_i.view(nele, u_nloc)
    r0 = r0.view(nele, u_nloc)
    # shape function
    if func_space.jac_s.shape[0] != 0:
        j = func_space.jac_s[:, :, E_F_b, :, :]
    else:
        j = func_space.jac_s  # will pass an empty tensor [] and sdet_snlx will handle it.
    snx, sdetwei, snormal = sdet_snlx(
        snlx=func_space.element.snlx,
        x_loc=func_space.x_ref_in[E_F_b],
        sweight=func_space.element.sweight,
        nloc=func_space.element.nloc,
        sngi=func_space.element.sngi,
        sn=func_space.element.sn,
        real_snlx=None,
        is_get_f_det_normal=True,
        j=j,
        drst_duv=func_space.drst_duv,
    )
    sn = func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_b, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_b, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_b, ...]  # (batch_in, ndim, sngi)
    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = eta_e / h

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
    K *= mu_f

    # update residual
    r0[E_F_b, ...] -= torch.einsum('bmn,bn->bm', K, u_ith)
    # put in diagonal
    # diagK[E_F_b - batch_start_idx, ...] += torch.diagonal(K.view(batch_in, u_nloc, u_nloc),
    #                                                       dim1=-2, dim2=-1).view(batch_in, u_nloc)
    bdiagK[E_F_b - batch_start_idx, ...] += K
    # return r0, diagK, bdiagK


# @torch.jit.optimize_for_inference
@torch.jit.script
def _s_res_fb_all_face(
        r0, f_b, E_F_b,
        x_i,
        batch_start_idx: int,
        func_space: function_space.FuncSpaceTS,
        mu_f: float, eta_e: float,
) -> None:
    """boundary faces"""
    batch_in = f_b.shape[0]
    if batch_in < 1:  # nothing to do here.
        return None  # r0, diagK, bdiagK
    dev = func_space.dev
    nele = func_space.nele
    ndim = func_space.element.ndim
    nface = ndim + 1
    sngi = func_space.element.sngi
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    # get element parameters
    u_nloc = func_space.element.nloc
    x_i = x_i.view(nele, u_nloc)
    r0 = r0.view(nele, u_nloc)
    # with torch.profiler.record_function("GETTING FB SURFACE SF"):
    if func_space.jac_s.shape[0] != 0:
        j = func_space.jac_s[:, :, E_F_b, :, :]
    else:
        j = func_space.jac_s  # will pass an empty tensor [] and sdet_snlx will handle it.
    # shape function
    snx, sdetwei, snormal = sdet_snlx(
        snlx=func_space.element.snlx,
        x_loc=func_space.x_ref_in[E_F_b],
        sweight=func_space.element.sweight,
        nloc=func_space.element.nloc,
        sngi=func_space.element.sngi,
        sn=func_space.element.sn,
        real_snlx=None,
        is_get_f_det_normal=True,
        j=j,
        drst_duv=func_space.drst_duv,
    )
    # torch.cuda.synchronize()
    sn = func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_b, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_b, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_b, ...]  # (batch_in, ndim, sngi)
    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = eta_e / h

    # vec
    x_i_b = x_i[E_F_b, ...]
    # mat-vec
    Au = torch.zeros(batch_in, u_nloc, device=dev, dtype=torch.float64)
    # [vi nj] {du_i / dx_j}  consistent term
    snx_u = torch.mul(
        snx,  # (batch_in, ndim, nloc, sngi)
        x_i_b.view(batch_in, 1, u_nloc, 1)  # (batch_in, 1, u_nloc, 1)
    ).sum(2)  # (batch_in, ndim, sngi)
    snx_u_snormal = torch.mul(
        snx_u,  # (batch_in, ndim, sngi)
        snormal  # (batch_in, ndim, sngi)
    ).sum(1)  # (batch_in, sngi)
    snx_u_snormal_sn = torch.mul(
        snx_u_snormal.view(batch_in, 1, sngi),
        sn  # (batch_in, u_nloc, sngi)
    )  # (batch_in, u_nloc, sngi)
    Au -= torch.bmm(snx_u_snormal_sn, sdetwei.view(batch_in, sngi, 1)).squeeze()
    # {du_i / dx_j} [vi nj]  symmetry term
    sn_u = torch.bmm(
        x_i_b.view(batch_in, 1, u_nloc),  # (batch_in, 1, u_nloc)
        sn.view(batch_in, u_nloc, sngi)  # (batch_in, u_nloc, sngi)
    ).squeeze()  # (batch_in, sngi)
    snx_snormal = torch.mul(
        snx,  # (batch_in, ndim, nloc, sngi)
        snormal.view(batch_in, ndim, 1, sngi)  # (batch_in, ndim, 1, sngi)
    ).sum(1)  # (batch_in, nloc, sngi)
    sn_u_snx_snormal = torch.mul(
        sn_u.view(batch_in, 1, sngi),  # (batch_in, 1, sngi)
        snx_snormal  # (batch_in, nloc, sngi)
    )  # (batch_in, nloc, sngi)
    Au -= torch.bmm(sn_u_snx_snormal, sdetwei.view(batch_in, sngi, 1)).squeeze()
    # \gamma_e [v_i] [u_i]  penalty term
    sn_u_sn = torch.mul(
        sn_u.view(batch_in, 1, sngi),  # (batch_in, 1, sngi)
        sn  # (batch_in, u_nloc, sngi)
    )  # (batch_in, u_nloc, sngi)
    Au += torch.bmm(sn_u_sn, sdetwei.view(batch_in, sngi, 1)).squeeze() * gamma_e.view(batch_in, 1)
    Au *= mu_f

    # SCATTER
    for iface in range(nface):
        idx_iface = (f_b == iface)
        E_idx = E_F_b[idx_iface]
        # update residual
        r0[E_idx, ...] -= Au[idx_iface, ...]
    # torch.cuda.synchronize()
    # return r0, diagK, bdiagK


# @profile (line-by-line profile)
# @torch.compile
# @torch.jit.optimize_for_inference
@torch.jit.script
def _s_res_one_batch(
        r0, x_i,
        # diagK, bdiagK,
        idx_in_f,
        batch_start_idx: int,
        func_space: function_space.FuncSpaceTS,
        mu_f: float, eta_e: float,
):  # -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    surface integral (left hand side)
    """
    nbf = func_space.nbf
    glb_bcface_type = func_space.glb_bcface_type
    nface = func_space.element.ndim + 1

    # change view
    u_nloc = func_space.element.nloc

    # separate nbf to get internal face list and boundary face list
    F_i = torch.where(torch.logical_and(glb_bcface_type < 0,
                                        idx_in_f))[0]  # interior face of fluid subdomain
    F_inb = nbf[F_i]  # neighbour list of interior face
    F_inb = F_inb.type(torch.int64)
    F_b = torch.where(torch.logical_and(
        func_space.glb_bcface_type == 0,
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
    # for iface in range(nface):
    #     for nb_gi_aln in range(nface - 1):
    #         idx_iface = (f_i == iface) & (func_space.alnmt[F_i] == nb_gi_aln)
    #         if idx_iface.sum() < 1:
    #             # there is nothing to do here, go on
    #             continue
    #         _s_res_fi(  # r0, diagK, bdiagK = _s_res_fi(
    #             r0, f_i[idx_iface], E_F_i[idx_iface],
    #             f_inb[idx_iface], E_F_inb[idx_iface],
    #             x_i,
    #             diagK, bdiagK, batch_start_idx,
    #             nb_gi_aln,
    #             func_space,
    #             mu_f, eta_e
    #         )
    _s_res_fi_all_face(  # r0, diagK, bdiagK = _s_res_fi_all_face(
        r0,
        f_i, E_F_i,
        f_inb, E_F_inb,
        x_i,
        func_space, mu_f, eta_e
    )
    # boundary faces (dirichlet)
    # for iface in range(nface):
    #     idx_iface = f_b == iface
    #     _s_res_fb(  # r0, diagK, bdiagK = _s_res_fb(
    #         r0, f_b[idx_iface], E_F_b[idx_iface],
    #         x_i,
    #         diagK, bdiagK, batch_start_idx,
    #         func_space, mu_f, eta_e
    #     )
    _s_res_fb_all_face(
        r0, f_b, E_F_b,
        x_i,
        batch_start_idx,
        func_space, mu_f, eta_e
    )
    return r0


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

    if config.mg_opt_D == 1:
        RAR = csr_matrix((RARvalues.cpu().numpy(), cola, fina),
                         shape=(cg_nonods, cg_nonods))
        sf_nd_nb.set_data(RARmat_Um=RAR.tocsr())
    elif config.mg_opt_D == 3:  # use pyamg to smooth on P1CG, setup AMG here
        RAR = csr_matrix((RARvalues.cpu().numpy(), cola, fina),
                         shape=(cg_nonods, cg_nonods))
        # RAR_ml = pyamg.ruge_stuben_solver(RAR.tocsr())
        RAR_ml = pyamg.smoothed_aggregation_solver(RAR.tocsr())
        sf_nd_nb.set_data(RARmat_Um=RAR_ml)
    elif config.mg_opt_D == 4:  # SA wrapped with pytorch -- will be computing on torch device.
        RAR = csr_matrix((RARvalues.cpu().numpy(), cola, fina),
                         shape=(cg_nonods, cg_nonods))
        RAR_ml = sa_amg.SASolver(RAR.tocsr(),
                                 omega=config.jac_wei)
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
    # cudart.cudaProfilerStart()
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
    # cudart.cudaProfilerStop()
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

    if config.mg_opt_D == 1:  # directly solve on P1CG
        e_i = torch.zeros(cg_nonods, device=dev, dtype=torch.float64)
        e_direct = sp.sparse.linalg.spsolve(
            sf_nd_nb.RARmat_Um,
            r1.contiguous().view(-1).cpu().numpy())
        e_direct = np.reshape(e_direct, (cg_nonods))
        e_i += torch.tensor(e_direct, device=dev, dtype=torch.float64)
    elif config.mg_opt_D == 3:  # use pyamg to smooth on P1CG
        e_i = torch.zeros(cg_nonods, device=dev, dtype=torch.float64)
        e_direct = sf_nd_nb.RARmat_Um.solve(
            r1.contiguous().view(-1).cpu().numpy(),
            maxiter=1,
            tol=1e-10)
        # e_direct = np.reshape(e_direct, (cg_nonods))
        e_i += torch.tensor(e_direct, device=dev, dtype=torch.float64)
    elif config.mg_opt_D == 4:  # SA wrapped with pytorch -- will be computing on torch device.
        e_i = sf_nd_nb.RARmat_Um.solve(
            r1.contiguous().view(-1),
            maxiter=1,
            tol=1e-10)
        e_i = e_i.view(-1)
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
        ngi=sf_nd_nb.vel_func_space.element.ngi,
        real_nlx=None,
        j=sf_nd_nb.vel_func_space.jac_v,
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
