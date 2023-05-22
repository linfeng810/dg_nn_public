#!/usr/bin/env python3

"""
Wrap solvers into one function here.
We have MG solver, and MG-preconditioned GMRES solver.
"""
import torch
import numpy as np
import scipy as sp
import config
import multigrid_linearelastic as mg
from volume_mf_linear_elastic import get_residual_only, get_residual_and_smooth_once
from config import sf_nd_nb


nonods = config.nonods
dev = config.dev
nloc = config.nloc
ndim = config.ndim


def _multigrid_one_cycle(u_i, u_n, u_bc, f, u_rhs=0):
    """
    given initial vector and right hand side,
    do one MG cycle.
    TODO: MG determined by config parameters:
        - pmg?
        - amg? (SFC)
        - which levels of SFCs to visit?
        - pre/post smooth step?
    """
    r0 = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)
    u_i = u_i.view(nonods, ndim)
    cg_nonods = sf_nd_nb.cg_nonods

    for its1 in range(config.pre_smooth_its):
        r0 *= 0
        r0, u_i = get_residual_and_smooth_once(
            r0, u_i, u_n, u_bc, f, u_rhs)
    # get residual on PnDG
    r0 *= 0
    r0 = get_residual_only(
        r0,
        u_i, u_n, u_bc, f, u_rhs)

    if not config.is_pmg:  # PnDG to P1CG
        r1 = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
        for idim in range(ndim):
            r1[:, idim] += torch.mv(sf_nd_nb.I_cf, mg.p3dg_to_p1dg_restrictor(r0[:, idim]))
    else:  # PnDG down one order each time, eventually go to P1CG
        r_p, e_p = mg.p_mg_pre(r0)
        r1 = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
        ilevel = config.ele_p - 1
        for idim in range(ndim):
            r1[:, idim] += torch.mv(sf_nd_nb.I_cf, r_p[ilevel][:, idim])
    if not config.is_sfc:  # two-grid method
        e_i = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
        e_direct = sp.sparse.linalg.spsolve(
            sf_nd_nb.RAR,
            r1.contiguous().view(-1).cpu().numpy())
        e_direct = np.reshape(e_direct, (cg_nonods, ndim))
        e_i += torch.tensor(e_direct, device=dev, dtype=torch.float64)
    else:  # multi-grid method
        ncurve = 1  # always use 1 sfc
        N = len(sf_nd_nb.sfc_data.space_filling_curve_numbering)
        inverse_numbering = np.zeros((N, ncurve), dtype=int)
        inverse_numbering[:, 0] = np.argsort(sf_nd_nb.sfc_data.space_filling_curve_numbering[:, 0])
        r1_sfc = r1[inverse_numbering[:, 0], :].view(cg_nonods, ndim)

        # # if we do the presmooth steps inside mg_on_P1CG, there's no need to pass in rr1 and e_i
        # e_i = torch.zeros((cg_nonods,1), device=dev, dtype=torch.float64)
        # rr1 = r1_sfc.detach().clone()
        # rr1_l2_0 = torch.linalg.norm(rr1.view(-1),dim=0)
        # rr1_l2 = 10.
        # go to SFC coarse grid levels and do 1 mg cycles there
        e_i = mg.mg_on_P1CG(
            r1_sfc.view(cg_nonods, ndim),
            sf_nd_nb.sfc_data.variables_sfc,
            sf_nd_nb.sfc_data.nlevel,
            sf_nd_nb.sfc_data.nodes_per_level
        )
        # reverse to original order
        e_i = e_i[sf_nd_nb.sfc_data.space_filling_curve_numbering[:, 0] - 1, :].view(cg_nonods, ndim)
    if not config.is_pmg:  # from P1CG to P3DG
        # prolongate error to fine grid
        e_i0 = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)
        for idim in range(ndim):
            e_i0[:, idim] += mg.p1dg_to_p3dg_prolongator(torch.mv(sf_nd_nb.I_fc, e_i[:, idim]))
    else:  # from P1CG to P1DG, then go one order up each time while also do post smoothing
        # prolongate error to P1DG
        ilevel = config.ele_p - 1
        for idim in range(ndim):
            e_p[ilevel][:, idim] += torch.mv(sf_nd_nb.I_fc, e_i[:, idim])
        r_p, e_p = mg.p_mg_post(e_p, r_p)
        e_i0 = e_p[0]
    # correct fine grid solution
    u_i += e_i0
    # post smooth
    for its1 in range(config.post_smooth_its):
        r0 *= 0
        r0, u_i = get_residual_and_smooth_once(
            r0, u_i, u_n, u_bc, f, u_rhs)
    # r0l2 = torch.linalg.norm(r0.view(-1), dim=0) / r0_init  # fNorm

    return u_i, r0


def multigrid_solver(u_i, u_n, u_bc, f,
                     tol):
    """
    use multigrid as a solver
    """
    # if transient, u_i is the field value of last time-step.
    # otherwise, it is zero.

    # first do one cycle to get initial residual
    r0 = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)
    dummy = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)
    dummy = get_rhs(u_n, u_bc, f)
    u_n *= 0
    u_bc *= 0
    f *= 0
    u_i, _ = _multigrid_one_cycle(u_i, u_n, u_bc, f, u_rhs=dummy)
    r0 *= 0
    r0 = get_residual_only(r0,
                           u_i, u_n, u_bc, f, u_rhs=dummy)
    r0l2_init = torch.linalg.norm(r0.view(-1), dim=0)
    r0l2 = torch.tensor([1.], device=dev, dtype=torch.float64)

    # now we do MG cycles
    its = 1
    while r0l2 > tol and its < config.jac_its:
        u_i, r = _multigrid_one_cycle(u_i, u_n, u_bc, f, u_rhs=dummy)
        # r0 *= 0
        # r0 = get_residual_only(r0,
        #                        c_i, c_n, c_bc, f, c_rhs=dummy)
        r0l2 = torch.linalg.norm(r.view(-1), dim=0) / r0l2_init
        its += 1
        print('its=', its, 'fine grid rel residual l2 norm=', r0l2.cpu().numpy())
    return u_i


def gmres_mg_solver(u_i, u_n, u_bc, f,
                    tol):
    """
    use mg left preconditioned gmres as a solver
    """
    u_i = u_i.view(-1)
    m = config.gmres_m
    v_m = torch.zeros(m+1, nonods * ndim, device=dev, dtype=torch.float64)  # V_m
    h_m = torch.zeros(m+1, m, device=dev, dtype=torch.float64)  # \bar H_m
    r0 = torch.zeros(nonods * ndim, device=dev, dtype=torch.float64)
    dummy = torch.zeros(nonods * ndim, device=dev, dtype=torch.float64)
    r0l2 = 1.
    its = 0
    e_1 = torch.zeros(m + 1, device=dev, dtype=torch.float64)  # this is a unit vector in the least square prob.
    e_1[0] += 1
    while r0l2 > tol and its < config.gmres_its:
        h_m *= 0
        v_m *= 0
        r0 *= 0
        r0 = get_residual_only(r0,
                               u_i, u_n, u_bc, f)
        r0 = r0.view(nonods, ndim)
        r0, _ = _multigrid_one_cycle(u_i=torch.zeros(nonods, ndim, device=dev, dtype=torch.float64),
                                     u_n=dummy,
                                     u_bc=dummy,
                                     f=dummy,
                                     u_rhs=r0)
        r0 = r0.view(-1)
        beta = torch.linalg.norm(r0)
        v_m[0, :] += r0 / beta
        w = r0  # this should place w in the same memory as r0 so that we don't take two nonods memory space
        for j in range(0, m):
            w *= 0
            w = get_residual_only(r0=w,
                                  u_i=v_m[j, :],
                                  u_n=dummy,
                                  u_bc=dummy,
                                  f=dummy,
                                  u_rhs=dummy)  # providing rhs=0, b-Ax is -Ax
            w = w.view(nonods, ndim)
            w *= -1.
            w, _ = _multigrid_one_cycle(u_i=torch.zeros(nonods, ndim, device=dev, dtype=torch.float64),
                                        # ↑ here I believe we create another nonods memory usage
                                        u_n=dummy,
                                        u_bc=dummy,
                                        f=dummy,
                                        u_rhs=w)
            w = w.view(-1)
            for i in range(0, j+1):
                h_m[i, j] = torch.linalg.vecdot(w, v_m[i, :])
                w -= h_m[i, j] * v_m[i, :]
            h_m[j+1, j] = torch.linalg.norm(w)
            v_m[j+1, :] += w / h_m[j+1, j]
        # solve least-square problem
        q, r = torch.linalg.qr(h_m, mode='complete')  # h_m: (m+1)xm, q: (m+1)x(m+1), r: (m+1)xm
        e_1 *= 0
        e_1[0] += beta
        y_m = torch.linalg.solve(r[0:m, 0:m], q[0:m+1, 0:m].T @ e_1)  # y_m: m
        # update c_i and get residual
        u_i += torch.einsum('ji,j->i', v_m[0:m, :], y_m)
        r0l2 = torch.linalg.norm(q[:, m:m+1].T @ e_1)
        r0 *= 0
        r0 = get_residual_only(r0,
                               u_i, u_n, u_bc, f)
        r0 = r0.view(-1)
        r0l2 = torch.linalg.norm(r0)
        print('its=', its, 'fine grid rel residual l2 norm=', r0l2.cpu().numpy())
        its += 1
    return u_i


def gmres_solver(c_i, c_n, c_bc, f,
                 tol):
    """
    use mg left preconditioned gmres as a solver
    """
    c_i = c_i.view(-1)
    m = config.gmres_m
    v_m = torch.zeros(m+1, nonods, device=dev, dtype=torch.float64)  # V_m
    h_m = torch.zeros(m+1, m, device=dev, dtype=torch.float64)  # \bar H_m
    r0 = torch.zeros(nonods, device=dev, dtype=torch.float64)
    dummy = torch.zeros(nonods, device=dev, dtype=torch.float64)
    r0l2 = 1.
    its = 0
    e_1 = torch.zeros(m + 1, device=dev, dtype=torch.float64)
    e_1[0] += 1
    while r0l2 > tol and its < config.gmres_its:
        h_m *= 0
        v_m *= 0
        r0 *= 0
        r0 = get_residual_only(r0,
                               c_i, c_n, c_bc, f).view(-1)
        beta = torch.linalg.norm(r0)
        v_m[0, :] += r0 / beta
        w = r0  # this should place w in the same memory as r0 so that we don't take two nonods memory space
        for j in range(0, m):
            w *= 0
            w = get_residual_only(r0=w,
                                  u_i=v_m[j, :],
                                  u_n=dummy,
                                  u_bc=dummy,
                                  f=dummy,
                                  u_rhs=dummy).view(-1)  # providing rhs=0, b-Ax is -Ax
            w *= -1.
            for i in range(0, j+1):
                h_m[i, j] = torch.linalg.vecdot(w, v_m[i, :])
                w -= h_m[i, j] * v_m[i, :]
            h_m[j+1, j] = torch.linalg.norm(w)
            v_m[j+1, :] += w / h_m[j+1, j]
        # solve least-square problem
        q, r = torch.linalg.qr(h_m, mode='complete')  # h_m: (m+1)xm, q: (m+1)x(m+1), r: (m+1)xm
        e_1 *= 0
        e_1[0] += beta
        y_m = torch.linalg.solve(r[0:m, 0:m], q[0:m+1, 0:m].T @ e_1)  # y_m: m
        # update c_i and get residual
        u_i += torch.einsum('ji,j->i', v_m[0:m, :], y_m)
        r0l2 = torch.linalg.norm(q[:, m:m+1].T @ e_1)
        # r0 *= 0
        # r0 = get_residual_only(r0,
        #                        c_i, c_n, c_bc, f)
        # r0l2 = torch.linalg.norm(r0)
        print('its=', its, 'fine grid rel residual l2 norm=', r0l2.cpu().numpy())
        its += 1
    return c_i


def get_rhs(u_n, u_bc, f):
    """
    get right-hand side at each time step.
    """
    u_rhs = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)
    dummy = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)
    u_rhs = get_residual_only(u_rhs, dummy, u_n, u_bc, f)
    return u_rhs
