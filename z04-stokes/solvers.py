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
# from volume_mf_linear_elastic import get_residual_only, get_residual_and_smooth_once
# from volume_mf_he import get_residual_only, get_residual_and_smooth_once
from volume_mf_st import get_residual_only, get_residual_and_smooth_once, update_rhs, \
    get_r0_l2_norm

import volume_mf_st
from config import sf_nd_nb


# nonods = config.nonods
dev = config.dev
# nloc = config.nloc
ndim = config.ndim


def _multigrid_one_cycle(x_i, x_rhs):
    """
    given initial vector and right hand side,
    do one MG cycle.
    """
    u_nonods = sf_nd_nb.vel_func_space.nonods
    p_nonods = sf_nd_nb.pre_func_space.nonods
    r0 = torch.zeros(u_nonods * ndim + p_nonods, device=dev, dtype=torch.float64)
    r0_list = [r0[0:u_nonods*ndim],
               r0[u_nonods*ndim:u_nonods*ndim+p_nonods]]
    # u_i = u_i.view(nonods, ndim)
    # du_i = du_i.view(nonods, ndim)
    x_i[0] = x_i[0].view(u_nonods, ndim)
    x_i[1] = x_i[1].view(p_nonods)

    cg_nonods = sf_nd_nb.vel_func_space.cg_nonods

    for its1 in range(config.pre_smooth_its):
        r0 *= 0
        r0_list, x_i = get_residual_and_smooth_once(
            r0_list, x_i, x_rhs)
    # get residual on PnDG
    r0 *= 0
    r0_list = get_residual_only(
        r0_list,
        x_i, x_rhs)

    if not config.is_pmg:  # PnDG to P1CG
        r1 = torch.zeros(cg_nonods, ndim+1, device=dev, dtype=torch.float64)
        for idim in range(ndim):
            r1[:, idim] += torch.mv(sf_nd_nb.I_cd, mg.vel_pndg_to_p1dg_restrictor(r0_list[0][:, idim]))
        r1[:, -1] += torch.mv(sf_nd_nb.I_cd, mg.pre_pndg_to_p1dg_restrictor(r0_list[1]))
    else:  # PnDG down one order each time, eventually go to P1CG
        raise NotImplemented('P-multigrid down 1 order each time is not implemented for Stokes.')
        # r_p, e_p = mg.p_mg_pre(r0)
        # r1 = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
        # ilevel = config.ele_p - 1
        # for idim in range(ndim):
        #     r1[:, idim] += torch.mv(sf_nd_nb.I_cf, r_p[ilevel][:, idim])
    if not config.is_sfc:  # two-grid method
        e_i = torch.zeros(cg_nonods, ndim+1, device=dev, dtype=torch.float64)
        e_direct = sp.sparse.linalg.spsolve(
            sf_nd_nb.RARmat,
            r1.contiguous().view(-1).cpu().numpy())
        e_direct = np.reshape(e_direct, (cg_nonods, ndim))
        e_i += torch.tensor(e_direct, device=dev, dtype=torch.float64)
    else:  # multi-grid method
        ncurve = 1  # always use 1 sfc
        N = len(sf_nd_nb.sfc_data.space_filling_curve_numbering)
        inverse_numbering = np.zeros((N, ncurve), dtype=int)
        inverse_numbering[:, 0] = np.argsort(sf_nd_nb.sfc_data.space_filling_curve_numbering[:, 0])
        r1_sfc = r1[inverse_numbering[:, 0], :].view(cg_nonods, ndim+1)

        # # if we do the presmooth steps inside mg_on_P1CG, there's no need to pass in rr1 and e_i
        # e_i = torch.zeros((cg_nonods,1), device=dev, dtype=torch.float64)
        # rr1 = r1_sfc.detach().clone()
        # rr1_l2_0 = torch.linalg.norm(rr1.view(-1),dim=0)
        # rr1_l2 = 10.
        # go to SFC coarse grid levels and do 1 mg cycles there
        e_i = mg.mg_on_P1CG(
            r1_sfc.view(cg_nonods, ndim+1),
            sf_nd_nb.sfc_data.variables_sfc,
            sf_nd_nb.sfc_data.nlevel,
            sf_nd_nb.sfc_data.nodes_per_level
        )
        # reverse to original order
        e_i = e_i[sf_nd_nb.sfc_data.space_filling_curve_numbering[:, 0] - 1, :].view(cg_nonods, ndim+1)
    if not config.is_pmg:  # from P1CG to P3DG
        # prolongate error to fine grid
        e_i0_vel = torch.zeros(u_nonods, ndim, device=dev, dtype=torch.float64)
        e_i0_pre = torch.zeros(p_nonods, device=dev, dtype=torch.float64)
        for idim in range(ndim):
            e_i0_vel[:, idim] += mg.vel_p1dg_to_pndg_prolongator(torch.mv(sf_nd_nb.I_dc, e_i[:, idim]))
        e_i0_pre += mg.pre_p1dg_to_pndg_prolongator(torch.mv(sf_nd_nb.I_dc, e_i[:, -1]))
    else:  # from P1CG to P1DG, then go one order up each time while also do post smoothing
        raise NotImplemented('P-multigrid down 1 order each time is not implemented for Stokes.')
        # # prolongate error to P1DG
        # ilevel = config.ele_p - 1
        # for idim in range(ndim):
        #     e_p[ilevel][:, idim] += torch.mv(sf_nd_nb.I_fc, e_i[:, idim])
        # r_p, e_p = mg.p_mg_post(e_p, r_p)
        # e_i0 = e_p[0]
    # correct fine grid solution
    x_i[0] += e_i0_vel.view(config.nele, -1, ndim)
    x_i[1] += e_i0_pre.view(config.nele, -1)
    # post smooth
    for its1 in range(config.post_smooth_its):
        r0 *= 0
        r0_list, x_i = get_residual_and_smooth_once(
            r0_list, x_i, x_rhs)
    # r0l2 = torch.linalg.norm(r0.view(-1), dim=0) / r0_init  # fNorm

    return x_i, r0


def multigrid_solver(x_i, p_i, x_rhs, tol):
    """
    use multigrid as a solver
    """
    # get parameter
    u_nonods = sf_nd_nb.vel_func_space.nonods
    p_nonods = sf_nd_nb.pre_func_space.nonods
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    # first do one cycle to get initial residual
    r0_vel = torch.zeros(u_nonods, ndim, device=dev, dtype=torch.float64)
    r0_pre = torch.zeros(p_nonods, device=dev, dtype=torch.float64)
    r0 = [r0_vel, r0_pre]
    x_i, _ = _multigrid_one_cycle(x_i, x_rhs)
    x_rhs = update_rhs(x_rhs, x_i[1])
    p_i += x_i[1].view(p_i.shape)  # p += dp

    r0_vel *= 0
    r0_pre *= 0
    r0 = get_residual_only(r0,
                           x_i, x_rhs)
    r0l2_init = get_r0_l2_norm(r0)
    r0l2 = torch.tensor([1.], device=dev, dtype=torch.float64)

    # now we do MG cycles
    its = 1
    while r0l2 > tol and its < config.jac_its:
        x_i, r = _multigrid_one_cycle(x_i, x_rhs)
        x_rhs = update_rhs(x_rhs, x_i[1])
        p_i += x_i[1].view(p_i.shape)  # p += dp
        x_i[1] *= 0
        r0l2 = get_r0_l2_norm(r) / r0l2_init
        its += 1
        print('its=', its, 'fine grid rel residual l2 norm=', r0l2.cpu().numpy())
    return x_i


def gmres_mg_solver(x_i, p_i, x_rhs,
                    tol):
    """
    use mg left preconditioned gmres as a solver
    """
    u_nonods = sf_nd_nb.vel_func_space.nonods
    p_nonods = sf_nd_nb.pre_func_space.nonods
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    m = config.gmres_m
    v_m = torch.zeros(m+1, u_nonods * ndim + p_nonods, device=dev, dtype=torch.float64)  # V_m
    h_m = torch.zeros(m+1, m, device=dev, dtype=torch.float64)  # \bar H_m
    r0 = torch.zeros(u_nonods * ndim + p_nonods, device=dev, dtype=torch.float64)
    r0_list = [r0[0:u_nonods*ndim], r0[u_nonods*ndim:u_nonods*ndim+p_nonods]]
    dummy = [torch.zeros(u_nonods * ndim, device=dev, dtype=torch.float64),
             torch.zeros(p_nonods, device=dev, dtype=torch.float64)]
    r0l2 = 1.
    its = 0
    e_1 = torch.zeros(m + 1, device=dev, dtype=torch.float64)  # this is a unit vector in the least square prob.
    e_1[0] += 1
    while r0l2 > tol and its < config.gmres_its:
        h_m *= 0
        v_m *= 0
        r0 *= 0
        r0_list = get_residual_only(r0_list,
                                    x_i, x_rhs)
        # r0 = r0.view(nonods, ndim)  # why change view here...?
        r0_list, _ = _multigrid_one_cycle(x_i=[torch.zeros(u_nonods, ndim, device=dev, dtype=torch.float64),
                                          torch.zeros(p_nonods, device=dev, dtype=torch.float64)],
                                          x_rhs=r0_list)
        r0 = r0.view(-1)  # remember r0_list is a 'view' of r0.
        beta = torch.linalg.norm(r0)
        v_m[0, :] += r0 / beta
        w = r0  # this should place w in the same memory as r0 so that we don't take two nonods memory space
        for j in range(0, m):
            w *= 0
            w_list = [w[0:u_nonods*ndim], w[u_nonods*ndim:u_nonods*ndim+p_nonods]]
            w_list = get_residual_only(r0=w_list,
                                       x_i=[v_m[j, 0:u_nonods*ndim],
                                            v_m[j, u_nonods*ndim:u_nonods*ndim+p_nonods]],
                                       x_rhs=dummy)  # providing rhs=0, b-Ax is -Ax
            # w = w.view(nonods, ndim)  # why change view here...?
            w *= -1.
            w_list, _ = _multigrid_one_cycle([torch.zeros(u_nonods, ndim, device=dev, dtype=torch.float64),
                                              torch.zeros(p_nonods, device=dev, dtype=torch.float64)],
                                             # ↑ here I believe we create another nonods memory usage
                                             x_rhs=w_list)
            w = w.view(-1)  # w_list is a 'view' of w
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
        dx_i = torch.einsum('ji,j->i', v_m[0:m, :], y_m)
        x_i[0] += dx_i[0:u_nonods*ndim].view(config.nele, u_nloc, ndim)
        x_i[1] += dx_i[u_nonods*ndim:u_nonods*ndim+p_nonods].view(config.nele, p_nloc)
        # update x_rhs and p
        x_rhs = update_rhs(x_rhs, x_i[1])
        p_i += x_i[1]
        # r0l2 = torch.linalg.norm(q[:, m:m+1].T @ e_1)
        r0 *= 0
        r0_list = get_residual_only(r0_list,
                                    x_i, x_rhs)
        r0 = r0.view(-1)
        r0l2 = torch.linalg.norm(r0)
        print('its=', its, 'fine grid rel residual l2 norm=', r0l2.cpu().numpy())
        its += 1
    return x_i


def gmres_solver(c_i, c_n, c_bc, f,
                 tol):
    """
    use mg left preconditioned gmres as a solver
    """
    raise Exception('GMRES solver not implemented!')
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
