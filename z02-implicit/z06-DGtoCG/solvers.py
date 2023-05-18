#!/usr/bin/env python3

"""
Wrap solvers into one function here.
We have MG solver, and MG-preconditioned GMRES solver.
"""
import torch
import numpy as np
import scipy as sp
import config
import multi_grid
from volume_integral import get_residual_only, get_residual_and_smooth_once
from config import sf_nd_nb


nonods = config.nonods
dev = config.dev
nloc = config.nloc


def _multigrid_one_cycle(c_i, c_n, c_bc, f, c_rhs):
    """
    given initial vector and right hand side,
    do one MG cycle.
    TODO: MG determined by config parameters:
        - pmg?
        - amg? (SFC)
        - which levels of SFCs to visit?
        - pre/post smooth step?
    """
    r0 = torch.zeros(nonods, device=dev, dtype=torch.float64)
    c_i = c_i.view(-1, 1, nloc)
    cg_nonods = sf_nd_nb.cg_nonods

    for its1 in range(config.pre_smooth_its):
        # on fine grid
        r0 *= 0
        r0, c_i = get_residual_and_smooth_once(
            r0,
            c_i, c_n, c_bc, f, c_rhs)

    # residual on PnDG
    r0 *= 0
    r0 = get_residual_only(
        r0,
        c_i, c_n, c_bc, f, c_rhs)

    if not config.is_pmg:
        # P1DG to P1CG
        r1 = multi_grid.p3dg_to_p1dg_restrictor(r0)
        r1 = torch.matmul(sf_nd_nb.I_cf, r1)
    else:  # do visit each p grid (p-multigrid)
        r_p, e_p = multi_grid.p_mg_pre(r0)
        # now restrict P1DG to P1CG
        ilevel = config.ele_p - 1
        r1 = torch.mv(sf_nd_nb.I_cf, r_p[ilevel])

    e_i = torch.zeros((cg_nonods, 1), device=dev, dtype=torch.float64)
    # rr1 = r1_sfc.detach().clone()
    # rr1_l2_0 = torch.linalg.norm(rr1.view(-1), dim=0)
    # rr1_l2 = 10.
    its1 = 0
    # while its1 < config.mg_its[0] and rr1_l2 > config.mg_tol:
    if config.is_sfc:  # smooth on P1CG
        # reordering node according to SFC
        ncurve = 1  # always use 1 sfc
        N = len(sf_nd_nb.sfc_data.space_filling_curve_numbering)
        inverse_numbering = np.zeros((N, ncurve), dtype=int)
        inverse_numbering[:, 0] = np.argsort(sf_nd_nb.sfc_data.space_filling_curve_numbering[:, 0])
        r1_sfc = r1[inverse_numbering[:, 0]].view(1, 1, cg_nonods)
        for _ in range(config.pre_smooth_its):
            # smooth (solve) on level 1 coarse grid (R^T A R e = r1)
            rr1 = r1_sfc - torch.sparse.mm(sf_nd_nb.sfc_data.variables_sfc[0][0], e_i).view(-1)
            # rr1_l2 = torch.linalg.norm(rr1.view(-1), dim=0) / rr1_l2_0

            diagA1 = 1. / sf_nd_nb.sfc_data.variables_sfc[0][2]
            e_i = e_i.view(cg_nonods, 1) + config.jac_wei * torch.mul(diagA1, rr1.view(-1)).view(-1, 1)
        # for _ in range(100):
        if True:  # SFC multi-grid saw-tooth iteration
            rr1 = r1_sfc - torch.sparse.mm(sf_nd_nb.sfc_data.variables_sfc[0][0], e_i).view(-1)
            # rr1_l2 = torch.linalg.norm(rr1.view(-1), dim=0) / rr1_l2_0
            # use SFC to generate a series of coarse grid
            # and iterate there (V-cycle saw-tooth fasion)
            # then return a residual on level-1 grid (P1CG)
            e_i = multi_grid.mg_on_P1CG(
                r1_sfc.view(cg_nonods, 1),
                rr1,
                e_i,
                sf_nd_nb.sfc_data.space_filling_curve_numbering,
                sf_nd_nb.sfc_data.variables_sfc,
                sf_nd_nb.sfc_data.nlevel,
                sf_nd_nb.sfc_data.nodes_per_level)

        else:  # direct solver on first SFC coarsened grid (thus constitutes a 3-level MG)
            rr1 = r1_sfc - torch.sparse.mm(variables_sfc[0][0], e_i).view(-1)
            # rr1_l2 = torch.linalg.norm(rr1.view(-1), dim=0) / rr1_l2_0
            level = 1

            # restrict residual
            sfc_restrictor = torch.nn.Conv1d(in_channels=1,
                                             out_channels=1, kernel_size=2,
                                             stride=2, padding='valid', bias=False)
            sfc_restrictor.weight.data = \
                torch.tensor([[1., 1.]],
                             dtype=torch.float64,
                             device=config.dev).view(1, 1, 2)
            b = torch.nn.functional.pad(rr1, (0, 1), "constant", 0)  # residual on level1 sfc coarse grid
            with torch.no_grad():
                b = sfc_restrictor(b)
                # np.savetxt('b.txt', b.view(-1).cpu().numpy(), delimiter=',')
            # get operator
            a_sfc_l1 = variables_sfc[level][0]
            cola = a_sfc_l1.col_indices().detach().clone().cpu().numpy()
            fina = a_sfc_l1.crow_indices().detach().clone().cpu().numpy()
            vals = a_sfc_l1.values().detach().clone().cpu().numpy()
            # direct solve on level1 sfc coarse grid
            from scipy.sparse import csr_matrix, linalg
            # np.savetxt('a_sfc_l0.txt', variables_sfc[0][0].to_dense().cpu().numpy(), delimiter=',')
            # np.savetxt('a_sfc_l1.txt', variables_sfc[1][0].to_dense().cpu().numpy(), delimiter=',')
            a_on_l1 = csr_matrix((vals, cola, fina), shape=(nodes_per_level[level], nodes_per_level[level]))
            # np.savetxt('a_sfc_l1_sp.txt', a_on_l1.todense(), delimiter=',')
            e_i_direct = linalg.spsolve(a_on_l1, b.view(-1).cpu().numpy())
            # prolongation
            e_ip1 = torch.tensor(e_i_direct, device=config.dev, dtype=torch.float64).view(1, 1, -1)
            CNN1D_prol_odd = torch.nn.Upsample(scale_factor=nodes_per_level[level - 1] / nodes_per_level[level])
            e_ip1 = CNN1D_prol_odd(e_ip1.view(1, 1, -1))
            e_i += torch.squeeze(e_ip1).view(cg_nonods, 1)
            # e_i += e_ip1[0, 0, space_filling_curve_numbering[:, 0] - 1].view(-1,1)

            for _ in range(config.post_smooth_its):
                # smooth (solve) on level 1 coarse grid (R^T A R e = r1)
                rr1 = r1_sfc - torch.sparse.mm(variables_sfc[0][0], e_i).view(-1)
                # rr1_l2 = torch.linalg.norm(rr1.view(-1), dim=0) / rr1_l2_0

                diagA1 = 1. / variables_sfc[0][2]
                e_i = e_i.view(cg_nonods, 1) + config.jac_wei * torch.mul(diagA1, rr1.view(-1)).view(-1, 1)

        its1 += 1
        # print('its1: %d, residual on P1CG: '%(its1), rr1_l2)
        # reverse to original order
        e_i = e_i[sf_nd_nb.sfc_data.space_filling_curve_numbering[:, 0] - 1, 0].view(-1, 1)

    else:  # direc solve on P1CG R^T A R e = r1?
        e_direct = sp.sparse.linalg.spsolve(sf_nd_nb.RARmat, r1.contiguous().view(-1).cpu().numpy())
        # np.savetxt('RARmat.txt', RARmat.toarray(), delimiter=',')
        e_i = e_i.view(-1)
        # print(e_i - torch.tensor(e_direct, device=dev, dtype=torch.float64))
        e_i += torch.tensor(e_direct, device=dev, dtype=torch.float64)

    if not config.is_pmg:  # from P1CG to P1DG
        e_i0 = torch.mv(sf_nd_nb.I_fc, e_i.view(-1))
        e_i0 = multi_grid.p1dg_to_p3dg_prolongator(e_i0)
    else:  # do visit all p grids (p multi-grid)
        # prolongate error from P1CG to P1DG
        ilevel = config.ele_p - 1
        e_p[ilevel] += torch.mv(sf_nd_nb.I_fc, e_i.view(-1))
        r_p, e_p = multi_grid.p_mg_post(e_p, r_p)
        e_i0 = e_p[0]
    c_i = c_i.view(-1) + e_i0.view(-1)

    for _ in range(config.post_smooth_its):
        # post smooth
        r0 *= 0
        r0, c_i = get_residual_and_smooth_once(
            r0,
            c_i, c_n, c_bc, f, c_rhs)
    return c_i, r0


def multigrid_solver(c_i, c_n, c_bc, f,
                     tol):
    """
    use multigrid as a solver
    """
    # if transient, c_i is the field value of last time-step.
    # otherwise, it is zero.

    # first do one cycle to get initial residual
    r0 = torch.zeros(nonods, device=dev, dtype=torch.float64)
    dummy = torch.zeros(nonods, device=dev, dtype=torch.float64)
    dummy = get_rhs(c_n, c_bc, f)
    c_n *= 0
    c_bc *= 0
    f *= 0
    c_i, _ = _multigrid_one_cycle(c_i, c_n, c_bc, f, c_rhs=dummy)
    r0 *= 0
    r0 = get_residual_only(r0,
                           c_i, c_n, c_bc, f, c_rhs=dummy)
    r0l2_init = torch.linalg.norm(r0.view(-1), dim=0)
    r0l2 = torch.tensor([1.], device=dev, dtype=torch.float64)

    # now we do MG cycles
    its = 1
    while r0l2 > tol and its < config.jac_its:
        c_i, r = _multigrid_one_cycle(c_i, c_n, c_bc, f, c_rhs=dummy)
        # r0 *= 0
        # r0 = get_residual_only(r0,
        #                        c_i, c_n, c_bc, f, c_rhs=dummy)
        r0l2 = torch.linalg.norm(r.view(-1), dim=0) / r0l2_init
        its += 1
        print('its=', its, 'fine grid rel residual l2 norm=', r0l2.cpu().numpy())
    return c_i


def gmres_mg_solver(c_i, c_n, c_bc, f,
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
                               c_i, c_n, c_bc, f)
        r0, _ = _multigrid_one_cycle(c_i=torch.zeros(nonods, device=dev, dtype=torch.float64),
                                     c_n=dummy,
                                     c_bc=dummy,
                                     f=dummy,
                                     c_rhs=r0)
        beta = torch.linalg.norm(r0)
        v_m[0, :] += r0 / beta
        w = r0  # this should place w in the same memory as r0 so that we don't take two nonods memory space
        for j in range(0, m):
            w *= 0
            w = get_residual_only(r0=w,
                                  c_i=v_m[j, :],
                                  c_n=dummy,
                                  c_bc=dummy,
                                  f=dummy,
                                  c_rhs=dummy)  # providing rhs=0, b-Ax is -Ax
            w *= -1.
            w, _ = _multigrid_one_cycle(c_i=torch.zeros(nonods, device=dev, dtype=torch.float64),
                                        # ↑ here I believe we create another nonods memory usage
                                        c_n=dummy,
                                        c_bc=dummy,
                                        f=dummy,
                                        c_rhs=w)
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
        c_i += torch.einsum('ji,j->i', v_m[0:m, :], y_m)
        r0l2 = torch.linalg.norm(q[:, m:m+1].T @ e_1)
        r0 *= 0
        r0 = get_residual_only(r0,
                               c_i, c_n, c_bc, f)
        r0l2 = torch.linalg.norm(r0)
        print('its=', its, 'fine grid rel residual l2 norm=', r0l2.cpu().numpy())
        its += 1
    return c_i


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
                               c_i, c_n, c_bc, f)
        beta = torch.linalg.norm(r0)
        v_m[0, :] += r0 / beta
        w = r0  # this should place w in the same memory as r0 so that we don't take two nonods memory space
        for j in range(0, m):
            w *= 0
            w = get_residual_only(r0=w,
                                  c_i=v_m[j, :],
                                  c_n=dummy,
                                  c_bc=dummy,
                                  f=dummy,
                                  c_rhs=dummy)  # providing rhs=0, b-Ax is -Ax
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
        c_i += torch.einsum('ji,j->i', v_m[0:m, :], y_m)
        r0l2 = torch.linalg.norm(q[:, m:m+1].T @ e_1)
        # r0 *= 0
        # r0 = get_residual_only(r0,
        #                        c_i, c_n, c_bc, f)
        # r0l2 = torch.linalg.norm(r0)
        print('its=', its, 'fine grid rel residual l2 norm=', r0l2.cpu().numpy())
        its += 1
    return c_i


def get_rhs(c_n, c_bc, f):
    """
    get right-hand side at each time step.
    """
    c_rhs = torch.zeros(nonods, device=dev, dtype=torch.float64)
    dummy = torch.zeros(nonods, device=dev, dtype=torch.float64)
    c_rhs = get_residual_only(c_rhs, dummy, c_n, c_bc, f)
    return c_rhs
