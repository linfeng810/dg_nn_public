#!/usr/bin/env python3

"""
Wrap solvers into one function here.
We have MG solver, and MG-preconditioned GMRES solver.
"""
import torch
import numpy as np
import scipy as sp
from types import NoneType
from tqdm import tqdm
import config
import preconditioner
import volume_mf_he

import volume_mf_st
from config import sf_nd_nb


# nonods = config.nonods
dev = config.dev
# nloc = config.nloc
ndim = config.ndim
nele = config.nele
nele_f = config.nele_f
nele_s = config.nele_s


# def _multigrid_one_cycle(x_i, x_rhs):
#     """
#     given initial vector and right hand side,
#     do one MG cycle.
#     """
#     u_nonods = sf_nd_nb.vel_func_space.nonods
#     p_nonods = sf_nd_nb.pre_func_space.nonods
#     r0 = torch.zeros(u_nonods * ndim + p_nonods, device=dev, dtype=torch.float64)
#     r0_list = [r0[0:u_nonods*ndim],
#                r0[u_nonods*ndim:u_nonods*ndim+p_nonods]]
#     # u_i = u_i.view(nonods, ndim)
#     # du_i = du_i.view(nonods, ndim)
#     x_i[0] = x_i[0].view(u_nonods, ndim)
#     x_i[1] = x_i[1].view(p_nonods)
#
#     cg_nonods = sf_nd_nb.vel_func_space.cg_nonods
#
#     for its1 in range(config.pre_smooth_its):
#         r0 *= 0
#         r0_list, x_i = get_residual_and_smooth_once(
#             r0_list, x_i, x_rhs)
#     # get residual on PnDG
#     r0 *= 0
#     r0_list = get_residual_only(
#         r0_list,
#         x_i, x_rhs)
#
#     if not config.is_pmg:  # PnDG to P1CG
#         r1 = torch.zeros(cg_nonods, ndim+1, device=dev, dtype=torch.float64)
#         for idim in range(ndim):
#             r1[:, idim] += torch.mv(sf_nd_nb.I_cd, mg.vel_pndg_to_p1dg_restrictor(r0_list[0][:, idim]))
#         r1[:, -1] += torch.mv(sf_nd_nb.I_cd, mg.pre_pndg_to_p1dg_restrictor(r0_list[1]))
#     else:  # PnDG down one order each time, eventually go to P1CG
#         raise NotImplemented('P-multigrid down 1 order each time is not implemented for Stokes.')
#         # r_p, e_p = mg.p_mg_pre(r0)
#         # r1 = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
#         # ilevel = config.ele_p - 1
#         # for idim in range(ndim):
#         #     r1[:, idim] += torch.mv(sf_nd_nb.I_cf, r_p[ilevel][:, idim])
#     if not config.is_sfc:  # two-grid method
#         e_i = torch.zeros(cg_nonods, ndim+1, device=dev, dtype=torch.float64)
#         e_direct = sp.sparse.linalg.spsolve(
#             sf_nd_nb.RARmat,
#             r1.contiguous().view(-1).cpu().numpy())
#         e_direct = np.reshape(e_direct, (cg_nonods, ndim))
#         e_i += torch.tensor(e_direct, device=dev, dtype=torch.float64)
#     else:  # multi-grid method
#         ncurve = 1  # always use 1 sfc
#         N = len(sf_nd_nb.sfc_data.space_filling_curve_numbering)
#         inverse_numbering = np.zeros((N, ncurve), dtype=int)
#         inverse_numbering[:, 0] = np.argsort(sf_nd_nb.sfc_data.space_filling_curve_numbering[:, 0])
#         r1_sfc = r1[inverse_numbering[:, 0], :].view(cg_nonods, ndim+1)
#
#         # # if we do the presmooth steps inside mg_on_P1CG, there's no need to pass in rr1 and e_i
#         # e_i = torch.zeros((cg_nonods,1), device=dev, dtype=torch.float64)
#         # rr1 = r1_sfc.detach().clone()
#         # rr1_l2_0 = torch.linalg.norm(rr1.view(-1),dim=0)
#         # rr1_l2 = 10.
#         # go to SFC coarse grid levels and do 1 mg cycles there
#         e_i = mg.mg_on_P1CG(
#             r1_sfc.view(cg_nonods, ndim+1),
#             sf_nd_nb.sfc_data.variables_sfc,
#             sf_nd_nb.sfc_data.nlevel,
#             sf_nd_nb.sfc_data.nodes_per_level
#         )
#         # reverse to original order
#         e_i = e_i[sf_nd_nb.sfc_data.space_filling_curve_numbering[:, 0] - 1, :].view(cg_nonods, ndim+1)
#     if not config.is_pmg:  # from P1CG to P3DG
#         # prolongate error to fine grid
#         e_i0_vel = torch.zeros(u_nonods, ndim, device=dev, dtype=torch.float64)
#         e_i0_pre = torch.zeros(p_nonods, device=dev, dtype=torch.float64)
#         for idim in range(ndim):
#             e_i0_vel[:, idim] += mg.vel_p1dg_to_pndg_prolongator(torch.mv(sf_nd_nb.I_dc, e_i[:, idim]))
#         e_i0_pre += mg.pre_p1dg_to_pndg_prolongator(torch.mv(sf_nd_nb.I_dc, e_i[:, -1]))
#     else:  # from P1CG to P1DG, then go one order up each time while also do post smoothing
#         raise NotImplemented('P-multigrid down 1 order each time is not implemented for Stokes.')
#         # # prolongate error to P1DG
#         # ilevel = config.ele_p - 1
#         # for idim in range(ndim):
#         #     e_p[ilevel][:, idim] += torch.mv(sf_nd_nb.I_fc, e_i[:, idim])
#         # r_p, e_p = mg.p_mg_post(e_p, r_p)
#         # e_i0 = e_p[0]
#     # correct fine grid solution
#     x_i[0] += e_i0_vel.view(config.nele, -1, ndim)
#     x_i[1] += e_i0_pre.view(config.nele, -1)
#     # post smooth
#     for its1 in range(config.post_smooth_its):
#         r0 *= 0
#         r0_list, x_i = get_residual_and_smooth_once(
#             r0_list, x_i, x_rhs)
#     # r0l2 = torch.linalg.norm(r0.view(-1), dim=0) / r0_init  # fNorm
#
#     return x_i, r0
#
#
# def multigrid_solver(x_i, p_i, x_rhs, tol):
#     """
#     use multigrid as a solver
#     """
#     # get parameter
#     u_nonods = sf_nd_nb.vel_func_space.nonods
#     p_nonods = sf_nd_nb.pre_func_space.nonods
#     u_nloc = sf_nd_nb.vel_func_space.element.nloc
#     p_nloc = sf_nd_nb.pre_func_space.element.nloc
#
#     # first do one cycle to get initial residual
#     r0_vel = torch.zeros(u_nonods, ndim, device=dev, dtype=torch.float64)
#     r0_pre = torch.zeros(p_nonods, device=dev, dtype=torch.float64)
#     r0 = [r0_vel, r0_pre]
#     x_i, _ = _multigrid_one_cycle(x_i, x_rhs)
#     x_rhs = update_rhs(x_rhs, x_i[1])
#     p_i += x_i[1].view(p_i.shape)  # p += dp
#
#     r0_vel *= 0
#     r0_pre *= 0
#     r0 = get_residual_only(r0,
#                            x_i, x_rhs)
#     r0l2_init = get_r0_l2_norm(r0)
#     r0l2 = torch.tensor([1.], device=dev, dtype=torch.float64)
#
#     # now we do MG cycles
#     its = 1
#     while r0l2 > tol and its < config.jac_its:
#         x_i, r = _multigrid_one_cycle(x_i, x_rhs)
#         x_rhs = update_rhs(x_rhs, x_i[1])
#         p_i += x_i[1].view(p_i.shape)  # p += dp
#         x_i[1] *= 0
#         r0l2 = get_r0_l2_norm(r) / r0l2_init
#         its += 1
#         print('its=', its, 'fine grid rel residual l2 norm=', r0l2.cpu().numpy())
#     return x_i


def gmres_mg_solver(x_i, x_rhs,
                    tol,
                    nullspace=None,
                    include_adv=False,
                    x_k=None, u_bc=None):
    """
    use mg left preconditioned gmres as a solver
    """
    u_nonods = sf_nd_nb.vel_func_space.nonods
    p_nonods = sf_nd_nb.pre_func_space.nonods
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    real_no_dofs = nele_f * u_nloc * ndim + nele_f * p_nloc + nele_s * u_nloc * ndim
    total_no_dofs = nele * u_nloc * ndim + nele * p_nloc + nele * u_nloc * ndim

    m = config.gmres_m
    v_m = torch.zeros(m+1, real_no_dofs, device=dev, dtype=torch.float64)  # V_m
    v_m_j_full_length = torch.zeros(total_no_dofs, device=dev, dtype=torch.float64)  # full length v vector
    h_m = torch.zeros(m+1, m, device=dev, dtype=torch.float64)  # \bar H_m
    r0 = torch.zeros(total_no_dofs, device=dev, dtype=torch.float64)
    r0_real = torch.zeros(real_no_dofs, device=dev, dtype=torch.float64)
    r0_dict = volume_mf_st.slicing_x_i(r0)

    x_dummy = torch.zeros(total_no_dofs, device=dev, dtype=torch.float64)

    r0l2 = 1.
    sf_nd_nb.its = 0
    e_1 = torch.zeros(m + 1, device=dev, dtype=torch.float64)  # this is a unit vector in the least square prob.
    e_1[0] += 1

    isUpperTriPrecond = True
    while r0l2 > tol and sf_nd_nb.its < config.gmres_its:
        h_m *= 0
        v_m *= 0
        r0 *= 0
        # get residual
        r0 = volume_mf_st.get_residual_only(
            r0, x_i, x_rhs,
            include_adv=include_adv,
            x_k=x_k,
            u_bc=u_bc[0])
        r0 = volume_mf_he.get_residual_only(
            r0, x_k, x_i, x_rhs, include_s_blk=True, include_itf=True,
        )
        # apply left preconditioner
        if isUpperTriPrecond:
            r0 = preconditioner.fsi_precond_all(r0, x_k, u_bc)
        else:
            r0 = preconditioner.fsi_precond_all2(r0, x_k, u_bc)

        r0_real = get_res_vec_from_dict(x_i=r0, x_res=r0_real)
        beta = torch.linalg.norm(r0_real)
        # beta = torch.linalg.norm(r0)
        v_m[0, :] += r0_real / beta
        w = r0  # this should place w in the same memory as r0 so that we don't take two nonods memory space
        w_real = r0_real
        w_dict = volume_mf_st.slicing_x_i(w)
        for j in tqdm(range(0, m), disable=config.disabletqdm):
            v_m_j_full_length *= 0
            v_m_j_full_length = get_dict_from_res_vec(x_res=v_m[j, :], x_i=v_m_j_full_length)
            w *= 0
            # w = A * v_m[j]
            w = volume_mf_st.get_residual_only(
                r0=w,
                x_i=v_m_j_full_length,
                x_rhs=0,
                include_adv=include_adv,
                x_k=x_k,
                u_bc=u_bc[0])
            w = volume_mf_he.get_residual_only(
                r0=w,
                x_k=x_k,
                x_i=v_m_j_full_length,
                x_rhs=0,
                include_s_blk=True,
                include_itf=True,
            )
            w *= -1.  # providing rhs=0, b-Ax is -Ax
            # apply preconditioner
            if isUpperTriPrecond:
                w = preconditioner.fsi_precond_all(x_i=w, x_k=x_k, u_bc=u_bc)
            else:
                w = preconditioner.fsi_precond_all2(x_i=w, x_k=x_k, u_bc=u_bc)
            w_real *= 0
            w_real = get_res_vec_from_dict(x_i=w, x_res=w_real)
            for i in range(0, j+1):
                h_m[i, j] = torch.linalg.vecdot(w_real, v_m[i, :])
                w_real -= h_m[i, j] * v_m[i, :]

            h_m[j+1, j] = torch.linalg.norm(w_real)
            v_m[j+1, :] += w_real / h_m[j+1, j]
            sf_nd_nb.its += 1
        # solve least-square problem
        q, r = torch.linalg.qr(h_m, mode='complete')  # h_m: (m+1)xm, q: (m+1)x(m+1), r: (m+1)xm
        e_1[0] = 0
        e_1[0] += beta
        y_m = torch.linalg.solve(r[0:m, 0:m], q[0:m+1, 0:m].T @ e_1)  # y_m: m
        # update c_i and get residual
        dx_i = torch.einsum('ji,j->i', v_m[0:m, :], y_m)
        x_dummy *= 0
        x_dummy = get_dict_from_res_vec(x_i=x_dummy, x_res=dx_i)
        x_i += x_dummy

        # r0l2 = torch.linalg.norm(q[:, m:m+1].T @ e_1)
        r0 *= 0
        # get residual
        r0 = volume_mf_st.get_residual_only(
            r0, x_i, x_rhs,
            include_adv=include_adv,
            x_k=x_k,
            u_bc=u_bc[0])
        r0 = volume_mf_he.get_residual_only(
            r0, x_k, x_i, x_rhs, include_s_blk=True, include_itf=True,
        )
        # r0 = r0.view(-1)
        r0l2 = volume_mf_st.get_r0_l2_norm(r0)
        print('its=', sf_nd_nb.its, 'fine grid rel residual l2 norm= all, vel, pre, disp', r0l2.cpu().numpy(),
              torch.linalg.norm(r0_dict['vel']).cpu().numpy(),
              torch.linalg.norm(r0_dict['pre']).cpu().numpy(),
              torch.linalg.norm(r0_dict['disp']).cpu().numpy(),
              'isUpperTriPrecond', isUpperTriPrecond)
        # isUpperTriPrecond = not isUpperTriPrecond
    return x_i, sf_nd_nb.its


def partitioned_solver(x_i, x_rhs,
                       tol,
                       include_adv=False,
                       x_k=None, u_bc=None):
    """
    use mg left preconditioned gmres as a solver
    """
    u_nonods = sf_nd_nb.vel_func_space.nonods
    p_nonods = sf_nd_nb.pre_func_space.nonods
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    real_no_dofs = nele_f * u_nloc * ndim + nele_f * p_nloc + nele_s * u_nloc * ndim
    total_no_dofs = nele * u_nloc * ndim + nele * p_nloc + nele * u_nloc * ndim

    m = config.gmres_m
    v_m = torch.zeros(m+1, real_no_dofs, device=dev, dtype=torch.float64)  # V_m
    v_m_j_full_length = torch.zeros(total_no_dofs, device=dev, dtype=torch.float64)  # full length v vector
    h_m = torch.zeros(m+1, m, device=dev, dtype=torch.float64)  # \bar H_m
    r0 = torch.zeros(total_no_dofs, device=dev, dtype=torch.float64)
    r0_real = torch.zeros(real_no_dofs, device=dev, dtype=torch.float64)
    r0_dict = volume_mf_st.slicing_x_i(r0)

    x_dummy = torch.zeros(total_no_dofs, device=dev, dtype=torch.float64)

    r0l2 = 1.
    sf_nd_nb.its = 0
    e_1 = torch.zeros(m + 1, device=dev, dtype=torch.float64)  # this is a unit vector in the least square prob.
    e_1[0] += 1
    while r0l2 > tol and sf_nd_nb.its < config.gmres_its * sf_nd_nb.nits:
        h_m *= 0
        v_m *= 0
        r0 *= 0
        # get residual
        r0 = volume_mf_st.get_residual_only(
            r0, x_i, x_rhs,
            include_adv=include_adv,
            x_k=x_k,
            u_bc=u_bc[0])
        r0 = volume_mf_he.get_residual_only(
            r0, x_k, x_i, x_rhs, include_s_blk=True, include_itf=True,
        )
        # apply left preconditioner
        if sf_nd_nb.nits % 2 == 0:
            r0 = preconditioner.fsi_precond_all(r0, x_k, u_bc)
        else:
            r0 = preconditioner.fsi_precond_all2(r0, x_k, u_bc)

        r0_real = get_res_vec_from_dict(x_i=r0, x_res=r0_real)
        beta = torch.linalg.norm(r0_real)
        # beta = torch.linalg.norm(r0)
        v_m[0, :] += r0_real / beta
        w = r0  # this should place w in the same memory as r0 so that we don't take two nonods memory space
        w_real = r0_real
        w_dict = volume_mf_st.slicing_x_i(w)
        for j in tqdm(range(0, m), disable=config.disabletqdm):
            v_m_j_full_length *= 0
            v_m_j_full_length = get_dict_from_res_vec(x_res=v_m[j, :], x_i=v_m_j_full_length)
            w *= 0
            # w = A * v_m[j]
            w = volume_mf_st.get_residual_only(
                r0=w,
                x_i=v_m_j_full_length,
                x_rhs=0,
                include_adv=include_adv,
                x_k=x_k,
                u_bc=u_bc[0])
            w = volume_mf_he.get_residual_only(
                r0=w,
                x_k=x_k,
                x_i=v_m_j_full_length,
                x_rhs=0,
                include_s_blk=True,
                include_itf=True,
            )
            w *= -1.  # providing rhs=0, b-Ax is -Ax
            # apply preconditioner
            if sf_nd_nb.nits % 2 == 0:
                w = preconditioner.fsi_precond_all(x_i=w, x_k=x_k, u_bc=u_bc)
            else:
                w = preconditioner.fsi_precond_all2(x_i=w, x_k=x_k, u_bc=u_bc)
            w_real *= 0
            w_real = get_res_vec_from_dict(x_i=w, x_res=w_real)
            for i in range(0, j+1):
                h_m[i, j] = torch.linalg.vecdot(w_real, v_m[i, :])
                w_real -= h_m[i, j] * v_m[i, :]

            h_m[j+1, j] = torch.linalg.norm(w_real)
            v_m[j+1, :] += w_real / h_m[j+1, j]
            sf_nd_nb.its += 1
        # solve least-square problem
        q, r = torch.linalg.qr(h_m, mode='complete')  # h_m: (m+1)xm, q: (m+1)x(m+1), r: (m+1)xm
        e_1[0] = 0
        e_1[0] += beta
        y_m = torch.linalg.solve(r[0:m, 0:m], q[0:m+1, 0:m].T @ e_1)  # y_m: m
        # update c_i and get residual
        dx_i = torch.einsum('ji,j->i', v_m[0:m, :], y_m)
        x_dummy *= 0
        x_dummy = get_dict_from_res_vec(x_i=x_dummy, x_res=dx_i)
        x_i += x_dummy

        # r0l2 = torch.linalg.norm(q[:, m:m+1].T @ e_1)
        r0 *= 0
        # get residual
        r0 = volume_mf_st.get_residual_only(
            r0, x_i, x_rhs,
            include_adv=include_adv,
            x_k=x_k,
            u_bc=u_bc[0])
        r0 = volume_mf_he.get_residual_only(
            r0, x_k, x_i, x_rhs, include_s_blk=True, include_itf=True,
        )
        # r0 = r0.view(-1)
        r0l2 = volume_mf_st.get_r0_l2_norm(r0)
        print('its=', sf_nd_nb.its, 'fine grid rel residual l2 norm= all, vel, pre, disp', r0l2.cpu().numpy(),
              torch.linalg.norm(r0_dict['vel']).cpu().numpy(),
              torch.linalg.norm(r0_dict['pre']).cpu().numpy(),
              torch.linalg.norm(r0_dict['disp']).cpu().numpy())
    return x_i, sf_nd_nb.its


def fluid_gmres_mg_solver(x_i, x_rhs,
                          tol,
                          x_k, u_bc):
    """
    solve fluid subdomain only
    this is part of the partitioned solver
    """
    u_nonods = sf_nd_nb.vel_func_space.nonods
    p_nonods = sf_nd_nb.pre_func_space.nonods
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    real_no_dofs = nele_f * u_nloc * ndim + nele_f * p_nloc
    total_no_dofs = nele * u_nloc * ndim + nele * p_nloc + nele * u_nloc * ndim

    m = config.gmres_m
    v_m = torch.zeros(m + 1, real_no_dofs, device=dev, dtype=torch.float64)  # V_m
    v_m_j_full_length = torch.zeros(total_no_dofs, device=dev, dtype=torch.float64)  # full length v vector
    h_m = torch.zeros(m + 1, m, device=dev, dtype=torch.float64)  # \bar H_m
    r0 = torch.zeros(total_no_dofs, device=dev, dtype=torch.float64)
    r0_real = torch.zeros(real_no_dofs, device=dev, dtype=torch.float64)
    r0_dict = volume_mf_st.slicing_x_i(r0)

    x_dummy = torch.zeros(total_no_dofs, device=dev, dtype=torch.float64)

    r0l2 = 1.
    sf_nd_nb.its = 0
    e_1 = torch.zeros(m + 1, device=dev, dtype=torch.float64)  # this is a unit vector in the least square prob.
    e_1[0] += 1

    while r0l2 > tol and sf_nd_nb.its < config.gmres_its * sf_nd_nb.nits:
        h_m *= 0
        v_m *= 0
        r0 *= 0
        # get residual
        r0 = volume_mf_st.get_residual_only(
            r0, x_i, x_rhs,
            include_adv=True,
            x_k=x_k,
            u_bc=u_bc[0])
        # apply left preconditioner
        ...

        r0_real = get_res_vec_from_dict(x_i=r0, x_res=r0_real)
        beta = torch.linalg.norm(r0_real)
        # beta = torch.linalg.norm(r0)
        v_m[0, :] += r0_real / beta
        w = r0  # this should place w in the same memory as r0 so that we don't take two nonods memory space
        w_real = r0_real
        w_dict = volume_mf_st.slicing_x_i(w)
        for j in tqdm(range(0, m), disable=config.disabletqdm):
            v_m_j_full_length *= 0
            v_m_j_full_length = get_dict_from_res_vec(x_res=v_m[j, :], x_i=v_m_j_full_length)
            w *= 0
            # w = A * v_m[j]
            w = volume_mf_st.get_residual_only(
                r0=w,
                x_i=v_m_j_full_length,
                x_rhs=0,
                include_adv=True,
                x_k=x_k,
                u_bc=u_bc[0])
            w = volume_mf_he.get_residual_only(
                r0=w,
                x_k=x_k,
                x_i=v_m_j_full_length,
                x_rhs=0,
                include_s_blk=True,
                include_itf=True,
            )
            w *= -1.  # providing rhs=0, b-Ax is -Ax
            # apply preconditioner
            if sf_nd_nb.nits % 2 == 0:
                w = preconditioner.fsi_precond_all(x_i=w, x_k=x_k, u_bc=u_bc)
            else:
                w = preconditioner.fsi_precond_all2(x_i=w, x_k=x_k, u_bc=u_bc)
            w_real *= 0
            w_real = get_res_vec_from_dict(x_i=w, x_res=w_real)
            for i in range(0, j+1):
                h_m[i, j] = torch.linalg.vecdot(w_real, v_m[i, :])
                w_real -= h_m[i, j] * v_m[i, :]

            h_m[j+1, j] = torch.linalg.norm(w_real)
            v_m[j+1, :] += w_real / h_m[j+1, j]
            sf_nd_nb.its += 1
        # solve least-square problem
        q, r = torch.linalg.qr(h_m, mode='complete')  # h_m: (m+1)xm, q: (m+1)x(m+1), r: (m+1)xm
        e_1[0] = 0
        e_1[0] += beta
        y_m = torch.linalg.solve(r[0:m, 0:m], q[0:m+1, 0:m].T @ e_1)  # y_m: m
        # update c_i and get residual
        dx_i = torch.einsum('ji,j->i', v_m[0:m, :], y_m)
        x_dummy *= 0
        x_dummy = get_dict_from_res_vec(x_i=x_dummy, x_res=dx_i)
        x_i += x_dummy

        # r0l2 = torch.linalg.norm(q[:, m:m+1].T @ e_1)
        r0 *= 0
        # get residual
        r0 = volume_mf_st.get_residual_only(
            r0, x_i, x_rhs,
            include_adv=True,
            x_k=x_k,
            u_bc=u_bc[0])
        r0 = volume_mf_he.get_residual_only(
            r0, x_k, x_i, x_rhs, include_s_blk=True, include_itf=True,
        )
        # r0 = r0.view(-1)
        r0l2 = volume_mf_st.get_r0_l2_norm(r0)
        print('its=', sf_nd_nb.its, 'fine grid rel residual l2 norm= all, vel, pre, disp', r0l2.cpu().numpy(),
              torch.linalg.norm(r0_dict['vel']).cpu().numpy(),
              torch.linalg.norm(r0_dict['pre']).cpu().numpy(),
              torch.linalg.norm(r0_dict['disp']).cpu().numpy())
    return x_i


def solid_gmres_mg_solver():
    ...


# def right_gmres_mg_solver(
#         x_i, x_rhs,
#         tol,
#         nullspace=None):
#     """
#     use mg right preconditioned gmres as a solver
#     """
#     u_nonods = sf_nd_nb.vel_func_space.nonods
#     p_nonods = sf_nd_nb.pre_func_space.nonods
#     u_nloc = sf_nd_nb.vel_func_space.element.nloc
#     p_nloc = sf_nd_nb.pre_func_space.element.nloc
#
#     m = config.gmres_m
#     v_m = torch.zeros(m+1, u_nonods * ndim + p_nonods, device=dev, dtype=torch.float64)  # V_m
#     h_m = torch.zeros(m+1, m, device=dev, dtype=torch.float64)  # \bar H_m
#     r0 = torch.zeros(u_nonods * ndim + p_nonods, device=dev, dtype=torch.float64)
#     r0_u, r0_p = volume_mf_st.slicing_x_i(r0)
#     x_dummy = torch.zeros(u_nonods * ndim + p_nonods, device=dev, dtype=torch.float64)
#     x_u_dummy, x_p_dummy = volume_mf_st.slicing_x_i(x_dummy)
#     r0l2 = 1.
#     its = 0
#     e_1 = torch.zeros(m + 1, device=dev, dtype=torch.float64)  # this is a unit vector in the least square prob.
#     e_1[0] += 1
#     while r0l2 > tol and its < config.gmres_its:
#         h_m *= 0
#         v_m *= 0
#         r0 *= 0
#         r0 = get_residual_only(r0,
#                                x_i, x_rhs)
#         # r0 = r0.view(nonods, ndim)  # why change view here...?
#         # # apply preconditioner
#         # x_u_dummy *= 0
#         # for icycle in range(4):
#         #     x_u_dummy = volume_mf_st.vel_blk_precon(
#         #         x_u_dummy,
#         #         r0_u)
#         # r0_u *= 0
#         # r0_u += x_u_dummy
#         # # print('r0_u norm: ', torch.linalg.norm(r0_u.view(-1)))
#         # # r0_u = volume_mf_st.vel_blk_precon_direct_inv(r0_u)
#         # r0_p = volume_mf_st.pre_blk_precon(r0_p)
#
#         # remove null space
#         if type(nullspace) is not NoneType:
#             r0 = remove_nullspace(r0, nullspace)
#         beta = torch.linalg.norm(r0)
#         v_m[0, :] += r0 / beta
#         w = r0  # this should place w in the same memory as r0 so that we don't take two nonods memory space
#         w_u, w_p = volume_mf_st.slicing_x_i(w)
#         for j in tqdm(range(0, m)):
#             w *= 0
#             # apply right preconditioner
#             w_u = volume_mf_st.vel_blk_precon(
#                 x_u=w_u,
#                 x_rhs=v_m[j, 0:u_nonods * ndim],
#             )
#             w_p += v_m[j, u_nonods * ndim:u_nonods * ndim + p_nonods].view(w_p.shape)
#             w_p = volume_mf_st.pre_blk_precon(w_p)
#             x_dummy *= 0
#             x_dummy = get_residual_only(
#                 r0=x_dummy,
#                 x_i=torch.zeros(u_nonods * ndim + p_nonods, device=dev, dtype=torch.float64),
#                 x_rhs=w)  # providing rhs=0, b-Ax is -Ax
#             x_dummy *= -1.
#             w *= 0
#             w += x_dummy
#             for i in range(0, j+1):
#                 h_m[i, j] = torch.linalg.vecdot(w, v_m[i, :])
#                 w -= h_m[i, j] * v_m[i, :]
#
#             # remove null space
#             if type(nullspace) is not NoneType:
#                 w = remove_nullspace(w, nullspace)
#             h_m[j+1, j] = torch.linalg.norm(w)
#             v_m[j+1, :] += w / h_m[j+1, j]
#         # solve least-square problem
#         q, r = torch.linalg.qr(h_m, mode='complete')  # h_m: (m+1)xm, q: (m+1)x(m+1), r: (m+1)xm
#         e_1[0] = 0
#         e_1[0] += beta
#         y_m = torch.linalg.solve(r[0:m, 0:m], q[0:m+1, 0:m].T @ e_1)  # y_m: m
#         # update c_i and get residual
#         dx_i = torch.einsum('ji,j->i', v_m[0:m, :], y_m)
#         # apply right preconditioner
#         x_dummy *= 0
#         x_u_dummy = volume_mf_st.vel_blk_precon(
#             x_u=x_u_dummy,
#             x_rhs=dx_i[0:u_nonods * ndim],
#         )
#         x_p_dummy += dx_i[u_nonods * ndim:u_nonods * ndim + p_nonods].view(x_p_dummy.shape)
#         x_p_dummy = volume_mf_st.pre_blk_precon(x_p_dummy)
#         x_i += x_dummy
#
#         # r0l2 = torch.linalg.norm(q[:, m:m+1].T @ e_1)
#         r0 *= 0
#         r0 = get_residual_only(r0,
#                                x_i, x_rhs)
#         # r0 = r0.view(-1)
#         r0l2 = torch.linalg.norm(r0)
#         print('its=', its, 'fine grid rel residual l2 norm=', r0l2.cpu().numpy())
#         its += 1
#     return x_i
#
#
# def gmres_solver(c_i, c_n, c_bc, f,
#                  tol):
#     """
#     use mg left preconditioned gmres as a solver
#     """
#     raise Exception('GMRES solver not implemented!')
#     c_i = c_i.view(-1)
#     m = config.gmres_m
#     v_m = torch.zeros(m+1, nonods, device=dev, dtype=torch.float64)  # V_m
#     h_m = torch.zeros(m+1, m, device=dev, dtype=torch.float64)  # \bar H_m
#     r0 = torch.zeros(nonods, device=dev, dtype=torch.float64)
#     dummy = torch.zeros(nonods, device=dev, dtype=torch.float64)
#     r0l2 = 1.
#     its = 0
#     e_1 = torch.zeros(m + 1, device=dev, dtype=torch.float64)
#     e_1[0] += 1
#     while r0l2 > tol and its < config.gmres_its:
#         h_m *= 0
#         v_m *= 0
#         r0 *= 0
#         r0 = get_residual_only(r0,
#                                c_i, c_n, c_bc, f).view(-1)
#         beta = torch.linalg.norm(r0)
#         v_m[0, :] += r0 / beta
#         w = r0  # this should place w in the same memory as r0 so that we don't take two nonods memory space
#         for j in range(0, m):
#             w *= 0
#             w = get_residual_only(r0=w,
#                                   u_i=v_m[j, :],
#                                   u_n=dummy,
#                                   u_bc=dummy,
#                                   f=dummy,
#                                   u_rhs=dummy).view(-1)  # providing rhs=0, b-Ax is -Ax
#             w *= -1.
#             for i in range(0, j+1):
#                 h_m[i, j] = torch.linalg.vecdot(w, v_m[i, :])
#                 w -= h_m[i, j] * v_m[i, :]
#             h_m[j+1, j] = torch.linalg.norm(w)
#             v_m[j+1, :] += w / h_m[j+1, j]
#         # solve least-square problem
#         q, r = torch.linalg.qr(h_m, mode='complete')  # h_m: (m+1)xm, q: (m+1)x(m+1), r: (m+1)xm
#         e_1 *= 0
#         e_1[0] += beta
#         y_m = torch.linalg.solve(r[0:m, 0:m], q[0:m+1, 0:m].T @ e_1)  # y_m: m
#         # update c_i and get residual
#         u_i += torch.einsum('ji,j->i', v_m[0:m, :], y_m)
#         r0l2 = torch.linalg.norm(q[:, m:m+1].T @ e_1)
#         # r0 *= 0
#         # r0 = get_residual_only(r0,
#         #                        c_i, c_n, c_bc, f)
#         # r0l2 = torch.linalg.norm(r0)
#         print('its=', its, 'fine grid rel residual l2 norm=', r0l2.cpu().numpy())
#         its += 1
#     return c_i


def remove_nullspace(r, n):
    """remove nullspace n from r"""
    dim_null = n.shape[0]
    for dim in range(dim_null):
        r -= torch.inner(r, n[dim, :]) * n[dim, :]
    return r


def get_res_vec_from_dict(x_i, x_res, fsi=0):
    """
    get residual vector from a vector of length total_no_dofs.

    input:

    x_i: length (nele * u_nloc * ndim + nele * p_nloc + nele * u_nloc * ndim)

    output:

    x_res: a tensor of length (nele_f * u_nloc * ndim + nele_f * p_nloc + nele_s * u_nloc * ndim)

    flag:

    fsi == 0 -- both fluid and solid subdomain
    fsi == 1 -- fluid subdomain only
    fsi == 2 -- solid subdomain only
    """
    x_res *= 0
    x_in_shape = x_res.shape
    x_res = x_res.view(-1)

    x_i_dict = volume_mf_st.slicing_x_i(x_i)
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    if fsi == 0:
        x_res[0:nele_f * u_nloc * ndim] += x_i_dict['vel'][0:nele_f, ...].view(-1)
        x_res[nele_f * u_nloc * ndim:
              nele_f * u_nloc * ndim + nele_f * p_nloc] += \
            x_i_dict['pre'][0:nele_f, ...].view(-1)
        x_res[nele_f * (u_nloc * ndim + p_nloc):
              nele_f * (u_nloc * ndim + p_nloc) + nele_s * u_nloc * ndim] += \
            x_i_dict['disp'][nele_f:nele, ...].view(-1)
    elif fsi == 1:
        x_res[0:nele_f * u_nloc * ndim] += x_i_dict['vel'][0:nele_f, ...].view(-1)
        x_res[nele_f * u_nloc * ndim:
              nele_f * u_nloc * ndim + nele_f * p_nloc] += \
            x_i_dict['pre'][0:nele_f, ...].view(-1)
    elif fsi == 2:
        x_res[0:nele_s * u_nloc * ndim] += \
            x_i_dict['disp'][nele_f:nele, ...].view(-1)
    return x_res.view(x_in_shape)


def get_dict_from_res_vec(x_res, x_i, fsi=0):
    """
    create a vector of length total_no_dofs from a residual vector of length real_no_dofs

    input:
    x_res: a tensor of length (nele_f * u_nloc * ndim + nele_f * p_nloc + nele_s * u_nloc * ndim)

    output:
    x_i_dict: store x_res in correspoding positions and leave other positions zero

    flag:
    fsi = 0 -- both fluid and solid subdomain
    fsi = 1 -- fluid subdomain only
    fsi = 2 -- solid subdomain only
    """
    x_res = x_res.view(-1)
    x_i *= 0
    x_i_dict = volume_mf_st.slicing_x_i(x_i)

    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    if fsi == 1:
        x_i_dict['vel'][0:nele_f, ...] += x_res[0:nele_f * u_nloc * ndim].view(nele_f, u_nloc, ndim)
        x_i_dict['pre'][0:nele_f, ...] += x_res[nele_f * u_nloc * ndim:
                                                nele_f * u_nloc * ndim + nele_f * p_nloc].view(nele_f, p_nloc)
    elif fsi == 2:
        x_i_dict['disp'][nele_f:nele, ...] += x_res.view(nele_s, u_nloc, ndim)
    else:
        x_i_dict['vel'][0:nele_f, ...] += x_res[0:nele_f * u_nloc * ndim].view(nele_f, u_nloc, ndim)
        x_i_dict['pre'][0:nele_f, ...] += x_res[nele_f * u_nloc * ndim:
                                                nele_f * u_nloc * ndim + nele_f * p_nloc].view(nele_f, p_nloc)
        x_i_dict['disp'][nele_f:nele, ...] += x_res[nele_f * (u_nloc * ndim + p_nloc):
                                                    nele_f * (u_nloc * ndim + p_nloc)
                                                    + nele_s * u_nloc * ndim].view(nele_s, u_nloc, ndim)

    return x_i.view(-1)
