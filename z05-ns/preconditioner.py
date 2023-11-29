""" collect all preconditioners here. """

import torch
import numpy as np
import scipy as sp
import config
import volume_mf_st  # stokes i.e. fluid (actually also contains advetion so it's navier-stokes)
import volume_mf_he  # hyper-elastic i.e. solid
from config import sf_nd_nb
import multigrid_linearelastic as mg

nele = config.nele
nele_f = config.nele_f
nele_s = config.nele_s
dev = config.dev
ndim = config.ndim


def disp_precond_all(x_i, x_rhs, x_k, isReturnResidual=False):
    """
    given initial vector and right hand side,
    do one MG cycle,
    for the solid block (displacement)
    """
    u_nloc = sf_nd_nb.disp_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    total_no_dofs = nele * u_nloc * ndim * 2 + nele * p_nloc

    r0 = torch.zeros(total_no_dofs, device=dev, dtype=torch.float64)
    # u_i = u_i.view(nonods, ndim)
    # du_i = du_i.view(nonods, ndim)
    r0_dict = volume_mf_st.slicing_x_i(r0)
    x_i_dict = volume_mf_st.slicing_x_i(x_i)
    x_k_dict = volume_mf_st.slicing_x_i(x_k)
    cg_nonods = sf_nd_nb.sparse_s.cg_nonods

    # pre smooth
    for its1 in range(config.pre_smooth_its):
        r0 *= 0
        r0, x_i = volume_mf_he.get_residual_and_smooth_once(
            r0, x_k, x_i, x_rhs, include_s_blk=True, include_itf=False
        )
    # get residual on PnDG
    r0 *= 0
    r0 = volume_mf_he.get_residual_only(
        r0, x_k, x_i, x_rhs, include_s_blk=True, include_itf=False)

    if not config.is_pmg:  # PnDG to P1CG
        r1 = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
        for idim in range(ndim):
            r1[:, idim] += torch.mv(
                sf_nd_nb.sparse_s.I_cf,
                mg.vel_pndg_to_p1dg_restrictor(r0_dict['disp'][nele_f:nele, :, idim]).view(-1)
            )
    else:  # PnDG down one order each time, eventually go to P1CG
        raise NotImplementedError('pmg not implemented!')
        # r_p, e_p = mg.p_mg_pre(r0)
        # r1 = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
        # ilevel = config.ele_p - 1
        # for idim in range(ndim):
        #     r1[:, idim] += torch.mv(sf_nd_nb.I_cf, r_p[ilevel][:, idim])
    if config.mg_opt_S == 1:  # two-grid method
        e_i = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
        e_direct = sp.sparse.linalg.spsolve(
            sf_nd_nb.RARmat_S,
            r1.contiguous().view(-1).cpu().numpy())
        e_direct = np.reshape(e_direct, (cg_nonods, ndim))
        e_i += torch.tensor(e_direct, device=dev, dtype=torch.float64)
    elif config.mg_opt_S == 3:  # amg method
            e_i = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
            e_direct = sf_nd_nb.RARmat_S.solve(
                r1.contiguous().view(-1).cpu().numpy(),
                maxiter=1,  # do one cycle
                tol=1e-10,
            )
            e_direct = np.reshape(e_direct, (cg_nonods, ndim))
            e_i += torch.tensor(e_direct, device=dev, dtype=torch.float64)
    elif config.mg_opt_S == 4:  # amg method but using pytorch
        e_i = sf_nd_nb.RARmat_S.solve(
            r1.contiguous().view(-1),
            maxiter=1,  # do one cycle
            tol=1e-10,
        )
        e_i = e_i.view(cg_nonods, ndim)
    else:  # multi-grid method
        ncurve = 1  # always use 1 sfc
        N = len(sf_nd_nb.sfc_data_S.space_filling_curve_numbering)
        inverse_numbering = np.zeros((N, ncurve), dtype=int)
        inverse_numbering[:, 0] = np.argsort(sf_nd_nb.sfc_data_S.space_filling_curve_numbering[:, 0])
        r1_sfc = r1[inverse_numbering[:, 0], :].view(cg_nonods, ndim)

        # go to SFC coarse grid levels and do 1 mg cycles there
        e_i = mg.mg_on_P1CG(
            r1_sfc.view(cg_nonods, ndim),
            sf_nd_nb.sfc_data_S.variables_sfc,
            sf_nd_nb.sfc_data_S.nlevel,
            sf_nd_nb.sfc_data_S.nodes_per_level,
            cg_nonods
        )
        # reverse to original order
        e_i = e_i[sf_nd_nb.sfc_data_S.space_filling_curve_numbering[:, 0] - 1, :].view(cg_nonods, ndim)
    if not config.is_pmg:  # from P1CG to P3DG
        # prolongate error to fine grid
        e_i0 = torch.zeros(nele_s * u_nloc, ndim, device=dev, dtype=torch.float64)
        for idim in range(ndim):
            e_i0[:, idim] += mg.vel_p1dg_to_pndg_prolongator(torch.mv(
                sf_nd_nb.sparse_s.I_fc,
                e_i[:, idim]))
    else:  # from P1CG to P1DG, then go one order up each time while also do post smoothing
        raise Exception('pmg not implemented')
        # # prolongate error to P1DG
        # ilevel = config.ele_p - 1
        # for idim in range(ndim):
        #     e_p[ilevel][:, idim] += torch.mv(sf_nd_nb.I_fc, e_i[:, idim])
        # r_p, e_p = mg.p_mg_post(e_p, r_p)
        # e_i0 = e_p[0]
    # correct fine grid solution
    x_i_dict['disp'][nele_f:nele, ...] += e_i0.view(nele_s, u_nloc, ndim)
    # du_i += e_i0
    # post smooth
    for its1 in range(config.pre_smooth_its):
        r0 *= 0
        r0, x_i = volume_mf_he.get_residual_and_smooth_once(
            r0, x_k, x_i, x_rhs, include_s_blk=True, include_itf=False
        )
    # r0l2 = torch.linalg.norm(r0.view(-1), dim=0) / r0_init  # fNorm
    if isReturnResidual:
        r0l2 = torch.linalg.norm(r0_dict['disp'][nele_f:nele, ...].view(-1))
        return x_i, r0l2
    return x_i


def fsi_precond_all(x_i, x_k, u_bc):
    """
    this is the whole preconditioner for fsi system.
    original system is:
    [  F     G    I_uS  ] [ u ]
    [  D     0    I_pS  ] [ p ]
    [ I_Su  I_Sp    S   ] [ d ]
    the preconditioner is:
                            (-1)
    [ P_F    G    I_uS  ]          [ x_i vel  ]
    [  0    P_p   I_pS  ]          [ x_i pre  ]
    [  0     0     P_S  ]          [ x_i disp ]

    here u_bc pass in is the full list of u_bc.
    we'll figure out which u_bc to use...
    """
    x_temp = torch.zeros_like(x_i, device=dev, dtype=torch.float64)
    x_temp_dict = volume_mf_st.slicing_x_i(x_temp)
    x_i_dict = volume_mf_st.slicing_x_i(x_i)
    x_k_dict = volume_mf_st.slicing_x_i(x_k)

    # first apply P_S
    if nele_s > 0:
        x_temp = disp_precond_all(x_i=x_temp, x_rhs=x_i, x_k=x_k)
        # move x_temp to x_i
        x_i_dict['disp'] *= 0
        x_i_dict['disp'] += x_temp_dict['disp']

    if nele_f < 1:  # no fluid element
        return x_i
    # apply top right 2x1 block (I_uS and I_pS)
    x_temp *= 0
    x_temp_dict['vel'] += x_i_dict['vel']
    x_temp_dict['pre'] += x_i_dict['pre']
    x_temp = volume_mf_st.get_residual_only(
        r0=x_temp,
        x_i=x_i,
        x_rhs=0,
        include_adv=False,
        a00=False, a01=False, a10=False, a11=False,
        include_itf=True,
        x_k=x_k,
        u_bc=u_bc[0]
    )
    # move x_temp to x_i
    x_i_dict['vel'] *= 0
    x_i_dict['pre'] *= 0
    x_i_dict['vel'] += x_temp_dict['vel']
    x_i_dict['pre'] += x_temp_dict['pre']

    # apply pressure schur complement preconditioner P_p
    volume_mf_st.pre_precond_all(
        x_p=x_i_dict['pre'][0:nele_f, ...],
        include_adv=True,
        u_n=x_k_dict['vel'][0:nele_f, ...] - sf_nd_nb.u_m[0:nele_f, ...],
        u_bc=u_bc[0][0:nele_f, ...],  # vel diri bc
    )

    # apply velocity preconditioner P_F (note that G is also applied in this function)
    x_i = volume_mf_st.vel_precond_all(
        x_i=x_i,
        x_k=x_k,
        u_bc=u_bc[0],
        include_adv=True
    )
    return x_i


def fsi_precond_all2(x_i, x_k, u_bc):
    """
    this is the whole preconditioner for fsi system.
    original system is:
    [  F     G    I_uS  ] [ u ]
    [  D     0    I_pS  ] [ p ]
    [ I_Su  I_Sp    S   ] [ d ]
    the preconditioner is:
                            (-1)
    [ P_F    G      0  ]          [ x_i vel  ]
    [  0    P_p     0  ]          [ x_i pre  ]
    [ I_Su  I_Sp    S  ]          [ x_i disp ]

    here u_bc pass in is the full list of u_bc.
    we'll figure out which u_bc to use...
    """
    x_temp = torch.zeros_like(x_i, device=dev, dtype=torch.float64)
    x_temp_dict = volume_mf_st.slicing_x_i(x_temp)
    x_i_dict = volume_mf_st.slicing_x_i(x_i)
    x_k_dict = volume_mf_st.slicing_x_i(x_k)

    # first apply pressure schur complement preconditioner P_p
    if nele_f > 0:  # has fluid element
        volume_mf_st.pre_precond_all(
            x_p=x_i_dict['pre'][0:nele_f, ...],
            include_adv=True,
            u_n=x_k_dict['vel'][0:nele_f, ...] - sf_nd_nb.u_m[0:nele_f, ...],
            u_bc=u_bc[0][0:nele_f, ...],  # vel diri bc
        )

        # apply velocity preconditioner P_F (note that G is also applied in this function)
        x_i = volume_mf_st.vel_precond_all(
            x_i=x_i,
            x_k=x_k,
            u_bc=u_bc[0],
            include_adv=True
        )

    # apply bottom left 1x2 block (I_Su and I_Sp)
    if nele_s < 1:  # no solid element
        return x_i
    x_temp *= 0
    x_temp_dict['disp'] += x_i_dict['disp']
    x_temp = volume_mf_he.get_residual_only(
        r0=x_temp,
        x_k=x_k,
        x_i=x_i,
        x_rhs=0,
        include_s_blk=False,
        include_itf=True,
    )
    # move x_temp to x_i
    x_i_dict['disp'] *= 0
    x_i_dict['disp'] += x_temp_dict['disp']
    x_temp *= 0

    # then apply P_S
    x_temp = disp_precond_all(x_i=x_temp, x_rhs=x_i, x_k=x_k)
    # move x_temp to x_i
    x_i_dict['disp'] *= 0
    x_i_dict['disp'] += x_temp_dict['disp']
    return x_i


def fluid_only_precond(x_i, x_k, u_bc):
    x_i_dict = volume_mf_st.slicing_x_i(x_i)
    x_k_dict = volume_mf_st.slicing_x_i(x_k)
    volume_mf_st.pre_precond_all(
            x_p=x_i_dict['pre'][0:nele_f, ...],
            include_adv=True,
            u_n=x_k_dict['vel'][0:nele_f, ...] - sf_nd_nb.u_m[0:nele_f, ...],
            u_bc=u_bc[0][0:nele_f, ...],  # vel diri bc
        )

    # apply velocity preconditioner P_F (note that G is also applied in this function)
    x_i = volume_mf_st.vel_precond_all(
        x_i=x_i,
        x_k=x_k,
        u_bc=u_bc[0],
        include_adv=True
    )
    return x_i


def solid_only_precond(x_i, x_k):
    x_temp = torch.zeros_like(x_i, device=dev, dtype=torch.float64)
    x_temp_dict = volume_mf_st.slicing_x_i(x_temp)
    x_i_dict = volume_mf_st.slicing_x_i(x_i)
    x_k_dict = volume_mf_st.slicing_x_i(x_k)

    x_temp = disp_precond_all(x_i=x_temp, x_rhs=x_i, x_k=x_k)
    # move x_temp to x_i
    x_i_dict['disp'] *= 0
    x_i_dict['disp'] += x_temp_dict['disp']
    return x_i


def apply_I_SF(x_i, x_k):
    x_temp = torch.zeros_like(x_i, device=dev, dtype=torch.float64)
    x_temp_dict = volume_mf_st.slicing_x_i(x_temp)
    x_i_dict = volume_mf_st.slicing_x_i(x_i)
    x_k_dict = volume_mf_st.slicing_x_i(x_k)

    x_temp *= 0
    x_temp_dict['disp'] += x_i_dict['disp']
    x_temp = volume_mf_he.get_residual_only(
        r0=x_temp,
        x_k=x_k,
        x_i=x_i,
        x_rhs=0,
        include_s_blk=False,
        include_itf=True,
    )
    # move x_temp to x_i
    x_i_dict['disp'] *= 0
    x_i_dict['disp'] += x_temp_dict['disp']
    return x_i


def apply_I_FS(x_i, x_k, u_bc):
    x_temp = torch.zeros_like(x_i, device=dev, dtype=torch.float64)
    x_temp_dict = volume_mf_st.slicing_x_i(x_temp)
    x_i_dict = volume_mf_st.slicing_x_i(x_i)
    x_k_dict = volume_mf_st.slicing_x_i(x_k)

    # apply top right 2x1 block (I_uS and I_pS)
    x_temp *= 0
    x_temp_dict['vel'] += x_i_dict['vel']
    x_temp_dict['pre'] += x_i_dict['pre']
    x_temp = volume_mf_st.get_residual_only(
        r0=x_temp,
        x_i=x_i,
        x_rhs=0,
        include_adv=False,
        a00=False, a01=False, a10=False, a11=False,
        include_itf=True,
        x_k=x_k,
        u_bc=u_bc[0]
    )
    # move x_temp to x_i
    x_i_dict['vel'] *= 0
    x_i_dict['pre'] *= 0
    x_i_dict['vel'] += x_temp_dict['vel']
    x_i_dict['pre'] += x_temp_dict['pre']
    return x_i
