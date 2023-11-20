"""this is to test SFC multigrid solver"""

import numpy as np
import torch
import config
import multigrid_linearelastic as mg
import volume_mf_he
import preconditioner
import volume_mf_st
from config import sf_nd_nb

ndim = config.ndim
dev=config.dev


def test_sfc_mg(Amat, rhs, sfc_data, sparse_in):
    n = Amat.shape[0]
    nele = int(n / ndim / 10)  # 10 is nloc
    nloc = 10
    r0 = torch.zeros(n, dtype=torch.float64, device=config.dev)
    x = torch.zeros(n, dtype=torch.float64, device=config.dev)
    cg_nonods = sparse_in.cg_nonods

    # get diagonal blocks of A and inverse it to use as a smoother
    blk_size = ndim * 10
    Amat_view = (Amat.view(ndim, nele, 10, ndim, nele, 10).
                 permute(1, 2, 0, 4, 5, 3).contiguous().
                 view(nele * blk_size, nele * blk_size))
    smoother = [Amat_view[i*blk_size:(i+1)*blk_size, i*blk_size:(i+1)*blk_size]
                for i in range(nele)]
    smoother = torch.stack(smoother, dim=0)
    smoother = torch.inverse(smoother)

    # get initial residual
    r0 *= 0
    _get_residual_and_smooth_once(r0, x, Amat, rhs, smooth=False)
    r0l2_init = torch.linalg.norm(r0)
    r0l2 = torch.tensor([1], device=config.dev, dtype=torch.float64)

    I_cf = sparse_in.I_cf
    I_fc = sparse_in.I_fc

    # MG cycles
    its = 1
    while r0l2 > 1e-8 and its < 400:
        # presmooth
        for ii in range(3):
            r0 *= 0
            _get_residual_and_smooth_once(r0, x, Amat, rhs, smooth=True, smoother=smoother)
        # get residaul on PnDG
        r0 *= 0
        _get_residual_and_smooth_once(r0, x, Amat, rhs, smooth=False)
        r1 = torch.zeros(cg_nonods, ndim, dtype=torch.float64, device=config.dev)
        for idim in range(ndim):
            r1[:, idim] += torch.mv(
                sf_nd_nb.sparse_s.I_cf,
                mg.vel_pndg_to_p1dg_restrictor(r0.view(ndim, -1)[idim, :]).view(-1)
            )

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
        # prolongate error to fine grid
        e_i0 = torch.zeros(nele * nloc, ndim, device=dev, dtype=torch.float64)
        for idim in range(ndim):
            e_i0[:, idim] += mg.vel_p1dg_to_pndg_prolongator(torch.mv(
                sf_nd_nb.sparse_s.I_fc,
                e_i[:, idim]))
        x += e_i0.view(nele, nloc, ndim).permute(2,0,1).contiguous().view(x.shape)

        # post smooth
        for ii in range(3):
            r0 *= 0
            _get_residual_and_smooth_once(r0, x, Amat, rhs, smooth=True, smoother=smoother)
        r0l2 = torch.linalg.norm(r0.view(-1), dim=0) / r0l2_init  # residual norm
        its += 1
        print('its, ', its, 'res, ', r0l2)
    return its, r0l2


def _get_residual_and_smooth_once(
        r0, x, Amat, rhs, smooth=False, smoother=None,
):
    r0 += rhs - Amat @ x
    if smooth:
        x += 2./3. * torch.einsum('bij,bj->bi', smoother, r0.view(-1, 10 * ndim)).view(x.shape)


def solve_with_precond(x_i, x_rhs, x_k):
    """ use preconditioner: disp_precond_all
    to solve the solid block A_s x = b
    """
    its = 0
    r0l2 = 1
    nele = config.nele
    nele_f = config.nele_f

    u_nloc = sf_nd_nb.disp_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    total_no_dofs = nele * u_nloc * ndim * 2 + nele * p_nloc

    r0 = torch.zeros(total_no_dofs, device=dev, dtype=torch.float64)
    r0 *= 0
    r0 = volume_mf_he.get_residual_only(
        r0, x_k, x_i, x_rhs, include_s_blk=True, include_itf=False)
    r0_dict = volume_mf_st.slicing_x_i(r0)
    r0l2_init = torch.linalg.norm(r0_dict['disp'][nele_f:nele, ...].view(-1))
    print('initial residual (with 0 guess) is ', r0l2_init)
    while its < 400 and r0l2 > 1e-8:
        x_i, r0l2 = preconditioner.disp_precond_all(x_i, x_rhs, x_k, isReturnResidual=True)
        r0l2 /= r0l2_init
        its += 1
        print('its, ', its, 'residual relative to init, ', r0l2)

