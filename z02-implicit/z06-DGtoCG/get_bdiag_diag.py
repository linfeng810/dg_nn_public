""" get block diagonal and diagonal of
left-hand side matrix (diffusion operator)

this will be stored and reused during
the iteration of the linear solver
"""

import torch
import numpy as np
import config
import function_space
from config import sf_nd_nb
from typing import List, Tuple
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


def get_bdiag_diag():
    """
    getting block diagonal and diagonal of diffusion matrix K
    """
    nnn = 1
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele_f, dtype=int)
    u_nloc = sf_nd_nb.vel_func_space.element.nloc

    i = 0
    # volume integral
    batch_in = nele

    bdiagK = torch.zeros(batch_in, u_nloc, u_nloc, device=dev, dtype=config.dtype)
    _k_res_one_batch(  # r0, diagK, bdiagK = _k_res_one_batch(
        bdiagK,
        # idx_in,
        sf_nd_nb.vel_func_space,
        config.mu_f, nele_f,
    )
    # surface integral
    idx_in_f = torch.zeros(nele * nface, dtype=torch.bool, device=dev)
    idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True

    _s_res_one_batch(  # r0, diagK, bdiagK = _s_res_one_batch(
        bdiagK,
        idx_in_f, brk_pnt[i],
        sf_nd_nb.vel_func_space,
        config.mu_f, config.eta_e
    )

    diagK = torch.diagonal(bdiagK, dim1=1, dim2=2).view(batch_in, u_nloc)

    sf_nd_nb.set_data(diagK=diagK, bdiagK=bdiagK)


# @torch.jit.optimize_for_inference
# @torch.jit.script
def _k_res_one_batch(
        bdiagK,
        # idx_in,  # those without explicit type declaration default to torch.Tensor
        func_space: function_space.FuncSpaceTS,
        mu_f: float, nele_f: int,
) -> None:  # Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_in = bdiagK.shape[0]
    u_nloc = func_space.element.nloc
    nele = func_space.nele
    dev = func_space.dev

    # diagK = diagK.view(-1, u_nloc)
    bdiagK = bdiagK.view(-1, u_nloc, u_nloc)

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

    # this saves memory
    nx_ndetwei = torch.mul(
        nx,
        ndetwei.view(batch_in, 1, 1, -1)
    )
    bdiagK += torch.einsum(
        'bimg,bing->bmn',
        nx,
        nx_ndetwei,
    ) * mu_f


@torch.jit.script
def _s_res_fi_all_face(
        bdiagK,
        func_space: function_space.FuncSpaceTS,
        mu_f: float, eta_e: float,
) -> None:
    """all internal faces *inside* element contribution
    computed in 1 batch
    no matter local face index or neighbour face gi pnts alignment
    will deal those when putting into global matrix/residual
    at the end of this function"""
    dev = func_space.dev
    nele = func_space.nele
    ndim = func_space.element.ndim
    nface = ndim + 1
    # get element parameters
    u_nloc = func_space.element.nloc

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

    # K block
    K = torch.zeros(nele, nface, u_nloc, u_nloc, device=dev, dtype=bdiagK.dtype)
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

    snx_snormal = torch.bmm(
        snx.permute(0, 1, 4, 3, 2).reshape(-1, u_nloc, ndim),
        snormal.permute(0, 1, 3, 2).reshape(-1, ndim, 1)
    )  # (nele * nface * sngi, u_nloc, 1)
    if False:
        snx_snormal_sn = torch.einsum(
            'fng,bfgm->bfmng',
            sn.view(nface, u_nloc, sngi),
            snx_snormal.view(nele, nface, sngi, u_nloc)
        )
        snx_snormal_sn_sdetwei = torch.einsum(
            'bfmng,bfg->bfmn',
            snx_snormal_sn,  # (nele, nface, nloc, nloc, sngi)
            sdetwei  # .view(nele, nface, sngi)
        )
    else:  # let's switch multiply sequence to save memory
        snx_snormal_sdetwei = torch.mul(
            snx_snormal.view(nele, nface, sngi, u_nloc),
            sdetwei.view(nele, nface, sngi, 1)
        )
        snx_snormal_sn_sdetwei = torch.einsum(
            'bfgm,fng->bfmn',
            snx_snormal_sdetwei,  # (nele, nface, sngi, nloc)
            sn.view(nface, u_nloc, sngi),
        )
    K += snx_snormal_sn_sdetwei * (-0.5)  # consistent term
    K += snx_snormal_sn_sdetwei.transpose(2, 3) * (-0.5)  # symmetry term
    sn_sn = torch.einsum(
        'fmg,fng->fmng',
        sn, sn,  # (nface, nloc, sngi)
    )
    sn_sn_sdetwei = torch.einsum(
        'fmng,bfg->bfmn',  # (nele, nface, nloc, nloc)
        sn_sn, sdetwei  # (nele, nface, sngi)
    )
    K += sn_sn_sdetwei * gamma_e.unsqueeze(2).unsqueeze(3)  # penalty term

    K *= mu_f

    # set boundary face to 0
    K = K.view(nele * nface, u_nloc, u_nloc)
    K *= (func_space.glb_bcface_type < 0).view(-1, 1, 1).to(K.dtype)

    # put them to r0, diagK and bdiagK (SCATTER)
    K = K.view(nele, nface, u_nloc, u_nloc)
    for iface in range(nface):
        bdiagK += K[:, iface, :, :]
    a = 0


# @torch.jit.optimize_for_inference
@torch.jit.script
def _s_res_fb_all_face(
        f_b, E_F_b,
        bdiagK,
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

    # block K
    K = torch.zeros(batch_in, u_nloc, u_nloc,
                    device=dev, dtype=bdiagK.dtype)
    # [vi nj] {du_i / dx_j}  consistent term
    snx_snormal = torch.bmm(
        snx.permute(0, 3, 2, 1).reshape(-1, u_nloc, ndim),
        snormal.permute(0, 2, 1).reshape(-1, ndim, 1)
    )  # (batch_in * sngi, u_nloc, 1)
    snx_snormal_sn = torch.einsum(
        'bmg,bgn->bmng',
        sn,  # (batch_in, nloc, sngi)
        snx_snormal.view(batch_in, sngi, u_nloc)
    )
    snx_snormal_sn_sdetwei = torch.einsum(
        'bmng,bg->bmn',
        snx_snormal_sn,  # (batch_in, nloc, nloc, sngi)
        sdetwei  # (batch_in, sngi)
    )
    K -= snx_snormal_sn_sdetwei
    # {dv_i / dx_j} [ui nj]  symmetry term
    K -= snx_snormal_sn_sdetwei.transpose(1, 2)
    # \gamma_e [v_i] [u_i]  penalty term
    sn_sn = torch.einsum('bmg,bng->bmng', sn, sn)
    sn_sn_sdetwei = torch.einsum('bmng,bg->bmn', sn_sn, sdetwei)
    K += sn_sn_sdetwei * gamma_e.view(batch_in, 1, 1)
    K *= mu_f

    # SCATTER
    for iface in range(nface):
        idx_iface = (f_b == iface)
        E_idx = E_F_b[idx_iface]
        bdiagK[E_idx - batch_start_idx, ...] += K[idx_iface]
    # return r0, diagK, bdiagK


# @profile (line-by-line profile)
# @torch.compile
# @torch.jit.optimize_for_inference
@torch.jit.script
def _s_res_one_batch(
        bdiagK,
        idx_in_f,
        batch_start_idx: int,
        func_space: function_space.FuncSpaceTS,
        mu_f: float, eta_e: float,
) -> None:  # -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    _s_res_fi_all_face(  # r0, diagK, bdiagK = _s_res_fi_all_face(
        bdiagK,
        func_space, mu_f, eta_e
    )
    _s_res_fb_all_face(
        f_b, E_F_b,
        bdiagK, batch_start_idx,
        func_space, mu_f, eta_e
    )
