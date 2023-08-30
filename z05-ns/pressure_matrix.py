import torch
import numpy as np
from tqdm import tqdm
import time, os.path
from scipy.sparse import csr_matrix, linalg
import config
import volume_mf_st
from config import sf_nd_nb
if config.ndim == 2:
    from shape_function import get_det_nlx as get_det_nlx
    from shape_function import sdet_snlx as sdet_snlx
else:
    from shape_function import get_det_nlx_3d as get_det_nlx
    from shape_function import sdet_snlx_3d as sdet_snlx
import torch.nn.functional as F
import torch.nn as nn
import sfc as sf  # to be compiled ...
import map_to_sfc_matrix as map_sfc
import multigrid_linearelastic as mg_le

nele = config.nele
ndim = config.ndim
nface = ndim + 1
dev = config.dev


def _apply_pressure_mat(
        r0, x_i, x_rhs,
        include_adv=False, u_n=None, u_bc=None,
        doSmooth=False
):
    """apply pressure convection-diffusion matrix  F_p
    or apply pressure laplacian  L_p
    i.e. r0 <- x_rhs - F_p * x_i
    oder r0 <- x_rhs - L_p * x_i

    if we want to obtain matvec,
    -    r0 = 0
    -    x_rhs = 0
    -    x_i = vector
    if we want get L_p^{-1}x w/ one multi-grid cycle,
    -    r0 = 0
    -    x_rhs = vector
    -    x_i = 0

    input r_0, x_i, and x_rhs should be of size (p_nonods)
        """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele, dtype=int)
    # retrive func space / element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    if include_adv:
        u_n = u_n.view(nele, u_nloc, ndim)
        u_bc = u_bc.view(nele, u_nloc, ndim)

    # add precalculated rhs to residual
    r0 += x_rhs.view(r0.shape)
    for i in range(nnn):
        # volume integral
        idx_in = torch.zeros(nele, device=dev, dtype=bool)  # element indices in this batch
        idx_in[brk_pnt[i]:brk_pnt[i + 1]] = True
        batch_in = int(torch.sum(idx_in))
        # dummy diagA and bdiagA
        diagK = torch.zeros(batch_in, p_nloc, device=dev, dtype=torch.float64)
        bdiagK = torch.zeros(batch_in, p_nloc, p_nloc, device=dev, dtype=torch.float64)

        r0, diagK, bdiagK = _k_res_one_batch(
            r0, x_i,
            diagK, bdiagK,
            idx_in,
            include_adv, u_n
        )
        # surface integral
        idx_in_f = torch.zeros(nele * nface, dtype=bool, device=dev)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0, diagK, bdiagK = _s_res_one_batch(
            r0, x_i,
            diagK, bdiagK,
            idx_in_f, brk_pnt[i],
            include_adv, u_n, u_bc,
        )
        if doSmooth:
            x_i = x_i.view(nele, p_nloc)
            if config.blk_solver == 'direct':
                bdiagK = torch.inverse(bdiagK.view(batch_in, p_nloc, p_nloc))
                x_i[idx_in, :] += config.jac_wei * torch.einsum(
                    'bmn,bn->bm',
                    bdiagK,
                    r0.view(nele, p_nloc)[idx_in, :]
                ).view(batch_in, p_nloc)
            else:
                raise Exception('can only choose direct solver for pressure laplacian block')
    return r0, x_i


def _k_res_one_batch(
        r0, x_i,
        diagK, bdiagK,
        idx_in,
        include_adv,
        u_n,
):
    """this contains volume integral part of the residual update
    """
    batch_in = diagK.shape[0]
    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc

    r0 = r0.view(-1, p_nloc)
    x_i = x_i.view(-1, p_nloc)
    diagK = diagK.view(-1, p_nloc)
    bdiagK = bdiagK.view(-1, p_nloc, p_nloc)

    # get shape function and derivatives
    q = sf_nd_nb.pre_func_space.element.n
    v = sf_nd_nb.vel_func_space.element.n  # velocity shape functions
    qx, ndetwei = get_det_nlx(
        nlx=sf_nd_nb.pre_func_space.element.nlx,
        x_loc=sf_nd_nb.pre_func_space.x_ref_in[idx_in],
        weight=sf_nd_nb.pre_func_space.element.weight,
        nloc=p_nloc,
        ngi=sf_nd_nb.pre_func_space.element.ngi
    )

    # local K
    K = torch.zeros(batch_in, p_nloc, p_nloc, device=dev, dtype=torch.float64)

    # Nx_i Nx_j
    K += torch.einsum('bimg,bing,bg->bmn', qx, qx, ndetwei) * config.mu
    if sf_nd_nb.isTransient:
        # ni nj
        K += torch.einsum('mg,ng,bg->bmn', q, q, ndetwei) * config.rho / config.dt * sf_nd_nb.bdfscm.gamma
    if include_adv:
        K += torch.einsum(
            'lg,bli,bing,mg,bg->bmn',
            v,
            u_n[idx_in, ...],
            qx,
            q,
            ndetwei
        )
    # update residual of velocity block K
    r0[idx_in, ...] -= torch.einsum('bmn,bn->bm', K, x_i[idx_in, ...])
    # get diagonal of velocity block K
    diagK += torch.diagonal(K.view(batch_in, p_nloc, p_nloc)
                            , dim1=1, dim2=2).view(batch_in, p_nloc)
    bdiagK[idx_in, ...] += K

    return r0, diagK, bdiagK


def _s_res_one_batch(
        r0, x_i,
        diagK, bdiagK,
        idx_in_f,
        batch_start_idx,
        include_adv, u_n, u_bc,
):
    # get essential data
    nbf = sf_nd_nb.vel_func_space.nbf
    alnmt = sf_nd_nb.vel_func_space.alnmt

    # change view
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    # r0[0] = r0[0].view(-1, u_nloc, ndim)
    # r0[1] = r0[1].view(-1, p_nloc)
    # x_i[0] = x_i[0].view(-1, u_nloc, ndim)
    # x_i[1] = x_i[1].view(-1, p_nloc)

    # separate nbf to get internal face list and boundary face list
    F_i = torch.where(torch.logical_and(alnmt >= 0, idx_in_f))[0]  # interior face
    # for boundary faces, only include velocity Neumann boundary faces (which is the Diri bc for pressure)
    F_b = torch.where(torch.logical_and(
        torch.logical_and(alnmt < 0, sf_nd_nb.vel_func_space.glb_bcface_type == 0),
        idx_in_f))[0]  # boundary face
    F_inb = nbf[F_i]  # neighbour list of interior face
    F_inb = F_inb.type(torch.int64)

    # create two lists of which element f_i / f_b is in
    E_F_i = torch.floor_divide(F_i, nface)
    E_F_b = torch.floor_divide(F_b, nface)
    E_F_inb = torch.floor_divide(F_inb, nface)

    # local face number
    f_i = torch.remainder(F_i, nface)
    f_b = torch.remainder(F_b, nface)
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
                include_adv, u_n
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
                include_adv, u_bc
            )
    else:
        raise Exception('2D hyper-elasticity not implemented!')
    return r0, diagK, bdiagK


def _s_res_fi(
        r_in, f_i, E_F_i,
        f_inb, E_F_inb,
        x_i_in,
        diagK, bdiagK, batch_start_idx,
        nb_gi_aln,
        include_adv, u_n
):
    """internal faces"""
    batch_in = f_i.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)

    # get element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    x_i = x_i_in.view(-1, p_nloc)
    r_i = r_in.view(-1, p_nloc)

    # shape function on this side
    sqx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.pre_func_space.element.snlx,
        x_loc=sf_nd_nb.pre_func_space.x_ref_in[E_F_i],
        sweight=sf_nd_nb.pre_func_space.element.sweight,
        nloc=sf_nd_nb.pre_func_space.element.nloc,
        sngi=sf_nd_nb.pre_func_space.element.sngi
    )
    sq = sf_nd_nb.pre_func_space.element.sn[f_i, ...]  # (batch_in, nloc, sngi)
    sv = sf_nd_nb.vel_func_space.element.sn[f_i, ...]  # velocity shape functions on faces
    sqx = sqx[dummy_idx, f_i, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_i, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_i, ...]  # (batch_in, ndim)

    # shape function on the other side
    sqx_nb, _, snormal_nb = sdet_snlx(
        snlx=sf_nd_nb.pre_func_space.element.snlx,
        x_loc=sf_nd_nb.pre_func_space.x_ref_in[E_F_inb],
        sweight=sf_nd_nb.pre_func_space.element.sweight,
        nloc=sf_nd_nb.pre_func_space.element.nloc,
        sngi=sf_nd_nb.pre_func_space.element.sngi
    )
    # get faces we want
    sq_nb = sf_nd_nb.pre_func_space.element.sn[f_inb, ...]  # (batch_in, nloc, sngi)
    sv_nb = sf_nd_nb.vel_func_space.element.sn[f_inb, ...]  # velocity shape function on neighbouring face
    sqx_nb = sqx_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim, nloc, sngi)
    snormal_nb = snormal_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim)
    # change gaussian points order on other side
    nb_aln = sf_nd_nb.pre_func_space.element.gi_align[nb_gi_aln, :]  # nb_aln for pressure element
    sqx_nb = sqx_nb[..., nb_aln]
    # don't forget to change gaussian points order on sn_nb!
    sq_nb = sq_nb[..., nb_aln]
    sv_nb = sv_nb[..., nb_aln]

    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = config.eta_e / h
    p_ith = x_i[E_F_i, ...]
    p_inb = x_i[E_F_inb, ...]

    # K block
    K = torch.zeros(batch_in, p_nloc, p_nloc, device=dev, dtype=torch.float64)
    # this side
    # [q n_j] {dp / dx_j}  consistent term
    K += torch.einsum(
        'bmg,bj,bjng,bg->bmn',
        sq,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        sqx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (-0.5)
    # {dq / dx_j} [p n_j]  symmetry term
    K += torch.einsum(
        'bjmg,bng,bj,bg->bmn',
        sqx,  # (batch_in, ndim, nloc, sngi)
        sq,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        sdetwei,  # (batch_in, sngi)
    ) * (-0.5)
    # \gamma_e * [q][p]  penalty term
    K += torch.einsum(
        'bmg,bng,bg,b->bmn',
        sq,  # (batch_in, nloc, sngi)
        sq,  # (batch_in, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        gamma_e,  # (batch_in)
    )
    K *= config.mu
    # Edge stabilisation (vel gradient penalty term)
    if config.isES:
        # \gamma_e h^2 [q / x_j] [p / x_j]
        K += torch.einsum(
            'b,bjmg,bjng,bg->bmn',
            config.gammaES * h ** 2,  # (batch_in)
            sqx,  # (batch_in, ndim, nloc, sngi)
            sqx,  # (batch_in, ndim, nloc, sngi)
            sdetwei,  # (batch_in, sngi)
        ) * (.5)
    if include_adv:
        # get upwind vel
        wknk_ave = torch.einsum(
            'bmg,bmi,bi->bg',
            sv,  # (batch_in, u_nloc, sngi)
            u_n[E_F_i, :, :],  # (batch_in, u_nloc, sngi)
            snormal,  # (batch_in, ndim)
        ) * 0.5 + torch.einsum(
            'bmg,bmi,bi->bg',
            sv_nb,  # (batch_in, u_nloc, sngi)
            u_n[E_F_inb, :, :],  # (batch_in, u_nloc, sngi)
            snormal,  # (batch_in, ndim)
        ) * 0.5
        wknk_upwd = 0.5 * (wknk_ave - torch.abs(wknk_ave))
        K += -torch.einsum(
            'bg,bmg,bng,bg->bmn',
            wknk_upwd,  # (batch_in, sngi)
            sq,  # (batch_in, p_nloc, sngi)
            sq,  # (batch_in, p_nloc, sngi)
            sdetwei,  # (bathc_in, sngi)
        )
    # update residual
    r_i[E_F_i, ...] -= torch.einsum('bmn,bn->bm', K, p_ith)
    # put diagonal into diagK and bdiagK
    diagK[E_F_i - batch_start_idx, :] += torch.diagonal(K.view(batch_in, p_nloc, p_nloc),
                                                           dim1=1, dim2=2).view(batch_in, p_nloc)
    bdiagK[E_F_i - batch_start_idx, :, :] += K

    # other side
    K *= 0
    # [q n_j] {dp / dx_j}  consistent term
    K += torch.einsum(
        'bmg,bj,bjng,bg->bmn',
        sq,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        sqx_nb,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (-0.5)
    # {dq / dx_j} [p n_j]  symmetry term
    K += torch.einsum(
        'bjmg,bng,bj,bg->bmn',
        sqx,  # (batch_in, ndim, nloc, sngi)
        sq_nb,  # (batch_in, nloc, sngi)
        snormal_nb,  # (batch_in, ndim)
        sdetwei,  # (batch_in, sngi)
    ) * (-0.5)
    # \gamma_e * [q][p]  penalty term
    K += torch.einsum(
        'bmg,bng,bg,b->bmn',
        sq,  # (batch_in, nloc, sngi)
        sq_nb,  # (batch_in, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        gamma_e,  # (batch_in)
    ) * (-1.)  # because n2 \cdot n1 = -1
    K *= config.mu
    # Edge stabilisation (vel gradient penalty term)
    if config.isES:
        # \gamma_e h^2 [q / x_j] [p / x_j]
        K += torch.einsum(
            'b,bjmg,bjng,bg->bmn',
            config.gammaES * h ** 2,  # (batch_in)
            sqx,  # (batch_in, ndim, nloc, sngi)
            sqx_nb,  # (batch_in, ndim, nloc, sngi)
            sdetwei,  # (batch_in, sngi)
        ) * (-.5)
    if include_adv:
        K += -torch.einsum(
            'bg,bmg,bng,bg->bmn',
            wknk_upwd,  # (batch_in, sngi)
            sq,  # (batch_in, u_nloc, sngi)
            sq_nb,  # (batch_in, u_nloc, sngi)
            sdetwei,  # (bathc_in, sngi)
        ) * (-1.)
    # update residual
    r_i[E_F_i, ...] -= torch.einsum('bmn,bn->bm', K, p_inb)

    # this concludes surface integral on interior faces.
    return r_in, diagK, bdiagK


def _s_res_fb(
        r_in, f_b, E_F_b,
        x_i_in,
        diagK, bdiagK, batch_start_idx,
        include_adv, u_bc
):
    """boundary faces"""
    batch_in = f_b.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    if batch_in < 1:  # nothing to do here.
        return r_in, diagK, bdiagK
    # get element parameters
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    r_i = r_in.view(-1, p_nloc)
    x_i = x_i_in.view(-1, p_nloc)

    # shape function
    sqx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.pre_func_space.element.snlx,
        x_loc=sf_nd_nb.pre_func_space.x_ref_in[E_F_b],
        sweight=sf_nd_nb.pre_func_space.element.sweight,
        nloc=sf_nd_nb.pre_func_space.element.nloc,
        sngi=sf_nd_nb.pre_func_space.element.sngi
    )
    sv = sf_nd_nb.vel_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    sq = sf_nd_nb.pre_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)

    sqx = sqx[dummy_idx, f_b, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_b, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_b, ...]  # (batch_in, ndim)
    if ndim == 3:
        gamma_e = config.eta_e / torch.sqrt(torch.sum(sdetwei, -1))
    else:
        gamma_e = config.eta_e / torch.sum(sdetwei, -1)

    p_ith = x_i[E_F_b, ...]

    # block K
    K = torch.zeros(batch_in, p_nloc, p_nloc,
                    device=dev, dtype=torch.float64)
    # [q nj] {dp / dx_j}  consistent term
    K -= torch.einsum(
        'bmg,bj,bjng,bg->bmn',
        sq,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        sqx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    )
    # {dq / dx_j} [p nj]  symmetry term
    K -= torch.einsum(
        'bjmg,bng,bj,bg->bmn',
        sqx,  # (batch_in, ndim, nloc, sngi)
        sq,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim)
        sdetwei,  # (batch_in, sngi)
    )  # .unsqueeze(2).unsqueeze(4).expand(batch_in, u_nloc, ndim, u_nloc, ndim)
    # \gamma_e [q] [p]  penalty term
    K += torch.einsum(
        'bmg,bng,bg,b->bmn',
        sq,  # (batch_in, nloc, sngi)
        sq,  # (batch_in, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
        gamma_e,  # (batch_in)
    )
    K *= config.mu
    if include_adv:
        # get upwind velocity
        wknk_ave = torch.einsum(
            'bmg,bmi,bi->bg',
            sv,  # (batch_in, u_nloc, sngi)
            u_bc[E_F_b, ...],  # (batch_in, u_nloc, ndim)
            snormal,  # (batch_in, ndim)
        )
        wknk_upwd = 0.5 * (wknk_ave - torch.abs(wknk_ave))
        K += -torch.einsum(
            'bg,bmg,bng,bg->bmn',
            wknk_upwd,
            sq,  # (batch_in, u_nloc, sngi)
            sq,  # (batch_in, u_nloc, sngi)
            sdetwei,  # (batch_in, sngi)
        )
    # update residual
    r_i[E_F_b, ...] -= torch.einsum('bmn,bn->bm', K, p_ith)
    # put in diagonal
    diagK[E_F_b - batch_start_idx, :] += torch.diagonal(K.view(batch_in, p_nloc, p_nloc),
                                                        dim1=-2, dim2=-1).view(batch_in, p_nloc)
    bdiagK[E_F_b - batch_start_idx, ...] += K
    del K

    return r_in, diagK, bdiagK


def _calc_RAR_mf_color(
        I_fc, I_cf,
        whichc, ncolor,
        fina, cola, ncola
):
    """
    get operator on P1CG grid, i.e. RAR
    where R is prolongator/restrictor
    via coloring method.
    """
    import time
    start_time = time.time()
    cg_nonods = sf_nd_nb.pre_func_space.cg_nonods
    p1dg_nonods = sf_nd_nb.pre_func_space.p1dg_nonods
    nonods = sf_nd_nb.pre_func_space.nonods
    value = torch.zeros(ncola, device=dev, dtype=torch.float64)  # NNZ entry values
    dummy = torch.zeros(nonods, device=dev, dtype=torch.float64)  # dummy variable of same length as PnDG
    Rm = torch.zeros(nonods, device=dev, dtype=torch.float64)
    ARm = torch.zeros(nonods, device=dev, dtype=torch.float64)
    RARm = torch.zeros(cg_nonods, device=dev, dtype=torch.float64)
    mask = torch.zeros(cg_nonods, device=dev, dtype=torch.float64)  # color vec
    for color in tqdm(range(1, ncolor + 1), disable=config.disabletqdm):
        # print('color: ', color)
        mask *= 0
        mask += torch.tensor((whichc == color),
                             device=dev,
                             dtype=torch.float64)  # 1 if true; 0 if false
        Rm *= 0
        Rm += mg_le.pre_p1dg_to_pndg_prolongator(
            torch.mv(I_fc, mask)  # I_fc is from continuous to discontinuous P1 mesh.
        )  # (p_nonods)
        ARm *= 0
        ARm, _ = _apply_pressure_mat(
            r0=ARm,
            x_i=Rm,
            x_rhs=dummy,
            include_adv=False,  # only get pressure laplacian.
            doSmooth=False,
        )
        ARm *= -1.  # (p_nonods)
        RARm *= 0
        RARm += torch.mv(I_cf, mg_le.pre_pndg_to_p1dg_restrictor(ARm))  # (cg_nonods)
        # add to RARvalue
        for i in range(RARm.shape[0]):
            for count in range(fina[i], fina[i + 1]):
                j = cola[count]
                value[count] += RARm[i] * mask[j]
        # print('finishing (another) one color, time comsumed: ', time.time() - start_time)
    return value


# multi-grid on P1CG of pressure laplacian
def _mg_smooth_one_level(level, e_i, b, variables_sfc):
    """
    do one smooth step on mg level = level
    """
    e_i = e_i.view(variables_sfc[level][2])
    rr_i = torch.zeros_like(e_i, device=config.dev, dtype=torch.float64)
    a_sfc_sparse, diagA, _ = variables_sfc[level]
    rr_i += torch.mv(a_sfc_sparse, e_i)
    rr_i *= -1
    rr_i += b.view(-1)
    e_i += rr_i / diagA.view(-1) * config.jac_wei
    e_i = e_i.view(1, 1, -1)
    rr_i = rr_i.view(1, 1, -1)
    return e_i, rr_i


def _get_a_diaga(
        level,
        fin_sfc_nonods,
        fina_sfc_all_un,
        cola_sfc_all_un,
        a_sfc
):
    '''
    extract operators and its diagonals on one level of SFC
    coarse grid

    # Input
    level : int
        level-th grid to extract
    fin_sfc_nonods : numpy array (integer) (nlevel+1)
        a list containing the starting index in a_sfc of each level
        actual length will be longer than nlevel+1, extras are 0
    fina_sfc_all_un : numpy array (float)
        fina on all levels
    cola_sfc_all_un : numpy array (float)
        cola on all levels
    a_sfc : numpy/torch array (ndim, ndim, :)
        a_sfc values on all levels

    # Output
    a_sfc_level_sparse : a 2-d list of torch coo sparse tensor
        list dimension (ndim, ndim)
        coo sparse tensor dimension (nonods, nonods)
        this stores operator A as a block matrix
        in SFC ordering for the specified level
    diagonal : torch tensor, (ndim,nonods)
        Diagonal values in a_sfc_level_sparse
    nonods: scalar
        Number of nodes in the *specified* level. N.B. this is *NOT*
        the global number of nodes.

    '''
    # level 0 is the highest level;
    # level nlevel-1 is the lowest level;
    # subtracting 1 because fortran indexes from 1 but python indexes from 0

    start_index = fin_sfc_nonods[level] - 1
    end_index = fin_sfc_nonods[level + 1] - 1
    nonods = end_index - start_index

    diagonal = torch.zeros((nonods), dtype=torch.float64, device=config.dev)
    a_indices = []
    a_values = []
    for i in range(start_index, end_index):
        for j in range(fina_sfc_all_un[i] - 1, fina_sfc_all_un[i + 1] - 1):
            a_indices.append([i - start_index, cola_sfc_all_un[j] - 1 - start_index])
            a_values.append(a_sfc[j])
    a_indices = np.asarray(a_indices).transpose()
    a_indices = torch.tensor(a_indices, device=config.dev)
    a_values = np.asarray(a_values)  # now a_values has shape (nnz)
    a_values = torch.tensor(a_values, dtype=torch.float64, device=config.dev)
    # convert to sparse
    a_sfc_level_sparse = torch.sparse_coo_tensor(
                a_indices,
                a_values,
                (nonods, nonods),
                dtype=torch.float64,
                device=config.dev).to_sparse_csr()
    # find diagonal
    for i in range(nonods):
        i += start_index
        for j in range(fina_sfc_all_un[i] - 1, fina_sfc_all_un[i + 1] - 1):
            if i == cola_sfc_all_un[j] - 1:
                diagonal[i - start_index] = a_sfc[j]

    return a_sfc_level_sparse, diagonal, nonods


def _mg_on_P1CG(r0, variables_sfc, nlevel, nodes_per_level):
    """
    Multi-grid cycle on P1CG mesh.
    Takes in residaul on P1CG mesh (restricted from residual on PnDG mesh),
    do 1 mg cycle on the SFC levels, then spit out an error correction on
    P1CG mesh (add it to the input e_i0).
    """
    cg_nonods = sf_nd_nb.vel_func_space.cg_nonods
    # get residual on each level
    sfc_restrictor = torch.nn.Conv1d(in_channels=1,
                                     out_channels=1, kernel_size=2,
                                     stride=2, padding='valid', bias=False)
    sfc_restrictor.weight.data = torch.tensor([[1., 1.]],
                                              dtype=torch.float64,
                                              device=config.dev).view(1, 1, 2)

    smooth_start_level = config.smooth_start_level
    r0 = r0.view(cg_nonods).view(1, 1, cg_nonods)
    r_s = [r0]  # collection of r
    e_s = [torch.zeros(1, 1, cg_nonods, device=config.dev, dtype=torch.float64)]  # collec. of e
    for level in range(0, smooth_start_level):
        # pre-smooth
        for its1 in range(config.pre_smooth_its):
            e_s[level], _ = _mg_smooth_one_level(
                level=level,
                e_i=e_s[level],
                b=r_s[level],
                variables_sfc=variables_sfc)
        # get residual on this level
        _, rr = _mg_smooth_one_level(
            level=level,
            e_i=e_s[level],
            b=r_s[level],
            variables_sfc=variables_sfc)
        # restriction
        rr = F.pad(rr.view(1, 1, -1), (0, 1), "constant", 0)
        e_i = F.pad(e_s[level].view(1, 1, -1), (0, 1), "constant", 0)
        with torch.no_grad():
            rr = sfc_restrictor(rr)
            r_s.append(rr.view(1, 1, -1))
            e_i = sfc_restrictor(e_i)
            e_s.append(e_i.view(1, 1, -1))
    for level in range(smooth_start_level, -1, -1):
        if level == smooth_start_level:
            # direct solve on smooth_start_level
            a_on_l = variables_sfc[level][0]
            e_i_direct = linalg.spsolve(a_on_l.tocsr(),
                                        r_s[level].view(-1).cpu().numpy())
            e_s[level] = torch.tensor(e_i_direct, device=config.dev, dtype=torch.float64)
        else:  # smooth
            # prolongation
            CNN1D_prol_odd = nn.Upsample(scale_factor=nodes_per_level[level] / nodes_per_level[level + 1])
            e_s[level] += CNN1D_prol_odd(e_s[level + 1].view(1, 1, -1))
            # post smooth
            for its1 in range(config.post_smooth_its):
                e_s[level], _ = _mg_smooth_one_level(
                    level=level,
                    e_i=e_s[level],
                    b=r_s[level],
                    variables_sfc=variables_sfc)
    return e_s[0].view(cg_nonods).contiguous()


def _mg_on_P0CG_prep(fina, cola, RARvalues):
    '''
    # Prepare for Multi-grid cycle on P0DG mesh

    This function forms space filling curve. Then form
    a series of coarse grid and operators thereon.

    # Input
    fina : torch tensor, (nele+1)
        sparsity - start of rows - of RAR matrix
    cola : torch tensor, (ncola)
        sparsity - column indices - of RAR matrix
    RARvalues : torch tensor, (ncola, ndim, ndim)
        ncola (ndim, ndim) block of RAR matrix

    # Output
    sfc : numpy list (nele)
        space filling curve index for ele.
    variables_sfc : list (nlevel)
        a list of all ingredients one needs to perform a smoothing
        step on level-th grid. Each list member is a list of the
        following member:
        [0] a_sfc_sparse : a 2-D list of torch coo sparse tensor,
            list shape (ndim, ndim)
            coo sparse tensor shape (nonods_level, nonods_level)
            coarse level grid operator
        [1] diag_weights : torch tensor, (ndim, nonods_level)
            diagonal of coarse grid operator
        [2] nonods : integer
            number of nodes on level-th grid
    nlevel : scalar, int
        number of SFC coarse grid levels
    nodes_per_level : list of int, (nlevel)
        number of nodes (DOFs) on each level
    '''

    cg_nonods = sf_nd_nb.pre_func_space.cg_nonods
    dummy = np.zeros((cg_nonods))

    starting_node = 1  # setting according to BY
    graph_trim = -10  # ''
    ncurve = 1  # ''
    nele = config.nele
    ncola = cola.shape[0]
    start_time = time.time()
    print('to get space filling curve...', time.time() - start_time)
    if os.path.isfile(config.filename[:-4] + '_sfc.npy'):
        print('pre-calculated sfc exists. readin from file...')
        sfc = np.load(config.filename[:-4] + '_sfc.npy')
    else:
        _, sfc = \
            sf.ncurve_python_subdomain_space_filling_curve(
                cola + 1, fina + 1, starting_node, graph_trim, ncurve
            )  # note that fortran array index start from 1, so cola and fina should +1.
        np.save(config.filename[:-4] + '_sfc.npy', sfc)
    print('to get sfc operators...', time.time() - start_time)

    # get coarse grid info
    max_nlevel = sf.calculate_nlevel_sfc(cg_nonods) + 1
    max_nonods_sfc_all_grids = 5 * cg_nonods
    max_ncola_sfc_all_un = 10 * ncola
    a_sfc, fina_sfc_all_un, cola_sfc_all_un, ncola_sfc_all_un, b_sfc, \
        ml_sfc, fin_sfc_nonods, nonods_sfc_all_grids, nlevel = \
        map_sfc.best_sfc_mapping_to_sfc_matrix_unstructured(
            a=RARvalues.cpu().numpy(),
            b=dummy,
            ml=dummy,
            fina=fina + 1,
            cola=cola + 1,
            sfc_node_ordering=sfc[:, 0],
            max_nonods_sfc_all_grids=max_nonods_sfc_all_grids,
            max_ncola_sfc_all_un=max_ncola_sfc_all_un,
            max_nlevel=max_nlevel,
            ncola=ncola, nonods=cg_nonods)
    print('back from sfc operator fortran subroutine,', time.time() - start_time)
    nodes_per_level = [fin_sfc_nonods[i] - fin_sfc_nonods[i - 1] for i in range(1, nlevel + 1)]
    # print(fin_sfc_nonods.shape)
    a_sfc = a_sfc[:ncola_sfc_all_un]  # trim following zeros.
    del b_sfc, ml_sfc
    # choose a level to directly solve on. then we'll iterate from there and levels up
    if config.smooth_start_level < 0:
        # for level in range(1,nlevel):
        #     if nodes_per_level[level] < 2:
        #         config.smooth_start_level = level
        #         break
        config.smooth_start_level += nlevel
    print('start_level: ', config.smooth_start_level)
    variables_sfc = []
    for level in range(config.smooth_start_level + 1):
        variables_sfc.append(_get_a_diaga(
            level,
            fin_sfc_nonods,
            fina_sfc_all_un,
            cola_sfc_all_un,
            a_sfc
        ))
    # build A on smooth_start_level. all levels before this are smooth levels,
    # on smooth_start_level, we use direct solve. Therefore, A is replaced with a
    # scipy csr_matrix.
    level = config.smooth_start_level
    a_sfc_l = variables_sfc[level][0]  # this is a torch csr tensors.
    cola = a_sfc_l.col_indices().detach().clone().cpu().numpy()
    fina = a_sfc_l.crow_indices().detach().clone().cpu().numpy()
    vals = np.zeros((cola.shape[0]), dtype=np.float64)
    vals += a_sfc_l.values().detach().clone().cpu().numpy()
    a_on_l = csr_matrix((vals, cola, fina),
                        shape=(nodes_per_level[level], nodes_per_level[level]))
    variables_sfc[level] = (a_on_l, 0, nodes_per_level[level])
    return sfc, variables_sfc, nlevel, nodes_per_level


def get_RAR_and_sfc_data_Lp(
        whichc, ncolor,
        fina, cola, ncola,
):
    """
    prepare for MG on SFC-coarse grids for pressure laplacian operator
    """
    RARvalues = _calc_RAR_mf_color(
        sf_nd_nb.I_dc,
        sf_nd_nb.I_cd,
        whichc, ncolor,
        fina, cola, ncola,
    )
    cg_nonods = sf_nd_nb.pre_func_space.cg_nonods
    if not config.is_sfc:
        RAR = csr_matrix((RARvalues.cpu(), cola, fina),
                         shape=(cg_nonods, cg_nonods))
        sf_nd_nb.set_data(RARmat_Lp=RAR)
    # get SFC, coarse grid and operators on coarse grid. Store them to save computational time?
    space_filling_curve_numbering, variables_sfc, nlevel, nodes_per_level = \
        _mg_on_P0CG_prep(fina, cola, RARvalues)
    sf_nd_nb.sfc_data_Lp.set_data(
        space_filling_curve_numbering=space_filling_curve_numbering,
        variables_sfc=variables_sfc,
        nlevel=nlevel,
        nodes_per_level=nodes_per_level
    )


def pre_precond_Fp(x_p, u_n, u_bc):
    """given vector x_p, apply pressure block's preconditioner:
    x_p <- -F_p x_p
    here F_p is the convection-diffusion(-mass * rho/dt) matrix of pressure
    """
    # x_p should be of dimension (nele, p_nloc) or (p_nonods)
    x_temp = torch.zeros_like(x_p, device=dev, dtype=torch.float64)
    x_temp, _ = _apply_pressure_mat(
        x_temp,
        x_p,
        torch.zeros_like(x_p, device=dev, dtype=torch.float64),
        include_adv=config.include_adv,
        u_n=u_n,
        u_bc=u_bc,
        doSmooth=False,
    )
    x_p *= 0
    x_p += x_temp.view(x_p.shape)
    return x_p.view(-1)


def pre_precond_invLp(x_p, x_rhs):
    """
    this function applies L_p^-1 on the input vector x_p (of size p_nonods),
    L_p^-1 is part of the pressure preconditioner: P_s = Q^-1 F_p L_p^-1

    x_p <- L_p^-1 x_rhs
    """
    nonods = sf_nd_nb.pre_func_space.nonods
    nloc = sf_nd_nb.pre_func_space.element.nloc
    x_p = x_p.view(nonods)  # should start with zero

    cg_nonods = sf_nd_nb.pre_func_space.cg_nonods
    r0 = torch.zeros(nonods, device=dev, dtype=torch.float64)

    # pre smooth
    for its1 in range(config.pre_smooth_its):
        r0 *= 0
        r0, x_p = _apply_pressure_mat(
            r0, x_p, x_rhs=x_rhs,
            include_adv=False,
            doSmooth=True,
        )

    # get residual on PnDG
    r0 *= 0
    r0, _ = _apply_pressure_mat(
        r0, x_p, x_rhs=x_rhs,
        include_adv=False,
        doSmooth=False,
    )

    # restrict residual
    if not config.is_pmg:
        r1 = torch.zeros(cg_nonods, device=dev, dtype=torch.float64)
        r1 += torch.mv(sf_nd_nb.I_cd, mg_le.pre_pndg_to_p1dg_restrictor(r0))
    else:
        raise NotImplemented('PMG not implemented!')

    # smooth/solve on P1CG grid
    if not config.is_sfc:  # two-grid method
        e_i = torch.zeros(cg_nonods, device=dev, dtype=torch.float64)
        e_direct = linalg.spsolve(
            sf_nd_nb.RARmat_Lp,
            r1.contiguous().view(-1).cpu().numpy()
        )
        e_i += torch.tensor(e_direct, device=dev, dtype=torch.float64).view(cg_nonods)
    else:  # multi-grid on space-filling curve generated grids
        ncurve = 1  # always use 1 sfc
        N = len(sf_nd_nb.sfc_data_Lp.space_filling_curve_numbering)
        inverse_numbering = np.zeros((N, ncurve), dtype=int)
        inverse_numbering[:, 0] = np.argsort(sf_nd_nb.sfc_data_Lp.space_filling_curve_numbering[:, 0])
        r1_sfc = r1[inverse_numbering[:, 0]].view(cg_nonods)

        # go to SFC coarse grid levels and do 1 mg cycles there
        e_i = _mg_on_P1CG(
            r1_sfc.view(cg_nonods),
            sf_nd_nb.sfc_data_Lp.variables_sfc,
            sf_nd_nb.sfc_data_Lp.nlevel,
            sf_nd_nb.sfc_data_Lp.nodes_per_level
        )
        # reverse to original order
        e_i = e_i[sf_nd_nb.sfc_data_Lp.space_filling_curve_numbering[:, 0] - 1].view(cg_nonods)

    # prolongate residual
    if not config.is_pmg:
        e_i0 = torch.zeros(nonods, device=dev, dtype=torch.float64)
        e_i0 += mg_le.pre_p1dg_to_pndg_prolongator(torch.mv(sf_nd_nb.I_dc, e_i))
    else:
        raise Exception('pmg not implemented')

    # correct fine grid solution
    x_p += e_i0.view(x_p.shape)

    # post smooth
    for its1 in range(config.pre_smooth_its):
        r0 *= 0
        r0, x_p = _apply_pressure_mat(
            r0, x_p, x_rhs=x_rhs,
            include_adv=False,
            doSmooth=True,
        )
    # print('x_rhs norm: ', torch.linalg.norm(x_rhs.view(-1)), 'r0 norm: ', torch.linalg.norm(r0.view(-1)))
    return x_p.view(nele, nloc)


def pre_precond_invQ(x_p):
    """given vector x_p, apply pressure block's preconditioner:
    x_p <- Q^-1 x_p
    here Q is the mass matrix of pressure space
    """
    # x_p should be of dimension (nele, p_nloc) or (p_nonods)
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    x_p = x_p.view(nele, p_nloc)
    q = sf_nd_nb.pre_func_space.element.n
    _, qdetwei = get_det_nlx(
        nlx=sf_nd_nb.pre_func_space.element.nlx,
        x_loc=sf_nd_nb.pre_func_space.x_ref_in,
        weight=sf_nd_nb.pre_func_space.element.weight,
        nloc=p_nloc,
        ngi=sf_nd_nb.pre_func_space.element.ngi
    )
    Q = torch.einsum(
        'mg,ng,bg->bmn',
        q,
        q,
        qdetwei
    ) / config.mu  # 1/nu Q as pressure preconditioner
    Q = torch.linalg.inv(Q)
    x_temp = torch.einsum('bmn,bn->bm', Q, x_p)
    x_p *= 0
    x_p += x_temp
    return x_p.view(-1)


def _apply_discrete_laplacian(x_in, x_p):
    """
    given a vector x_p, apply
    x_in <- x_in - (B Q^{-1} B^T) x_p

    this can be used to apply multigrid to approximate
    inv( B Q^{-1} B^T )

    ** I just realise it's impossible to get the diagonal of BQ-1Bt
    matrix-freely, so we can't use this. **
    """
    u_nonods = sf_nd_nb.vel_func_space.nonods
    p_nonods = sf_nd_nb.pre_func_space.nonods
    x_u_temp = torch.zeros(u_nonods, ndim, device=dev, dtype=torch.float64)
    x_p_temp = torch.zeros(p_nonods, device=dev, dtype=torch.float64)
    u_dummy = torch.zeros(u_nonods, ndim, device=dev, dtype=torch.float64)
    p_dummy = torch.zeros(p_nonods, device=dev, dtype=torch.float64)

    # G x_p  (or B^T x_p)
    x_u_temp = volume_mf_st.get_residual_only(
        r0=x_u_temp,
        x_i=x_p,
        x_rhs=p_dummy,
        a00=False,
        a01=True,
        a10=False,
        a11=False,
    )
    x_u_temp *= -1

    # Q^{-1} x_u
    # get shape functions
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    n = sf_nd_nb.vel_func_space.element.n
    nx, ndetwei = get_det_nlx(
        nlx=sf_nd_nb.vel_func_space.element.nlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in,
        weight=sf_nd_nb.vel_func_space.element.weight,
        nloc=u_nloc,
        ngi=sf_nd_nb.vel_func_space.element.ngi
    )
    Q = torch.einsum(
        'mg,ng,bg->bmn',
        n,
        n,
        ndetwei
    )
    invQ = torch.linalg.inv(Q)
    x_u_temp = torch.einsum(
        'bmn,bni->bmi',
        invQ,
        x_u_temp.view(nele, u_nloc, ndim)
    )

    # D x_u  (or B x_u, or G^T x_u)
    x_p_temp = volume_mf_st.get_residual_only(
        r0=x_p_temp,
        x_i=x_u_temp,
        x_rhs=u_dummy,
        a00=False,
        a01=False,
        a10=True,
        a11=False,
    )
    x_p_temp *= -1

    x_in *= 0
    x_in += x_p_temp
    return x_in
