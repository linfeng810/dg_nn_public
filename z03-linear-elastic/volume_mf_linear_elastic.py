#!/usr/bin/env python3
'''
This file implements volume integral term
for the linear elastic problem
in finest grid and P0DG grid matrix-free-ly.
Hence the name "volume_mf_linear_elastic".
'''
import torch
from torch import Tensor
import config
from config import sf_nd_nb
import numpy as np
import multigrid_linearelastic as mg_le
import mesh_init
if config.ndim == 2:
    from shape_function import get_det_nlx as get_det_nlx
    from shape_function import sdet_snlx as sdet_snlx
else:
    from shape_function import get_det_nlx_3d as get_det_nlx
    from shape_function import sdet_snlx_3d as sdet_snlx

torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)

dev = config.dev
nele = config.nele 
mesh = config.mesh 
nonods = config.nonods 
ngi = config.ngi
ndim = config.ndim
nloc = config.nloc 
nface = config.ndim+1
sngi = config.sngi
cijkl = config.cijkl
lam = config.lam
mu = config.mu
dt = config.dt 
rho = config.rho 


def calc_RAR_mf_color(I_fc, I_cf,
                      whichc, ncolor,
                      fina, cola, ncola,
                      n=sf_nd_nb.n, nlx=sf_nd_nb.nlx, weight=sf_nd_nb.weight,
                      sn=sf_nd_nb.sn, snlx=sf_nd_nb.snlx, sweight=sf_nd_nb.sweight,
                      x_ref_in=sf_nd_nb.x_ref_in,
                      nbele=sf_nd_nb.nbele, nbf=sf_nd_nb.nbf,
                      ):
    """
    get operator on P1CG grid, i.e. RAR
    where R is prolongator/restrictor,
    via coloring method.
    *ONLY* values of csr format RAR is returned.
    shape (ncola, ndim, ndim)
    """
    import time
    start_time = time.time()
    cg_nonods = sf_nd_nb.cg_nonods
    p1dg_nonods = config.p1dg_nonods
    value = torch.zeros(ncola, ndim, ndim, device=dev, dtype=torch.float64)  # NNZ entry values
    dummy = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)  # dummy variable of same length as PnDG
    Rm = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)
    ARm = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)
    RARm = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
    mask = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)  # color vec
    for color in range(1, ncolor + 1):
        print('color: ', color)
        for jdim in range(ndim):
            mask *= 0
            mask[:, jdim] += torch.tensor((whichc == color),
                                          device=dev,
                                          dtype=torch.float64)  # 1 if true; 0 if false
            Rm *= 0
            for idim in range(ndim):
                # Rm[:, idim] += torch.mv(I_fc, mask[:, idim].view(-1))  # (p1dg_nonods, ndim)
                Rm[:, idim] += mg_le.p1dg_to_p3dg_prolongator(
                    torch.mv(I_fc, mask[:, idim].view(-1))
                )  # (p3dg_nonods, ndim)
            ARm *= 0
            ARm = get_residual_only(ARm,
                                    Rm, dummy, dummy, dummy)
            ARm *= -1.  # (p3dg_nonods, ndim)
            # RARm = multi_grid.p3dg_to_p1dg_restrictor(ARm)  # (p1dg_nonods, )
            RARm *= 0
            for idim in range(ndim):
                RARm[:,idim] += torch.mv(I_cf, mg_le.p3dg_to_p1dg_restrictor(ARm[:,idim]))  # (cg_nonods, ndim)
            for idim in range(ndim):
                # add to value
                for i in range(RARm.shape[0]):
                    for count in range(fina[i], fina[i + 1]):
                        j = cola[count]
                        value[count, idim, jdim] += RARm[i, idim] * mask[j, jdim]
        print('finishing (another) one color, time comsumed: ', time.time()-start_time)

    # RAR = torch.sparse_csr_tensor(crow_indices=fina,
    #                               col_indices=cola,
    #                               values=value,
    #                               size=(cg_nonods, cg_nonods),
    #                               device=dev)
    return value


def get_residual_and_smooth_once(
        r0,
        u_i, u_n, u_bc, f, u_rhs=0):
    """
    update residual
    r0 = b(boundary) + f(body force) - (K+S)*u_i
        + M/dt*u_n (transient last timestep)
    """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0,nnn+1)/nnn*nele, dtype=int)
    # add precalculated rhs to residual
    r0 += u_rhs
    for i in range(nnn):
        # volume integral
        idx_in = torch.zeros(nele, dtype=bool, device=dev)
        idx_in[brk_pnt[i]:brk_pnt[i+1]] = True
        batch_in = torch.sum(idx_in)
        # dummy diagA and bdiagA
        diagA = torch.zeros(batch_in, nloc, ndim, device=dev, dtype=torch.float64)
        bdiagA = torch.zeros(batch_in, nloc, ndim, nloc, ndim, device=dev, dtype=torch.float64)
        r0, diagA, bdiagA = k_mf_one_batch(r0, u_i, u_n, f,
                                           diagA, bdiagA,
                                           idx_in)
        # surface integral
        idx_in_f = torch.zeros(nele * nface, dtype=bool, device=dev)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0, diagA, bdiagA = s_mf_one_batch(r0, u_i, u_bc,
                                           diagA, bdiagA,
                                           idx_in_f, brk_pnt[i])
        # smooth once
        if config.blk_solver == 'direct':
            bdiagA = torch.inverse(bdiagA.view(batch_in, nloc*ndim, nloc*ndim))
            u_i = u_i.view(nele, nloc*ndim)
            u_i[idx_in, :] += config.jac_wei * torch.einsum('...ij,...j->...i',
                                                            bdiagA,
                                                            r0.view(nele, nloc*ndim)[idx_in, :])
        if config.blk_solver == 'jacobi':
            new_b = torch.einsum('...ij,...j->...i',
                                 bdiagA.view(batch_in, nloc * ndim, nloc * ndim),
                                 u_i.view(nele, nloc*ndim)[idx_in, :])\
                    + config.jac_wei * r0.view(nele, nloc*ndim)[idx_in, :]
            new_b = new_b.view(-1)
            diagA = diagA.view(-1)
            u_i = u_i.view(nele, nloc*ndim)
            u_i_partial = u_i[idx_in, :]
            for its in range(3):
                u_i_partial += ((new_b - torch.einsum('...ij,...j->...i',
                                                      bdiagA.view(batch_in, nloc * ndim, nloc * ndim),
                                                      u_i_partial).view(-1))
                                / diagA).view(-1, nloc*ndim)
            u_i[idx_in, :] = u_i_partial.view(-1, nloc*ndim)
    r0 = r0.view(nele*nloc, ndim)
    u_i = u_i.view(nele*nloc, ndim)
    return r0, u_i


def get_residual_only(
        r0,
        u_i, u_n, u_bc, f, u_rhs=0):
    """
    update residual
    r0 = b(boundary) + f(body force) - (K+S)*u_i
        + M/dt*u_n (transient last timestep)
    """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0,nnn+1)/nnn*nele, dtype=int)
    # add pre-computed right hand side to residual
    r0 += u_rhs
    for i in range(nnn):
        # volume integral
        idx_in = torch.zeros(nele, dtype=torch.bool)
        idx_in[brk_pnt[i]:brk_pnt[i+1]] = True
        batch_in = torch.sum(idx_in)
        # dummy diagA and bdiagA
        diagA = torch.zeros(batch_in, nloc, ndim, device=dev, dtype=torch.float64)
        bdiagA = torch.zeros(batch_in, nloc, ndim, nloc, ndim, device=dev, dtype=torch.float64)
        r0, diagA, bdiagA = k_mf_one_batch(r0, u_i, u_n, f,
                                           diagA, bdiagA,
                                           idx_in)
        # surface integral
        idx_in_f = torch.zeros(nele * nface, dtype=torch.bool, device=dev)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0, diagA, bdiagA = s_mf_one_batch(r0, u_i, u_bc,
                                           diagA, bdiagA,
                                           idx_in_f, brk_pnt[i])
    r0 = r0.view(nele*nloc, ndim)
    return r0


def k_mf_one_batch(r0, u_i, u_n, f,
                   diagA, bdiagA,
                   idx_in: Tensor):  # pls make sure idx_in is passed in as a torch Tensor on dev. Otherwise moving
    # it across devices will be very time-consuming!
    '''
    update residual's volume integral parts
    r0 <- r0 - K*u_i + M*f
    if transient, also:
          r0 + M/dt*u_n - M/dt*u_i
    '''

    # get essential data
    n = sf_nd_nb.n; nlx = sf_nd_nb.nlx
    x_ref_in = sf_nd_nb.x_ref_in
    weight = sf_nd_nb.weight

    batch_in = torch.sum(idx_in)
    # change view
    r0 = r0.view(-1, nloc, ndim)
    u_i = u_i.view(-1, nloc, ndim)
    diagA = diagA.view(-1, nloc, ndim)
    bdiagA = bdiagA.view(-1, nloc, ndim, nloc, ndim)
    # get shape function derivatives
    nx, detwei = get_det_nlx(nlx, x_ref_in[idx_in], weight)
    # make all sf tensors in shape (batch_in, ndim, nloc(i), nloc(j), sngi)
    ni = n.unsqueeze(0).unsqueeze(2).expand(batch_in, -1, nloc, -1)
    nj = n.unsqueeze(0).unsqueeze(1).expand(batch_in, nloc, -1, -1)
    nxi = nx.unsqueeze(3).expand(-1, -1, -1, nloc, -1)  # expand on nloc(jnod)
    nxj = nx.unsqueeze(2).expand(-1, -1, nloc, -1, -1)  # expand on nloc(inod)
    detweiv = detwei.unsqueeze(1).unsqueeze(2).expand(-1, nloc, nloc, -1)

    # declare K
    K = torch.zeros(batch_in, nloc, ndim, nloc, ndim, device=dev, dtype=torch.float64)
    # print('in k_mf_one_batch(), batch_in = ', batch_in)
    # ni nj
    for idim in range(ndim):
        K[:, :, idim, :, idim] += torch.sum(torch.mul(torch.mul(ni, nj), detweiv), -1)
    f = f.view(nele, nloc, ndim)
    r0[idx_in, ...] += torch.einsum('...ijkl,...kl->...ij', K, f[idx_in, ...])
    if config.isTransient:
        K *= rho / dt
        u_n = u_n.view(nele, nloc, ndim)
        r0[idx_in, ...] += torch.einsum('...ijkl,...kl->...ij', K,
                                        u_n[idx_in,...]-u_i[idx_in,...])
    else:
        K *= 0

    # epsilon_kl C_ijkl epsilon_ij
    for [idim, jdim, kdim, ldim] in config.ijkldim_nz:
        K[:, :, idim, :, kdim] += torch.sum(torch.mul(torch.mul(
            nxi[:, jdim, :, :, :], nxj[:, ldim, :, :, :]), detweiv), -1) * cijkl[idim, jdim, kdim, ldim]
    # K[:, :, 0, :, 0] += torch.sum(torch.mul(torch.mul(
    #     nxi[:, 0, :, :, :], nxj[:, 0, :, :, :]), detweiv), -1) * (lam + 2 * mu)
    # K[:, :, 0, :, 0] += torch.sum(torch.mul(torch.mul(
    #     nxi[:, 1, :, :, :], nxj[:, 1, :, :, :]), detweiv), -1) * mu
    # K[:, :, 0, :, 1] += torch.sum(torch.mul(torch.mul(
    #     nxi[:, 0, :, :, :], nxj[:, 1, :, :, :]), detweiv), -1) * lam
    # K[:, :, 0, :, 1] += torch.sum(torch.mul(torch.mul(
    #     nxi[:, 1, :, :, :], nxj[:, 0, :, :, :]), detweiv), -1) * mu
    # K[:, :, 1, :, 0] += torch.sum(torch.mul(torch.mul(
    #     nxi[:, 0, :, :, :], nxj[:, 1, :, :, :]), detweiv), -1) * mu
    # K[:, :, 1, :, 0] += torch.sum(torch.mul(torch.mul(
    #     nxi[:, 1, :, :, :], nxj[:, 0, :, :, :]), detweiv), -1) * lam
    # K[:, :, 1, :, 1] += torch.sum(torch.mul(torch.mul(
    #     nxi[:, 0, :, :, :], nxj[:, 0, :, :, :]), detweiv), -1) * mu
    # K[:, :, 1, :, 1] += torch.sum(torch.mul(torch.mul(
    #     nxi[:, 1, :, :, :], nxj[:, 1, :, :, :]), detweiv), -1) * (lam + 2 * mu)

    # update residual -K*u_i
    r0[idx_in,...] -= torch.einsum('...ijkl,...kl->...ij', K, u_i[idx_in, ...])

    # get diagonal
    diagA += torch.diagonal(K.view(batch_in,nloc*ndim, nloc*ndim), dim1=1, dim2=2).view(batch_in, nloc, ndim)
    bdiagA += K
    return r0, diagA, bdiagA


def s_mf_one_batch(r, u_i, u_bc,
                   diagA, bdiagA,
                   idx_in_f: Tensor,  # make sure idx_in_f is passed in as Tensor on designated dev.
                   # Moving it across devices is time-consuming.
                   batch_start_idx):
    """
    update residual's surface integral part for one batch of elements
    r0 <- r0 - S*u_i + S*u_bc
    """

    # get essential data
    nbf = sf_nd_nb.nbf
    alnmt = sf_nd_nb.alnmt

    u_i = u_i.view(nele, nloc, ndim)
    r = r.view(nele, nloc, ndim)
    u_bc = u_bc.view(nele, nloc, ndim)

    # first lets separate nbf to get two list of F_i and F_b
    F_i = torch.where(torch.logical_and(alnmt >= 0, idx_in_f))[0]  # interior face
    F_b = torch.where(torch.logical_and(alnmt < 0, idx_in_f))[0]  # boundary face
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

    # for interior faces, update residual
    # r <= r - S*u_i
    # let's hope that each element has only one boundary face.
    for iface in range(nface):
        for nb_gi_aln in range(nface-1):
            idx_iface = (f_i == iface) & (sf_nd_nb.alnmt[F_i] == nb_gi_aln)
            if idx_iface.sum() < 1:
                # there is nothing to do here, go on
                continue
            r, diagA, bdiagA = _S_fi(
                r, f_i[idx_iface], E_F_i[idx_iface],
                f_inb[idx_iface], E_F_inb[idx_iface],
                u_i,
                diagA, bdiagA, batch_start_idx,
                nb_gi_aln)

    # update residual for boundary faces
    # r <= r + S*u_bc - S*u_i
    if ndim == 3:
        for iface in range(nface):
            idx_iface = f_b == iface
            r, diagA, bdiagA = _S_fb(
                r, f_b[idx_iface], E_F_b[idx_iface],
                u_i, u_bc,
                diagA, bdiagA, batch_start_idx)
    else:
        r, diagA, bdiagA = _S_fb(
            r, f_b, E_F_b,
            u_i, u_bc,
            diagA, bdiagA, batch_start_idx)
    return r, diagA, bdiagA


def _S_fi(r, f_i: Tensor, E_F_i: Tensor,
          f_inb: Tensor, E_F_inb: Tensor,
          u_i,
          diagS, diagS20, batch_start_idx,
          nb_gi_aln):
    """
    this function add interior face S*c contribution
    to r
    """

    # get essential data
    sn = sf_nd_nb.sn
    snlx = sf_nd_nb.snlx
    x_ref_in = sf_nd_nb.x_ref_in
    sweight = sf_nd_nb.sweight

    # faces can be passed in by batches to fit memory/GPU cores
    batch_in = f_i.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)

    # make all tensors in shape (nele, nface, ndim, nloc(inod), nloc(jnod), sngi)
    # all these expansion are views of original tensor,
    # i.e. they point to same memory as sn/snx
    sni = sn.unsqueeze(0).expand(batch_in, -1, -1, -1) \
        .unsqueeze(3).expand(-1, -1, -1, nloc, -1)  # expand on nloc(jnod)
    snj = sn.unsqueeze(0).expand(batch_in, -1, -1, -1) \
        .unsqueeze(2).expand(-1, -1, nloc, -1, -1)  # expand on nloc(inod)

    # get shape function derivatives
    # this side
    snx, sdetwei, snormal = sdet_snlx(snlx, x_ref_in[E_F_i], sweight)
    # now tensor shape are:
    # snx | snx_nb         (batch_in, nface, ndim, nloc, sngi)
    # sdetwei | sdetwei_nb (batch_in, nface, sngi)
    # snormal | snormal_nb (batch_in, nface, ndim)
    mu_e = config.eta_e / torch.sum(sdetwei[dummy_idx, f_i, :], -1)  # mu_e for each face (batch_in)
    snxi = snx.unsqueeze(4) \
        .expand(-1, -1, -1, -1, nloc, -1)  # expand on nloc(jnod)
    snxj = snx.unsqueeze(3) \
        .expand(-1, -1, -1, nloc, -1, -1)  # expand on nloc(inod)
    # make all tensors in shape (nele, nface, nloc(inod), nloc(jnod), sngi)
    # are views of snormal/sdetwei, taken same memory
    snormalv = snormal.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) \
        .expand(-1, -1, -1, nloc, nloc, sngi)
    sdetweiv = sdetwei.unsqueeze(2).unsqueeze(3) \
        .expand(-1, -1, nloc, nloc, -1)

    # other side.
    snx_nb, _, snormal_nb = sdet_snlx(snlx, x_ref_in[E_F_inb], sweight)
    # change gausian pnts alignment on the other side use nb_gi_aln
    nb_aln = sf_nd_nb.gi_align[nb_gi_aln, :]
    snx_nb = snx_nb[..., nb_aln]
    snj_nb = snj[..., nb_aln]
    # snxi_nb = snx_nb.unsqueeze(4) \
    #     .expand(-1, -1, -1, -1, nloc, -1)  # expand on nloc(jnod)
    snxj_nb = snx_nb.unsqueeze(3) \
        .expand(-1, -1, -1, nloc, -1, -1)  # expand on nloc(inod)
    snormalv_nb = snormal_nb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) \
        .expand(-1, -1, -1, nloc, nloc, sngi)
    # sdetweiv_nb = sdetwei_nb.unsqueeze(2).unsqueeze(3) \
    #     .expand(-1, -1, nloc, nloc, -1)
    # make mu_e in shape (batch_in, nloc, nloc)
    mu_ev = mu_e.unsqueeze(-1).unsqueeze(-1).expand(-1, nloc, nloc)

    # this side
    S = torch.zeros(batch_in, nloc, ndim, nloc, ndim,
                    device=dev, dtype=torch.float64)  # local S matrix

    for [idim, jdim, kdim, ldim] in config.ijkldim_nz:
        # epsilon(u)_ij * 0.5 * C_ijkl v_i n_j
        S[..., idim, :, kdim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                sni[dummy_idx, f_i, :, :, :], # v_i
                0.5*snxj[dummy_idx, f_i, ldim, :,:,:] ), # 0.5*du_k/dx_l of epsilon_kl
                snormalv[dummy_idx, f_i, jdim, :,:,:]), # n_j
                sdetweiv[dummy_idx, f_i, :,:,:]),
                -1) \
            * cijkl[idim, jdim, kdim, ldim] * (-0.5)
        S[..., idim, :, ldim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                sni[dummy_idx, f_i, :, :, :], # v_i
                0.5*snxj[dummy_idx, f_i, kdim, :,:,:] ), # 0.5*du_l/dx_k of epsilon_kl
                snormalv[dummy_idx, f_i, jdim, :,:,:]), # n_j
                sdetweiv[dummy_idx, f_i, :,:,:]),
                -1) \
            * cijkl[idim, jdim, kdim, ldim] * (-0.5)
        # u_i n_j * 0.5 * C_ijkl epsilon(v)_kl
        S[...,kdim,:,idim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                snj[dummy_idx, f_i, :, :, :], # u_i
                0.5*snxi[dummy_idx, f_i, ldim, :,:,:] ), # 0.5*dv_k/dx_l of epsilon_kl
                snormalv[dummy_idx, f_i, jdim, :,:,:]), # n_j
                sdetweiv[dummy_idx, f_i, :,:,:]),
                -1)\
            * cijkl[idim,jdim,kdim,ldim]*(-0.5)
        S[..., ldim,:,idim] +=\
            torch.sum(torch.mul(torch.mul(torch.mul(
                snj[dummy_idx, f_i, :, :, :], # u_i
                0.5*snxi[dummy_idx, f_i, kdim, :,:,:] ), # 0.5*dv_l/dx_k of epsilon_kl
                snormalv[dummy_idx,f_i, jdim, :,:,:]), # n_j
                sdetweiv[dummy_idx,f_i, :,:,:]),
                -1)\
            * cijkl[idim,jdim,kdim,ldim]*(-0.5)
    # mu * u_i v_i
    for idim in range(ndim):
        S[...,idim,:,idim] += torch.mul(torch.sum(torch.mul(torch.mul(
            sni[dummy_idx, f_i, :,:,:],
            snj[dummy_idx, f_i, :,:,:]),
            sdetweiv[dummy_idx, f_i, :,:,:]),
            -1), mu_ev)
    # multiply S and c_i and add to (subtract from) r
    r[E_F_i, :, :] -= torch.einsum('...ijkl,...kl->...ij', S, u_i[E_F_i, :, :])
    # put diagonal of S into diagS
    diagS[E_F_i-batch_start_idx, :, :] += torch.diagonal(S.view(batch_in, nloc*ndim, nloc*ndim),
                                                         dim1=1, dim2=2).view(batch_in, nloc, ndim)
    diagS20[E_F_i-batch_start_idx, ...] += S
    # print('torch.flip')
    # print(torch.flip(snxj_nb[dummy_idx, f_inb, ldim, :,:,:],[-1]) )
    # print('torch inx')
    # print(snxj_nb[dummy_idx, f_inb, ldim, :,:,:][...,mesh_init.sgi_order_on_other_side(sngi)] )
    # print(torch.allclose(torch.flip(snxj_nb[dummy_idx, f_inb, ldim, :,:,:],[-1]),
    #                     snxj_nb[dummy_idx, f_inb, ldim, :,:,:][...,mesh_init.sgi_order_on_other_side(sngi)] ))
    # other side
    S *= 0.  # local S matrix
    for [idim, jdim, kdim, ldim] in config.ijkldim_nz:
        # TODO: We might be able to simplify this computation leveraging
        #   the minor symmetry of C_ijkl.
        # 0.5 C_ijkl epsilon(u)_kl v_i n_j
        S[..., idim, :, kdim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                sni[dummy_idx, f_i, :, :, :],  # v_i
                0.5*snxj_nb[dummy_idx, f_inb, ldim, :,:,:]), # 0.5*du_k/dx_l of epsilon_kl
                snormalv[dummy_idx, f_i, jdim, :,:,:]), # n_j
                sdetweiv[dummy_idx, f_i, :,:,:]),
                -1)\
            * cijkl[idim,jdim,kdim,ldim]*(-0.5)
        S[..., idim, :, ldim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                sni[dummy_idx, f_i, :, :, :], # v_i
                0.5*snxj_nb[dummy_idx, f_inb, kdim, :,:,:]), # 0.5*du_l/dx_k of epsilon_kl
                snormalv[dummy_idx, f_i, jdim, :,:,:]), # n_j
                sdetweiv[dummy_idx, f_i, :,:,:]),
                -1)\
            * cijkl[idim,jdim,kdim,ldim]*(-0.5)
        # u_i n_j * 0.5 * C_ijkl epsilon(v)_kl
        S[..., kdim, :, idim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                snj_nb[dummy_idx, f_inb, :, :, :], # u_i
                0.5*snxi[dummy_idx, f_i, ldim, :,:,:] ), # 0.5*dv_k/dx_l of epsilon_kl
                snormalv_nb[dummy_idx, f_inb, jdim, :,:,:]), # n_j
                sdetweiv[dummy_idx, f_i, :,:,:]),
                -1)\
            * cijkl[idim,jdim,kdim,ldim]*(-0.5)
        S[..., ldim, :, idim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                snj_nb[dummy_idx, f_inb, :, :, :], # u_i
                0.5*snxi[dummy_idx, f_i, kdim, :,:,:] ), # 0.5*dv_l/dx_k of epsilon_kl
                snormalv_nb[dummy_idx, f_inb, jdim, :,:,:]), # n_j
                sdetweiv[dummy_idx, f_i, :,:,:]),
                -1)\
            * cijkl[idim,jdim,kdim,ldim]*(-0.5)
    # mu * u_i v_i
    for idim in range(ndim):
        S[..., idim, :, idim] += torch.mul(torch.sum(torch.mul(torch.mul(
            sni[dummy_idx, f_i, :,:,:],
            snj_nb[dummy_idx, f_inb, :,:,:]),
            sdetweiv[dummy_idx, f_i, :,:,:]),
            -1), -mu_ev)
    # this S is off-diagonal contribution, therefore no need to put in diagS
    # multiply S and c_i and add to (subtract from) r
    r[E_F_i, :, :] -= torch.einsum('...ijkl,...kl->...ij', S, u_i[E_F_inb, :, :])

    return r, diagS, diagS20


def _S_fb(
        r, f_b: Tensor, E_F_b: Tensor,
        u_i, u_bc,
        diagS, diagS20, batch_start_idx):
    """
    update residual for boundary faces
    """

    # get essential data
    sn = sf_nd_nb.sn
    snlx = sf_nd_nb.snlx
    x_ref_in = sf_nd_nb.x_ref_in
    sweight = sf_nd_nb.sweight

    # faces can be passed in by batches to fit memory/GPU cores
    batch_in = f_b.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)

    # make all tensors in shape (nele, nface, ndim, nloc(inod), nloc(jnod), sngi)
    # all these expansion are views of original tensor,
    # i.e. they point to same memory as sn/snx
    sni = sn.unsqueeze(0).expand(nele, -1, -1, -1) \
        .unsqueeze(3).expand(-1, -1, -1, nloc, -1)  # expand on nloc(jnod)
    snj = sn.unsqueeze(0).expand(nele, -1, -1, -1) \
        .unsqueeze(2).expand(-1, -1, nloc, -1, -1)  # expand on nloc(inod)

    # get shape function derivatives
    snx, sdetwei, snormal = sdet_snlx(snlx, x_ref_in[E_F_b], sweight)
    mu_e = config.eta_e / torch.sum(sdetwei[dummy_idx, f_b, :], -1)
    snxi = snx.unsqueeze(4) \
        .expand(-1, -1, -1, -1, nloc, -1)  # expand on nloc(jnod)
    snxj = snx.unsqueeze(3) \
        .expand(-1, -1, -1, nloc, -1, -1)  # expand on nloc(inod)
    # make all tensors in shape (nele, nface, nloc(inod), nloc(jnod), sngi)
    # are views of snormal/sdetwei, taken same memory
    snormalv = snormal.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) \
        .expand(-1, -1, -1, nloc, nloc, sngi)
    sdetweiv = sdetwei.unsqueeze(2).unsqueeze(3) \
        .expand(-1, -1, nloc, nloc, -1)
    # make mu_e in shape (batch_in, nloc, nloc)
    mu_e = mu_e.unsqueeze(-1).unsqueeze(-1) \
        .expand(-1, nloc, nloc)

    # only one side
    S = torch.zeros(batch_in, nloc, ndim, nloc, ndim,
                    device=dev, dtype=torch.float64)  # local S matrix
    # u_i n_j * 0.5 * C_ijkl epsilon(v)_kl
    for [idim, jdim, kdim, ldim] in config.ijkldim_nz:
        S[..., kdim, :, idim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                snj[dummy_idx, f_b, :, :, :],  # u_i
                0.5 * snxi[dummy_idx, f_b, ldim, :, :, :]),  # 0.5*dv_k/dx_l of epsilon_kl
                snormalv[dummy_idx, f_b, jdim, :, :, :]),  # n_j
                sdetweiv[dummy_idx, f_b, :, :, :]),
                -1) \
            * cijkl[idim, jdim, kdim, ldim] * (-1.0)
        S[..., ldim, :, idim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                snj[dummy_idx, f_b, :, :, :],  # u_i
                0.5 * snxi[dummy_idx, f_b, kdim, :, :, :]),  # 0.5*dv_l/dx_k of epsilon_kl
                snormalv[dummy_idx, f_b, jdim, :, :, :]),  # n_j
                sdetweiv[dummy_idx, f_b, :, :, :]),
                -1) \
            * cijkl[idim, jdim, kdim, ldim] * (-1.0)
    # mu * u_i v_i
    for idim in range(ndim):
        S[..., idim, :, idim] += torch.mul(
            torch.sum(torch.mul(torch.mul(
                sni[dummy_idx, f_b, :, :, :],
                snj[dummy_idx, f_b, :, :, :]),
                sdetweiv[dummy_idx, f_b, :, :, :]),
                -1), mu_e)
    # calculate b_bc and add to r:
    # r <- r + b_bc
    r[E_F_b, :, :] += torch.einsum('...ijkl,...kl->...ij', S, u_bc[E_F_b, :, :])

    # 0.5 C_ijkl epsilon(u)_kl v_i n_j
    for [idim, jdim, kdim, ldim] in config.ijkldim_nz:
        S[..., idim, :, kdim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                sni[dummy_idx, f_b, :, :, :],  # v_i
                0.5 * snxj[dummy_idx, f_b, ldim, :, :, :]),  # 0.5*du_k/dx_l of epsilon_kl
                snormalv[dummy_idx, f_b, jdim, :, :, :]),  # n_j
                sdetweiv[dummy_idx, f_b, :, :, :]),
                -1) \
            * cijkl[idim, jdim, kdim, ldim] * (-1.0)
        S[..., idim, :, ldim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                sni[dummy_idx, f_b, :, :, :],  # v_i
                0.5 * snxj[dummy_idx, f_b, kdim, :, :, :]),  # 0.5*du_l/dx_k of epsilon_kl
                snormalv[dummy_idx, f_b, jdim, :, :, :]),  # n_j
                sdetweiv[dummy_idx, f_b, :, :, :]),
                -1) \
            * cijkl[idim, jdim, kdim, ldim] * (-1.0)
    # multiply S and u_i and add to (subtract from) r
    r[E_F_b, :, :] -= torch.einsum('...ijkl,...kl->...ij', S, u_i[E_F_b, :, :])
    diagS[E_F_b - batch_start_idx, :, :] += torch.diagonal(S.view(batch_in, nloc*ndim, nloc*ndim),
                                                           dim1=-2, dim2=-1).view(batch_in, nloc, ndim)
    diagS20[E_F_b - batch_start_idx, ...] += S

    return r, diagS, diagS20


def pmg_get_residual_and_smooth_once(r0, u_i, po: int):
    """
    do one smooth step on p-order DG mesh
    input:
    r0 : rhs residual on this p level
    u_i : lhs error vector on this p level
    po : order
    """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0,nnn+1)/nnn*nele, dtype=int)
    for i in range(nnn):
        # volume integral
        idx_in = torch.zeros(nele, dtype=bool, device=dev)
        idx_in[brk_pnt[i]:brk_pnt[i+1]] = True
        batch_in = torch.sum(idx_in)
        # dummy diagA and bdiagA
        diagA = torch.zeros(batch_in, mg_le.p_nloc(po), ndim,
                            device=dev, dtype=torch.float64)
        bdiagA = torch.zeros(batch_in, mg_le.p_nloc(po), ndim, mg_le.p_nloc(po), ndim,
                             device=dev, dtype=torch.float64)
        r0, diagA, bdiagA = _pmg_k_mf_one_batch(r0, u_i,
                                                diagA, bdiagA,
                                                idx_in, po)
        # surface integral
        idx_in_f = torch.zeros(nele * nface, dtype=bool, device=dev)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0, diagA, bdiagA = _pmg_s_mf_one_batch(r0, u_i,
                                                diagA, bdiagA,
                                                idx_in_f, brk_pnt[i],
                                                po)
        # smooth once
        if config.blk_solver == 'direct':
            bdiagA = torch.inverse(bdiagA.view(batch_in, mg_le.p_nloc(po)*ndim, mg_le.p_nloc(po)*ndim))
            u_i = u_i.view(nele, mg_le.p_nloc(po)*ndim)
            u_i[idx_in, :] += config.jac_wei * torch.einsum('...ij,...j->...i',
                                                            bdiagA,
                                                            r0.view(nele, mg_le.p_nloc(po)*ndim)[idx_in, :])
        if config.blk_solver == 'jacobi':
            new_b = torch.einsum('...ij,...j->...i',
                                 bdiagA.view(batch_in, mg_le.p_nloc(po)*ndim, mg_le.p_nloc(po)*ndim),
                                 u_i.view(nele, mg_le.p_nloc(po)*ndim)[idx_in, :])\
                    + config.jac_wei * r0.view(nele, mg_le.p_nloc(po)*ndim)[idx_in, :]
            new_b = new_b.view(-1)
            diagA = diagA.view(-1)
            u_i = u_i.view(nele, mg_le.p_nloc(po)*ndim)
            u_i_partial = u_i[idx_in, :]
            for its in range(3):
                u_i_partial += ((new_b - torch.einsum('...ij,...j->...i',
                                                      bdiagA.view(batch_in,
                                                                  mg_le.p_nloc(po)*ndim,
                                                                  mg_le.p_nloc(po)*ndim),
                                                      u_i_partial).view(-1))
                                / diagA).view(-1, mg_le.p_nloc(po)*ndim)
            u_i[idx_in, :] = u_i_partial.view(-1, mg_le.p_nloc(po) * ndim)
    r0 = r0.view(-1, ndim)
    u_i = u_i.view(-1, ndim)
    return r0, u_i


def pmg_get_residual_only(r0, u_i, po: int):
    """
    get residual of error equation on p-order DG mesh
    i.e. rr_p = r_p - Ae
    input:
    r0 : rhs residual on this p level (r_p)
    u_i : lhs error vector on this p level (e)
    po : order
    """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0,nnn+1)/nnn*nele, dtype=int)
    for i in range(nnn):
        # volume integral
        idx_in = torch.zeros(nele, dtype=bool, device=dev)
        idx_in[brk_pnt[i]:brk_pnt[i+1]] = True
        batch_in = torch.sum(idx_in)
        # dummy diagA and bdiagA
        diagA = torch.zeros(batch_in, mg_le.p_nloc(po), ndim, device=dev, dtype=torch.float64)
        bdiagA = torch.zeros(batch_in, mg_le.p_nloc(po), ndim, mg_le.p_nloc(po), ndim,
                             device=dev, dtype=torch.float64)
        r0, diagA, bdiagA = _pmg_k_mf_one_batch(r0, u_i,
                                                diagA, bdiagA,
                                                idx_in, po)
        # surface integral
        idx_in_f = torch.zeros(nele * nface, dtype=bool, device=dev)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0, diagA, bdiagA = _pmg_s_mf_one_batch(r0, u_i,
                                                diagA, bdiagA,
                                                idx_in_f, brk_pnt[i],
                                                po)
    r0 = r0.view(-1, ndim)
    return r0


def _pmg_k_mf_one_batch(
        r0, u_i,
        diagA, bdiagA,
        idx_in: Tensor, po: int):
    '''
    update residual's volume integral parts
    r0 <- r0 - K*u_i
    '''

    # get essential data
    n = sf_nd_nb.n
    nlx = sf_nd_nb.nlx
    x_ref_in = sf_nd_nb.x_ref_in
    weight = sf_nd_nb.weight

    batch_in = torch.sum(idx_in)
    # change view
    r0 = r0.view(-1, mg_le.p_nloc(po), ndim)
    u_i = u_i.view(-1, mg_le.p_nloc(po), ndim)
    diagA = diagA.view(-1, mg_le.p_nloc(po), ndim)
    bdiagA = bdiagA.view(-1, mg_le.p_nloc(po), ndim, mg_le.p_nloc(po), ndim)
    # get shape function derivatives
    nx, detwei = get_det_nlx(nlx, x_ref_in[idx_in], weight)
    # make all sf tensors in shape (batch_in, ndim, nloc(i), nloc(j), sngi)
    # ni = n.unsqueeze(0).unsqueeze(2).expand(batch_in, -1, nloc, -1)
    # nj = n.unsqueeze(0).unsqueeze(1).expand(batch_in, nloc, -1, -1)
    nxi = nx.unsqueeze(3).expand(-1, -1, -1, nloc, -1)  # expand on nloc(jnod)
    nxj = nx.unsqueeze(2).expand(-1, -1, nloc, -1, -1)  # expand on nloc(inod)
    detweiv = detwei.unsqueeze(1).unsqueeze(2).expand(-1, nloc, nloc, -1)

    # declare K
    K = torch.zeros(batch_in, nloc, ndim, nloc, ndim, device=dev, dtype=torch.float64)

    # epsilon_kl C_ijkl epsilon_ij
    for [idim, jdim, kdim, ldim] in config.ijkldim_nz:
        K[:, :, idim, :, kdim] += torch.sum(torch.mul(torch.mul(
            nxi[:, jdim, :, :, :], nxj[:, ldim, :, :, :]), detweiv), -1) * cijkl[idim, jdim, kdim, ldim]
    # K[:, :, 0, :, 0] += torch.sum(torch.mul(torch.mul(
    #     nxi[:, 0, :, :, :], nxj[:, 0, :, :, :]), detweiv), -1) * (lam + 2 * mu)
    # K[:, :, 0, :, 0] += torch.sum(torch.mul(torch.mul(
    #     nxi[:, 1, :, :, :], nxj[:, 1, :, :, :]), detweiv), -1) * mu
    # K[:, :, 0, :, 1] += torch.sum(torch.mul(torch.mul(
    #     nxi[:, 0, :, :, :], nxj[:, 1, :, :, :]), detweiv), -1) * lam
    # K[:, :, 0, :, 1] += torch.sum(torch.mul(torch.mul(
    #     nxi[:, 1, :, :, :], nxj[:, 0, :, :, :]), detweiv), -1) * mu
    # K[:, :, 1, :, 0] += torch.sum(torch.mul(torch.mul(
    #     nxi[:, 0, :, :, :], nxj[:, 1, :, :, :]), detweiv), -1) * mu
    # K[:, :, 1, :, 0] += torch.sum(torch.mul(torch.mul(
    #     nxi[:, 1, :, :, :], nxj[:, 0, :, :, :]), detweiv), -1) * lam
    # K[:, :, 1, :, 1] += torch.sum(torch.mul(torch.mul(
    #     nxi[:, 0, :, :, :], nxj[:, 0, :, :, :]), detweiv), -1) * mu
    # K[:, :, 1, :, 1] += torch.sum(torch.mul(torch.mul(
    #     nxi[:, 1, :, :, :], nxj[:, 1, :, :, :]), detweiv), -1) * (lam + 2 * mu)

    # get operator on this p-level
    K = torch.einsum('pi,...ijkl,kq->...pjql',
                     mg_le.p_restrictor(3, po),
                     K,
                     mg_le.p_prolongator(po, 3)).contiguous()
    # update residual -K*u_i
    r0[idx_in,...] -= torch.einsum('...ijkl,...kl->...ij', K, u_i[idx_in, ...])

    # get diagonal
    diagA += torch.diagonal(K.view(batch_in, mg_le.p_nloc(po)*ndim, mg_le.p_nloc(po)*ndim),
                            dim1=1, dim2=2).contiguous().view(batch_in, mg_le.p_nloc(po), ndim)
    bdiagA += K
    return r0, diagA, bdiagA


def _pmg_s_mf_one_batch(r, u_i,
                        diagA, bdiagA,
                        idx_in_f: Tensor, batch_start_idx,
                        po):
    """
    update residual's surface integral part for one batch of elements
    r0 <- r0 - S*u_i + S*u_bc
    """

    # get essential data
    nbf = sf_nd_nb.nbf
    alnmt = sf_nd_nb.alnmt

    u_i = u_i.view(nele, mg_le.p_nloc(po), ndim)
    r = r.view(nele, mg_le.p_nloc(po), ndim)

    # first lets separate nbf to get two list of F_i and F_b
    F_i = torch.where(torch.logical_and(alnmt >= 0, idx_in_f))[0]  # interior face
    F_b = torch.where(torch.logical_and(alnmt < 0, idx_in_f))[0]  # boundary face
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

    # for interior faces, update residual
    # r <= r - S*u_i
    # let's hope that each element has only one boundary face.
    for iface in range(nface):
        for nb_gi_aln in range(nface-1):
            idx_iface = (f_i == iface) & (sf_nd_nb.alnmt[F_i] == nb_gi_aln)
            if idx_iface.sum < 1:
                # there is nothing to do here, go on
                continue
            r, diagA, bdiagA = _pmg_S_fi(
                r, f_i[idx_iface], E_F_i[idx_iface],
                f_inb[idx_iface], E_F_inb[idx_iface],
                u_i,
                diagA, bdiagA, batch_start_idx,
                nb_gi_aln, po)

    # update residual for boundary faces
    # r <= r + S*u_bc - S*u_i
    if ndim == 3:
        for iface in range(nface):
            idx_iface = f_b == iface
            r, diagA, bdiagA = _S_fb(
                r, f_b[idx_iface], E_F_b[idx_iface],
                u_i,
                diagA, bdiagA, batch_start_idx, po)
    else:
        r, diagA, bdiagA = _pmg_S_fb(
            r, f_b, E_F_b,
            u_i,
            diagA, bdiagA, batch_start_idx, po)
    return r, diagA, bdiagA


def _pmg_S_fi(
        r, f_i: Tensor, E_F_i: Tensor,
        f_inb: Tensor, E_F_inb: Tensor,
        u_i,
        diagS, diagS20, batch_start_idx,
        nb_gi_aln, po):
    """
    this function add interior face S*c contribution
    to r
    """

    # get essential data
    sn = sf_nd_nb.sn
    snlx = sf_nd_nb.snlx
    x_ref_in = sf_nd_nb.x_ref_in
    sweight = sf_nd_nb.sweight

    # faces can be passed in by batches to fit memory/GPU cores
    batch_in = f_i.shape[0]
    dummy_idx = torch.arange(0, batch_in, dtype=torch.int64, device=dev)

    # make all tensors in shape (nele, nface, ndim, nloc(inod), nloc(jnod), sngi)
    # all these expansion are views of original tensor,
    # i.e. they point to same memory as sn/snx
    sni = sn.unsqueeze(0).expand(batch_in, -1, -1, -1) \
        .unsqueeze(3).expand(-1, -1, -1, nloc, -1)  # expand on nloc(jnod)
    snj = sn.unsqueeze(0).expand(batch_in, -1, -1, -1) \
        .unsqueeze(2).expand(-1, -1, nloc, -1, -1)  # expand on nloc(inod)

    # get shape function derivatives
    # this side
    snx, sdetwei, snormal = sdet_snlx(snlx, x_ref_in[E_F_i], sweight)
    # now tensor shape are:
    # snx | snx_nb         (batch_in, nface, ndim, nloc, sngi)
    # sdetwei | sdetwei_nb (batch_in, nface, sngi)
    # snormal | snormal_nb (batch_in, nface, ndim)
    mu_e = config.eta_e / torch.sum(sdetwei[dummy_idx, f_i, :], -1)  # mu_e for each face (batch_in)
    snxi = snx.unsqueeze(4) \
        .expand(-1, -1, -1, -1, nloc, -1)  # expand on nloc(jnod)
    snxj = snx.unsqueeze(3) \
        .expand(-1, -1, -1, nloc, -1, -1)  # expand on nloc(inod)
    # make all tensors in shape (nele, nface, nloc(inod), nloc(jnod), sngi)
    # are views of snormal/sdetwei, taken same memory
    snormalv = snormal.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) \
        .expand(-1, -1, -1, nloc, nloc, sngi)
    sdetweiv = sdetwei.unsqueeze(2).unsqueeze(3) \
        .expand(-1, -1, nloc, nloc, -1)

    # other side.
    snx_nb, sdetwei_nb, snormal_nb = sdet_snlx(snlx, x_ref_in[E_F_inb], sweight)
    # change gausian pnts alignment on the other side use nb_gi_aln
    nb_aln = sf_nd_nb.gi_align[nb_gi_aln, :]
    snx_nb = snx_nb[..., nb_aln]
    snj_nb = snj[..., nb_aln]
    # snxi_nb = snx_nb.unsqueeze(4) \
    #     .expand(-1, -1, -1, -1, nloc, -1)  # expand on nloc(jnod)
    snxj_nb = snx_nb.unsqueeze(3) \
        .expand(-1, -1, -1, nloc, -1, -1)  # expand on nloc(inod)
    snormalv_nb = snormal_nb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) \
        .expand(-1, -1, -1, nloc, nloc, sngi)
    # sdetweiv_nb = sdetwei_nb.unsqueeze(2).unsqueeze(3) \
    #     .expand(-1, -1, nloc, nloc, -1)
    # make mu_e in shape (batch_in, nloc, nloc)
    mu_ev = mu_e.unsqueeze(-1).unsqueeze(-1).expand(-1, nloc, nloc)

    # this side
    S = torch.zeros(batch_in, nloc, ndim, nloc, ndim,
                    device=dev, dtype=torch.float64)  # local S matrix

    for [idim, jdim, kdim, ldim] in config.ijkldim_nz:
        # epsilon(u)_ij * 0.5 * C_ijkl v_i n_j
        S[..., idim, :, kdim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                sni[dummy_idx, f_i, :, :, :], # v_i
                0.5*snxj[dummy_idx, f_i, ldim, :,:,:] ), # 0.5*du_k/dx_l of epsilon_kl
                snormalv[dummy_idx, f_i, jdim, :,:,:]), # n_j
                sdetweiv[dummy_idx, f_i, :,:,:]),
                -1) \
            * cijkl[idim, jdim, kdim, ldim] * (-0.5)
        S[..., idim, :, ldim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                sni[dummy_idx, f_i, :, :, :], # v_i
                0.5*snxj[dummy_idx, f_i, kdim, :,:,:] ), # 0.5*du_l/dx_k of epsilon_kl
                snormalv[dummy_idx, f_i, jdim, :,:,:]), # n_j
                sdetweiv[dummy_idx, f_i, :,:,:]),
                -1) \
            * cijkl[idim, jdim, kdim, ldim] * (-0.5)
        # u_i n_j * 0.5 * C_ijkl epsilon(v)_kl
        S[...,kdim,:,idim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                snj[dummy_idx, f_i, :, :, :], # u_i
                0.5*snxi[dummy_idx, f_i, ldim, :,:,:] ), # 0.5*dv_k/dx_l of epsilon_kl
                snormalv[dummy_idx, f_i, jdim, :,:,:]), # n_j
                sdetweiv[dummy_idx, f_i, :,:,:]),
                -1)\
            * cijkl[idim,jdim,kdim,ldim]*(-0.5)
        S[..., ldim,:,idim] +=\
            torch.sum(torch.mul(torch.mul(torch.mul(
                snj[dummy_idx, f_i, :, :, :], # u_i
                0.5*snxi[dummy_idx, f_i, kdim, :,:,:] ), # 0.5*dv_l/dx_k of epsilon_kl
                snormalv[dummy_idx,f_i, jdim, :,:,:]), # n_j
                sdetweiv[dummy_idx,f_i, :,:,:]),
                -1)\
            * cijkl[idim,jdim,kdim,ldim]*(-0.5)
    # mu * u_i v_i
    for idim in range(ndim):
        S[...,idim,:,idim] += torch.mul(torch.sum(torch.mul(torch.mul(
            sni[dummy_idx, f_i, :,:,:],
            snj[dummy_idx, f_i, :,:,:]),
            sdetweiv[dummy_idx, f_i, :,:,:]),
            -1), mu_ev)
    # get S on p order grid
    S = torch.einsum('pi,...ijkl,kq->...pjql',
                     mg_le.p_restrictor(3, po),
                     S,
                     mg_le.p_prolongator(po, 3)).contiguous()
    # multiply S and c_i and add to (subtract from) r
    r[E_F_i, :, :] -= torch.einsum('...ijkl,...kl->...ij', S, u_i[E_F_i, :, :])
    # put diagonal of S into diagS
    diagS[E_F_i-batch_start_idx, :, :] += torch.diagonal(
        S.view(batch_in, mg_le.p_nloc(po)*ndim, mg_le.p_nloc(po)*ndim),
        dim1=1, dim2=2).view(batch_in, mg_le.p_nloc(po), ndim)
    diagS20[E_F_i-batch_start_idx, ...] += S

    # other side
    S = torch.zeros(batch_in, nloc, ndim, nloc, ndim,
                    device=dev, dtype=torch.float64)  # local S matrix
    for [idim, jdim, kdim, ldim] in config.ijkldim_nz:
        # 0.5 C_ijkl epsilon(u)_kl v_i n_j
        S[..., idim, :, kdim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                sni[dummy_idx, f_i, :, :, :],  # v_i
                0.5*snxj_nb[dummy_idx, f_inb, ldim, :,:,:]), # 0.5*du_k/dx_l of epsilon_kl
                snormalv[dummy_idx, f_i, jdim, :,:,:]), # n_j
                sdetweiv[dummy_idx, f_i, :,:,:]),
                -1)\
            * cijkl[idim,jdim,kdim,ldim]*(-0.5)
        S[..., idim, :, ldim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                sni[dummy_idx, f_i, :, :, :], # v_i
                0.5*snxj_nb[dummy_idx, f_inb, kdim, :,:,:]), # 0.5*du_l/dx_k of epsilon_kl
                snormalv[dummy_idx, f_i, jdim, :,:,:]), # n_j
                sdetweiv[dummy_idx, f_i, :,:,:]),
                -1)\
            * cijkl[idim,jdim,kdim,ldim]*(-0.5)
        # u_i n_j * 0.5 * C_ijkl epsilon(v)_kl
        S[..., kdim, :, idim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                snj_nb[dummy_idx, f_inb, :, :, :], # u_i
                0.5*snxi[dummy_idx, f_i, ldim, :,:,:] ), # 0.5*dv_k/dx_l of epsilon_kl
                snormalv_nb[dummy_idx, f_inb, jdim, :,:,:]), # n_j
                sdetweiv[dummy_idx, f_i, :,:,:]),
                -1)\
            * cijkl[idim,jdim,kdim,ldim]*(-0.5)
        S[..., ldim, :, idim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                snj_nb[dummy_idx, f_inb, :, :, :], # u_i
                0.5*snxi[dummy_idx, f_i, kdim, :,:,:] ), # 0.5*dv_l/dx_k of epsilon_kl
                snormalv_nb[dummy_idx, f_inb, jdim, :,:,:]), # n_j
                sdetweiv[dummy_idx, f_i, :,:,:]),
                -1)\
            * cijkl[idim,jdim,kdim,ldim]*(-0.5)
    # mu * u_i v_i
    for idim in range(ndim):
        S[..., idim, :, idim] += torch.mul(torch.sum(torch.mul(torch.mul(
            sni[dummy_idx, f_i, :,:,:],
            snj_nb[dummy_idx, f_inb, :,:,:]),
            sdetweiv[dummy_idx, f_i, :,:,:]),
            -1), -mu_ev)
    # this S is off-diagonal contribution, therefore no need to put in diagS
    # get S on p order grid
    S = torch.einsum('pi,...ijkl,kq->...pjql',
                     mg_le.p_restrictor(3, po),
                     S,
                     mg_le.p_prolongator(po, 3))
    # multiply S and c_i and add to (subtract from) r
    r[E_F_i, :, :] -= torch.einsum('...ijkl,...kl->...ij', S, u_i[E_F_inb, :, :])

    return r, diagS, diagS20


def _pmg_S_fb(
        r, f_b: Tensor, E_F_b: Tensor,
        u_i,
        diagS, diagS20, batch_start_idx, po):
    """
    update residual for boundary faces
    """

    # get essential data
    sn = sf_nd_nb.sn
    snlx = sf_nd_nb.snlx
    x_ref_in = sf_nd_nb.x_ref_in
    sweight = sf_nd_nb.sweight

    # faces can be passed in by batches to fit memory/GPU cores
    batch_in = f_b.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)

    # make all tensors in shape (nele, nface, ndim, nloc(inod), nloc(jnod), sngi)
    # all these expansion are views of original tensor,
    # i.e. they point to same memory as sn/snx
    sni = sn.unsqueeze(0).expand(nele, -1, -1, -1) \
        .unsqueeze(3).expand(-1, -1, -1, nloc, -1)  # expand on nloc(jnod)
    snj = sn.unsqueeze(0).expand(nele, -1, -1, -1) \
        .unsqueeze(2).expand(-1, -1, nloc, -1, -1)  # expand on nloc(inod)

    # get shape function derivatives
    snx, sdetwei, snormal = sdet_snlx(snlx, x_ref_in[E_F_b], sweight)
    mu_e = config.eta_e / torch.sum(sdetwei[dummy_idx, f_b, :], -1)
    snxi = snx.unsqueeze(4) \
        .expand(-1, -1, -1, -1, nloc, -1)  # expand on nloc(jnod)
    snxj = snx.unsqueeze(3) \
        .expand(-1, -1, -1, nloc, -1, -1)  # expand on nloc(inod)
    # make all tensors in shape (nele, nface, nloc(inod), nloc(jnod), sngi)
    # are views of snormal/sdetwei, taken same memory
    snormalv = snormal.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) \
        .expand(-1, -1, -1, nloc, nloc, sngi)
    sdetweiv = sdetwei.unsqueeze(2).unsqueeze(3) \
        .expand(-1, -1, nloc, nloc, -1)
    # make mu_e in shape (batch_in, nloc, nloc)
    mu_e = mu_e.unsqueeze(-1).unsqueeze(-1) \
        .expand(-1, nloc, nloc)

    # only one side
    S = torch.zeros(batch_in, nloc, ndim, nloc, ndim,
                    device=dev, dtype=torch.float64)  # local S matrix
    # u_i n_j * 0.5 * C_ijkl epsilon(v)_kl
    for [idim, jdim, kdim, ldim] in config.ijkldim_nz:
        S[..., kdim, :, idim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                snj[dummy_idx, f_b, :, :, :],  # u_i
                0.5 * snxi[dummy_idx, f_b, ldim, :, :, :]),  # 0.5*dv_k/dx_l of epsilon_kl
                snormalv[dummy_idx, f_b, jdim, :, :, :]),  # n_j
                sdetweiv[dummy_idx, f_b, :, :, :]),
                -1) \
            * cijkl[idim, jdim, kdim, ldim] * (-1.0)
        S[..., ldim, :, idim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                snj[dummy_idx, f_b, :, :, :],  # u_i
                0.5 * snxi[dummy_idx, f_b, kdim, :, :, :]),  # 0.5*dv_l/dx_k of epsilon_kl
                snormalv[dummy_idx, f_b, jdim, :, :, :]),  # n_j
                sdetweiv[dummy_idx, f_b, :, :, :]),
                -1) \
            * cijkl[idim, jdim, kdim, ldim] * (-1.0)
    # mu * u_i v_i
    for idim in range(ndim):
        S[..., idim, :, idim] += torch.mul(
            torch.sum(torch.mul(torch.mul(
                sni[dummy_idx, f_b, :, :, :],
                snj[dummy_idx, f_b, :, :, :]),
                sdetweiv[dummy_idx, f_b, :, :, :]),
                -1), mu_e)
    # 0.5 C_ijkl epsilon(u)_kl v_i n_j
    for [idim, jdim, kdim, ldim] in config.ijkldim_nz:
        S[..., idim, :, kdim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                sni[dummy_idx, f_b, :, :, :],  # v_i
                0.5 * snxj[dummy_idx, f_b, ldim, :, :, :]),  # 0.5*du_k/dx_l of epsilon_kl
                snormalv[dummy_idx, f_b, jdim, :, :, :]),  # n_j
                sdetweiv[dummy_idx, f_b, :, :, :]),
                -1) \
            * cijkl[idim, jdim, kdim, ldim] * (-1.0)
        S[..., idim, :, ldim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                sni[dummy_idx, f_b, :, :, :],  # v_i
                0.5 * snxj[dummy_idx, f_b, kdim, :, :, :]),  # 0.5*du_l/dx_k of epsilon_kl
                snormalv[dummy_idx, f_b, jdim, :, :, :]),  # n_j
                sdetweiv[dummy_idx, f_b, :, :, :]),
                -1) \
            * cijkl[idim, jdim, kdim, ldim] * (-1.0)
    # get S on p order grid
    S = torch.einsum('pi,...ijkl,kq->...pjql',
                     mg_le.p_restrictor(3, po),
                     S,
                     mg_le.p_prolongator(po, 3)).contiguous()
    # multiply S and u_i and add to (subtract from) r
    r[E_F_b, :, :] -= torch.einsum('...ijkl,...kl->...ij', S, u_i[E_F_b, :, :])
    diagS[E_F_b - batch_start_idx, :, :] += torch.diagonal(
        S.view(batch_in, mg_le.p_nloc(po)*ndim, mg_le.p_nloc(po)*ndim),
        dim1=-2, dim2=-1).view(batch_in, mg_le.p_nloc(po), ndim)
    diagS20[E_F_b - batch_start_idx, ...] += S

    return r, diagS, diagS20
