#!/usr/bin/env python3
'''
This file implements surface integral term
in finest grid and P0DG grid matrix-free-ly.
Hence the name "surface_integral_mf".
'''
import torch 
import config
import mesh_init
from config import sf_nd_nb
import numpy as np

import shape_function

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

eta_e=config.eta_e # penalty coefficient

## main function
def S_mf(r, sn, snlx, x_ref_in, sweight,
         nbele, nbf, c_bc, c_i,
         diagA, bdiagA):
    '''
    This function compute the Sc contribution to residual:
        r <- r - S*c
    where r is residual on finest grid,
        S is surface integral term matrix,
        c is field variable.

    # Input
    r : torch tensor (nonods)
        residual vector that hasn't taken account of S*c
    sn : numpy array (nface, nloc, sngi)
        shape function at face quad pnts
    # snx : torch tensor (nele, nface, ndim, nloc, sngi)
    #     shape func derivatives at face quad pnts
    # sdetwei : torch tensor (nele, nface, sngi)
    #     det x quad weights for surface integral
    # snormal : torch tensor (nele, nface, ndim)
    #     unit out normal of face
    snlx :
    x_ref_in :
    sweight :
    nbele : python list (nele x nface)
        list of neighbour element index
    nbf : python list (nele x nloc)
        list of neighbour face index
    c_bc : torch tensor (nonods)
        Dirichlet boundary values at boundary nodes, 0 otherwise
    c_i : torch tensor (nonods)
        field value at i-th iteration (last iteration)

    # Output
    r : torch tensor (nonods)
        residual vector that has taken account of S*c
    diagS : torch tensor (nonods)
        diagonal of surface integral matrix
    ~~b_bc : torch tensor (nonods)~~
        ~~rhs vector accounting for Dirichlet boundary conditions~~
    '''

    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0,nnn+1)/nnn*nele, dtype=int)
    diagA = diagA.view(nele, nloc)
    bdiagA = bdiagA.view(nele, nloc, nloc)
    for i in range(nnn):
        idx_in_f = np.zeros(nele*nface, dtype=bool)
        idx_in_f[brk_pnt[i]*3:brk_pnt[i+1]*3] = True
        r, diagA, bdiagA = S_mf_one_batch(r, c_i, c_bc,
                                          sn, snlx, x_ref_in, sweight,
                                          nbele, nbf,
                                          diagA, bdiagA,
                                          idx_in_f)
    r = r.view(-1).contiguous()
    diagA = diagA.view(-1).contiguous()
    bdiagA = bdiagA.contiguous()

    return r, diagA, bdiagA


def S_mf_one_batch(r, c_i, c_bc,
                   diagA, bdiagA,
                   idx_in_f, batch_start_idx):

    # get essential data
    nbf = sf_nd_nb.nbf
    alnmt = sf_nd_nb.alnmt
    c_i = c_i.view(nele, nloc)
    r = r.view(nele, nloc)
    c_bc = c_bc.view(nele, nloc)

    # first lets separate nbf to get two list of F_i and F_b
    F_i = np.where(alnmt >= 0 & idx_in_f)[0]  # interior face
    F_b = np.where(alnmt < 0 & idx_in_f)[0]  # boundary face
    F_inb = nbf[F_i]  # neighbour list of interior face
    F_inb = F_inb.astype(np.int64)

    # create two lists of which element f_i / f_b is in
    E_F_i = np.floor_divide(F_i, nface)
    E_F_b = np.floor_divide(F_b, nface)
    E_F_inb = np.floor_divide(F_inb, nface)

    # local face number
    f_i = np.mod(F_i, nface)
    f_b = np.mod(F_b, nface)
    f_inb = np.mod(F_inb, nface)

    # diagS = torch.zeros(nonods, device=dev, dtype=torch.float64)
    # diagS = diagS.view(nele, nloc)
    #
    # bdiagS = torch.zeros(nele, nloc, nloc, device=dev, dtype=torch.float64)

    # for interior faces, update r
    # r <- r-S*c
    # use r+= or r-= to make sure in-place assignment to avoid copy
    # update 3 local faces separately to avoid change r with multiple values
    # idx_iface_all = np.zeros(F_i.shape[0], dtype=bool)
    for iface in range(nface):
        for nb_gi_aln in range(nface-1):
            idx_iface = (f_i == iface) & (sf_nd_nb.alnmt[F_i] == nb_gi_aln)
            # idx_iface_all += idx_iface
            # idx_iface = idx_iface & idx_in_f
            r, diagA, bdiagA = S_fi(
                r, f_i[idx_iface], E_F_i[idx_iface], F_i[idx_iface],
                f_inb[idx_iface], E_F_inb[idx_iface], F_inb[idx_iface],
                c_i, diagA, bdiagA, batch_start_idx,
                nb_gi_aln)

        # # if split and compute separately (to save memory)
        # idx_iface = f_i == iface
        # # break the whole idx_iface list into nnn parts
        # # and do S_fi for each part
        # # so that we have a smaller "batch" size to solve memory issue
        # nnn = config.no_batch
        # brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * idx_iface.shape[0], dtype=int)
        # # [0,
        # #        int(idx_iface.shape[0] / 4.),
        # #        int(idx_iface.shape[0] / 4. * 2.),
        # #        int(idx_iface.shape[0] / 4. * 3.),
        # #        idx_iface.shape[0]]
        # for i in range(nnn):
        #     idx_list = np.zeros_like(idx_iface, dtype=bool)
        #     idx_list[brk_pnt[i]:brk_pnt[i + 1]] += idx_iface[brk_pnt[i]:brk_pnt[i + 1]]
        #     r, diagS, bdiagS = S_fi(
        #         r, f_i[idx_list], E_F_i[idx_list], F_i[idx_list],
        #         f_inb[idx_list], E_F_inb[idx_list], F_inb[idx_list],
        #         sn, snlx, x_ref_in, sweight, c_i, diagS, bdiagS)

    # for boundary faces, update r
    # r <= r + b_bc - S*c
    # let's hope that each element has only one boundary face.
    r, diagA, bdiagA = S_fb(r, f_b, E_F_b, F_b,
                            c_i, c_bc, diagA, bdiagA, batch_start_idx)

    return r, diagA, bdiagA


def S_fi(r, f_i, E_F_i, F_i, 
         f_inb, E_F_inb, F_inb, 
         c_i,
         diagS, bdiagS, batch_start_idx,
         nb_gi_aln):  # neighbour face gaussian points aliagnment
    '''
    this function add interior face S*c contribution
    to r
    '''
    # get essential data
    sn = sf_nd_nb.sn
    snlx = sf_nd_nb.snlx
    x_ref_in = sf_nd_nb.x_ref_in
    sweight = sf_nd_nb.sweight

    # faces can be passed in by batches to fit memory/GPU cores
    batch_in = f_i.shape[0]
    dummy_idx = np.arange(0,batch_in)  # this is to use with idx f_i

    # make all tensors in shape (nele, nface, ndim, nloc(inod), nloc(jnod), sngi)
    # all these expansion are views of original tensor,
    # i.e. they point to same memory as sn/snx
    sni = sn.unsqueeze(0).expand(batch_in,-1,-1,-1)\
        .unsqueeze(3).expand(-1,-1,-1,nloc,-1) # expand on nloc(jnod)
    snj = sn.unsqueeze(0).expand(batch_in,-1,-1,-1)\
        .unsqueeze(2).expand(-1,-1,nloc,-1,-1) # expand on nloc(inod)

    # get shape function derivatives
    # this side.
    snx, sdetwei, snormal = shape_function.sdet_snlx_3d(snlx, x_ref_in[E_F_i], sweight)
    # now tensor shape are:
    # snx | snx_nb         (batch_in, nface, ndim, nloc, sngi)
    # sdetwei | sdetwei_nb (batch_in, nface, sngi)
    # snormal | snormal_nb (batch_in, nface, ndim)
    mu_e = eta_e/torch.sum(sdetwei[dummy_idx, f_i,:],-1)
    snxi = snx.unsqueeze(4)\
        .expand(-1,-1,-1,-1,nloc,-1) # expand on nloc(jnod)
    snxj = snx.unsqueeze(3)\
        .expand(-1,-1,-1,nloc,-1,-1) # expand on nloc(inod)
    # make all tensors in shape (nele, nface, nloc(inod), nloc(jnod), sngi)
    # are views of snormal/sdetwei, taken same memory
    snormalv = snormal.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)\
        .expand(-1,-1,-1,nloc,nloc,sngi)
    sdetweiv = sdetwei.unsqueeze(2).unsqueeze(3)\
        .expand(-1,-1,nloc,nloc,-1)
    # other side.
    snx_nb, _, snormal_nb = shape_function.sdet_snlx_3d(snlx, x_ref_in[E_F_inb], sweight)
    # change gausian pnts alignment on the other side use nb_gi_aln
    nb_aln = sf_nd_nb.gi_align[nb_gi_aln, :]
    snx_nb = snx_nb[..., nb_aln]
    snj_nb = snj[..., nb_aln]
    # snxi_nb = snx_nb.unsqueeze(4) \
    #     .expand(-1, -1, -1, -1, nloc, -1)  # expand on nloc(jnod)
    snxj_nb = snx_nb.unsqueeze(3) \
        .expand(-1, -1, -1, nloc, -1, -1)  # expand on nloc(inod)
    # make all tensors in shape (nele, nface, nloc(inod), nloc(jnod), sngi)
    # are views of snormal/sdetwei, taken same memory
    snormalv_nb = snormal_nb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) \
        .expand(-1, -1, -1, nloc, nloc, sngi)
    # sdetweiv_nb = sdetwei_nb.unsqueeze(2).unsqueeze(3) \
    #     .expand(-1, -1, nloc, nloc, -1)
    # print('sn shape sn memory usage',snj.shape, snj.storage().size())
    # print('snx shape snx memory usage',snxi.shape, snxi.storage().size())
    # print('snormal shape snormal memory usage',snormal.shape, snormal.storage().size())
    # print('sdetwei shape sdetwei memory usage',sdetwei.shape, sdetwei.storage().size())

    # make mu_e in shape (batch_in, nloc, nloc)
    mu_e = mu_e.unsqueeze(-1).unsqueeze(-1)\
        .expand(-1,nloc,nloc)
    
    # this side
    S = torch.zeros(batch_in, nloc, nloc,
        device=dev, dtype=torch.float64) # local S matrix
    # n_j nx_i
    for idim in range(ndim):
        S[:,:nloc,:nloc] += torch.sum(torch.mul(torch.mul(torch.mul(
            snj[dummy_idx, f_i, :, :, :],
            snxi[dummy_idx, f_i, idim, :,:,:]),
            snormalv[dummy_idx,f_i, idim, :,:,:]),
            sdetweiv[dummy_idx,f_i, :,:,:]),
            -1)*(-0.5)
    # njx ni
    for idim in range(ndim):
        S[:,:nloc,:nloc] += torch.sum(torch.mul(torch.mul(torch.mul(
            snxj[dummy_idx, f_i,idim,:,:,:],
            sni[dummy_idx, f_i, :,:,:]),
            snormalv[dummy_idx,f_i, idim, :,:,:]),
            sdetweiv[dummy_idx, f_i, :,:,:]),
            -1)*(-0.5)
    # nj ni
    S[:,:nloc,:nloc] += torch.mul(
        torch.sum(torch.mul(torch.mul(
        sni[dummy_idx, f_i, :,:,:],
        snj[dummy_idx, f_i, :,:,:]),
        sdetweiv[dummy_idx, f_i, :,:,:]),
        -1)   ,   mu_e)
    # multiply S and c_i and add to (subtract from) r 
    r[E_F_i,:] -= torch.matmul(S, c_i[E_F_i,:].view(batch_in,nloc,1)).squeeze()
    # put diagonal of S into diagS
    diagS[E_F_i-batch_start_idx,:] += torch.diagonal(S,dim1=-2,dim2=-1)
    bdiagS[E_F_i-batch_start_idx, ...] += S

    # other side
    S = torch.zeros(batch_in, nloc, nloc,
        device=dev, dtype=torch.float64) # local S matrix
    # Nj2 * Ni1x * n2
    for idim in range(ndim):
        S[:,:nloc,:nloc] += torch.sum(torch.mul(torch.mul(torch.mul(
            snj_nb[dummy_idx, f_inb, :, :, :],
            snxi[dummy_idx, f_i, idim, :,:,:]),
            snormalv_nb[dummy_idx,f_inb, idim, :,:,:]),
            sdetweiv[dummy_idx,f_i, :,:,:]),
            -1)*(-0.5)
    # Nj2x * Ni1 * n1
    for idim in range(ndim):
        S[:,:nloc,:nloc] += torch.sum(torch.mul(torch.mul(torch.mul(
            snxj_nb[dummy_idx, f_inb,idim,:,:,:],
            sni[dummy_idx, f_i, :,:,:]),
            snormalv[dummy_idx,f_i, idim, :,:,:]),
            sdetweiv[dummy_idx, f_i, :,:,:]),
            -1)*(-0.5)
    # Nj2n2 * Ni1n1 ! n2 \cdot n1 = -1
    S[:,:nloc,:nloc] += torch.mul(
        torch.sum(torch.mul(torch.mul(
        sni[dummy_idx, f_i, :,:,:],
        snj_nb[dummy_idx, f_inb, :,:,:]),
        sdetweiv[dummy_idx, f_i, :,:,:]),
        -1)   ,   -mu_e)
    # this S is off-diagonal contribution, therefore no need to put in diagS
    # multiply S and c_i and add to (subtract from) r 
    r[E_F_i,:] -= torch.matmul(S, c_i[E_F_inb,:].view(batch_in,nloc,1)).squeeze()

    return r, diagS, bdiagS


def S_fb(r, f_b, E_F_b, F_b,
    c_i, c_bc,
    diagS, bdiagS, batch_start_idx):
    '''
    This function add boundary face S*c_i contribution
    to residual 
    r <- r - S*c_i
    and 
    S*c_bc contribution to rhs, and then, also 
    residual
    r ,_ r + S*c_bc
    '''

    # get essential data
    sn = sf_nd_nb.sn
    snlx = sf_nd_nb.snlx
    x_ref_in = sf_nd_nb.x_ref_in
    sweight = sf_nd_nb.sweight

    # faces can be passed in by batches to fit memory/GPU cores
    batch_in = f_b.shape[0]
    dummy_idx = np.arange(0, batch_in)

    # make all tensors in shape (nele, nface, ndim, nloc(inod), nloc(jnod), sngi)
    # all these expansion are views of original tensor,
    # i.e. they point to same memory as sn/snx
    sni = sn.unsqueeze(0).expand(batch_in,-1,-1,-1)\
        .unsqueeze(3).expand(-1,-1,-1,nloc,-1) # expand on nloc(jnod)
    snj = sn.unsqueeze(0).expand(batch_in,-1,-1,-1)\
        .unsqueeze(2).expand(-1,-1,nloc,-1,-1) # expand on nloc(inod)

    # get shaps function derivatives
    snx, sdetwei, snormal = shape_function.sdet_snlx_3d(snlx, x_ref_in[E_F_b], sweight)
    mu_e = eta_e/torch.sum(sdetwei[dummy_idx, f_b,:],-1)
    snxi = snx.unsqueeze(4)\
        .expand(-1,-1,-1,-1,nloc,-1) # expand on nloc(jnod)
    snxj = snx.unsqueeze(3)\
        .expand(-1,-1,-1,nloc,-1,-1) # expand on nloc(inod)
    # make all tensors in shape (nele, nface, nloc(inod), nloc(jnod), sngi)
    # are views of snormal/sdetwei, taken same memory
    snormalv = snormal.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)\
        .expand(-1,-1,-1,nloc,nloc,sngi)
    sdetweiv = sdetwei.unsqueeze(2).unsqueeze(3)\
        .expand(-1,-1,nloc,nloc,-1)
    # make mu_e in shape (batch_in, nloc, nloc)
    mu_e = mu_e.unsqueeze(-1).unsqueeze(-1)\
        .expand(-1,nloc,nloc)
    
    S = torch.zeros(batch_in, nloc, nloc,
        device=dev, dtype=torch.float64) # local S matrix
    # # Nj1 * Ni1x * n1
    for idim in range(ndim):
        S[:,:nloc,:nloc] -= torch.sum(torch.mul(torch.mul(torch.mul(
            snj[dummy_idx, f_b, :, :, :],
            snxi[dummy_idx, f_b, idim, :,:,:]),
            snormalv[dummy_idx,f_b, idim, :,:,:]),
            sdetweiv[dummy_idx,f_b, :,:,:]),
            -1) # *(-1.0)
    # # Nj1n1 * Ni1n1 ! n1 \cdot n1 = 1
    S[:,:nloc,:nloc] += torch.mul(
        torch.sum(torch.mul(torch.mul(
        sni[dummy_idx, f_b, :,:,:],
        snj[dummy_idx, f_b, :,:,:]),
        sdetweiv[dummy_idx, f_b, :,:,:]),
        -1)   ,   mu_e)
    # calculate b_bc and add to r
    # r <- r + b_bc
    r[E_F_b, :] += torch.matmul(S, c_bc[E_F_b,:].view(batch_in,nloc,1)).squeeze()
    # diagS[E_F_b,:] += torch.diagonal(S,dim1=-2,dim2=-1)
    # # Nj1x * Ni1 * n1
    for idim in range(ndim):
        S[:,:nloc,:nloc] -= torch.sum(torch.mul(torch.mul(torch.mul(
            sni[dummy_idx, f_b, :, :, :],
            snxj[dummy_idx, f_b, idim, :,:,:]),
            snormalv[dummy_idx,f_b, idim, :,:,:]),
            sdetweiv[dummy_idx,f_b, :,:,:]),
            -1) # *(-1.0)
    # calculate S*c and add to (subtract from) r
    r[E_F_b, :] -= torch.matmul(S, c_i[E_F_b,:].view(batch_in,nloc,1)).squeeze()
    diagS[E_F_b-batch_start_idx,:] += torch.diagonal(S,dim1=-2,dim2=-1)
    bdiagS[E_F_b-batch_start_idx, ...] += S
    
    return r, diagS, bdiagS
