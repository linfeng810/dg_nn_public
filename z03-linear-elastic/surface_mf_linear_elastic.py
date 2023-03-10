#!/usr/bin/env python3
'''
This file is built upon "surface_integral_mf".

This file implements surface integral term
for the linear elastic problem
in finest grid and P0DG grid matrix-free-ly.
Hence the name "surface_mf_linear_elastic".
'''
import torch 
import config 
import numpy as np
from opt_einsum import contract
import cProfile, pstats, io
from pstats import SortKey

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

eta_e = config.eta_e

## main function
# @profile
def S_mf(r, sn, snx, sdetwei, snormal, 
         nbele, nbf, u_bc, u_i):
    '''
    This function compute the Sc contribution to residual:
        r <- r - S*c + b_bc
    where r is residual on finest grid,
        S is surface integral term matrix,
        c is field variable.

    # Input
    r : torch tensor (nonods, ndim)
        residual vector that hasn't taken account of S*c
    sn : numpy array (nface, nloc, sngi)
        shape function at face quad pnts
    snx : torch tensor (nele, nface, ndim, nloc, sngi)
        shape func derivatives at face quad pnts
    sdetwei : torch tensor (nele, nface, sngi)
        det x quad weights for surface integral
    snormal : torch tensor (nele, nface, ndim)
        unit out normal of face
    nbele : python list (nele x nface)
        list of neighbour element index
    nbf : python list (nele x nloc)
        list of neighbour face index
    u_bc : torch tensor (nonods, ndim)
        Dirichlet boundary values at boundary nodes, 0 otherwise
    u_i : torch tensor (nonods, ndim)
        field value at i-th iteration (last iteration)

    # Output
    r : torch tensor (nonods, ndim)
        residual vector that has taken account of S*u
    diagS : torch tensor (nonods, ndim)
        diagonal of surface integral matrix
    '''
    
    u_i = u_i.view(nele, nloc, ndim)
    r = r.view(nele, nloc, ndim)
    u_bc = u_bc.view(nele,nloc, ndim)

    # output declaration
    diagS = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)
    diagS20 = torch.zeros(nele, nloc, ndim, nloc, ndim, device=dev, dtype=torch.float64)

    # first lets separate nbf to get two list of F_i and F_b
    F_i = np.where(np.logical_not(np.isnan(nbf)))[0] # interior face
    F_b = np.where(np.isnan(nbf))[0]   # boundary face
    F_inb = -nbf[F_i] # neighbour list of interior face
    F_inb = F_inb.astype(np.int64)

    # create two lists of which element f_i / f_b is in
    E_F_i = np.floor_divide(F_i,3)
    E_F_b = np.floor_divide(F_b,3)
    E_F_inb = np.floor_divide(F_inb,3)

    # local face number
    f_i = np.mod(F_i,3)
    f_b = np.mod(F_b,3)
    f_inb = np.mod(F_inb,3)

    diagS = diagS.view(nele, nloc, ndim)
    # f_i_iface = []
    # E_F_i_iface = []
    #
    # for iface in range(nface):
    #     f_i_iface.append(f_i[f_i == iface])
    #     _ = E_F_i[f_i == iface]
    #     _ = F_i[f_i == iface]
    #     _ = f_inb[f_i == iface]
    #     _ = E_F_inb[f_i == iface]
    #     _ = F_inb[f_i == iface]

    # for interior faces, update r
    # r <- r-S*c
    # use r+= or r-= to make sure in-place assignment to avoid copy
    # update 3 local faces separately to avoid change r with multiple values
    for iface in range(nface):
        # pr = cProfile.Profile()
        # pr.enable()
        # print('iface=',iface)
        indices = f_i==iface
        r, diagS, diagS20 = S_fi(r, f_i[indices], E_F_i[indices], F_i[indices],
                f_inb[indices], E_F_inb[indices], F_inb[indices],
                sn, snx, snormal, sdetwei, u_i, diagS, diagS20)
        # pr.disable()
        # s = io.StringIO()
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())

    # for boundary faces, update r
    # r <= r + b_bc - S*c
    # let's hope that each element has only one boundary face.
    r, diagS, diagS20 = S_fb(r, f_b, E_F_b, F_b,
             sn, snx, snormal, sdetwei, u_i, u_bc, diagS, diagS20)

    r = r.view(nonods, ndim).contiguous()
    diagS = diagS.view(nonods, ndim).contiguous()
    diagS20 = diagS20.view(nele, nloc*ndim, nloc*ndim).contiguous()
    return r, diagS, diagS20

# @profile
def S_fi(r, f_i, E_F_i, F_i, 
         f_inb, E_F_inb, F_inb, 
         sn, snx, snormal, sdetwei, u_i,
         diagS, diagS20):
    '''
    this function add interior face S*c contribution
    to r
    '''

    # faces can be passed in by batches to fit memory/GPU cores
    batch_in = f_i.shape[0]
    mu_e = eta_e/torch.sum(sdetwei[E_F_i, f_i, :], -1)  # mu_e for each face (batch_in)

    # make all tensors in shape (nele, nface, ndim, nloc(inod), nloc(jnod), sngi)
    # all these expansion are views of original tensor, 
    # i.e. they point to same memory as sn/snx
    sni = sn.unsqueeze(0).expand(nele,-1,-1,-1)\
        .unsqueeze(3).expand(-1,-1,-1,nloc,-1) # expand on nloc(jnod)
    snj = sn.unsqueeze(0).expand(nele,-1,-1,-1)\
        .unsqueeze(2).expand(-1,-1,nloc,-1,-1) # expand on nloc(inod)
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
    mu_ev = mu_e.unsqueeze(-1).unsqueeze(-1).expand(-1,nloc,nloc)
    
    # this side
    S = torch.zeros(batch_in, nloc, ndim, nloc, ndim,
                    device=dev, dtype=torch.float64)  # local S matrix

    # prefetch E_F_i / f_i indices
    sn_fi = sn[f_i, ...]
    snx_fi = snx[E_F_i, f_i, :, ...]
    sn_finb = torch.flip(sn[f_inb, ...], [-1])  # flip gaussian pnts on other side
    snx_finb = torch.flip(snx[E_F_inb, f_inb, ...], [-1])  # flip gaussian pnts on ...
    snormal_fi = snormal[E_F_i, f_i, :]
    sdetwei_fi = sdetwei[E_F_i, f_i, :]
    # for idim in range(ndim):
    #     for jdim in range(ndim):
    #         for kdim in range(ndim):
    #             for ldim in range(ndim):
    for [idim,jdim,kdim,ldim] in config.ijkldim_nz :
        # 0.5 C_ijkl epsilon(u)_kl v_i n_j
        S[..., idim, :, kdim] += \
            torch.einsum('...ig,...jg,...,...g->...ij',
                     sn_fi,
                     snx_fi[:,ldim,...],
                     snormal_fi[:,jdim],
                     sdetwei_fi)\
            * cijkl[idim,jdim,kdim,ldim]*(-0.5) * 0.5
            # torch.sum(torch.mul(torch.mul(torch.mul(
            #     sni[E_F_i, f_i, :, :, :], # v_i
            #     0.5*snxj[E_F_i, f_i, ldim, :,:,:] ), # 0.5*du_k/dx_l of epsilon_kl
            #     snormalv[E_F_i,f_i, jdim, :,:,:]), # n_j
            #     sdetweiv[E_F_i,f_i, :,:,:]),
            #     -1)\
        S[..., idim, :, ldim] += \
            torch.einsum('...ig,...jg,...,...g->...ij',
                     sn_fi,
                     snx_fi[:,kdim,...],
                     snormal_fi[:,jdim],
                     sdetwei_fi)\
            * cijkl[idim,jdim,kdim,ldim]*(-0.5) * 0.5
            # torch.sum(torch.mul(torch.mul(torch.mul(
            #     sni[E_F_i, f_i, :, :, :], # v_i
            #     0.5*snxj[E_F_i, f_i, kdim, :,:,:] ), # 0.5*du_l/dx_k of epsilon_kl
            #     snormalv[E_F_i,f_i, jdim, :,:,:]), # n_j
            #     sdetweiv[E_F_i,f_i, :,:,:]),
            #     -1)\
        # u_i n_j * 0.5 * C_ijkl epsilon(v)_kl
        S[..., kdim, :, idim] += \
            torch.einsum('...jg,...ig,...,...g->...ij',  # g is gaussian point, i, j are iloc/jloc
                     sn_fi,
                     snx_fi[:,ldim,...],
                     snormal_fi[:,jdim],
                     sdetwei_fi) \
            * cijkl[idim, jdim, kdim, ldim] * (-0.5) * 0.5
            # torch.sum(torch.mul(torch.mul(torch.mul(
            #     snj[E_F_i, f_i, :, :, :], # u_i
            #     0.5*snxi[E_F_i, f_i, ldim, :,:,:] ), # 0.5*dv_k/dx_l of epsilon_kl
            #     snormalv[E_F_i,f_i, jdim, :,:,:]), # n_j
            #     sdetweiv[E_F_i,f_i, :,:,:]),
            #     -1)\
            # *cijkl[idim,jdim,kdim,ldim]*(-0.5)
        S[..., ldim, :, idim] += \
            torch.einsum('...jg,...ig,...,...g->...ij',  # g is gaussian point, i, j are iloc/jloc
                     sn_fi,
                     snx_fi[:,kdim,...],
                     snormal_fi[:,jdim],
                     sdetwei_fi) \
            * cijkl[idim, jdim, kdim, ldim] * (-0.5) * 0.5
            # torch.sum(torch.mul(torch.mul(torch.mul(
            #     snj[E_F_i, f_i, :, :, :], # u_i
            #     0.5*snxi[E_F_i, f_i, kdim, :,:,:] ), # 0.5*dv_l/dx_k of epsilon_kl
            #     snormalv[E_F_i,f_i, jdim, :,:,:]), # n_j
            #     sdetweiv[E_F_i,f_i, :,:,:]),
            #     -1)\
            # *cijkl[idim,jdim,kdim,ldim]*(-0.5)
    # mu * u_i v_i
    for idim in range(ndim):
        S[..., idim, :, idim] += \
            torch.einsum('...ig,...jg,...g,...->...ij',
                     sn_fi,
                     sn_fi,
                     sdetwei_fi,
                     mu_e)
            # torch.mul(
            # torch.sum(torch.mul(torch.mul(
            # sni[E_F_i, f_i, :,:,:],
            # snj[E_F_i, f_i, :,:,:]),
            # sdetweiv[E_F_i, f_i, :,:,:]),
            # -1)   ,   mu_ev)
    # multiply S and c_i and add to (subtract from) r 
    # r[E_F_i, :, :] -= torch.einsum('...ijkl,...kl->...ij', S, u_i[E_F_i, :, :])
    r[E_F_i, :, :] -= torch.bmm(S.view(batch_in, nloc*ndim, nloc*ndim),
                                u_i[E_F_i, ...].view(batch_in, nloc*ndim, 1)).view(batch_in, nloc, ndim)
    # put diagonal of S into diagS
    diagS[E_F_i, :, :] += torch.diagonal(torch.diagonal(S, dim1=1, dim2=3), dim1=1, dim2=2)
    diagS20[E_F_i, ...] += S

    # other side
    # S *= 0.  # local S matrix
    S = torch.zeros(batch_in, nloc, nloc, ndim, ndim,
                    device=dev, dtype=torch.float64)  # local S matrix
    for [idim,jdim,kdim,ldim] in config.ijkldim_nz :
        # 0.5 C_ijkl epsilon(u)_kl v_i n_j
        S[..., idim, kdim] += \
            torch.einsum('...ig,...jg,...,...g->...ij',
                     sn_fi,
                     snx_finb[:,ldim,...],
                     snormal_fi[:,jdim],
                     sdetwei_fi)\
            * cijkl[idim,jdim,kdim,ldim]* (-0.5) * 0.5
            # torch.sum(torch.mul(torch.mul(torch.mul(
            #     sni[E_F_i, f_i, :, :, :], # v_i
            #     0.5*torch.flip(snxj[E_F_inb, f_inb, ldim, :,:,:],[-1]) ), # 0.5*du_k/dx_l of epsilon_kl
            #     snormalv[E_F_i,f_i, jdim, :,:,:]), # n_j
            #     sdetweiv[E_F_i,f_i, :,:,:]),
            #     -1)\
            # *cijkl[idim,jdim,kdim,ldim]*(-0.5)
        S[..., idim, ldim] += \
            torch.einsum('...ig,...jg,...,...g->...ij',
                     sn_fi,
                     snx_finb[:, kdim, ...],
                     snormal_fi[:, jdim],
                     sdetwei_fi) \
            * cijkl[idim, jdim, kdim, ldim] * (-0.5) * 0.5
            # torch.sum(torch.mul(torch.mul(torch.mul(
            #     sni[E_F_i, f_i, :, :, :], # v_i
            #     0.5*torch.flip(snxj[E_F_inb, f_inb, kdim, :,:,:],[-1]) ), # 0.5*du_l/dx_k of epsilon_kl
            #     snormalv[E_F_i,f_i, jdim, :,:,:]), # n_j
            #     sdetweiv[E_F_i,f_i, :,:,:]),
            #     -1)\
            # *cijkl[idim,jdim,kdim,ldim]*(-0.5)
        # u_i n_j * 0.5 * C_ijkl epsilon(v)_kl
        S[..., kdim, idim] += \
            torch.einsum('...jg,...ig,...,...g->...ij',
                     sn_finb,
                     snx_fi[:, ldim, ...],
                     snormal_fi[:, jdim],
                     sdetwei_fi) \
            * cijkl[idim, jdim, kdim, ldim] * (-0.5) * 0.5
            # torch.sum(torch.mul(torch.mul(torch.mul(
            #     torch.flip(snj[E_F_inb, f_inb, :, :, :],[-1]), # u_i
            #     0.5*snxi[E_F_i, f_i, ldim, :,:,:] ), # 0.5*dv_k/dx_l of epsilon_kl
            #     snormalv[E_F_inb,f_inb, jdim, :,:,:]), # n_j
            #     sdetweiv[E_F_i,f_i, :,:,:]),
            #     -1)\
            # *cijkl[idim,jdim,kdim,ldim]*(-0.5)
        S[..., ldim, idim] += \
            torch.einsum('...jg,...ig,...,...g->...ij',
                     sn_finb,
                     snx_fi[:, kdim, ...],
                     snormal_fi[:, jdim],
                     sdetwei_fi) \
            * cijkl[idim, jdim, kdim, ldim] * (-0.5) * 0.5
            # torch.sum(torch.mul(torch.mul(torch.mul(
            #     torch.flip(snj[E_F_inb, f_inb, :, :, :],[-1]), # u_i
            #     0.5*snxi[E_F_i, f_i, kdim, :,:,:] ), # 0.5*dv_l/dx_k of epsilon_kl
            #     snormalv[E_F_inb,f_inb, jdim, :,:,:]), # n_j
            #     sdetweiv[E_F_i,f_i, :,:,:]),
            #     -1)\
            # *cijkl[idim,jdim,kdim,ldim]*(-0.5)
    # mu * u_i v_i
    for idim in range(ndim):
        S[..., idim, idim] += \
            torch.einsum('...ig,...jg,...g,...->...ij',
                     sn_fi,
                     sn_finb,
                     sdetwei_fi,
                     mu_e) * (-1.0)
            # torch.mul(
            # torch.sum(torch.mul(torch.mul(
            # sni[E_F_i, f_i, :,:,:],
            # torch.flip(snj[E_F_inb, f_inb, :,:,:],[-1])),
            # sdetweiv[E_F_i, f_i, :,:,:]),
            # -1), -mu_ev)
    # this S is off-diagonal contribution, therefore no need to put in diagS
    # multiply S and c_i and add to (subtract from) r 
    r[E_F_i, :, :] -= torch.einsum('...ijkl,...jl->...ik', S, u_i[E_F_inb, :, :])

    return r, diagS, diagS20

def S_fb(r, f_b, E_F_b, F_b,
    sn, snx, snormal, sdetwei, u_i, u_bc,
    diagS, diagS20):
    '''
    This function add boundary face S*c_i contribution
    to residual 
    r <- r - S*u_i
    and 
    S*c_bc contribution to rhs, and then, also 
    residual
    r <- r + S*u_bc
    '''

    # faces can be passed in by batches to fit memory/GPU cores
    batch_in = f_b.shape[0]
    mu_e = eta_e/torch.sum(sdetwei[E_F_b, f_b,:],-1)

    # make all tensors in shape (nele, nface, ndim, nloc(inod), nloc(jnod), sngi)
    # all these expansion are views of original tensor, 
    # i.e. they point to same memory as sn/snx
    sni = sn.unsqueeze(0).expand(nele,-1,-1,-1)\
        .unsqueeze(3).expand(-1,-1,-1,nloc,-1) # expand on nloc(jnod)
    snj = sn.unsqueeze(0).expand(nele,-1,-1,-1)\
        .unsqueeze(2).expand(-1,-1,nloc,-1,-1) # expand on nloc(inod)
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
    
    # only one side
    S = torch.zeros(batch_in, nloc, nloc, ndim, ndim,
                    device=dev, dtype=torch.float64)  # local S matrix
    # u_i n_j * 0.5 * C_ijkl epsilon(v)_kl
    # for idim in range(ndim):
    #     for jdim in range(ndim):
    #         for kdim in range(ndim):
    #             for ldim in range(ndim):
    for [idim, jdim, kdim, ldim] in config.ijkldim_nz:
        S[..., kdim, idim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                snj[E_F_b, f_b, :, :, :], # u_i
                0.5*snxi[E_F_b, f_b, ldim, :,:,:] ), # 0.5*dv_k/dx_l of epsilon_kl
                snormalv[E_F_b,f_b, jdim, :,:,:]), # n_j
                sdetweiv[E_F_b,f_b, :,:,:]),
                -1)\
            *cijkl[idim,jdim,kdim,ldim]*(-1.0)
        S[..., ldim, idim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                snj[E_F_b, f_b, :, :, :], # u_i
                0.5*snxi[E_F_b, f_b, kdim, :,:,:] ), # 0.5*dv_l/dx_k of epsilon_kl
                snormalv[E_F_b,f_b, jdim, :,:,:]), # n_j
                sdetweiv[E_F_b,f_b, :,:,:]),
                -1)\
            *cijkl[idim,jdim,kdim,ldim]*(-1.0)
    # mu * u_i v_i
    for idim in range(ndim):
        S[..., idim, idim] += torch.mul(
            torch.sum(torch.mul(torch.mul(
            sni[E_F_b, f_b, :,:,:],
            snj[E_F_b, f_b, :,:,:]),
            sdetweiv[E_F_b, f_b, :,:,:]),
            -1)   ,   mu_e)
    # calculate b_bc and add to r:
    # r <- r + b_bc
    r[E_F_b, :, :] += torch.einsum('...ijkl,...jl->...ik', S, u_bc[E_F_b, :, :])

    # 0.5 C_ijkl epsilon(u)_kl v_i n_j
    # for idim in range(ndim):
    #     for jdim in range(ndim):
    #         for kdim in range(ndim):
    #             for ldim in range(ndim):
    for [idim, jdim, kdim, ldim] in config.ijkldim_nz:
        S[..., idim, kdim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                sni[E_F_b, f_b, :, :, :], # v_i
                0.5*snxj[E_F_b, f_b, ldim, :,:,:] ), # 0.5*du_k/dx_l of epsilon_kl
                snormalv[E_F_b,f_b, jdim, :,:,:]), # n_j
                sdetweiv[E_F_b,f_b, :,:,:]),
                -1)\
            *cijkl[idim,jdim,kdim,ldim]*(-1.0)
        S[..., idim, ldim] += \
            torch.sum(torch.mul(torch.mul(torch.mul(
                sni[E_F_b, f_b, :, :, :], # v_i
                0.5*snxj[E_F_b, f_b, kdim, :,:,:] ), # 0.5*du_l/dx_k of epsilon_kl
                snormalv[E_F_b,f_b, jdim, :,:,:]), # n_j
                sdetweiv[E_F_b,f_b, :,:,:]),
                -1)\
            *cijkl[idim,jdim,kdim,ldim]*(-1.0)
    # multiply S and u_i and add to (subtract from) r
    r[E_F_b, :, :] -= torch.einsum('...ijkl,...jl->...ik', S, u_i[E_F_b, :, :])
    diagS[E_F_b, :, :] += torch.diagonal(torch.diagonal(S, dim1=1, dim2=2), dim1=1, dim2=2)
    diagS20[E_F_b, ...] += torch.permute(S, (0, 1, 3, 2, 4))
    
    return r, diagS, diagS20
