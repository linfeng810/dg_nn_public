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

eta_e=36. # penalty coefficient

## main function
def S_mf(r, sn, snx, sdetwei, snormal, 
         nbele, nbf, u_bc, u_i):
    '''
    This function compute the Sc contribution to residual:
        r <- r - S*c
    where r is residual on finest grid,
        S is surface integral term matrix,
        c is field variable.

    # Input
    r : torch tensor (ndim, nonods)
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
    u_bc : torch tensor (ndim, nonods)
        Dirichlet boundary values at boundary nodes, 0 other wise
    u_i : torch tensor (ndim, nonods)
        field value at i-th iteration (last iteration)

    # Output
    r : torch tensor (ndim, nonods)
        residual vector that has taken account of S*u
    diagS : torch tensor (ndim, nonods)
        diagonal of surface integral matrix
    '''
    
    u_i = u_i.view(ndim, nele,nloc)
    r = r.view(ndim, nele,nloc)
    u_bc = u_bc.view(ndim, nele,nloc)

    # output declaration
    diagS = torch.zeros(ndim, nonods, device = dev, dtype=torch.float64)
    

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

    diagS = torch.zeros(nonods, device=dev, dtype=torch.float64)
    diagS = diagS.view(nele,nloc)

    # for interior faces, update r
    # r <- r-S*c
    # use r+= or r-= to make sure in-place assignment to avoid copy
    # update 3 local faces separately to avoid change r with multiple values
    for iface in range(nface):
        r, diagS = S_fi(r, f_i[f_i==iface], E_F_i[f_i==iface], F_i[f_i==iface], 
                f_inb[f_i==iface], E_F_inb[f_i==iface], F_inb[f_i==iface], 
                sn, snx, snormal, sdetwei, c_i, diagS)
        
    # for boundary faces, update r
    # r <= r + b_bc - S*c
    # let's hope that each element has only one boundary face.
    r,diagS = S_fb(r, f_b, E_F_b, F_b,
             sn, snx, snormal, sdetwei, c_i, c_bc, diagS)

    r = r.view(-1).contiguous()
    diagS = diagS.view(-1).contiguous()
    return r, diagS

def S_fi(r, f_i, E_F_i, F_i, 
         f_inb, E_F_inb, F_inb, 
         sn, snx, snormal, sdetwei, c_i,
         diagS):
    '''
    this function add interior face S*c contribution
    to r
    '''

    # faces can be passed in by batches to fit memory/GPU cores
    batch_in = f_i.shape[0]
    mu_e = eta_e/torch.sum(sdetwei[E_F_i, f_i,:],-1)

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
            snj[E_F_i, f_i, :, :, :],
            snxi[E_F_i, f_i, idim, :,:,:]),
            snormalv[E_F_i,f_i, idim, :,:,:]),
            sdetweiv[E_F_i,f_i, :,:,:]),
            -1)*(-0.5)
    # njx ni
    for idim in range(ndim):
        S[:,:nloc,:nloc] += torch.sum(torch.mul(torch.mul(torch.mul(
            snxj[E_F_i, f_i,idim,:,:,:],
            sni[E_F_i, f_i, :,:,:]),
            snormalv[E_F_i,f_i, idim, :,:,:]),
            sdetweiv[E_F_i, f_i, :,:,:]),
            -1)*(-0.5)
    # nj ni
    S[:,:nloc,:nloc] += torch.mul(
        torch.sum(torch.mul(torch.mul(
        sni[E_F_i, f_i, :,:,:],
        snj[E_F_i, f_i, :,:,:]),
        sdetweiv[E_F_i, f_i, :,:,:]),
        -1)   ,   mu_e)
    # multiply S and c_i and add to (subtract from) r 
    r[E_F_i,:] -= torch.matmul(S, c_i[E_F_i,:].view(batch_in,nloc,1)).squeeze()
    # put diagonal of S into diagS
    diagS[E_F_i,:] += torch.diagonal(S,dim1=-2,dim2=-1)

    # other side
    S = torch.zeros(batch_in, nloc, nloc,
        device=dev, dtype=torch.float64) # local S matrix
    # Nj2 * Ni1x * n2
    for idim in range(ndim):
        S[:,:nloc,:nloc] += torch.sum(torch.mul(torch.mul(torch.mul(
            torch.flip(snj[E_F_inb, f_inb, :, :, :], [-1]), # torch.flip make copy of data. might be memory-intensive
            snxi[E_F_i, f_i, idim, :,:,:]),
            snormalv[E_F_inb,f_inb, idim, :,:,:]),
            sdetweiv[E_F_i,f_i, :,:,:]),
            -1)*(-0.5)
    # Nj2x * Ni1 * n1
    for idim in range(ndim):
        S[:,:nloc,:nloc] += torch.sum(torch.mul(torch.mul(torch.mul(
            torch.flip(snxj[E_F_inb, f_inb,idim,:,:,:],[-1]),
            sni[E_F_i, f_i, :,:,:]),
            snormalv[E_F_i,f_i, idim, :,:,:]),
            sdetweiv[E_F_i, f_i, :,:,:]),
            -1)*(-0.5)
    # Nj2n2 * Ni1n1 ! n2 \cdot n1 = -1
    S[:,:nloc,:nloc] += torch.mul(
        torch.sum(torch.mul(torch.mul(
        sni[E_F_i, f_i, :,:,:],
        torch.flip(snj[E_F_inb, f_inb, :,:,:],[-1])),
        sdetweiv[E_F_i, f_i, :,:,:]),
        -1)   ,   -mu_e)
    # this S is off-diagonal contribution, therefore no need to put in diagS
    # multiply S and c_i and add to (subtract from) r 
    r[E_F_i,:] -= torch.matmul(S, c_i[E_F_inb,:].view(batch_in,nloc,1)).squeeze()

    return r, diagS

def S_fb(r, f_b, E_F_b, F_b,
    sn, snx, snormal, sdetwei, c_i, c_bc,
    diagS):
    '''
    This function add boundary face S*c_i contribution
    to residual 
    r <- r - S*c_i
    and 
    S*c_bc contribution to rhs, and then, also 
    residual
    r ,_ r + S*c_bc
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
    
    S = torch.zeros(batch_in, nloc, nloc,
        device=dev, dtype=torch.float64) # local S matrix
    # # Nj1 * Ni1x * n1
    for idim in range(ndim):
        S[:,:nloc,:nloc] -= torch.sum(torch.mul(torch.mul(torch.mul(
            snj[E_F_b, f_b, :, :, :],
            snxi[E_F_b, f_b, idim, :,:,:]),
            snormalv[E_F_b,f_b, idim, :,:,:]),
            sdetweiv[E_F_b,f_b, :,:,:]),
            -1) # *(-1.0)
    # # Nj1n1 * Ni1n1 ! n1 \cdot n1 = 1
    S[:,:nloc,:nloc] += torch.mul(
        torch.sum(torch.mul(torch.mul(
        sni[E_F_b, f_b, :,:,:],
        snj[E_F_b, f_b, :,:,:]),
        sdetweiv[E_F_b, f_b, :,:,:]),
        -1)   ,   mu_e)
    # calculate b_bc and add to r
    # r <- r + b_bc
    r[E_F_b, :] += torch.matmul(S, c_bc[E_F_b,:].view(batch_in,nloc,1)).squeeze()
    # diagS[E_F_b,:] += torch.diagonal(S,dim1=-2,dim2=-1)
    # # Nj1x * Ni1 * n1
    for idim in range(ndim):
        S[:,:nloc,:nloc] -= torch.sum(torch.mul(torch.mul(torch.mul(
            sni[E_F_b, f_b, :, :, :],
            snxj[E_F_b, f_b, idim, :,:,:]),
            snormalv[E_F_b,f_b, idim, :,:,:]),
            sdetweiv[E_F_b,f_b, :,:,:]),
            -1) # *(-1.0)
    # calculate S*c and add to (subtract from) r
    r[E_F_b, :] -= torch.matmul(S, c_i[E_F_b,:].view(batch_in,nloc,1)).squeeze()
    diagS[E_F_b,:] += torch.diagonal(S,dim1=-2,dim2=-1)
    
    return r,diagS