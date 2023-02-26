#!/usr/bin/env python3
'''
This file implements volume integral term
for the linear elastic problem
in finest grid and P0DG grid matrix-free-ly.
Hence the name "volume_mf_linear_elastic".
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
cijkl = config.cijkl
lam = config.lam
mu = config.mu
dt = config.dt 
rho = config.rho 

def K_mf(r, n, nx, detwei, u_i, f, u_old=0):
    '''
    This function compute the Ku contribution to the residual:
        r <- r - K*u
    where r is the residual on the finest grid,
        K is volume integral term matrix (may contain mass if transient)
        u is field variable.

    # Input
    r : torch tensor (ndim, nonods)
        residual vector that hasn't taken into account of K*c
    n : torch tensor (nloc, ngi)
        shape function at reference element quadrature pnts
    nx : torch tensor (nele, ndim, nloc, ngi)
        shape func derivatives at quad pnts
    detwei : torch tensor (nele, ngi)
        det x quad weights for volume integral
    u_i : torch tensor (ndim, nonods)
        field value at i-th iteration (last iteration)
    f : torch tensor (ndim, nonods)
        right hand side force.
    u_old : torch tensor (ndim, nonods)
        (optional), field value at last timestep, if transient

    # Output
    r : torch tensor (ndim, nonods)
        residual vector that has taken into account of K*u
    diagK : torch tensor (ndim, nonods)
        diagonal of volume integral matrix (may contain mass if transient)
    '''

    u_i = u_i.view(ndim, nele, nloc)
    r = r.view(ndim, nele, nloc)
    f = f.view(ndim, nele, nloc)

    # output declaration
    diagK = torch.zeros(ndim, nele, nonods, device=dev, dtype=torch.float64)

    # make shape function etc. in shape
    # (nele, nloc(inod), nloc(jnod), ngi) 
    #      or
    # (nele, ndim, nloc(inod), nloc(jnod), ngi)
    # all expansions are view of original tensor
    # so that they point to same memory address
    ni = n.unsqueeze(0).unsqueeze(2).expand(nele,-1,nloc,-1)
    nj = n.unsqueeze(0).unsqueeze(1).expand(nele,nloc,-1,-1)
    nxi = nx.unsqueeze(3).expand(-1,-1,-1,nloc,-1)
    nxj = nx.unsqueeze(2).expand(-1,-1,nloc,-1,-1)
    detweiv = detwei.unsqueeze(1).unsqueeze(2).expand(-1,nloc,nloc,-1)

    # declare K
    K = torch.zeros(ndim,ndim,nele,nloc,nloc, device=dev, dtype=torch.float64)

    # ni nj
    K+= torch.sum(torch.mul(torch.mul(
        ni,nj),detweiv),-1).unsqueeze(0).unsqueeze(1).expand(ndim,ndim,-1,-1,-1)
    r += torch.einsum('ij...kl,j...l->i...k', K, f) # rhs force
    if (config.isTransient) :
        K *= rho/dt 
        r += torch.einsum('ij...kl,j...l->i...k', 
            K, (u_old.view(ndim,nele,nloc)-u_i) )
        diagK += torch.permute(
            torch.diagonal(
                torch.diagonal(K,dim1=-2,dim2=-1)
                , dim1=0,dim2=1),
            (2,0,1))
    else :
        K *= 0

    # epsilon_kl C_ijkl epsilon_ij
    K[0,0,:,:,:] += \
        torch.sum(
            torch.mul(torch.mul(
                nxi[:,0,:,:,:],nxj[:,0,:,:,:]
            ), detweiv[:,:,:,:])
        ,-1)*(lam+2*mu)
    K[0,0,:,:,:] += \
        torch.sum(
            torch.mul(torch.mul(
                nxi[:,1,:,:,:],nxj[:,1,:,:,:]
            ), detweiv[:,:,:,:])
        ,-1)*(mu)
    K[0,1,:,:,:] += \
        torch.sum(
            torch.mul(torch.mul(
                nxi[:,0,:,:,:],nxj[:,1,:,:,:]
            ), detweiv[:,:,:,:])
        ,-1)*(lam)
    K[0,1,:,:,:] += \
        torch.sum(
            torch.mul(torch.mul(
                nxi[:,1,:,:,:],nxj[:,0,:,:,:]
            ), detweiv[:,:,:,:])
        ,-1)*(lam)
    K[1,0,:,:,:] += \
        torch.sum(
            torch.mul(torch.mul(
                nxi[:,0,:,:,:],nxj[:,1,:,:,:]
            ), detweiv[:,:,:,:])
        ,-1)*(mu)
    K[1,0,:,:,:] += \
        torch.sum(
            torch.mul(torch.mul(
                nxi[:,1,:,:,:],nxj[:,0,:,:,:]
            ), detweiv[:,:,:,:])
        ,-1)*(lam)
    K[1,1,:,:,:] += \
        torch.sum(
            torch.mul(torch.mul(
                nxi[:,0,:,:,:],nxj[:,0,:,:,:]
            ), detweiv[:,:,:,:])
        ,-1)*(mu)
    K[1,1,:,:,:] += \
        torch.sum(
            torch.mul(torch.mul(
                nxi[:,1,:,:,:],nxj[:,1,:,:,:]
            ), detweiv[:,:,:,:])
        ,-1)*(lam+2*mu)

    # add to residual
    r -= torch.einsum('ij...kl,j...l->i...k', K, u_i) # rhs force
    # extract diagonal
    diagK += torch.permute(
        torch.diagonal(
            torch.diagonal(K,dim1=-2,dim2=-1)
            , dim1=0,dim2=1),
        (2,0,1))
    # make memory contiguous
    r = r.view(ndim,nonods).contiguous()
    diagK = diagK.view(ndim,nonods).contiguous()
    return r, diagK

def RKR_mf(n, nx, detwei, R):
    '''
    This function compute the K operator on P0DG grid
    i.e. R^T * K * R

    # Input
    n : torch tensor (nloc, ngi)
        shape function at reference element quadrature pnts
    nx : torch tensor (nele, ndim, nloc, ngi)
        shape func derivatives at quad pnts
    detwei : torch tensor (nele, ngi)
        det x quad weights for volume integral
    R : torch tensor (nloc)
        restrictor
        
    # Output
    diagRKR : torch tensor (ndim, nele)
        diagonal or RKR matrix
    RKRvalues : torch tensor (nele, ndim, ndim)
        values of RKR sparse matrix.
    '''

    # make shape function etc. in shape
    # (nele, nloc(inod), nloc(jnod), ngi) 
    #      or
    # (nele, ndim, nloc(inod), nloc(jnod), ngi)
    # all expansions are view of original tensor
    # so that they point to same memory address
    ni = n.unsqueeze(0).unsqueeze(2).expand(nele,-1,nloc,-1)
    nj = n.unsqueeze(0).unsqueeze(1).expand(nele,nloc,-1,-1)
    nxi = nx.unsqueeze(3).expand(-1,-1,-1,nloc,-1)
    nxj = nx.unsqueeze(2).expand(-1,-1,nloc,-1,-1)
    detweiv = detwei.unsqueeze(1).unsqueeze(2).expand(-1,nloc,nloc,-1)

    # declare K
    K = torch.zeros(ndim,ndim,nele,nloc,nloc, device=dev, dtype=torch.float64)

    # ni nj
    K+= torch.sum(torch.mul(torch.mul(
        ni,nj),detweiv),-1).unsqueeze(0).unsqueeze(1).expand(ndim,ndim,-1,-1,-1)
    if (config.isTransient) :
        K *= rho/dt 
        diagK += torch.permute(
            torch.diagonal(
                torch.diagonal(K,dim1=-2,dim2=-1)
                , dim1=0,dim2=1),
            (2,0,1))
    else :
        K *= 0

    # epsilon_kl C_ijkl epsilon_ij
    K[0,0,:,:,:] += \
        torch.sum(
            torch.mul(torch.mul(
                nxi[:,0,:,:,:],nxj[:,0,:,:,:]
            ), detweiv[:,:,:,:])
        ,-1)*(lam+2*mu)
    K[0,0,:,:,:] += \
        torch.sum(
            torch.mul(torch.mul(
                nxi[:,1,:,:,:],nxj[:,1,:,:,:]
            ), detweiv[:,:,:,:])
        ,-1)*(mu)
    K[0,1,:,:,:] += \
        torch.sum(
            torch.mul(torch.mul(
                nxi[:,0,:,:,:],nxj[:,1,:,:,:]
            ), detweiv[:,:,:,:])
        ,-1)*(lam)
    K[0,1,:,:,:] += \
        torch.sum(
            torch.mul(torch.mul(
                nxi[:,1,:,:,:],nxj[:,0,:,:,:]
            ), detweiv[:,:,:,:])
        ,-1)*(lam)
    K[1,0,:,:,:] += \
        torch.sum(
            torch.mul(torch.mul(
                nxi[:,0,:,:,:],nxj[:,1,:,:,:]
            ), detweiv[:,:,:,:])
        ,-1)*(mu)
    K[1,0,:,:,:] += \
        torch.sum(
            torch.mul(torch.mul(
                nxi[:,1,:,:,:],nxj[:,0,:,:,:]
            ), detweiv[:,:,:,:])
        ,-1)*(lam)
    K[1,1,:,:,:] += \
        torch.sum(
            torch.mul(torch.mul(
                nxi[:,0,:,:,:],nxj[:,0,:,:,:]
            ), detweiv[:,:,:,:])
        ,-1)*(mu)
    K[1,1,:,:,:] += \
        torch.sum(
            torch.mul(torch.mul(
                nxi[:,1,:,:,:],nxj[:,1,:,:,:]
            ), detweiv[:,:,:,:])
        ,-1)*(lam+2*mu)

    # declare output
    RKRvalues = torch.einsum('...ij,i,j->...', K, R, R)
    RKRvalues = torch.permute(RKRvalues,[2,0,1])
    # extract diagonal
    diagRKR = torch.zeros(ndim,nele,device=dev, dtype=torch.float64)
    for idim in range(ndim):
        diagRKR[idim,:] += RKRvalues[:,idim,idim]
    
    return diagRKR, RKRvalues

def calc_RAR(diagRSR, diagRKR, RSRvalues, RKRvalues, fina, cola):
    '''
    calculate the sum of RKR and RSR to get RAR

    # Input:
    diagRSR : torch tensor (ndim, nele)
        diagonal of RSR
    diagRKR : torch tensor (ndim, nele)
        diagonal of RKR
    RSRvalues : torch tensor (ncola, ndim, ndim)
        values of RSR matrix
    RKRvalues : torch tensor (nele, ndim, ndim)
        values of RKR matrix
    fina : torch tensor (nele+1)
        connectivity matrix, start of rows
    cola : torch tensor (nonod)
        connectivity matrix, column indices

    # Output
    diagRAR : torch tensor (ndim, nele)
        diagonal of RAR
    RARvalues : torch tensor (ncola, ndim, ndim)
        values of RAR matrix, has same sparsity as RSR
    '''
    diagRAR = diagRSR + diagRKR 

    RARvalues = torch.zeros_like(RSRvalues, device=dev, dtype=torch.float64)
    for ele in range(nele):
        for spIdx in range(fina[ele], fina[ele+1]):
            if (cola[spIdx]==ele) :
                RARvalues[spIdx,:,:] = RSRvalues[spIdx,:,:] + RKRvalues[ele,:,:]

    return diagRAR , RARvalues