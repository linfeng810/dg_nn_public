#!/usr/bin/env python3
"""
This file contains matrix assembly for direct solver.
This is for linear elastic problem.
The volume integral and surface integral terms are
taken from `volume_mf_linear_elastic.py` and
`surface_mf_linear_elastic.py` respectively.
"""

import torch
import config
import numpy as np

ndim = config.ndim
nonods = config.nonods
nloc = config.nloc
nele = config.nele
mu = config.mu
lam = config.lam
dev = config.dev
rho = config.rho
dt = config.dt
nface = config.nface
cijkl = config.cijkl
ngi = config.ngi
sngi = config.sngi

eta_e=36.  # penalty coefficient

def SK_matrix(n, nx, detwei,
              sn, snx, sdetwei, snormal,
              nbele, nbf,
              f, u_bc,
              fina, cola, ncola):
    '''
    This function takes in shape functions and compute
    S and K matrices.

    Input
    -----
    n, nx, detwei,                : volume shape functions
    sn, snx, sdetwei, snormal,    : surface shape functions
    nbele, nbf,                   : neighbours
    f, u_bc,                      : rhs force and Dirichlet boundary conditions
    fina, cola, ncola             : sparsity

    Output
    ------
    SK : scipy block sparse matrix, shape (ndim x nonods, ndim x nonods)
        matrix S+K (+M if transient). use the scipy method `bsr_matrix`
        to create. Block shape is (ndim*nloc, ndim*nloc).
        Overall sparsity is element connectivity.
    rhs_f : np array (nele*ndim*nloc)
        rhs vector that matches the SK matrix block.
    '''
    from scipy.sparse import bsr_matrix

    # K
    # make shape function etc. in shape
    # (nele, nloc(inod), nloc(jnod), ngi)
    #      or
    # (nele, ndim, nloc(inod), nloc(jnod), ngi)
    # all expansions are view of original tensor
    # so that they point to same memory address
    ni = n.unsqueeze(0).unsqueeze(2).expand(nele, -1, nloc, -1)
    nj = n.unsqueeze(0).unsqueeze(1).expand(nele, nloc, -1, -1)
    nxi = nx.unsqueeze(3).expand(-1, -1, -1, nloc, -1)
    nxj = nx.unsqueeze(2).expand(-1, -1, nloc, -1, -1)
    detweiv = detwei.unsqueeze(1).unsqueeze(2).expand(-1, nloc, nloc, -1)
    # declare K
    K = torch.zeros(ndim, ndim, nele, nloc, nloc, device=dev, dtype=torch.float64)
    # ni nj; rhs_f calculation; Mass term if transient
    for idim in range(ndim):
        K[idim, idim, ...] += torch.sum(torch.mul(torch.mul(ni, nj), detweiv), -1)
    f = f.view(ndim, nele, nloc)
    rhs_f = torch.einsum('ij...kl,j...l->i...k', K, f)  # rhs force
    rhs_f = rhs_f.cpu().numpy()  # shape (ndim, nele, nloc)
    rhs_f = np.transpose(rhs_f, (1,0,2))  # shape (nele, ndim, nloc)
    rhs_f = np.reshape(rhs_f, (nele*ndim*nloc))
    # np.savetxt('rhs_f.txt', rhs_f, delimiter=',')
    # Kvalues = K.cpu().numpy()  # shape(ndim,ndim,nele,nloc,nloc)
    # Kvalues = np.transpose(Kvalues, (2, 0, 3, 1, 4))  # shape(nele,ndim,nloc,ndim,nloc)
    # Kvalues = np.reshape(Kvalues, (nele, ndim * nloc, ndim * nloc))
    # np.savetxt('Kmass.txt', Kvalues[0,:,:], delimiter=',')

    if config.isTransient:
        K *= rho / dt
    else:
        K *= 0
    # epsilon_kl C_ijkl epsilon_ij
    K[0, 0, :, :, :] += torch.sum(torch.mul(torch.mul(
        nxi[:, 0, :, :, :], nxj[:, 0, :, :, :]), detweiv), -1) * (lam + 2 * mu)
    K[0, 0, :, :, :] += torch.sum(torch.mul(torch.mul(
        nxi[:, 1, :, :, :], nxj[:, 1, :, :, :]), detweiv), -1) * mu
    K[0, 1, :, :, :] += torch.sum(torch.mul(torch.mul(
        nxi[:, 0, :, :, :], nxj[:, 1, :, :, :]), detweiv), -1) * lam
    K[0, 1, :, :, :] += torch.sum(torch.mul(torch.mul(
        nxi[:, 1, :, :, :], nxj[:, 0, :, :, :]), detweiv), -1) * mu
    K[1, 0, :, :, :] += torch.sum(torch.mul(torch.mul(
        nxi[:, 0, :, :, :], nxj[:, 1, :, :, :]), detweiv), -1) * mu
    K[1, 0, :, :, :] += torch.sum(torch.mul(torch.mul(
        nxi[:, 1, :, :, :], nxj[:, 0, :, :, :]), detweiv), -1) * lam
    K[1, 1, :, :, :] += torch.sum(torch.mul(torch.mul(
        nxi[:, 0, :, :, :], nxj[:, 0, :, :, :]), detweiv), -1) * mu
    K[1, 1, :, :, :] += torch.sum(torch.mul(torch.mul(
        nxi[:, 1, :, :, :], nxj[:, 1, :, :, :]), detweiv), -1) * (lam + 2 * mu)
    Kvalues = K.cpu().numpy()  # shape(ndim,ndim,nele,nloc,nloc)
    Kvalues = np.transpose(Kvalues, (2,0,3,1,4))  # shape(nele,ndim,nloc,ndim,nloc)
    Kvalues = np.reshape(Kvalues, (nele, ndim*nloc, ndim*nloc))

    # S
    # first lets separate nbf to get two list of F_i and F_b
    F_i = np.where(np.logical_not(np.isnan(nbf)))[0]  # interior face
    F_b = np.where(np.isnan(nbf))[0]  # boundary face
    F_inb = -nbf[F_i]  # neighbour list of interior face
    F_inb = F_inb.astype(np.int64)

    # create two lists of which element f_i / f_b is in
    E_F_i = np.floor_divide(F_i, 3)
    E_F_b = np.floor_divide(F_b, 3)
    E_F_inb = np.floor_divide(F_inb, 3)

    # local face number
    f_i = np.mod(F_i, 3)
    f_b = np.mod(F_b, 3)
    f_inb = np.mod(F_inb, 3)
    Svalues = np.zeros((ncola, ndim*nloc, ndim*nloc), dtype=np.float64)
    # for interior faces, assemble S
    # update 3 local faces separately to avoid change r with multiple values
    for iface in range(nface):
        Svalues = S_fi(Svalues, f_i[f_i == iface], E_F_i[f_i == iface], F_i[f_i == iface],
                       f_inb[f_i == iface], E_F_inb[f_i == iface], F_inb[f_i == iface],
                       sn, snx, snormal, sdetwei,
                       fina, cola)
    # for boundary faces, assemble S and b_bc
    # let's hope that each element has only one boundary face.
    Svalues, rhs_bc = S_fb(Svalues, f_b, E_F_b, F_b,
                           sn, snx, snormal, sdetwei, u_bc,
                           fina, cola)
    # add Kvalues to Svalues
    for i in range(nele):
        for j in range(fina[i],fina[i+1]):
            if i==cola[j] :
                Svalues[j,:,:] += Kvalues[i,:,:]
    SK = bsr_matrix((Svalues, cola, fina), shape=(ndim*nonods, ndim*nonods))
    # np.savetxt('rhs_f.txt', rhs_f, delimiter=',')
    # np.savetxt('rhs_bc.txt', rhs_bc, delimiter=',')
    rhs_f += rhs_bc
    return SK, rhs_f


def S_fi(Svalues, f_i, E_F_i, F_i,
         f_inb, E_F_inb, F_inb,
         sn, snx, snormal, sdetwei,
         fina, cola):
    '''
    this function assemble interior faces' contribution
    to S matrix, add to Svalues (ncola, ndim*nloc, ndim*nloc)
    '''

    # faces can be passed in by batches to fit memory/GPU cores
    batch_in = f_i.shape[0]
    mu_e = eta_e / torch.sum(sdetwei[E_F_i, f_i, :], -1)

    # make all tensors in shape (nele, nface, ndim, nloc(inod), nloc(jnod), sngi)
    # all these expansion are views of original tensor,
    # i.e. they point to same memory as sn/snx
    sni = sn.unsqueeze(0).expand(nele, -1, -1, -1) \
        .unsqueeze(3).expand(-1, -1, -1, nloc, -1)  # expand on nloc(jnod)
    snj = sn.unsqueeze(0).expand(nele, -1, -1, -1) \
        .unsqueeze(2).expand(-1, -1, nloc, -1, -1)  # expand on nloc(inod)
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

    # this side
    S = torch.zeros(ndim, ndim, batch_in, nloc, nloc,
                    device=dev, dtype=torch.float64)  # local S matrix
    for idim in range(2):
        for jdim in range(2):
            for kdim in range(2):
                for ldim in range(2):
                    # 0.5 C_ijkl epsilon(u)_kl v_i n_j
                    S[idim, kdim, :, :nloc, :nloc] += \
                        torch.sum(torch.mul(torch.mul(torch.mul(
                            sni[E_F_i, f_i, :, :, :],  # v_i
                            0.5 * snxj[E_F_i, f_i, ldim, :, :, :]),  # 0.5*du_k/dx_l of epsilon_kl
                            snormalv[E_F_i, f_i, jdim, :, :, :]),  # n_j
                            sdetweiv[E_F_i, f_i, :, :, :]),
                            -1) \
                        * cijkl[idim, jdim, kdim, ldim] * (-0.5)
                    S[idim, ldim, :, :nloc, :nloc] += \
                        torch.sum(torch.mul(torch.mul(torch.mul(
                            sni[E_F_i, f_i, :, :, :],  # v_i
                            0.5 * snxj[E_F_i, f_i, kdim, :, :, :]),  # 0.5*du_l/dx_k of epsilon_kl
                            snormalv[E_F_i, f_i, jdim, :, :, :]),  # n_j
                            sdetweiv[E_F_i, f_i, :, :, :]),
                            -1) \
                        * cijkl[idim, jdim, kdim, ldim] * (-0.5)
                    # u_i n_j * 0.5 * C_ijkl epsilon(v)_kl
                    S[kdim, idim, :, :nloc, :nloc] += \
                        torch.sum(torch.mul(torch.mul(torch.mul(
                            snj[E_F_i, f_i, :, :, :],  # u_i
                            0.5 * snxi[E_F_i, f_i, ldim, :, :, :]),  # 0.5*dv_k/dx_l of epsilon_kl
                            snormalv[E_F_i, f_i, jdim, :, :, :]),  # n_j
                            sdetweiv[E_F_i, f_i, :, :, :]),
                            -1) \
                        * cijkl[idim, jdim, kdim, ldim] * (-0.5)
                    S[ldim, idim, :, :nloc, :nloc] += \
                        torch.sum(torch.mul(torch.mul(torch.mul(
                            snj[E_F_i, f_i, :, :, :],  # u_i
                            0.5 * snxi[E_F_i, f_i, kdim, :, :, :]),  # 0.5*dv_l/dx_k of epsilon_kl
                            snormalv[E_F_i, f_i, jdim, :, :, :]),  # n_j
                            sdetweiv[E_F_i, f_i, :, :, :]),
                            -1) \
                        * cijkl[idim, jdim, kdim, ldim] * (-0.5)
    # mu * u_i v_i
    for idim in range(2):
        S[idim, idim, :, :nloc, :nloc] += torch.mul(
            torch.sum(torch.mul(torch.mul(
                sni[E_F_i, f_i, :, :, :],
                snj[E_F_i, f_i, :, :, :]),
                sdetweiv[E_F_i, f_i, :, :, :]),
                -1), mu_e)
    # put S contribution to Svalues
    Sdetach = S.cpu().numpy()  # shape (ndim, ndim, batch_in, nloc, nloc)
    Sdetach = np.transpose(Sdetach, (2,0,3,1,4))  # shape(batch_in, ndim, nloc, ndim, nloc)
    Sdetach = np.reshape(Sdetach, (batch_in, ndim*nloc, ndim*nloc))
    for i in range(E_F_i.shape[0]) :
        e = E_F_i[i]
        for j in range(fina[e], fina[e+1]) :
            col = cola[j]
            if e==col :  # this is diagonal block
                Svalues[j,:,:] += Sdetach[i,:,:]

    # other side
    S *= 0.  # local S matrix
    for idim in range(2):
        for jdim in range(2):
            for kdim in range(2):
                for ldim in range(2):
                    # 0.5 C_ijkl epsilon(u)_kl v_i n_j
                    S[idim, kdim, :, :nloc, :nloc] += \
                        torch.sum(torch.mul(torch.mul(torch.mul(
                            sni[E_F_i, f_i, :, :, :],  # v_i
                            0.5 * torch.flip(snxj[E_F_inb, f_inb, ldim, :, :, :], [-1])),  # 0.5*du_k/dx_l of epsilon_kl
                            snormalv[E_F_i, f_i, jdim, :, :, :]),  # n_j
                            sdetweiv[E_F_i, f_i, :, :, :]),
                            -1) \
                        * cijkl[idim, jdim, kdim, ldim] * (-0.5)
                    S[idim, ldim, :, :nloc, :nloc] += \
                        torch.sum(torch.mul(torch.mul(torch.mul(
                            sni[E_F_i, f_i, :, :, :],  # v_i
                            0.5 * torch.flip(snxj[E_F_inb, f_inb, kdim, :, :, :], [-1])),  # 0.5*du_l/dx_k of epsilon_kl
                            snormalv[E_F_i, f_i, jdim, :, :, :]),  # n_j
                            sdetweiv[E_F_i, f_i, :, :, :]),
                            -1) \
                        * cijkl[idim, jdim, kdim, ldim] * (-0.5)
                    # u_i n_j * 0.5 * C_ijkl epsilon(v)_kl
                    S[kdim, idim, :, :nloc, :nloc] += \
                        torch.sum(torch.mul(torch.mul(torch.mul(
                            torch.flip(snj[E_F_inb, f_inb, :, :, :], [-1]),  # u_i
                            0.5 * snxi[E_F_i, f_i, ldim, :, :, :]),  # 0.5*dv_k/dx_l of epsilon_kl
                            snormalv[E_F_inb, f_inb, jdim, :, :, :]),  # n_j
                            sdetweiv[E_F_i, f_i, :, :, :]),
                            -1) \
                        * cijkl[idim, jdim, kdim, ldim] * (-0.5)
                    S[ldim, idim, :, :nloc, :nloc] += \
                        torch.sum(torch.mul(torch.mul(torch.mul(
                            torch.flip(snj[E_F_inb, f_inb, :, :, :], [-1]),  # u_i
                            0.5 * snxi[E_F_i, f_i, kdim, :, :, :]),  # 0.5*dv_l/dx_k of epsilon_kl
                            snormalv[E_F_inb, f_inb, jdim, :, :, :]),  # n_j
                            sdetweiv[E_F_i, f_i, :, :, :]),
                            -1) \
                        * cijkl[idim, jdim, kdim, ldim] * (-0.5)
    # mu * u_i v_i
    for idim in range(2):
        S[idim, idim, :, :nloc, :nloc] += torch.mul(
            torch.sum(torch.mul(torch.mul(
                sni[E_F_i, f_i, :, :, :],
                torch.flip(snj[E_F_inb, f_inb, :, :, :], [-1])),
                sdetweiv[E_F_i, f_i, :, :, :]),
                -1), -mu_e)
    # this S is off-diagonal contribution
    # add to Svalues
    Sdetach = S.cpu().numpy()  # shape (ndim, ndim, batch_in, nloc, nloc)
    Sdetach = np.transpose(Sdetach, (2,0,3,1,4))  # shape (batch_in, ndim, nloc, ndim, nloc)
    Sdetach = np.reshape(Sdetach, (batch_in, ndim*nloc, ndim*nloc))
    for i in range(E_F_i.shape[0]):
        e = E_F_i[i]
        e2 = E_F_inb[i]
        for j in range(fina[e], fina[e+1]) :
            col = cola[j]
            if e2==col :  # this is the neighbouring off-diagonal block
                Svalues[j,:,:] += Sdetach[i,:,:]
    return Svalues


def S_fb(Svalues, f_b, E_F_b, F_b,
         sn, snx, snormal, sdetwei, u_bc,
         fina, cola):
    '''
    This function assembles the boundary face's
    contribution to S matrix, and, the Dirichlet
    boundary conditions on rhs vector.
    '''

    # faces can be passed in by batches to fit memory/GPU cores
    batch_in = f_b.shape[0]
    mu_e = eta_e / torch.sum(sdetwei[E_F_b, f_b, :], -1)

    # make all tensors in shape (nele, nface, ndim, nloc(inod), nloc(jnod), sngi)
    # all these expansion are views of original tensor,
    # i.e. they point to same memory as sn/snx
    sni = sn.unsqueeze(0).expand(nele, -1, -1, -1) \
        .unsqueeze(3).expand(-1, -1, -1, nloc, -1)  # expand on nloc(jnod)
    snj = sn.unsqueeze(0).expand(nele, -1, -1, -1) \
        .unsqueeze(2).expand(-1, -1, nloc, -1, -1)  # expand on nloc(inod)
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
    S = torch.zeros(ndim, ndim, batch_in, nloc, nloc,
                    device=dev, dtype=torch.float64)  # local S matrix
    # u_i n_j * 0.5 * C_ijkl epsilon(v)_kl
    for idim in range(ndim):
        for jdim in range(ndim):
            for kdim in range(ndim):
                for ldim in range(ndim):
                    S[kdim, idim, :, :nloc, :nloc] += \
                        torch.sum(torch.mul(torch.mul(torch.mul(
                            snj[E_F_b, f_b, :, :, :],  # u_i
                            0.5 * snxi[E_F_b, f_b, ldim, :, :, :]),  # 0.5*dv_k/dx_l of epsilon_kl
                            snormalv[E_F_b, f_b, jdim, :, :, :]),  # n_j
                            sdetweiv[E_F_b, f_b, :, :, :]),
                            -1) \
                        * cijkl[idim, jdim, kdim, ldim] * (-1.0)
                    S[ldim, idim, :, :nloc, :nloc] += \
                        torch.sum(torch.mul(torch.mul(torch.mul(
                            snj[E_F_b, f_b, :, :, :],  # u_i
                            0.5 * snxi[E_F_b, f_b, kdim, :, :, :]),  # 0.5*dv_l/dx_k of epsilon_kl
                            snormalv[E_F_b, f_b, jdim, :, :, :]),  # n_j
                            sdetweiv[E_F_b, f_b, :, :, :]),
                            -1) \
                        * cijkl[idim, jdim, kdim, ldim] * (-1.0)
                    # print('i j k l cijkl: ', idim, jdim, kdim, ldim, cijkl[idim, jdim, kdim, ldim])
    # mu * u_i v_i
    for idim in range(2):
        S[idim, idim, :, :nloc, :nloc] += torch.mul(
            torch.sum(torch.mul(torch.mul(
                sni[E_F_b, f_b, :, :, :],
                snj[E_F_b, f_b, :, :, :]),
                sdetweiv[E_F_b, f_b, :, :, :]),
                -1), mu_e)
    # multiply S and u_bc to get bc contribution to rhs vector
    rhs_bc = torch.zeros(ndim, nele, nloc, device=dev, dtype=torch.float64)
    rhs_bc[:, E_F_b, :] += torch.einsum('ij...kl,j...l->i...k', S, u_bc[:, E_F_b, :])
    rhs_bc = rhs_bc.cpu().numpy()  # shape (ndim, nele, nloc)
    rhs_bc = np.transpose(rhs_bc, (1,0,2))  # shape (nele, ndim, nloc)
    rhs_bc = np.reshape(rhs_bc, (nele*ndim*nloc))

    # 0.5 C_ijkl epsilon(u)_kl v_i n_j
    for idim in range(ndim):
        for jdim in range(ndim):
            for kdim in range(ndim):
                for ldim in range(ndim):
                    S[idim, kdim, :, :nloc, :nloc] += \
                        torch.sum(torch.mul(torch.mul(torch.mul(
                            sni[E_F_b, f_b, :, :, :],  # v_i
                            0.5 * snxj[E_F_b, f_b, ldim, :, :, :]),  # 0.5*du_k/dx_l of epsilon_kl
                            snormalv[E_F_b, f_b, jdim, :, :, :]),  # n_j
                            sdetweiv[E_F_b, f_b, :, :, :]),
                            -1) \
                        * cijkl[idim, jdim, kdim, ldim] * (-1.0)
                    S[idim, ldim, :, :nloc, :nloc] += \
                        torch.sum(torch.mul(torch.mul(torch.mul(
                            sni[E_F_b, f_b, :, :, :],  # v_i
                            0.5 * snxj[E_F_b, f_b, kdim, :, :, :]),  # 0.5*du_l/dx_k of epsilon_kl
                            snormalv[E_F_b, f_b, jdim, :, :, :]),  # n_j
                            sdetweiv[E_F_b, f_b, :, :, :]),
                            -1) \
                        * cijkl[idim, jdim, kdim, ldim] * (-1.0)
    # add S to Svalues
    S = S.cpu().numpy()  # shape (ndim, ndim, batch_in, nloc, nloc)
    S = np.transpose(S, (2, 0, 3, 1, 4))  # shape (batch_in, ndim, nloc, ndim, nloc)
    S = np.reshape(S, (batch_in, ndim*nloc, ndim*nloc))
    for i in range(E_F_b.shape[0]) :
        e = E_F_b[i]
        for j in range(fina[e], fina[e+1]) :
            col = cola[j]
            if e==col :  # this is the digonal block
                Svalues[j,:,:] += S[i,:,:]
    return Svalues, rhs_bc
