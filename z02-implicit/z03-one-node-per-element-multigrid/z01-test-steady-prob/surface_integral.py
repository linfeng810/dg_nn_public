import torch
import config 
import numpy as np
from scipy.sparse import coo_matrix 
from mesh_init import face_iloc,face_iloc2
from mesh_init import sgi2 as pnt_to_sgi2

torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)

dev = config.dev
nele = config.nele 
mesh = config.mesh 
nonods = config.nonods 
ngi = config.ngi
ndim = config.ndim
nloc = config.nloc 
dt = config.dt 
tend = config.tend 
tstart = config.tstart

#####################################
# now we assemble S (surface integral)
#####################################

# first let's define surface shape functions
# note that we're using mass lumping and
# only one value per node is required
# that is either 1/3 or 1/6 multiplied by 
# the curve length

#####
# build sparsity
#####
# let's build in coo format first - easier to construct
# then transform to csr format - more efficient to do linear algebra operations
def S_Minv_sparse(sn, snx, sdetwei, snormal, x_all, nbele, nbf, c_bc):
    # input:
    # sn - shape function at face quadrature pnts, numpy (nface, nloc, sngi)
    # snx - shape func derivatives at face quadrature pnts, 
    #       torch (nele, nface, ndim, nloc, sngi)
    # sdetwei - det time quadrature weights for surface integral,
    #           torch (nele, nface, sngi)
    # snormal - unit out normal of face, torch (nele, nface, ndim)
    # x_all - numpy array (nonods, ndim)
    # nbele - list of neighbour element index
    # nbf - list of neighbour face index
    # c_bc - Dirichlet boundary values at boundary nodes, 0 otherwise. This is to impose Diri bcs weakly
    #        torch (nonods)
    
    # output:
    # diagS, diagonal of S, torch tensor (nonods) 
    # S - surface integral matrix, torch.sparse_csr_tensor
    # b_bc - rhs vector accounting for boundary conditions, torch tensor (nonods)
    
    b_bc = torch.zeros(nonods, dtype=torch.float64, device=dev)
    # transfer shape functions and weights to cpu numpy array. 
    # would be easier to flip and do multiplication
    # we are not in parallel anyway.
    snx = snx.cpu().numpy()
    sdetwei = sdetwei.cpu().numpy()
    snormal = snormal.cpu().numpy()

    # S matrix 
    indices=[] # indices of entries, a list of lists
    values=[]  # values to be add to S

    ### this is the *simple* interior penalty method. not accurate. delete later.
    if (True):
        # coeff = np.asarray([1./8., 3./8., 3./8., 1./8.]) # lumped face mass
        coeff = np.asarray([1./6., 1./3., 1./3., 1./6.])
        # coeff = np.asarray([1./4., 1./4., 1./4., 1./4.])*1e8
        for ele in range(nele):
            # loop over surfaces
            for iface in range(3):
                glb_iface = ele*3+iface 
                ele2 = nbele[glb_iface]
                if (np.isnan(ele2)):
                    # this is a boundary face without neighbouring element
                    farea=np.linalg.norm(x_all[ele*nloc+iface,:]-x_all[ele*nloc+(iface+1)%3,:])
                    dx = farea/4. # use farea to approximate dx
                    ifaceloc=0
                    for iloc in face_iloc(iface):
                        glb_iloc = ele*nloc+iloc
                        indices.append([glb_iloc, glb_iloc])
                        values.append(1e2*coeff[ifaceloc]*farea/dx)
                        b_bc[glb_iloc] = b_bc[glb_iloc] + 1e2*coeff[ifaceloc]*farea/dx * c_bc[glb_iloc]
                        # print(glb_iloc, coeff[ifaceloc]*farea/dx, \
                        #     c_bc[glb_iloc].cpu().numpy(), b_bc[glb_iloc].cpu().numpy(),\
                        #     )
                        ifaceloc+=1
                    # S matrix value                  face node   0--1--2--3
                    # values.append(1./6.*farea/dx)   # 0     diag
                    # print(c_bc[ele*nloc+0])
                    # b_bc[ele*nloc+0] = 1./6.*farea/dx * c_bc[ele*nloc+0]
                    # print(b_bc[ele*nloc+0])
                    # values.append(1./3.*farea/dx)   # 1     diag
                    # b_bc[ele*nloc+1] = 1./3.*farea/dx * c_bc[ele*nloc+1]
                    # print(b_bc[ele*nloc+1])
                    # values.append(1./3.*farea/dx)   # 2     diag
                    # b_bc[ele*nloc+2] = 1./3.*farea/dx * c_bc[ele*nloc+2]
                    # print(b_bc[ele*nloc+2])
                    # values.append(1./6.*farea/dx)   # 3     diag
                    # b_bc[ele*nloc+3] = 1./6.*farea/dx * c_bc[ele*nloc+3]
                    # print(b_bc[ele*nloc+3])
                    continue 
                ele2 = int(abs(ele2))
                glb_iface2 = int(abs(nbf[glb_iface]))
                iface2 = glb_iface2 % 3
                dx=np.linalg.norm(x_all[ele*nloc+9,:]-x_all[int(ele2)*nloc+9,:])/4. # dx/(order+1)
                # print(ele, iface, dx,x_all[ele*nloc+9,:], x_all[int(ele2)*nloc+9,:])
                farea=np.linalg.norm(x_all[ele*nloc+iface,:]-x_all[ele*nloc+(iface+1)%3,:])
                # print(ele, iface, farea)
                # print(ele, ele2, '|', iface, iface2, '|', face_iloc(iface), face_iloc2(iface2))
                ifaceloc = 0
                for iloc,iloc2 in zip(face_iloc(iface), face_iloc2(iface2)):
                    glb_iloc = ele*nloc+iloc 
                    glb_iloc2 = int(abs(ele2*nloc+iloc2))
                    # print(ele, ele2, '|', iface, iface2, '|', iloc, iloc2, '|', glb_iloc, glb_iloc2)
                    # print('\t',x_all[glb_iloc]-x_all[glb_iloc2])
                                
                    indices.append([glb_iloc, glb_iloc])
                    indices.append([glb_iloc, glb_iloc2])

                    # S matrix value                  face node   0--1--2--3
                    values.append(coeff[ifaceloc]*farea/dx)   # 0     diag  
                    values.append(-coeff[ifaceloc]*farea/dx)  # 0     off-diag
                    # values.append(1./3.*farea/dx)   # 1     diag
                    # values.append(-1./3.*farea/dx)  # 1     off-diag
                    # values.append(1./3.*farea/dx)   # 2     diag
                    # values.append(-1./3.*farea/dx)  # 2     off-diag
                    # values.append(1./6.*farea/dx)   # 3     diag
                    # values.append(-1./6.*farea/dx)  # 3     off-diag

                    ifaceloc+=1


    ### Classic IP method as addressed in Arnold et al. 2002
    if (False):
        eta_e = 36. # penalty coefficient
        for ele in range(nele):
            for iface in range(config.nface):
                mu_e = eta_e/np.sum(sdetwei[ele,iface,:]) # penalty coeff
                glb_iface = ele*3+iface 
                ele2 = nbele[glb_iface]
                if (np.isnan(ele2)):
                    # this is a boundary face
                    # print('ele %d ele2 nan iface %d iface2 nan'%(ele,iface))
                    for inod in range(nloc):
                        glb_inod = ele*nloc+inod 
                        # this side 
                        # Note to myself: 
                        # j -> jnod   i -> inod
                        # 1 -> ele | iface   2 -> ele2 | iface2
                        for jnod in range(nloc):
                            glb_jnod = ele*nloc+jnod 
                            nnx = 0
                            nxn = 0
                            nn = 0
                            # maybe we can replace these two idim/sgi loops with np.einsum and optimize the path
                            for sgi in range(config.sngi):
                                for idim in range(ndim):
                                    nnx += sn[iface,jnod,sgi]*snx[ele,iface,idim,inod,sgi]*snormal[ele,iface,idim]*sdetwei[ele,iface,sgi] # Nj1 * Ni1x * n1
                                    nxn += snx[ele,iface,idim,jnod,sgi]*sn[iface,inod,sgi]*snormal[ele,iface,idim]*sdetwei[ele,iface,sgi] # Nj1x * Ni1 * n1
                                nn += sn[iface,jnod,sgi]*sn[iface,inod,sgi]*sdetwei[ele,iface,sgi] # Nj1n1 * Ni1n1 ! n1 \cdot n1 = 1
                            # sum 
                            indices.append([glb_inod, glb_jnod])
                            values.append(-nnx-nxn+mu_e*nn)
                            # values.append(-nxn+mu_e*nn) # revirie beatrice
                            # values.append(mu_e*nn) # chris suggest to try deleting nn term
                            # print('glbi, %d, glbj, %d, nnx %.16f, nxn %.16f, nn %.16f'%(glb_inod,glb_jnod,nnx,nxn,nn))
                            b_bc[glb_inod] = b_bc[glb_inod] + c_bc[glb_jnod] * (-nnx+mu_e*nn)
                            # print('bbc glbi %d globj %d c_bc %f'%(glb_inod, glb_jnod, c_bc[glb_jnod]))
                            # b_bc[glb_inod] = b_bc[glb_inod] + c_bc[glb_jnod] * (-nxn+mu_e*nn) # according to Beatrice eq. (2.23)+1, there is no nnx term in Diri bc.
                            # b_bc[glb_inod] = b_bc[glb_inod] + c_bc[glb_jnod] * (mu_e*nn) # chris suggest to try deleting nn term
                    continue 
                ele2 = int(abs(ele2))
                glb_iface2 = int(abs(nbf[glb_iface]))
                iface2 = glb_iface2%3
                # print('ele %d ele2 %d iface %d iface2 %d'%(ele,ele2,iface,iface2))
                for inod in range(nloc):
                    glb_inod = ele*nloc+inod 
                    # this side 
                    # Note to myself: 
                    # j -> jnod   i -> inod
                    # 1 -> ele | iface   2 -> ele2 | iface2
                    for jnod in range(nloc):
                        glb_jnod = ele*nloc+jnod 
                        nnx = 0
                        nxn = 0
                        nn = 0
                        # maybe we can replace these two idim/sgi loops with np.einsum and optimize the path
                        for sgi in range(config.sngi):
                            for idim in range(ndim):
                                nnx += sn[iface,jnod,sgi]*snx[ele,iface,idim,inod,sgi]*snormal[ele,iface,idim]*sdetwei[ele,iface,sgi] # Nj1 * Ni1x * n1
                                nxn += snx[ele,iface,idim,jnod,sgi]*sn[iface,inod,sgi]*snormal[ele,iface,idim]*sdetwei[ele,iface,sgi] # Nj1x * Ni1 * n1
                            nn += sn[iface,jnod,sgi]*sn[iface,inod,sgi]*sdetwei[ele,iface,sgi] # Nj1n1 * Ni1n1 ! n1 \cdot n1 = 1
                        # sum 
                        indices.append([glb_inod, glb_jnod])
                        values.append(-0.5*nnx-0.5*nxn+mu_e*nn)
                        # print('glbi, %d, glbj, %d, nnx %.16f, nxn %.16f, nn %.16f'%(glb_inod,glb_jnod,nnx,nxn,nn))
                    # other side
                    for jnod2 in range(nloc):
                        glb_jnod2 = ele2*nloc+jnod2 
                        nnx = 0
                        nxn = 0
                        nn = 0
                        for sgi in range(config.sngi):
                            sgi2 = pnt_to_sgi2(sgi) # match surface quadrature pnts on other side
                            for idim in range(ndim):
                                nnx += sn[iface2,jnod2,sgi2]*snx[ele,iface,idim,inod,sgi]*snormal[ele2,iface2,idim]*sdetwei[ele,iface,sgi] # Nj2 * Ni1x * n2
                                nxn += snx[ele2,iface2,idim,jnod2,sgi2]*sn[iface,inod,sgi]*snormal[ele,iface,idim]*sdetwei[ele,iface,sgi] # Nj2x * Ni1 * n1
                            nn += (-1.)*sn[iface2,jnod2,sgi2]*sn[iface,inod,sgi]*sdetwei[ele,iface,sgi] # Nj2n2 * Ni1n1 ! n2 \cdot n1 = -1
                        # sum
                        indices.append([glb_inod, glb_jnod2])
                        values.append(-0.5*nnx-0.5*nxn+mu_e*nn)
                        # print('glbi, %d, glbj, %d, nnx %.16f, nxn %.16f, nn %.16f'%(glb_inod,glb_jnod2,nnx,nxn,nn))

    values = torch.tensor(values)
    # print(values)
    indices = torch.transpose(torch.tensor(indices),0,1)



    S_scipy = coo_matrix((values, (indices[0,:].numpy(), indices[1,:].numpy()) ), shape=(nonods, nonods))
    S_scipy = S_scipy.tocsr()  # this transformation will altomatically add entries at same position, perfect for assembling
    diagS = torch.tensor(S_scipy.diagonal(),device=dev).view(nonods)
    S = torch.sparse_csr_tensor(crow_indices=torch.tensor(S_scipy.indptr), \
        col_indices=torch.tensor(S_scipy.indices), \
        values=S_scipy.data, \
        size=(nonods, nonods), \
        device=dev)
    
    np.savetxt('fina.txt', S_scipy.indptr,delimiter=',')
    np.savetxt('cola.txt', S_scipy.indices,delimiter=',')
    

    # # inverse of mass matrix Minv_sparse
    # indices=[]
    # values=[]
    # for ele in range(nele):
    #     for iloc in range(nloc):
    #         for jloc in range(nloc):
    #             glb_iloc = int( ele*nloc+iloc )
    #             glb_jloc = int( ele*nloc+jloc )
    #             indices.append([glb_iloc, glb_jloc])
    #             values.append(Minv[ele,iloc,jloc])
    # values = torch.tensor(values)
    # indices = torch.transpose(torch.tensor(indices),0,1)
    # Minv_scipy = coo_matrix((values,(indices[0,:].numpy(), indices[1,:].numpy())), shape=(nonods, nonods))
    # Minv_scipy = Minv_scipy.tocsr() 
    # Minv = torch.sparse_csr_tensor( crow_indices=torch.tensor(Minv_scipy.indptr), \
    #     col_indices=torch.tensor(Minv_scipy.indices), \
    #     values=Minv_scipy.data, \
    #     size=(nonods, nonods) , \
    #     device=dev)

    return diagS, S, b_bc


def RSR_matrix(S, R):
    '''
    Calculated S operator on level 1 coarse grid.

    Input:
    S - S matrix (nonods, nonods), sparse
    R - rotation matrix (nloc, 1)

    Output:
    diagRSR - diagonal of restricted operator, (nele)
    RSR - restricted operator, (nele, nele), torch csr sparse matrix
    RTbig - restrictor, torch csr sparse matrix(nele, nonods)
    '''

    # Rbig matrix (nonods, nele)
    # [R  .. .. .. ..]
    # [.. R  .. .. ..]
    # [.. .. R  .. ..]
    # [.. .. .. R  ..]
    # [ :  :  :  :  :]
    # [.. .. .. .. R ]
    indices=[] # indices of entries, a list of lists
    values=[]  # values to be add to S
    for ele in range(nele):
        for inod in range(nloc):
            glb_inod = ele*nloc+inod 
            indices.append([glb_inod,ele])
            values.append(R[inod].cpu().numpy())

    indices = np.asarray(indices)
    

    values = np.asarray(values)
    
    Rbig_scipy = coo_matrix((values, ( indices[:,0], indices[:,1] ) ), shape=(nonods, nele))
    Rbig_scipy = Rbig_scipy.tocsr()
    Rbig = torch.sparse_csr_tensor(crow_indices=torch.tensor(Rbig_scipy.indptr), \
        col_indices=torch.tensor(Rbig_scipy.indices), \
        values=Rbig_scipy.data, \
        size=(nonods, nele), \
        device=dev)
    
    SR = torch.sparse.mm(S,Rbig)
    Rbig = torch.transpose(Rbig, dim0=0, dim1=1)
    RSR = torch.sparse.mm(Rbig,SR)
    del SR

    diagRSR = torch.zeros((nele,1), device=dev, dtype=torch.float64)
    for ele in range(nele):
        diagRSR[ele,0] = RSR[ele,ele]
    return diagRSR, RSR, Rbig

def RSR_matrix_color(S,R,whichc,ncolor, fina, cola, ncola):
    '''
    calculate operator R on the first level coarse grid
    (one node per element) by coloring methotd.
    
    #input
    S : S on the fine grid, torch csr sparse matrix, (nonods, nonods)
    R : rotation matrix per element, (nonods)
    whichc : element color, (nele)
    ncolor : number of colors to go through, integer
    ncloa : number of NNZ in RAR, integer

    #output
    diagRAR : diagonal of RAR operator
    RAR : operator R on the level-1 coarse grid, torch csr sparse matrix, 
    (nele, nele)
    RTbig restrictor, torch csr sparse matrix(nele, nonods)
    '''
    indices=[] # indices of entries, a list of lists
    values=[]  # values to be add to S
    for ele in range(nele):
        for inod in range(nloc):
            glb_inod = ele*nloc+inod 
            indices.append([glb_inod,ele])
            values.append(R[inod].cpu().numpy())

    indices = np.asarray(indices)
    values = np.asarray(values)
    Rbig_scipy = coo_matrix((values, ( indices[:,0], indices[:,1] ) ), shape=(nonods, nele))
    Rbig_scipy = Rbig_scipy.tocsr()
    Rbig = torch.sparse_csr_tensor(crow_indices=torch.tensor(Rbig_scipy.indptr), \
        col_indices=torch.tensor(Rbig_scipy.indices), \
        values=Rbig_scipy.data, \
        size=(nonods, nele), \
        device=dev)
    RTbig = torch.transpose(Rbig, dim0=0, dim1=1)

    value = np.zeros(ncola) # NNZ entry values
    for color in range(1,ncolor+1):
        mask = (whichc == color) # 1 if true; 0 if false
        mask = torch.tensor(mask, device=dev, dtype=torch.float64)
        color_vec = torch.mv(Rbig, mask) # (nonods, 1)
        color_vec = torch.mv(S, color_vec) # (nonods, 1)
        color_vec = torch.mv(RTbig, color_vec) # (nele, 1)
        # add to value 
        for i in range(color_vec.shape[0]):
            for count in range(fina[i], fina[i+1]):
                j = cola[count]
                value[count] = value[count] + color_vec[i]*mask[j]

    RSR = torch.sparse_csr_tensor(crow_indices=fina, \
        col_indices=cola, \
        values=value, \
        size=(nele, nele), \
        device=dev)

    diagRSR = torch.zeros((nele,1), device=dev, dtype=torch.float64)
    for ele in range(nele):
        diagRSR[ele,0] = RSR[ele,ele]

    return diagRSR, RSR, RTbig