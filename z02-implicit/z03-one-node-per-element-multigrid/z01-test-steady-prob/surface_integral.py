import torch
import config 
import numpy as np
from scipy.sparse import coo_matrix 
from mesh_init import face_iloc,face_iloc2

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
def S_Minv_sparse(x_all, nbele, nbf, c_bc):
    # input:
    # x_all - numpy array (nonods, ndim)
    # nbele - list of neighbour element index
    # nbf - list of neighbour face index
    # c_bc - Dirichlet boundary values at boundary nodes, 0 otherwise. This is to impose Diri bcs weakly
    #        (nonods)
    
    # output:
    # diagS, diagonal of S, torch tensor (nonods) 
    # S - surface integral matrix, torch.sparse_csr_tensor
    # b_bc - rhs vector accounting for boundary conditions, torch tensor (nonods)
    
    b_bc = torch.zeros(nonods, dtype=torch.float64, device=dev)
    # S matrix 
    indices=[] # indices of entries, a list of lists
    values=[]  # values to be add to S
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
    np.savetxt('indices.txt',S.to_dense().cpu().numpy(),delimiter=',')
    np.savetxt('fina', S_scipy.indptr,delimiter=',')
    np.savetxt('cola', S_scipy.indices,delimiter=',')

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