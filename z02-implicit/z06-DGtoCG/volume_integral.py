import torch 
from torch.nn import Module 
import config
from config import sf_nd_nb
import numpy as np

import multi_grid
import shape_function
from surface_integral_mf import S_mf_one_batch

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
nface = config.nface


def get_residual_and_smooth_once(
        r0,
        c_i, c_n, c_bc, f):
    '''
    update residual, do block Jacobi smooth once, by batches.
    '''
    k = config.k
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0,nnn+1)/nnn*nele, dtype=int)
    # diagA = torch.zeros_like(r0, device=dev, dtype=torch.float64)
    # bdiagA = torch.zeros(nele, nloc, nloc, device=dev, dtype=torch.float64)
    for i in range(nnn):
        # volume integral
        idx_in = np.zeros(nele, dtype=bool)
        idx_in[brk_pnt[i]:brk_pnt[i+1]] = True
        batch_in = np.sum(idx_in)
        diagA = torch.zeros(batch_in, nloc, device=dev, dtype=torch.float64)
        bdiagA = torch.zeros(batch_in, nloc, nloc, device=dev, dtype=torch.float64)
        r0, diagA, bdiagA = K_mf_one_batch(r0, c_i, c_n, f, k, dt,
                                           diagA, bdiagA,
                                           idx_in)
        # surface integral
        idx_in_f = np.zeros(nele * nface, dtype=bool)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0, diagA, bdiagA = S_mf_one_batch(r0, c_i, c_bc,
                                           diagA, bdiagA,
                                           idx_in_f, brk_pnt[i])
        # one smooth step
        if config.blk_solver == 'direct':
            bdiagA = torch.inverse(bdiagA)
            c_i = c_i.view(nele, nloc)
            c_i[idx_in, :] += config.jac_wei * torch.einsum('...ij,...j->...i',
                                                            bdiagA,
                                                            r0.view(nele, nloc)[idx_in, :])
            # c_i = c_i.view(-1, 1, nloc)
        if config.blk_solver == 'jacobi':
            new_b = torch.einsum('...ij,...j->...i', bdiagA, c_i.view(nele, nloc)[idx_in, :])\
                    + config.jac_wei * r0.view(nele, nloc)[idx_in, :]
            new_b = new_b.view(-1)  # batch_in * nloc
            diagA = diagA.view(-1)  # batch_in * nloc
            c_i = c_i.view(nele, nloc)
            c_i_partial = c_i[idx_in, :]
            for its in range(3):
                c_i_partial += ((new_b - torch.einsum('...ij,...j->...i',
                                                       bdiagA,
                                                       c_i_partial).view(-1))
                                   / diagA).view(-1, nloc)
            c_i[idx_in, :] = c_i_partial.view(-1, nloc)
        if config.blk_solver == 'none':
            # then use point jacobi iteration
            c_i = c_i.view(nele, nloc)
            c_i[idx_in, :] += config.jac_wei * r0.view(nele, nloc)[idx_in, :] / diagA
    r0 = r0.view(-1)
    c_i = c_i.view(-1)

    return r0, c_i


def get_residual_only(
        r0,
        c_i, c_n, c_bc, f):
    '''
    update residual, do block Jacobi smooth once, by batches.
    '''
    k = config.k
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0,nnn+1)/nnn*nele, dtype=int)
    # diagA = torch.zeros_like(r0, device=dev, dtype=torch.float64)
    # bdiagA = torch.zeros(nele, nloc, nloc, device=dev, dtype=torch.float64)
    for i in range(nnn):
        # volume integral
        idx_in = np.zeros(nele, dtype=bool)
        idx_in[brk_pnt[i]:brk_pnt[i+1]] = True
        batch_in = np.sum(idx_in)
        # here diagA and bdiagA are dummy variables since we won't need them to update c_i.
        diagA = torch.zeros(batch_in, nloc, device=dev, dtype=torch.float64)
        bdiagA = torch.zeros(batch_in, nloc, nloc, device=dev, dtype=torch.float64)
        r0, diagA, bdiagA = K_mf_one_batch(r0, c_i, c_n, f, k, dt,
                                           diagA, bdiagA,
                                           idx_in)
        # surface integral
        idx_in_f = np.zeros(nele * nface, dtype=bool)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0, diagA, bdiagA = S_mf_one_batch(r0, c_i, c_bc,
                                           diagA, bdiagA,
                                           idx_in_f, brk_pnt[i])
    r0 = r0.view(-1)
    return r0


def K_mf(r0, c_i, c_n, k, dt, n, nlx, x_ref_in, weight):
    '''
    update residual:     r0 <- r0 - K*c_i
    if transient, also do:     r0 <- r0 + M/dt * c_n

    # input
    c_i - node values at last jacobi iteration, i.e. c^n_i, (batch_size, 1, nloc)
    c_n  - node values at last timestep, i.e. c^(n-1)_i, (batch_size, 1, nloc)
    k  - diffusion coefficient at node, right now we simplify it using constant k. (1)
    dt - timestep. (1)
    n  - shape function Ni, (ngi, nloc)
    nlx : torch tensor ()
        shape function derivative on reference element
    x_ref_in : torch tensor ()
        nodes coordinates
    weight : torch tensor (ngi,)
        volume integral quadrature weights

    # output
    bdiagA - block diagonal of matrix A (batch_size, nloc, nloc)
    diagA - diagonal of lhs matrix A_1 (M/dt + K), (batch_size, nloc)
    r1  - part of residual b-A_1 c_i, (batch_size, 1, nloc)
    '''
    diagA = torch.zeros_like(r0, device=dev, dtype=torch.float64)
    bdiagA = torch.zeros(nele, nloc, nloc, device=dev, dtype=torch.float64)
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0,nnn+1)/nnn*nele, dtype=int)
    for i in range(nnn):
        idx_in = np.zeros(nele, dtype=bool)
        idx_in[brk_pnt[i]:brk_pnt[i+1]] = True
        r0, diagA, bdiagA = K_mf_one_batch(r0, c_i, c_n, k, dt,
                                           diagA, bdiagA,
                                           idx_in,
                                           n, nlx, x_ref_in, weight)

    return bdiagA, diagA, r0


def K_mf_one_batch(r0, c_i, c_n, f, k, dt,
                   diagA, bdiagA,
                   idx_in):
    # get essential data
    n = sf_nd_nb.n; nlx = sf_nd_nb.nlx
    x_ref_in = sf_nd_nb.x_ref_in
    weight = sf_nd_nb.weight

    batch_in = idx_in.shape[0]
    # change view
    r0 = r0.view(-1, nloc)
    c_i = c_i.view(-1, nloc)
    diagA = diagA.view(-1, nloc)
    bdiagA = bdiagA.view(-1, nloc, nloc)
    # get shape function derivatives
    nx, detwei = shape_function.get_det_nlx_3d(nlx, x_ref_in[idx_in], weight)
    nxnx = torch.zeros(batch_in, nloc, nloc, device=dev, dtype=torch.float64)
    # stiffness matrix
    for idim in range(ndim):
        nxnx += torch.einsum('...ig,...jg,...g->...ij', nx[:, idim, :, :], nx[:, idim, :, :], detwei)
    del nx
    nxnx *= k  # scalar multiplication, (batch_in, nloc, nloc)

    nn = torch.einsum('ig,jg,...g->...ij', n, n, detwei)
    f = f.view(nele, nloc)
    r0[idx_in, ...] += torch.einsum('...ij,...j->...i', nn, f[idx_in, ...])
    if config.isTransient:
        print('I go to transient...')
        nxnx = nn/dt + nxnx  # this is (M/dt + K), (batch_in, nloc, nloc)
        c_n = c_n.view(-1, nloc)
        r0[idx_in, ...] += torch.einsum('...ij,...j->...i', nn/dt, c_n[idx_in, ...])  # (batch_in, nloc)
    r0[idx_in, ...] -= torch.einsum('...ij,...j->...i', nxnx, c_i[idx_in, ...])  # (batch_in, nloc)
    diagA += torch.diagonal(nxnx, offset=0, dim1=-2, dim2=-1)  # (batch_in, nloc)
    bdiagA += nxnx  # (batch_in, nloc, nloc)
    return r0, diagA, bdiagA


# mass and stifness operator on level 1 coarse grid (per element condensation)
class mk_lv1(Module):
    def __init__(self):
        super(mk_lv1, self).__init__()

    def forward(self, e_i, r1_1, k, dt, n, nx, detwei, R):
        ### input
        # e_i - correction at last jacobi iteration, (batch_size, 1)
        # r1_1  - residual passed from fine mesh, (batch_size, 1)
        # k  - diffusion coefficient at node, right now we simplify it using constant k. (1)
        # dt - timestep. (1)
        # n  - shape function Ni, (ngi, nloc)
        # nx - shape function derivatives Nix & Niy, (batch_size, ndim, ngi, nloc)
        # detwei - determinant times GI weight, (batch_size, ngi)
        # R - rotation matrix (restrictor), (nloc)
        ### output
        # diagRAR - diagonal of lhs matrix A_1 (=M/dt + K) restricted by R, (batch_size, 1)
        # rr1  - residual of residual equation r1_1 - A_1 e_i, (batch_size, 1)

        batch_in = e_i.shape[0]
        # stiffness matrix 
        # nx1nx1 = torch.mul(nx[:,0,:,:].view(batch_in, ngi, nloc), \
        #     detwei.unsqueeze(-1).expand(batch_in, ngi, nloc)) # (batch_in, ngi, nloc)
        # nx1nx1 = torch.bmm(torch.transpose(nx[:,0,:,:].view(batch_in, ngi, nloc), 1,2), \
        #     nx1nx1) # (batch_in, nloc, nloc)
        # # print('nx1nx1',nx1nx1)
        # nx2nx2 = torch.mul(nx[:,1,:,:].view(batch_in, ngi, nloc), \
        #     detwei.unsqueeze(-1).expand(batch_in, ngi, nloc)) # (batch_in, ngi, nloc)
        # nx2nx2 = torch.bmm(torch.transpose(nx[:,1,:,:].view(batch_in, ngi, nloc), 1,2), \
        #     nx2nx2) # (batch_in, nloc, nloc)
        nx1nx1 = torch.einsum('...ig,...jg,...g->...ij', nx[:, 0, :, :], nx[:, 0, :, :], detwei)
        nx2nx2 = torch.einsum('...ig,...jg,...g->...ij', nx[:, 1, :, :], nx[:, 1, :, :], detwei)
        del nx
        nxnx = (nx1nx1+nx2nx2)*k # scalar multiplication, (batch_in, nloc, nloc)
        del nx1nx1 , nx2nx2 

        # print('nxnx', nxnx)
        # mass matrix
        # nn = torch.mul(n.unsqueeze(0).expand(batch_in, ngi, nloc), \
        #     detwei.unsqueeze(-1).expand(batch_in, ngi, nloc))   # (batch_in, ngi, nloc)
        # nn = torch.bmm(torch.transpose(n,0,1).unsqueeze(0).expand(batch_in, nloc, ngi), \
        #     nn) # (batch_in, nloc, nloc)
        nn = torch.einsum('ig,jg,...g->...ij', n, n, detwei).contiguous()
        # for ele in range(batch_in):
        #     np.savetxt('nn'+str(ele)+'.txt',nn[ele,:,:].view(nloc,nloc)/dt,delimiter=',')
        
        if (config.isTransient) :
            nxnx = nn/dt + nxnx # this is (M/dt + K), (batch_in, nloc, nloc)
        
        diagRAR = torch.bmm(nxnx, R.view(nloc,1).unsqueeze(0).expand(batch_in, nloc, 1)) # (batch_in, nloc, 1)
        diagRAR = torch.bmm(R.view(1,nloc).unsqueeze(0).expand(batch_in, 1, nloc), diagRAR) # (batch_in, 1, 1)
        
        rr1 = r1_1 - torch.mul(diagRAR.view(batch_in,-1), e_i.view(batch_in,-1))
        
        return diagRAR.view(batch_in,1), rr1.view(batch_in,1)


def calc_RKR(n, nx, detwei, R, k=1,dt=1):
    '''
    Calculate RKR matrix (diagonal) on level-1 coarse grid (P0DG).
    Can also include M/dt if transient.

    # Input

    n : torch tensor, (nloc, ngi)
        shape function N_i
    nx : torch tensor, (nele, ndim, nloc, ngi)
        derivative of shape functions dN_i/dx
    detwei : torch tensor, (nele, ngi)
        determinant x weights
    R : torch tensor, (nloc, 1)
        restrictor operator
    k : scalar
        diffusion coefficient
    dt : scalar
        time step (s)

    # output
    RKR : torch tensor, (nele,1)
        if transient: R*(K+M/dt)*R;
        if stable: R*K*R
    '''

    batch_in = nx.shape[0]
    # stiffness matrix 
    # nx1nx1 = torch.mul(nx[:,0,:,:].view(batch_in, ngi, nloc), \
    #     detwei.unsqueeze(-1).expand(batch_in, ngi, nloc)) # (batch_in, ngi, nloc)
    # nx1nx1 = torch.bmm(torch.transpose(nx[:,0,:,:].view(batch_in, ngi, nloc), 1,2), \
    #     nx1nx1) # (batch_in, nloc, nloc)
    # # print('nx1nx1',nx1nx1)
    # nx2nx2 = torch.mul(nx[:,1,:,:].view(batch_in, ngi, nloc), \
    #     detwei.unsqueeze(-1).expand(batch_in, ngi, nloc)) # (batch_in, ngi, nloc)
    # nx2nx2 = torch.bmm(torch.transpose(nx[:,1,:,:].view(batch_in, ngi, nloc), 1,2), \
    #     nx2nx2) # (batch_in, nloc, nloc)
    nx1nx1 = torch.einsum('...ig,...jg,...g->...ij', nx[:, 0, :, :], nx[:, 0, :, :], detwei)
    nx2nx2 = torch.einsum('...ig,...jg,...g->...ij', nx[:, 1, :, :], nx[:, 1, :, :], detwei)
    del nx
    nxnx = (nx1nx1+nx2nx2)*k # scalar multiplication, (batch_in, nloc, nloc)
    del nx1nx1 , nx2nx2 

    # print('nxnx', nxnx)
    # mass matrix
    # nn = torch.mul(n.unsqueeze(0).expand(batch_in, nloc, ngi), \
    #     detwei.unsqueeze(1).expand(batch_in, nloc, ngi))   # (batch_in, nloc, ngi)
    # nn = torch.bmm(torch.transpose(n,0,1).unsqueeze(0).expand(batch_in, nloc, ngi), \
    #     nn) # (batch_in, nloc, nloc)
    nn = torch.einsum('ig,jg,...g->...ij', n, n, detwei)
    # for ele in range(batch_in):
    #     np.savetxt('nn'+str(ele)+'.txt',nn[ele,:,:].view(nloc,nloc)/dt,delimiter=',')
    
    if (config.isTransient) :
        nxnx = nn/dt + nxnx # this is (M/dt + K), (batch_in, nloc, nloc)
    # print(nxnx)
    RKR = torch.bmm(nxnx, R.view(nloc,1).unsqueeze(0).expand(batch_in, nloc, 1)) # (batch_in, nloc, 1)
    RKR = torch.bmm(R.view(1,nloc).unsqueeze(0).expand(batch_in, 1, nloc), RKR) # (batch_in, 1, 1)

    return RKR.view(batch_in)

def calc_RAR(RKR, RSR, diagRSR):
    '''
    Calulate the sum of RKR and RSR to get RAR

    # Input:
    RKR : torch tensor, (nele)
        R*K*R, or R*(K+M/dt)*R if transient
    RSR : torch sparse tensor, (nele, nele)
        R*S*R

    # output:
    RAR : torch sparse tensor, (nele, nele)
        R*A*R, operator on (P0DG) coarse grid
    diagRAR : torch tensor, (nele)
        diagonal of RAR
    '''
    nele = config.nele
    diagRAR = diagRSR.view(-1) + RKR
    fina = RSR.crow_indices() 
    cola = RSR.col_indices()
    values = RSR.values()
    for ele in range(nele):
        for spIdx in range(fina[ele], fina[ele+1]):
            if (cola[spIdx]==ele) :
                values[spIdx] += RKR[ele]
    RAR = torch.sparse_csr_tensor(fina, cola, values, size=[nele,nele])
    # RAR = RSR
    
    return RAR, diagRAR


def RKR_DG_to_CG(n, nx, detwei, I_fc, I_cf, k=1, dt=1):
    '''
    Compute RKR, where R is from P1DG to P1CG

    Input
    -----
    n : torch tensor (nloc, ngi)
        shape function on reference element
    nx : torch tensor (nele, ndim, nloc, ngi)
        shape func derivative
    detwei : torch tensor (nele, ngi)
        determinant times quadrature weight
    I_fc : torch csr tensor (nonods, cg_nonods)
        prolongator from P1CG to P1DG
    I_cf : torch csr tensor (cg_nonods, nonods)
        restrictor from P1DG to P1CG
    k : scaler
        diffusion coefficient
    dt : scaler
        time step

    Output
    ------
    diagRKR : torch tensor (cg_nonods)
        diagonal of I_cf * K * I_fc
    RKR : torch csr tensor (cg_nonods, cg_nonods)
    '''
    K = torch.zeros(nele, nloc, nloc, device=dev, dtype=torch.float64)
    if config.isTransient:
        K += torch.einsum('ig,jg,...g->...ij', n, n, detwei)/dt
    for idim in range(config.ndim):
        K += torch.einsum('...ig,...jg,...g->...ij', nx[:,idim,:,:], nx[:,idim,:,:], detwei)*k
    # Transform K to scr
    K_fina = torch.arange(0, nloc*nonods+1, nloc)
    K_cola = torch.einsum('ij,k->ikj',
                          torch.arange(0,nonods, dtype=torch.long).view(nele,nloc),
                          torch.ones(nloc, dtype=torch.long)).contiguous().view(-1)
    K_csr = torch.sparse_csr_tensor(crow_indices=K_fina,
                                    col_indices=K_cola,
                                    values=K.view(-1),
                                    device=dev, dtype=torch.float64, size=(nonods, nonods))

    RKR = torch.sparse.mm(K_csr, I_fc)
    RKR = torch.sparse.mm(I_cf, RKR)
    diagRKR = torch.zeros(config.cg_nonods, device=dev, dtype=torch.float64)
    for inod in range(config.cg_nonods):
        diagRKR[inod] = RKR[inod, inod]
    return diagRKR, RKR


def RKR_DG_to_CG_color(n, nx, detwei, I_fc, I_cf,
                       whichc, ncolor,
                       fina, cola, ncola,
                       k=1, dt=1):
    '''
    Compute RKR, where R is from P1DG to P1CG
    via coloring probing method.

    Input
    -----
    n : torch tensor (nloc, ngi)
        shape function on reference element
    nx : torch tensor (nele, ndim, nloc, ngi)
        shape func derivative
    detwei : torch tensor (nele, ngi)
        determinant times quadrature weight
    I_fc : torch csr tensor (nonods, cg_nonods)
        prolongator from P1CG to P1DG
    I_cf : torch csr tensor (cg_nonods, nonods)
        restrictor from P1DG to P1CG
    k : scaler
        diffusion coefficient
    dt : scaler
        time step

    Output
    ------
    diagRKR : torch tensor (cg_nonods)
        diagonal of I_cf * K * I_fc
    RKR : torch csr tensor (cg_nonods, cg_nonods)
    '''
    K = torch.zeros(nele, nloc, nloc, device=dev, dtype=torch.float64)
    if config.isTransient:
        K += torch.einsum('ig,jg,...g->...ij', n, n, detwei)/dt
    for idim in range(config.ndim):
        K += torch.einsum('...ig,...jg,...g->...ij', nx[:,idim,:,:], nx[:,idim,:,:], detwei)*k
    # Transform K to scr
    K_fina = torch.arange(0, nloc*nonods+1, nloc)
    K_cola = torch.einsum('ij,k->ikj',
                          torch.arange(0,nonods, dtype=torch.long).view(nele,nloc),
                          torch.ones(nloc, dtype=torch.long)).contiguous().view(-1)
    K_csr = torch.sparse_csr_tensor(crow_indices=K_fina,
                                    col_indices=K_cola,
                                    values=K.view(-1),
                                    device=dev, dtype=torch.float64, size=(nonods, nonods))

    # RKR = torch.sparse.mm(K_csr, I_fc)
    # RKR = torch.sparse.mm(I_cf, RKR)
    cg_nonods = config.cg_nonods
    value = torch.zeros(ncola, device=dev, dtype=torch.float64)  # NNZ entry values
    for color in range(1,ncolor+1):
        mask = (whichc == color) # 1 if true; 0 if false
        mask = torch.tensor(mask, device=dev, dtype=torch.float64)
        print('color: ', color)
        Rm = torch.mv(I_fc, mask) # (nonods, 1)
        KRm = torch.mv(K_csr, Rm) # (nonods, 1)
        RKRm = torch.mv(I_cf, KRm) # (cg_nonods, 1)
        # add to value
        for i in range(RKRm.shape[0]):
            for count in range(fina[i], fina[i+1]):
                j = cola[count]
                value[count] += RKRm[i]*mask[j]

    RKR = torch.sparse_csr_tensor(crow_indices=fina,
                                  col_indices=cola,
                                  values=value,
                                  size=(cg_nonods, cg_nonods),
                                  device=dev)

    diagRKR = torch.zeros(config.cg_nonods, device=dev, dtype=torch.float64)
    for inod in range(config.cg_nonods):
        diagRKR[inod] = RKR[inod, inod]

    return diagRKR, RKR


def RAR_DG_to_CG(diagRKR, RKR, diagRSR, RSR):
    '''
    add RSR and RAR
    '''
    RAR = RSR + RKR
    del RSR, RKR
    diagRAR = diagRSR + diagRKR
    del diagRSR, diagRKR
    return diagRAR, RAR


def calc_RAR_mf_color(I_fc, I_cf,
                      whichc, ncolor,
                      fina, cola, ncola):
    '''
    get RAR matrix-freely via coloring method

    # Input:
    Volume shape functions:
        n, nx, detwei
    Surface shape functions:
        sn, snx, setwei, snormal
    Neighbouring info for surface integral:
        nbele, nbf
    Restrictor/prolongator: (p1dg <-> p1dg)
        I_cf, I_fc
    Coloring:
        whichc, color
    RAR sparsity:
        fina, cola, ncola

    # Output:
    RAR: torch csr tensor, (cg_nonods, cg_nonods)
        I_cf * A * I_fc
    '''
    import surface_integral_mf
    import time
    start_time = time.time()
    cg_nonods = sf_nd_nb.cg_nonods
    value = torch.zeros(ncola, device=dev, dtype=torch.float64)  # NNZ entry values
    dummy = torch.zeros(nonods, device=dev, dtype=torch.float64)  # dummy variable of same length as PnDG
    ARm = torch.zeros(nonods, device=dev, dtype=torch.float64)
    # nx, detwei = Det_nlx.forward(x_ref_in, weight)
    for color in range(1, ncolor + 1):
        mask = (whichc == color)  # 1 if true; 0 if false
        mask = torch.tensor(mask, device=dev, dtype=torch.float64)
        print('color: ', color)
        Rm = torch.mv(I_fc, mask)  # (p1dg_nonods, 1)
        Rm = multi_grid.p1dg_to_p3dg_prolongator(Rm)  # (p3dg_nonods, )
        ARm *= 0
        ARm = get_residual_only(ARm,
                                Rm, dummy, dummy, dummy)
        ARm *= -1.  # (p3dg_nonods, )
        RARm = multi_grid.p3dg_to_p1dg_restrictor(ARm)  # (p1dg_nonods, )
        RARm = torch.mv(I_cf, RARm)  # (cg_nonods, 1)
        # add to value
        for i in range(RARm.shape[0]):
            for count in range(fina[i], fina[i + 1]):
                j = cola[count]
                value[count] += RARm[i] * mask[j]
        print('finishing (another) one color, time comsumed: ', time.time()-start_time)

    RAR = torch.sparse_csr_tensor(crow_indices=fina,
                                  col_indices=cola,
                                  values=value,
                                  size=(cg_nonods, cg_nonods),
                                  device=dev)
    return RAR
