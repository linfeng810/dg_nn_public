import torch 
from torch.nn import Module 
import config
from config import sf_nd_nb
import numpy as np

import multi_grid
from surface_integral_mf import S_mf_one_batch
from types import NoneType
if config.ndim == 2:
    from shape_function import get_det_nlx as get_det_nlx
    from shape_function import sdet_snlx as sdet_snlx
else:
    from shape_function import get_det_nlx_3d as get_det_nlx
    from shape_function import sdet_snlx_3d as sdet_snlx

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
sngi = config.sngi
eta_e = config.eta_e


def get_residual_and_smooth_once(
        r0,
        c_i, c_n, c_bc, f, c_rhs=0):
    '''
    update residual, do block Jacobi smooth once, by batches.
    '''
    k = config.k
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0,nnn+1)/nnn*nele, dtype=int)
    # diagA = torch.zeros_like(r0, device=dev, dtype=torch.float64)
    # bdiagA = torch.zeros(nele, nloc, nloc, device=dev, dtype=torch.float64)
    # if type(c_rhs) != NoneType:
    r0 += c_rhs  # precalculated rhs
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
        c_i, c_n, c_bc, f, c_rhs=0):
    '''
    update residual, do block Jacobi smooth once, by batches.
    '''
    k = config.k
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0,nnn+1)/nnn*nele, dtype=int)
    # diagA = torch.zeros_like(r0, device=dev, dtype=torch.float64)
    # bdiagA = torch.zeros(nele, nloc, nloc, device=dev, dtype=torch.float64)
    # if type(c_rhs) != NoneType:
    r0 += c_rhs  # precalculated rhs
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

    batch_in = np.sum(idx_in)
    # change view
    r0 = r0.view(-1, nloc)
    c_i = c_i.view(-1, nloc)
    diagA = diagA.view(-1, nloc)
    bdiagA = bdiagA.view(-1, nloc, nloc)
    # get shape function derivatives
    nx, detwei = get_det_nlx(nlx, x_ref_in[idx_in], weight)
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


def pmg_get_residual_and_smooth_once(
        r0,
        c_i, po: int):
    """
    update residual, do block Jacobi smooth once, by batches.
    """
    k = config.k
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0,nnn+1)/nnn*nele, dtype=int)
    nloc_po = multi_grid.p_nloc(po)  # nloc at this p-level (po output p level)
    for i in range(nnn):
        # volume integral
        idx_in = np.zeros(nele, dtype=bool)
        idx_in[brk_pnt[i]:brk_pnt[i+1]] = True
        batch_in = np.sum(idx_in)
        diagA = torch.zeros(batch_in, nloc_po,
                            device=dev, dtype=torch.float64)
        bdiagA = torch.zeros(batch_in, nloc_po, nloc_po,
                             device=dev, dtype=torch.float64)
        r0, diagA, bdiagA = _pmg_k_mf_one_batch(r0, c_i,
                                                diagA, bdiagA,
                                                idx_in, po)
        # surface integral
        idx_in_f = np.zeros(nele * nface, dtype=bool)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0, diagA, bdiagA = _pmg_s_mf_one_batch(r0, c_i,
                                                diagA, bdiagA,
                                                idx_in_f, brk_pnt[i], po)
        # one smooth step
        if config.blk_solver == 'direct':
            bdiagA = torch.inverse(bdiagA.view(batch_in, nloc_po, nloc_po))
            c_i = c_i.view(nele, nloc_po)
            c_i[idx_in, :] += config.jac_wei * torch.einsum('...ij,...j->...i',
                                                            bdiagA,
                                                            r0.view(nele, nloc_po)[idx_in, :])
            # c_i = c_i.view(-1, 1, nloc)
        if config.blk_solver == 'jacobi':
            new_b = torch.einsum('...ij,...j->...i',
                                 bdiagA.view(batch_in, nloc_po, nloc_po),
                                 c_i.view(nele, nloc_po)[idx_in, :])\
                    + config.jac_wei * r0.view(nele, nloc_po)[idx_in, :]
            new_b = new_b.view(-1)  # batch_in * nloc
            diagA = diagA.view(-1)  # batch_in * nloc
            c_i = c_i.view(nele, nloc_po)
            c_i_partial = c_i[idx_in, :]
            for its in range(3):
                c_i_partial += ((new_b - torch.einsum('...ij,...j->...i',
                                                      bdiagA.view(batch_in,
                                                                  nloc_po,
                                                                  nloc_po),
                                                      c_i_partial).view(-1))
                                / diagA).view(-1, nloc_po)
            c_i[idx_in, :] = c_i_partial.view(-1, nloc_po)
        if config.blk_solver == 'none':
            # then use point jacobi iteration
            c_i = c_i.view(nele, nloc_po)
            c_i[idx_in, :] += config.jac_wei * r0.view(nele, nloc_po)[idx_in, :] / diagA
    r0 = r0.view(-1)
    c_i = c_i.view(-1)

    return r0, c_i


def pmg_get_residual_only(
        r0,
        c_i, po: int):
    """
    update residual, do block Jacobi smooth once, by batches.
    """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0,nnn+1)/nnn*nele, dtype=int)
    nloc_po = multi_grid.p_nloc(po)
    for i in range(nnn):
        # volume integral
        idx_in = np.zeros(nele, dtype=bool)
        idx_in[brk_pnt[i]:brk_pnt[i+1]] = True
        batch_in = np.sum(idx_in)
        # here diagA and bdiagA are dummy variables since we won't need them to update c_i.
        diagA = torch.zeros(batch_in, nloc_po, device=dev, dtype=torch.float64)
        bdiagA = torch.zeros(batch_in, nloc_po, nloc_po, device=dev, dtype=torch.float64)
        r0, diagA, bdiagA = _pmg_k_mf_one_batch(r0, c_i,
                                                diagA, bdiagA,
                                                idx_in, po)
        # surface integral
        idx_in_f = np.zeros(nele * nface, dtype=bool)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0, diagA, bdiagA = _pmg_s_mf_one_batch(r0, c_i,
                                                diagA, bdiagA,
                                                idx_in_f, brk_pnt[i], po)
    r0 = r0.view(-1)
    return r0


def _pmg_k_mf_one_batch(
        r0, c_i,
        diagA, bdiagA,
        idx_in, po: int):
    # get essential data
    n = sf_nd_nb.n; nlx = sf_nd_nb.nlx
    x_ref_in = sf_nd_nb.x_ref_in
    weight = sf_nd_nb.weight

    nloc_po = multi_grid.p_nloc(po)

    batch_in = idx_in.shape[0]
    # change view
    r0 = r0.view(-1, nloc_po)
    c_i = c_i.view(-1, nloc_po)
    diagA = diagA.view(-1, nloc_po)
    bdiagA = bdiagA.view(-1, nloc_po, nloc_po)
    # get shape function derivatives
    nx, detwei = get_det_nlx(nlx, x_ref_in[idx_in], weight)
    nxnx = torch.zeros(batch_in, nloc, nloc, device=dev, dtype=torch.float64)
    # stiffness matrix
    for idim in range(ndim):
        nxnx += torch.einsum('...ig,...jg,...g->...ij', nx[:, idim, :, :], nx[:, idim, :, :], detwei)
    del nx
    nxnx *= config.k  # scalar multiplication, (batch_in, nloc, nloc)

    nn = torch.einsum('ig,jg,...g->...ij', n, n, detwei)
    if config.isTransient:
        print('I go to transient...')
        nxnx = nn/dt + nxnx  # this is (M/dt + K), (batch_in, nloc, nloc)
    # get operator on this p-level
    nxnx = torch.einsum('pi,...ij,jq->...pq',
                        multi_grid.p_restrictor(p_in=config.ele_p, p_out=po),
                        nxnx,
                        multi_grid.p_prolongator(p_in=po, p_out=config.ele_p)).contiguous()
    # update residual -K*c_i
    r0[idx_in, ...] -= torch.einsum('...ij,...j->...i', nxnx, c_i[idx_in, ...])  # (batch_in, nloc)
    # get diagonal
    diagA += torch.diagonal(nxnx.view(batch_in, nloc_po, nloc_po),
                            offset=0, dim1=-2, dim2=-1).contiguous().view(batch_in, nloc_po)
    bdiagA += nxnx  # (batch_in, nloc, nloc)
    return r0, diagA, bdiagA


def _pmg_s_mf_one_batch(
        r, c_i,
        diagA, bdiagA,
        idx_in_f, batch_start_idx,
        po: int):
    # get essential data
    nbf = sf_nd_nb.nbf
    alnmt = sf_nd_nb.alnmt

    nloc_po = multi_grid.p_nloc(po)
    c_i = c_i.view(nele, nloc_po)
    r = r.view(nele, nloc_po)

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
        for nb_gi_aln in range(nface - 1):
            idx_iface = (f_i == iface) & (sf_nd_nb.alnmt[F_i] == nb_gi_aln)
            # idx_iface_all += idx_iface
            # idx_iface = idx_iface & idx_in_f
            r, diagA, bdiagA = _pmg_S_fi(
                r, f_i[idx_iface], E_F_i[idx_iface],
                f_inb[idx_iface], E_F_inb[idx_iface],
                c_i, diagA, bdiagA, batch_start_idx,
                nb_gi_aln, po)

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
    # FIXED: let's hope that each element has only one boundary face.
    #        This isn't the case in 3D!
    for iface in range(nface):
        idx_iface = f_b == iface
        r, diagA, bdiagA = _pmg_S_fb(r, f_b[idx_iface], E_F_b[idx_iface],
                                     c_i, diagA, bdiagA, batch_start_idx, po)

    return r, diagA, bdiagA


def _pmg_S_fi(
        r, f_i, E_F_i,
        f_inb, E_F_inb,
        c_i,
        diagS, bdiagS, batch_start_idx,
        nb_gi_aln,
        po: int):  # neighbour face gaussian points aliagnment
    '''
    this function add interior face S*c contribution
    to r
    '''
    # faces can be passed in by batches to fit memory/GPU cores
    batch_in = f_i.shape[0]
    if batch_in < 1:  # nothing to do here. return to what called me.
        return r, diagS, bdiagS
    dummy_idx = np.arange(0, batch_in)  # this is to use with idx f_i

    # get essential data
    sn = sf_nd_nb.sn
    snlx = sf_nd_nb.snlx
    x_ref_in = sf_nd_nb.x_ref_in
    sweight = sf_nd_nb.sweight

    nloc_po = multi_grid.p_nloc(po)

    # make all tensors in shape (nele, nface, ndim, nloc(inod), nloc(jnod), sngi)
    # all these expansion are views of original tensor,
    # i.e. they point to same memory as sn/snx
    sni = sn.unsqueeze(0).expand(batch_in, -1, -1, -1) \
        .unsqueeze(3).expand(-1, -1, -1, nloc, -1)  # expand on nloc(jnod)
    snj = sn.unsqueeze(0).expand(batch_in, -1, -1, -1) \
        .unsqueeze(2).expand(-1, -1, nloc, -1, -1)  # expand on nloc(inod)

    # get shape function derivatives
    # this side.
    snx, sdetwei, snormal = sdet_snlx(snlx, x_ref_in[E_F_i], sweight)
    # now tensor shape are:
    # snx | snx_nb         (batch_in, nface, ndim, nloc, sngi)
    # sdetwei | sdetwei_nb (batch_in, nface, sngi)
    # snormal | snormal_nb (batch_in, nface, ndim)
    mu_e = eta_e / torch.sum(sdetwei[dummy_idx, f_i, :], -1)
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
    mu_e = mu_e.unsqueeze(-1).unsqueeze(-1) \
        .expand(-1, nloc, nloc)

    # this side
    S = torch.zeros(batch_in, nloc, nloc,
                    device=dev, dtype=torch.float64)  # local S matrix
    # n_j nx_i
    for idim in range(ndim):
        S[:, :nloc, :nloc] += torch.sum(torch.mul(torch.mul(torch.mul(
            snj[dummy_idx, f_i, :, :, :],
            snxi[dummy_idx, f_i, idim, :, :, :]),
            snormalv[dummy_idx, f_i, idim, :, :, :]),
            sdetweiv[dummy_idx, f_i, :, :, :]),
            -1) * (-0.5)
    # njx ni
    for idim in range(ndim):
        S[:, :nloc, :nloc] += torch.sum(torch.mul(torch.mul(torch.mul(
            snxj[dummy_idx, f_i, idim, :, :, :],
            sni[dummy_idx, f_i, :, :, :]),
            snormalv[dummy_idx, f_i, idim, :, :, :]),
            sdetweiv[dummy_idx, f_i, :, :, :]),
            -1) * (-0.5)
    # nj ni
    S[:, :nloc, :nloc] += torch.mul(
        torch.sum(torch.mul(torch.mul(
            sni[dummy_idx, f_i, :, :, :],
            snj[dummy_idx, f_i, :, :, :]),
            sdetweiv[dummy_idx, f_i, :, :, :]),
            -1), mu_e)
    # get S on p order grid
    S = torch.einsum('pi,...ij,jq->...pq',
                     multi_grid.p_restrictor(p_in=config.ele_p, p_out=po),
                     S,
                     multi_grid.p_prolongator(p_in=po, p_out=config.ele_p)).contiguous()
    # multiply S and c_i and add to (subtract from) r
    r[E_F_i, :] -= torch.matmul(S, c_i[E_F_i, :].view(batch_in, nloc_po, 1)).squeeze()
    # put diagonal of S into diagS
    diagS[E_F_i - batch_start_idx, :] += torch.diagonal(S.view(batch_in, nloc_po, nloc_po),
                                                        dim1=-2, dim2=-1).view(batch_in, nloc_po)
    bdiagS[E_F_i - batch_start_idx, ...] += S

    # other side
    S = torch.zeros(batch_in, nloc, nloc,
                    device=dev, dtype=torch.float64)  # local S matrix
    # Nj2 * Ni1x * n2
    for idim in range(ndim):
        S[:, :nloc, :nloc] += torch.sum(torch.mul(torch.mul(torch.mul(
            snj_nb[dummy_idx, f_inb, :, :, :],
            snxi[dummy_idx, f_i, idim, :, :, :]),
            snormalv_nb[dummy_idx, f_inb, idim, :, :, :]),
            sdetweiv[dummy_idx, f_i, :, :, :]),
            -1) * (-0.5)
    # Nj2x * Ni1 * n1
    for idim in range(ndim):
        S[:, :nloc, :nloc] += torch.sum(torch.mul(torch.mul(torch.mul(
            snxj_nb[dummy_idx, f_inb, idim, :, :, :],
            sni[dummy_idx, f_i, :, :, :]),
            snormalv[dummy_idx, f_i, idim, :, :, :]),
            sdetweiv[dummy_idx, f_i, :, :, :]),
            -1) * (-0.5)
    # Nj2n2 * Ni1n1 ! n2 \cdot n1 = -1
    S[:, :nloc, :nloc] += torch.mul(
        torch.sum(torch.mul(torch.mul(
            sni[dummy_idx, f_i, :, :, :],
            snj_nb[dummy_idx, f_inb, :, :, :]),
            sdetweiv[dummy_idx, f_i, :, :, :]),
            -1), -mu_e)
    # this S is off-diagonal contribution, therefore no need to put in diagS
    # get S on p order grid
    S = torch.einsum('pi,...ij,jq->...pq',
                     multi_grid.p_restrictor(p_in=config.ele_p, p_out=po),
                     S,
                     multi_grid.p_prolongator(p_in=po, p_out=config.ele_p))
    # multiply S and c_i and add to (subtract from) r
    r[E_F_i, :] -= torch.matmul(S, c_i[E_F_inb, :].view(batch_in, nloc_po, 1)).squeeze()

    return r, diagS, bdiagS


def _pmg_S_fb(
        r, f_b, E_F_b,
        c_i,
        diagS, bdiagS, batch_start_idx, po: int):
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
    if batch_in < 1:  # there is nothing to do here. return to what called me.
        return r, diagS, bdiagS
    dummy_idx = np.arange(0, batch_in)

    # get essential data
    sn = sf_nd_nb.sn
    snlx = sf_nd_nb.snlx
    x_ref_in = sf_nd_nb.x_ref_in
    sweight = sf_nd_nb.sweight

    nloc_po = multi_grid.p_nloc(po)

    # make all tensors in shape (nele, nface, ndim, nloc(inod), nloc(jnod), sngi)
    # all these expansion are views of original tensor,
    # i.e. they point to same memory as sn/snx
    sni = sn.unsqueeze(0).expand(batch_in, -1, -1, -1) \
        .unsqueeze(3).expand(-1, -1, -1, nloc, -1)  # expand on nloc(jnod)
    snj = sn.unsqueeze(0).expand(batch_in, -1, -1, -1) \
        .unsqueeze(2).expand(-1, -1, nloc, -1, -1)  # expand on nloc(inod)

    # get shaps function derivatives
    snx, sdetwei, snormal = sdet_snlx(snlx, x_ref_in[E_F_b], sweight)
    mu_e = eta_e / torch.sum(sdetwei[dummy_idx, f_b, :], -1)
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

    S = torch.zeros(batch_in, nloc, nloc,
                    device=dev, dtype=torch.float64)  # local S matrix
    # # Nj1 * Ni1x * n1
    for idim in range(ndim):
        S[:, :nloc, :nloc] -= torch.sum(torch.mul(torch.mul(torch.mul(
            snj[dummy_idx, f_b, :, :, :],
            snxi[dummy_idx, f_b, idim, :, :, :]),
            snormalv[dummy_idx, f_b, idim, :, :, :]),
            sdetweiv[dummy_idx, f_b, :, :, :]),
            -1)  # *(-1.0)
    # # Nj1n1 * Ni1n1 ! n1 \cdot n1 = 1
    S[:, :nloc, :nloc] += torch.mul(
        torch.sum(torch.mul(torch.mul(
            sni[dummy_idx, f_b, :, :, :],
            snj[dummy_idx, f_b, :, :, :]),
            sdetweiv[dummy_idx, f_b, :, :, :]),
            -1), mu_e)
    # # Nj1x * Ni1 * n1
    for idim in range(ndim):
        S[:, :nloc, :nloc] -= torch.sum(torch.mul(torch.mul(torch.mul(
            sni[dummy_idx, f_b, :, :, :],
            snxj[dummy_idx, f_b, idim, :, :, :]),
            snormalv[dummy_idx, f_b, idim, :, :, :]),
            sdetweiv[dummy_idx, f_b, :, :, :]),
            -1)  # *(-1.0)
    # get S on p-order grid
    S = torch.einsum('pi,...ij,jq->...pq',
                     multi_grid.p_restrictor(p_in=config.ele_p, p_out=po),
                     S,
                     multi_grid.p_prolongator(p_in=po, p_out=config.ele_p))
    # calculate S*c and add to (subtract from) r
    r[E_F_b, :] -= torch.matmul(S, c_i[E_F_b, :].view(batch_in, nloc_po, 1)).squeeze()
    diagS[E_F_b - batch_start_idx, :] += torch.diagonal(S.view(batch_in, nloc_po, nloc_po),
                                                        dim1=-2, dim2=-1).view(batch_in, nloc_po)
    bdiagS[E_F_b - batch_start_idx, ...] += S

    return r, diagS, bdiagS
