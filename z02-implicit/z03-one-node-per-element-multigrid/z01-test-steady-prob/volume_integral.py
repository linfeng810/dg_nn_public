import torch 
from torch.nn import Module 
import config 
import numpy as np

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

# assemble mass and stiffness matrix
# 
# equation:
# (M/dt + K)c^n + S c^n = (M/dt)c^(n-1)
# 
#    A_1    c^n + A_2 c^n =   b
# 
# to form a Jacobi iteration, we need:
# 1) diag(A) = diag(A_1) + diag(A_2)
# 2) residual r = b - A_1 c^n_i - A_2 c^n_i
# 
# in this subroutine we will calculate:
# diag(A_1)
# first part of residual r_1 = b - A_1 c^n_i
class mk(Module):
    def __init__(self):
        super(mk, self).__init__()

    def forward(self, c_i, c_n, b_bc, k, dt, n, nx, detwei):
        ### input
        # c_i - node values at last jacobi iteration, i.e. c^n_i, (batch_size, 1, nloc)
        # c_n  - node values at last timestep, i.e. c^(n-1)_i, (batch_size, 1, nloc)
        # b_bc - boundary condition contributions to the rhs, (batch_size, 1, nloc)
        # k  - diffusion coefficient at node, right now we simplify it using constant k. (1)
        # dt - timestep. (1)
        # n  - shape function Ni, (ngi, nloc)
        # nx - shape function derivatives Nix & Niy, (batch_size, ndim, ngi, nloc)
        # detwei - determinant times GI weight, (batch_size, ngi)
        ### output
        # diagA - diagonal of lhs matrix A_1 (M/dt + K), (batch_size, nloc)
        # r1  - part of residual b-A_1 c_i, (batch_size, 1, nloc)

        batch_in = c_i.shape[0]
        # stiffness matrix 
        nx1nx1 = torch.mul(nx[:,0,:,:].view(batch_in, ngi, nloc), \
            detwei.unsqueeze(-1).expand(batch_in, ngi, nloc)) # (batch_in, ngi, nloc)
        nx1nx1 = torch.bmm(torch.transpose(nx[:,0,:,:].view(batch_in, ngi, nloc), 1,2), \
            nx1nx1) # (batch_in, nloc, nloc)
        # print('nx1nx1',nx1nx1)
        nx2nx2 = torch.mul(nx[:,1,:,:].view(batch_in, ngi, nloc), \
            detwei.unsqueeze(-1).expand(batch_in, ngi, nloc)) # (batch_in, ngi, nloc)
        nx2nx2 = torch.bmm(torch.transpose(nx[:,1,:,:].view(batch_in, ngi, nloc), 1,2), \
            nx2nx2) # (batch_in, nloc, nloc)
        del nx
        nxnx = (nx1nx1+nx2nx2)*k # scalar multiplication, (batch_in, nloc, nloc)
        del nx1nx1 , nx2nx2 
        
        # print('nxnx', nxnx)
        # mass matrix
        nn = torch.mul(n.unsqueeze(0).expand(batch_in, ngi, nloc), \
            detwei.unsqueeze(-1).expand(batch_in, ngi, nloc))   # (batch_in, ngi, nloc)
        nn = torch.bmm(torch.transpose(n,0,1).unsqueeze(0).expand(batch_in, nloc, ngi), \
            nn) # (batch_in, nloc, nloc)
        # for ele in range(batch_in):
        #     np.savetxt('nn'+str(ele)+'.txt',nn[ele,:,:].view(nloc,nloc)/dt,delimiter=',')
        
        b = torch.zeros(batch_in, nloc, 1, device=dev, dtype=torch.float64)
        b = b + b_bc.view(batch_in, nloc,1)
        if (config.isTransient) :
            nxnx = nn/dt + nxnx # this is (M/dt + K), (batch_in, nloc, nloc)
        
            b = b + torch.matmul(nn/dt,torch.transpose(c_n,1,2)) # batch matrix-vector multiplication, 
                        # input1: (batch_in, nloc, nloc)
                # input2: (batch_in, nloc, 1)
                # broadcast over batch
                # output: (batch_in, nloc, 1)
        
        r1 = torch.matmul(nxnx, torch.transpose(c_i,1,2)) # batch m-v multiplication of (M/dt+K)*c_i
        
        r1 = b-r1 
        r1 = torch.transpose(r1,1,2) # return to (batch_in, 1, nloc)
        
        diagA = torch.diagonal(nxnx, offset=0, dim1=-2, dim2=-1).contiguous()  # use continuous thus diagonal are stored contiguously
           # otherwise by default diagonal returns memory position of diagonal in originally stored tensor
           # wouldn't be able to do e.g. .view
        
        return diagA, r1

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
        nx1nx1 = torch.mul(nx[:,0,:,:].view(batch_in, ngi, nloc), \
            detwei.unsqueeze(-1).expand(batch_in, ngi, nloc)) # (batch_in, ngi, nloc)
        nx1nx1 = torch.bmm(torch.transpose(nx[:,0,:,:].view(batch_in, ngi, nloc), 1,2), \
            nx1nx1) # (batch_in, nloc, nloc)
        # print('nx1nx1',nx1nx1)
        nx2nx2 = torch.mul(nx[:,1,:,:].view(batch_in, ngi, nloc), \
            detwei.unsqueeze(-1).expand(batch_in, ngi, nloc)) # (batch_in, ngi, nloc)
        nx2nx2 = torch.bmm(torch.transpose(nx[:,1,:,:].view(batch_in, ngi, nloc), 1,2), \
            nx2nx2) # (batch_in, nloc, nloc)
        del nx
        nxnx = (nx1nx1+nx2nx2)*k # scalar multiplication, (batch_in, nloc, nloc)
        del nx1nx1 , nx2nx2 

        # print('nxnx', nxnx)
        # mass matrix
        nn = torch.mul(n.unsqueeze(0).expand(batch_in, ngi, nloc), \
            detwei.unsqueeze(-1).expand(batch_in, ngi, nloc))   # (batch_in, ngi, nloc)
        nn = torch.bmm(torch.transpose(n,0,1).unsqueeze(0).expand(batch_in, nloc, ngi), \
            nn) # (batch_in, nloc, nloc)
        # for ele in range(batch_in):
        #     np.savetxt('nn'+str(ele)+'.txt',nn[ele,:,:].view(nloc,nloc)/dt,delimiter=',')
        
        if (config.isTransient) :
            nxnx = nn/dt + nxnx # this is (M/dt + K), (batch_in, nloc, nloc)
        
        diagRAR = torch.bmm(nxnx, R.view(nloc,1).unsqueeze(0).expand(batch_in, nloc, 1)) # (batch_in, nloc, 1)
        diagRAR = torch.bmm(R.view(1,nloc).unsqueeze(0).expand(batch_in, 1, nloc), diagRAR) # (batch_in, 1, 1)
        
        rr1 = r1_1 - torch.mul(diagRAR.view(batch_in,-1), e_i.view(batch_in,-1))
        
        return diagRAR.view(batch_in,1), rr1.view(batch_in,1)