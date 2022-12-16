import torch 
from torch.nn import Module 
import config 

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
# 1/dt M( c^(n*) - c^(n-1) ) + K c^(n-1) = 0
# ==>
# M c^(n*) = (M-dt*K) c^(n-1)
# to form a Jacobi iteration, here we calculate 
# M and b = (M-dt*K) c^(n-1)
class mk(Module):
    def __init__(self):
        super(mk, self).__init__()

    def forward(self, c, k, dt, n, nx, detwei):
        ### input
        # c  - node values at last timestep, (batch_size, 1, nloc)
        # k  - diffusion coefficient at node, right now we simplify it using constant k. (1)
        # dt - timestep. (1)
        # n  - shape function Ni, (ngi, nloc)
        # nx - shape function derivatives Nix & Niy, (batch_size, ndim, ngi, nloc)
        # detwei - determinant times GI weight, (batch_size, ngi)
        ### output
        # nn - mass matrix (consistent), (batch_size, nloc, nloc)
        # b  - rhs of Mc=b, (batch_size, 1, nloc)

        batch_in = c.shape[0]
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
        # for ele in range(batch_in):
        #     np.savetxt('nxnx'+str(ele)+'.txt',nxnx[ele,:,:].view(nloc,nloc),delimiter=',')
        
        # print('nxnx', nxnx)
        # mass matrix
        nn = torch.mul(n.unsqueeze(0).expand(batch_in, ngi, nloc), \
            detwei.unsqueeze(-1).expand(batch_in, ngi, nloc))   # (batch_in, ngi, nloc)
        nn = torch.bmm(torch.transpose(n,0,1).unsqueeze(0).expand(batch_in, nloc, ngi), \
            nn) # (batch_in, nloc, nloc)
        
        nxnx = nn - nxnx*dt # this is (M-dt K), (batch_in, nloc, nloc)
        b = torch.matmul(nxnx,torch.transpose(c,1,2)) # batch matrix-vector multiplication, 
            # input1: (batch_in, nloc, nloc)
            # input2: (batch_in, nloc, 1)
            # broadcast over batch
            # output: (batch_in, nloc, 1)
        b = torch.transpose(b,1,2) # return to (batch_in, 1, nloc)
        
        return nn, b