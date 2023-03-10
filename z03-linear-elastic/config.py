# configuration
import toughio
import numpy as np
import torch

torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)
#
# device
dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dev="cpu" # if force funning on cpu
torch.manual_seed(0)

#####################################################
# time step settings
#####################################################
dt = 1e8 # timestep
tstart=0 # starting time
tend=1e8 # end time, we'll need ~2s for the modal problem to reach static state
isTransient=False # decide if we are doing transient simulation
solver='iterative' # 'direct' or 'iterative'

#####################################################
# read mesh and build connectivity
#####################################################
filename='square_refine5.msh' # directory to mesh file (gmsh)
mesh = toughio.read_mesh(filename) # mesh object

# mesh info
nele = mesh.n_cells # number of elements
cubic = True # elemnet type (cubic)
if cubic:
    nloc = 10  # number of nodes in an element
    ngi = 13 # number of quadrature points
    sngi = 4 # number of surface quadrature
nonods = nloc*nele # number of nodes
ndim = 2 # dimesnion of the problem
nface = 3 # number of element faces
ndglno=np.arange(0,nonods) # local to global


######################
jac_its = 1e1  # max jacobi iteration steps
jac_wei = 2./3.  # jacobi weight
mg_its = 1          # mg cycle
mg_smooth_its = 1 # smooth step

####################
# material property
####################
E = 2.5
nu = 0.25  # or 0.49, or 0.4999
lam = E*nu/(1.+nu)/(1.-2.*nu)
mu = E/2.0/(1.+nu)
print('Lame coefficient: lamda, mu', lam, mu)
# lam = 1.0; mu = 1.0
kdiff = 1.0
# print('lam, mu', lam, mu)
rho = 1.
a = torch.eye(2)
kijkl = torch.einsum('ik,jl->ijkl',a,a)  # k tensor for double diffusion
cijkl = lam*torch.einsum('ij,kl->ijkl',a,a)\
    +mu*torch.einsum('ik,jl->ijkl',a,a)\
    +mu*torch.einsum('il,jk->ijkl',a,a) # c_ijkl elasticity tensor

ijkldim_nz = [[0,0,0,0], [0,0,1,1], [0,1,0,1], [0,1,1,0],
              [1,0,0,1], [1,0,1,0], [1,1,0,0], [1,1,1,1]]  # non-zero indices of cijkl
# print('cijkl=', cijkl)

#####################
# rhs body force 
def rhs_f(x_all):
    # takes in coordinates numpy array (nonods, ndim)
    # output body force: torch tensor (nele*nloc, ndim)
    f = np.zeros((nonods, ndim), dtype=np.float64)
    f[:, 0] += -2.0*mu*np.power(np.pi,3)*\
        np.cos(np.pi*x_all[:,1]) * np.sin(np.pi*x_all[:,1])\
        * (2*np.cos(2*np.pi*x_all[:,0])-1)
    f[:, 1] += 2.0*mu*np.power(np.pi,3)*\
        np.cos(np.pi*x_all[:,0]) * np.sin(np.pi*x_all[:,0])\
        * (2*np.cos(2*np.pi*x_all[:,1])-1)
    f = torch.tensor(f, device=dev, dtype=torch.float64)
    fNorm = torch.linalg.norm(f.view(-1), dim=0)
    # f *=0
    # np.savetxt('f.txt', f.cpu().numpy(), delimiter=',')
    # np.savetxt('x_all.txt', x_all, delimiter=',')
    return f, fNorm


####################
# discretisation settings
classicIP = True  # boolean
eta_e = 36.*E  # penalty coefficient
print('Surface jump penalty coefficient eta_e: ', eta_e)
