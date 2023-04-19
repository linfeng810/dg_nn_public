# configuration
import toughio
import numpy as np
import torch
import sys
import cmmn_data

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
filename='square.msh' # directory to mesh file (gmsh)
if len(sys.argv) > 1:
    filename = sys.argv[1]
mesh = toughio.read_mesh(filename) # mesh object
sf_nd_nb = cmmn_data.SfNdNb()

# mesh info
nele = mesh.n_cells # number of elements
ele_type = 'cubic'  # 'linear' or 'cubic'
if ele_type=='cubic':
    nloc = 10  # number of nodes in an element
    ngi = 13 # number of quadrature points
    sngi = 4 # number of surface quadrature
    snloc = 4  # number of nodes per face
elif ele_type=='linear':
    nloc = 3
    ngi = 3
    sngi = 2
    snloc = 2
else:
    raise Exception("Element type is not acceptable.")
nonods = nloc*nele # number of nodes
p1dg_nonods = 3*nele  # number of nodes on P1DG grid
ndim = 2 # dimesnion of the problem
nface = 3 # number of element faces
ndglno=np.arange(0,nonods) # local to global


######################
jac_its = 500  # max jacobi iteration steps on PnDG (overall MG cycles)
jac_resThres = 1.e-10  # convergence criteria
jac_wei = 2./3. # jacobi weight
mg_its = [1, 1, 1, 1, 1, 1, 1]          # smooth steps on each level: P1CG(SFC0), SFC1, ...
mg_tol = 0.1    # multigrid smoother raletive residual tolorance (if we want)
mg_smooth_its = 1
pre_smooth_its = 3
post_smooth_its = 3  # thus we have a V(pre,post)-cycle
smooth_start_level = -1  # choose a level to directly solve on. then we'll iterate from there and levels up
if len(sys.argv) > 2:
    smooth_start_level = int(sys.argv[2])
if len(sys.argv) > 3:
    pre_smooth_its = int(sys.argv[3])
    post_smooth_its = int(sys.argv[3])
is_mass_weighted = False  # mass-weighted SFC-level restriction/prolongation
blk_solver = 'direct'  # block Jacobian iteration's block (10x10) -- 'direct' direct inverse
# 'jacobi' do 3 jacobi iteration (approx. inverse)
is_pmg = False  # whether visiting each order DG grid (p-multigrid)
is_sfc = True  # whether visiting SFC levels (otherwise will directly solve on P1CG)
print('this is V(%d,%d) cycle'%(pre_smooth_its, post_smooth_its), 'with PMG?', is_pmg, 'with SFC?', is_sfc)

####################
# material property
####################
E = 2.5
nu = 0.25  # or 0.49, or 0.4999
lam = E*nu/(1.+nu)/(1.-2.*nu)
mu = E/2.0/(1.+nu)
lam = torch.tensor(lam, device=dev, dtype=torch.float64)
mu = torch.tensor(mu, device=dev, dtype=torch.float64)
print('Lame coefficient: lamda, mu', lam, mu)
# lam = 1.0; mu = 1.0
kdiff = 1.0
# print('lam, mu', lam, mu)
rho = 1.
a = torch.eye(2, device=dev)
kijkl = torch.einsum('ik,jl->ijkl',a,a)  # k tensor for double diffusion
cijkl = lam*torch.einsum('ij,kl->ijkl',a,a)\
    +mu*torch.einsum('ik,jl->ijkl',a,a)\
    +mu*torch.einsum('il,jk->ijkl',a,a)  # c_ijkl elasticity tensor

ijkldim_nz = [[0,0,0,0], [0,0,1,1], [0,1,0,1], [0,1,1,0],
              [1,0,0,1], [1,0,1,0], [1,1,0,0], [1,1,1,1]]  # non-zero indices of cijkl
# print('cijkl=', cijkl)

#####################
# rhs body force 
def rhs_f(x_all, mu):
    # takes in coordinates numpy array (nonods, ndim)
    # output body force: torch tensor (nele*nloc, ndim)
    mu = mu.cpu().numpy()
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

# no of batches in mf volume and surface integral
no_batch = 1
print('No of batch: ', no_batch)
