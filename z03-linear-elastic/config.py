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
filename='z11-cube-mesh/cube.msh' # directory to mesh file (gmsh)
if len(sys.argv) > 1:
    filename = sys.argv[1]
mesh = toughio.read_mesh(filename) # mesh object
sf_nd_nb = cmmn_data.SfNdNb()

# mesh info
nele = mesh.n_cells # number of elements
ele_type = 'linear'  # 'linear' or 'cubic' or 'quadratic'
print('element order: ', ele_type)
ndim = 3  # dimesnion of the problem
if ndim == 2:
    if ele_type == 'cubic':
        nloc = 10  # number of nodes in an element
        ngi = 13 # number of quadrature points
        sngi = 4 # number of surface quadrature
        snloc = 4  # number of nodes per face
        ele_p = 3  # order of elements
    elif ele_type == 'linear':
        nloc = 3
        ngi = 3
        sngi = 2
        snloc = 2
        ele_p = 1  # order of elements
    else:
        raise Exception("Element type is not acceptable.")
    p1dg_nonods = 3*nele  # number of nodes on P1DG grid
    nface = 3 # number of element faces
    p1cg_nloc = 3
else:  # ndim = 3
    if ele_type == 'cubic':
        nloc = 20
        ngi = 24
        snloc = 10
        sngi = 9
        ele_p = 3  # order of elements
    elif ele_type == 'linear':
        nloc = 4
        ngi = 4
        snloc = 3
        sngi = 3
        ele_p = 2  # order of elements
    elif ele_type == 'quadratic':
        nloc = 10
        ngi = 11
        snloc = 6
        sngi = 6
        ele_p = 1  # order of elements
    else:
        raise Exception("Element type is not acceptable.")
    p1dg_nonods = 4*nele
    nface = 4
    p1cg_nloc = 4
nonods = nloc*nele  # number of nodes
ndglno=np.arange(0,nonods) # local to global

linear_solver = 'gmres-mg'  # linear solver: either 'gmres' or 'mg' or 'gmres-mg' (preconditioned gmres)
tol = 1.e-10  # convergence tolerance for linear solver (e.g. MG)
######################
jac_its = 500  # max jacobi iteration steps on PnDG (overall MG cycles)
jac_resThres = tol  # convergence criteria
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
is_sfc = False  # whether visiting SFC levels (otherwise will directly solve on P1CG)
print('MG parameters: \n this is V(%d,%d) cycle'%(pre_smooth_its, post_smooth_its),
      'with PMG?', is_pmg,
      'with SFC?', is_sfc)
print('jacobi block solver is: ', blk_solver)

# gmres parameters
gmres_m = 10  # restart
gmres_its = 100  # max GMRES steps
print('linear solver is: ', linear_solver)
if linear_solver == 'gmres' or linear_solver == 'gmres-mg':
    print('gmres paraters: restart=', gmres_m)

# non-linear iteration parameters
n_its_max = 10
n_tol = 1.e-10
relax_coeff = 0.1

####################
# material property
####################
problem = 'hyper-elastic'  # 'hyper-elastic' or 'linear-elastic'
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
a = torch.eye(ndim, device=dev)
kijkl = torch.einsum('ik,jl->ijkl',a,a)  # k tensor for double diffusion
cijkl = lam*torch.einsum('ij,kl->ijkl',a,a)\
    +mu*torch.einsum('ik,jl->ijkl',a,a)\
    +mu*torch.einsum('il,jk->ijkl',a,a)  # c_ijkl elasticity tensor

if ndim == 2:
    ijkldim_nz = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0],
                  [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]]  # non-zero indices of cijkl
else:
    ijkldim_nz = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 2, 2], [0, 1, 0, 1],
                  [0, 1, 1, 0], [0, 2, 0, 2], [0, 2, 2, 0], [1, 0, 0, 1],
                  [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 2, 2],
                  [1, 2, 1, 2], [1, 2, 2, 1], [2, 0, 0, 2], [2, 0, 2, 0],
                  [2, 1, 1, 2], [2, 1, 2, 1], [2, 2, 0, 0], [2, 2, 1, 1], [2, 2, 2, 2]]  # non-zero indices of cijkl

# print('cijkl=', cijkl)


####################
# discretisation settings
classicIP = True  # boolean
eta_e = 36.*E  # penalty coefficient
print('Surface jump penalty coefficient eta_e: ', eta_e)

# no of batches in mf volume and surface integral
no_batch = 1
print('No of batch: ', no_batch)
