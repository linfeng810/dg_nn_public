# configuration
import meshio
import numpy as np
import torch
import sys
import cmmn_data
import time
# from function_space import FuncSpace, Element

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
filename='z31-cube-mesh/cube_diri_neu.msh' # directory to mesh file (gmsh)
filename='z32-square-mesh/square_poiseuille.msh'
if len(sys.argv) > 1:
    filename = sys.argv[1]
mesh = meshio.read(filename) # mesh object
sf_nd_nb = cmmn_data.SfNdNb()

# mesh info
nele = mesh.cell_data['gmsh:geometrical'][-1].shape[0]  # number of elements
ele_p = 3  # velocity element order (2 or higher)
ele_p_pressure = ele_p - 1  # pressure element order
print('element order: ', ele_p)

ndim = 2  # dimesnion of the problem


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
is_sfc = True  # whether visiting SFC levels (otherwise will directly solve on P1CG)
print('MG parameters: \n this is V(%d,%d) cycle'%(pre_smooth_its, post_smooth_its),
      'with PMG?', is_pmg,
      'with SFC?', is_sfc)
print('jacobi block solver is: ', blk_solver)

# gmres parameters
gmres_m = 20  # restart
gmres_its = 20  # max GMRES steps
print('linear solver is: ', linear_solver)
if linear_solver == 'gmres' or linear_solver == 'gmres-mg':
    print('gmres paraters: restart=', gmres_m)

# non-linear iteration parameters
n_its_max = 50
n_tol = 1.e-7
relax_coeff = 1.

####################
# material property
####################
problem = 'ldc'  # 'hyper-elastic' or 'linear-elastic' or 'stokes' or 'ns' or 'kovasznay' or 'poiseuille'
# or 'ldc' = lid-driven cavity
# E = 2.5
# nu = 0.25  # or 0.49, or 0.4999
# lam = E*nu/(1.+nu)/(1.-2.*nu)
# mu = E/2.0/(1.+nu)
lam = 10
mu = 1
E = mu * (3*lam + 2*mu) / (lam + mu)
nu = lam + 2. * mu / 3.
lam = torch.tensor(lam, device=dev, dtype=torch.float64)
mu = torch.tensor(mu, device=dev, dtype=torch.float64)
print('Lame coefficient: lamda, mu', lam, mu)
# lam = 1.0; mu = 1.0
kdiff = 1.0
# print('lam, mu', lam, mu)
rho = 1.
a = torch.eye(ndim, device=dev, dtype=torch.float64)
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

if True:
    mu = 1/5000  # this is diffusion coefficient (viscosity)
    _Re = int(1/mu)
    hasNullSpace = True  # to remove null space, adding 1 to a pressure diagonal node
    is_pressure_stablise = False  # to add stablise term h[p][q] to pressure block or not.
    include_adv = True  # if Navier-Stokes, include advection term.
    print('viscosity, Re, hasNullSpade, is_pressure_stabilise?', mu, _Re, hasNullSpace, is_pressure_stablise)

    isSetInitial = False  # whether to use a precalculated fields (u and p) as initial condition
    initDataFile = 'Re100.pt'
    print('initial condition: '+initDataFile)

# Edge stabilisation (for convection-dominant and not-fine-enough mesh) (like SUPG but simpler)
# c.f. Burman & Hansbo CMAME 2004
isES = True
gammaES = 2.5e-2  # stabilisation parameter

####################
# discretisation settings
classicIP = True  # boolean
eta_e = 36.  # penalty coefficient
print('Surface jump penalty coefficient eta_e: ', eta_e)

# no of batches in mf volume and surface integral
no_batch = 1
print('No of batch: ', no_batch)

case_name = '_'+problem+'Re'+str(_Re)+'_p'+str(ele_p)+'p'+str(ele_p_pressure)+\
            '_'+time.strftime("%Y%m%d-%H%M%S")  # this is used in output vtk.
print('case name is: '+case_name)
