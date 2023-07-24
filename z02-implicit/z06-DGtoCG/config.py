# configuration
import meshio
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
isTransient=False # decide if we are do transient simulation
solver='iterative' # 'direct' or 'iterative'

#####################################################
# read mesh and build connectivity
#####################################################
filename='z04-cube-mesh/cube_only_diri.msh' # directory to mesh file (gmsh)
if len(sys.argv) > 1:
    filename = sys.argv[1]
mesh = meshio.read(filename) # mesh object
sf_nd_nb = cmmn_data.SfNdNb()

# mesh info
nele = mesh.cell_data['gmsh:geometrical'][-1].shape[0]  # number of elements
ele_type = 'cubic'  # 'linear' or 'cubic' or 'quadratic'
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
        sngi = 12
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
ndglno=np.arange(0,nonods)  # local to global

linear_solver = 'gmres-mg'  # linear solver: either 'gmres' or 'mg' or 'gmres-mg' (preconditioned gmres)
tol = 1.e-10  # convergence tolerance for linear solver (e.g. MG)
######################
jac_its = 2000  # max jacobi iteration steps on PnDG (overall MG cycles)
jac_wei = 2./3.  # jacobi weight
mg_its = [1, 1, 1, 1, 1, 1, 1]          # smooth steps on each level: P1CG(SFC0), SFC1, ...
mg_tol = 0.1    # multigrid smoother raletive residual tolorance (if we want)
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
# 'none' don't use block jacobian, but use point jacobian. Usually not stable.
is_pmg = False  # whether visiting each order DG grid (p-multigrid)
is_sfc = True  # whether visiting SFC levels (otherwise will directly solve on P1CG)
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

####################
# discretisation settings
classicIP = True # boolean
eta_e = 36.

# no of batches in mf volume and surface integral
no_batch = 1

# material property (data)
k = 1.  # diffusion coefficient
