# configuration
import toughio
import numpy as np
import torch
import sys

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
filename='square_2M.msh' # directory to mesh file (gmsh)
if len(sys.argv) > 1:
    filename = sys.argv[1]
mesh = toughio.read_mesh(filename) # mesh object

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
cg_ndglno=[]
cg_nonods=[]


######################
jac_its = 500  # max jacobi iteration steps on PnDG (overall MG cycles)
jac_wei = 2./3. # jacobi weight
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
print('this is V(%d,%d) cycle'%(pre_smooth_its, post_smooth_its))

####################
# discretisation settings
classicIP = True # boolean
eta_e = 36.

# no of batches in mf volume and surface integral
no_batch = 16