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
isTransient=False # decide if we are do transient simulation
solver='iterative' # 'direct' or 'iterative'

#####################################################
# read mesh and build connectivity
#####################################################
filename='square.msh' # directory to mesh file (gmsh)
mesh = toughio.read_mesh(filename) # mesh object

# mesh info
nele = mesh.n_cells # number of elements
ele_type = 'linear'  # 'linear' or 'cubic'
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
ndim = 2 # dimesnion of the problem
nface = 3 # number of element faces
ndglno=np.arange(0,nonods) # local to global 


######################
jac_its = 1e5 # max jacobi iteration steps
jac_wei = 1. # jacobi weight
mg_its = 1          # mg cycle
mg_smooth_its = 1 # smooth step


####################
# discretisation settings
classicIP = True # boolean
eta_e = 36.