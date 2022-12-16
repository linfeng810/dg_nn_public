# configuration
import toughio
import numpy as np
import torch

#
# device
dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dev="cpu" # if force funning on cpu

#####################################################
# time step settings
#####################################################
dt = 1e-3 # timestep
tstart=0 # starting time
tend=5 # end time, we'll need ~2s for the modal problem to reach static state

#####################################################
# read mesh and build connectivity
#####################################################
filename='square.msh' # directory to mesh file (gmsh)
mesh = toughio.read_mesh(filename) # mesh object

# mesh info
nele = mesh.n_cells # number of elements
cubic = True # elemnet type (cubic)
if (cubic) :
    nloc = 10  # number of nodes in an element
    ngi = 13 # number of quadrature points
nonods = nloc*nele # number of nodes
ndim = 2 # dimesnion of the problem
ndglno=np.arange(0,nonods) # local to global 


######################
jac_its = 1e3 # max jacobi iteration steps
jac_wei = 2./3.