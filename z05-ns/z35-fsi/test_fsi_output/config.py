# configuration
import meshio
import numpy as np
import torch
import sys
import cmmn_data
import time
# from function_space import FuncSpace, Element
from parse_terminal_input import args

torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)
disabletqdm = False
#
# device
dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dev="cpu" # if force funning on cpu
torch.manual_seed(0)

isFSI = True  # fsi problem or not

#####################################################
# time step settings
#####################################################
dt = 1  # timestep
if args.dt is not None:
    dt = args.dt
tstart = 0.  # starting time
tend = 1 * dt  # end time
isTransient = True  # decide if we are doing transient simulation
isAdvExp = False  # treat advection term explicitly
if isTransient:
    time_order = 3  # time discretisation order
    print('dt, tstart, tend, temporal order:', dt, tstart, tend, time_order, 'treat adv explicitly?', isAdvExp)
solver = 'iterative'  # 'direct' or 'iterative'

#####################################################
# read mesh and build connectivity
#####################################################
filename='z31-cube-mesh/cube_diri_neu.msh' # directory to mesh file (gmsh)
filename='z32-square-mesh/square_poiseuille_r2.msh'
# filename = 'z32-square-mesh/square.msh'
filename = 'z34-bfs/bfs.msh'
filename = 'z33-fpc/fpc.msh'
filename = 'fs_square.msh'
if args.filename is not None:
    filename = args.filename
# if len(sys.argv) > 1:
#     filename = sys.argv[1]
mesh = meshio.read(filename) # mesh object
isoparametric = True  # use iso-parametric geometry
sf_nd_nb = cmmn_data.SfNdNb()
use_fict_dt_in_vel_precond = True
sf_nd_nb.use_fict_dt_in_vel_precond = use_fict_dt_in_vel_precond  # add mass matrix to velocity block preconditioner
sf_nd_nb.fict_dt = 0.0025  # coefficient multiply to mass matrix add to vel blk precond
print('use fictitious timestep in velocity block preconditioner? (to make it diagonal dominant)',
      sf_nd_nb.use_fict_dt_in_vel_precond,
      'coeff', sf_nd_nb.fict_dt)
sf_nd_nb.isTransient = isTransient
sf_nd_nb.dt = dt

# mesh info
if not isFSI:
    nele = mesh.cell_data['gmsh:geometrical'][-1].shape[0]  # number of elements
else:
    nele_f = mesh.cell_data['gmsh:geometrical'][-2].shape[0]
    nele_s = mesh.cell_data['gmsh:geometrical'][-1].shape[0]
    nele = nele_f + nele_s
ele_p = 3  # velocity element order (2 or higher)
ele_p_pressure = ele_p - 1  # pressure element order
print('element order: ', ele_p)

ndim = 2  # dimesnion of the problem


linear_solver = 'gmres-mg'  # linear solver: either 'gmres' or 'mg' or 'gmres-mg' (preconditioned gmres)
tol = 1.e-8  # convergence tolerance for linear solver (e.g. MG)
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
# if len(sys.argv) > 2:
#     smooth_start_level = int(sys.argv[2])
# if len(sys.argv) > 3:
#     pre_smooth_its = int(sys.argv[3])
#     post_smooth_its = int(sys.argv[3])
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
gmres_m = 80  # restart
gmres_its = 400  # max GMRES steps
print('linear solver is: ', linear_solver)
if linear_solver == 'gmres' or linear_solver == 'gmres-mg':
    print('gmres paraters: restart=', gmres_m, 'max restart: ', gmres_its)

# non-linear iteration parameters
n_its_max = 15
n_tol = 1.e-6
relax_coeff = 1.

####################
# material property
####################
problem = 'fsi-test'  # 'hyper-elastic' or 'linear-elastic' or 'stokes' or 'ns' or 'kovasznay' or 'poiseuille'
# or 'ldc' = lid-driven cavity or 'tgv' = taylor-green vortex
# or 'bfs' = backward facing step
# or 'fpc' = flow-past cylinder
# or 'fsi-test' = test fluid-structure boundary
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
if isFSI:
    rho_s = 1.  # solid density at initial configuration
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
    mu = 1./10000  # this is diffusion coefficient (viscosity)
    _Re = int(1/mu)
    hasNullSpace = False  # to remove null space, adding 1 to a pressure diagonal node
    is_pressure_stablise = False  # to add stablise term h[p][q] to pressure block or not.
    include_adv = True  # if Navier-Stokes, include advection term.
    if isAdvExp:
        include_adv = False  # treat advection explicitly, no longer need to include adv in left-hand matrix.
    print('viscosity, Re, hasNullSpace, is_pressure_stabilise?', mu, _Re, hasNullSpace, is_pressure_stablise)

    initialCondition = 2  # 1. use zero as initial condition
    # 2. solve steady stokes as initial condition
    # 3. to use a precalculated fields (u and p) in a file as initial condition
    if initialCondition == 3:
        initDataFile = 'Re109_t20.00.pt'
        print('use this data file as initial condition: '+initDataFile)

# === all kinds of stabilisation for convection-dominant flow ===
# Edge stabilisation (for convection-dominant and not-fine-enough mesh) (like SUPG but simpler)
# c.f. Burman & Hansbo CMAME 2004
# this will make iterative solver less effective!
isES = True
gammaES = 0.01  # stabilisation parameter
sf_nd_nb.isES = isES
# Petrov-Galerkin stabilisation
isPetrovGalerkin = False
isPetrovGalerkinFace = False
sf_nd_nb.isPetrovGalerkin = isPetrovGalerkin
sf_nd_nb.isPetrovGalerkinFace = isPetrovGalerkinFace
# grad-div stab + interior penalty stab (see Niklas Fehn 2021 TUM PhD thesis pp 50-51)
isGradDivStab = False
zeta = 1.  # stab coefficient
print('is Edge stabilisation?', isES, gammaES,
      '\nis Petrov Galerkin Stabilisation? on face?', isPetrovGalerkin, isPetrovGalerkinFace,
      '\nis grad-div stabilisation? coefficient is', isGradDivStab, zeta)

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
# case_name = '_bfsRe109_p3p2_20230828-190846'
print('case name is: '+case_name)
