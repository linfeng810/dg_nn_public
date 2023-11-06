#!/usr/bin/env python3

####################################################
# preamble
####################################################
# import
import numpy as np
import torch
from torch.nn import Conv1d,Sequential,Module
import scipy as sp
# import time
from scipy.sparse import coo_matrix, bsr_matrix
from tqdm import tqdm

import cmmn_data
import config
import fsi_output
import output
import shape_function
import sparsity
import volume_mf_diff
from function_space import FuncSpace, Element
from config import sf_nd_nb
import mesh_init
from color import color2
import multigrid_linearelastic as mg
import bc_f
import time

starttime = time.time()

# for pretty print out torch tensor
# torch.set_printoptions(sci_mode=False)
torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)

dev = config.dev
nele = config.nele
nele_f = config.nele_f
nele_s = config.nele_s
ndim = config.ndim
dt = config.dt
tend = config.tend
tstart = config.tstart

print('computation on ',dev)

# define element
quad_degree = config.ele_p*2
vel_ele = Element(ele_order=config.ele_p, gi_order=quad_degree, edim=ndim, dev=dev)
print('quadrature degree: ', quad_degree)

if True:  # scale mesh
    mesh_init.scale_mesh(mesh=config.mesh, origin=np.zeros(3), scale=np.asarray([1, 1, 4]))

vel_func_space = FuncSpace(vel_ele, name="Velocity", mesh=config.mesh, dev=dev)
sf_nd_nb.set_data(vel_func_space=vel_func_space,
                  p1cg_nonods=vel_func_space.cg_nonods)

fluid_spar, solid_spar = sparsity.get_subdomain_sparsity(
    vel_func_space.cg_ndglno,
    config.nele_f,
    config.nele_s,
    vel_func_space.cg_nonods
)
sf_nd_nb.set_data(sparse_f=fluid_spar, sparse_s=solid_spar)

print('nele=', nele)

print('1. time elapsed, ',time.time()-starttime)

"""getting boundary condition and rhs force, all problem are defined in fsi_bc"""
u_bc, f, fNorm = bc_f.diff_bc(
    ndim, vel_func_space.bc_node_list, vel_func_space.x_all,
    prob=config.problem,
    t=0)

tstep = int(np.ceil((tend-tstart)/dt)) + 1
if not sf_nd_nb.isTransient:
    tstep = 2
    sf_nd_nb.set_data(bdfscm=cmmn_data.BDFdata(order=config.time_order))
else:
    sf_nd_nb.set_data(bdfscm=cmmn_data.BDFdata(order=config.time_order))

print('no of time steps to compute: ', tstep)

u_nonods = sf_nd_nb.vel_func_space.nonods
u_nloc = sf_nd_nb.vel_func_space.element.nloc

no_total_dof = nele * vel_ele.nloc
if config.solver == 'iterative':
    print('i am going to time loop')
    print('8. time elapsed, ',time.time()-starttime)
    # print("Using quit()")
    # quit()
    r0l2all = []
    # time loop
    r0 = torch.zeros(no_total_dof,
                     device=dev, dtype=torch.float64)

    x_i = torch.zeros(no_total_dof, device=dev, dtype=torch.float64)
    x_rhs = torch.zeros(no_total_dof, device=dev, dtype=torch.float64)
    # let's create a list of tensors to store J+1 previoius timestep values
    # for time integrator
    x_all_previous = [torch.zeros(no_total_dof, dtype=torch.float64, device=dev)
                      for _ in range(config.time_order+1)]

    # solve a stokes problem as initial velocity
    if config.initialCondition == 2:
        raise ValueError('using Stokes soln as initial cond is not yet supported in FSI solver...')
    elif config.initialCondition == 1:
        x_all_previous[0] *= 0
        # ana_sln = bc_f.ana_soln(problem=config.problem, t=tstart)
        # x_all_previous[0]['vel'] += ana_sln[0:u_nonods*ndim].view(nele, -1, ndim)
        # x_all_previous[0]['pre'] += ana_sln[u_nonods*ndim:].view(nele, -1)

    elif config.initialCondition == 3:
        x_all_previous[0] *= 0
        x_all_previous[0] += torch.load(config.initDataFile)

    x_i *= 0
    x_i += x_all_previous[0]

    # save initial condition to vtk
    fsi_output.output_diff_vtu(x_i, vel_func_space, itime=0)

    t = tstart  # physical time (start time)

    alpha_u_n = torch.zeros(u_nonods, ndim, device=dev, dtype=torch.float64).view(nele, -1, ndim)

    for itime in range(1, tstep):  # time loop
        wall_time_start = time.time()
        sf_nd_nb.ntime = itime
        # for the starting steps, use 1st, 2nd then 3rd order BDF.
        if sf_nd_nb.isTransient:
            if itime <= config.time_order:
                if itime == 1:
                    sf_nd_nb.set_data(bdfscm=cmmn_data.BDFdata(order=1))
                elif itime == 2:
                    sf_nd_nb.set_data(bdfscm=cmmn_data.BDFdata(order=2))
                elif itime == 3:
                    sf_nd_nb.set_data(bdfscm=cmmn_data.BDFdata(order=3))
            if True:
                alpha_u_n *= 0
                for i in range(0, sf_nd_nb.bdfscm.order):
                    alpha_u_n += sf_nd_nb.bdfscm.alpha[i] * x_all_previous[i]['vel'].view(alpha_u_n.shape)

        t += dt
        print('====physical time: ', t, ' ====')
        # get boundary and rhs body force condition
        u_bc, f, fNorm = bc_f.diff_bc(
            ndim,
            vel_func_space.bc_node_list,
            vel_func_space.x_all,
            prob=config.problem,
            t=t
        )

        # save bc condition to vtk
        x_i += u_bc[0].view(x_i.shape)
        fsi_output.output_diff_vtu(x_i, vel_func_space, itime=0)

        x_i *= 0
        x_i += x_all_previous[0]  # use last timestep p as start value

        r0l2 = torch.tensor(1, device=dev, dtype=torch.float64)  # linear solver residual l2 norm
        its = 0  # linear solver iteration
        nr0l2 = 1  # non-linear solver residual l2 norm
        sf_nd_nb.nits = 0  # newton iteration step
        r0 *= 0

        total_its = 0  # total linear iteration number / restart

        sf_nd_nb.nits += 1
        print('============')  # start new non-linear iteration
        sf_nd_nb.Kmatinv = None

        # print('going to solve for mesh displacement and move the mesh...')
        x_i = volume_mf_diff.solve_for_diff(x_i, f, u_bc, alpha_u_n,
                                            t)

        # store this step in case we want to use this for next timestep
        for ii in range(len(x_all_previous)-1, 0, -1):
            x_all_previous[ii] *= 0
            x_all_previous[ii] += x_all_previous[ii-1]
        x_all_previous[0] *= 0
        x_all_previous[0] += x_i

        # save x_i at this Re to reuse as the initial condition for higher Re
        torch.save(x_i, config.filename + config.case_name + '_t%.2f.pt' % t)
        print('total its on this non-lienar step ', total_its)
        total_its = 0

        # output to vtk
        fsi_output.output_diff_vtu(x_i, vel_func_space,
                                   itime,
                                   )

        # get l2 error
        x_ana = bc_f.ana_soln(config.problem, t=0)
        u_l2, u_linf = volume_mf_diff.get_l2_error(x_i, x_ana)
        print('after solving, compare to previous timestep, l2 error is: \n',
              u_l2, '\n',
              )
        print('l infinity error is: \n',
              u_linf, '\n',
              )
        # print('total its / restart ', total_its)

        print('wall time on this timestep: ', time.time() - wall_time_start)

    # END OF TIME LOOP

print('10. done output, time elaspsed: ', time.time()-starttime)

