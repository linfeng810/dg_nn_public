#!/usr/bin/env python3

####################################################
# preamble
####################################################
# import 
import toughio
import numpy as np
import torch
from torch.nn import Conv1d,Sequential,Module
import scipy as sp
# import time
from scipy.sparse import coo_matrix, bsr_matrix
from tqdm import tqdm
import config
import output
import shape_function
import volume_mf_st
from function_space import FuncSpace, Element
import solvers
# import volume_mf_he
from config import sf_nd_nb
import mesh_init
# from mesh_init import face_iloc,face_iloc2
# from shape_function import SHATRInew, det_nlx, sdet_snlx
import volume_mf_st as integral_mf
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
# mesh = config.mesh
# nonods = config.nonods
# p1dg_nonods = config.p1dg_nonods
# ngi = config.ngi
ndim = config.ndim
# nloc = config.nloc
dt = config.dt
tend = config.tend
tstart = config.tstart

print('computation on ',dev)

# define element
vel_ele = Element(ele_order=config.ele_p, gi_order=config.ele_p*2, edim=ndim, dev=dev)
pre_ele = Element(ele_order=config.ele_p_pressure, gi_order=config.ele_p*2, edim=ndim,dev=dev)
print('ele pair: ', vel_ele.ele_order, pre_ele.ele_order)
# define function space
# if ndim == 2:
#     x_all, nbf, nbele, alnmt, fina, cola, ncola, \
#         bc, \
#         cg_ndglno, cg_nonods, cg_bc = mesh_init.init()
# else:
#     alnmt: object
#     [x_all, nbf, nbele, alnmt, fina, cola, ncola, bc, cg_ndglno, cg_nonods] = mesh_init.init_3d()
vel_func_space = FuncSpace(vel_ele, name="Velocity", mesh=config.mesh, dev=dev)
pre_func_space = FuncSpace(pre_ele, name="Pressure", mesh=config.mesh, dev=dev)
sf_nd_nb.set_data(vel_func_space=vel_func_space,
                  pre_func_space=pre_func_space,
                  p1cg_nonods=vel_func_space.cg_nonods)

print('nele=', nele)
# config.sf_nd_nb.set_data(nbele=nbele, nbf=nbf, alnmt=alnmt)
np.savetxt('cg_ndglno.txt', sf_nd_nb.vel_func_space.cg_ndglno, delimiter=',')  # cg_ndglno is stored in sf_nd_nb in mesh_init.init

if True:# P1CG connectivity and coloring
    fina, cola, ncola = mesh_init.p1cg_sparsity(sf_nd_nb.vel_func_space)
    whichc, ncolor = color2(fina=fina, cola=cola, nnode=sf_nd_nb.p1cg_nonods)
# np.savetxt('whichc.txt', whichc, delimiter=',')
print('ncolor', ncolor, 'whichc type', whichc.dtype)
print('cg_nonods', sf_nd_nb.p1cg_nonods, 'ncola (p1cg sparsity)', ncola)
print('1. time elapsed, ',time.time()-starttime)
# #####################################################
# # shape functions
# #####################################################
# # get shape functions on reference element
# [n,nlx,weight,sn,snlx,sweight] = SHATRInew(config.nloc,
#                                            config.ngi, config.ndim, config.snloc, config.sngi)
# # put numpy array to torch tensor in expected device
# sf_nd_nb.set_data(n=torch.tensor(n, device=config.dev, dtype=torch.float64))
# sf_nd_nb.set_data(nlx=torch.tensor(nlx, device=dev, dtype=torch.float64))
# sf_nd_nb.set_data(weight=torch.tensor(weight, device=dev, dtype=torch.float64))
# sf_nd_nb.set_data(sn=torch.tensor(sn, dtype=torch.float64, device=dev))
# sf_nd_nb.set_data(snlx=torch.tensor(snlx, device=dev, dtype=torch.float64))
# sf_nd_nb.set_data(sweight=torch.tensor(sweight, device=dev, dtype=torch.float64))
# del n, nlx, weight, sn, snlx, sweight
# print('3. time elapsed, ',time.time()-starttime)
#######################################################
# assemble local mass matrix and stiffness matrix
#######################################################


# Mk = mk()
# Mk.to(device=dev)
print('4. time elapsed, ',time.time()-starttime)

####################################################
# time loop
####################################################
# x_ref_in = np.empty((nele, ndim, nloc))
# for ele in range(nele):
#     for iloc in range(nloc):
#         glb_iloc = ele*nloc+iloc
#         for idim in range(ndim):
#             x_ref_in[ele, idim, iloc] = x_all[glb_iloc, idim]
# sf_nd_nb.set_data(x_ref_in = torch.tensor(x_ref_in,
#                                           device=dev,
#                                           dtype=torch.float64,
#                                           requires_grad=False))
# del x_ref_in

# initical condition
u = torch.zeros(nele, vel_ele.nloc, ndim, device=dev, dtype=torch.float64)  # now we have a vector filed to solve
p = torch.zeros(nele, pre_ele.nloc, device=dev, dtype=torch.float64)
u, f, fNorm = bc_f.vel_bc_f(ndim, vel_func_space.bc, u, vel_func_space.x_all, prob=config.problem)  # 'linear-elastic')

# print(u)
tstep = int(np.ceil((tend-tstart)/dt)) + 1
u = u.reshape(nele, vel_ele.nloc, ndim) # reshape doesn't change memory allocation.
u_bc = u.detach().clone() # this stores Dirichlet boundary *only*, otherwise zero.

u = torch.rand_like(u)*0  # initial guess
# to test vtk output, set u and p be ana sol here
if False:
    u = u.view(vel_func_space.nonods, ndim)
    x = vel_func_space.x_all[:, 0]
    y = vel_func_space.x_all[:, 1]
    z = vel_func_space.x_all[:, 2]
    u[:, 0] += torch.tensor(-2./3. * np.sin(x)**3, device=dev)
    u[:, 1] += torch.tensor(
        np.sin(x)**2 * (y * np.cos(x) - z * np.sin(x)),
        device=dev
    )
    u[:, 2] += torch.tensor(
        np.sin(x)**2 * (z * np.cos(x) + y * np.sin(x)),
        device=dev
    )
    p = p.view(pre_func_space.nonods)
    x = pre_func_space.x_all[:, 0]
    p += torch.tensor(np.sin(x), device=dev)
# output to vtk
vtk = output.File(config.filename+config.case_name+'_v%d.vtu' % 0)
vtk.write_vector(u, 'velocity', sf_nd_nb.vel_func_space)
vtk.write_end()
vtk = output.File(config.filename+config.case_name+'_p%d.vtu' % 0)
vtk.write_scaler(p, 'pressure', sf_nd_nb.pre_func_space)
vtk.write_end()

# u = torch.ones_like(u)
u_all = np.empty([tstep, nele, vel_ele.nloc, ndim])
u_all[0, :, :, :] = u.view(nele, vel_ele.nloc, ndim).cpu().numpy()
p_all = np.empty([tstep, nele, pre_ele.nloc])
p_all[0, ...] = p.view(nele, -1).cpu().numpy()

print('5. time elapsed, ',time.time()-starttime)
print('6. time elapsed, ',time.time()-starttime)

if False:  # per element condensation Rotation matrix (diagonal)
    R = torch.tensor([1./30., 1./30., 1./30., 3./40., 3./40., 3./40.,
                     3./40., 3./40., 3./40., 9./20.], device=dev, dtype=torch.float64)
    # R = torch.tensor([1./10., 1./10., 1./10., 1./10., 1./10., 1./10.,
    #         1./10., 1./10., 1./10., 1./10.], device=dev, dtype=torch.float64)
if True:  # from PnDG to P1CG
    p1dg_nonods = vel_func_space.p1dg_nonods
    cg_ndglno = vel_func_space.cg_ndglno
    cg_nonods = vel_func_space.cg_nonods
    I_fc_colx = np.arange(0, p1dg_nonods)
    I_fc_coly = cg_ndglno
    I_fc_val = np.ones(p1dg_nonods)
    I_cf = coo_matrix((I_fc_val, (I_fc_coly, I_fc_colx)),
                      shape=(cg_nonods, p1dg_nonods))  # fine to coarse == DG to CG
    I_cf = I_cf.tocsr()
    print('don assemble I_cf I_fc', time.time()-starttime)
    no_dgnodes = I_cf.sum(axis=1)
    print('done suming', time.time()-starttime)
    # weight by 1 / overlapping dg nodes number
    for i in range(p1dg_nonods):
        I_fc_val[i] /= no_dgnodes[I_fc_coly[i]]
    print('done weighting', time.time()-starttime)
    # for i in tqdm(range(cg_nonods)):
    #     # no_dgnodes = np.sum(I_cf[i,:])
    #     I_cf[i,:] /= no_dgnodes[i]
    #     I_fc[:,i] /= no_dgnodes[i]
    # print('done weighting', time.time()-starttime)
    I_cf = coo_matrix((I_fc_val, (I_fc_coly, I_fc_colx)),
                      shape=(cg_nonods, p1dg_nonods))  # fine to coarse == DG to CG
    I_cf = I_cf.tocsr()
    I_fc = coo_matrix((I_fc_val, (I_fc_colx, I_fc_coly)),
                      shape=(p1dg_nonods, cg_nonods))  # coarse to fine == CG to DG
    I_fc = I_fc.tocsr()
    print('done transform to csr', time.time()-starttime)
    # transfer to torch device
    I_fc = torch.sparse_csr_tensor(crow_indices=torch.tensor(I_fc.indptr),
                                   col_indices=torch.tensor(I_fc.indices),
                                   values=I_fc.data,
                                   size=(p1dg_nonods, cg_nonods),
                                   device=dev)
    I_cf = torch.sparse_csr_tensor(crow_indices=torch.tensor(I_cf.indptr),
                                   col_indices=torch.tensor(I_cf.indices),
                                   values=I_cf.data,
                                   size=(cg_nonods, p1dg_nonods),
                                   device=dev)
    sf_nd_nb.set_data(I_dc=I_fc, I_cd=I_cf)
#         np.savetxt('I_fc.txt', I_fc.to_dense().cpu().numpy(), delimiter=',')
#         np.savetxt('I_cf.txt', I_cf.to_dense().cpu().numpy(), delimiter=',')
print('7. time elapsed, ',time.time()-starttime)

if (config.solver=='iterative') :
    print('i am going to time loop')
    print('8. time elapsed, ',time.time()-starttime)
    # print("Using quit()")
    # quit()
    r0l2all = []
    # time loop
    r0 = torch.zeros(nele * (vel_ele.nloc * ndim + pre_ele.nloc),
                     device=dev, dtype=torch.float64)
    r0_list = [r0[0:nele * vel_ele.nloc * ndim].view(nele, -1, ndim),
               r0[nele * vel_ele.nloc * ndim:nele * (vel_ele.nloc * ndim + pre_ele.nloc)].view(nele, -1)]
    if config.problem == 'stokes':
        u_i = torch.zeros(nele, vel_ele.nloc, ndim, device=dev, dtype=torch.float64)
        u_rhs = torch.zeros(nele, vel_ele.nloc, ndim, device=dev, dtype=torch.float64)
        p_i = torch.zeros_like(r0_list[1], device=dev, dtype=torch.float64)
        dp_i = torch.zeros_like(r0_list[1], device=dev, dtype=torch.float64)
        p_rhs = torch.zeros_like(r0_list[1], device=dev, dtype=torch.float64)
        x_i = [u_i, dp_i]
        x_rhs = [u_rhs, p_rhs]
        for itime in range(1, tstep):  # time loop
            u_n = u.view(nele, vel_ele.nloc, ndim)  # store last timestep value to un
            u_i += u_n  # newton iteration initial value taken as last time step value
            p_i += p.view(nele, pre_ele.nloc)  # use last timestep p as start value
            dp_i *= 0

            r0l2 = torch.tensor(1, device=dev, dtype=torch.float64)  # linear solver residual l2 norm
            its = 0  # linear solver iteration
            nr0l2 = 1  # non-linear solver residual l2 norm
            nits = 0  # newton iteration step
            r0 *= 0

            while nits < config.n_its_max:
                u_rhs *= 0
                p_rhs *= 0
                x_rhs = integral_mf.get_rhs(
                    x_rhs=x_rhs, p_i=p_i, u_bc=u_bc, f=f
                )
                nr0l2 = r0l2  # stokes is linear prob, we will simply use linear residual as the non-linear residual
                print('============')
                print('nits = ', nits, 'non-linear residual = ', nr0l2.cpu().numpy())
                if nr0l2 < config.n_tol:
                    # non-linear iteration converged
                    break
                # prepare for MG on SFC-coarse grids
                RARvalues = integral_mf.calc_RAR_mf_color(
                    I_fc, I_cf,
                    whichc, ncolor,
                    fina, cola, ncola,
                )
                from scipy.sparse import bsr_matrix

                if not config.is_sfc:
                    RAR = bsr_matrix((RARvalues.cpu().numpy(), cola, fina),
                                     shape=((ndim+1) * cg_nonods, (ndim+1) * cg_nonods))
                    sf_nd_nb.set_data(RARmat=RAR.tocsr())
                # np.savetxt('RAR.txt', RAR.toarray(), delimiter=',')
                RARvalues = torch.permute(RARvalues, (1, 2, 0)).contiguous()  # (ndim+1,ndim+1,ncola)
                # get SFC, coarse grid and operators on coarse grid. Store them to save computational time?
                space_filling_curve_numbering, variables_sfc, nlevel, nodes_per_level = \
                    mg.mg_on_P0DG_prep(fina, cola, RARvalues)
                sf_nd_nb.sfc_data.set_data(
                    space_filling_curve_numbering=space_filling_curve_numbering,
                    variables_sfc=variables_sfc,
                    nlevel=nlevel,
                    nodes_per_level=nodes_per_level
                )
                # print('9. time elapsed, ', time.time() - starttime)

                dp_i *= 0  # solve for delta p_i and u_i
                if config.linear_solver == 'mg':
                    x_i = solvers.multigrid_solver(x_i, p_i, x_rhs, config.tol)
                elif config.linear_solver == 'gmres-mg':
                    x_i = solvers.gmres_mg_solver(x_i, p_i, x_rhs, config.tol) # min(1.e-3*nr0l2, 1.e-3))
                else:
                    raise Exception('choose a valid solver...')
                # get final residual after we're back to fine grid
                r0 *= 0
                r0_list = integral_mf.get_residual_only(
                    r0_list, x_i, x_rhs
                )

                r0l2 = integral_mf.get_r0_l2_norm(r0) / fNorm
                r0l2all.append(r0l2.cpu().numpy())
                print('its=', its, 'residual l2 norm=', r0l2.cpu().numpy(),
                      'abs res=', integral_mf.get_r0_l2_norm(r0).cpu().numpy(),
                      'fNorm', fNorm.cpu().numpy())
                # nr0l2 = torch.linalg.norm(du_i.view(-1))
                nr0l2 = r0l2
                # print('norm of delta u ', nr0l2.cpu().numpy())
                # u_i += du_i.view(nele, nloc, ndim) * config.relax_coeff
                nits += 1

            # if converges,
            x_i[0] = x_i[0].view(nele, -1, ndim)
            x_i[1] = x_i[1].view(nele, -1)

            # combine inner/inter element contribution
            u_all[itime, :, :, :] = x_i[0].cpu().numpy()
            p_all[itime, ...] = x_i[1].cpu().numpy()

            # output to vtk
            vtk = output.File(config.filename + config.case_name + '_v%d.vtu' % itime)
            vtk.write_vector(x_i[0], 'velocity', sf_nd_nb.vel_func_space)
            vtk.write_end()
            vtk = output.File(config.filename + config.case_name + '_p%d.vtu' % itime)
            vtk.write_scaler(x_i[1], 'pressure', sf_nd_nb.pre_func_space)
            vtk.write_end()

    np.savetxt('r0l2all.txt', np.asarray(r0l2all), delimiter=',')

if config.solver == 'direct':
    # raise NotImplemented('Direct solver for Stokes problem is not implemented!')
    u_nonods = vel_ele.nloc * nele
    p_nonods = pre_ele.nloc * nele
    # # let build the lhs matrix column by column...
    #
    # # a bunch of dummies...
    # x_dummy = torch.zeros(u_nonods * ndim + p_nonods,
    #                       device=dev, dtype=torch.float64)
    # u_dummy = x_dummy[0:u_nonods*ndim]
    # p_dummy = x_dummy[u_nonods*ndim:u_nonods*ndim+p_nonods]
    # x_list = [u_dummy, p_dummy]
    # rhs_dummy = torch.zeros(u_nonods*ndim+p_nonods, device=dev, dtype=torch.float64)
    # rhs_dummy_list = [rhs_dummy[0:u_nonods*ndim],
    #                   rhs_dummy[u_nonods*ndim:u_nonods*ndim+p_nonods]]
    # # assemble column by column
    # Amat = torch.zeros(u_nonods * ndim + p_nonods, u_nonods * ndim + p_nonods,
    #                    device=dev, dtype=torch.float64)
    # rhs = torch.zeros(u_nonods * ndim + p_nonods,
    #                   device=dev, dtype=torch.float64)
    # probe = torch.zeros(u_nonods * ndim + p_nonods,
    #                     device=dev, dtype=torch.float64)
    # probe_list = [probe[0:u_nonods*ndim],
    #               probe[u_nonods*ndim:u_nonods*ndim+p_nonods]]
    # rhs_list = [rhs[0:u_nonods*ndim],
    #             rhs[u_nonods*ndim:u_nonods*ndim+p_nonods]]
    # rhs_list = integral_mf.get_rhs(
    #     x_rhs=rhs_list,
    #     p_i=p_dummy,
    #     u_bc=u_bc,
    #     f=f
    # )
    # rhs_np = rhs.cpu().numpy()
    #
    # for inod in tqdm(range(u_nonods)):
    #     for jdim in range(ndim):
    #         u_dummy *= 0
    #         p_dummy *= 0
    #         rhs_dummy *= 0
    #         probe *= 0
    #         probe[jdim * u_nonods + inod] += 1.
    #         x_list = integral_mf.get_residual_only(
    #             r0=x_list,
    #             x_i=probe_list,
    #             x_rhs=rhs_dummy_list,
    #         )
    #         # add to Amat
    #         Amat[:, jdim*u_nonods + inod] -= x_dummy
    # # get pressure columns
    # for inod in tqdm(range(p_nonods)):
    #     u_dummy *= 0
    #     p_dummy *= 0
    #     rhs_dummy *= 0
    #     probe *= 0
    #     probe[ndim * u_nonods + inod] += 1.
    #     x_list = integral_mf.get_residual_only(
    #         r0=x_list,
    #         x_i=probe_list,
    #         x_rhs=rhs_dummy_list,
    #     )
    #     # put in Amat
    #     Amat[:, ndim*u_nonods + inod] -= x_dummy
    # Amat_np = Amat.cpu().numpy()
    # Amat_sp = sp.sparse.csr_matrix(Amat_np)

    import stokes_assemble
    print('im going to assemble', time.time()-starttime)
    Amat_sp, rhs_np = stokes_assemble.assemble(u_bc, f)
    print('done assemble, going to write', time.time()-starttime)
    Amat_np = Amat_sp.todense()
    print('Amat cond number is: ', np.linalg.cond(Amat_np))
    # print('Amat rank is: ', np.linalg.matrix_rank(Amat_np))
    # np.savetxt('rhs_assemble.txt', rhs_np, delimiter=',')
    # np.savetxt('Amat_assemble.txt', Amat_np, delimiter=',')
    print('doing writing. im goingt o solve', time.time()-starttime)
    x_sol = sp.sparse.linalg.spsolve(Amat_sp, rhs_np)
    u_sol = x_sol[0:ndim*u_nonods]
    p_sol = x_sol[ndim*u_nonods:ndim*u_nonods+p_nonods]
    u_all[1, ...] = u_sol.reshape(u_all[1, ...].shape)
    p_all[1, ...] = p_sol.reshape(p_all[1, ...].shape)
    # output to vtk
    vtk = output.File(config.filename + config.case_name + '_v%d.vtu' % 1)
    vtk.write_vector(torch.tensor(u_sol, dtype=torch.float64), 'velocity', sf_nd_nb.vel_func_space)
    vtk.write_end()
    vtk = output.File(config.filename + config.case_name + '_p%d.vtu' % 1)
    vtk.write_scaler(torch.tensor(p_sol, dtype=torch.float64), 'pressure', sf_nd_nb.pre_func_space)
    vtk.write_end()


#############################################################
# write output
#############################################################
# output 1: 
# to output, we need to change u_all to 2D array
# (tstep, ndim*nonods)
u_all = np.reshape(u_all, (tstep, nele*vel_ele.nloc*ndim))
p_all = np.reshape(p_all, (tstep, nele*pre_ele.nloc))
np.savetxt('u_all.txt', u_all, delimiter=',')
np.savetxt('p_all.txt', p_all, delimiter=',')
np.savetxt('x_all.txt', vel_func_space.x_all, delimiter=',')
np.savetxt('p_x_all.txt', pre_func_space.x_all, delimiter=',')
print('10. done output, time elaspsed: ', time.time()-starttime)
print(torch.cuda.memory_summary())
