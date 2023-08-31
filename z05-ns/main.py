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
import pressure_matrix
import ns_assemble

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
quad_degree = config.ele_p*3
vel_ele = Element(ele_order=config.ele_p, gi_order=quad_degree, edim=ndim, dev=dev)
pre_ele = Element(ele_order=config.ele_p_pressure, gi_order=quad_degree, edim=ndim,dev=dev)
print('ele pair: ', vel_ele.ele_order, pre_ele.ele_order, 'quadrature degree: ', quad_degree)
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
# u = torch.zeros(nele, vel_ele.nloc, ndim, device=dev, dtype=torch.float64)  # now we have a vector filed to solve
u_bc, f, fNorm = bc_f.vel_bc_f(ndim, vel_func_space.bc_node_list, vel_func_space.x_all,
                               prob=config.problem,
                               t=0)
# output to vtk
vtk = output.File('bc_diri.vtu')
vtk.write_vector(u_bc[0], 'velocity', sf_nd_nb.vel_func_space)
vtk.write_end()
# print(u)
tstep = int(np.ceil((tend-tstart)/dt)) + 1
if not sf_nd_nb.isTransient:
    tstep = 2
else:
    sf_nd_nb.set_data(bdfscm=cmmn_data.BDFdata(order=config.time_order))

print('no of time steps to compute: ', tstep)

u = torch.zeros(nele, vel_ele.nloc, ndim, device=dev, dtype=torch.float64)  # initial guess
p = torch.zeros(nele, pre_ele.nloc, device=dev, dtype=torch.float64)
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

u_nonods = sf_nd_nb.vel_func_space.nonods
p_nonods = sf_nd_nb.pre_func_space.nonods
if (config.solver=='iterative') :
    print('i am going to time loop')
    print('8. time elapsed, ',time.time()-starttime)
    # print("Using quit()")
    # quit()
    r0l2all = []
    # time loop
    r0 = torch.zeros(nele * (vel_ele.nloc * ndim + pre_ele.nloc),
                     device=dev, dtype=torch.float64)
    r0_list = volume_mf_st.slicing_x_i(r0)
    if True:
        x_i = torch.zeros(u_nonods * ndim + p_nonods, device=dev, dtype=torch.float64)
        x_rhs = torch.zeros(u_nonods * ndim + p_nonods, device=dev, dtype=torch.float64)
        x_i_list = volume_mf_st.slicing_x_i(x_i)
        x_i_n = torch.zeros(u_nonods * ndim + p_nonods, device=dev, dtype=torch.float64)  # last step soln
        u_n, p_n = volume_mf_st.slicing_x_i(x_i_n)
        u_k = torch.zeros(u_nonods * ndim, device=dev, dtype=torch.float64)  # non-linear velocity
        # get pressure laplacian on P1CG, will be stored in
        # sf_nd_nb.sfc_data_Lp
        pressure_matrix.get_RAR_and_sfc_data_Lp(
            whichc, ncolor,
            fina, cola, ncola
        )

        # DEBUG: assemble Lp col by col
        if False:
            import Lp_assemble_colbycol
            Lpmat = Lp_assemble_colbycol.vanilla_assemble_Lp()
            np.savetxt('Lpmat_colbycolassemble.txt', Lpmat.todense(), delimiter=',')
        # solve a stokes problem as initial velocity
        if sf_nd_nb.isTransient and not config.isSetInitial:
            print('going to solve steady stokes problem as initial condition')
            sf_nd_nb.isTransient = False  # temporarily omit transient terms
            x_i *= 0
            x_i += x_i_n  # use last timestep p as start value
            r0 *= 0
            x_rhs *= 0
            x_rhs = volume_mf_st.get_rhs(
                x_rhs=x_rhs, u_bc=u_bc, f=f,
                include_adv=False,
                u_n=u_n  # last *timestep* velocity
            )
            RARvalues = integral_mf.calc_RAR_mf_color(
                I_fc, I_cf,
                whichc, ncolor,
                fina, cola, ncola,
                include_adv=False,
                u_n=u_k,
                u_bc=u_bc[0]
            )
            from scipy.sparse import bsr_matrix

            if not config.is_sfc:
                RAR = bsr_matrix((RARvalues.cpu().numpy(), cola, fina),
                                 shape=((ndim) * cg_nonods, (ndim) * cg_nonods))
                sf_nd_nb.set_data(RARmat=RAR.tocsr())
            # np.savetxt('RAR.txt', RAR.toarray(), delimiter=',')
            RARvalues = torch.permute(RARvalues, (1, 2, 0)).contiguous()  # (ndim, ndim, ncola)
            # get SFC, coarse grid and operators on coarse grid. Store them to save computational time?
            space_filling_curve_numbering, variables_sfc, nlevel, nodes_per_level = \
                mg.mg_on_P0CG_prep(fina, cola, RARvalues)
            sf_nd_nb.sfc_data.set_data(
                space_filling_curve_numbering=space_filling_curve_numbering,
                variables_sfc=variables_sfc,
                nlevel=nlevel,
                nodes_per_level=nodes_per_level
            )

            if config.linear_solver == 'gmres-mg':
                x_i, _ = solvers.gmres_mg_solver(
                    x_i, x_rhs,
                    tol=config.tol,
                    include_adv=False,
                    u_k=u_k,
                    u_bc=u_bc[0]
                )
            print('finishing solving stokes problem as initial conidtion.')
            x_i_n *= 0
            x_i_n += x_i

            sf_nd_nb.isTransient = config.isTransient  # change back to settings in config

        elif sf_nd_nb.isTransient and not config.isSetInitial:
            x_i_n *= 0
            x_i_n += bc_f.ana_soln(problem=config.problem, t=tstart)

        elif config.isSetInitial:
            x_i_n *= 0
            x_i_n += torch.load(config.initDataFile)

        p_ana_ave = 0
        if config.hasNullSpace:
            # remove average from pressure
            p_ana = x_i_n[u_nonods * ndim:u_nonods * ndim + p_nonods]
            p_ana_ave = ns_assemble.get_ave_pressure(p_ana.cpu().numpy())

        x_i *= 0
        x_i += x_i_n
        u_all[0, :, :, :] = x_i_list[0].cpu().numpy()
        p_all[0, ...] = x_i_list[1].cpu().numpy()

        # save initial condition to vtk
        vtk = output.File(config.filename + config.case_name + '_v%d.vtu' % 0)
        vtk.write_vector(u_n, 'velocity', sf_nd_nb.vel_func_space)
        vtk.write_end()
        vtk = output.File(config.filename + config.case_name + '_p%d.vtu' % 0)
        vtk.write_scaler(p_n-p_ana_ave, 'pressure', sf_nd_nb.pre_func_space)
        vtk.write_end()

        t = tstart  # physical time (start time)

        alpha_u_n = torch.zeros(u_nonods, ndim, device=dev, dtype=torch.float64)

        for itime in range(1, tstep):  # time loop
            # for the starting steps, use 1st, 2nd then 3rd order BDF.
            if False and itime < config.time_order:  # has analytical soln
                print('itime = ', itime, 'getting ana soln...')
                t += dt
                x_i *= 0
                x_i += bc_f.ana_soln(config.problem, t=t)

                # if converges,
                x_i_n *= 0
                x_i_n += x_i  # store this step in case we want to use this for next timestep

                # combine inner/inter element contribution
                u_all[itime, :, :, :] = x_i_list[0].cpu().numpy()
                p_all[itime, ...] = x_i_list[1].cpu().numpy()

                # output to vtk
                vtk = output.File(config.filename + config.case_name + '_v%d.vtu' % itime)
                vtk.write_vector(x_i_list[0], 'velocity', sf_nd_nb.vel_func_space)
                vtk.write_end()
                vtk = output.File(config.filename + config.case_name + '_p%d.vtu' % itime)
                vtk.write_scaler(x_i_list[1], 'pressure', sf_nd_nb.pre_func_space)
                vtk.write_end()

                continue

            if sf_nd_nb.isTransient:
                if itime <= config.time_order:
                    if itime == 1:
                        sf_nd_nb.set_data(bdfscm=cmmn_data.BDFdata(order=1))
                        alpha_u_n *= 0
                        alpha_u_n += u_n.view(alpha_u_n.shape)
                    elif itime == 2:
                        sf_nd_nb.set_data(bdfscm=cmmn_data.BDFdata(order=2))
                        alpha_u_n *= 0
                        alpha_u_n += sf_nd_nb.bdfscm.alpha[0] * u_n.view(alpha_u_n.shape)
                        alpha_u_n += sf_nd_nb.bdfscm.alpha[1] * torch.tensor(
                            u_all[itime - 2, :, :, :],
                            device=dev, dtype=torch.float64).view(alpha_u_n.shape)
                    elif itime == 3:
                        sf_nd_nb.set_data(bdfscm=cmmn_data.BDFdata(order=3))
                        alpha_u_n *= 0
                        alpha_u_n += sf_nd_nb.bdfscm.alpha[0] * u_n.view(alpha_u_n.shape)
                        alpha_u_n += sf_nd_nb.bdfscm.alpha[1] * torch.tensor(
                            u_all[itime - 2, :, :, :],
                            device=dev, dtype=torch.float64).view(alpha_u_n.shape)
                        alpha_u_n += sf_nd_nb.bdfscm.alpha[2] * torch.tensor(
                            u_all[itime - 3, :, :, :],
                            device=dev, dtype=torch.float64).view(alpha_u_n.shape)
                else:
                    alpha_u_n *= 0
                    alpha_u_n += sf_nd_nb.bdfscm.alpha[0] * u_n.view(alpha_u_n.shape)
                    for i in range(1, config.time_order):
                        alpha_u_n += sf_nd_nb.bdfscm.alpha[i] * torch.tensor(
                            u_all[itime - i - 1, :, :, :],
                            device=dev, dtype=torch.float64).view(alpha_u_n.shape)

            t += dt
            print('====physical time: ', t, ' ====')
            # get boundary and rhs body force condition
            u_bc, f, fNorm = bc_f.vel_bc_f(
                ndim,
                vel_func_space.bc_node_list,
                vel_func_space.x_all,
                prob=config.problem,
                t=t
            )

            x_i *= 0
            x_i += x_i_n  # use last timestep p as start value

            r0l2 = torch.tensor(1, device=dev, dtype=torch.float64)  # linear solver residual l2 norm
            its = 0  # linear solver iteration
            nr0l2 = 1  # non-linear solver residual l2 norm
            nits = 0  # newton iteration step
            r0 *= 0

            total_its = 0  # total linear iteration number / restart

            while nits < config.n_its_max:
                sf_nd_nb.Kmatinv = None
                u_k *= 0
                u_k += x_i_list[0].view(u_k.shape)  # non-linear velocity
                x_rhs *= 0
                x_rhs = volume_mf_st.get_rhs(
                    x_rhs=x_rhs, u_bc=u_bc, f=f,
                    include_adv=config.include_adv,
                    u_n=alpha_u_n,  # previous *timesteps* velocity multiplied by BDF extrapolation coeffs
                    isAdvExp=config.isAdvExp,  # whether to treat advection explicity
                    u_k=alpha_u_n,  # non-lienar velocity
                )
                # get non-linear residual
                r0 *= 0
                r0 = volume_mf_st.get_residual_only(
                    r0, x_i, x_rhs,
                    include_adv=config.include_adv,
                    u_n=u_k,  # last *non-linear iteration step* velocity
                    u_bc=u_bc[0],
                )
                nr0l2 = torch.linalg.norm(r0)  # stokes is linear prob, we will simply use linear residual as the non-linear residual
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
                    include_adv=config.include_adv,
                    u_n=u_k,
                    u_bc=u_bc[0]
                )
                from scipy.sparse import bsr_matrix

                if not config.is_sfc:
                    RAR = bsr_matrix((RARvalues.cpu().numpy(), cola, fina),
                                     shape=((ndim) * cg_nonods, (ndim) * cg_nonods))
                    sf_nd_nb.set_data(RARmat=RAR.tocsr())
                # np.savetxt('RAR.txt', RAR.toarray(), delimiter=',')
                RARvalues = torch.permute(RARvalues, (1, 2, 0)).contiguous()  # (ndim, ndim, ncola)
                # get SFC, coarse grid and operators on coarse grid. Store them to save computational time?
                space_filling_curve_numbering, variables_sfc, nlevel, nodes_per_level = \
                    mg.mg_on_P0CG_prep(fina, cola, RARvalues)
                sf_nd_nb.sfc_data.set_data(
                    space_filling_curve_numbering=space_filling_curve_numbering,
                    variables_sfc=variables_sfc,
                    nlevel=nlevel,
                    nodes_per_level=nodes_per_level
                )
                # print('9. time elapsed, ', time.time() - starttime)

                # dp_i *= 0  # solve for delta p_i and u_i
                if config.linear_solver == 'gmres-mg':
                    # nullspace = torch.zeros(u_nonods * ndim + p_nonods, device=dev, dtype=torch.float64)
                    # nullspace[u_nonods * ndim:-1] += 1.
                    # nullspace = nullspace.view(1, -1)
                    # x_i, its = solvers.gmres_mg_solver(
                    #     x_i, x_rhs,
                    #     tol=1e-12,  # max(min(1.e-5*nr0l2, 1.e-5), 1.e-11),
                    #     include_adv=config.include_adv,
                    #     u_k=u_k,
                    #     u_bc=u_bc[0],
                    #     nullspace=nullspace,
                    # )
                    x_i, its = solvers.gmres_mg_solver(
                        x_i, x_rhs,
                        tol=max(min(1.e-1*nr0l2, 1.e-1), 1.e-11),  # config.tol,
                        include_adv=config.include_adv,
                        u_k=u_k,
                        u_bc=u_bc[0]
                    )
                    total_its += its
                elif config.linear_solver == 'right-gmres-mg':
                    raise ValueError('right-gmre-mg solver is not implemented!')
                    x_i = solvers.right_gmres_mg_solver(x_i, x_rhs, config.tol)
                else:
                    raise Exception('choose a valid solver...')
                # get final residual after we're back to fine grid
                r0 *= 0
                r0 = volume_mf_st.get_residual_only(
                    r0, x_i, x_rhs,
                    include_adv=config.include_adv,
                    u_n=u_k,
                    u_bc=u_bc[0],
                )

                r0l2 = torch.linalg.norm(r0) / fNorm
                r0l2all.append(r0l2.cpu().numpy())
                print('its=', its, 'residual l2 norm=', r0l2.cpu().numpy(),
                      'abs res=', torch.linalg.norm(r0).cpu().numpy(),
                      'fNorm', fNorm.cpu().numpy())

                # nr0l2 = r0l2

                nits += 1

            # if converges,
            x_i_n *= 0
            x_i_n += x_i  # store this step in case we want to use this for next timestep

            # save x_i at this Re to reuse as the initial condition for higher Re
            torch.save(x_i, config.filename + config.case_name + '_t%.2f.pt' % t)
            print('total its / restart ', total_its)
            total_its = 0

            # combine inner/inter element contribution
            u_all[itime, :, :, :] = x_i_list[0].cpu().numpy()
            p_all[itime, ...] = x_i_list[1].cpu().numpy()

            # output to vtk
            vtk = output.File(config.filename + config.case_name + '_v%d.vtu' % itime)
            vtk.write_vector(x_i_list[0], 'velocity', sf_nd_nb.vel_func_space)
            vtk.write_end()
            vtk = output.File(config.filename + config.case_name + '_p%d.vtu' % itime)
            vtk.write_scaler(x_i_list[1], 'pressure', sf_nd_nb.pre_func_space)
            vtk.write_end()

            # get l2 error
            x_ana = bc_f.ana_soln(config.problem, t=t)
            x_ana[0:u_nonods*ndim] += torch.tensor(
                u_all[itime - 1, :, :, :],
                device=dev, dtype=torch.float64).view(-1)
            x_ana[u_nonods * ndim: u_nonods*ndim + p_nonods] += torch.tensor(
                p_all[itime - 1, ...],
                device=dev, dtype=torch.float64).view(-1)
            if config.hasNullSpace:
                # remove average from pressure
                p_ave = ns_assemble.get_ave_pressure(x_i_list[1].cpu().numpy())
                x_i[u_nonods * ndim:u_nonods * ndim + p_nonods] -= p_ave
                p_ana = x_ana[u_nonods * ndim:u_nonods * ndim + p_nonods]
                p_ana_ave = ns_assemble.get_ave_pressure(p_ana.cpu().numpy())
                p_ana -= p_ana_ave
            u_l2, p_l2, u_linf, p_linf = volume_mf_st.get_l2_error(x_i, x_ana)
            print('after solving, compare to previous timestep, l2 error is: \n',
                  'velocity ', u_l2, '\n',
                  'pressure ', p_l2)
            print('l infinity error is: \n',
                  'velocity ', u_linf, '\n',
                  'pressure ', p_linf)
            # print('total its / restart ', total_its)

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
    # rhs = integral_mf.get_rhs(
    #     x_rhs=rhs,
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
    #         x_dummy = integral_mf.get_residual_only(
    #             r0=x_dummy,
    #             x_i=probe,
    #             x_rhs=0,
    #             include_p=True,
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
    #     x_dummy = integral_mf.get_residual_only(
    #         r0=x_dummy,
    #         x_i=probe,
    #         x_rhs=0,
    #         include_p=True,
    #     )
    #     # put in Amat
    #     Amat[:, ndim*u_nonods + inod] -= x_dummy
    # Amat_np = Amat.cpu().numpy()
    # Amat_sp = sp.sparse.csr_matrix(Amat_np)

    print('im going to assemble', time.time()-starttime)
    # save bc to vtk to check if it's correct
    vtk = output.File('bc_diri.vtu')
    vtk.write_vector(u_bc[0], 'velocity_diri', sf_nd_nb.vel_func_space)
    vtk.write_end()
    if len(u_bc) > 1:  # has neumann bc
        vtk = output.File('bc_neu.vtu')
        vtk.write_vector(u_bc[1], 'velocity_neu', sf_nd_nb.vel_func_space)
        vtk.write_end()
    indices_st = []
    values_st = []
    rhs_np, indices_st, values_st = ns_assemble.assemble(u_bc, f, indices_st, values_st)

    if False:
        # apply velocity boundary strongly
        bclist = np.zeros(u_nonods*ndim+p_nonods, dtype=bool)
        for bci in sf_nd_nb.vel_func_space.bc:
            for inod in bci:
                bclist[inod*ndim:(inod+1)*ndim] = True
        # print('boundary nodes list: ', bclist)
        # set i'th rows and columns in Amat be 0 except for diagonal.
        # for all i in bclist
        Amat_ncola = Amat_sp.nnz
        Amat_cola = Amat_sp.indices
        Amat_fina = Amat_sp.indptr
        u_bc = u_bc.reshape(-1)
        for inod in range(u_nonods*ndim+p_nonods):
            if bclist[inod]:
                # this is boundary node row, just set everything to 0
                Amat_sp.data[Amat_fina[inod]:Amat_fina[inod+1]] *= 0
                for j in range(Amat_fina[inod], Amat_fina[inod + 1]):
                    jnod = Amat_cola[j]
                    if inod == jnod:
                        # we're on boundary nodes' diagonal
                        Amat_sp.data[j] += 1
                        rhs_np[inod] *= 0
                        rhs_np[inod] += u_bc[inod]
            else:
                for j in range(Amat_fina[inod], Amat_fina[inod+1]):
                    jnod = Amat_cola[j]
                    if bclist[jnod]:
                        if inod != jnod:
                            # this is a boundary node, and we're not on diagonal
                            # put off-diagonal contribution to rhs then set entry to 0
                            rhs_np[inod] -= Amat_sp.data[j] * u_bc[jnod]
                            Amat_sp.data[j] *= 0
    if False:  # liftig pressure 0 node
        pre_ref_node = u_nonods*ndim  # pressure reference node
        Amat_ncola = Amat_sp.nnz
        Amat_cola = Amat_sp.indices
        Amat_fina = Amat_sp.indptr
        for inod in range(u_nonods*ndim+p_nonods):
            if inod == pre_ref_node:
                # this is the row of pressure reference node, set everything to 0
                Amat_sp.data[Amat_fina[inod]:Amat_fina[inod+1]] *= 0
                for j in range(Amat_fina[inod], Amat_fina[inod+1]):
                    jnod = Amat_cola[j]
                    if inod == jnod:
                        # this is the diagonal
                        Amat_sp.data[j] += 1
                        rhs_np[inod] *= 0  # we are setting ref pressure node to 0
            else:
                for j in range(Amat_fina[inod], Amat_fina[inod+1]):
                    jnod = Amat_cola[j]
                    if jnod == pre_ref_node:
                        # this is the column of ref pressure node
                        if inod != jnod:  # and, we're not on diagonal
                            rhs_np[inod] -= Amat_sp.data[j] * 0
                            Amat_sp.data[j] *= 0

    print('done assemble, going to write', time.time()-starttime)
    # Amat_np = Amat_sp.todense()
    # print('Amat cond number is: ', np.linalg.cond(Amat_np))
    # print('Amat rank is: ', np.linalg.matrix_rank(Amat_np))
    # np.savetxt('rhs_assemble.txt', rhs_np, delimiter=',')
    # np.savetxt('Amat_assemble.txt', Amat_np, delimiter=',')
    print('doing writing. im goingt o solve', time.time()-starttime)
    x_sol = np.zeros(u_nonods*ndim+p_nonods, dtype=np.float64)
    for nit in tqdm(range(config.n_its_max), disable=config.disabletqdm):
        # non-linear iteration
        indices_ns = []
        values_ns = []
        indices_ns += indices_st
        values_ns += values_st
        # x_sol *= 0
        # x_sol += 1
        # u_bc[0] *= 0
        # u_bc[0] += 1
        # np.savetxt('u_bc.txt', u_bc[0].view(-1).cpu().numpy(), delimiter=',')
        rhs_c, indices_ns, values_ns  = ns_assemble.assemble_adv(x_sol, u_bc, indices_ns, values_ns)
        if config.problem == 'stokes':
            indices_ns = indices_st
            values_ns = values_st
            # Cmat_sp *= 0; rhs_c *= 0  # back to stokes system

        # Cmat_np = Cmat_sp.todense()
        # np.savetxt('Cmat_assemble.txt', Cmat_np, delimiter=',')

        # Allmat = Amat_sp + Cmat_sp  # this step is memory hungry because sparse matrix is convereted to dense
        Allmat = ns_assemble.assemble_csr_mat(indices_ns, values_ns)
        rhs_all = rhs_np + rhs_c
        # # get non-linear residual
        # rhs_all -= Allmat @ x_sol
        non_linear_res_l2 = np.linalg.norm(rhs_all - Allmat @ x_sol)
        print('nit = ', nit, 'residual l2 norm = ', non_linear_res_l2)
        if non_linear_res_l2 < config.n_tol:
            print('non-linear iteration converged.')
            break
        x_sol = sp.sparse.linalg.spsolve(Allmat, rhs_all)
        # # remove average from pressure
        # p_sol = x_sol[u_nonods*ndim:u_nonods*ndim+p_nonods]
        # p_ave = ns_assemble.get_ave_pressure(p_sol)
        # p_sol -= p_ave

    # get l2 error
    x_ana = bc_f.ana_soln(config.problem, t=tend)
    if config.hasNullSpace:
        # remove average from pressure
        p_sol = x_sol[u_nonods * ndim:u_nonods * ndim + p_nonods]
        p_ave = ns_assemble.get_ave_pressure(p_sol)
        p_sol -= p_ave
        p_ana = x_ana[u_nonods*ndim:u_nonods*ndim+p_nonods]
        p_ana_ave = ns_assemble.get_ave_pressure(p_ana.cpu().numpy())
        p_ana -= p_ana_ave
    u_l2, p_l2, u_linf, p_linf = volume_mf_st.get_l2_error(x_sol, x_ana)
    print('after solving, l2 error is: \n',
          'velocity ', u_l2, '\n',
          'pressure ', p_l2)
    print('l infinity error is: \n',
          'velocity ', u_linf, '\n',
          'pressure ', p_linf)
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
# print(torch.cuda.memory_summary())
