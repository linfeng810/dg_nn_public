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
import petrov_galerkin
import shape_function
import sparsity
import volume_mf_st, volume_mf_um, volume_mf_he, volume_mf_um_on_fix_mesh
from function_space import FuncSpace, Element
import solvers
from config import sf_nd_nb
import mesh_init
# from mesh_init import face_iloc,face_iloc2
# from shape_function import SHATRInew, det_nlx, sdet_snlx
# import volume_mf_st as integral_mf
from color import color2
import multigrid_linearelastic as mg
import bc_f
import time
import pressure_matrix
import ns_assemble
import materials

starttime = time.time()

# for pretty print out torch tensor
# torch.set_printoptions(sci_mode=False)
torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)

dev = config.dev
nele = config.nele
nele_f = config.nele_f
nele_s = config.nele_s
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
quad_degree = config.ele_p*2
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
pre_func_space = FuncSpace(pre_ele, name="Pressure", mesh=config.mesh, dev=dev,
                           not_iso_parametric=True, x_element=vel_ele)  # super-parametric pressure ele.
sf_nd_nb.set_data(vel_func_space=vel_func_space,
                  pre_func_space=pre_func_space,
                  p1cg_nonods=vel_func_space.cg_nonods)

disp_func_space = FuncSpace(vel_ele, name="Displacement", mesh=config.mesh, dev=dev,
                            get_pndg_ndglbno=True)  # displacement func space
sf_nd_nb.set_data(disp_func_space=disp_func_space)

material = materials.NeoHookean(sf_nd_nb.disp_func_space.element.nloc,
                                ndim, dev, config.mu_s, config.lam_s)
# material = materials.LinearElastic(sf_nd_nb.disp_func_space.element.nloc,
#                                    ndim, dev, config.mu_s, config.lam_s)
# material = materials.STVK(sf_nd_nb.disp_func_space.element.nloc,
#                           ndim, dev, config.mu_s, config.lam_s)
sf_nd_nb.set_data(material=material)

fluid_spar, solid_spar = sparsity.get_subdomain_sparsity(
    vel_func_space.cg_ndglno,
    config.nele_f,
    config.nele_s,
    vel_func_space.cg_nonods
)
sf_nd_nb.set_data(sparse_f=fluid_spar, sparse_s=solid_spar)

if False:  # test mesh displacement
    x_i = torch.zeros(vel_func_space.nonods * ndim * 2 + pre_func_space.nonods, device=dev, dtype=torch.float64)
    x_i_dict = volume_mf_st.slicing_x_i(x_i)
    x_i_dict['vel'] += (torch.sin(vel_func_space.x_ref_in[:, 0, :]) *
                        torch.sin(vel_func_space.x_ref_in[:, 1, :])).unsqueeze(dim=-1).expand(-1, -1, ndim)
    x_i_dict['pre'] += torch.tensor(
        np.sin(pre_func_space.x_all[:, 0]) * np.sin(pre_func_space.x_all[:, 1]),
        device=dev, dtype=torch.float64
    ).view(nele, -1)
    for itime in range(30):
        print('itime', itime)
        x_i_dict['disp'][nele_f:nele, ...] += (
            torch.ones(nele_s, disp_func_space.element.nloc, ndim, device=dev, dtype=torch.float64,)
                                              ) * 0.01
        x_i_dict['disp'] = volume_mf_um.solve_for_mesh_disp(x_i_dict['disp'])
        # x_i_dict['disp'] = volume_mf_um_on_fix_mesh.solve_for_mesh_disp(x_i_dict['disp'])

        # move velocity mesh
        sf_nd_nb.vel_func_space.x_ref_in *= 0
        sf_nd_nb.vel_func_space.x_ref_in += sf_nd_nb.disp_func_space.x_ref_in \
                                            + x_i_dict['disp'].permute(0, 2, 1)

        sf_nd_nb.pre_func_space.x_ref_in *= 0
        sf_nd_nb.pre_func_space.x_ref_in += sf_nd_nb.vel_func_space.x_ref_in

        # update x_all after moving mesh
        sf_nd_nb.vel_func_space.get_x_all_after_move_mesh()
        sf_nd_nb.pre_func_space.get_x_all_after_move_mesh()

        fsi_output.output_fsi_vtu(x_i, vel_func_space, pre_func_space, disp_func_space, itime)
    raise SystemExit("Code stopped at this point")

print('nele=', nele)

# if True:# P1CG connectivity and coloring
#     fina, cola, ncola = mesh_init.p1cg_sparsity(sf_nd_nb.vel_func_space)
#     whichc, ncolor = color2(fina=fina, cola=cola, nnode=sf_nd_nb.p1cg_nonods)
# # np.savetxt('whichc.txt', whichc, delimiter=',')
# print('ncolor', ncolor, 'whichc type', whichc.dtype)
# print('cg_nonods', sf_nd_nb.p1cg_nonods, 'ncola (p1cg sparsity)', ncola)
print('1. time elapsed, ',time.time()-starttime)

"""solve a fsi problem"""
u_bc, f, fNorm = bc_f.fsi_bc(ndim, vel_func_space.bc_node_list, vel_func_space.x_all,
                             prob=config.problem,
                             t=0)
"""as a test we first solve a fluid only problem"""
# u_bc, f, fNorm = bc_f.vel_bc_f(ndim, vel_func_space.bc_node_list, vel_func_space.x_all,
#                                prob=config.problem,
#                                t=0)
"""as another test we solve a solid only problem"""
# u_bc, f, fNorm = bc_f.he_bc_f(ndim, disp_func_space.bc_node_list, disp_func_space.x_all,
#                               prob=config.problem
#                               )

tstep = int(np.ceil((tend-tstart)/dt)) + 1
if not sf_nd_nb.isTransient:
    tstep = 2
    sf_nd_nb.set_data(bdfscm=cmmn_data.BDFdata(order=config.time_order))
else:
    sf_nd_nb.set_data(bdfscm=cmmn_data.BDFdata(order=config.time_order))

print('no of time steps to compute: ', tstep)

# u = torch.zeros(nele, vel_ele.nloc, ndim, device=dev, dtype=torch.float64)  # initial guess
# p = torch.zeros(nele, pre_ele.nloc, device=dev, dtype=torch.float64)
# # to test vtk output, set u and p be ana sol here
# if False:
#     u = u.view(vel_func_space.nonods, ndim)
#     x = vel_func_space.x_all[:, 0]
#     y = vel_func_space.x_all[:, 1]
#     z = vel_func_space.x_all[:, 2]
#     u[:, 0] += torch.tensor(-2./3. * np.sin(x)**3, device=dev)
#     u[:, 1] += torch.tensor(
#         np.sin(x)**2 * (y * np.cos(x) - z * np.sin(x)),
#         device=dev
#     )
#     u[:, 2] += torch.tensor(
#         np.sin(x)**2 * (z * np.cos(x) + y * np.sin(x)),
#         device=dev
#     )
#     p = p.view(pre_func_space.nonods)
#     x = pre_func_space.x_all[:, 0]
#     p += torch.tensor(np.sin(x), device=dev)
#
# # u = torch.ones_like(u)
# u_all = np.empty([tstep, nele, vel_ele.nloc, ndim])
# u_all[0, :, :, :] = u.view(nele, vel_ele.nloc, ndim).cpu().numpy()
# p_all = np.empty([tstep, nele, pre_ele.nloc])
# p_all[0, ...] = p.view(nele, -1).cpu().numpy()

# print('5. time elapsed, ',time.time()-starttime)
# print('6. time elapsed, ',time.time()-starttime)

# if False:  # per element condensation Rotation matrix (diagonal)
#     R = torch.tensor([1./30., 1./30., 1./30., 3./40., 3./40., 3./40.,
#                      3./40., 3./40., 3./40., 9./20.], device=dev, dtype=torch.float64)
#     # R = torch.tensor([1./10., 1./10., 1./10., 1./10., 1./10., 1./10.,
#     #         1./10., 1./10., 1./10., 1./10.], device=dev, dtype=torch.float64)
# if True:  # from PnDG to P1CG
#     p1dg_nonods = vel_func_space.p1dg_nonods
#     cg_ndglno = vel_func_space.cg_ndglno
#     cg_nonods = vel_func_space.cg_nonods
#     I_fc_colx = np.arange(0, p1dg_nonods)
#     I_fc_coly = cg_ndglno
#     I_fc_val = np.ones(p1dg_nonods)
#     I_cf = coo_matrix((I_fc_val, (I_fc_coly, I_fc_colx)),
#                       shape=(cg_nonods, p1dg_nonods))  # fine to coarse == DG to CG
#     I_cf = I_cf.tocsr()
#     print('don assemble I_cf I_fc', time.time()-starttime)
#     no_dgnodes = I_cf.sum(axis=1)
#     print('done suming', time.time()-starttime)
#     # weight by 1 / overlapping dg nodes number
#     for i in range(p1dg_nonods):
#         I_fc_val[i] /= no_dgnodes[I_fc_coly[i]]
#     print('done weighting', time.time()-starttime)
#     # for i in tqdm(range(cg_nonods)):
#     #     # no_dgnodes = np.sum(I_cf[i,:])
#     #     I_cf[i,:] /= no_dgnodes[i]
#     #     I_fc[:,i] /= no_dgnodes[i]
#     # print('done weighting', time.time()-starttime)
#     I_cf = coo_matrix((I_fc_val, (I_fc_coly, I_fc_colx)),
#                       shape=(cg_nonods, p1dg_nonods))  # fine to coarse == DG to CG
#     I_cf = I_cf.tocsr()
#     I_fc = coo_matrix((I_fc_val, (I_fc_colx, I_fc_coly)),
#                       shape=(p1dg_nonods, cg_nonods))  # coarse to fine == CG to DG
#     I_fc = I_fc.tocsr()
#     print('done transform to csr', time.time()-starttime)
#     # transfer to torch device
#     I_fc = torch.sparse_csr_tensor(crow_indices=torch.tensor(I_fc.indptr),
#                                    col_indices=torch.tensor(I_fc.indices),
#                                    values=I_fc.data,
#                                    size=(p1dg_nonods, cg_nonods),
#                                    device=dev)
#     I_cf = torch.sparse_csr_tensor(crow_indices=torch.tensor(I_cf.indptr),
#                                    col_indices=torch.tensor(I_cf.indices),
#                                    values=I_cf.data,
#                                    size=(cg_nonods, p1dg_nonods),
#                                    device=dev)
#     sf_nd_nb.set_data(I_dc=I_fc, I_cd=I_cf)
#         np.savetxt('I_fc.txt', I_fc.to_dense().cpu().numpy(), delimiter=',')
#         np.savetxt('I_cf.txt', I_cf.to_dense().cpu().numpy(), delimiter=',')
print('7. time elapsed, ',time.time()-starttime)

u_nonods = sf_nd_nb.vel_func_space.nonods
p_nonods = sf_nd_nb.pre_func_space.nonods
u_nloc = sf_nd_nb.vel_func_space.element.nloc
if False and config.isoparametric:  # test curvilinear elements
    # now we are deforming the mesh accroding to a known displacement:
    # d = (i j k) * a sin(pi(x+1)) sin(pi(y+1)) sin(pi(z+1))
    # a = 0.15
    # vel space
    v_x_all = sf_nd_nb.vel_func_space.x_all
    if ndim == 2:
        d_x_all = 0.15 * np.sin(np.pi * (v_x_all[:,0] + 1)) * np.sin(np.pi * (v_x_all[:,1] + 1))
        d_x_all = np.stack((d_x_all, d_x_all), axis=-1)
    else:  # ndim == 3
        d_x_all = 0.15 * np.sin(np.pi * (v_x_all[:, 0] + 1)) * np.sin(np.pi * (v_x_all[:, 1] + 1)) \
            * np.sin(np.pi * (v_x_all[:, 2] + 1))
        d_x_all = np.stack((d_x_all, d_x_all, d_x_all), axis=-1)
    sf_nd_nb.vel_func_space.x_all += d_x_all
    sf_nd_nb.vel_func_space.x_ref_in += torch.tensor(d_x_all, device=dev, dtype=torch.float64
                                                     ).view(nele, -1, ndim).transpose(dim0=-1, dim1=-2)
    # pre space
    p_x_all = sf_nd_nb.pre_func_space.x_all
    if ndim == 2:
        d_x_all = 0.15 * np.sin(np.pi * (p_x_all[:, 0] + 1)) * np.sin(np.pi * (p_x_all[:, 1] + 1))
        d_x_all = np.stack((d_x_all, d_x_all), axis=-1)
    else:  # ndim == 3
        d_x_all = 0.15 * np.sin(np.pi * (p_x_all[:, 0] + 1)) * np.sin(np.pi * (p_x_all[:, 1] + 1)) \
            * np.sin(np.pi * (p_x_all[:, 2] + 1))
        d_x_all = np.stack((d_x_all, d_x_all, d_x_all), axis=-1)
    sf_nd_nb.pre_func_space.x_all += d_x_all
    sf_nd_nb.pre_func_space.x_ref_in += torch.tensor(d_x_all, device=dev, dtype=torch.float64
                                                     ).view(nele, -1, ndim).transpose(dim0=-1, dim1=-2)

if not config.isFSI:
    no_total_dof = nele * (vel_ele.nloc * ndim+ pre_ele.nloc)  # velocity, pressure
else:
    no_total_dof = nele * (vel_ele.nloc * ndim * 2 + pre_ele.nloc)  # velocity, displacmeent, pressure
if config.solver=='iterative':
    print('i am going to time loop')
    print('8. time elapsed, ',time.time()-starttime)
    # print("Using quit()")
    # quit()
    r0l2all = []
    # time loop
    r0 = torch.zeros(no_total_dof,
                     device=dev, dtype=torch.float64)
    r0_dict = volume_mf_st.slicing_x_i(r0)
    if True:
        x_i = torch.zeros(no_total_dof, device=dev, dtype=torch.float64)
        x_rhs = torch.zeros(no_total_dof, device=dev, dtype=torch.float64)
        u_m = torch.zeros(nele, u_nloc, ndim, device=dev, dtype=torch.float64)
        sf_nd_nb.set_data(u_m=u_m.view(nele, -1, ndim))
        x_i_dict = volume_mf_st.slicing_x_i(x_i)
        # x_i_n = torch.zeros(no_total_dof, device=dev, dtype=torch.float64)  # last step soln
        # x_i_n_dict = volume_mf_st.slicing_x_i(x_i_n)
        # let's create a list of tensors to store J+1 previoius timestep values
        # for time integrator
        # x_all_previous[0] is x_all_t (previoius timestep)
        # x_all_previous[i] is x_all_(t-i) (i step before)
        x_all_previous = [volume_mf_st.slicing_x_i(torch.zeros(no_total_dof, dtype=torch.float64, device=dev))
                          for _ in range(config.time_order+1)]
        # non-linear velocity/displacement/pressure (don't actually needs pressure though)
        x_i_k = torch.zeros(no_total_dof, device=dev, dtype=torch.float64)  # non-linear velocity/disp/pressure
        x_i_k_dict = volume_mf_st.slicing_x_i(x_i_k)

        # DEBUG: assemble Lp col by col
        if False:
            import Lp_assemble_colbycol
            Lpmat = Lp_assemble_colbycol.vanilla_assemble_Lp()
            np.savetxt('Lpmat_colbycolassemble.txt', Lpmat.todense(), delimiter=',')
        # solve a stokes problem as initial velocity
        if config.initialCondition == 2:
            raise ValueError('using Stokes soln as initial cond is not yet supported in FSI solver...')
            # print('going to solve steady stokes problem as initial condition')
            # sf_nd_nb.isTransient = False  # temporarily omit transient terms
            # sf_nd_nb.use_fict_dt_in_vel_precond = False
            # sf_nd_nb.isES = False
            # x_i *= 0
            # x_i += x_i_n  # use last timestep p as start value
            # r0 *= 0
            # x_rhs *= 0
            # x_rhs = volume_mf_st.get_rhs(
            #     x_rhs=x_rhs, u_bc=u_bc, f=f,
            #     include_adv=False,
            #     u_n=u_n  # last *timestep* velocity
            # )
            # RARvalues = integral_mf.calc_RAR_mf_color(
            #     I_fc, I_cf,
            #     whichc, ncolor,
            #     fina, cola, ncola,
            #     include_adv=False,
            #     u_n=u_k,
            #     u_bc=u_bc[0]
            # )
            # from scipy.sparse import bsr_matrix
            #
            # if not config.is_sfc:
            #     RAR = bsr_matrix((RARvalues.cpu().numpy(), cola, fina),
            #                      shape=((ndim) * cg_nonods, (ndim) * cg_nonods))
            #     sf_nd_nb.set_data(RARmat=RAR.tocsr())
            # # np.savetxt('RAR.txt', RAR.toarray(), delimiter=',')
            # RARvalues = torch.permute(RARvalues, (1, 2, 0)).contiguous()  # (ndim, ndim, ncola)
            # # get SFC, coarse grid and operators on coarse grid. Store them to save computational time?
            # space_filling_curve_numbering, variables_sfc, nlevel, nodes_per_level = \
            #     mg.mg_on_P1CG_prep(fina, cola, RARvalues)
            # sf_nd_nb.sfc_data.set_data(
            #     space_filling_curve_numbering=space_filling_curve_numbering,
            #     variables_sfc=variables_sfc,
            #     nlevel=nlevel,
            #     nodes_per_level=nodes_per_level
            # )
            #
            # if config.linear_solver == 'gmres-mg':
            #     x_i, _ = solvers.gmres_mg_solver(
            #         x_i, x_rhs,
            #         tol=config.tol,
            #         include_adv=False,
            #         u_k=u_k,
            #         u_bc=u_bc[0]
            #     )
            # print('finishing solving stokes problem as initial conidtion.')
            # x_i_n *= 0
            # x_i_n += x_i
            #
            # sf_nd_nb.isTransient = config.isTransient  # change back to settings in config
            # sf_nd_nb.use_fict_dt_in_vel_precond = config.use_fict_dt_in_vel_precond
            # sf_nd_nb.isES = config.isES

        elif config.initialCondition == 1:
            x_all_previous[0]['all'] *= 0
            # ana_sln = bc_f.ana_soln(problem=config.problem, t=tstart)
            # x_all_previous[0]['vel'] += ana_sln[0:u_nonods*ndim].view(nele, -1, ndim)
            # x_all_previous[0]['pre'] += ana_sln[u_nonods*ndim:].view(nele, -1)

        elif config.initialCondition == 3:
            x_all_previous[0]['all'] *= 0
            x_all_previous[0]['all'] += torch.load(config.initDataFile)

        p_ana_ave = 0
        if config.hasNullSpace:
            # remove average from pressure
            p_ana = x_all_previous[0]['pre']
            p_ana_ave = ns_assemble.get_ave_pressure(p_ana.cpu().numpy())

        x_i *= 0
        x_i += x_all_previous[0]['all']
        # u_all[0, :, :, :] = x_i_list[0].cpu().numpy()
        # p_all[0, ...] = x_i_list[1].cpu().numpy()

        # save initial condition to vtk
        fsi_output.output_fsi_vtu(x_i, vel_func_space, pre_func_space, disp_func_space, itime=0)

        t = tstart  # physical time (start time)

        alpha_u_n = torch.zeros(u_nonods, ndim, device=dev, dtype=torch.float64).view(nele, -1, ndim)
        d_n_dt = torch.zeros_like(alpha_u_n, device=dev, dtype=torch.float64)  # disp. (1st deri.) integrator w/o dt
        d_n_dt2 = torch.zeros_like(alpha_u_n, device=dev, dtype=torch.float64)  # disp. (2nd deri.) integrator w/o dt

        if sf_nd_nb.isPetrovGalerkin:  # get projection operator
            sf_nd_nb.projection_one_order_lower = petrov_galerkin.get_projection_one_order_lower(
                k=vel_func_space.element.ele_order,
                ndim=config.ndim
            )

        for itime in range(1, tstep):  # time loop
            wall_time_start = time.time()
            sf_nd_nb.ntime = itime
            # for the starting steps, use 1st, 2nd then 3rd order BDF.
            if False and itime < config.time_order:  # has analytical soln
                print('itime = ', itime, 'getting ana soln...')
                t += dt
                x_i *= 0
                x_i += bc_f.ana_soln(config.problem, t=t)

                # if converges,
                x_i_n *= 0
                x_i_n += x_i  # store this step in case we want to use this for next timestep

                # output to vtk
                fsi_output.output_fsi_vtu(x_i, vel_func_space, pre_func_space, disp_func_space, itime)

                continue

            if sf_nd_nb.isTransient:
                if itime <= config.time_order:
                    if itime == 1:
                        sf_nd_nb.set_data(bdfscm=cmmn_data.BDFdata(order=1))
                        # alpha_u_n *= 0
                        # alpha_u_n += x_all_previous[0]['vel'].view(alpha_u_n.shape)
                        # d_n_dt *= 0
                        # d_n_dt -= sf_nd_nb.bdfscm.alpha[0] * x_all_previous[0]['disp'].view(d_n_dt.shape)
                        # d_n_dt2 *= 0
                        # d_n_dt2 += sf_nd_nb.bdfscm.beta[1] * x_all_previous[0]['disp'].view(d_n_dt2.shape)
                        # # assume 0 initial displacement for displacement thus no need to account for
                        # # displacement at tstep=-1:
                        # #   d_n_dt2 += d_(n-1) * sf_nd_nb.bdfscm.beta[1]
                    elif itime == 2:
                        sf_nd_nb.set_data(bdfscm=cmmn_data.BDFdata(order=2))
                        # alpha_u_n *= 0
                        # alpha_u_n += sf_nd_nb.bdfscm.alpha[0] * x_all_previous[0]['vel'].view(alpha_u_n.shape)
                        # alpha_u_n += sf_nd_nb.bdfscm.alpha[1] * x_all_previous[1]['vel'].view(alpha_u_n.shape)
                        # d_n_dt *= 0
                        # d_n_dt -= sf_nd_nb.bdfscm.alpha[0] * x_all_previous[0]['disp'].view(d_n_dt.shape)
                        # d_n_dt -= sf_nd_nb.bdfscm.alpha[1] * x_all_previous[1]['disp'].view(d_n_dt.shape)
                        # d_n_dt2 *= 0
                        # d_n_dt2 += sf_nd_nb.bdfscm.beta[1] * x_all_previous[0]['disp'].view(d_n_dt2.shape)
                        # d_n_dt2 += sf_nd_nb.bdfscm.beta[2] * x_all_previous[1]['disp'].view(d_n_dt2.shape)
                        # d_n_dt2 += sf_nd_nb.bdfscm.beta[3] * x_all_previous[2]['disp'].view(d_n_dt2.shape)
                    elif itime == 3:
                        sf_nd_nb.set_data(bdfscm=cmmn_data.BDFdata(order=3))
                        # alpha_u_n *= 0
                        # alpha_u_n += sf_nd_nb.bdfscm.alpha[0] * x_all_previous[0]['vel'].view(alpha_u_n.shape)
                        # alpha_u_n += sf_nd_nb.bdfscm.alpha[1] * x_all_previous[1]['vel'].view(alpha_u_n.shape)
                        # alpha_u_n += sf_nd_nb.bdfscm.alpha[2] * x_all_previous[2]['vel'].view(alpha_u_n.shape)
                        # d_n_dt *= 0
                        # d_n_dt -= sf_nd_nb.bdfscm.alpha[0] * x_all_previous[0]['disp'].view(d_n_dt.shape)
                        # d_n_dt -= sf_nd_nb.bdfscm.alpha[1] * x_all_previous[1]['disp'].view(d_n_dt.shape)
                        # d_n_dt -= sf_nd_nb.bdfscm.alpha[2] * x_all_previous[2]['disp'].view(d_n_dt.shape)
                        # d_n_dt2 *= 0
                        # d_n_dt2 += sf_nd_nb.bdfscm.beta[1] * x_all_previous[0]['disp'].view(d_n_dt2.shape)
                        # d_n_dt2 += sf_nd_nb.bdfscm.beta[2] * x_all_previous[1]['disp'].view(d_n_dt2.shape)
                        # d_n_dt2 += sf_nd_nb.bdfscm.beta[3] * x_all_previous[2]['disp'].view(d_n_dt2.shape)
                        # d_n_dt2 += sf_nd_nb.bdfscm.beta[4] * x_all_previous[3]['disp'].view(d_n_dt2.shape)
                if True:
                    alpha_u_n *= 0
                    d_n_dt *= 0
                    for i in range(0, sf_nd_nb.bdfscm.order):
                        alpha_u_n += sf_nd_nb.bdfscm.alpha[i] * x_all_previous[i]['vel'].view(alpha_u_n.shape)
                        d_n_dt -= sf_nd_nb.bdfscm.alpha[i] * x_all_previous[i]['disp'].view(d_n_dt.shape)
                    d_n_dt2 *= 0
                    for i in range(1, sf_nd_nb.bdfscm.order + 2):
                        d_n_dt2 += sf_nd_nb.bdfscm.beta[i] * x_all_previous[i - 1]['disp'].view(d_n_dt2.shape)

            t += dt
            print('====physical time: ', t, ' ====')
            # get boundary and rhs body force condition
            u_bc, f, fNorm = bc_f.fsi_bc(
                ndim,
                vel_func_space.bc_node_list,
                vel_func_space.x_all,
                prob=config.problem,
                t=t
            )
            # u_bc, f, fNorm = bc_f.vel_bc_f(
            #     ndim,
            #     vel_func_space.bc_node_list,
            #     vel_func_space.x_all,
            #     prob=config.problem,
            #     t=t
            # )
            # u_bc, f, fNorm = bc_f.he_bc_f(
            #     ndim,
            #     vel_func_space.bc_node_list,
            #     vel_func_space.x_all,
            #     prob=config.problem,
            #     t=t
            # )
            # if use grad-div stabilisation or edge stabilisation, get elementwise volume-averaged velocity here
            if config.isGradDivStab or sf_nd_nb.isES:
                u_ave = petrov_galerkin.get_ave_vel(x_all_previous[0]['vel'])
                sf_nd_nb.set_data(u_ave=u_ave)

            x_i *= 0
            x_i += x_all_previous[0]['all']  # use last timestep p as start value
            x_i_k *= 0  # use last timestep value as non-linear iteration start value
            x_i_k += x_all_previous[0]['all']
            d_n_dt += sf_nd_nb.bdfscm.gamma * x_i_k_dict['disp'].view(d_n_dt.shape)
            d_n_dt2 += sf_nd_nb.bdfscm.beta[0] * x_i_k_dict['disp'].view(d_n_dt2.shape)

            r0l2 = torch.tensor(1, device=dev, dtype=torch.float64)  # linear solver residual l2 norm
            its = 0  # linear solver iteration
            nr0l2 = 1  # non-linear solver residual l2 norm
            sf_nd_nb.nits = 0  # newton iteration step
            r0 *= 0

            total_its = 0  # total linear iteration number / restart

            while sf_nd_nb.nits < config.n_its_max:
                sf_nd_nb.nits += 1
                print('============')  # start new non-linear iteration
                sf_nd_nb.Kmatinv = None

                # get rhs and non-linear residual
                r0 *= 0
                x_rhs *= 0
                print('before getting rhs and r0, lets check x_i_k and x_i')
                print('norm x_i_k vel pre disp',
                      torch.linalg.norm(x_i_k_dict['vel']).cpu().numpy(),
                      torch.linalg.norm(x_i_k_dict['pre']).cpu().numpy(),
                      torch.linalg.norm(x_i_k_dict['disp']).cpu().numpy())
                print('norm x_i vel pre disp',
                      torch.linalg.norm(x_i_dict['vel']).cpu().numpy(),
                      torch.linalg.norm(x_i_dict['pre']).cpu().numpy(),
                      torch.linalg.norm(x_i_dict['disp']).cpu().numpy())
                print('difference between x_k and x_i',
                      torch.linalg.norm(x_i_k_dict['vel'] - x_i_dict['vel']).cpu().numpy(),
                      torch.linalg.norm(x_i_k_dict['pre'] - x_i_dict['pre']).cpu().numpy(),
                      torch.linalg.norm(x_i_k_dict['disp'] - x_i_dict['disp']).cpu().numpy())
                # get rhs for fluid
                x_rhs = volume_mf_st.get_rhs(
                    x_rhs=x_rhs, u_bc=u_bc, f=f,
                    include_adv=config.include_adv,
                    u_n=alpha_u_n,  # previous *timesteps* velocity multiplied by BDF extrapolation coeffs
                    isAdvExp=config.isAdvExp,  # whether to treat advection explicity
                    u_k=x_i_k_dict['vel'][0:nele_f, ...],  # non-lienar velocity
                    d_n=d_n_dt,  # previous *timesteps* displacement x BDF coeff.
                )
                # get rhs and non-linear residual for solid
                print('r0_dict pointer', r0_dict['all'].data_ptr())
                x_rhs, r0_dict = volume_mf_he.get_rhs(
                    rhs_in=x_rhs,  # right-hand side
                    u=x_i_k_dict['disp'],  # displacement at current non-linear step
                    u_bc=u_bc,  # boundary condition(s)
                    f=f,  # body force
                    u_n=d_n_dt2,  # displacement at last timestep(s) (used in BDF scheme)
                    is_get_nonlin_res=True,
                    x_k_dict=x_i_k_dict,
                    r0_dict=r0_dict,
                )
                print('r0_dict pointer', r0_dict['all'].data_ptr())
                print('before adding fluid residual, non-linear residual is ', torch.linalg.norm(r0_dict['vel']),
                      torch.linalg.norm(r0_dict['pre']),
                      torch.linalg.norm(r0_dict['disp']))
                # we're solving for correction for displacement in solid
                # so we set starting value to 0
                x_i_dict['disp'][nele_f:nele_f + nele_s, ...] *= 0
                # get fluid subdomain non-linear residual
                # print('is x_i and x_k the same? ', torch.linalg.norm(x_i_dict['vel'] - x_i_k_dict['vel']))
                print('mesh velocity norm... ', torch.linalg.norm(sf_nd_nb.u_m.view(-1)))
                r0 = volume_mf_st.get_residual_only(
                    r0, x_i, x_rhs,
                    include_adv=config.include_adv,
                    x_k=x_i_k,  # last *non-linear iteration step* velocity
                    u_bc=u_bc[0],
                    include_itf=True,
                )
                print('r0_dict pointer', r0_dict['all'].data_ptr())
                nr0l2 = volume_mf_st.get_r0_l2_norm(r0)
                print('nits = ', sf_nd_nb.nits, 'non-linear residual = ', nr0l2.cpu().numpy(),
                      'vel, pre, disp', torch.linalg.norm(r0_dict['vel']),
                      torch.linalg.norm(r0_dict['pre']),
                      torch.linalg.norm(r0_dict['disp']))
                x_rhs_dict = volume_mf_st.slicing_x_i(x_rhs)
                print('rhs norm: vel, pre, disp', torch.linalg.norm(x_rhs_dict['vel']),
                      torch.linalg.norm(x_rhs_dict['pre']),
                      torch.linalg.norm(x_rhs_dict['disp']))
                if nr0l2 < config.n_tol:
                    # non-linear iteration converged
                    break
                # since we're moving mesh and changing pressure_func_space,
                # we need to update pressure Laplacian operator on coarse grid
                # get pressure laplacian on P1CG, will be stored in
                # sf_nd_nb.sfc_data_Lp
                pressure_matrix.get_RAR_and_sfc_data_Lp(
                    sf_nd_nb.sparse_f.whichc, sf_nd_nb.sparse_f.ncolor,
                    sf_nd_nb.sparse_f.fina,
                    sf_nd_nb.sparse_f.cola,
                    sf_nd_nb.sparse_f.ncola
                )
                volume_mf_st.get_RAR_and_sfc_data_Fp(x_i_k, u_bc)
                volume_mf_he.get_RAR_and_sfc_data_Sp(x_i_k)
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
                        tol=max(min(1.e-1 * nr0l2, 1.e-1), 1.e-11),  # config.tol,
                        include_adv=config.include_adv,
                        x_k=x_i_k,
                        u_bc=u_bc
                    )
                    total_its += its
                elif config.linear_solver == 'right-gmres-mg':
                    raise ValueError('right-gmre-mg solver is not implemented!')
                    # x_i = solvers.right_gmres_mg_solver(x_i, x_rhs, config.tol)
                else:
                    raise Exception('choose a valid solver...')

                # let's get non-linear residual here
                # define the residual as the l2 norm of difference between two iterations
                r_vel = torch.linalg.norm((x_i_dict['vel'] - x_i_k_dict['vel'])[0:nele_f, ...]).cpu().numpy()
                r_pre = torch.linalg.norm((x_i_dict['pre'] - x_i_k_dict['pre'])[0:nele_f, ...]).cpu().numpy()
                r_disp = torch.linalg.norm(x_i_dict['disp'][nele_f:nele, ...]).cpu().numpy()
                r_max_vel = torch.max(torch.abs(x_i_dict['vel'] -
                                                x_i_k_dict['vel'])[0:nele_f, ...].view(-1)).cpu().numpy()
                r_max_pre = torch.max(torch.abs(x_i_dict['pre'] -
                                                x_i_k_dict['pre'])[0:nele_f, ...].view(-1)).cpu().numpy()
                r_disp_max = torch.max(torch.abs(x_i_dict['disp'][nele_f:nele, ...].view(-1))).cpu().numpy()
                print('difference between 2 non-linear iteration norm: vel, pre, disp: ', r_vel, r_pre, r_disp)
                print('max diff between 2 non-linear steps: vel, pre, disp: ', r_max_vel, r_max_pre, r_disp_max)

                # before update displacement, let's first update time derivative of displacement (vel & acceleration)
                # in trasient terms (interface structure velocity d_n_dt use in fluid interface bc, and
                # d^2 d / dt^2 used in solid subdomain)
                d_n_dt += sf_nd_nb.bdfscm.gamma * x_i_dict['disp'].view(d_n_dt.shape) * config.relax_coeff
                d_n_dt2 += sf_nd_nb.bdfscm.beta[0] * x_i_dict['disp'].view(d_n_dt2.shape) * config.relax_coeff
                # update displacement (since we're solving the increment of displacement)
                x_i_k_dict['disp'][nele_f:nele, ...] += x_i_dict['disp'][nele_f:nele, ...] * config.relax_coeff
                x_i_dict['disp'][nele_f:nele, ...] *= 0
                x_i_dict['disp'][nele_f:nele, ...] += x_i_k_dict['disp'][nele_f:nele, ...]
                # update nonlinear velocity
                x_i_k_dict['vel'] *= 0
                x_i_k_dict['vel'] += x_i_dict['vel']  # non-linear velocity is updated here
                # update pressure
                x_i_k_dict['pre'] *= 0
                x_i_k_dict['pre'] += x_i_dict['pre']  # we need pressure to update fluid stress for solid subdomain

                if sf_nd_nb.nits % 1 == 0:  # move mesh every n non-linear steps
                    pass
                    # solving for new mesh displacement/velocity
                    # I think it's more sensible to compute the mesh displacement rather than
                    # mesh velocity. We can easily get mesh velocity with BDF scheme.
                    print('going to solve for mesh displacement and move the mesh...')
                    x_i_dict['disp'] = volume_mf_um.solve_for_mesh_disp(x_i_dict['disp'], t)
                    # x_i_dict['disp'] = volume_mf_um_on_fix_mesh.solve_for_mesh_disp(x_i_dict['disp'])
                    # get mesh velocity and move mesh
                    u_m *= 0  # (use BDF scheme)
                    u_m += x_i_dict['disp'] * sf_nd_nb.bdfscm.gamma / dt
                    for ii in range(1, sf_nd_nb.bdfscm.order):
                        u_m -= x_all_previous[ii]['disp'] * sf_nd_nb.bdfscm.alpha[ii] / dt
                    sf_nd_nb.set_data(u_m=u_m)  # store in commn data so that we can use it everywhere.
                    # move velocity mesh
                    sf_nd_nb.vel_func_space.x_ref_in *= 0
                    sf_nd_nb.vel_func_space.x_ref_in += sf_nd_nb.disp_func_space.x_ref_in \
                        + x_i_dict['disp'].permute(0, 2, 1)

                    sf_nd_nb.pre_func_space.x_ref_in *= 0
                    sf_nd_nb.pre_func_space.x_ref_in += sf_nd_nb.vel_func_space.x_ref_in

                    # since we have changed x_i_dict['disp'], we need to update d_n_dt and d_n_dt2 and x_i_k
                    d_n_dt += sf_nd_nb.bdfscm.gamma * (x_i_dict['disp']
                                                       - x_i_k_dict['disp']).view(d_n_dt.shape) * config.relax_coeff
                    d_n_dt2 += sf_nd_nb.bdfscm.beta[0] * (x_i_dict['disp']
                                                          - x_i_k_dict['disp']).view(d_n_dt2.shape) * config.relax_coeff
                    x_i_k_dict['disp'] *= 0
                    x_i_k_dict['disp'] += x_i_dict['disp']

                # output non-linear iteration steps to vtk for debugging
                # put structure velocity in 'vel'
                x_i_dict['vel'][nele_f:nele, ...] *= 0
                x_i_dict['vel'][nele_f:nele, ...] += d_n_dt.view(nele, -1, ndim)[nele_f:nele, ...] / sf_nd_nb.dt
                # if converges,
                sf_nd_nb.vel_func_space.get_x_all_after_move_mesh()
                sf_nd_nb.pre_func_space.get_x_all_after_move_mesh()
                fsi_output.output_fsi_vtu(x_i, vel_func_space, pre_func_space, disp_func_space,
                                          itime*100 + sf_nd_nb.nits)

            # explicit mesh movement -- only move mesh at the end of a timestep
            if False:
                # solving for new mesh displacement/velocity
                # I think it's more sensible to compute the mesh displacement rather than
                # mesh velocity. We can easily get mesh velocity with BDF scheme.
                print('going to solve for mesh displacement and move the mesh...')
                x_i_dict['disp'] = volume_mf_um.solve_for_mesh_disp(x_i_dict['disp'])
                # x_i_dict['disp'] = volume_mf_um_on_fix_mesh.solve_for_mesh_disp(x_i_dict['disp'])
                # get mesh velocity and move mesh
                u_m *= 0  # (use BDF scheme)
                u_m += x_i_dict['disp'] * sf_nd_nb.bdfscm.gamma / dt
                for ii in range(1, sf_nd_nb.bdfscm.order + 1):
                    u_m -= x_all_previous[ii]['disp'] * sf_nd_nb.bdfscm.alpha[ii-1] / dt
                sf_nd_nb.set_data(u_m=u_m)  # store in commn data so that we can use it everywhere.
                # move velocity mesh
                sf_nd_nb.vel_func_space.x_ref_in *= 0
                sf_nd_nb.vel_func_space.x_ref_in += sf_nd_nb.disp_func_space.x_ref_in \
                    + x_i_dict['disp'].permute(0, 2, 1)

                sf_nd_nb.pre_func_space.x_ref_in *= 0
                sf_nd_nb.pre_func_space.x_ref_in += sf_nd_nb.vel_func_space.x_ref_in

            x_i_dict['disp'] *= 0
            x_i_dict['disp'] += x_i_k_dict['disp']
            # put structure velocity in 'vel'
            x_i_dict['vel'][nele_f:nele, ...] *= 0
            x_i_dict['vel'][nele_f:nele, ...] += d_n_dt.view(nele, -1, ndim)[nele_f:nele, ...] / sf_nd_nb.dt
            # store this step in case we want to use this for next timestep
            for ii in range(len(x_all_previous)-1, 0, -1):
                x_all_previous[ii]['all'] *= 0
                x_all_previous[ii]['all'] += x_all_previous[ii-1]['all']
            x_all_previous[0]['all'] *= 0
            x_all_previous[0]['all'] += x_i

            # save x_i at this Re to reuse as the initial condition for higher Re
            torch.save(x_i, config.filename + config.case_name + '_t%.2f.pt' % t)
            print('total its / restart ', total_its)
            total_its = 0

            # output to vtk
            # if converges,
            sf_nd_nb.vel_func_space.get_x_all_after_move_mesh()
            sf_nd_nb.pre_func_space.get_x_all_after_move_mesh()
            fsi_output.output_fsi_vtu(x_i, vel_func_space, pre_func_space, disp_func_space, itime)

            # # get l2 error
            # x_ana = bc_f.ana_soln(config.problem, t=t)
            # x_ana[0:u_nonods*ndim] += torch.tensor(
            #     u_all[itime - 1, :, :, :],
            #     device=dev, dtype=torch.float64).view(-1)
            # x_ana[u_nonods * ndim: u_nonods*ndim + p_nonods] += torch.tensor(
            #     p_all[itime - 1, ...],
            #     device=dev, dtype=torch.float64).view(-1)
            # if config.hasNullSpace:
            #     # remove average from pressure
            #     p_ave = ns_assemble.get_ave_pressure(x_i_dict['pre'].cpu().numpy())
            #     x_i[u_nonods * ndim:u_nonods * ndim + p_nonods] -= p_ave
            #     p_ana = x_ana[u_nonods * ndim:u_nonods * ndim + p_nonods]
            #     p_ana_ave = ns_assemble.get_ave_pressure(p_ana.cpu().numpy())
            #     p_ana -= p_ana_ave
            # u_l2, p_l2, u_linf, p_linf = volume_mf_st.get_l2_error(x_i, x_ana)
            # print('after solving, compare to previous timestep, l2 error is: \n',
            #       'velocity ', u_l2, '\n',
            #       'pressure ', p_l2)
            # print('l infinity error is: \n',
            #       'velocity ', u_linf, '\n',
            #       'pressure ', p_linf)
            # # print('total its / restart ', total_its)

            print('wall time on this timestep: ', time.time() - wall_time_start)

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
    vtk.write_scalar(torch.tensor(p_sol, dtype=torch.float64), 'pressure', sf_nd_nb.pre_func_space)
    vtk.write_end()


#############################################################
# write output
#############################################################
# output 1:
# to output, we need to change u_all to 2D array
# (tstep, ndim*nonods)
# u_all = np.reshape(u_all, (tstep, nele*vel_ele.nloc*ndim))
# p_all = np.reshape(p_all, (tstep, nele*pre_ele.nloc))
# np.savetxt('u_all.txt', u_all, delimiter=',')
# np.savetxt('p_all.txt', p_all, delimiter=',')
# np.savetxt('x_all.txt', vel_func_space.x_all, delimiter=',')
# np.savetxt('p_x_all.txt', pre_func_space.x_all, delimiter=',')
print('10. done output, time elaspsed: ', time.time()-starttime)
# print(torch.cuda.memory_summary())
