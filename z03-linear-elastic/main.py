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
from config import sf_nd_nb
import mesh_init 
# from mesh_init import face_iloc,face_iloc2
from shape_function import SHATRInew, det_nlx, sdet_snlx
from surface_integral import S_Minv_sparse, RSR_matrix, RSR_matrix_color, RSR_mf_color
import volume_mf_linear_elastic
import surface_mf_linear_elastic
from volume_integral import mk, mk_lv1, calc_RKR, calc_RAR
from color import color2
import multigrid_linearelastic as mg
import time 

starttime = time.time()

# for pretty print out torch tensor
# torch.set_printoptions(sci_mode=False)
torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)

dev = config.dev
nele = config.nele 
mesh = config.mesh 
nonods = config.nonods
p1dg_nonods = config.p1dg_nonods
ngi = config.ngi
ndim = config.ndim
nloc = config.nloc 
dt = config.dt 
tend = config.tend 
tstart = config.tstart

print('computation on ',dev)
print('nele=', nele)

x_all, nbf, nbele, fina, cola, ncola, \
    bc1, bc2, bc3, bc4, \
    cg_ndglno, cg_nonods, cg_bc = mesh_init.init()
nbele = torch.tensor(nbele, device=dev)
nbf = torch.tensor(nbf, device=dev)
config.sf_nd_nb.set_data(nbele=nbele, nbf=nbf)
np.savetxt('cg_ndglno.txt', cg_ndglno, delimiter=',')
if False:  # P0DG connectivity and coloring
    # coloring and get probing vector
    [whichc, ncolor] = color2(fina=fina, cola=cola, nnode = nele)
if True:# P1CG connectivity and coloring
    fina, cola, ncola = mesh_init.p1cg_sparsity(cg_ndglno)
    whichc, ncolor = color2(fina=fina, cola=cola, nnode=cg_nonods)
# np.savetxt('whichc.txt', whichc, delimiter=',')
print('ncolor', ncolor, 'whichc type', whichc.dtype)
print('cg_nonods', cg_nonods, 'ncola (p1cg sparsity)', ncola)
print('1. time elapsed, ',time.time()-starttime)
#####################################################
# shape functions
#####################################################
# get shape functions on reference element
[n,nlx,weight,sn,snlx,sweight] = SHATRInew(config.nloc,
                                           config.ngi, config.ndim, config.snloc, config.sngi)
# put numpy array to torch tensor in expected device
sf_nd_nb.set_data(n = torch.tensor(n, device=config.dev, dtype=torch.float64))
sf_nd_nb.set_data(nlx = torch.tensor(nlx, device=dev, dtype=torch.float64))
sf_nd_nb.set_data(weight = weight)
sf_nd_nb.set_data(sn = torch.tensor(sn, dtype=torch.float64, device=dev))
sf_nd_nb.set_data(snlx = snlx)
sf_nd_nb.set_data(sweight = sweight)
del n, nlx, weight, sn, snlx, sweight
print('3. time elapsed, ',time.time()-starttime)
#######################################################
# assemble local mass matrix and stiffness matrix
#######################################################


# Mk = mk()
# Mk.to(device=dev)
print('4. time elapsed, ',time.time()-starttime)

####################################################
# time loop
####################################################
x_ref_in = np.empty((nele, ndim, nloc))
for ele in range(nele):
    for iloc in range(nloc):
        glb_iloc = ele*nloc+iloc
        x_ref_in[ele,0,iloc] = x_all[glb_iloc,0]
        x_ref_in[ele,1,iloc] = x_all[glb_iloc,1]
sf_nd_nb.set_data(x_ref_in = torch.tensor(x_ref_in,
                                          device=dev,
                                          dtype=torch.float64,
                                          requires_grad=False))
del x_ref_in

# initical condition
u = torch.zeros(nele, nloc, ndim, device=dev, dtype=torch.float64)  # now we have a vector filed to solve

# apply boundary conditions (4 Dirichlet bcs)
for inod in bc1:
    u[int(inod/nloc), inod%nloc, :]=0.
for inod in bc2:
    u[int(inod/nloc), inod%nloc, :]=0.
for inod in bc3:
    u[int(inod/nloc), inod%nloc, :]=0.
for inod in bc4:
    u[int(inod/nloc), inod%nloc, :]=0.
    # x_inod = x_ref_in[inod//10, 0, inod%10]
    # u[:,inod]= torch.sin(torch.pi*x_inod)
# print(u)
tstep=int(np.ceil((tend-tstart)/dt))+1
u = u.reshape(nele, nloc, ndim) # reshape doesn't change memory allocation.
u_bc = u.detach().clone() # this stores Dirichlet boundary *only*, otherwise zero.

f, fNorm = config.rhs_f(x_all, config.mu) # rhs force

u = torch.rand_like(u)*0  # initial guess
# u = torch.ones_like(u)
u_all=np.empty([tstep, nele, nloc, ndim])
u_all[0, :, :, :]=u.view(nele, nloc, ndim).cpu().numpy()
print('5. time elapsed, ',time.time()-starttime)
print('6. time elapsed, ',time.time()-starttime)

if False:  # per element condensation Rotation matrix (diagonal)
    R = torch.tensor([1./30., 1./30., 1./30., 3./40., 3./40., 3./40.,
                     3./40., 3./40., 3./40., 9./20.], device=dev, dtype=torch.float64)
    # R = torch.tensor([1./10., 1./10., 1./10., 1./10., 1./10., 1./10.,
    #         1./10., 1./10., 1./10., 1./10.], device=dev, dtype=torch.float64)
if True:  # from PnDG to P1CG
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
    # PnDG to P1DG
    # if config.ele_type == 'cubic':
    if False:
        # I_13 = np.asarray([
        #     [1./ 10, 0, 0, 27./ 40, -9./ 40, -9./ 40, -9./ 40, -9./ 40, 27./ 40, 9./ 20],
        #     [0, 1./ 10, 0, -9./ 40, 27./ 40, 27./ 40, -9./ 40, -9./ 40, -9./ 40, 9./ 20],
        #     [0, 0, 1./ 10, -9./ 40, -9./ 40, -9./ 40, 27./ 40, 27./ 40, -9./ 40, 9./ 20],
        # ], dtype=np.float64)  # from P3DG to P1DG, restrictor
        I_31 = np.asarray([
            [1., 0, 0],
            [0, 1., 0],
            [0, 0, 1.],
            [2./ 3, 1./ 3, 0],
            [1./ 3, 2./ 3, 0],
            [0, 2./ 3, 1./ 3],
            [0, 1./ 3, 2./ 3],
            [1./ 3, 0, 2./ 3],
            [2./ 3, 0, 1./ 3],
            [1./ 3, 1./ 3, 1./ 3]
        ], dtype=np.float64)  # P1DG to P3DG, element-wise prolongation operator
        I_13 = np.transpose(I_31).copy()
        I_31_big = bsr_matrix((I_31.repeat(nele, axis=0).reshape(10, nele, 3).transpose((1, 0, 2)),
                               np.arange(0,nele), np.arange(0,nele+1)),
                              shape=(nonods, p1dg_nonods))
        I_31_big = I_31_big.tocsr()
        I_13_big = bsr_matrix((I_13.repeat(nele, axis=0).reshape(3, nele, 10).transpose((1, 0, 2)),
                               np.arange(0,nele), np.arange(0,nele+1)),
                              shape=(p1dg_nonods, nonods))
        I_13_big = I_13_big.tocsr()
        I_31_big = torch.sparse_csr_tensor(crow_indices=torch.tensor(I_31_big.indptr),
                                           col_indices=torch.tensor(I_31_big.indices),
                                           values=I_31_big.data,
                                           size=(nonods, p1dg_nonods),
                                           device=dev)
        I_13_big = torch.sparse_csr_tensor(crow_indices=torch.tensor(I_13_big.indptr),
                                           col_indices=torch.tensor(I_13_big.indices),
                                           values=I_13_big.data,
                                           size=(p1dg_nonods, nonods),
                                           device=dev)
        # np.savetxt('I_dc.txt', I_fc.to_dense().cpu().numpy(), delimiter=',')
        # np.savetxt('I_cd.txt', I_cf.to_dense().cpu().numpy(), delimiter=',')
        I_fc = torch.sparse.mm(I_31_big, I_fc)
        del I_31_big
        I_cf = torch.sparse.mm(I_cf, I_13_big)
        del I_13_big
        # np.savetxt('I_31_big.txt', I_31_big.to_dense().cpu().numpy(), delimiter=',')
        # np.savetxt('I_13_big.txt', I_13_big.to_dense().cpu().numpy(), delimiter=',')
#         np.savetxt('I_fc.txt', I_fc.to_dense().cpu().numpy(), delimiter=',')
#         np.savetxt('I_cf.txt', I_cf.to_dense().cpu().numpy(), delimiter=',')
print('7. time elapsed, ',time.time()-starttime)

if (config.solver=='iterative') :
    print('i am going to time loop')
    print('8. time elapsed, ',time.time()-starttime)
    # print("Using quit()")
    # quit()
    r0l2all=[]
    # time loop
    r0 = torch.zeros(nele*nloc, ndim, device=dev, dtype=torch.float64)
    for itime in tqdm(range(1,tstep)):
        u_n = u.view(nele, nloc, ndim)  # store last timestep value to un
        u_i = u_n  # jacobi iteration initial value taken as last time step value

        r0l2 = 1
        its = 0
        r0 *= 0

        # prepare for MG on SFC-coarse grids
        RARvalues = volume_mf_linear_elastic.calc_RAR_mf_color(
            I_fc, I_cf,
            whichc, ncolor,
            fina, cola, ncola)
        from scipy.sparse import bsr_matrix
        RAR = bsr_matrix((RARvalues.cpu().numpy(), cola, fina), shape=(ndim*cg_nonods, ndim*cg_nonods))
        # np.savetxt('RAR.txt', RAR.toarray(), delimiter=',')
        RARvalues = torch.permute(RARvalues, (1,2,0)).contiguous()  # (ndim,ndim,ncola)
        # get SFC, coarse grid and operators on coarse grid. Store them to save computational time?
        space_filling_curve_numbering, variables_sfc, nlevel, nodes_per_level = \
            mg.mg_on_P0DG_prep(fina, cola, RARvalues)
        print('9. time elapsed, ', time.time()-starttime)

        # get initial residual on PnDG
        r0 *= 0
        r0 = volume_mf_linear_elastic.get_residual_only(
            r0,
            u_i, u_n, u_bc, f)
        r0_init = torch.linalg.norm(r0.view(-1), dim=0)
        # sawtooth iteration : sooth one time at each white dot
        # fine grid    o   o - o   o - o   o
        #               \ /     \ /     \ /  ...
        # coarse grid    o       o       o
        while (r0l2>config.jac_resThres and its<config.jac_its):
            u_i = u_i.view(nonods, ndim)
            # from torch.profiler import profile, record_function, ProfilerActivity
            # with profile(activities=[
            #         ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            #     print('time1', time.time()-starttime)
            #     r0, u_i = volume_mf_linear_elastic.get_residual_and_smooth_once(
            #         r0, u_i, u_n, u_bc, f)
            #     print('time2', time.time()-starttime)
            #
            # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=100))
            ## on fine grid
            # get diagA and residual at fine grid r0
            for its1 in range(config.pre_smooth_its):
                r0 *= 0
                r0, u_i = volume_mf_linear_elastic.get_residual_and_smooth_once(
                    r0, u_i, u_n, u_bc, f)
            # get residual on PnDG
            r0 *= 0
            r0 = volume_mf_linear_elastic.get_residual_only(
                r0,
                u_i, u_n, u_bc, f)

            if False:  # PnDG to P0DG - TODO: this should be permanantly disabled...
                # per element condensation
                # passing r0 to next level coarse grid and solve Ae=r0
                r1 = torch.einsum('...ij,i->...j', r0.view(nele, nloc, ndim), R)  # restrict residual to coarser mesh,
                # (nele, ndim)
                r1 = torch.transpose(r1, dim0=0, dim1=1)  # shape (ndim, nele)
            if not config.is_pmg:  # PnDG to P1CG
                r1 = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
                for idim in range(ndim):
                    r1[:, idim] += torch.mv(I_cf, mg.p3dg_to_p1dg_restrictor(r0[:, idim]))
            else:  # PnDG down one order each time, eventually go to P1CG
                r_p, e_p = mg.p_mg_pre(r0)
                r1 = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
                ilevel = 3 - 1
                for idim in range(ndim):
                    r1[:, idim] += torch.mv(I_cf, r_p[ilevel][:, idim])
            if not config.is_sfc:  # two-grid method
                e_i = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
                e_direct = sp.sparse.linalg.spsolve(
                    RAR.tocsr(),
                    r1.contiguous().view(-1).cpu().numpy())
                e_direct = np.reshape(e_direct, (cg_nonods, ndim))
                e_i += torch.tensor(e_direct, device=dev, dtype=torch.float64)
            else:  # multi-grid method
                ncurve = 1  # always use 1 sfc
                N = len(space_filling_curve_numbering)
                inverse_numbering = np.zeros((N, ncurve), dtype=int)
                inverse_numbering[:, 0] = np.argsort(space_filling_curve_numbering[:, 0])
                r1_sfc = r1[inverse_numbering[:, 0], :].view(cg_nonods, ndim)

                # # if we do the presmooth steps inside mg_on_P1CG, there's no need to pass in rr1 and e_i
                # e_i = torch.zeros((cg_nonods,1), device=dev, dtype=torch.float64)
                # rr1 = r1_sfc.detach().clone()
                # rr1_l2_0 = torch.linalg.norm(rr1.view(-1),dim=0)
                # rr1_l2 = 10.
                # go to SFC coarse grid levels and do 1 mg cycles there
                e_i = mg.mg_on_P1CG(
                    r1_sfc.view(cg_nonods, ndim),
                    variables_sfc,
                    nlevel,
                    nodes_per_level
                )
                # reverse to original order
                e_i = e_i[space_filling_curve_numbering[:, 0] - 1, :].view(cg_nonods, ndim)
            if not config.is_pmg:  # from P1CG to P3DG
                # prolongate error to fine grid
                e_i0 = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)
                for idim in range(ndim):
                    e_i0[:,idim] += mg.p1dg_to_p3dg_prolongator(torch.mv(I_fc, e_i[:,idim]))
            else:  # from P1CG to P1DG, then go one order up each time while also do post smoothing
                # prolongate error to P1DG
                ilevel = 3-1
                for idim in range(ndim):
                    e_p[ilevel][:, idim] += torch.mv(I_fc, e_i[:, idim])
                r_p, e_p = mg.p_mg_post(e_p, r_p)
                e_i0 = e_p[0]
            # correct fine grid solution
            u_i += e_i0
            # post smooth
            for its1 in range(config.post_smooth_its):
                r0 *= 0
                r0, u_i = volume_mf_linear_elastic.get_residual_and_smooth_once(
                    r0, u_i, u_n, u_bc, f)
            r0l2 = torch.linalg.norm(r0.view(-1),dim=0)/r0_init  # fNorm

            print('its=',its,'fine grid residual l2 norm=',r0l2.cpu().numpy(),
                  'abs res=',torch.linalg.norm(r0.view(-1),dim=0).cpu().numpy(),
                  'r0_init', r0_init.cpu().numpy())
            r0l2all.append(r0l2.cpu().numpy())

            its+=1

        # get final residual after we're back to fine grid
        r0 *= 0
        r0 = volume_mf_linear_elastic.get_residual_only(
            r0,
            u_i, u_n, u_bc, f)

        r0l2 = torch.linalg.norm(r0.view(-1), dim=0)/fNorm
        r0l2all.append(r0l2.cpu().numpy())
        print('its=',its,'residual l2 norm=',r0l2.cpu().numpy(),
              'abs res=',torch.linalg.norm(r0.view(-1),dim=0).cpu().numpy(),
              'fNorm', fNorm.cpu().numpy())
            
        # if jacobi converges,
        u = u_i.view(nele, nloc, ndim)

        # combine inner/inter element contribution
        u_all[itime, :, :, :] = u.cpu().numpy()

    np.savetxt('r0l2all.txt', np.asarray(r0l2all), delimiter=',')

if (config.solver=='direct'):
    from assemble_matrix_for_direct_solver import SK_matrix
    # from assemble_double_diffusion_direct_solver import SK_matrix
    with torch.no_grad():
        nx, detwei = Det_nlx.forward(x_ref_in, weight)
    # assemble S+K matrix and rhs (force + bc)
    SK, rhs_b = SK_matrix(n, nx, detwei,
                          sn, snx, sdetwei, snormal,
                          nbele, nbf, f, u_bc,
                          fina, cola, ncola)
    np.savetxt('SK.txt', SK.toarray(), delimiter=',')
    np.savetxt('rhs_b.txt', rhs_b, delimiter=',')
    print('8. (direct solver) done assmeble, time elapsed: ', time.time()-starttime)
    # direct solver in scipy
    SK = SK.tocsr()
    u_i = sp.sparse.linalg.spsolve(SK, rhs_b)  # shape nele*ndim*nloc
    print('9. (direct solver) done solve, time elapsed: ', time.time()-starttime)
    u_i = np.reshape(u_i, (nele, nloc, ndim))
    # store to c_all to print out
    u_all[1,:,:,:] = u_i

#############################################################
# write output
#############################################################
# output 1: 
# to output, we need to change u_all to 2D array
# (tstep, ndim*nonods)
u_all = np.reshape(u_all, (tstep, nele*nloc*ndim))
np.savetxt('u_all.txt', u_all, delimiter=',')
np.savetxt('x_all.txt', x_all, delimiter=',')
print('10. done output, time elaspsed: ', time.time()-starttime)
print(torch.cuda.memory_summary())