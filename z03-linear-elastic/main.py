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
import solvers
import volume_mf_he
from config import sf_nd_nb
import mesh_init
# from mesh_init import face_iloc,face_iloc2
from shape_function import SHATRInew, det_nlx, sdet_snlx
if config.problem == 'linear-elastic':
    import volume_mf_linear_elastic as integral_mf
else:
    import volume_mf_he as integral_mf
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

if ndim == 2:
    x_all, nbf, nbele, alnmt, fina, cola, ncola, \
        bc, \
        cg_ndglno, cg_nonods, cg_bc = mesh_init.init()
else:
    alnmt: object
    [x_all, nbf, nbele, alnmt, fina, cola, ncola, bc, cg_ndglno, cg_nonods] = mesh_init.init_3d()
nbele = torch.tensor(nbele, device=dev)
nbf = torch.tensor(nbf, device=dev)
alnmt = torch.tensor(alnmt, device=dev)
config.sf_nd_nb.set_data(nbele=nbele, nbf=nbf, alnmt=alnmt)
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
sf_nd_nb.set_data(n=torch.tensor(n, device=config.dev, dtype=torch.float64))
sf_nd_nb.set_data(nlx=torch.tensor(nlx, device=dev, dtype=torch.float64))
sf_nd_nb.set_data(weight=torch.tensor(weight, device=dev, dtype=torch.float64))
sf_nd_nb.set_data(sn=torch.tensor(sn, dtype=torch.float64, device=dev))
sf_nd_nb.set_data(snlx=torch.tensor(snlx, device=dev, dtype=torch.float64))
sf_nd_nb.set_data(sweight=torch.tensor(sweight, device=dev, dtype=torch.float64))
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
        for idim in range(ndim):
            x_ref_in[ele, idim, iloc] = x_all[glb_iloc, idim]
sf_nd_nb.set_data(x_ref_in = torch.tensor(x_ref_in,
                                          device=dev,
                                          dtype=torch.float64,
                                          requires_grad=False))
del x_ref_in

# initical condition
u = torch.zeros(nele, nloc, ndim, device=dev, dtype=torch.float64)  # now we have a vector filed to solve

u, f, fNorm = bc_f.bc_f(ndim, bc, u, x_all, prob=config.problem)  # 'linear-elastic')

# print(u)
tstep=int(np.ceil((tend-tstart)/dt))+1
u = u.reshape(nele, nloc, ndim) # reshape doesn't change memory allocation.
u_bc = u.detach().clone() # this stores Dirichlet boundary *only*, otherwise zero.

u = torch.rand_like(u)*0  # initial guess
# output to vtk
vtk = output.File(config.filename+config.case_name+'_%d.vtu' % 0, x_all)
vtk.write(u, 'displacement')

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
    sf_nd_nb.set_data(I_fc=I_fc, I_cf=I_cf)
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
    if config.problem == 'linear-elastic':
        for itime in range(1,tstep):
            u_n = u.view(nele, nloc, ndim)  # store last timestep value to un
            u_i = u_n  # jacobi iteration initial value taken as last time step value

            r0l2 = 1
            its = 0
            r0 *= 0

            # prepare for MG on SFC-coarse grids
            RARvalues = integral_mf.calc_RAR_mf_color(
                I_fc, I_cf,
                whichc, ncolor,
                fina, cola, ncola)
            from scipy.sparse import bsr_matrix
            if not config.is_sfc:
                RAR = bsr_matrix((RARvalues.cpu().numpy(), cola, fina), shape=(ndim*cg_nonods, ndim*cg_nonods))
                sf_nd_nb.set_data(RARmat=RAR.tocsr())
            # np.savetxt('RAR.txt', RAR.toarray(), delimiter=',')
            RARvalues = torch.permute(RARvalues, (1,2,0)).contiguous()  # (ndim,ndim,ncola)
            # get SFC, coarse grid and operators on coarse grid. Store them to save computational time?
            space_filling_curve_numbering, variables_sfc, nlevel, nodes_per_level = \
                mg.mg_on_P0DG_prep(fina, cola, RARvalues)
            sf_nd_nb.sfc_data.set_data(
                space_filling_curve_numbering=space_filling_curve_numbering,
                variables_sfc=variables_sfc,
                nlevel=nlevel,
                nodes_per_level=nodes_per_level
            )
            print('9. time elapsed, ', time.time()-starttime)

            if config.linear_solver == 'mg':
                u_i = solvers.multigrid_solver(u_i, u_n, u_bc, f, config.tol)
            elif config.linear_solver == 'gmres-mg':
                u_i = solvers.gmres_mg_solver(u_i, u_n, u_bc, f, config.tol)
            else:
                raise Exception('choose a valid solver...')
            # get final residual after we're back to fine grid
            r0 *= 0
            r0 = integral_mf.get_residual_only(
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
    else:  # config.problem == 'hyper-elastic'
        du_i = torch.zeros(nele, nloc, ndim, device=dev, dtype=torch.float64)
        u_rhs = torch.zeros(nele, nloc, ndim, device=dev, dtype=torch.float64)
        for itime in range(1, tstep):  # time loop
            u_n = u.view(nele, nloc, ndim)  # store last timestep value to un
            u_i = u_n.detach().clone()  # newton iteration initial value taken as last time step value

            r0l2 = 1  # linear solver residual l2 norm
            its = 0  # linear solver iteration
            nr0l2 = 1  # non-linear solver residual l2 norm
            nits = 0  # newton iteration step
            r0 *= 0

            while nits < config.n_its_max:
                u_rhs *= 0
                u_rhs = volume_mf_he.get_rhs(rhs=u_rhs, u=u_i, u_bc=u_bc, f=f, u_n=u_n)
                nr0l2 = torch.linalg.norm(u_rhs.view(-1))
                print('============')
                print('nits = ', nits, 'non-linear residual = ', nr0l2.cpu().numpy())
                if nr0l2 < config.n_tol:
                    # non-linear iteration converged
                    break
                # prepare for MG on SFC-coarse grids
                RARvalues = volume_mf_he.calc_RAR_mf_color(
                    I_fc, I_cf,
                    whichc, ncolor,
                    fina, cola, ncola,
                    u_i
                )
                from scipy.sparse import bsr_matrix

                if not config.is_sfc:
                    RAR = bsr_matrix((RARvalues.cpu().numpy(), cola, fina), shape=(ndim * cg_nonods, ndim * cg_nonods))
                    sf_nd_nb.set_data(RARmat=RAR.tocsr())
                # np.savetxt('RAR.txt', RAR.toarray(), delimiter=',')
                RARvalues = torch.permute(RARvalues, (1, 2, 0)).contiguous()  # (ndim,ndim,ncola)
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

                du_i *= 0  # solve for delta u_i
                if config.linear_solver == 'mg':
                    du_i = solvers.multigrid_solver(u_i, du_i, u_rhs, config.tol)
                elif config.linear_solver == 'gmres-mg':
                    du_i = solvers.gmres_mg_solver(u_i, du_i, u_rhs, min(1.e-3*nr0l2, 1.e-3))
                else:
                    raise Exception('choose a valid solver...')
                # get final residual after we're back to fine grid
                r0 *= 0
                r0 = volume_mf_he.get_residual_only(
                    r0, u_i, du_i, u_rhs
                )

                r0l2 = torch.linalg.norm(r0.view(-1), dim=0) / fNorm
                r0l2all.append(r0l2.cpu().numpy())
                print('its=', its, 'residual l2 norm=', r0l2.cpu().numpy(),
                      'abs res=', torch.linalg.norm(r0.view(-1), dim=0).cpu().numpy(),
                      'fNorm', fNorm.cpu().numpy())
                nr0l2 = torch.linalg.norm(du_i.view(-1))
                print('norm of delta u ', nr0l2.cpu().numpy())
                u_i += du_i.view(nele, nloc, ndim) * config.relax_coeff
                nits += 1

            # if jacobi converges,
            u = u_i.view(nele, nloc, ndim)

            # combine inner/inter element contribution
            u_all[itime, :, :, :] = u.cpu().numpy()

            # output to vtk
            vtk = output.File(config.filename+config.case_name+'_%d.vtu' % itime, x_all)
            vtk.write(u, 'displacement')

    np.savetxt('r0l2all.txt', np.asarray(r0l2all), delimiter=',')

if (config.solver=='direct'):
    # from assemble_matrix_for_direct_solver import SK_matrix
    # # from assemble_double_diffusion_direct_solver import SK_matrix
    # with torch.no_grad():
    #     nx, detwei = Det_nlx.forward(x_ref_in, weight)
    # # assemble S+K matrix and rhs (force + bc)
    # SK, rhs_b = SK_matrix(n, nx, detwei,
    #                       sn, snx, sdetwei, snormal,
    #                       nbele, nbf, f, u_bc,
    #                       fina, cola, ncola)
    # np.savetxt('SK.txt', SK.toarray(), delimiter=',')
    # np.savetxt('rhs_b.txt', rhs_b, delimiter=',')
    # print('8. (direct solver) done assmeble, time elapsed: ', time.time()-starttime)
    # # direct solver in scipy
    # SK = SK.tocsr()
    # u_i = sp.sparse.linalg.spsolve(SK, rhs_b)  # shape nele*ndim*nloc

    # ----- new matrix assemble for direct solver -----
    dummy1 = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)
    dummy2 = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)
    dummy3 = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)
    dummy4 = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)
    Amat = torch.zeros(nonods * ndim, nonods * ndim, device=dev, dtype=torch.float64)
    rhs = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)
    probe = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)
    np.savetxt('f.txt', f.cpu().numpy(), delimiter=',')
    dummy1 *= 0
    dummy2 *= 0
    rhs = integral_mf.get_residual_only(
        r0=rhs,
        u_i=dummy1,
        u_n=dummy2,
        u_bc=u_bc,
        f=f)
    for inod in tqdm(range(nonods)):
        for jdim in range(ndim):
            dummy1 *= 0
            dummy2 *= 0
            dummy3 *= 0
            dummy4 *= 0
            probe *= 0
            probe[inod, :] = 1.
            Amat[:, inod + jdim * nonods] -= integral_mf.get_residual_only(
                r0=dummy1,
                u_i=probe,
                u_n=dummy2,
                u_bc=dummy3,
                f=dummy4).view(-1)
    np.savetxt('Amat.txt', Amat.cpu().numpy(), delimiter=',')
    np.savetxt('rhs.txt', rhs.cpu().numpy(), delimiter=',')
    Amat_np = sp.sparse.csr_matrix(Amat.cpu().numpy())
    rhs_np = rhs.view(-1).cpu().numpy()
    print('im going to solve', time.time() - starttime)
    u_i = sp.sparse.linalg.spsolve(Amat_np, rhs_np)
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
