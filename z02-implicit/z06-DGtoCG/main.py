#!/usr/bin/env python3

####################################################
# preamble
####################################################
# import
import time, os.path

import volume_integral

starttime = time.time()

import toughio 
import numpy as np
import torch
import scipy as sp
# import time
from scipy.sparse import coo_matrix, bsr_matrix
from tqdm import tqdm
import config
from config import sf_nd_nb
import mesh_init 
# from mesh_init import face_iloc,face_iloc2
from shape_function import SHATRInew, det_nlx, sdet_snlx, get_det_nlx
from surface_integral import S_Minv_sparse
from volume_integral import calc_RAR_mf_color, get_residual_and_smooth_once, \
    get_residual_only
from color import color2
import multi_grid, solvers

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
    [x_all, nbf, nbele, fina, cola, ncola, bc1, bc2, bc3, bc4, cg_ndgln, cg_nonods, cg_bc] = mesh_init.init()
    config.sf_nd_nb.set_data(nbele=nbele, nbf=nbf)
else:
    [x_all, nbf, nbele, alnmt, fina, cola, ncola, bc, cg_ndgln, cg_nonods] = mesh_init.init_3d()
    config.sf_nd_nb.set_data(nbele=nbele, nbf=nbf, alnmt=alnmt)
# [fina, cola, ncola] = mesh_init.connectivity(nbele)
np.savetxt('x_all.txt', x_all, delimiter=',')
if False:  # P0DG connectivity and coloring
    # coloring and get probing vector
    [whichc, ncolor] = color2(fina=fina, cola=cola, nnode = nele)
if True:  # P1CG connectivity and coloring
    fina, cola, ncola = mesh_init.p1cg_sparsity(cg_ndgln)
    whichc, ncolor = color2(fina=fina, cola=cola, nnode=cg_nonods)
#     starting_node = 1 # setting according to BY
#     graph_trim = -10  # ''
#     ncurve = 1        # ''
#     ncola = cola.shape[0]
#     dummy_vec = np.zeros(nonods)
#     import time
#     start_time = time.time()
#     print('to get space filling curve...', time.time()-start_time)
#     if os.path.isfile(config.filename[:-4] + '_sfc.npy'):
#         print('pre-calculated sfc exists. readin from file...')
#         sfc = np.load(config.filename[:-4] + '_sfc.npy')
#     else:
#         import sfc as sf  #
#         _, sfc = \
#             sf.ncurve_python_subdomain_space_filling_curve(
#             cola+1, fina+1, starting_node, graph_trim, ncurve,
#             ) # note that fortran array index start from 1, so cola and fina should +1.
#         np.save(config.filename[:-4] + '_sfc.npy', sfc)
#     print('to get sfc operators...', time.time()-start_time)
# exit(0)
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
sf_nd_nb.set_data(n = torch.tensor(n, device=config.dev, dtype=torch.float64))
sf_nd_nb.set_data(nlx = torch.tensor(nlx, device=dev, dtype=torch.float64))
sf_nd_nb.set_data(weight = torch.tensor(weight, device=dev, dtype=torch.float64))
sf_nd_nb.set_data(sn = torch.tensor(sn, dtype=torch.float64, device=dev))
sf_nd_nb.set_data(snlx = torch.tensor(snlx, dtype=torch.float64, device=dev))
sf_nd_nb.set_data(sweight = torch.tensor(sweight, dtype=torch.float64, device=dev))
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
sf_nd_nb.set_data(x_ref_in = torch.tensor(x_ref_in, device=dev, requires_grad=False))
del x_ref_in
# print(x_ref_in)
# initical condition
c = torch.zeros(nele*nloc, device=dev, dtype=torch.float64)
if ndim == 2:
    # apply boundary conditions (4 Dirichlet bcs)
    for inod in bc1:
        c[inod]=0.
        # x_inod = x_ref_in[inod//10, 0, inod%10]
        # y_inod = x_ref_in[inod//10, 1, inod%10]
        # print('bc1 inod %d x %f y %f'%(inod,x_inod, y_inod))
        # c[inod]= x_inod
    for inod in bc2:
        c[inod]=0.
        # x_inod = x_ref_in[inod//10, 0, inod%10]
        # y_inod = x_ref_in[inod//10, 1, inod%10]
        # print('bc2 inod %d x %f y %f'%(inod,x_inod, y_inod))
        # c[inod]= x_inod
    for inod in bc3:
        c[inod]=0.
        # x_inod = x_ref_in[inod//10, 0, inod%10]
        # y_inod = x_ref_in[inod//10, 1, inod%10]
        # print('bc3 inod %d x %f y %f'%(inod,x_inod, y_inod))
        # c[inod]= x_inod
    for inod in bc4:
        x_inod = sf_nd_nb.x_ref_in[inod//nloc, 0, inod%nloc]
        c[inod]= torch.sin(torch.pi*x_inod)
        # y_inod = x_ref_in[inod//10, 1, inod%10]
        # print('bc4 inod %d x %f y %f'%(inod,x_inod, y_inod))
        # c[inod]= x_inod
        # print("x, c", x_inod.cpu().numpy(), c[inod])
    f = x_all[:, 0] * 0  # right-hand side source
else:  # (6 Dirichlet bcs)
    for bci in bc:
        for inod in bci:
            x_inod = sf_nd_nb.x_ref_in[inod//nloc, :, inod%nloc]
            # c[inod] = torch.sin(torch.pi*2*x_inod[0]) \
            #     * torch.sin(torch.pi*2*x_inod[1]) \
            #     * torch.sin(torch.pi*2*x_inod[2])
            c[inod] = torch.exp(-x_inod[0] - x_inod[1] - x_inod[2])
            # c[inod] = x_inod[1]
    # right hand side
    # f = 12*np.pi**2 * np.sin(2*np.pi*x_all[:, 0]) \
    #                * np.sin(2*np.pi*x_all[:, 1]) \
    #                * np.sin(2*np.pi*x_all[:, 2])
    f = -3. * np.exp(-x_all[:, 0] - x_all[:, 1] - x_all[:,2])
    f = torch.tensor(f, device=dev, dtype=torch.float64)
    # f *= 0; f += 1
tstep=int(np.ceil((tend-tstart)/dt))+1
c = c.reshape(nele,nloc) # reshape doesn't change memory allocation.
# c = torch.tensor(c, dtype=torch.float64, device=dev).view(-1,1,nloc)
c_bc = c.detach().clone() # this stores Dirichlet boundary *only*, otherwise zero.

# print('c_bc',c_bc)
c = torch.rand_like(c)*0
c_all=np.empty([tstep,nonods])
c_all[0,:]=c.view(-1).cpu().numpy()[:]
print('5. time elapsed, ',time.time()-starttime)

print('6. time elapsed, ',time.time()-starttime)

if False:  # per element condensation Rotation matrix (diagonal)
    if nloc == 10:
        # R = torch.tensor([1./30., 1./30., 1./30., 3./40., 3./40., 3./40., \
        #     3./40., 3./40., 3./40., 9./20.], device=dev, dtype=torch.float64)
        R = torch.tensor([1./10., 1./10., 1./10., 1./10., 1./10., 1./10., \
            1./10., 1./10., 1./10., 1./10.], device=dev, dtype=torch.float64)
    elif nloc == 3:
        R = torch.tensor([1./3., 1./3., 1./3.], device=dev, dtype=torch.float64)
    else:
        raise Exception('Element type is not accepted. Please check nloc.')
if True:  # from PnDG to P1CG
    I_fc_colx = np.arange(0, p1dg_nonods)
    I_fc_coly = cg_ndgln
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
    # np.savetxt('I_cf.txt', I_cf.to_dense().cpu().numpy(), delimiter=',')
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
                               np.arange(0, nele), np.arange(0, nele + 1)),
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
        # np.savetxt('I_fc.txt', I_fc.to_dense().cpu().numpy(), delimiter=',')
        # np.savetxt('I_cf.txt', I_cf.to_dense().cpu().numpy(), delimiter=',')
print('7. time elapsed, ',time.time()-starttime)

if (config.solver=='iterative') :
    print('i am going to time loop')
    print('8. time elapsed, ',time.time()-starttime)
    r0l2all=[]
    # time loop
    for itime in tqdm(range(1,tstep)):
        c_n = c.view(-1,1,nloc) # store last timestep value to cn
        c_i = c_n # jacobi iteration initial value taken as last time step value 

        r0l2=1
        its=0
        r0 = torch.zeros(config.nonods, device=dev, dtype=torch.float64)

        # prepare for MG on SFC-coarse grids
        RAR = calc_RAR_mf_color(I_fc, I_cf,
                                whichc, ncolor,
                                fina, cola, ncola)
        print(torch.cuda.mem_get_info(device=dev))
        print('RAR fina cola len: ', RAR.crow_indices().shape, RAR.col_indices().shape)
        print('finishing getting RAR: ', time.time() - starttime)
        # get RARmat (scipy csr format) for direct solver on P1CG (two-grid cycle)
        RARmat = sp.sparse.csr_matrix((RAR.values().cpu().numpy(),
                                       RAR.col_indices().cpu().numpy(),
                                       RAR.crow_indices().cpu().numpy()),
                                      shape=(cg_nonods, cg_nonods))
        # get SFC, coarse grid and operators on coarse grid. Store them to save computational time?
        space_filling_curve_numbering, variables_sfc, nlevel, nodes_per_level = \
            multi_grid.mg_on_P1CG_prep(RAR)
        # del RAR
        # np.savetxt('sfc.txt', space_filling_curve_numbering, delimiter=',')
        print('9. time elapsed, ', time.time()-starttime)
        # V-cycle
        # fine grid    o   o - o   o - o   o  
        #               \ /     \ /     \ /  ...
        # coarse grid    o       o       o
        if config.linear_solver == 'mg':
            c_i = solvers.multigrid_solver(c_i, c_n, c_bc, f,
                                           config.tol,
                                           space_filling_curve_numbering,
                                           variables_sfc,
                                           nlevel,
                                           nodes_per_level)
        if config.linear_solver == 'gmres-mg':
            c_i = solvers.gmres_mg_solver(c_i, c_n, c_bc, f,
                                          config.tol,
                                          space_filling_curve_numbering,
                                          variables_sfc,
                                          nlevel,
                                          nodes_per_level)
        if config.linear_solver == 'gmres':
            c_i = solvers.gmres_solver(c_i, c_n, c_bc, f,
                                       config.tol,
                                       space_filling_curve_numbering,
                                       variables_sfc,
                                       nlevel,
                                       nodes_per_level)

        # get final residual after we get back to fine mesh
        # c_i = c_i.view(-1,1,nloc)
        print('10. finishing cycles...', time.time()-starttime)
        r0 *= 0
        r0 = get_residual_only(
            r0,
            c_i, c_n, c_bc, f)

        r0l2 = torch.linalg.norm(r0,dim=0)
        r0l2all.append(r0l2.cpu().numpy())
        print('its=',its,'residual l2 norm=',r0l2.cpu().numpy())
            
        # if jacobi converges,
        c = c_i.view(nonods)
        c_all[itime,:]=c.view(-1).cpu().numpy()[:]

    np.savetxt('r0l2all.txt', np.asarray(r0l2all), delimiter=',')

if (config.solver=='direct'):
    # [diagS, S, b_bc] = S_Minv_sparse(sn, snx, sdetwei, snormal, \
    #     x_all, nbele, nbf, c_bc.view(-1))
    # # first transfer S and b_bc to scipy csr spM and np array
    # fina = S.crow_indices().cpu().numpy()
    # cola = S.col_indices().cpu().numpy()
    # values = S.values().cpu().numpy()
    # S_sp = sp.sparse.csr_matrix((values, cola, fina), shape=(nonods, nonods))
    # b_bc_np = b_bc.cpu().numpy()
    #
    # ### then assemble K as scipy csr spM
    # # calculate shape functions from element nodes coordinate
    # nx, detwei = get_det_nlx(nlx, x_ref_in, weight)
    # # transfer to cpu
    # nx = nx.cpu().numpy() # (nele, ndim, nloc, ngi)
    # detwei = detwei.cpu().numpy() # (nele, ngi)
    # indices = []
    # values = []
    # k = np.asarray([[1.,0.], [0.,1.]]) # this is diffusion coefficient | homogeneous, diagonal
    # for ele in range(nele):
    #     for iloc in range(nloc):
    #         glob_iloc = ele*nloc + iloc
    #         for jloc in range(nloc):
    #             glob_jloc = ele*nloc + jloc
    #             nxnx = 0
    #             for idim in range(ndim):
    #                 for gi in range(ngi):
    #                     nxnx += nx[ele,idim,iloc,gi] * k[idim,idim] * nx[ele,idim,jloc,gi] * detwei[ele,gi]
    #             indices.append([glob_iloc, glob_jloc])
    #             values.append(nxnx)
    #             # print(glob_iloc, glob_jloc, nxnx,';')
    # values = np.asarray(values)
    # indices = np.asarray(indices)
    # print(indices.shape)
    # K_sp = sp.sparse.coo_matrix((values, (indices[:,0], indices[:,1]) ), shape=(nonods, nonods))
    # K_sp = K_sp.tocsr()
    #
    # ## direct solver in scipy
    # c_i = sp.sparse.linalg.spsolve(S_sp+K_sp, b_bc_np)
    #
    # ## store to c_all to print out
    # c_all[1,:] = c_i
    #
    # if True:  # output a series of matrices for fourier analysis.
    #     np.savetxt('Amat.txt', (S_sp+K_sp).toarray(), delimiter=',')
    #
    #
    #     # '''
    #     # Continuous Galerkin discretisation
    #     # '''
    #     # # we need cg_ndgln, OK got it.
    #     # Kcg_idx = []
    #     # Kcg_val = []
    #     # for ele in range(nele):
    #     #     for iloc in range(3):
    #     #         glob_iloc = cg_ndgln[ele*3 + iloc]
    #     #         for jloc in range(3):
    #     #             glob_jloc = cg_ndgln[ele*3 + jloc]
    #     #             nxnx = 0
    #     #             for idim in range(ndim):
    #     #                 for gi in range(3):
    #     #                     nxnx += nx[ele,idim,iloc,gi] * k[idim,idim] * nx[ele,idim,jloc,gi] * detwei[ele,gi]
    #     #             # print(glob_iloc, glob_jloc, nxnx)
    #     #             Kcg_idx.append([glob_iloc, glob_jloc])
    #     #             Kcg_val.append(nxnx)
    #     # Kcg_idx = np.asarray(Kcg_idx)
    #     # Kcg_val = np.asarray(Kcg_val)
    #     # Kcg = sp.sparse.coo_matrix((Kcg_val, (Kcg_idx[:,0], Kcg_idx[:,1]) ), shape=(cg_nonods, cg_nonods))
    #     # Kcg = Kcg.tocsr()
    #     # # cg boundary condition strongly impose:
    #     # for bc in cg_bc:
    #     #     for inod in bc:
    #     #         Kcg[inod,:] = 0.
    #     #         Kcg[:,inod] = 0.
    #     #         Kcg[inod,inod] = 1.
    #     # np.savetxt('Kmat_cg.txt', Kcg.toarray(), delimiter=',')
    #
    #     # # let psi in P1CG, phi in P1DG
    #     # # compute projection matrix from P1DG to P1CG
    #     # # [psi_i psi_j]^(-1) * [psi_i phi_k]
    #     # #     M_psi    ^(-1) *    P_psi_phi
    #     # M_psi_idx = []
    #     # M_psi_val = []
    #     # P_psi_idx = []
    #     # P_psi_val = []
    #     # M_phi_idx = []
    #     # M_phi_val = []
    #     # for ele in range(nele):
    #     #     for iloc in range(3):
    #     #         glob_iloc = cg_ndgln[ele * 3 + iloc]
    #     #         glob_lloc = ele * 3 + iloc
    #     #         for jloc in range(3):
    #     #             kloc = jloc
    #     #             glob_jloc = cg_ndgln[ele * 3 + jloc]
    #     #             glob_kloc = ele * 3 + kloc
    #     #             # print(glob_jloc, glob_kloc)
    #     #             nn = 0.
    #     #             for gi in range(3):
    #     #                 nn += n[iloc,gi] * n[jloc,gi] * detwei[ele,gi]
    #     #             M_psi_idx.append([glob_iloc, glob_jloc])
    #     #             M_psi_val.append(nn.cpu())
    #     #             P_psi_idx.append([glob_iloc, glob_kloc])
    #     #             P_psi_val.append(nn.cpu())
    #     #             M_phi_idx.append([glob_lloc, glob_kloc])
    #     #             M_phi_val.append(nn.cpu())
    #     # M_psi_idx = np.asarray(M_psi_idx)
    #     # M_psi_val = np.asarray(M_psi_val)
    #     # M_psi = sp.sparse.coo_matrix((M_psi_val, (M_psi_idx[:, 0], M_psi_idx[:, 1])), shape=(cg_nonods, cg_nonods))
    #     # M_psi = M_psi.tocsr()
    #     # P_psi_idx = np.asarray(P_psi_idx)
    #     # P_psi_val = np.asarray(P_psi_val)
    #     # P_psi = sp.sparse.coo_matrix((P_psi_val, (P_psi_idx[:, 0], P_psi_idx[:, 1])), shape=(cg_nonods, p1dg_nonods))
    #     # P_psi = P_psi.tocsr()
    #     # M_phi_idx = np.asarray(M_phi_idx)
    #     # M_phi_val = np.asarray(M_phi_val)
    #     # M_phi = sp.sparse.coo_matrix((M_phi_val, (M_phi_idx[:, 0], M_phi_idx[:, 1])), shape=(p1dg_nonods, p1dg_nonods))
    #     # M_phi = M_phi.tocsr()
    #     # np.savetxt('Mpsimat.txt', M_psi.toarray(), delimiter=',')
    #     # np.savetxt('Ppsimat.txt', P_psi.toarray(), delimiter=',')
    #     # np.savetxt('Mphimat.txt', M_phi.toarray(), delimiter=',')
    #     np.savetxt('cg_ndglno.txt', cg_ndgln, delimiter=',')

    # # ----- new matrix assemble for direct solver -----
    # dummy1 = torch.zeros(nonods, device=dev, dtype=torch.float64)
    # dummy2 = torch.zeros(nonods, device=dev, dtype=torch.float64)
    # dummy3 = torch.zeros(nonods, device=dev, dtype=torch.float64)
    # dummy4 = torch.zeros(nonods, device=dev, dtype=torch.float64)
    # Amat = torch.zeros(nonods, nonods, device=dev, dtype=torch.float64)
    # rhs = torch.zeros(nonods, device=dev, dtype=torch.float64)
    # probe = torch.zeros(nonods, device=dev, dtype=torch.float64)
    # np.savetxt('f.txt', f.cpu().numpy(), delimiter=',')
    # dummy1 *= 0
    # dummy2 *= 0
    # rhs = volume_integral.get_residual_only(r0=rhs,
    #                                         c_i=dummy1,
    #                                         c_n=dummy2,
    #                                         c_bc=c_bc,
    #                                         f=f)
    # for inod in tqdm(range(nonods)):
    #     dummy1 *= 0
    #     dummy2 *= 0
    #     dummy3 *= 0
    #     dummy4 *= 0
    #     probe *= 0
    #     probe[inod] = 1.
    #     Amat[:, inod] -= volume_integral.get_residual_only(r0=dummy1,
    #                                                        c_i=probe,
    #                                                        c_n=dummy2,
    #                                                        c_bc=dummy3,
    #                                                        f=dummy4)
    # np.savetxt('Amat.txt', Amat.cpu().numpy(), delimiter=',')
    # np.savetxt('rhs.txt', rhs.cpu().numpy(), delimiter=',')
    # Amat_np = sp.sparse.csr_matrix(Amat.cpu().numpy())
    # rhs_np = rhs.cpu().numpy()

    # proper assembly matrix
    import diffusion_3d_assemble
    print('im going to assemble', time.time()-starttime)
    Amat_np, rhs_np = diffusion_3d_assemble.assemble(c_bc, f)
    print('im going to solve', time.time()-starttime)
    np.savetxt('f.txt', f.cpu().numpy(), delimiter=',')
    np.savetxt('Amat.txt', Amat_np.toarray(), delimiter=',')
    np.savetxt('rhs.txt', rhs_np, delimiter=',')
    c_i = sp.sparse.linalg.spsolve(Amat_np, rhs_np)
    print('ive done solving', time.time() - starttime)
    c_all[1,:] = c_i

#############################################################
# write output
#############################################################
# output 1: 
# c_all = np.asarray(c_all)[::1,:]
np.savetxt('c_all.txt', c_all, delimiter=',')
np.savetxt('x_all.txt', x_all, delimiter=',')
print(torch.cuda.memory_summary())
