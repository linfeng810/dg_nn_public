#!/usr/bin/env python3

####################################################
# preamble
####################################################
# import
import time, os.path

starttime = time.time()

import toughio 
import numpy as np
import torch
from torch.nn import Conv1d,Sequential,Module
import scipy as sp
# import time
from scipy.sparse import coo_matrix, bsr_matrix
from tqdm import tqdm
import config
import mesh_init 
# from mesh_init import face_iloc,face_iloc2
from shape_function import SHATRInew, det_nlx, sdet_snlx, get_det_nlx
from surface_integral import S_Minv_sparse, RSR_DG_to_CG, RSR_DG_to_CG_color
import surface_integral_mf
from volume_integral import K_mf, calc_RKR, calc_RAR, \
    RKR_DG_to_CG, RKR_DG_to_CG_color, RAR_DG_to_CG, \
    calc_RAR_mf_color, get_residual_and_smooth_once, \
    get_residual_only
from color import color2
import multi_grid

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

[x_all, nbf, nbele, fina, cola, ncola, bc1, bc2, bc3, bc4, cg_ndgln, cg_nonods, cg_bc] = mesh_init.init()
# [fina, cola, ncola] = mesh_init.connectivity(nbele)
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
n = torch.tensor(n, device=config.dev, dtype=torch.float64)
nlx = torch.tensor(nlx, device=config.dev, dtype=torch.float64)
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
x_ref_in = torch.tensor(x_ref_in, device=dev, requires_grad=False)
# print(x_ref_in)
# initical condition
c = torch.zeros(nele*nloc, device=dev, dtype=torch.float64)
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
    x_inod = x_ref_in[inod//nloc, 0, inod%nloc]
    c[inod]= torch.sin(torch.pi*x_inod)
    # y_inod = x_ref_in[inod//10, 1, inod%10]
    # print('bc4 inod %d x %f y %f'%(inod,x_inod, y_inod))
    # c[inod]= x_inod
    # print("x, c", x_inod.cpu().numpy(), c[inod])


tstep=int(np.ceil((tend-tstart)/dt))+1
c = c.reshape(nele,nloc) # reshape doesn't change memory allocation.
# c = torch.tensor(c, dtype=torch.float64, device=dev).view(-1,1,nloc)
c_bc = c.detach().clone() # this stores Dirichlet boundary *only*, otherwise zero.

# print('c_bc',c_bc)
c = torch.rand_like(c)*0
c_all=np.empty([tstep,nonods])
c_all[0,:]=c.view(-1).cpu().numpy()[:]
print('5. time elapsed, ',time.time()-starttime)
## surface integral 

# # prepare shape functions ** only once ** store for all future usages
# [snx, sdetwei, snormal] = sdet_snlx(snlx, x_ref_in, sweight)
# # with torch.no_grad():
# #     nx, detwei = Det_nlx.forward(x_ref_in, weight)
# nx, detwei = get_det_nlx(nlx, x_ref_in, weight)

# put numpy array to torch tensor in expected device
sn = torch.tensor(sn, dtype=torch.float64, device=dev)
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
        RAR = calc_RAR_mf_color(n, nlx, weight,
                                sn, snlx, sweight,
                                x_ref_in,
                                nbele, nbf,
                                I_fc, I_cf,
                                whichc, ncolor,
                                fina, cola, ncola)
        print(torch.cuda.mem_get_info(device=dev))
        print('RAR fina cola len: ', RAR.crow_indices().shape, RAR.col_indices().shape)
        print('finishing getting RAR: ', time.time()-starttime)
        # get SFC, coarse grid and operators on coarse grid. Store them to save computational time?
        space_filling_curve_numbering, variables_sfc, nlevel, nodes_per_level = \
            multi_grid.mg_on_P0DG_prep(RAR, x_ref_in)
        # del RAR
        # np.savetxt('sfc.txt', space_filling_curve_numbering, delimiter=',')
        print('9. time elapsed, ', time.time()-starttime)
        # V-cycle
        # fine grid    o   o - o   o - o   o  
        #               \ /     \ /     \ /  ...
        # coarse grid    o       o       o
        while (r0l2>1e-9 and its<config.jac_its):
            c_i = c_i.view(-1,1,nloc)

            for its1 in range(config.pre_smooth_its):
                # on fine grid
                r0 *= 0
                r0, c_i = get_residual_and_smooth_once(
                    r0,
                    c_i, c_n, c_bc, x_ref_in,
                    n, nlx, weight,
                    sn, snlx, sweight,
                    nbele, nbf)

            # residual on PnDG
            r0 *= 0
            r0 = get_residual_only(
                r0,
                c_i, c_n, c_bc, x_ref_in,
                n, nlx, weight,
                sn, snlx, sweight,
                nbele, nbf)
            
            if False:  # PnDG to P0DG
                # per element condensation
                # passing r0 to next level coarse grid and solve Ae=r0
                r1 = torch.matmul(r0.view(-1,nloc), R.view(nloc,1)) # restrict residual to coarser mesh, (nele, 1)
            if True:  # P1DG to P1CG
                r1 = multi_grid.p3dg_to_p1dg_restrictor(r0)
                r1 = torch.matmul(I_cf, r1)
            # reordering node according to SFC
            ncurve = 1  # always use 1 sfc
            N = len(space_filling_curve_numbering)
            inverse_numbering = np.zeros((N, ncurve), dtype=int)
            inverse_numbering[:, 0] = np.argsort(space_filling_curve_numbering[:, 0])
            r1_sfc = r1[inverse_numbering[:, 0]].view(1, 1, config.cg_nonods)

            e_i = torch.zeros((cg_nonods,1), device=dev, dtype=torch.float64)
            rr1 = r1_sfc.detach().clone()
            rr1_l2_0 = torch.linalg.norm(rr1.view(-1),dim=0)
            rr1_l2 = 10.
            its1=0
            # while its1 < config.mg_its[0] and rr1_l2 > config.mg_tol:
            if True:  # smooth on P1CG
                for _ in range(config.pre_smooth_its):
                    # smooth (solve) on level 1 coarse grid (R^T A R e = r1)
                    rr1 = r1_sfc - torch.sparse.mm(variables_sfc[0][0], e_i).view(-1)
                    rr1_l2 = torch.linalg.norm(rr1.view(-1), dim=0) / rr1_l2_0

                    diagA1 = 1. / variables_sfc[0][2]
                    e_i = e_i.view(cg_nonods, 1) + config.jac_wei * torch.mul(diagA1, rr1.view(-1)).view(-1, 1)
                # for _ in range(100):
                if True:  # SFC multi-grid saw-tooth iteration
                    rr1 = r1_sfc - torch.sparse.mm(variables_sfc[0][0], e_i).view(-1)
                    rr1_l2 = torch.linalg.norm(rr1.view(-1), dim=0) / rr1_l2_0
                    # use SFC to generate a series of coarse grid
                    # and iterate there (V-cycle saw-tooth fasion)
                    # then return a residual on level-1 grid (P1CG)
                    e_i = multi_grid.mg_on_P0DG(
                        r1_sfc.view(cg_nonods, 1),
                        rr1,
                        e_i,
                        space_filling_curve_numbering,
                        variables_sfc,
                        nlevel,
                        nodes_per_level)

                else:  # direct solver on first SFC coarsened grid (thus constitutes a 3-level MG)
                    rr1 = r1_sfc - torch.sparse.mm(variables_sfc[0][0], e_i).view(-1)
                    rr1_l2 = torch.linalg.norm(rr1.view(-1), dim=0) / rr1_l2_0
                    level = 1

                    # restrict residual
                    sfc_restrictor = torch.nn.Conv1d(in_channels=1,
                                                     out_channels=1, kernel_size=2,
                                                     stride=2, padding='valid', bias=False)
                    sfc_restrictor.weight.data = \
                        torch.tensor([[1., 1.]],
                                     dtype=torch.float64,
                                     device=config.dev).view(1, 1, 2)
                    b = torch.nn.functional.pad(rr1, (0, 1), "constant", 0)  # residual on level1 sfc coarse grid
                    with torch.no_grad():
                        b = sfc_restrictor(b)
                        # np.savetxt('b.txt', b.view(-1).cpu().numpy(), delimiter=',')
                    # get operator
                    a_sfc_l1 = variables_sfc[level][0]
                    cola = a_sfc_l1.col_indices().detach().clone().cpu().numpy()
                    fina = a_sfc_l1.crow_indices().detach().clone().cpu().numpy()
                    vals = a_sfc_l1.values().detach().clone().cpu().numpy()
                    # direct solve on level1 sfc coarse grid
                    from scipy.sparse import csr_matrix, linalg
                    # np.savetxt('a_sfc_l0.txt', variables_sfc[0][0].to_dense().cpu().numpy(), delimiter=',')
                    # np.savetxt('a_sfc_l1.txt', variables_sfc[1][0].to_dense().cpu().numpy(), delimiter=',')
                    a_on_l1 = csr_matrix((vals, cola, fina), shape=(nodes_per_level[level], nodes_per_level[level]))
                    # np.savetxt('a_sfc_l1_sp.txt', a_on_l1.todense(), delimiter=',')
                    e_i_direct = linalg.spsolve(a_on_l1, b.view(-1).cpu().numpy())
                    # prolongation
                    e_ip1 = torch.tensor(e_i_direct, device=config.dev, dtype=torch.float64).view(1, 1, -1)
                    CNN1D_prol_odd = torch.nn.Upsample(scale_factor=nodes_per_level[level - 1] / nodes_per_level[level])
                    e_ip1 = CNN1D_prol_odd(e_ip1.view(1, 1, -1))
                    e_i += torch.squeeze(e_ip1).view(cg_nonods, 1)
                    # e_i += e_ip1[0, 0, space_filling_curve_numbering[:, 0] - 1].view(-1,1)

                    for _ in range(config.post_smooth_its):
                        # smooth (solve) on level 1 coarse grid (R^T A R e = r1)
                        rr1 = r1_sfc-torch.sparse.mm(variables_sfc[0][0], e_i).view(-1)
                        rr1_l2 = torch.linalg.norm(rr1.view(-1),dim=0)/rr1_l2_0

                        diagA1 = 1./variables_sfc[0][2]
                        e_i = e_i.view(cg_nonods,1) + config.jac_wei * torch.mul(diagA1, rr1.view(-1)).view(-1,1)

                its1 += 1
                # print('its1: %d, residual on P1CG: '%(its1), rr1_l2)
            # reverse to original order
            e_i = e_i[space_filling_curve_numbering[:, 0] - 1, 0].view(-1,1)

            if False:  # direc solve on P1CG R^T A R e = r1?
                e_direct = sp.sparse.linalg.spsolve(RARmat, r1.contiguous().view(-1).cpu().numpy())
                # np.savetxt('RARmat.txt', RARmat.toarray(), delimiter=',')
                e_i = e_i.view(-1)
                # print(e_i - torch.tensor(e_direct, device=dev, dtype=torch.float64))
                e_i += torch.tensor(e_direct, device=dev, dtype=torch.float64)

            # np.savetxt('e_i_back.txt', e_i.cpu().numpy(), delimiter=',')
            if False:  # from P0DG to PnDG
                # pass e_i back to fine mesh
                e_i0 = torch.sparse.mm(torch.transpose(RTbig, dim0=0, dim1=1), e_i.view(-1,1))
            if True:  # from P1CG to P1DG
                e_i0 = torch.mv(I_fc, e_i.view(-1))
                e_i0 = multi_grid.p1dg_to_p3dg_prolongator(e_i0)
            c_i = c_i.view(-1) + e_i0.view(-1)

            for _ in range(config.post_smooth_its):
                # post smooth
                r0 *= 0
                r0, c_i = get_residual_and_smooth_once(
                    r0,
                    c_i, c_n, c_bc, x_ref_in,
                    n, nlx, weight,
                    sn, snlx, sweight,
                    nbele, nbf)
            # np.savetxt('c_i.txt', c_i.cpu().numpy(), delimiter=',')
            # np.savetxt('r0.txt', r0.cpu().numpy(), delimiter=',')

            r0l2 = torch.linalg.norm(r0,dim=0)
            # print('its=',its,'fine grid residual l2 norm=',r0l2.cpu().numpy())
            print('P1CG its=', its1, 'its=', its, 'fine grid residual l2 norm=', r0l2.cpu().numpy())
            r0l2all.append(r0l2.cpu().numpy())

            its+=1

        # get final residual after we get back to fine mesh
        # c_i = c_i.view(-1,1,nloc)
        print('10. finishing cycles...', time.time()-starttime)
        r0 *= 0
        r0 = get_residual_only(
            r0,
            c_i, c_n, c_bc, x_ref_in,
            n, nlx, weight,
            sn, snlx, sweight,
            nbele, nbf)

        r0l2 = torch.linalg.norm(r0,dim=0)
        r0l2all.append(r0l2.cpu().numpy())
        print('its=',its,'residual l2 norm=',r0l2.cpu().numpy())
            
        # if jacobi converges,
        c = c_i.view(nonods)
        c_all[itime,:]=c.view(-1).cpu().numpy()[:]

    np.savetxt('r0l2all.txt', np.asarray(r0l2all), delimiter=',')

if (config.solver=='direct'):
    [diagS, S, b_bc] = S_Minv_sparse(sn, snx, sdetwei, snormal, \
        x_all, nbele, nbf, c_bc.view(-1))
    # first transfer S and b_bc to scipy csr spM and np array
    fina = S.crow_indices().cpu().numpy()
    cola = S.col_indices().cpu().numpy()
    values = S.values().cpu().numpy()
    S_sp = sp.sparse.csr_matrix((values, cola, fina), shape=(nonods, nonods))
    b_bc_np = b_bc.cpu().numpy() 

    ### then assemble K as scipy csr spM
    # calculate shape functions from element nodes coordinate
    nx, detwei = get_det_nlx(nlx, x_ref_in, weight)
    # transfer to cpu
    nx = nx.cpu().numpy() # (nele, ndim, nloc, ngi)
    detwei = detwei.cpu().numpy() # (nele, ngi)
    indices = []
    values = []
    k = np.asarray([[1.,0.], [0.,1.]]) # this is diffusion coefficient | homogeneous, diagonal
    for ele in range(nele):
        for iloc in range(nloc):
            glob_iloc = ele*nloc + iloc 
            for jloc in range(nloc):
                glob_jloc = ele*nloc + jloc 
                nxnx = 0
                for idim in range(ndim):
                    for gi in range(ngi):
                        nxnx += nx[ele,idim,iloc,gi] * k[idim,idim] * nx[ele,idim,jloc,gi] * detwei[ele,gi]
                indices.append([glob_iloc, glob_jloc])
                values.append(nxnx)
                # print(glob_iloc, glob_jloc, nxnx,';')
    values = np.asarray(values)
    indices = np.asarray(indices)
    print(indices.shape)
    K_sp = sp.sparse.coo_matrix((values, (indices[:,0], indices[:,1]) ), shape=(nonods, nonods))
    K_sp = K_sp.tocsr()

    ## direct solver in scipy
    c_i = sp.sparse.linalg.spsolve(S_sp+K_sp, b_bc_np)

    ## store to c_all to print out
    c_all[1,:] = c_i

    if True:  # output a series of matrices for fourier analysis.
        np.savetxt('Amat.txt', (S_sp+K_sp).toarray(), delimiter=',')


        # '''
        # Continuous Galerkin discretisation
        # '''
        # # we need cg_ndgln, OK got it.
        # Kcg_idx = []
        # Kcg_val = []
        # for ele in range(nele):
        #     for iloc in range(3):
        #         glob_iloc = cg_ndgln[ele*3 + iloc]
        #         for jloc in range(3):
        #             glob_jloc = cg_ndgln[ele*3 + jloc]
        #             nxnx = 0
        #             for idim in range(ndim):
        #                 for gi in range(3):
        #                     nxnx += nx[ele,idim,iloc,gi] * k[idim,idim] * nx[ele,idim,jloc,gi] * detwei[ele,gi]
        #             # print(glob_iloc, glob_jloc, nxnx)
        #             Kcg_idx.append([glob_iloc, glob_jloc])
        #             Kcg_val.append(nxnx)
        # Kcg_idx = np.asarray(Kcg_idx)
        # Kcg_val = np.asarray(Kcg_val)
        # Kcg = sp.sparse.coo_matrix((Kcg_val, (Kcg_idx[:,0], Kcg_idx[:,1]) ), shape=(cg_nonods, cg_nonods))
        # Kcg = Kcg.tocsr()
        # # cg boundary condition strongly impose:
        # for bc in cg_bc:
        #     for inod in bc:
        #         Kcg[inod,:] = 0.
        #         Kcg[:,inod] = 0.
        #         Kcg[inod,inod] = 1.
        # np.savetxt('Kmat_cg.txt', Kcg.toarray(), delimiter=',')

        # # let psi in P1CG, phi in P1DG
        # # compute projection matrix from P1DG to P1CG
        # # [psi_i psi_j]^(-1) * [psi_i phi_k]
        # #     M_psi    ^(-1) *    P_psi_phi
        # M_psi_idx = []
        # M_psi_val = []
        # P_psi_idx = []
        # P_psi_val = []
        # M_phi_idx = []
        # M_phi_val = []
        # for ele in range(nele):
        #     for iloc in range(3):
        #         glob_iloc = cg_ndgln[ele * 3 + iloc]
        #         glob_lloc = ele * 3 + iloc
        #         for jloc in range(3):
        #             kloc = jloc
        #             glob_jloc = cg_ndgln[ele * 3 + jloc]
        #             glob_kloc = ele * 3 + kloc
        #             # print(glob_jloc, glob_kloc)
        #             nn = 0.
        #             for gi in range(3):
        #                 nn += n[iloc,gi] * n[jloc,gi] * detwei[ele,gi]
        #             M_psi_idx.append([glob_iloc, glob_jloc])
        #             M_psi_val.append(nn.cpu())
        #             P_psi_idx.append([glob_iloc, glob_kloc])
        #             P_psi_val.append(nn.cpu())
        #             M_phi_idx.append([glob_lloc, glob_kloc])
        #             M_phi_val.append(nn.cpu())
        # M_psi_idx = np.asarray(M_psi_idx)
        # M_psi_val = np.asarray(M_psi_val)
        # M_psi = sp.sparse.coo_matrix((M_psi_val, (M_psi_idx[:, 0], M_psi_idx[:, 1])), shape=(cg_nonods, cg_nonods))
        # M_psi = M_psi.tocsr()
        # P_psi_idx = np.asarray(P_psi_idx)
        # P_psi_val = np.asarray(P_psi_val)
        # P_psi = sp.sparse.coo_matrix((P_psi_val, (P_psi_idx[:, 0], P_psi_idx[:, 1])), shape=(cg_nonods, p1dg_nonods))
        # P_psi = P_psi.tocsr()
        # M_phi_idx = np.asarray(M_phi_idx)
        # M_phi_val = np.asarray(M_phi_val)
        # M_phi = sp.sparse.coo_matrix((M_phi_val, (M_phi_idx[:, 0], M_phi_idx[:, 1])), shape=(p1dg_nonods, p1dg_nonods))
        # M_phi = M_phi.tocsr()
        # np.savetxt('Mpsimat.txt', M_psi.toarray(), delimiter=',')
        # np.savetxt('Ppsimat.txt', P_psi.toarray(), delimiter=',')
        # np.savetxt('Mphimat.txt', M_phi.toarray(), delimiter=',')
        np.savetxt('cg_ndglno.txt', cg_ndgln, delimiter=',')

#############################################################
# write output
#############################################################
# output 1: 
# c_all = np.asarray(c_all)[::1,:]
np.savetxt('c_all.txt', c_all, delimiter=',')
np.savetxt('x_all.txt', x_all, delimiter=',')
print(torch.cuda.memory_summary())