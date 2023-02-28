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
# from scipy.sparse import coo_matrix 
from tqdm import tqdm
import config
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
ngi = config.ngi
ndim = config.ndim
nloc = config.nloc 
dt = config.dt 
tend = config.tend 
tstart = config.tstart

print('computation on ',dev)
print('nele=', nele)

[x_all, nbf, nbele, fina, cola, ncola, bc1, bc2, bc3, bc4 ] =mesh_init.init()
# [fina, cola, ncola] = mesh_init.connectivity(nbele)
# coloring and get probing vector
[whichc, ncolor] = color2(fina=fina, cola=cola, nnode = nele)
# np.savetxt('whichc.txt', whichc, delimiter=',')
print('ncolor', ncolor, 'whichc type', whichc.dtype)
print('1. time elapsed, ',time.time()-starttime)
#####################################################
# shape functions
#####################################################
# get shape functions on reference element
[n,nlx,weight,sn,snlx,sweight] = SHATRInew(config.nloc, \
    config.ngi, config.ndim)
nlx = torch.tensor(nlx, device=dev, dtype=torch.float64)
## set weights in det_nlx
Det_nlx = det_nlx(nlx)
Det_nlx.to(dev)
print('2. time elapsed, ',time.time()-starttime)
# filter for calc jacobian
calc_j11_j12_filter = torch.transpose(nlx[0,:,:],0,1) # dN/dx
calc_j11_j12_filter = calc_j11_j12_filter.unsqueeze(1) # (ngi, 1, nloc)
calc_j21_j22_filter = torch.transpose(nlx[1,:,:],0,1) # dN/dy
calc_j21_j22_filter = calc_j21_j22_filter.unsqueeze(1) # (ngi, 1, nloc)
# print(Det_nlx.calc_j11.weight.shape)
# print(nlx.shape)
# print(calc_j21_j22_filter.shape)
Det_nlx.calc_j11.weight.data = calc_j11_j12_filter
Det_nlx.calc_j12.weight.data = calc_j11_j12_filter
Det_nlx.calc_j21.weight.data = calc_j21_j22_filter
Det_nlx.calc_j22.weight.data = calc_j21_j22_filter
# print(Det_nlx.calc_j11.weight.shape)
# print(Det_nlx.calc_j11.weight.data)
print('3. time elapsed, ',time.time()-starttime)
#######################################################
# assemble local mass matrix and stiffness matrix
#######################################################


Mk = mk()
Mk.to(device=dev)
n = torch.tensor(n, device=dev, dtype=torch.float64)

Mk1 = mk_lv1() # level 1 mass/stiffness operator 
Mk.to(device=dev)
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

# initical condition
u = torch.zeros(ndim, nonods, device=dev, dtype=torch.float64) # now we have a vector filed to solve

# apply boundary conditions (4 Dirichlet bcs)
for inod in bc1:
    u[:,inod]=0.
for inod in bc2:
    u[:,inod]=0.
for inod in bc3:
    u[:,inod]=0.
for inod in bc4:
    u[:,inod]=0.    

tstep=int(np.ceil((tend-tstart)/dt))+1
u = u.reshape(ndim,nele,nloc) # reshape doesn't change memory allocation.
u_bc = u.detach().clone() # this stores Dirichlet boundary *only*, otherwise zero.

f = config.rhs_f(x_all) # rhs force

u = torch.rand_like(u) # initial guess
u_all=np.empty([tstep,ndim,nonods])
u_all[0,:,:]=u.view(ndim,nonods).cpu().numpy()[:]
print('5. time elapsed, ',time.time()-starttime)
## surface integral 

# surface shape functions 
# output :
# snx: (nele, nface, ndim, nloc, sngi)
# sdetwei: (nele, nface, sgni)
[snx, sdetwei, snormal] = sdet_snlx(snlx, x_ref_in, sweight)
# put numpy array to torch tensor in expected device
sn = torch.tensor(sn, dtype=torch.float64, device=dev)
print('6. time elapsed, ',time.time()-starttime)

# per element condensation Rotation matrix (diagonal)
# R = torch.tensor([1./30., 1./30., 1./30., 3./40., 3./40., 3./40., \
#     3./40., 3./40., 3./40., 9./20.], device=dev, dtype=torch.float64)
R = torch.tensor([1./10., 1./10., 1./10., 1./10., 1./10., 1./10., \
    1./10., 1./10., 1./10., 1./10.], device=dev, dtype=torch.float64)
print('7. time elapsed, ',time.time()-starttime)

if (config.solver=='iterative') :
    # # surface integral operator restrcted by R
    # # [diagRSR, RSR, RTbig] = RSR_matrix(S,R) # matmat multiplication
    # [diagS, S, b_bc] = S_Minv_sparse(sn, snx, sdetwei, snormal, \
    #     x_all, nbele, nbf, u_bc.view(-1))
    # [diagRSR, RSR, RTbig] = RSR_matrix_color(S,R, whichc, ncolor, fina, cola, ncola) # matvec multiplication
    # del S, diagS, b_bc

    # go matrix free RSR calculation
    [diagRSR, RSRvalues] = RSR_mf_color(R, whichc, ncolor, fina, cola, ncola,
                                  sn, snx, sdetwei, snormal, nbele, nbf)
    # print('diagRSR',diagRSR)
    print('i am going to time loop')
    print('8. time elapsed, ',time.time()-starttime)
    # print("Using quit()")
    # quit()
    r0l2all=[]
    # time loop
    r0 = torch.zeros(ndim, nonods, device=dev, dtype=torch.float64)
    for itime in tqdm(range(1,tstep)):
        u_n = u.view(-1,ndim,nloc) # store last timestep value to un
        u_i = u_n # jacobi iteration initial value taken as last time step value 

        r0l2=1
        its=0
        r0 *= 0
        ## prepare for MG on SFC-coarse grids
        with torch.no_grad():
            nx, detwei = Det_nlx.forward(x_ref_in, weight)
        # RKR = calc_RKR(n=n, nx=nx, detwei=detwei, R=R, k=1,dt=1)
        [diagRKR, RKRvalues] = volume_mf_linear_elastic.RKR_mf(n, nx, detwei, R)
        [diagRAR, RARvalues] = volume_mf_linear_elastic.calc_RAR(\
            diagRSR, diagRKR, RSRvalues, RKRvalues, fina, cola)
        del diagRKR, diagRSR, RKRvalues, RSRvalues # we will only use diagRAR and RARvalues
        RARvalues = torch.permute(RARvalues, (1,2,0)).contiguous() # (ndim,ndim,ncola)
        # get SFC, coarse grid and operators on coarse grid. Store them to save computational time?
        space_filling_curve_numbering, variables_sfc, nlevel, nodes_per_level = \
            mg.mg_on_P0DG_prep(fina, cola, RARvalues)
        print('9. time elapsed, ', time.time()-starttime)
        # calculate shape functions from element nodes coordinate
        with torch.no_grad():
            nx, detwei = Det_nlx.forward(x_ref_in, weight)
        # sawtooth iteration : sooth one time at each white dot
        # fine grid    o   o - o   o - o   o  
        #               \ /     \ /     \ /  ...
        # coarse grid    o       o       o
        while (r0l2>1e-9 and its<config.jac_its):
            u_i = u_i.view(ndim, nonods)
            
            ## on fine grid
            # get diagA and residual at fine grid r0
            with torch.no_grad():
                r0, _ = volume_mf_linear_elastic.K_mf(
                    r0, n, nx, detwei, u_i, f )
            # r0 *= 0.
            # r0 = r0.view(nonods,1) - torch.sparse.mm(S, c_i.view(nonods,1))
            [r0, _] = surface_mf_linear_elastic.S_mf(r0,
                            sn, snx, sdetwei, snormal, nbele, nbf, u_bc, u_i)
            
            # per element condensation
            # passing r0 to next level coarse grid and solve Ae=r0
            r1 = torch.matmul(r0.view(ndim,nele,nloc),R) # restrict residual to coarser mesh, (ndim, nele)

            # for its1 in range(config.mg_its):
                
            ## use SFC to generate a series of coarse grid
            # and iterate there (V-cycle saw-tooth fasion)
            # then return an error on level-1 grid (P0DG)
            e1 = mg.mg_smooth(r1, 
                space_filling_curve_numbering, 
                variables_sfc, 
                nlevel, 
                nodes_per_level)
            
            ## smooth (solve) on level 1 coarse grid (R^T A R e = r1)
            e1 = volume_mf_linear_elastic.RAR_smooth(r1, e1, 
                                                     RARvalues,
                                                     fina, cola, 
                                                     diagRAR)

            # pass e_1 back to fine mesh and correct u_i
            # print(R.view(nloc,1).shape)
            # print(e1.view(ndim,1,nele).shape)
            u_i += torch.matmul(R.view(nloc,1), e1.view(ndim,1,nele)).view(ndim,nonods)
            ## finally give residual
            # get diagA and residual at fine grid r0
            r0 *=0
            with torch.no_grad():
                r0, diagK = volume_mf_linear_elastic.K_mf(
                    r0, n, nx, detwei, u_i, f )
            [r0, diagS] = surface_mf_linear_elastic.S_mf(r0,
                            sn, snx, sdetwei, snormal, nbele, nbf, u_bc, u_i)
            # I think we need to smooth one time at finest grid as well.
            u_i += config.jac_wei * r0 / (diagS+diagK)
            r0l2 = torch.linalg.norm(r0.view(-1),dim=0)
            print('its=',its,'fine grid residual l2 norm=',r0l2.cpu().numpy())
            r0l2all.append(r0l2.cpu().numpy())

            its+=1

        ## smooth a final time after we get back to fine mesh
        # c_i = c_i.view(-1,1,nloc)
        
        # # calculate shape functions from element nodes coordinate
        # with torch.no_grad():
        #     nx, detwei = Det_nlx.forward(x_ref_in, weight)

        r0 *= 0
        with torch.no_grad():
            r0, diagK = volume_mf_linear_elastic.K_mf(
                r0, n, nx, detwei, u_i, f)
        [r0, diagS] = surface_mf_linear_elastic.S_mf(r0,
                                                     sn, snx, sdetwei, snormal, nbele, nbf, u_bc, u_i)
        # I think we need to smooth one time at finest grid as well.
        u_i += config.jac_wei * r0 / (diagS + diagK)
        r0l2 = torch.linalg.norm(r0.view(-1), dim=0)
        r0l2all.append(r0l2.cpu().numpy())
        print('its=',its,'residual l2 norm=',r0l2.cpu().numpy())
            
        # if jacobi converges,
        u = u_i.view(nonods)

        # combine inner/inter element contribution
        u_all[itime,:,:]=u.view(ndim, nonods).cpu().numpy()

    np.savetxt('r0l2all.txt', np.asarray(r0l2all), delimiter=',')

if (config.solver=='direct'):
    from assemble_matrix_for_direct_solver import SK_matrix
    with torch.no_grad():
        nx, detwei = Det_nlx.forward(x_ref_in, weight)
    # assemble S+K matrix and rhs (force + bc)
    SK, rhs_b = SK_matrix(n, nx, detwei,
                          sn, snx, sdetwei, snormal,
                          nbele, nbf, f, u_bc,
                          fina, cola, ncola)
    np.savetxt('SK.txt', SK.toarray(), delimiter=',')
    # direct solver in scipy
    SK = SK.tocsr()
    u_i = sp.sparse.linalg.spsolve(SK, rhs_b)  # shape nele*ndim*nloc
    u_i = np.reshape(u_i, (nele, ndim, nloc))
    u_i = np.transpose(u_i, (1,0,2))
    u_i = np.reshape(u_i, (ndim, nele*nloc))
    # store to c_all to print out
    u_all[1,:,:] = u_i

#############################################################
# write output
#############################################################
# output 1: 
# to output, we need to change u_all to 2D array
# (tstep, ndim*nonods)
u_all = np.reshape(u_all, (tstep, ndim*nonods))
np.savetxt('u_all.txt', u_all, delimiter=',')
np.savetxt('x_all.txt', x_all, delimiter=',')
print(torch.cuda.memory_summary())