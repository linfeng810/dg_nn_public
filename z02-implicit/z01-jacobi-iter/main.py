#!/usr/bin/env python3

####################################################
# preamble
####################################################
# import 
import toughio 
import numpy as np
import torch
from torch.nn import Conv1d,Sequential,Module
# import time
# from scipy.sparse import coo_matrix 
from tqdm import tqdm
import config
import mesh_init 
# from mesh_init import face_iloc,face_iloc2
from shape_function import SHATRInew, det_nlx
from surface_integral import S_Minv_sparse
from volume_integral import mk

# for pretty print out torch tensor
# torch.set_printoptions(sci_mode=False)

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
print('totel ele no. ', nele)

[x_all, nbf, nbele, bc1, bc2, bc3, bc4 ] =mesh_init.init()

#####################################################
# shape functions
#####################################################

## set weights in det_nlx
[n, nlx, weight] = SHATRInew(nloc, ngi, ndim)
Det_nlx = det_nlx(nlx)
Det_nlx.to(dev)

# filter for calc jacobian
calc_j11_j12_filter = np.transpose(nlx[0,:,:]) # dN/dx
calc_j11_j12_filter = torch.tensor(calc_j11_j12_filter, device=dev).unsqueeze(1) # (ngi, 1, nloc)
calc_j21_j22_filter = np.transpose(nlx[1,:,:]) # dN/dy
calc_j21_j22_filter = torch.tensor(calc_j21_j22_filter, device=dev).unsqueeze(1) # (ngi, 1, nloc)
# print(Det_nlx.calc_j11.weight.shape)
# print(nlx.shape)
# print(calc_j21_j22_filter.shape)
Det_nlx.calc_j11.weight.data = calc_j11_j12_filter
Det_nlx.calc_j12.weight.data = calc_j11_j12_filter
Det_nlx.calc_j21.weight.data = calc_j21_j22_filter
Det_nlx.calc_j22.weight.data = calc_j21_j22_filter
# print(Det_nlx.calc_j11.weight.shape)
# print(Det_nlx.calc_j11.weight.data)

#######################################################
# assemble local mass matrix and stiffness matrix
#######################################################


Mk = mk()
Mk.to(device=dev)
n = torch.transpose(torch.tensor(n, device=dev),0,1)


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
c = np.ones(nonods)
# c = np.asarray([1.000000000000000000e+00,2.000000000000000000e+00,1.493838904965597569e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.449966370388888537e+00,1.469033186016506010e+00,1.347141914183232148e+00,1.152750854595107066e+00,1.160140167446554349e+00,2.000000000000000000e+00,2.000000000000000000e+00,1.604796987540639153e+00,2.000000000000000000e+00,2.000000000000000000e+00,1.924636785854314303e+00,1.777579115307477586e+00,1.598073528854923264e+00,1.739355725613122461e+00,1.880373604087656636e+00,2.000000000000000000e+00,1.754399305660250041e+00,1.860698503999662634e+00,1.912555293209513341e+00,1.832992867485747412e+00,1.793603819785748232e+00,1.828777781554110415e+00,1.904292029262314001e+00,1.950644311037632583e+00,1.870696672554985618e+00,2.000000000000000000e+00,2.000000000000000000e+00,1.875466519341594918e+00,2.000000000000000000e+00,2.000000000000000000e+00,1.970084772125898498e+00,1.921416927406050412e+00,1.909446495780714681e+00,1.959869059419381498e+00,1.966051717905866525e+00,1.708788247347079237e+00,1.590406631659132985e+00,1.598421738447025842e+00,1.645239968676932341e+00,1.606731501002102158e+00,1.592745392397802329e+00,1.597020424780011716e+00,1.635550814925486929e+00,1.669762327596505669e+00,1.621135676584691820e+00,1.588358489223376413e+00,1.453113046928694585e+00,1.584281940384167919e+00,1.571851118994411767e+00,1.530738882663677636e+00,1.503649385313970965e+00,1.544286115996887920e+00,1.582328190551923974e+00,1.585873858125989688e+00,1.557301076290084296e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.224463585125032283e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.076461232701819482e+00,1.168821855317328406e+00,1.139537447109792545e+00,1.052369945083597269e+00,1.061829941916374143e+00,1.391892842440358713e+00,1.000000000000000000e+00,1.251255031108101878e+00,1.278956361726842017e+00,1.149832667989081125e+00,1.085427114030233176e+00,1.167440717592010957e+00,1.293160324816911677e+00,1.340071754548867178e+00,1.220546144132276023e+00,1.000000000000000000e+00,1.465468887236520246e+00,1.266761006261882372e+00,1.151271503687611775e+00,1.305470291792105275e+00,1.360533856633150629e+00,1.302941410124508259e+00,1.228825867503321900e+00,1.145876673913813315e+00,1.257219388930236326e+00,1.737056362497977124e+00,1.658859748710394433e+00,1.839779815233241633e+00,1.726003219039905501e+00,1.700235873718550117e+00,1.722120836381548115e+00,1.781083135021837682e+00,1.801759549431894536e+00,1.769540568893342769e+00,1.752270443868931826e+00,1.642824389936767826e+00,1.722402891657959900e+00,1.601318832181879204e+00,1.678888388057656833e+00,1.706738930968299872e+00,1.683622891114131548e+00,1.643092127937530122e+00,1.610131826480696304e+00,1.623806846994243491e+00,1.659712817768047133e+00,2.000000000000000000e+00,1.658795287023820730e+00,1.597301394522557372e+00,1.846920713034494455e+00,1.734544945483164735e+00,1.659963175674846303e+00,1.645024360694456789e+00,1.766236358058662415e+00,1.894559238560293846e+00,1.750396877319061684e+00,1.668209330932912682e+00,2.000000000000000000e+00,1.853874193600455245e+00,1.751994352722610504e+00,1.857143353739321601e+00,1.927765923022346950e+00,1.882183199279316987e+00,1.798558452053737211e+00,1.734623621035235930e+00,1.823910141926509132e+00,1.610450060843217734e+00,1.416911105440552099e+00,1.295550763684644391e+00,1.506742415673864466e+00,1.442391602695812791e+00,1.376821805580491720e+00,1.341504056475049333e+00,1.414273724536526444e+00,1.518900368834024350e+00,1.426227949409357487e+00,1.435834199812773937e+00,1.618397721589056815e+00,1.583313378420909023e+00,1.467038039402340344e+00,1.524616777838820658e+00,1.590381285395748634e+00,1.582518822908904665e+00,1.538800727920351941e+00,1.488600745986405682e+00,1.528243591423988024e+00,1.501419304636960828e+00,1.628227026803016120e+00,1.282718202701798704e+00,1.571271161698147845e+00,1.611625878276715662e+00,1.535405207842922559e+00,1.404060315890601762e+00,1.325333598289352599e+00,1.391144332285839891e+00,1.475470254153124605e+00])
# apply boundary conditions (two Dirichlet bcs)
for inod in bc1:
    c[inod]=1.
for inod in bc2:
    c[inod]=2.

tstep=int(np.ceil((tend-tstart)/dt))+1
c = c.reshape(nele,nloc) # reshape doesn't change memory allocation.
c = torch.tensor(c, dtype=torch.float64, device=dev).view(-1,1,nloc)
c_all=np.empty([tstep,nonods])
c_all[0,:]=c.view(-1).cpu().numpy()[:]

# surface integral 
[diagS, S] = S_Minv_sparse(x_all, nbele, nbf)

r1l2all=[]
# time loop
for itime in tqdm(range(1,tstep)):
    c_n = c.view(-1,1,nloc) # store last timestep value to cn
    c_i = c_n # jacobi iteration initial value taken as last time step value 

    r1l2=1
    its=0
    # for its in range(0,config.jac_its):
    while (r1l2>1e-13 and its<config.jac_its):
        c_i = c_i.view(-1,1,nloc)
        # print('its=',its,'c_i shape', c_i.shape)
        # calculate shape functions from element nodes coordinate
        # print(x_ref_in.shape)
        with torch.no_grad():
            nx, detwei = Det_nlx.forward(x_ref_in, weight)

        # mass matrix and rhs
        with torch.no_grad():
            [diagA,r1] = Mk.forward(c_i, c_n,k=1,dt=dt,n=n,nx=nx,detwei=detwei)

        r1 = r1.view(nonods,1) - torch.sparse.mm(S, c_i.view(nonods,1))
        # np.savetxt('c_i.txt',c_i.view(-1).cpu().numpy(), delimiter=',')
        # np.savetxt('r1.txt',r1.view(-1).cpu().numpy(), delimiter=',')
        diagA = diagA.view(nonods,1)+diagS.view(nonods,1)
        diagA = 1./diagA
        
        c_i = c_i.view(nonods,1) + config.jac_wei * torch.mul(diagA, r1)

        r1l2 = torch.linalg.norm(r1,dim=0)[0]
        its+=1
        r1l2all.append(r1l2)

        print('its=',its,'residual l2 norm=',r1l2.cpu().numpy())
        
    # if jacobi converges,
    c = c_i.view(nonods)

    # apply boundary conditions (two Dirichlet bcs)
    for inod in bc1:
        c[inod]=1.
    for inod in bc2:
        c[inod]=2.

    # combine inner/inter element contribution
    c_all[itime,:]=c.view(-1).cpu().numpy()[:]

np.savetxt('r0l2all.txt', np.asarray(r1l2all), delimiter=',')

#############################################################
# write output
#############################################################
# output 1: 
c_all = np.asarray(c_all)[::10,:]
np.savetxt('c_all.txt', c_all, delimiter=',')
np.savetxt('x_all.txt', x_all, delimiter=',')