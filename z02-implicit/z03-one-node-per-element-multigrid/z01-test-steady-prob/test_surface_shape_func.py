#!/usr/bin/env python3

import numpy as np 
import shape_function 
import config 
import torch

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
torch.set_printoptions(precision=16)

'''
output: 

n, shape functions on a reference element at quadrature points, 
        numpy array (nloc, ngi)

nlx, shape function deriatives on a reference element at quad pnts,
        numpy array (ndim, nloc, ngi)

weight, quad pnts weights, np array (ngi)

sn, surface shape functions on a reference element, numpy array (nface,nloc,sngi)

snlx_all, shape function derivatives on surface, on a reference element, 
        numpy array (nface, ndim, nloc, sngi)

sweight, quad pnts weights on face, np array (sngi)
'''
[n,nlx,weight,sn,snlx,sweight] = shape_function.SHATRInew(config.nloc, \
    config.ngi, config.ndim)



#####################################
# test two elements input
#
x_ref_in1 = np.asarray([ 1., 0., \
            0., 1., \
            0., 0., \
            2./3., 1./3., \
            1./3., 2./3., \
            0., 2./3., \
            0., 1./3., \
            1./3., 0., \
            2./3., 0., \
            1./3., 1./3.])
x_ref_in1 = x_ref_in1.reshape((nloc,ndim))
# x_ref_in2 = np.asarray([ 2.0, 0.0, \
#             0.0, 1.0, \
#             0.0, 0.0, \
#             4./3., 1./3., \
#             2./3., 2./3., \
#             0., 2./3., \
#             0., 1./3., \
#             2./3., 0., \
#             4./3., 0., \
#             2./3., 1./3.])
rota_mat = np.asarray([[np.cos(np.pi/2), np.sin(np.pi/2)],\
    [-np.sin(np.pi/2), np.cos(np.pi/2)]]).transpose()
# print(rota_mat)
x_ref_in2 = np.dot(rota_mat, np.transpose(x_ref_in1)).transpose()
x_ref_in2 = x_ref_in2.reshape((nloc,ndim))

x_ref_in1 = np.transpose(x_ref_in1)
x_ref_in2 = np.transpose(x_ref_in2)
x_ref_in = np.stack((x_ref_in1,x_ref_in2), axis=0)
x_ref_in = torch.tensor(x_ref_in, requires_grad=False, device=dev).view(2,2,nloc)

print('xin size', x_ref_in.shape)
# print('xin type', x_ref_in)

# output :
# snx: (nele, nface, ndim, nloc, sngi)
# sdetwei: (nele, nface, sgni)
[snx, sdetwei, snormal] = shape_function.sdet_snlx(snlx, x_ref_in, sweight)

# print('snx, ', snx)
# print('sdetwei, ', sdetwei)
print(snormal)

# integrate over faces
# let's do integration on 3 faces
# first integration of sn
nface = 3
int_sn_ds = np.zeros([3,10])
for iface in range(nface): 
    for iloc in range(config.nloc) : 
        int_sn_ds[iface,iloc] = np.dot(sn[iface,iloc,:], sdetwei[0,iface,:].cpu().numpy())
        # print('iface %d, iloc %d, int_sn_ds %f'%(iface,iloc,int_sn_ds))
        int_snx_ds = np.dot(snlx[iface,0,iloc,:],sweight)
        int_sny_ds = np.dot(snlx[iface,1,iloc,:],sweight)
        # print('iface %d, iloc %d, intsnx %f, intsny %f'%(iface,iloc,int_snx_ds, int_sny_ds))
np.savetxt('int_sn_ds.txt', int_sn_ds, delimiter=',')
###### int_f N ds pass! ########
# face=1
intNx_f=torch.zeros(2,nface,nloc,ndim, device=dev, dtype=torch.float64)
for ele in range(2):
    for inod in range(10):
        for iface in range(nface):
            for idim in range(ndim):
                intNx_f[ele,iface,inod,idim] = \
                    torch.dot(snx[ele,iface,idim,inod,:], sdetwei[ele,iface,:])
print('intNx_f', intNx_f)
np.savetxt('intNx_f1.txt', intNx_f[0,0,:,:].cpu().numpy(), delimiter=',')
np.savetxt('intNx_f2.txt', intNx_f[0,1,:,:].cpu().numpy(), delimiter=',')
np.savetxt('intNx_f3.txt', intNx_f[0,2,:,:].cpu().numpy(), delimiter=',')
###### int_f Nx ds pass #########

# int_f Njx Ni ds
intNxN_f = torch.zeros(2,nface,nloc,nloc, device=dev, dtype=torch.float64)
for ele in range(2):
    for iface in range(3):
        for inod in range(10):
            for jnod in range(10):
                for idim in range(ndim):
                    for sgi in range(config.sngi):
                        intNxN_f[ele,iface,inod,jnod] += snx[ele,iface,idim,jnod,sgi]*sn[iface,inod,sgi]*snormal[ele,iface,idim]*sdetwei[ele,iface,sgi]
np.savetxt('intNxN_f1.txt', intNxN_f[1,0,:,:].cpu().numpy(), delimiter=',')
np.savetxt('intNxN_f2.txt', intNxN_f[1,1,:,:].cpu().numpy(), delimiter=',')
np.savetxt('intNxN_f3.txt', intNxN_f[1,2,:,:].cpu().numpy(), delimiter=',')
###### int_f NxN ds pass #########

# how do we test the rotated "unit" triangle?
# let's use the rotation matrix to rotate intNx_f[:,iface=0,:,:]
ele1_intNx_f = intNx_f[0,:,:,:].cpu().numpy()
rotated_intNx = np.einsum('ij,klj->kli', rota_mat, ele1_intNx_f)
print(rotated_intNx)
# print(intNx_f[1,:,:,:,:])

# I would say we passed the test. Yay!
# Now we are good to do the classic interior penalty method!
ele=1
iface=0
print(3*torch.sum(sdetwei[ele,iface,:]).cpu().numpy())

sdetwei = sdetwei.cpu().numpy()
print(np.sum(sdetwei[ele,iface,:]))