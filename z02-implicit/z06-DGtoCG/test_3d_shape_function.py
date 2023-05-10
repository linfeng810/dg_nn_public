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
snloc = config.snloc
sngi = config.sngi
dt = config.dt
tend = config.tend
tstart = config.tstart
torch.set_printoptions(precision=16)

n, nlx, weight, sn, snlx, sweight = shape_function.SHATRInew(
    nloc=nloc, ngi=ngi, ndim=ndim, snloc=snloc, sngi=sngi
)
n = torch.tensor(n, device=dev)
nlx = torch.tensor(nlx, device=dev)
weight = torch.tensor(weight, device=dev)
sn = torch.tensor(sn, device=dev)
snlx = torch.tensor(snlx, device=dev)
sweight = torch.tensor(sweight, device=dev)
# # mass matrix
# nn = np.zeros((nloc, nloc))
# for inod in range(nloc):
#     for jnod in range(nloc):
#         nn[inod, jnod] = np.sum(n[inod, :] * n[jnod, :] * weight)
# np.savetxt('nn.txt', nn, delimiter=',')
#
# # stiffness matrix
# nxnx = np.zeros((nloc, nloc, ndim, ndim))
# for inod in range(nloc):
#     for jnod in range(nloc):
#         for idim in range(ndim):
#             for jdim in range(ndim):
#                 nxnx[inod, jnod, idim, jdim] = np.sum(nlx[idim, inod, :] * nlx[jdim, jnod, :] * weight)
# for idim in range(ndim):
#     for jdim in range(ndim):
#         np.savetxt('nxnx'+str(idim)+str(jdim)+'.txt', nxnx[:,:,idim,jdim], delimiter=',')

# face integral
# print(sn.shape, snlx.shape, sweight.shape)
if nloc == 20:  # cubic element
    x_ref_in = [
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0, 0, 0,
        2/3, 0, 1/3,
        1/3, 0, 2/3,
        2/3, 1/3, 0,
        1/3, 2/3, 0,
        0, 2/3, 1/3,
        0, 1/3, 2/3,
        2/3, 0, 0,
        1/3, 0, 0,
        0, 2/3, 0,
        0, 1/3, 0,
        0, 0, 2/3,
        0, 0, 1/3,
        0, 1/3, 1/3,
        1/3, 1/3, 1/3,
        1/3, 0, 1/3,
        1/3, 1/3, 0
    ]
    x_ref_in0 = torch.tensor(x_ref_in, device=dev,
                            dtype=torch.float64).reshape(-1,3).transpose(0,1).view(1,3,20)
    x_ref_in1 = x_ref_in0.detach().clone()
    x_ref_in1[:,1,:] *= 0.5

    alpha = np.pi/2
    beta = np.pi/2*0
    gamma = np.pi/2*0
    r_z = torch.tensor([np.cos(gamma), -np.sin(gamma), 0,
                       np.sin(gamma), np.cos(gamma),  0,
                       0, 0, 1], device=dev).reshape((3,3))
    r_y = torch.tensor([np.cos(beta), 0, np.sin(beta),
                       0, 1, 0,
                       -np.sin(beta), 0, np.cos(beta)], device=dev).reshape((3,3))
    r_x = torch.tensor([1, 0, 0,
                       0, np.cos(alpha), -np.sin(alpha),
                       0, np.sin(alpha), np.cos(alpha)], device=dev).reshape((3,3))
    r_mat = r_z @ r_y @ r_x
    x_ref_in2 = torch.matmul(r_mat, x_ref_in0)

    x_ref_in3 = torch.tensor(
        [0, 0, 0,
         0, 1, 0,
         1, 0, 0,
         0, 0, 1,
         1./3., 0, 0,
         2./3., 0, 0,
         0, 1./3., 0,
         0, 2./3., 0,
         1./3., 2./3., 0,
         2./3., 1./3., 0,
         0, 0, 1./3.,
         0, 0, 2./3.,
         0, 2./3., 1./3.,
         0, 1./3., 2./3.,
         2./3., 0, 1./3.,
         1./3., 0, 2./3.,
         1./3., 1./3., 1./3.,
         1./3., 1./3., 0,
         1./3., 0, 1./3.,
         0, 1./3., 1./3.
         ], device=dev, dtype=torch.float64
    ).reshape(-1,3).transpose(0,1).view(1,3,20)
    x_ref_in = torch.vstack((x_ref_in0, x_ref_in1, x_ref_in2, x_ref_in3))
    nx, detwei = shape_function.get_det_nlx_3d(nlx, x_ref_in, weight)
elif nloc == 4:  # linear element
    x_ref_in = [
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0, 0, 0,
    ]
    x_ref_in0 = torch.tensor(x_ref_in, device=dev,
                             dtype=torch.float64).reshape(-1, 3).transpose(0, 1).view(1, 3, 4)
    x_ref_in1 = x_ref_in0.detach().clone()
    x_ref_in1[:, 1, :] *= 0.5

    alpha = np.pi / 2
    beta = np.pi / 2 * 0
    gamma = np.pi / 2 * 0
    r_z = torch.tensor([np.cos(gamma), -np.sin(gamma), 0,
                        np.sin(gamma), np.cos(gamma), 0,
                        0, 0, 1], device=dev).reshape((3, 3))
    r_y = torch.tensor([np.cos(beta), 0, np.sin(beta),
                        0, 1, 0,
                        -np.sin(beta), 0, np.cos(beta)], device=dev).reshape((3, 3))
    r_x = torch.tensor([1, 0, 0,
                        0, np.cos(alpha), -np.sin(alpha),
                        0, np.sin(alpha), np.cos(alpha)], device=dev).reshape((3, 3))
    r_mat = r_z @ r_y @ r_x
    x_ref_in2 = torch.matmul(r_mat, x_ref_in0)

    x_ref_in3 = torch.tensor(
        [0, 0, 0,
         0, 1, 0,
         1, 0, 0,
         0, 0, 1,
         ], device=dev, dtype=torch.float64
    ).reshape(-1, 3).transpose(0, 1).view(1, 3, 4)
    x_ref_in4 = torch.tensor(
        [0, 0, 0,
         0, 1, 0,
         0, 0, 1,
         -2, 0, 0,
         ], device=dev, dtype=torch.float64
    ).reshape(-1, 3).transpose(0, 1).view(1, 3, 4)
    x_ref_in = torch.vstack((x_ref_in0, x_ref_in1, x_ref_in2, x_ref_in3, x_ref_in4))
    nx, detwei = shape_function.get_det_nlx_3d(nlx, x_ref_in, weight)
else:
    raise Exception('nloc %d element not implemented!' % nloc)


# mass matrix
nn = torch.zeros((x_ref_in.shape[0], nloc, nloc), device=dev, dtype=torch.float64)
for inod in range(nloc):
    for jnod in range(nloc):
        nn[:, inod, jnod] = torch.sum(n[inod, :] * n[jnod, :] * detwei[:, :], -1)

# stiffness matrix
nxnx = torch.einsum('...ilg,...img,...g->...lm', nx, nx, detwei)

# === face shape functions ===
snlx, sdetwei, snormal = shape_function.sdet_snlx_3d(snlx, x_ref_in, sweight)
snx_int = torch.einsum('...fdng,...fg->...fdn', snlx, sdetwei)
print(snx_int)

