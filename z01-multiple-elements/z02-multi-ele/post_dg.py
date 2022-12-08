#!/usr/bin/env python3

# This is to plot 2d dg result
# Cubic element#

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# input: 
# c - field variable 1d array (nonods)
# x_all - node coordinate, 2d array (nonods, ndim)

# x_all = np.empty([160,2])
# x_all = np.asarray([ 1.0, 0.0, \
#             0.0, 1.0, \
#             0.0, 0.0, \
#             2./3., 1./3., \
#             1./3., 2./3., \
#             0., 2./3., \
#             0., 1./3., \
#             1./3., 0., \
#             2./3., 0., \
#             1./3., 1./3.])
# x_all = np.reshape(x_all,(10,2))
x_data = pd.read_csv('/data2/linfeng/aicfd/z01-chrisemail/x_all.txt', header=None)
x_all = x_data.to_numpy()
print(x_all.shape)

# c = np.arange(1,11)
c_data = pd.read_csv('/data2/linfeng/aicfd/z01-chrisemail/c_all.txt', header=None)
c = c_data.to_numpy()
# print(c.shape)
ntime = c.shape[0]
nloc = 10
nele = int(x_all.shape[0]/nloc )
triangles = np.asarray([
    [1,4,9],    [9,4,10],   [4,5,10], 
    [10,5,6],   [5,2,6],    [9,10,8],
    [8,10,7],   [7,10,6],   [8,7,3]
])-1
levels = np.linspace(0,2,21)
# print(np.max(c))
# print(levels)
for itime in tqdm(range(100,ntime,10)):
    fig4, ax4 = plt.subplots()
    for ele in range(nele):
            
        current_ele_idx = np.arange(ele*nloc,ele*nloc+nloc) 
        x = x_all[current_ele_idx, 0]
        y = x_all[current_ele_idx, 1]
        z = c[itime,current_ele_idx]
        # print(z)

        ax4.set_aspect('equal')
        tcf = ax4.tricontourf(x, y, z,levels=levels )#, levels=levels  )
    fig4.colorbar(tcf)
    ax4.set_title('t = {0:.2f}'.format((itime)/100))
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')

    plt.savefig('z01-multiple-elements/z02-multi-ele/diffusion{:04}.png'.format(itime))
    plt.close()
plt.show()

# x = x_all[:, 0]
# y = x_all[:, 1]
# z = c[-1,:]
# fig4, ax4 = plt.subplots()
# ax4.set_aspect('equal')
# tcf = ax4.tricontourf(x, y, z,levels=levels )#, levels=levels  )
# fig4.colorbar(tcf)
# ax4.set_title('t = {0:.2f}'.format((10000)/100))
# ax4.set_xlabel('x')
# ax4.set_ylabel('y')
# plt.show()