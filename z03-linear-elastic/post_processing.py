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

x_data = pd.read_csv('./x_all.txt', header=None)
x_all = x_data.to_numpy()
print(x_all.shape)

# c = np.arange(1,11)
c_data = pd.read_csv('./c_all.txt', header=None)
c = c_data.to_numpy()


ntime = c.shape[0]
nloc = 10
nele = int(x_all.shape[0]/nloc )
nonods = x_all.shape[0]

# fetch final result
# c_final = c[1,:]-c_data_twogrid[1,:]
# calculate analytical soln
c_ana = np.zeros(nonods)
for inod in range(nonods):
    xi = x_all[inod,0]
    yi = x_all[inod,1]
    c_ana[inod] = np.sin(np.pi*xi) * np.sinh(np.pi*yi) / np.sinh(np.pi)
if (False): # ploting error rather than value
    c_error = c[1,:] - c_ana
    c[1,:] = c_error 
# print(c_error.max(), c_error.min(), np.linalg.norm(c_error))

c_max = c[1,:].max() 
c_min = c[1,:].min()
print(c_max,c_min)
print(c.shape)

triangles = np.asarray([
    [1,4,9],    [9,4,10],   [4,5,10], 
    [10,5,6],   [5,2,6],    [9,10,8],
    [8,10,7],   [7,10,6],   [8,7,3]
])-1
levels = np.linspace(c_min,c_max,21)
# print(np.max(c))
print(levels)

for itime in tqdm(range(0,ntime,1)):
    fig1, ax1 = plt.subplots()
    for ele in range(nele):
            
        current_ele_idx = np.arange(ele*nloc,ele*nloc+nloc) 
        x = x_all[current_ele_idx, 0]
        y = x_all[current_ele_idx, 1]
        z = c[itime,current_ele_idx]
        # z = c[current_ele_idx]
        # print(z)

        ax1.set_aspect('equal')
        tcf = ax1.tricontourf(x, y, z,levels=levels )#, levels=levels  )
    fig1.colorbar(tcf)
    # ax1.set_title('t = {0:.2f}'.format((itime)*1e-2))
    ax1.set_title('%d elements approx'%(nele))
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    
    # for ele in tqdm(range(nele)):
            
    #     current_ele_idx = np.arange(ele*nloc,ele*nloc+nloc) 
    #     x = x_all[current_ele_idx, 0]
    #     y = x_all[current_ele_idx, 1]
    #     # z = c[itime,current_ele_idx]
    #     z = c_ana[current_ele_idx]
    #     # print(z)

    #     ax2.set_aspect('equal')
    #     tcf = ax2.tricontourf(x, y, z,levels=levels )#, levels=levels  )
    # # fig2.colorbar(tcf)
    # # ax1.set_title('t = {0:.2f}'.format((itime)*1e-2))
    # ax2.set_title('analytical')
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')

    plt.savefig('diffusion{:04}.png'.format(itime))
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