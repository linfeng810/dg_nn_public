#!/usr/bin/env python3
import sys
sys.path.append('../../')
import numpy as np
import torch
import config
import fsi_output
import sparsity
import volume_mf_st
from function_space import FuncSpace, Element
from config import sf_nd_nb
import time
import materials

starttime = time.time()

# for pretty print out torch tensor
# torch.set_printoptions(sci_mode=False)
torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)

dev = config.dev
nele = config.nele
if config.isFSI:
    nele_f = config.nele_f
    nele_s = config.nele_s
ndim = config.ndim
dt = config.dt
tend = config.tend
tstart = config.tstart

print('computation on ',dev)

# define element
quad_degree = config.ele_p*3
vel_ele = Element(ele_order=config.ele_p, gi_order=quad_degree, edim=ndim, dev=dev)
pre_ele = Element(ele_order=config.ele_p_pressure, gi_order=quad_degree, edim=ndim,dev=dev)
print('ele pair: ', vel_ele.ele_order, pre_ele.ele_order, 'quadrature degree: ', quad_degree)

vel_func_space = FuncSpace(vel_ele, name="Velocity", mesh=config.mesh, dev=dev)
pre_func_space = FuncSpace(pre_ele, name="Pressure", mesh=config.mesh, dev=dev,
                           not_iso_parametric=True, x_element=vel_ele)  # super-parametric pressure ele.
sf_nd_nb.set_data(vel_func_space=vel_func_space,
                  pre_func_space=pre_func_space,
                  p1cg_nonods=vel_func_space.cg_nonods)
if config.isFSI:
    disp_func_space = FuncSpace(vel_ele, name="Displacement", mesh=config.mesh, dev=dev,
                                get_pndg_ndglbno=True)  # displacement func space
    sf_nd_nb.set_data(disp_func_space=disp_func_space)

material = materials.NeoHookean(sf_nd_nb.disp_func_space.element.nloc,
                                ndim, dev, config.mu, config.lam)
# material = materials.LinearElastic(nloc, ndim, dev, mu, lam)
sf_nd_nb.set_data(material=material)

fluid_spar, solid_spar = sparsity.get_subdomain_sparsity(
    vel_func_space.cg_ndglno,
    config.nele_f,
    config.nele_s,
    vel_func_space.cg_nonods
)
sf_nd_nb.set_data(sparse_f=fluid_spar, sparse_s=solid_spar)


x_i = torch.zeros(vel_func_space.nonods * ndim * 2 + pre_func_space.nonods, device=dev, dtype=torch.float64)
x_i_dict = volume_mf_st.slicing_x_i(x_i)
x_i_dict['vel'] += (torch.sin(vel_func_space.x_ref_in[:, 0, :]) *
                    torch.sin(vel_func_space.x_ref_in[:, 1, :])).unsqueeze(dim=-1).expand(-1, -1, ndim)
x_i_dict['pre'] += torch.tensor(
    np.sin(pre_func_space.x_all[:, 0]) * np.sin(pre_func_space.x_all[:, 1]),
    device=dev, dtype=torch.float64
).view(nele, -1)
x_i_dict['disp'] += (torch.exp(disp_func_space.x_ref_in[:, 0, :]) *
                     torch.exp(disp_func_space.x_ref_in[:, 1, :])).unsqueeze(dim=-1).expand(-1, -1, ndim)

fsi_output.output_fsi_vtu(x_i, vel_func_space, pre_func_space, disp_func_space, 0)
