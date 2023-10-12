"""
output FSI results to vtu
"""
import numpy as np
import torch

import config
import volume_mf_st
import output


def output_fsi_vtu(x_i, vel_func_space, pre_func_space, disp_func_space, itime):
    """
    output fsi results to vtu file,

    input
    -----
    x_i: dict
        the solution of the fsi system
    vel_func_space:
        velocity FunctionSpace
    pre_func_space:
        pressure FunctionSpace
    disp_func_space:
        displacement FunctionSpace
    itime: int
        current time step
    """
    x_i_dict = volume_mf_st.slicing_x_i(x_i)

    # write fluid velocity and pressure
    vtk = output.File(config.filename + config.case_name + '_f%d.vtu' % itime)
    vtk.write_head(vel_func_space)
    # write fluid/solid subdomain indicator
    idct = np.zeros(config.nele, dtype=np.int32)
    idct[0:config.nele_f] = 1  # fluid: 1, solid: 0
    vtk.write_cell_data(idct, 'subdomain')
    # write velocity
    vtk.write_vector(x_i_dict['vel'], 'velocity', vel_func_space)
    # write pressure
    # first project pressure to velocity space
    prolongator = pre_func_space.get_pndg_prolongator(
        p=pre_func_space.element.ele_order,
        gi_order=pre_func_space.element.gi_order,
        ndim=pre_func_space.element.ndim,
        dev=config.dev,
    )
    pressure_on_vel_space = torch.einsum(
        'mn,bn->bm', prolongator, x_i_dict['pre']
    )
    vtk.write_scalar(pressure_on_vel_space, 'pressure', vel_func_space)
    vtk.write_end()

    # write solid displacement (and mesh displacement)
    vtk = output.File(config.filename + config.case_name + '_s%d.vtu' % itime)
    vtk.write_head(disp_func_space)
    # write fluid/solid subdomain indicator
    vtk.write_cell_data(idct, 'subdomain')
    # write displacement
    vtk.write_vector(x_i_dict['disp'], 'displacement', disp_func_space)
    vtk.write_end()
