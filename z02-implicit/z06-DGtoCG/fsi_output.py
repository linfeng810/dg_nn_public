"""
output FSI results to vtu
"""
import numpy as np
import config
import output


def output_diff_vtu(x_i, scalar_func_space, itime):
    """output diffusion results to vtu file"""
    # write to vtk
    vtk = output.File(config.filename + config.case_name + '_%d.vtu' % itime)
    vtk.write_head(scalar_func_space)
    vtk.write_cell_data(data=np.zeros(config.nele, dtype=np.int32), name='cell_mark')
    vtk.write_scalar(x_i, name='c_i', func_space=scalar_func_space)
    vtk.write_end()
