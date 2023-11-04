# this is a tool to debug code.
# that takes in a field on quadrature points and output a csv file consists the following columns:
# coord_x, coord_y, coord_z, field_value
import torch
import numpy
from shape_function import _gi_pnts_tri, _gi_pnts_tetra
from config import dev


def write_gi_data_to_csv(field_val, ngi, func_space, fname):
    """
    this is a tool to debug code.
    that takes in a vector field on quadrature points and output a csv file consists the following columns:
    coord_x, coord_y, coord_z, field_value

    input:
    field_val: field values on quadrature points, should be of shape (nele, ngi)
    ngi: number of gi points
    func_space: on which function space is this field defined (we mainly use this to get element nodes info)
    """
    ndim = func_space.element.ndim
    if ndim == 3:
        raise ValueError('output gaussian points field value in 3D not implemented!')
    else:  # ndim == 2:
        L, weight, _ = _gi_pnts_tri(ngi)  # gaussian points on reference triangle element
        L = torch.tensor(L, device=dev)
        # L shape is (ngi, 3) 3 barycentric coordinates
        # get gi points coordinates
        nodal_pnts = func_space.x_ref_in[:, :, 0:3]  # 3 corner nodes coordinates of every element (nele, ndim, 3)
        gi_pnts = torch.einsum('bi,g->big', nodal_pnts[:, :, 0], L[:, 0]) \
            + torch.einsum('bi,g->big', nodal_pnts[:, :, 1], L[:, 1]) \
            + torch.einsum('bi,g->big', nodal_pnts[:, :, 2], L[:, 2])  # (nele, ndim, ngi)
        gi_pnts = gi_pnts.cpu().numpy()
        field_val_np = field_val.view(gi_pnts.shape).cpu().numpy()
        # open a file and write
        with open(fname, "w") as f:
            f.write('x,y,z,tau_x,tau_y\n')
            for ele in range(gi_pnts.shape[0]):
                for gi in range(gi_pnts.shape[2]):
                    f.write('%f,%f,%f,%f,%f\n' % (
                        gi_pnts[ele, 0, gi],
                        gi_pnts[ele, 1, gi],
                        0,  # z=0
                        field_val_np[ele, 0, gi],  # first component of vector field
                        field_val_np[ele, 1, gi],  # first component of vector field
                    ))
