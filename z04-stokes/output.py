import numpy as np

import config
# from config import sf_nd_nb
from function_space import FuncSpace


# nele = config.nele
# nonods = config.nonods
# nloc = config.nloc
ndim = config.ndim

VTK_LAGRANGE_TETRAHEDRON = 71


class File():
    """
    file handle to write vtu file
    for outputting and visualising

    c.f.
    https://people.math.sc.edu/Burkardt/data/vtu/vtu.html : ugridex.vtu sample file
    https://firedrakeproject.org/_modules/firedrake/output.html#File.write
    http://www.princeton.edu/~efeibush/viscourse/vtk.pdf
    """
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, "wb") as f:
            f.write(b'<?xml version="1.0" ?>\n')
            f.write(b'<VTKFile type="UnstructuredGrid" version="0.1" '
                    b'byte_order="BigEndian" '
                    b'header_type="UInt32">\n')
            f.write(b'<UnstructuredGrid>\n')
            # then we wait user to call write_vector or write_scaler
            # to write points coordinates and field values and xml file endings

    def write_vector(self, u, name, func_space: FuncSpace):
        # write a vector field (e.g. displacement)
        x_all = func_space.x_all
        nonods = func_space.nonods
        nloc = func_space.element.nloc
        perm = func_space.ref_node_order
        nele = func_space.nele

        u_np = u.view(nonods, ndim).cpu().numpy()
        typ = {np.dtype("float32"): "Float32",
               np.dtype("float64"): "Float64",
               np.dtype("int32"): "Int32",
               np.dtype("int64"): "Int64",
               np.dtype("uint8"): "UInt8"}[u_np.dtype]
        ncmp = 3  # vector has 3 components [MUST. if in 2D, append 0]
        with open(self.filename, "ab") as f:
            f.write(('<Piece NumberOfPoints="%d" '
                     'NumberOfCells="%d">\n' % (nonods, nele)).encode('ascii'))

            # write points coordinates
            f.write(b'<Points>\n')
            f.write(b'<DataArray type="Float64" NumberOfComponents="3" Format="ascii">\n')
            np.savetxt(f, x_all, delimiter=' ')
            f.write(b'</DataArray>\n'
                    b'</Points>\n')

            # write cell connectivity
            f.write(b'<Cells>\n'
                    b'<DataArray type="Int32" Name="connectivity" Format="ascii">\n')
            pndg_glbno = np.arange(0, nonods, dtype=np.int32)
            pndg_glbno = pndg_glbno.reshape((nele, nloc))
            # perm = sf_nd_nb.ref_node_order
            pndg_glbno = pndg_glbno[:, perm]
            np.savetxt(f, pndg_glbno, fmt='%d')
            f.write(b'</DataArray>\n'
                    b'<DataArray type="Int32" Name="offsets" Format="ascii">\n')
            offsets = np.arange(nloc, nonods + nloc, nloc, dtype=np.int32)
            np.savetxt(f, offsets, fmt='%d')
            f.write(b'</DataArray>\n'
                    b'<DataArray type="Int32" Name="types" Format="ascii">\n')
            np.savetxt(f, np.full(nele, VTK_LAGRANGE_TETRAHEDRON, dtype="uint8"), fmt='%d')
            f.write(b'</DataArray>\n'
                    b'</Cells>\n')
            f.write(('<PointData Vectors="%s">\n' % name).encode('ascii'))
            f.write(('<DataArray Name="%s" type="%s" '
                     'NumberOfComponents="%s" '
                     'format="ascii">\n' % (name, typ, ncmp)).encode('ascii'))
            np.savetxt(f, u_np, delimiter=' ')
            f.write(b'</DataArray>\n'
                    b'</PointData>\n'
                    b'</Piece>\n')
        return self.filename

    def write_end(self):
        with open(self.filename, "ab") as f:
            # finishing
            f.write(b'</UnstructuredGrid>\n'
                    b'</VTKFile>\n')
        return self.filename

    def write_scaler(self, p, name, func_space: FuncSpace):
        # write a scaler field (e.g. pressure)
        x_all = func_space.x_all
        nonods = func_space.nonods
        nloc = func_space.element.nloc
        perm = func_space.ref_node_order
        nele = func_space.nele

        p_np = p.view(nonods).cpu().numpy()
        typ = {np.dtype("float32"): "Float32",
               np.dtype("float64"): "Float64",
               np.dtype("int32"): "Int32",
               np.dtype("int64"): "Int64",
               np.dtype("uint8"): "UInt8"}[p_np.dtype]
        # ncmp = 3  # vector has 3 components [MUST. if in 2D, append 0]
        ncmp = 1
        with open(self.filename, "ab") as f:
            f.write(('<Piece NumberOfPoints="%d" '
                     'NumberOfCells="%d">\n' % (nonods, nele)).encode('ascii'))

            # write points coordinates
            f.write(b'<Points>\n')
            f.write(b'<DataArray type="Float64" NumberOfComponents="3" Format="ascii">\n')
            np.savetxt(f, x_all, delimiter=' ')
            f.write(b'</DataArray>\n'
                    b'</Points>\n')

            # write cell connectivity
            f.write(b'<Cells>\n'
                    b'<DataArray type="Int32" Name="connectivity" Format="ascii">\n')
            pndg_glbno = np.arange(0, nonods, dtype=np.int32)
            pndg_glbno = pndg_glbno.reshape((nele, nloc))
            # perm = sf_nd_nb.ref_node_order
            pndg_glbno = pndg_glbno[:, perm]
            np.savetxt(f, pndg_glbno, fmt='%d')
            f.write(b'</DataArray>\n'
                    b'<DataArray type="Int32" Name="offsets" Format="ascii">\n')
            offsets = np.arange(nloc, nonods + nloc, nloc, dtype=np.int32)
            np.savetxt(f, offsets, fmt='%d')
            f.write(b'</DataArray>\n'
                    b'<DataArray type="Int32" Name="types" Format="ascii">\n')
            np.savetxt(f, np.full(nele, VTK_LAGRANGE_TETRAHEDRON, dtype="uint8"), fmt='%d')
            f.write(b'</DataArray>\n'
                    b'</Cells>\n')
            f.write(('<PointData Scalers="%s">\n' % name).encode('ascii'))
            f.write(('<DataArray Name="%s" type="%s" '
                     'NumberOfComponents="%s" '
                     'format="ascii">\n' % (name, typ, ncmp)).encode('ascii'))
            np.savetxt(f, p_np, delimiter=' ')
            f.write(b'</DataArray>\n'
                    b'</PointData>\n'
                    b'</Piece>\n')
        return self.filename


# import collections
# import numpy as np
# import torch
# import config
# from config import sf_nd_nb
#
# from vtkmodules.vtkCommonDataModel import (
#     vtkLagrangeTriangle, vtkLagrangeTetra,
#     vtkLagrangeQuadrilateral, vtkLagrangeHexahedron, vtkLagrangeWedge
# )
#
#
# VTK_INTERVAL = 3
# VTK_TRIANGLE = 5
# VTK_QUADRILATERAL = 9
# VTK_TETRAHEDRON = 10
# VTK_HEXAHEDRON = 12
# VTK_WEDGE = 13
# #  Lagrange VTK cells:
# VTK_LAGRANGE_CURVE = 68
# VTK_LAGRANGE_TRIANGLE = 69
# VTK_LAGRANGE_QUADRILATERAL = 70
# VTK_LAGRANGE_TETRAHEDRON = 71
# VTK_LAGRANGE_HEXAHEDRON = 72
# VTK_LAGRANGE_WEDGE = 73
#
# OFunction = collections.namedtuple("OFunction", ["array", "name", "function"])
#
#
# def prepare_ofunction(ofunction, real):
#     array, name, _ = ofunction  # ofunction is a triple of (array, name:str, None)
#     if array.dtype.kind == "c":
#         if real:
#             arrays = (array.real, )
#             names = (name, )
#         else:
#             arrays = (array.real, array.imag)
#             names = (name + " (real part)", name + " (imaginary part)")
#     else:
#         arrays = (array, )
#         names = (name,)
#     return arrays, names
#
#
# def write_array_descriptor(f, ofunction, offset=None, parallel=False, real=True):
#     arrays, names = prepare_ofunction(ofunction, real)  # default is real array.
#     nbytes = 0
#     for array, name in zip(arrays, names):
#         shape = array.shape[1:]
#         ncmp = {0: "",
#                 1: "3",
#                 2: "9"}[len(shape)]
#         typ = {np.dtype("float32"): "Float32",
#                np.dtype("float64"): "Float64",
#                np.dtype("int32"): "Int32",
#                np.dtype("int64"): "Int64",
#                np.dtype("uint8"): "UInt8"}[array.dtype]
#         if parallel:
#             f.write(('<PDataArray Name="%s" type="%s" '
#                      'NumberOfComponents="%s" />' % (name, typ, ncmp)).encode('ascii'))
#         else:
#             if offset is None:
#                 raise ValueError("Must provide offset")
#             offset += nbytes
#             nbytes += (4 + array.nbytes)  # 4 is for the array size (uint32)
#             f.write(('<DataArray Name="%s" type="%s" '
#                      'NumberOfComponents="%s" '
#                      'format="appended" '
#                      'offset="%d" />\n' % (name, typ, ncmp, offset)).encode('ascii'))
#     return nbytes
#
#
# def get_byte_order(dtype):
#     import sys
#     native = {"little": "LittleEndian", "big": "BigEndian"}[sys.byteorder]
#     return {"=": native,
#             "|": "LittleEndian",
#             "<": "LittleEndian",
#             ">": "BigEndian"}[dtype.byteorder]
#
#
# def write_array(f, ofunction, real=False):
#     arrays, _ = prepare_ofunction(ofunction, real=real)
#     for array in arrays:
#         np.uint32(array.nbytes).tofile(f)
#         if get_byte_order(array.dtype) == "BigEndian":
#             array = array.byteswap()
#         array.tofile(f)
#
#
# def invert(list1, list2):
#     r"""Given two maps (lists) from [0..N] to nodes, finds a permutations between them.
#     :arg list1: a list of nodes.
#     :arg list2: a second list of nodes.
#     :returns: a list of integers, l, such that list1[x] = list2[l[x]]
#     """
#     if len(list1) != len(list2):
#         raise ValueError("Dimension of Paraview basis and Element basis unequal.")
#
#     def find_same(val, lst, tol=0.00000001):
#         for (idx, x) in enumerate(lst):
#             if np.linalg.norm(val - x) < tol:
#                 return idx
#         raise ValueError("Unable to establish permutation between Paraview basis and given element's basis.")
#     perm = [find_same(x, list2) for x in list1]
#     if len(set(perm)) != len(perm):
#         raise ValueError("Unable to establish permutation between Paraview basis and given element's basis.")
#     return perm
#
#
# def bary_to_cart(bar):
#     N = len(bar) - 1
#     mat = np.vstack([np.zeros(N), np.eye(N)])
#     return np.dot(bar, mat)
#
#
# def tet_barycentric_index(tet, index, order):
#     """
#     Wrapper for vtkLagrangeTetra::BarycentricIndex.
#     """
#     bindex = [-1, -1, -1, -1]
#     tet.BarycentricIndex(index, bindex, order)
#     return bary_to_cart(np.array(bindex) / order)
#
#
# def vtk_tet_local_to_cart(order):
#     r"""Produces a list of nodes for VTK's lagrange tet basis.
#     :arg order: the order of the tet
#     :return a list of arrays of floats
#     """
#     count = int((order + 1) * (order + 2) * (order + 3) // 6)
#     tet = vtkLagrangeTetra()
#     carts = [tet_barycentric_index(tet, i, order) for i in range(count)]
#     return carts
#
#
# def vtk_lagrange_tet_reorder(node_order):
#     degree = config.ele_p
#     vtk_local = vtk_tet_local_to_cart(degree)
#     my_local = node_order
#     return invert(vtk_local, my_local)
#
#
# class File(object):
#     """
#     output results to file
#     """
#     def __init__(self, filename):
#         self.filename = filename
#
#     def _write_single_vtu(self, basename, function_to_write):
#         # get connectivity
#         connectivity = config.ndglno.reshape(config.nele, config.nloc)  # PnDG local to global idx
#         perm = vtk_lagrange_tet_reorder(sf_nd_nb.ref_node_order)
#         connectivity = connectivity[:, perm]  # change node order to be the same as vtk lagrangian tetra order
#
#         coordinates = OFunction(array=sf_nd_nb.x_ref_in.view(config.nele * config.nloc, config.ndim).cpu().numpy(),
#                                 name='coordinates',
#                                 function=None)
#         num_points = config.nonods
#         num_cells = config.nele
#         fname = self.filename
#         with open(fname, "wb") as f:
#             # Running offset for appended data
#             offset = 0
#             f.write(b'<?xml version="1.0" ?>\n')
#             f.write(b'<VTKFile type="UnstructuredGrid" version="0.1" '
#                     b'byte_order="LittleEndian" '
#                     b'header_type="UInt32">\n')
#             f.write(b'<UnstructuredGrid>\n')
#
#             f.write(('<Piece NumberOfPoints="%d" '
#                      'NumberOfCells="%d">\n' % (num_points, num_cells)).encode('ascii'))
#             f.write(b'<Points>\n')
#             # Vertex coordinates
#             offset += write_array_descriptor(f, coordinates, offset=offset, real=True)
#             f.write(b'</Points>\n')
#
#             f.write(b'<Cells>\n')
#             offset += write_array_descriptor(f, connectivity, offset=offset)
#             offset += write_array_descriptor(f, offsets, offset=offset)
#             cell_types = np.full(num_cells, VTK_LAGRANGE_TETRAHEDRON, dtype="uint8")
#             types = OFunction(array=cell_types,
#                               name='types',
#                               function=None)
#             offset += write_array_descriptor(f, types, offset=offset)
#             f.write(b'</Cells>\n')
