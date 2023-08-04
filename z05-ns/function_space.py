import torch
from mesh_init import init_2d, init_3d
from shape_function import SHATRInew
# import config


# dev = config.dev


class Element(object):
    """
    this is a class that log element information.
    nloc, ngi, snloc, sngi,
    n, nlx, weight, sn, snlx, sweight at reference element
    ele_order
    """
    def __init__(self, ele_order: int, gi_order: int, edim: int, dev):
        self.ele_order = ele_order
        self.gi_order = gi_order
        self.ndim = edim
        # assuming simplex element (triangle or tetrahedron)
        if self.ndim == 3:
            self.nloc = int(1/6*(ele_order+1)*(ele_order+2)*(ele_order+3))
            self.snloc = int(1/2*(ele_order+2)*(ele_order+1))
            if self.gi_order == 6:
                self.ngi = 24
                # self.sngi = 9  # STOP using this. this is only 5 order precise
                self.sngi = 12  # this is 6 order.
            elif self.gi_order == 4:
                self.ngi = 11
                self.sngi = 6
            elif self.gi_order == 2:
                self.ngi = 4
                self.sngi = 3
            else:
                raise ValueError("the chosen gaussian integration order %d isn't accepted." % gi_order)
        elif self.ndim == 2:
            self.nloc = int(1/2*(ele_order+2)*(ele_order+1))
            self.snloc = ele_order+1
            if self.gi_order == 6:
                self.ngi = 12
                self.sngi = 4
            elif self.gi_order == 4:
                self.ngi = 6
                self.sngi = 3
            elif self.gi_order == 2:
                self.ngi = 3
                self.sngi = 2
            elif self.gi_order == 9:
                self.ngi = 19
                self.sngi = 5
            else:
                raise ValueError("the chosen gaussian integration order %d isn't accepted." % gi_order)
        else:
            raise ValueError("only support 2D or 3D elements.")
        # now we find shape functions on the reference element
        self.n, self.nlx, self.weight, \
            self.sn, self.snlx, self.sweight, \
            self.gi_align = \
            SHATRInew(self.nloc, self.ngi, self.ndim, self.snloc, self.sngi)
        # converse to torch tensor
        self.n = torch.tensor(self.n, device=dev, dtype=torch.float64)
        self.nlx = torch.tensor(self.nlx, device=dev, dtype=torch.float64)
        self.weight = torch.tensor(self.weight, device=dev, dtype=torch.float64)
        self.sn = torch.tensor(self.sn, device=dev, dtype=torch.float64)
        self.snlx = torch.tensor(self.snlx, device=dev, dtype=torch.float64)
        self.sweight = torch.tensor(self.sweight, device=dev, dtype=torch.float64)


class FuncSpace(object):
    """
    this is a function space object that stores:
    DOFs (for lagrangian elements, this is nodes): nods coordintes
        x_all (nonods, ndim) is numpy array
        x_ref_in (nele, ndim, nloc) is torch tensor
    nonods
    nbf, nbele, alnmt: element neighbouring info. *this seems to be the same for all mesh
    fina, cola, ncola: element connectivity. this seems useless!
    bc: boundary nodes, is a list of n numpy array, each array records nods on that boundary
    cg_ndglno
    cg_nonods
    """
    def __init__(self, element: Element, mesh, dev, name: str = ""):
        self.name = name
        print('initalising '+name+' function space')
        self.mesh = mesh
        self.nele = mesh.cell_data['gmsh:geometrical'][-1].shape[0]
        self.nonods = element.nloc * self.nele
        self.element = element
        self.ndim = self.element.ndim
        self.p1cg_nloc = self.ndim + 1
        self.p1dg_nonods = self.p1cg_nloc * self.nele
        if self.ndim == 2:
            self.x_all, \
                self.nbf, self.nbele, self.alnmt, self.glb_bcface_type, \
                self.fina, self.cola, self.ncola, \
                self.bc_node_list, self.cg_ndglno, self.cg_nonods, \
                self.ref_node_order, self.prolongator_from_p1dg = \
                init_2d(self.mesh, self.nele, self.nonods, self.element.nloc, nface=self.ndim + 1)
            self.restrictor_to_p1dg = torch.transpose(self.prolongator_from_p1dg, dim0=0, dim1=1)
        elif self.ndim == 3:
            self.x_all, \
                self.nbf, self.nbele, self.alnmt, self.glb_bcface_type, \
                self.fina, self.cola, self.ncola, \
                self.bc_node_list, self.cg_ndglno, self.cg_nonods, \
                self.ref_node_order, self.prolongator_from_p1dg = \
                init_3d(self.mesh, self.nele, self.nonods, self.element.nloc, nface=self.ndim+1)
            self.restrictor_to_p1dg = torch.transpose(self.prolongator_from_p1dg, dim0=0, dim1=1)
        # converse to torch.tensor
        self.x_ref_in = torch.tensor(
            self.x_all.reshape((self.nele, self.element.nloc, self.element.ndim)).transpose((0, 2, 1)),
            device=dev, dtype=torch.float64
        )
        self.nbele = torch.tensor(self.nbele, device=dev)
        self.nbf = torch.tensor(self.nbf, device=dev)
        self.alnmt = torch.tensor(self.alnmt, device=dev)
        self.glb_bcface_type = torch.tensor(self.glb_bcface_type, device=dev)
