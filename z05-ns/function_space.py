import torch

import sparsity
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
            elif self.gi_order == 9:
                self.ngi = 57
                self.sngi = 19
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
    def __init__(self, element: Element, mesh, dev, name: str = "",
                 not_iso_parametric=False, x_element=None,
                 get_pndg_ndglbno=False):
        """
        if we're using superparametric element, we should provice
        the element for geometry as well (might be higher order)
        """
        self.name = name
        print('initalising '+name+' function space')
        self.mesh = mesh
        self.element = element
        self.ndim = self.element.ndim
        self.p1cg_nloc = self.ndim + 1
        if self.ndim == 2:
            self.nele = mesh.cell_data_dict['gmsh:geometrical']['triangle'].shape[0]
            self.nonods = element.nloc * self.nele
            self.p1dg_nonods = self.p1cg_nloc * self.nele
            self.x_all, \
                self.nbf, self.nbele, self.alnmt, self.glb_bcface_type, \
                self.fina, self.cola, self.ncola, \
                self.bc_node_list, self.cg_ndglno, self.cg_nonods, \
                self.ref_node_order, self.prolongator_from_p1dg = \
                init_2d(self.mesh, self.nele, self.nonods, self.element.nloc, nface=self.ndim + 1)
            self.restrictor_to_p1dg = torch.transpose(self.prolongator_from_p1dg, dim0=0, dim1=1)
            # if not iso-paramatric, we store geometry DOFs in x_ref_in
            if not_iso_parametric:
                self.x_element = x_element
                self.x_x_all, \
                    self.x_nbf, self.x_nbele, self.x_alnmt, self.x_glb_bcface_type, \
                    self.x_fina, self.x_cola, self.x_ncola, \
                    self.x_bc_node_list, self.x_cg_ndglno, self.x_cg_nonods, \
                    self.x_ref_node_order, self.x_prolongator_from_p1dg = \
                    init_2d(self.mesh, self.nele, self.nonods, self.x_element.nloc, nface=self.ndim + 1)
                # converse to torch.tensor
                self.x_ref_in = torch.tensor(
                    self.x_x_all.reshape((self.nele, self.x_element.nloc, self.x_element.ndim)
                                         ).transpose((0, 2, 1)),
                    device=dev, dtype=torch.float64
                )
            else:
                self.x_element = element
                # converse to torch.tensor
                self.x_ref_in = torch.tensor(
                    self.x_all.reshape((self.nele, self.element.nloc, self.element.ndim)).transpose((0, 2, 1)),
                    device=dev, dtype=torch.float64
                )
        elif self.ndim == 3:
            self.nele = mesh.cell_data_dict['gmsh:geometrical']['tetra'].shape[0]
            self.nonods = element.nloc * self.nele
            self.p1dg_nonods = self.p1cg_nloc * self.nele
            self.x_all, \
                self.nbf, self.nbele, self.alnmt, self.glb_bcface_type, \
                self.fina, self.cola, self.ncola, \
                self.bc_node_list, self.cg_ndglno, self.cg_nonods, \
                self.ref_node_order, self.prolongator_from_p1dg = \
                init_3d(self.mesh, self.nele, self.nonods, self.element.nloc, nface=self.ndim+1)
            self.restrictor_to_p1dg = torch.transpose(self.prolongator_from_p1dg, dim0=0, dim1=1)
            # if not iso-paramatric, we store geometry DOFs in x_ref_in
            if not_iso_parametric:
                self.x_element = x_element
                self.x_x_all, \
                    self.x_nbf, self.x_nbele, self.x_alnmt, self.x_glb_bcface_type, \
                    self.x_fina, self.x_cola, self.x_ncola, \
                    self.x_bc_node_list, self.x_cg_ndglno, self.x_cg_nonods, \
                    self.x_ref_node_order, self.x_prolongator_from_p1dg = \
                    init_3d(self.mesh, self.nele, self.nonods, self.x_element.nloc, nface=self.ndim + 1)
                # converse to torch.tensor
                self.x_ref_in = torch.tensor(
                    self.x_x_all.reshape((self.nele, self.x_element.nloc, self.x_element.ndim)
                                         ).transpose((0, 2, 1)),
                    device=dev, dtype=torch.float64
                )
            else:
                self.x_element = element
                # converse to torch.tensor
                self.x_ref_in = torch.tensor(
                    self.x_all.reshape((self.nele, self.element.nloc, self.element.ndim)).transpose((0, 2, 1)),
                    device=dev, dtype=torch.float64
                )
        self.nbele = torch.tensor(self.nbele, device=dev)
        self.nbf = torch.tensor(self.nbf, device=dev)
        self.alnmt = torch.tensor(self.alnmt, device=dev)
        self.glb_bcface_type = torch.tensor(self.glb_bcface_type, device=dev)
        self.cell_volume = None  # to store element volume
        self._get_cell_volume()
        self.restrictor_1order = self._get_pndg_restrictor(self.x_element.ele_order,
                                                           self.x_element.gi_order,
                                                           self.x_element.ndim,
                                                           dev)
        self.not_iso_parametric = not_iso_parametric
        self.pndg_ndglbno_f = None  # to store pndg node global number (only fluid subdomain)
        self.pncg_nonods_f = None  # to store pncg nonods (only fluid subdomain)
        self.pndg_ndglbno = None  # to store pndg node global number (whole domain)
        self.pncg_nonods = None  # to store pncg nonods (whole domain)
        if get_pndg_ndglbno:
            self.pndg_ndglbno_f = sparsity.get_fluid_pndg_sparsity(self)
            self.pncg_nonods_f = self.pndg_ndglbno_f.shape[1]  # this is fluid subdomain PnCG_nonods
            self.pndg_ndglbno = sparsity.get_pndg_sparsity(self)
            self.pncg_nonods = self.pndg_ndglbno.shape[1]  # this is whole domain PnCG_nonods

    def _get_cell_volume(self):
        """this is to get element volume and store in self.cell_volume"""
        n = self.element.n
        import shape_function
        if self.ndim == 3:
            _, ndetwei = shape_function.get_det_nlx_3d(
                nlx=self.x_element.nlx,
                x_loc=self.x_ref_in,
                weight=self.element.weight,
                nloc=self.element.nloc,
                ngi=self.element.ngi,
                real_nlx=self.element.nlx
            )
        elif self.ndim == 2:
            _, ndetwei = shape_function.get_det_nlx(
                nlx=self.x_element.nlx,
                x_loc=self.x_ref_in,
                weight=self.element.weight,
                nloc=self.element.nloc,
                ngi=self.element.ngi,
                real_nlx=self.element.nlx
            )
        else:
            raise Exception('function space must reside in either 3D or 2D domain')
        self.cell_volume = torch.sum(ndetwei, dim=-1)

    def get_x_all_after_move_mesh(self):
        """
        after move the mesh, geometry nodes x_ref_in has been moved.
        we need to move the DOFs (x_all)
        as well for output, and for post-processing.

        if the space is super-parametric, we will use the pndg to
        p(n-1)dg projection to get x_all.

        if the space is iso-parametric, just copy x_ref_in to x_all.
        """
        if self.not_iso_parametric:
            x_all_new = torch.einsum(
                'mn,bin->bmi',
                self.restrictor_1order,
                self.x_ref_in,
            )
            self.x_all *= 0
            self.x_all += x_all_new.reshape(self.x_all.shape).cpu().numpy()
        else:
            self.x_all *= 0
            self.x_all += self.x_ref_in.permute(0, 2, 1).reshape(self.x_all.shape).cpu().numpy()

    @staticmethod
    def _get_pndg_restrictor(p, gi_order, ndim, dev):
        """
        get restrictor from order p element to order (p-1) element
        """
        p_ele = Element(p, gi_order, ndim, dev)
        p_1_ele = Element(p-1, gi_order, ndim, dev)
        np_1_np_1 = torch.einsum(
            'mg,ng,g->mn',
            p_1_ele.n,
            p_1_ele.n,
            p_1_ele.weight
        )  # p2dg_nloc x p2dg_nloc
        np_np_1 = torch.einsum(
            'mg,ng,g->mn',
            p_1_ele.n,
            p_ele.n,
            p_1_ele.weight,
        )  # p2dg_nloc x p3dg_nloc
        restrictor = torch.matmul(
            torch.inverse(np_1_np_1), np_np_1
        )  # p2dg_nloc x p3dg_nloc
        return restrictor

    @staticmethod
    def get_pndg_prolongator(p, gi_order, ndim, dev):
        # get prolongator from order p element to order (p+1) element
        p_ele = Element(p, gi_order, ndim, dev)
        p_1_ele = Element(p+1, gi_order, ndim, dev)
        np_1_np_1 = torch.einsum(
            'mg,ng,g->mn',
            p_1_ele.n,
            p_1_ele.n,
            p_1_ele.weight
        )
        np_1_np = torch.einsum(
            'mg,ng,g->mn',
            p_1_ele.n,
            p_ele.n,
            p_1_ele.weight,
        )
        prolongator = torch.matmul(
            torch.inverse(np_1_np_1),
            np_1_np,
        )
        return prolongator
