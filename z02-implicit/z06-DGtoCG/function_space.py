import torch

# import config
import sparsity
from mesh_init import init_2d, init_3d
from shape_function import SHATRInew
import config
from typing import List, Optional


# dev = config.dev


@torch.jit.script
class Element(object):
    """
    this is a class that log element information.
    nloc, ngi, snloc, sngi,
    n, nlx, weight, sn, snlx, sweight at reference element
    ele_order
    """
    def __init__(self, ele_order: int, gi_order: int, edim: int, dev, dtype: torch.dtype):
        self.ele_order = ele_order
        self.gi_order = gi_order
        self.ndim = edim
        self.nloc = 0
        self.snloc = 0
        self.ngi = 0
        self.sngi = 0
        self.dtype = dtype
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
            SHATRInew(self.nloc, self.ngi, self.ndim, self.snloc, self.sngi, dev, self.dtype)
        # # converse to torch tensor
        # self.n = torch.tensor(self.n, device=dev, dtype=torch.float64)
        # self.nlx = torch.tensor(self.nlx, device=dev, dtype=torch.float64)
        # self.weight = torch.tensor(self.weight, device=dev, dtype=torch.float64)
        # self.sn = torch.tensor(self.sn, device=dev, dtype=torch.float64)
        # self.snlx = torch.tensor(self.snlx, device=dev, dtype=torch.float64)
        # self.sweight = torch.tensor(self.sweight, device=dev, dtype=torch.float64)


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
        self.jac_v = torch.Tensor()
        self.jac_s = torch.Tensor()
        if self.ndim == 2:
            self.nele = mesh.cell_data_dict['gmsh:geometrical'][config.ele_key_2d[config.ele_p - 1]].shape[0]
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
            # Now we're only store geometry DOFs in x_ref_in AND x_all

            if not_iso_parametric:
                self.x_element = x_element
                # self.x_x_all, \
                #     self.x_nbf, self.x_nbele, self.x_alnmt, self.x_glb_bcface_type, \
                #     self.x_fina, self.x_cola, self.x_ncola, \
                #     self.x_bc_node_list, self.x_cg_ndglno, self.x_cg_nonods, \
                #     self.x_ref_node_order, self.x_prolongator_from_p1dg = \
                #     init_2d(self.mesh, self.nele, self.nonods, self.x_element.nloc, nface=self.ndim + 1)
                # # converse to torch.tensor
                # self.x_ref_in = torch.tensor(
                #     self.x_x_all.reshape((self.nele, self.x_element.nloc, self.x_element.ndim)
                #                          ).transpose((0, 2, 1)),
                #     device=dev, dtype=torch.float64
                # )
            else:
                self.x_element = element
            # converse to torch.tensor
            self.x_ref_in = torch.tensor(
                self.x_all.reshape((self.nele, -1, self.element.ndim)).transpose((0, 2, 1)),
                device=dev, dtype=config.dtype
            )
        elif self.ndim == 3:
            self.nele = mesh.cell_data_dict['gmsh:geometrical'][config.ele_key_3d[config.ele_p - 1]].shape[0]
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
            # NOW we're only store geometry x_ref_in and
            if not_iso_parametric:
                self.x_element = x_element
                # self.x_x_all, \
                #     self.x_nbf, self.x_nbele, self.x_alnmt, self.x_glb_bcface_type, \
                #     self.x_fina, self.x_cola, self.x_ncola, \
                #     self.x_bc_node_list, self.x_cg_ndglno, self.x_cg_nonods, \
                #     self.x_ref_node_order, self.x_prolongator_from_p1dg = \
                #     init_3d(self.mesh, self.nele, self.nonods, self.x_element.nloc, nface=self.ndim + 1)
                # # converse to torch.tensor
                # self.x_ref_in = torch.tensor(
                #     self.x_x_all.reshape((self.nele, self.x_element.nloc, self.x_element.ndim)
                #                          ).transpose((0, 2, 1)),
                #     device=dev, dtype=torch.float64
                # )
            else:
                self.x_element = element
            # converse to torch.tensor
            self.x_ref_in = torch.tensor(
                self.x_all.reshape((self.nele, -1, self.element.ndim)).transpose((0, 2, 1)),
                device=dev, dtype=config.dtype
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
        if self.ndim == 3:
            self.drst_duv = torch.tensor([  # --> this is for shape function face det/norm
                [[0, 0], [0, 1], [1, 0]],  # face 2-1-3
                [[1, 0], [0, 0], [0, 1]],  # face 0-2-3
                [[0, 1], [1, 0], [0, 0]],  # face 1-0-3
                [[-1, -1], [1, 0], [0, 1]],  # face 0-1-2
            ], device=config.dev, dtype=config.dtype)  # (nface, ndim, ndim-1)
        elif self.ndim == 2:
            self.drst_duv = torch.tensor(  # actual name is deta_dlam
                [[-1., 1.],
                 [0., -1.],
                 [1., 0.]],
                device=config.dev, dtype=config.dtype
            )

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
                real_nlx=self.element.nlx,
                j=self.jac_v,
            )
        elif self.ndim == 2:
            _, ndetwei = shape_function.get_det_nlx(
                nlx=self.x_element.nlx,
                x_loc=self.x_ref_in,
                weight=self.element.weight,
                nloc=self.element.nloc,
                ngi=self.element.ngi,
                real_nlx=self.element.nlx,
                j=self.jac_v,
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
        # since we're always store geometry DOFs in x_ref_in and x_all,
        # we don't need to project for non-iso-gemetric spaces.
        if False:  # self.not_iso_parametric:
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

    def store_jacobian(self, jac_v: torch.Tensor, jac_s: torch.Tensor):
        self.jac_v = jac_v
        self.jac_s = jac_s

    @staticmethod
    def _get_pndg_restrictor(p, gi_order, ndim, dev):
        """
        get restrictor from order p element to order (p-1) element
        """
        p_ele = Element(p, gi_order, ndim, dev, config.dtype)
        p_1_ele = Element(p-1, gi_order, ndim, dev, config.dtype)
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


@torch.jit.script
class FuncSpaceTS:
    def __init__(self,
                 alnmt,
                 bc_node_list: List[torch.Tensor],
                 cell_volume,
                 cg_ndglno,
                 cg_nonods: int,
                 cola,
                 dev: torch.device,
                 element: Element,
                 fina,
                 glb_bcface_type,
                 name: str,
                 nbele,
                 nbf,
                 ncola: int,
                 nele: int,
                 nonods: int,
                 not_iso_parametric: bool,
                 p1cg_nloc: int,
                 p1dg_nonods: int,
                 pncg_nonods: Optional[int],
                 pncg_nonods_f: Optional[int],
                 pndg_ndglbno: Optional[int],
                 pndg_ndglbno_f: Optional[int],
                 prolongator_from_p1dg,
                 ref_node_order,
                 restrictor_1order,
                 restrictor_to_p1dg,
                 x_element: Element,
                 x_ref_in,
                 jac_v: torch.Tensor,
                 jac_s: torch.Tensor,
                 ):
        self.alnmt = alnmt
        self.bc_node_list = bc_node_list
        self.cell_volume = cell_volume
        self.cg_ndglno = cg_ndglno
        self.cg_nonods = cg_nonods
        self.cola = cola
        self.dev = dev
        self.element = element
        self.fina = fina
        self.glb_bcface_type = glb_bcface_type
        self.name = name
        self.nbele = nbele
        self.nbf = nbf
        self.ncola = ncola
        self.nele = nele
        self.nonods = nonods
        self.not_iso_parametric = not_iso_parametric
        self.p1cg_nloc = p1cg_nloc
        self.p1dg_nonods = p1dg_nonods
        self.pncg_nonods = pncg_nonods
        self.pncg_nonods_f = pncg_nonods_f
        self.pndg_ndglbno = pndg_ndglbno
        self.pndg_ndglbno_f = pndg_ndglbno_f
        self.prolongator_from_p1dg = prolongator_from_p1dg
        self.ref_node_order = ref_node_order
        self.restrictor_1order = restrictor_1order
        self.restrictor_to_p1dg = restrictor_to_p1dg
        self.x_element = x_element
        self.x_ref_in = x_ref_in
        self.jac_v = jac_v
        self.jac_s = jac_s
        self.drst_duv = torch.tensor(0)
        if self.element.ndim == 3:
            self.drst_duv = torch.tensor([  # --> this is for shape function face det/norm
                [[0, 0], [0, 1], [1, 0]],  # face 2-1-3
                [[1, 0], [0, 0], [0, 1]],  # face 0-2-3
                [[0, 1], [1, 0], [0, 0]],  # face 1-0-3
                [[-1, -1], [1, 0], [0, 1]],  # face 0-1-2
            ], device=self.dev, dtype=config.dtype)  # (nface, ndim, ndim-1)
        elif self.element.ndim == 2:
            self.drst_duv = torch.tensor(  # actual name is deta_dlam
                [[-1., 1.],
                 [0., -1.],
                 [1., 0.]],
                device=self.dev, dtype=config.dtype
            )

    def store_jacobian(self, jac_v: torch.Tensor, jac_s: torch.Tensor):
        self.jac_v = jac_v
        self.jac_s = jac_s


def create_funcspacets_from_funcspace(funcspace: FuncSpace):
    dev = config.dev
    return FuncSpaceTS(
        alnmt=funcspace.alnmt,
        bc_node_list=[torch.tensor(lst, device=dev, dtype=torch.bool) for lst in funcspace.bc_node_list],
        cell_volume=funcspace.cell_volume,
        cg_ndglno=torch.tensor(funcspace.cg_ndglno, device=dev, dtype=torch.int64),
        cg_nonods=funcspace.cg_nonods,
        cola=torch.tensor(funcspace.cola, device=dev, dtype=torch.int64),
        dev=dev,
        element=funcspace.element,
        fina=torch.tensor(funcspace.fina, device=dev, dtype=torch.int64),
        glb_bcface_type=funcspace.glb_bcface_type,
        name=funcspace.name,
        nbele=funcspace.nbele,
        nbf=funcspace.nbf,
        ncola=funcspace.ncola,
        nele=funcspace.nele,
        nonods=funcspace.nonods,
        not_iso_parametric=funcspace.not_iso_parametric,
        p1cg_nloc=funcspace.p1cg_nloc,
        p1dg_nonods=funcspace.p1dg_nonods,
        pncg_nonods=funcspace.pncg_nonods,
        pncg_nonods_f=funcspace.pncg_nonods_f,
        pndg_ndglbno=funcspace.pndg_ndglbno,
        pndg_ndglbno_f=funcspace.pndg_ndglbno_f,
        prolongator_from_p1dg=funcspace.prolongator_from_p1dg,
        ref_node_order=torch.tensor(funcspace.ref_node_order, device=dev, dtype=torch.int64),
        restrictor_1order=funcspace.restrictor_1order,
        restrictor_to_p1dg=funcspace.restrictor_to_p1dg,
        x_element=funcspace.x_element,
        x_ref_in=funcspace.x_ref_in,
        jac_v=torch.Tensor(0),
        jac_s=torch.Tensor(0)
    )
