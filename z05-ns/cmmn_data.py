#!/usr/bin/env python3

from types import NoneType
import torch
# from function_space import FuncSpace


class SfNdNb:
    """
    this is a class of objects that stores essential and
    repetively occured discretisation space data, including
    1) volume/surface shape function, derivatives and quadrature
    weights (n, nlx, weight, sn, snlx, sweight),
    2) nodes coordinates on finest grid (highest p) (x_ref_in)
    3) neighouring face and elements list (nbele, nbf)
    """

    def __init__(self):
        self.vel_func_space = None
        self.pre_func_space = None
        self.p1cg_nonods = None
        # self.vel_I_prol = None  # velocity p prolongator (from P1DG to PnDG)
        # self.vel_I_rest = None  # velocity p restrictor (from PnDG to P1DG)
        # self.pre_I_prol = None  # velocity p prolongator (from P1DG to PnDG)
        # self.pre_I_rest = None  # velocity p restrictor (from PnDG to P1DG)
        self.I_dc = None  # prolongator from P1CG to P1DG
        self.I_cd = None  # restrictor from P1DG to P1CG
        self.RARmat = None  # operator on P1CG (velocity and pressure both here. they are same shape on P1CG)
        self.RARmat_Lp = None  # pressure laplacian on P1CG
        self.sfc_data = SFCdata()  # sfc data for velocity block (adv + transient + diff)
        self.sfc_data_Lp = SFCdata()  # sfc data for pressure laplacian
        self.Lpmatinv = None  # inverse of pressure Laplacian
        self.Kmatinv = None  # velocity block of stokes problem
        # velocity block of stokes problem - values and coordinates (coo format)
        self.indices_st = None
        self.values_st = None
        self.bdfscm = None  # bdf scheme

    def set_data(self,
                 vel_func_space=None,
                 pre_func_space=None,
                 p1cg_nonods=None,
                 # vel_I_prol=None,
                 # pre_I_prol=None,
                 I_cd=None,  # discontinuous P1DG to continuous P1CG prolongator
                 I_dc=None,  # continuous P1CG to discontinuous p1DG restrictor
                 RARmat=None,  # operator on P1CG, type: scipy csr sparse matrix
                 RARmat_Lp=None,  # pressure laplacian on P1CG
                 Kmatinv=None,  # inverse of velocity block of stokes problem
                 indices_st=None,
                 values_st=None,
                 bdfscm=None,
                 ):
        if type(vel_func_space) != NoneType:
            self.vel_func_space = vel_func_space
        if type(pre_func_space) != NoneType:
            self.pre_func_space = pre_func_space
        if type(p1cg_nonods) != NoneType:
            self.p1cg_nonods = p1cg_nonods
        # if type(vel_I_prol) != NoneType:
        #     self.vel_I_prol = vel_I_prol
        #     self.vel_I_rest = torch.transpose(vel_I_prol, dim0=0, dim1=1)
        # if type(pre_I_prol) != NoneType:
        #     self.pre_I_prol = pre_I_prol
        #     self.pre_I_rest = torch.transpose(pre_I_prol, dim0=0, dim1=1)
        if type(I_cd) != NoneType:
            self.I_cd = I_cd
        if type(I_dc) != NoneType:
            self.I_dc = I_dc
        if type(RARmat) != NoneType:
            self.RARmat = RARmat  # operator on P1CG, type: scipy csr sparse matrix
        if type(RARmat_Lp) != NoneType:
            self.RARmat_Lp = RARmat_Lp  # operator on P1CG, type: scipy csr sparse matrix
        if type(Kmatinv) != NoneType:
            self.Kmatinv = Kmatinv
        if indices_st is not None:
            self.indices_st = indices_st
        if values_st is not None:
            self.values_st = values_st
        if bdfscm is not None:
            self.bdfscm = bdfscm


class SFCdata:
    """
    SFC related data, including:
    space_filling_curve_numbering
    variables_sfc
    nlevel
    nodes_per_level
    """

    def __init__(self,
                 space_filling_curve_numbering=None,
                 variables_sfc=None,
                 nlevel=None,
                 nodes_per_level=None,
                 ):
        self.space_filling_curve_numbering = space_filling_curve_numbering
        self.variables_sfc = variables_sfc
        self.nlevel = nlevel
        self.nodes_per_level = nodes_per_level

    def set_data(self,
                 space_filling_curve_numbering=None,
                 variables_sfc=None,
                 nlevel=None,
                 nodes_per_level=None,
                 ):
        if type(space_filling_curve_numbering) != NoneType:
            self.space_filling_curve_numbering = space_filling_curve_numbering
        if type(variables_sfc) != NoneType:
            self.variables_sfc = variables_sfc
        if type(nlevel) != NoneType:
            self.nlevel = nlevel
        if type(nodes_per_level) != NoneType:
            self.nodes_per_level = nodes_per_level


class BDFdata:
    """store BDF scheme coefficients"""

    def __init__(self,
                 order):
        self.order = order
        if order == 1:
            self.gamma = 1
            self.alpha = [1]
        elif order == 2:
            self.gamma = 3/2
            self.alpha = [2, -1/2]
        elif order == 3:
            self.gamma = 11/6
            self.alpha = [3, -3/2, 1/3]

    def set_data(self,
                 gamma=None,
                 alpha=None,
                 ):
        if gamma is not None:
            self.gamma = gamma
        if alpha is not None:
            if len(alpha) != len(self.alpha):
                raise ValueError('the alpha coefficient you want to store is of wrong size for order '
                                 + self.order + ' BDF scheme')
            self.alpha = alpha

    def compute_coeff(self,
                      dt_list):
        """given a list of previous time steps,
        compute the coefficients.

        input: dt_n, dt_{n-1}, dt_{n-2}, ...

        use this when the time step is not constant"""
        if len(dt_list) != len(self.alpha):
            raise ValueError('the timesteps you input to compute '
                             'the BDF coefficient is of wrong size for order '
                             + self.order + ' BDF scheme')
        if self.order == 2:
            self.gamma = (2 * dt_list[0] + dt_list[1]) / (dt_list[0] + dt_list[1])
            self.alpha[0] = (dt_list[0] + dt_list[1]) / dt_list[1]
            self.alpha[1] = - dt_list[0]**2 / (dt_list[0] + dt_list[1]) / dt_list[1]
        elif self.order == 3:
            self.gamma = 1 + dt_list[0] / (dt_list[0] + dt_list[1]) \
                + dt_list[0] / (dt_list[0] + dt_list[1] + dt_list[2])
            self.alpha[0] = (dt_list[0] + dt_list[1]) * (dt_list[0] + dt_list[1] + dt_list[2]) \
                / dt_list[1] / (dt_list[1] + dt_list[2])
            self.alpha[1] = - dt_list[0]**2 * (dt_list[0] + dt_list[1] + dt_list[2]) \
                / (dt_list[0] + dt_list[1]) / dt_list[1] / dt_list[2]
            self.alpha[2] = dt_list[0]**2 * (dt_list[0] + dt_list[1]) \
                / (dt_list[0] + dt_list[1] + dt_list[2]) \
                / (dt_list[1] + dt_list[2]) / dt_list[2]
