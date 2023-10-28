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
        self.disp_func_space = None  # displacement function space
        self.p1cg_nonods = None
        self.sparse_f = Sparsity()  # sparsity for fluid subdomain
        self.sparse_s = Sparsity()  # sparsity for solid subdomain
        self.RARmat_F = None  # velocity blk on P1CG
        self.RARmat_Lp = None  # pressure laplacian on P1CG
        self.RARmat_S = None  # solid blk on P1DG
        self.RARmat_Um = None  # mesh displacement/velocity blk on P1DG
        self.sfc_data_F = SFCdata()  # sfc data for velocity block (adv + transient + diff)
        self.sfc_data_Lp = SFCdata()  # sfc data for pressure laplacian
        self.sfc_data_S = SFCdata()  # sfc data for solid block
        self.sfc_data_Um = SFCdata()  # sfc data for mesh displacement/velocity
        self.Lpmatinv = None  # inverse of pressure Laplacian
        self.Kmatinv = None  # velocity block of stokes problem
        # velocity block of stokes problem - values and coordinates (coo format)
        self.indices_st = None
        self.values_st = None
        self.bdfscm = None  # bdf scheme
        self.use_fict_dt_in_vel_precond = None  # add mass matrix to preconditioner (velocity block)
        self.fict_dt = 0.  # fictitious time step for fictitious mass matrix (added to velocity blk precond)
        self.isTransient = False  # if we're solving transient eq
        self.dt = None  # timestep
        self.isPetrovGalerkin = False  # flag to including Petrov-Galerkin stabilisation term
        self.isPetrovGalerkinFace = False  # flag to also include PG stab on surface
        self.projection_one_order_lower = None  # project high-order element to one-order lower element (used in getting
                                                # Petrov-Galerkin residual
        self.nits = 0  # current non-linear step is the (nits)-th step.
        self.its = 0  # current linear step is the (its)-th step.
        self.ntime = 0  # current number of timestep
        self.u_ave = None  # volume-averaged velocity (nele, ndim)
        self.isES = False  # edge-stabilisation or not
        self.u_m = None  # mesh velocity (nele, u_nloc, ndim)
        self.inter_stress_imbalance = None  # interface stress imbalance (nele, sngi)

        self.material = None  # structure material (e.g. NeoHookean, StVenant-Kirchoff)

    def set_data(self,
                 vel_func_space=None,
                 pre_func_space=None,
                 disp_func_space=None,  # displacement function space
                 p1cg_nonods=None,
                 # vel_I_prol=None,
                 # pre_I_prol=None,
                 sparse_s=None,  # solid subdomain sparsity
                 sparse_f=None,  # fluid subdomain sparsity
                 RARmat_F=None,  # velocity blk operator on P1CG, type: scipy csr sparse matrix
                 RARmat_Lp=None,  # pressure laplacian on P1CG
                 RARmat_S=None,  # solid blk operator on P1DG, type: scipy csr sparse matrix
                 RARmat_Um=None,  # mesh displacement/velocity blk operator on P1DG, type: scipy csr sparse matrix
                 Kmatinv=None,  # inverse of velocity block of stokes problem
                 indices_st=None,
                 values_st=None,
                 bdfscm=None,
                 u_ave=None,  # volume averaged velocity (nele, ndim)
                 u_m=None,  # mesh velocity (nele, u_nloc, ndim)
                 material=None,  # structure material (e.g. NeoHookean, StVenant-Kirchoff)
                 ):
        if type(vel_func_space) != NoneType:
            self.vel_func_space = vel_func_space
        if type(pre_func_space) != NoneType:
            self.pre_func_space = pre_func_space
        if disp_func_space is not None:
            self.disp_func_space = disp_func_space
        if type(p1cg_nonods) != NoneType:
            self.p1cg_nonods = p1cg_nonods
        # if type(vel_I_prol) != NoneType:
        #     self.vel_I_prol = vel_I_prol
        #     self.vel_I_rest = torch.transpose(vel_I_prol, dim0=0, dim1=1)
        # if type(pre_I_prol) != NoneType:
        #     self.pre_I_prol = pre_I_prol
        #     self.pre_I_rest = torch.transpose(pre_I_prol, dim0=0, dim1=1)
        if sparse_s is not None:
            self.sparse_s = sparse_s
        if sparse_f is not None:
            self.sparse_f = sparse_f
        if RARmat_F is not None:
            self.RARmat_F = RARmat_F  # velocity blk operator on P1CG, type: scipy csr sparse matrix
        if RARmat_Lp is not None:
            self.RARmat_Lp = RARmat_Lp  # operator on P1CG, type: scipy csr sparse matrix
        if RARmat_S is not None:
            self.RARmat_S = RARmat_S
        if RARmat_Um is not None:
            self.RARmat_Um = RARmat_Um
        if type(Kmatinv) != NoneType:
            self.Kmatinv = Kmatinv
        if indices_st is not None:
            self.indices_st = indices_st
        if values_st is not None:
            self.values_st = values_st
        if bdfscm is not None:
            self.bdfscm = bdfscm
        if u_ave is not None:
            self.u_ave = u_ave
        if u_m is not None:
            self.u_m = u_m
        if material is not None:
            self.material = material


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
    """store BDF scheme coefficients

    BDF-J scheme:

    1st order time derivative: du/dt = gamma u^{t+1} + sum_i alpha_i u^{t-i}, i from 1 to J

    2nd order time derivative: d^2u/dt^2 = sum_i beta_i u^{t+1-i}, i from 0 to J+1
    """

    def __init__(self,
                 order):
        self.order = order
        if order == 1:
            self.gamma = 1
            self.alpha = [1]
            self.beta = [1, -2, 1]
        elif order == 2:
            self.gamma = 3/2
            self.alpha = [2, -1/2]
            self.beta = [2, -5, 4, -1]
        elif order == 3:
            self.gamma = 11/6
            self.alpha = [3, -3/2, 1/3]
            self.beta = [35/12, -26/3, 19/2, -14/3, 11/12]

    def set_data(self,
                 gamma=None,
                 alpha=None,
                 beta=None
                 ):
        if gamma is not None:
            self.gamma = gamma
        if alpha is not None:
            if len(alpha) != len(self.alpha):
                raise ValueError('the alpha coefficient you want to store is of wrong size for order '
                                 + self.order + ' BDF scheme')
            self.alpha = alpha
        if beta is not None:
            if len(beta) != len(self.beta):
                raise ValueError('the beta coefficient you want to store is of wrong size for order '
                                 + self.order + ' BDF scheme')
            self.beta = beta

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


class Sparsity:
    def __init__(self, name=None):
        self.name = name
        self.fina = None
        self.cola = None
        self.ncola = None
        self.whichc = None
        self.ncolor = None
        self.I_fc = None
        self.I_cf = None
        self.cg_nonods = None
        self.p1dg_nonods = None

    def set_data(self,
                 fina=None,
                 cola=None,
                 ncola=None,
                 whichc=None,
                 ncolor=None,
                 I_fc=None,
                 I_cf=None,
                 cg_nonods=None,
                 p1dg_nonods=None,
                 ):
        if fina is not None:
            self.fina = fina
        if cola is not None:
            self.cola = cola
        if ncola is not None:
            self.ncola = ncola
        if whichc is not None:
            self.whichc = whichc
        if ncolor is not None:
            self.ncolor = ncolor
        if I_fc is not None:
            self.I_fc = I_fc
        if I_cf is not None:
            self.I_cf = I_cf
        if cg_nonods is not None:
            self.cg_nonods = cg_nonods
        if p1dg_nonods is not None:
            self.p1dg_nonods = p1dg_nonods
