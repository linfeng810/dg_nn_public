#!/usr/bin/env python3

"""
This is a collection of solid materials.
Presented as classes. Import the class one wishes to 
use into the volume/surface integral file.
"""

import torch
import config
from config import sf_nd_nb
if config.ndim == 2:
    from shape_function import get_det_nlx as get_det_nlx
    from shape_function import sdet_snlx as sdet_snlx
else:
    from shape_function import get_det_nlx_3d as get_det_nlx
    from shape_function import sdet_snlx_3d as sdet_snlx


class NeoHookean:
    # FIXME: F, C, CC, P, AA should all have values at nloc nodes,
    #    and for each node it has values at ngi gaussian points.
    #    so the shape should be (batch_in, nloc, ..., ngi)
    def __init__(self, nloc, ndim, dev, mu, lam):
        self.nloc = nloc
        self.ndim = ndim
        self.dev = dev
        self.mu = mu
        self.lam = lam

    def _calc_F(self, nlx, u, batch_in: int):
        # compute deformation gradient
        # here we assum nx shape is (batch_in, ndim, nloc, ngi)
        F = torch.einsum('bni,bjng->bgij', u.view(batch_in, self.nloc, self.ndim), nlx)
        F += torch.eye(self.ndim, device=self.dev, dtype=torch.float64)
        # output shape is (batch_in, ngi, ndim, ndim)
        return F

    def _calc_C(self, F):
        """
        compute right Cauchy-Green tensor
        C = F^T F
        """
        C = torch.einsum('bgik,bgij->bgkj', F, F)
        # output shape is (batch_in, ngi, ndim, ndim)
        return C

    def _calc_CC(self, C, F):
        """
        compute elasticity tensor $\mathbb C$
        $\mathbb C = \partial S / \partial C$
        """
        batch_in = C.shape[0]
        ngi = C.shape[1]
        invC = torch.linalg.inv(C)  # C^{-1}
        invCinvC = torch.einsum('bgij,bgkl->bgijkl', invC, invC)  # C^{-1} \otimes C^{-1}
        J = torch.linalg.det(F)  # this is J = det F, or J^2 = det C
        if torch.any(J <= 0):
            raise Exception(f'determinant of some element is negative! ',
                            f'consider larger relaxation... or there is a bug :/')
        mu_m_lam_lnJ = self.mu - self.lam * torch.log(J)
        CC = torch.zeros(batch_in, ngi, self.ndim, self.ndim, self.ndim, self.ndim,
                         device=self.dev, dtype=torch.float64)
        CC += self.lam * invCinvC
        CC += torch.einsum('bg,bgikjl->bgijkl', mu_m_lam_lnJ, invCinvC)
        CC += torch.einsum('bg,bgiljk->bgijkl', mu_m_lam_lnJ, invCinvC)
        # output shape is (batch_in, ngi, ndim, ndim, ndim, ndim)
        return CC

    def _calc_S(self, C, F):
        # compute PK2 tensor
        batch_in = C.shape[0]
        ngi = C.shape[1]
        invC = torch.linalg.inv(C)  # C^{-1}
        lnJ = torch.log(torch.linalg.det(F))  # ln J = ln det F
        S = torch.zeros(batch_in, ngi, self.ndim, self.ndim,
                        device=self.dev, dtype=torch.float64)
        S += self.mu * (torch.eye(self.ndim, device=self.dev, dtype=torch.float64)
                        - invC)
        S += self.lam * torch.einsum('bg,bgij->bgij', lnJ, invC)
        # output shape is (batch_in, ngi, ndim, ndim)
        return S

    def calc_P(self, nlx, u, batch_in: int):
        """
        compute PK1 tensor from given displacement
        """
        F = self._calc_F(nlx, u, batch_in)
        C = self._calc_C(F)
        S = self._calc_S(C, F)
        P = torch.einsum('bgij,bgjk->bikg', F, S)
        # output shape is (batch_in, ndim, ndim, ngi)
        return P

    def calc_AA(self, nlx, u, batch_in: int):
        """
        compute elasticity tensor \mathbb A
        at given intermediate state (displacement)
        \mathbb A = \partial P / \partial F
                  = delta S + F F C
        """
        ngi = nlx.shape[-1]
        F = self._calc_F(nlx, u, batch_in)
        C = self._calc_C(F)
        S = self._calc_S(C, F)
        CC = self._calc_CC(C, F)
        AA = torch.zeros(batch_in, self.ndim, self.ndim, self.ndim, self.ndim, ngi,
                         device=self.dev, dtype=torch.float64)
        AA += torch.einsum('bgiI,bgkK,bgIJKL->biJkLg', F, F, CC)
        AA += torch.einsum('ik,bgJL->biJkLg',
                           torch.eye(self.ndim, device=self.dev, dtype=torch.float64),
                           S)
        # output shape is (batch_in, ndim, ndim, ndim, ndim, ngi)
        return AA


class LinearElastic:
    def __init__(self, nloc, ndim, dev, mu, lam):
        self.nloc = nloc
        self.ndim = ndim
        self.dev = dev
        self.mu = mu
        self.lam = lam

    def _calc_F(self, nlx, u, batch_in: int):
        # compute deformation gradient
        # here we assum nx shape is (batch_in, ndim, nloc, ngi)
        F = torch.einsum('bni,bjng->bgij', u.view(batch_in, self.nloc, self.ndim), nlx)
        F += torch.eye(self.ndim, device=self.dev, dtype=torch.float64)
        # output shape is (batch_in, ngi, ndim, ndim)
        return F

    def _calc_C(self, F):
        """
        compute right Cauchy-Green tensor
        C = F^T F
        """
        C = torch.einsum('bgik,bgij->bgkj', F, F)
        # output shape is (batch_in, ngi, ndim, ndim)
        return C

    def _calc_CC(self):
        """
        compute elasticity tensor $\mathbb C$
        $\mathbb C = \partial S / \partial C$
                   = \lambda I_ij I_kl + \mu (I_ik I_jl + I_il I_jk)
        """
        I3 = torch.eye(self.ndim, device=self.dev, dtype=torch.float64)
        CC = self.lam * torch.einsum('ij,kl->ijkl', I3, I3) \
            + self.mu * torch.einsum('ik,jl->ijkl', I3, I3) \
            + self.mu * torch.einsum('il,jk->ijkl', I3, I3)
        # print(CC)
        # output shape is (ndim, ndim, ndim, ndim)
        return CC

    def calc_P(self, nx, u, batch_in):
        """
        compute PK1 tensor from given displacement
        """
        nabla_u = torch.einsum('bjng,bni->bijg', nx, u.view(batch_in, self.nloc, self.ndim))
        CC = self._calc_CC()
        P = torch.einsum('ijkl,bklg->bijg', CC, nabla_u)
        # output shape is (batch_in, ndim, ndim, ngi)
        return P

    def calc_AA(self, nx, u, batch_in):
        """
        compute elasticity tensor \mathbb A
        at given intermediate state (displacement)
        \mathbb A = \mathbb C
        """

        # although we don't need u and ele_list for linear elastic,
        # we keep them in the inputs list for ensure same function
        # interface for different materials.

        ngi = nx.shape[-1]
        AA = torch.zeros((batch_in,
                          self.ndim, self.ndim,
                          self.ndim, self.ndim,
                          ngi),
                         device=self.dev, dtype=torch.float64)
        AA += self._calc_CC().unsqueeze(0).unsqueeze(-1).expand(batch_in,
                                                                -1, -1, -1, -1,
                                                                ngi)
        # output shape is (batch_in, ndim, ndim, ndim, ndim, ngi)
        return AA
