#!/usr/bin/env python3

from types import NoneType

import torch


class SfNdNb:
    """
    this is a class of objects that stores essential and
    repetively occured discretisation space data, including
    1) volume/surface shape function, derivatives and quadrature
    weights (n, nlx, weight, sn, snlx, sweight),
    2) nodes coordinates on finest grid (highest p) (x_ref_in)
    3) neighouring face and elements list (nbele, nbf)
    """
    def __init__(self,
                 n=None,
                 nlx=None,
                 weight=None,  # volume shape function on ref. element
                 sn=None,
                 snlx=None,
                 sweight=None,  # surface shape function on ref. element
                 x_ref_in=None,  # nodes coordinate on PnDG (finest mesh)
                 nbele=None,  # neighbour element list
                 nbf=None,  # neighbour face list
                 cg_ndglno=None,  # P1CG connectivity matrix
                 cg_nonods=None,  # number of nodes on P1CG
                 alnmt=None,  # face alignment in 3D
                 ):
        self.n = n
        self.nlx = nlx
        self.weight = weight
        self.sn = sn
        self.snlx = snlx
        self.sweight = sweight
        self.x_ref_in = x_ref_in
        self.nbele = nbele
        self.nbf = nbf
        self.cg_ndglno = cg_ndglno
        self.cg_nonods = cg_nonods
        self.alnmt = alnmt
        self.I_31 = None
        self.I_13 = None
        self.gi_align = None  # gaussian points alignment (3 possibilities in 3D and 2 in 2D)
        self.I_cf = None  # discontinuous P1DG to continuous P1CG prolongator
        self.I_fc = None  # continuous P1CG to discontinuous p1DG restrictor


    def set_data(self,
                 n=None,
                 nlx=None,
                 weight=None,  # volume shape function on ref. element
                 sn=None,
                 snlx=None,
                 sweight=None,  # surface shape function on ref. element
                 x_ref_in=None,  # nodes coordinate on PnDG (finest mesh)
                 nbele=None,  # neighbour element list
                 nbf=None,  # neighbour face list
                 cg_ndglno=None,  # P1CG connectivity matrix
                 cg_nonods=None,  # number of nodes on P1CG
                 alnmt=None,  # face alignment in 3D
                 I_31=None,
                 gi_align=None,  # gaussian points alignment (3 possibilities in 3D and 2 in 2D)
                 I_cf = None,  # discontinuous P1DG to continuous P1CG prolongator
                 I_fc = None,  # continuous P1CG to discontinuous p1DG restrictor
                 ):
        if type(n) != NoneType:
            self.n = n
        if type(nlx) != NoneType:
            self.nlx = nlx
        if type(weight) != NoneType:
            self.weight = weight
        if type(sn) != NoneType:
            self.sn = sn
        if type(snlx) != NoneType:
            self.snlx = snlx
        if type(sweight) != NoneType:
            self.sweight = sweight
        if type(x_ref_in) != NoneType:
            self.x_ref_in = x_ref_in
        if type(nbele) != NoneType:
            self.nbele = nbele
        if type(nbf) != NoneType:
            self.nbf = nbf
        if type(cg_ndglno) != NoneType:
            self.cg_ndglno = cg_ndglno
        if type(cg_nonods) != NoneType:
            self.cg_nonods = cg_nonods
        if type(alnmt) != NoneType:
            self.alnmt = alnmt
        if type(I_31) != NoneType:
            self.I_31 = I_31
            self.I_13 = torch.transpose(I_31, dim0=0, dim1=1)
        if type(gi_align) != NoneType:
            self.gi_align = gi_align
        if type(I_cf) != NoneType:
            self.I_cf = I_cf
        if type(I_fc) != NoneType:
            self.I_fc = I_fc
