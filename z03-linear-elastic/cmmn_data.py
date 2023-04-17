#!/usr/bin/env python3

from types import NoneType


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
