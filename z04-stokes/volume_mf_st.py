#!/usr/bin/env python3

"""
all integrations for hyper-elastic
matric-free implementation
"""

import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm
import config
from config import sf_nd_nb
# import materials
# we sould be able to reuse all multigrid functions in linear-elasticity in hyper-elasticity
import multigrid_linearelastic as mg_le
if config.ndim == 2:
    from shape_function import get_det_nlx as get_det_nlx
    from shape_function import sdet_snlx as sdet_snlx
else:
    from shape_function import get_det_nlx_3d as get_det_nlx
    from shape_function import sdet_snlx_3d as sdet_snlx


dev = config.dev
nele = config.nele
mesh = config.mesh
ndim = config.ndim
nface = config.ndim+1


def calc_RAR_mf_color():
    ...


def get_residual_and_smooth_once():
    ...


def get_residual_only():
    ...


def _k_res_one_batch():
    ...


def _s_res_one_batch():
    ...


def _s_res_fi():
    """internal faces"""
    ...


def _s_res_fb():
    """boundary faces"""
    ...


def get_rhs():
    """get right-hand side"""
    ...
