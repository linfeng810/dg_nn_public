#!/usr/bin/env python3
""" this is to test surface integral of mixed formulation
used in stokes problem.
we will create two reference element sits symmetrycally 
about yz axial plane, and do surface integration of 
q.(v.n)
on the shared face."""

import torch
import numpy as np
from function_space import FuncSpace, Element
from shape_function import get_det_nlx_3d as get_det_nlx
from shape_function import sdet_snlx_3d as sdet_snlx

# get mesh

