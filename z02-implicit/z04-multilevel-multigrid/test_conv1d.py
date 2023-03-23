#!/usr/bin/env python3

####################################################
# preamble
####################################################
# import 
import numpy as np
import torch
from torch.nn import Conv1d,Sequential,Module
import config 

CNN1D_res = torch.nn.Conv1d(1, 1, kernel_size=2, stride=2, padding='valid', bias=False) 
filter_ = torch.tensor([[0.5, 0.5]],dtype=torch.float64, device=config.dev).view(1, 1, 2) 
CNN1D_res.weight.data = filter_

input = torch.tensor([1.,2.,3.,4.,5.,5.], device=config.dev, dtype=torch.float64).view(1,1,-1)
print(input)
numbering = np.asarray([2,3,4,1,0,5])
output = input[0,0,numbering]
print(output)