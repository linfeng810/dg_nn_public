#!/usr/bin/env python3

import numpy as np 
import torch

# device
dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dev="cpu" # if force funning on cpu
torch.set_printoptions(precision=16)

a = np.asarray([1.,0.,0.,1.,3./4.,4./3.,2.,6.])
a = np.reshape(a,(2,2,2))

a_tensor = torch.tensor(a,device=dev)
print(a_tensor)
b = torch.linalg.vector_norm(a_tensor[:,:,0]-a_tensor[:,:,1], dim=1)
print(a_tensor[:,:,0]-a_tensor[:,:,1])
print(b)