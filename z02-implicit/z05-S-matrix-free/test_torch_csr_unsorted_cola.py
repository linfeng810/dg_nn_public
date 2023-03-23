#!/usr/bin/env python3

import torch 

fina = [0,3,6,8,11,13]
cola = [0,1,3,1,2,4,0,1,1,2,3,4,2]
vals = [4.,2.,1.,5.,6.,7.,5.,6.,4.,1.,8.,4.,9.]
mat = torch.sparse_csr_tensor(
    crow_indices=fina,
    col_indices=cola,
    values=torch.tensor(vals)
)
print(mat.to_dense())