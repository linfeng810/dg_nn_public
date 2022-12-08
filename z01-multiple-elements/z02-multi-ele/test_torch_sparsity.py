#!/usr/bin/env python3

import torch 

# target: [[5.,0,0,0], [0,8.,0,0], [0,0,3,0],[0,6,0,0]]
a = torch.tensor([[0, 2.], [3, 0]]) # dummy input
a=a.to_sparse_csr()
print(a)
print(type(a))
print(a.layout)
crow_indices=torch.tensor([0,1,2,3,4])
col_indices=torch.tensor([0,1,2,1])
values=torch.tensor([5.,8.,3.,6.])
size=(4,4)
b=torch.sparse_csr_tensor(crow_indices, col_indices, values, size)
print(type(b))
print(b.layout)
print(b.to_dense())
print(b.crow_indices)
print(b)

indices = [[0,0], [1,1],[2,2],[3,1]]
indices = torch.tensor(indices)
indices = torch.transpose(indices, 0,1)
print(indices.shape)
print(indices)
values = [5,8,3,6]
c = torch.sparse_coo_tensor(indices, values)
print(c)