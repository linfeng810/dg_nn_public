import torch 
import numpy as np
# a = [1,2,0,0, 0,2,4,0, 0,0,9,0, 1,0,0,5]
# a = np.asarray(a)
# a = np.reshape(a,(4,4))
# a = torch.tensor(a)
# a = a.to_sparse_csr()
# b = torch.clone(a)

a = torch.randn(2, 3).to_sparse_csr().requires_grad_(True)
b = torch.randn(3, 2).to_sparse_csr().requires_grad_(True)
y = torch.sparse.mm(a, b)

print(a)
print(b)
c = torch.sparse.mm(a,b)
print(c)

'''
Conclusion:
torch.sparse.mm for two csr sparse matrices is not working
in torch 1.11.0
It's working in torch 1.13.0 (on ese-foundation).
'''