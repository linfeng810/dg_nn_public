# let build the lhs matrix column by column...
import scipy as sp
import torch
from tqdm import tqdm
import config
import pressure_matrix
from config import sf_nd_nb

dev = config.dev


def vanilla_assemble_Lp():
    p_nonods = sf_nd_nb.pre_func_space.nonods
    # a bunch of dummies...
    p_dummy = torch.zeros(p_nonods,
                          device=dev, dtype=torch.float64)
    rhs_dummy = torch.zeros(p_nonods, device=dev, dtype=torch.float64)
    # assemble column by column
    Amat = torch.zeros(p_nonods, p_nonods,
                       device=dev, dtype=torch.float64)
    rhs = torch.zeros(p_nonods,
                      device=dev, dtype=torch.float64)
    probe = torch.zeros(p_nonods,
                        device=dev, dtype=torch.float64)

    # get pressure columns
    for inod in tqdm(range(p_nonods)):
        p_dummy *= 0
        rhs_dummy *= 0
        probe *= 0
        probe[inod] += 1.
        p_dummy, _ = pressure_matrix._apply_pressure_mat(
            r0=p_dummy,
            x_i=probe,
            x_rhs=rhs_dummy,
            include_adv=False,
            doSmooth=False,
        )
        # put in Amat
        Amat[:, inod] -= p_dummy.view(-1)
    Amat_np = Amat.cpu().numpy()
    Amat_sp = sp.sparse.csr_matrix(Amat_np)
    return Amat_sp
