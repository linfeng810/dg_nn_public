""" smoothed aggregation AMG solver """
import pyamg
import torch
import numpy as np
import scipy.sparse as sp
import config
from cmmn_data import SFCdata

dev = config.dev


def get_rigid_body_mode(coor, ndim):
    """given coordinates on a mesh, return rigid body mode"""
    if type(coor) is not torch.Tensor:
        coor = torch.tensor(coor, dtype=config.dtype, device=config.dev)
    if ndim == 2:
        coords = coor[:, 0:2]
        x = coords[:, 0]
        y = coords[:, 1]
        nnod = x.shape[0]
        v = torch.zeros(3, nnod, ndim, dtype=config.dtype, device=config.dev)
        v[0, :, 0] += 1./nnod
        v[1, :, 1] += 1./nnod
        # rotation
        v[2, :, 0] += -y
        v[2, :, 1] += x
    elif ndim == 3:
        coords = coor[:, 0:3]
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        nnod = x.shape[0]
        v = torch.zeros(6, nnod, ndim, dtype=config.dtype, device=config.dev)
        v[0, :, 0] += 1./nnod
        v[1, :, 1] += 1./nnod
        v[2, :, 2] += 1./nnod
        # rotation
        v[3, :, 0] += y
        v[3, :, 1] += -x
        v[4, :, 1] += -z
        v[4, :, 2] += y
        v[5, :, 0] += z
        v[5, :, 2] += -x
    else:
        raise ValueError('ndim must be 2 or 3')
    # orthonormalize v
    v = v.view(-1, nnod * ndim)
    for i in range(v.shape[0]):
        for j in range(i):
            v[i] -= torch.dot(v[i], v[j]) * v[j]
        v[i] /= torch.linalg.norm(v[i])
    return v


def relax_rigid_body_mode(A, v, nsmooth=1):
    """relax rigid body mode with Jacobian smoother
    input
    A: CSR matrix
    v: nmode x nnod x ndim numpy array, where nmode is the number of rigid body modes
    nsmooth: number of smoothing steps
    output
    v: nmode x nnod x ndim numpy array -- after smoothing
    """
    diagA = A.diagonal()
    diagA = 1./diagA
    Dm12 = 1. / np.sqrt(np.abs(diagA))  # D^{-1/2}
    wei = 3. / 4. * np.linalg.norm(Dm12.transpose() @ A @ Dm12, np.inf)
    b = np.zeros_like(v)  # homogeneous right hand side
    for i in range(nsmooth):
        b -= wei * np.einsum('i,ib->bi', diagA, b.transpose() - A @ v.transpose())
    # # direct inverse to get v_s
    # v_s = sp.linalg.spsolve(A, v.transpose())
    #
    # v *= 0
    # v += v_s.transpose()
    return v


def print_out_eigen_vec(coor, v, ndim, name='rigid_body_mode.txt'):
    if type(v) is torch.Tensor:
        nmode = v.shape[0]
        v = v.view(nmode, -1, ndim).permute(1,2,0).contiguous().view(-1, ndim*nmode).cpu().numpy()
    else:
        nmode = v.shape[0]
        v = v.transpose().reshape(-1, ndim*nmode)
    np.savetxt(name,
               np.concatenate((coor, v),
                              axis=1), delimiter=',')


def create_sa_ml(A, v):
    """given a matrix A and its near null space v,
    using pyAMG, generate multi-level operators and prolongators
    and move them to designate torch device (config.dev)

    input:
    A: CSR matrix or BSR matrix
    v: nmod x cg1_nonods x ndim, near null space vectors

    output:
    sfc_data: cmmn_data.SFCdata object, has everything needed for AMG.
    """
    RAR_ml = config.pyAMGsmoother(A.tocsr(), B=v.cpu().numpy().transpose())
    nlevel = len(RAR_ml.levels)
    sfc_data = SFCdata()

    ml_torch = []
    nodes_per_level = []
    operators_on_level = []
    for level in RAR_ml.levels:
        nodes_per_level.append(level.A.shape[0])
        operators_on_level.append(
            (torch.sparse_csr_tensor(crow_indices=torch.tensor(level.A.indptr),
                                     col_indices=torch.tensor(level.A.indices),
                                     values=torch.tensor(level.A.data),
                                     size=level.A.shape,
                                     device=config.dev),
             torch.tensor(level.A.diagonal(), device=config.dev),
             level.A.shape[0],)
        )

    return ml_torch


class SASolver:
    """smoothed aggregation solver"""

    def __init__(self, A, v=None, omega=2./3.):
        """
        input:
        A: CSR matrix or BSR matrix
        v: nmod x ndof, near null space vectors
        omega: relaxation coefficient, i.e. Jacobi weight (we're only using point Jacobian smoother)
        """
        if v is not None:
            if type(v) is torch.Tensor:
                v = v.cpu().numpy()
            if v.shape[0] != A.shape[0]:
                v = v.transpose()
            ml = pyamg.smoothed_aggregation_solver(A.tocsr(), B=v)
        else:
            ml = pyamg.smoothed_aggregation_solver(A.tocsr())  # no near null space provided, pyamg will assume 1
        self.nlevel = len(ml.levels)
        self.sfc_data = SFCdata()
        self.levels = []
        self.jac_wei = omega
        for lvl in range(self.nlevel):
            # print('lvl', lvl)  # , 'invdiagA_l', invdiagA_l)
            level = ml.levels[lvl]
            A = level.A.tocsr()
            A_l = torch.sparse_csr_tensor(crow_indices=torch.tensor(A.indptr),
                                          col_indices=torch.tensor(A.indices),
                                          values=torch.tensor(A.data),
                                          size=A.shape,
                                          device=config.dev)
            if lvl < self.nlevel - 1:
                invdiagA_l = torch.tensor(level.A.diagonal(), device=config.dev)
                invdiagA_l[invdiagA_l != 0] = 1./invdiagA_l[invdiagA_l != 0]
                P = level.P.tocsr()
                P_l = torch.sparse_csr_tensor(crow_indices=torch.tensor(P.indptr),
                                              col_indices=torch.tensor(P.indices),
                                              values=torch.tensor(P.data),
                                              size=P.shape,
                                              device=config.dev)
                R = level.R.tocsr()
                R_l = torch.sparse_csr_tensor(crow_indices=torch.tensor(R.indptr),
                                              col_indices=torch.tensor(R.indices),
                                              values=torch.tensor(R.data),
                                              size=R.shape,
                                              device=config.dev)
            else:
                invdiagA_l = None
                P_l = None
                R_l = None
            self.levels.append(SALevel(A_l, P_l, R_l, invdiagA_l))

    def solve(self, b, x0=None, tol=1e-8, maxiter=100, cycle='V', presmooth=3, postsmooth=3):
        """
        solve Ax=b using smoothed aggregation AMG

        input:
        b: right hand side, torch tensor
        x0: initial guess, torch tensor
        tol: tolerance
        maxiter: maximum number of iterations

        output:
        x: solution, torch tensor. if x0 is given,
           x0 is modified in place. otherwise it is newly created.
        """
        if x0 is None:
            x0 = torch.zeros_like(b, device=config.dev, dtype=config.dtype)
        x = x0
        normb = torch.linalg.norm(b)
        if normb == 0:
            normb = 1.0

        it = 0

        while True:  # it <= maxiter and normr >= tol:
            if len(self.levels) == 1:
                # solve directly
                x = torch.linalg.inv(self.levels[0].A) @ b
            else:
                # V cycle
                self._solve_level(0, x, b, presmooth, postsmooth)  # x is changed in place
            it += 1
            normr = torch.linalg.norm(b - self.levels[0].A @ x)
            if normr < tol * normb:
                return x
            if it == maxiter:
                return x

    def _solve_level(self, lvl, x, b, presmooth=3, postsmooth=3):
        """Multigrid cycling.

        lvl: level
        x: initial guess and return correction
        b: right hand side on this level
        presmooth: number of presmoothing steps
        postsmooth: number of postsmoothing steps
        """
        A = self.levels[lvl].A
        Dinv = self.levels[lvl].invdiagA

        # presmooth
        for i in range(presmooth):
            x += self.jac_wei * Dinv * (b - A @ x)
        # get residual
        residual = b - A @ x
        # restrict residual
        coarse_b = self.levels[lvl].R @ residual
        coarse_x = torch.zeros_like(coarse_b)  # use 0 initial guess on next level

        # smooth on lvl-1 level
        if lvl == len(self.levels) - 2:
            # next level would be the last level, we'll direct solve on that
            coarse_x += torch.linalg.pinv(self.levels[lvl+1].A.to_dense()) @ coarse_b
        else:
            self._solve_level(lvl+1, coarse_x, coarse_b, presmooth, postsmooth)

        # prolongate correction and correct on this level
        x += self.levels[lvl].P @ coarse_x

        # post smooth
        for i in range(postsmooth):
            x += self.jac_wei * Dinv * (b - A @ x)


class SALevel:
    """
    store data for smoothed aggregation AMG solver on one level

    this includes:
    operators A,
    prolongator P,
    restrictor R,
    inverse of diagonal of A's
    """

    def __init__(self, A, P, R, invdiagA):
        self.A = A
        self.P = P
        self.R = R
        self.invdiagA = invdiagA

    def set_data(self, A, P, R, invdiagA):
        self.A = A
        self.P = P
        self.R = R
        self.invdiagA = invdiagA
