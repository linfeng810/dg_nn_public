"""matrix-free integral for mesh velocity
or mesh displacement
using linear elastic equation rather than diffusion equation"""

import torch
import numpy as np
import scipy as sp
import config
import materials
from config import sf_nd_nb
import multigrid_linearelastic as mg
from tqdm import tqdm

if config.ndim == 2:
    from shape_function import get_det_nlx as get_det_nlx
    from shape_function import sdet_snlx as sdet_snlx
else:
    from shape_function import get_det_nlx_3d as get_det_nlx
    from shape_function import sdet_snlx_3d as sdet_snlx

nele = config.nele
nele_f = config.nele_f
nele_s = config.nele_s
ndim = config.ndim
dev = config.dev
nface = ndim + 1


def solve_for_mesh_disp(
        x_i,
        t=None,
):
    """
    input: current non-linear step solution

    from the solid displacement at interface,
    solve a diffusion equation for the mesh
    displacement in the fluid subdomain

    then do a projection to make it C1 continuous

    x_i contains all displacement field (fluid + solid)
    we will separate them into x_i fluid subdomain to solve;
    and x_i solid subdomain as boundary condition
    """
    given_mesh_disp = False
    if given_mesh_disp:  # given mesh displacement to test ALE
        if ndim == 2:
            d = 0.125  # magnitude
            x_i *= 0
            x = sf_nd_nb.disp_func_space.x_ref_in[:, 0, :]
            y = sf_nd_nb.disp_func_space.x_ref_in[:, 1, :]
            x_i[:, :, 0] = 1 / 4 * torch.sin(2 * torch.pi * x) * (1 - torch.cos(2 * torch.pi * y)) \
                           * (1 - np.cos(2 * np.pi * t)) * d
            x_i[:, :, 1] = 1 / 4 * (1 - torch.cos(2 * torch.pi * x)) * torch.sin(2 * torch.pi * y) \
                           * (1 - np.cos(2 * np.pi * t)) * d
        else:
            raise NotImplementedError
        return x_i
    if torch.linalg.norm(x_i) < 1e-8:
        return x_i
    x_k = torch.zeros_like(x_i, device=dev, dtype=torch.float64)
    x_k += x_i
    x_i = _solve_diffusion(x_i, x_k)
    x_i = _project_to_make_continuous(x_i)
    x_i = _make_0_bc_strongly_enforced(x_i)
    x_i = x_i.view(nele, -1, ndim)
    return x_i


def _solve_diffusion(
        x_i, x_k
):
    """solving *linear elastic* equation for mesh displacement
    *not* diffusion equation."""
    x_i = x_i.view(nele, -1, ndim)
    x_rhs = torch.zeros_like(x_i, device=dev, dtype=torch.float64)
    # get RAR and SFC data
    _get_RAR_and_sfc_data_Um(x_k)
    # get rhs
    x_rhs = _get_rhs(x_rhs, x_i, x_k)

    if False:  # build lhs matrix column by column to test assemble
        nloc = sf_nd_nb.disp_func_space.element.nloc
        u_nonods = nele * nloc
        x_dummy = torch.zeros_like(x_i, device=dev, dtype=torch.float64).view(-1)
        if sf_nd_nb.Kmatinv is None:
            probe = torch.zeros_like(x_i, device=dev, dtype=torch.float64).view(-1)
            Amat = torch.zeros(nele * nloc * ndim, nele * nloc * ndim, device=dev, dtype=torch.float64)
            for inod in tqdm(range(nele * nloc)):
                for jdim in range(ndim):
                    x_dummy *= 0
                    probe *= 0
                    probe[jdim * u_nonods + inod] += 1
                    x_dummy, _ = get_residual_or_smooth(
                        r0=x_dummy,
                        x_i=probe,
                        x_rhs=0,
                        x_k=x_k,
                        do_smooth=False,
                    )
                    Amat[:, jdim * u_nonods + inod] -= x_dummy.view(-1)
            Amat_np = Amat[0:nele_f * nloc * ndim, 0:nele_f * nloc * ndim].cpu().numpy()
            Amat_sp = sp.sparse.csr_matrix(Amat_np)
            sf_nd_nb.Kmatinv = Amat_sp
        else:
            Amat_sp = sf_nd_nb.Kmatinv
        rhs = x_rhs.view(nele, nloc, ndim)[0:nele_f, :, :].cpu().numpy()
        x_sol = sp.sparse.linalg.spsolve(Amat_sp, rhs.reshape(-1))
        # print(x_sol)
        x_i = x_i.view(nele, -1, ndim)
        x_i[0:nele_f, ...] *= 0
        x_i[0:nele_f, ...] += torch.tensor(x_sol.reshape(nele_f, nloc, ndim), device=dev, dtype=torch.float64)
    # solve with left-preconditionerd GMRES
    x_i = x_i.view(nele, -1, ndim)
    x_i, its = _gmres_mg_solver(
        x_i, x_rhs, x_k, tol=1e-6
    )

    return x_i


def _project_to_make_fluid_continuous(
        x_i
):
    """
    make 0:nele_f of x_i[nele_f, nloc, ndim] C1 continuous
    """
    nloc = sf_nd_nb.disp_func_space.element.nloc
    x_i = x_i.view(nele, -1, ndim)
    x_i_f = np.zeros((nele_f, nloc, ndim), dtype=np.float64)
    x_i_f += x_i[0:nele_f, ...].cpu().numpy()
    pncg_nonods = sf_nd_nb.disp_func_space.pncg_nonods_f
    x_i_cg = np.zeros((pncg_nonods, ndim), dtype=np.float64)
    x_i_f = x_i_f.reshape((-1, ndim))
    for idim in range(ndim):
        x_i_cg[:, idim] += x_i_f[:, idim] @ sf_nd_nb.disp_func_space.pndg_ndglbno_f / \
                           np.array(sf_nd_nb.disp_func_space.pndg_ndglbno_f.sum(axis=0))[0, :]
        x_i_f[:, idim] *= 0
        x_i_f[:, idim] += sf_nd_nb.disp_func_space.pndg_ndglbno_f @ x_i_cg[:, idim]
    x_i[0:nele_f, ...] *= 0
    x_i[0:nele_f, ...] += torch.tensor(x_i_f, device=dev, dtype=torch.float64).view(nele_f, -1, ndim)
    return x_i


def _project_to_make_continuous(
        x_i
):
    """
    make whole domain x_i continuous

    at interface, honour solid displacement more by using large weight on solid side
    """
    nloc = sf_nd_nb.disp_func_space.element.nloc
    x_i = x_i.view(nele, -1, ndim)
    x_i_f = np.zeros((nele, nloc, ndim), dtype=np.float64)
    x_i_f += x_i.cpu().numpy()
    pncg_nonods = sf_nd_nb.disp_func_space.pncg_nonods
    x_i_cg = np.zeros((pncg_nonods, ndim), dtype=np.float64)
    x_i_f = x_i_f.reshape((-1, ndim))
    pndg_ndglbno = sf_nd_nb.disp_func_space.pndg_ndglbno
    pndg_ndglbno[nele_f * nloc:nele * nloc, :] *= 1000
    for idim in range(ndim):
        x_i_cg[:, idim] += x_i_f[:, idim] @ sf_nd_nb.disp_func_space.pndg_ndglbno / \
                           np.array(sf_nd_nb.disp_func_space.pndg_ndglbno.sum(axis=0))[0, :]
    pndg_ndglbno[nele_f * nloc:nele * nloc, :] /= 1000
    for idim in range(ndim):
        x_i_f[:, idim] *= 0
        x_i_f[:, idim] += sf_nd_nb.disp_func_space.pndg_ndglbno @ x_i_cg[:, idim]
    x_i *= 0
    x_i += torch.tensor(x_i_f, device=dev, dtype=torch.float64).view(nele, -1, ndim)
    return x_i


def _make_0_bc_strongly_enforced(x_i):
    """
    set displacement on fluid boundaries \partial\Omega_f
    as 0. We don't want any mesh displacement there.
    """
    nloc = sf_nd_nb.disp_func_space.element.nloc
    glb_bcface_type = sf_nd_nb.disp_func_space.glb_bcface_type
    idx_in_f = torch.zeros(nele * nface, dtype=torch.bool, device=dev)
    idx_in_f[0:nele_f * nface] = True

    F_b = torch.where(torch.logical_and(
        torch.logical_or(glb_bcface_type == 0,
                         glb_bcface_type == 1),
        idx_in_f))[0]  # boundary face (all boundaries of fluid subdomain, except interface)
    E_F_b = torch.floor_divide(F_b, nface)
    f_b = torch.remainder(F_b, nface)
    strong_bc_node_idx = torch.zeros(nele * nloc, dtype=torch.bool, device=dev)
    for iface in range(nface):
        idx_iface = f_b == iface
        if torch.sum(idx_iface) == 0:  # nothing to do here, continue
            continue
        sn = sf_nd_nb.disp_func_space.element.sn[iface, ...]
        local_node_idx = (torch.nonzero(sn.sum(-1) != 0)).reshape(-1)
        strong_bc_node_idx[
            (E_F_b[idx_iface].unsqueeze(-1).expand(-1, local_node_idx.shape[0]) * nloc + local_node_idx).view(-1)
        ] = True
        # x_i_0_bc[E_F_b[idx_iface]] += torch.einsum(
        #     'bni,bn->bni',
        #     x_i[E_F_b[idx_iface], ...],  # (batch_in, u_nloc, ndim)
        #     (sn.sum(-1) == 0).to(torch.int64),  # (batch_in, u_nloc)
        # )
    x_i = x_i.view(-1, ndim)
    x_i[strong_bc_node_idx, :] *= 0
    # x_i *= 0
    # x_i += x_i_0_bc
    return x_i


def _get_rhs(
        x_rhs, x_i, x_k
):
    """
    get rhs of the diffusion equation (interface
    displacement on solid side as a dirichlet bc)

    input:

    x_i has length (nele * u_nloc * ndim)
    i.e. it contains solid displacement

    x_rhs has length (nele_f * u_nloc * ndim)
    i.e. only in fluid displacement
    """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele_f, dtype=int)
    # idx_in = torch.zeros(nele, dtype=torch.bool)
    idx_in_f = torch.zeros(nele * nface, dtype=torch.bool, device=dev)

    # change view
    u_nloc = sf_nd_nb.disp_func_space.element.nloc

    for i in range(nnn):
        # idx_in *= False
        # # volume integral
        # idx_in[brk_pnt[i]:brk_pnt[i + 1]] = True
        # x_rhs = _k_rhs_one_batch(x_rhs, x_i, idx_in)
        # surface integral
        idx_in_f *= False
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        x_rhs = _s_rhs_one_batch(x_rhs, x_i, x_k, idx_in_f)

    return x_rhs


def _k_rhs_one_batch(
        x_rhs, x_k, idx_in
):
    batch_in = int(torch.sum(idx_in))
    if batch_in < 1:  # nothing to do here.
        return x_rhs
    nloc = sf_nd_nb.disp_func_space.element.nloc
    n = sf_nd_nb.disp_func_space.element.n
    nx, detwei = get_det_nlx(
        nlx=sf_nd_nb.disp_func_space.element.nlx,
        x_loc=sf_nd_nb.disp_func_space.x_ref_in[idx_in],
        weight=sf_nd_nb.disp_func_space.element.weight,
        nloc=nloc,
        ngi=sf_nd_nb.disp_func_space.element.ngi
    )

    # Nxi Nj P
    mu_th = torch.einsum('ng,bn->bg', n,
                         sf_nd_nb.mesh_material.mu[idx_in, ...])
    lam_th = torch.einsum('ng,bn->bg', n,
                          sf_nd_nb.mesh_material.lam[idx_in, ...])
    P = sf_nd_nb.mesh_material.calc_P(
        nx=nx,
        u=x_k.view(nele, nloc, ndim)[idx_in, ...],
        batch_in=batch_in,
        mu_gi=mu_th,
        lam_gi=lam_th,
    )  # PK1 stress evaluated at current state u
    x_rhs[idx_in, ...] -= torch.einsum(
        'bijg,bjmg,bg->bmi',  # i,j is idim and jdim; m, n is mloc and nloc
        P,  # (batch_in, ndim, ndim, ngi)
        # n,  # (nloc, ngi)
        nx,  # (batch_in, ndim, nloc, ngi)
        detwei,  # (batch_in, ngi)
    )
    return x_rhs


def _s_rhs_one_batch(
        x_rhs, x_i, x_k, idx_in_f
):
    # get essential data
    nbf = sf_nd_nb.disp_func_space.nbf
    alnmt = sf_nd_nb.disp_func_space.alnmt
    glb_bcface_type = sf_nd_nb.disp_func_space.glb_bcface_type

    # change view
    u_nloc = sf_nd_nb.disp_func_space.element.nloc

    # separate nbf to get internal face list and boundary face list
    F_i = torch.where(torch.logical_and(glb_bcface_type < 0,
                                        idx_in_f))[0]  # interior face of solid subdomain
    F_inb = nbf[F_i]  # neighbour list of interior face
    F_inb = F_inb.type(torch.int64)
    # interface contribution from solid side to the fluid interface
    F_itf = torch.where(torch.logical_and(
        sf_nd_nb.disp_func_space.glb_bcface_type == 4,
        idx_in_f))[0]  # boundary face  # (interface)
    F_itf_nb = nbf[F_itf].type(torch.int64)  # neighbour list of interface face

    # create two lists of which element f_i / f_b is in
    E_F_i = torch.floor_divide(F_i, nface)
    E_F_inb = torch.floor_divide(F_inb, nface)
    E_F_itf = torch.floor_divide(F_itf, nface)
    E_F_itf_nb = torch.floor_divide(F_itf_nb, nface)

    # local face number
    f_i = torch.remainder(F_i, nface)
    f_inb = torch.remainder(F_inb, nface)
    f_itf = torch.remainder(F_itf, nface)
    f_itf_nb = torch.remainder(F_itf_nb, nface)

    # # interior face term and interface (because interface has neighbour and needs nb_gi_aln)
    # for iface in range(nface):
    #     for nb_gi_aln in range(nface - 1):
    #         idx_iface = (f_i == iface) & (alnmt[F_i] == nb_gi_aln)
    #         if idx_iface.sum() > 0:
    #             x_rhs = _s_rhs_fi(
    #                 x_rhs, f_i[idx_iface], E_F_i[idx_iface],
    #                 f_inb[idx_iface], E_F_inb[idx_iface],
    #                 x_k,
    #                 nb_gi_aln)

    for iface in range(nface):
        idx_iface_itf = f_itf == iface
        x_rhs = _s_rhs_fitf(
            x_rhs,
            f_itf[idx_iface_itf], E_F_itf[idx_iface_itf],
            f_itf_nb[idx_iface_itf], E_F_itf_nb[idx_iface_itf],
            x_i, x_k
        )  # rhs terms from interface
    return x_rhs


def _s_rhs_fi(
        rhs,
        f_i, E_F_i,
        f_inb, E_F_inb,
        u,
        nb_gi_aln
):
    batch_in = f_i.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    nloc = sf_nd_nb.disp_func_space.element.nloc
    # shape function on this side
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.disp_func_space.element.snlx,
        x_loc=sf_nd_nb.disp_func_space.x_ref_in[E_F_i],
        sweight=sf_nd_nb.disp_func_space.element.sweight,
        nloc=sf_nd_nb.disp_func_space.element.nloc,
        sngi=sf_nd_nb.disp_func_space.element.sngi,
        sn=sf_nd_nb.disp_func_space.element.sn,
    )
    sn = sf_nd_nb.disp_func_space.element.sn[f_i, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_i, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_i, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_i, ...]  # (batch_in, ndim, sngi)

    # shape function on the other side
    snx_nb, _, snormal_nb = sdet_snlx(
        snlx=sf_nd_nb.disp_func_space.element.snlx,
        x_loc=sf_nd_nb.disp_func_space.x_ref_in[E_F_inb],
        sweight=sf_nd_nb.disp_func_space.element.sweight,
        nloc=sf_nd_nb.disp_func_space.element.nloc,
        sngi=sf_nd_nb.disp_func_space.element.sngi,
        sn=sf_nd_nb.disp_func_space.element.sn,
    )
    # change gaussian points order
    nb_aln = sf_nd_nb.disp_func_space.element.gi_align[nb_gi_aln, :]
    snx_nb = snx_nb[..., nb_aln]
    # fetch faces we want
    sn_nb = sf_nd_nb.disp_func_space.element.sn[f_inb, ...]  # (batch_in, nloc, sngi)
    snx_nb = snx_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim, nloc, sngi)
    snormal_nb = snormal_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim, sngi)
    # don't forget to change gaussian points order on sn_nb!
    sn_nb = sn_nb[..., nb_aln]
    snormal_nb = snormal_nb[..., nb_aln]

    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = config.eta_e / h
    u_i = u[E_F_i, ...]  # u^n on this side
    u_inb = u[E_F_inb, ...]  # u^n on the other side

    # [vi nj]{P^n_kl} term
    # P = sf_nd_nb.mesh_material.calc_P(nx=snx, u=u_i, batch_in=batch_in)
    # P_nb = sf_nd_nb.mesh_material.calc_P(nx=snx_nb, u=u_inb, batch_in=batch_in)
    # P *= 0.5
    # P_nb *= 0.5
    # P += P_nb  # this is {P^n} = 1/2 (P^1 + P^2)  average on both sides
    # new average P = P({F})
    mu_th = torch.einsum('bng,bn->bg', sn,
                         sf_nd_nb.mesh_material.mu[E_F_i, ...])
    lam_th = torch.einsum('bng,bn->bg', sn,
                          sf_nd_nb.mesh_material.lam[E_F_i, ...])
    mu_nb = torch.einsum('bng,bn->bg', sn,
                         sf_nd_nb.mesh_material.mu[E_F_inb, ...])
    lam_nb = torch.einsum('bng,bn->bg', sn,
                          sf_nd_nb.mesh_material.lam[E_F_inb, ...])
    P = sf_nd_nb.mesh_material.calc_P_ave(nx=snx, u=u_i, nx_nb=snx_nb, u_nb=u_inb, batch_in=batch_in,
                                          mu_gi_th=mu_th, lam_gi_th=lam_th,
                                          mu_gi_nb=mu_nb, lam_gi_nb=lam_nb)
    # this side + other side
    rhs[E_F_i, ...] += torch.einsum(
        'bmg,bjg,bijg,bg->bmi',  # i,j is idim/jdim; m, n is mloc/nloc
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        # sn,  # (batch_in, nloc, sngi)
        P,  # (batch_in, ndim, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
    )
    # del P, P_nb
    del P

    # [ui nj] {A (\nabla v)_kl] term
    # AA = sf_nd_nb.mesh_material.calc_AA(nx=snx, u=u_i, batch_in=batch_in)
    # use new AA = A({F})
    mu_th = torch.einsum('bng,bn->bg', sn,
                         sf_nd_nb.mesh_material.mu[E_F_i, ...])
    lam_th = torch.einsum('bng,bn->bg', sn,
                          sf_nd_nb.mesh_material.lam[E_F_i, ...])
    mu_nb = torch.einsum('bng,bn->bg', sn,
                         sf_nd_nb.mesh_material.mu[E_F_inb, ...])
    lam_nb = torch.einsum('bng,bn->bg', sn,
                          sf_nd_nb.mesh_material.lam[E_F_inb, ...])
    AA = sf_nd_nb.mesh_material.calc_AA_ave(nx=snx, u=u_i, nx_nb=snx_nb, u_nb=u_inb, batch_in=batch_in,
                                            mu_gi_th=mu_th, lam_gi_th=lam_th,
                                            mu_gi_nb=mu_nb, lam_gi_nb=lam_nb)
    # this side + other side
    rhs[E_F_i, ...] += torch.einsum(
        'bnijg,bijklg,blmg,bg->bmk',  # i/j : idim/jdim; m/n: mloc/nloc
        (
                torch.einsum('bni,bng,bjg->bnijg', u_i, sn, snormal)
                + torch.einsum('bni,bng,bjg->bnijg', u_inb, sn_nb, snormal_nb)
        ),  # (batch_in, nloc, ndim, ndim, sngi)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei  # (batch_in, sngi)
    ) * 0.5

    # penalty term
    # \gamma_e [vi nj] A [uk nl]
    # # this A is 1/2(A_this + A_nb)
    # AA += sf_nd_nb.mesh_material.calc_AA(nx=snx_nb, u=u_inb, batch_in=batch_in)
    # AA *= 0.5
    rhs[E_F_i, ...] -= torch.einsum(
        'b,bmg,bjg,bijklg,bnklg,bg->bmi',
        gamma_e,  # (batch_in)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        # config.cijkl,
        (
                torch.einsum('bnk,bng,blg->bnklg', u_i, sn, snormal)
                + torch.einsum('bnk,bng,blg->bnklg', u_inb, sn_nb, snormal_nb)
        ),  # (batch_in, nloc, ndim, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
    )
    return rhs


def _s_rhs_fitf(
        rhs,
        f_itf, E_F_itf,
        f_itf_nb, E_F_itf_nb,
        x_i, x_k
):
    batch_in = f_itf.shape[0]
    if batch_in < 1:  # nothing to do here.
        return rhs
    u_nloc = sf_nd_nb.disp_func_space.element.nloc
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    rhs = rhs.view(nele, u_nloc, ndim)
    x_i = x_i.view(nele, u_nloc, ndim)
    x_k = x_k.view(nele, u_nloc, ndim)

    # shape function
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.disp_func_space.element.snlx,
        x_loc=sf_nd_nb.disp_func_space.x_ref_in[E_F_itf],
        sweight=sf_nd_nb.disp_func_space.element.sweight,
        nloc=sf_nd_nb.disp_func_space.element.nloc,
        sngi=sf_nd_nb.disp_func_space.element.sngi,
        sn=sf_nd_nb.disp_func_space.element.sn,
    )
    sn = sf_nd_nb.disp_func_space.element.sn[f_itf, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_itf, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_itf, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_itf, ...]  # (batch_in, ndim, sngi)
    # neighbour face shape function
    sn_nb = sf_nd_nb.disp_func_space.element.sn[f_itf_nb, ...]
    # sn_nb = torch.zeros_like(sn, device=dev, dtype=torch.float64)
    for nb_gi_aln in range(ndim):  # 'ndim' alignnment of GI points on neighbour faces
        idx = sf_nd_nb.disp_func_space.alnmt[E_F_itf * nface + f_itf] == nb_gi_aln
        nb_aln = sf_nd_nb.disp_func_space.element.gi_align[nb_gi_aln, :]
        sn_nb[idx] = sn_nb[idx][..., nb_aln]
    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = config.eta_e / h
    u_s_nb = x_i[E_F_itf_nb, ...]
    u_k_nb = x_k[E_F_itf_nb, ...]
    u_k_th = x_k[E_F_itf, ...]

    # get elasticity tensor at face quadrature points
    # (batch_in, ndim, ndim, ndim, ndim, sngi)
    mu_th = torch.einsum('bng,bn->bg', sn,
                         sf_nd_nb.mesh_material.mu[E_F_itf, ...])
    lam_th = torch.einsum('bng,bn->bg', sn,
                          sf_nd_nb.mesh_material.lam[E_F_itf, ...])
    AA = sf_nd_nb.mesh_material.calc_AA(nx=snx, u=u_k_th, batch_in=batch_in,
                                        mu_gi=mu_th, lam_gi=lam_th)

    # u_Di nj A \delta v_kl
    rhs[E_F_itf, ...] -= torch.einsum(
        'bni,bng,bjg,bijklg,blmg,bg->bmk',  # could easily by wrong...
        u_s_nb,  # (batch_in, nloc, ndim)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch, sngi)
    )
    # penalty term
    # gamma_e v_i n_j A u_Dk n_l
    rhs[E_F_itf, ...] += torch.einsum(
        'b,bmg,bjg,bijklg,bng,blg,bnk,bg->bmi',  # again could easily be wrong...
        gamma_e,  # (batch_in)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        # config.cijkl,
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        u_s_nb,  # (batch_in, nloc, ndim)  # TODO: could be u_bc - u_i or sth like that to avoid 2 einsums
        sdetwei,  # (batch_in, sngi
    )

    # # add boundary contribution from lhs. (last 3 terms in eq 60c)
    # # u_i n_j A \nabla v_kl
    # rhs[E_F_itf, ...] += torch.einsum(
    #     'bni,bng,bjg,bijklg,blmg,bg->bmk',  # could easily by wrong...
    #     u_k_th,  # (batch_in, nloc, ndim)
    #     sn,  # (batch_in, nloc, sngi)
    #     snormal,  # (batch_in, ndim, sngi)
    #     AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
    #     snx,  # (batch_in, ndim, nloc, sngi)
    #     sdetwei,  # (batch, sngi)
    # )
    # # penalty term
    # # \gamma_e v_i n_j A u_k n_l
    # rhs[E_F_itf, ...] -= torch.einsum(
    #     'b,bmg,bjg,bijklg,bng,blg,bnk,bg->bmi',  # again could easily be wrong...
    #     gamma_e,  # (batch_in)
    #     sn,  # (batch_in, nloc, sngi)
    #     snormal,  # (batch_in, ndim, sngi)
    #     AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
    #     # config.cijkl,
    #     sn,  # (batch_in, nloc, sngi)
    #     snormal,  # (batch_in, ndim, sngi)
    #     u_k_th,  # (batch_in, nloc, ndim)
    #     sdetwei,  # (batch_in, sngi
    # )
    # del AA  # no longer need
    # mu_th = torch.einsum('bng,bn->bg', sn,
    #                      sf_nd_nb.mesh_material.mu[E_F_itf, ...])
    # lam_th = torch.einsum('bng,bn->bg', sn,
    #                       sf_nd_nb.mesh_material.lam[E_F_itf, ...])
    # P = sf_nd_nb.mesh_material.calc_P(nx=snx, u=u_k_th, batch_in=batch_in,
    #                                   mu_gi=mu_th, lam_gi=lam_th)
    # # [v_i n_j] {P_ij}
    # rhs[E_F_itf, ...] += torch.einsum(
    #     'bmg,bjg,bijg,bg->bmi',
    #     sn,  # (batch_in, nloc, sngi)
    #     snormal,  # (batch_in, ndim, sngi)
    #     P,  # (batch_in, ndim, ndim, sngi)
    #     sdetwei,  # (batch_in, sngi)
    # )
    return rhs


def get_residual_or_smooth(
        r0, x_i, x_rhs, x_k,
        do_smooth=False,
):
    """
    matrix free, integral, for terms in the diffusion equation
    """
    nnn = config.no_batch
    brk_pnt = np.asarray(np.arange(0, nnn + 1) / nnn * nele_f, dtype=int)
    u_nloc = sf_nd_nb.disp_func_space.element.nloc
    r0 = r0.view(nele, u_nloc, ndim)
    if type(x_rhs) is int:
        r0 += x_rhs
    else:
        x_rhs = x_rhs.view(nele, u_nloc, ndim)
        r0 += x_rhs
    for i in range(nnn):
        # volume integral
        idx_in = torch.zeros(nele, device=dev, dtype=torch.bool)  # element indices in this batch
        idx_in[brk_pnt[i]:brk_pnt[i + 1]] = True
        batch_in = int(torch.sum(idx_in))
        # dummy diagA and bdiagA
        diagK = torch.zeros(batch_in, u_nloc, ndim, device=dev, dtype=torch.float64)
        bdiagK = torch.zeros(batch_in, u_nloc, ndim, u_nloc, ndim, device=dev, dtype=torch.float64)
        r0, diagK, bdiagK = _k_res_one_batch(
            r0, x_i, x_k,
            diagK, bdiagK,
            idx_in
        )
        # surface integral
        idx_in_f = torch.zeros(nele * nface, dtype=torch.bool, device=dev)
        idx_in_f[brk_pnt[i] * nface:brk_pnt[i + 1] * nface] = True
        r0, diagK, bdiagK = _s_res_one_batch(
            r0, x_i, x_k,
            diagK, bdiagK,
            idx_in_f, brk_pnt[i]
        )
        if do_smooth:
            # smooth once
            bdiagK = torch.inverse(bdiagK.view(batch_in, u_nloc * ndim, u_nloc * ndim))
            x_i = x_i.view(nele, u_nloc, ndim)
            x_i[idx_in, :] += config.jac_wei * torch.einsum(
                '...ij,...j->...i',
                bdiagK,
                r0.view(nele, u_nloc * ndim)[idx_in, :]
            ).view(batch_in, u_nloc, ndim)
    r0 = r0.view(-1)
    x_i = x_i.view(-1)
    return r0, x_i


def _k_res_one_batch(
        r0, x_i, x_k,
        diagK, bdiagK,
        idx_in,
):
    batch_in = diagK.shape[0]
    u_nloc = sf_nd_nb.disp_func_space.element.nloc

    r0 = r0.view(nele, u_nloc, ndim)
    x_i = x_i.view(nele, u_nloc, ndim)
    x_k = x_k.view(nele, u_nloc, ndim)
    diagK = diagK.view(-1, u_nloc, ndim)
    bdiagK = bdiagK.view(-1, u_nloc, ndim, u_nloc, ndim)

    # get shape function and derivatives
    n = sf_nd_nb.disp_func_space.element.n
    nx, ndetwei = get_det_nlx(
        nlx=sf_nd_nb.disp_func_space.element.nlx,
        x_loc=sf_nd_nb.disp_func_space.x_ref_in[idx_in],
        weight=sf_nd_nb.disp_func_space.element.weight,
        nloc=u_nloc,
        ngi=sf_nd_nb.disp_func_space.element.ngi
    )
    mu_th = torch.einsum('ng,bn->bg', n,
                         sf_nd_nb.mesh_material.mu[idx_in, ...])
    lam_th = torch.einsum('ng,bn->bg', n,
                          sf_nd_nb.mesh_material.lam[idx_in, ...])
    AA = sf_nd_nb.mesh_material.calc_AA(nx=nx,
                                        u=x_k[idx_in, ...], batch_in=batch_in,
                                        mu_gi=mu_th, lam_gi=lam_th)
    K = torch.zeros(batch_in, u_nloc, ndim, u_nloc, ndim, device=dev, dtype=torch.float64)
    K += torch.einsum('bjmg,bijklg,blng,bg->bmink', nx, AA, nx, ndetwei)
    r0[idx_in, ...] -= torch.einsum(
        'bminj,bnj->bmi',
        K, x_i[idx_in, ...]
    )
    # get diagonal of velocity block K
    diagK += torch.diagonal(K.view(batch_in, u_nloc * ndim, u_nloc * ndim)
                            , dim1=1, dim2=2).view(batch_in, u_nloc, ndim)
    bdiagK[idx_in[0:nele_f], ...] += K

    return r0, diagK, bdiagK


def _s_res_one_batch(
        r0, x_i, x_k,
        diagK, bdiagK,
        idx_in_f,
        batch_start_idx
):
    """
    surface integral (left hand side)
    """
    nbf = sf_nd_nb.disp_func_space.nbf
    glb_bcface_type = sf_nd_nb.disp_func_space.glb_bcface_type

    # change view
    u_nloc = sf_nd_nb.disp_func_space.element.nloc

    # separate nbf to get internal face list and boundary face list
    F_i = torch.where(torch.logical_and(glb_bcface_type < 0,
                                        idx_in_f))[0]  # interior face of fluid subdomain
    F_inb = nbf[F_i]  # neighbour list of interior face
    F_inb = F_inb.type(torch.int64)
    F_b = torch.where(torch.logical_and(
        torch.logical_or(sf_nd_nb.disp_func_space.glb_bcface_type == 0,
                         torch.logical_or(
                             sf_nd_nb.disp_func_space.glb_bcface_type == 1,
                             sf_nd_nb.disp_func_space.glb_bcface_type == 4)),
        idx_in_f))[0]  # boundary face (all boundaries of fluid subdomain, including interface)
    # F_itf = torch.where(torch.logical_and(
    #     sf_nd_nb.disp_func_space.glb_bcface_type == 4,
    #     idx_in_f))[0]  # boundary face  # (interface)
    # F_itf_nb = nbf[F_itf].type(torch.int64)  # neighbour list of interface face

    # create two lists of which element f_i / f_b is in
    E_F_i = torch.floor_divide(F_i, nface)
    E_F_inb = torch.floor_divide(F_inb, nface)
    E_F_b = torch.floor_divide(F_b, nface)
    # E_F_itf = torch.floor_divide(F_itf, nface)
    # E_F_itf_nb = torch.floor_divide(F_itf_nb, nface)

    # local face number
    f_b = torch.remainder(F_b, nface)
    f_i = torch.remainder(F_i, nface)
    f_inb = torch.remainder(F_inb, nface)
    # f_itf = torch.remainder(F_itf, nface)
    # f_itf_nb = torch.remainder(F_itf_nb, nface)

    # for interior faces
    for iface in range(nface):
        for nb_gi_aln in range(nface - 1):
            idx_iface = (f_i == iface) & (sf_nd_nb.disp_func_space.alnmt[F_i] == nb_gi_aln)
            if idx_iface.sum() < 1:
                # there is nothing to do here, go on
                continue
            r0, diagK, bdiagK = _s_res_fi(
                r0, f_i[idx_iface], E_F_i[idx_iface],
                f_inb[idx_iface], E_F_inb[idx_iface],
                x_i, x_k,
                diagK, bdiagK, batch_start_idx,
                nb_gi_aln,
            )
    # boundary faces (fluid domain boundary and fluid interface)
    for iface in range(nface):
        idx_iface = f_b == iface
        r0, diagK, bdiagK = _s_res_fb(
            r0, f_b[idx_iface], E_F_b[idx_iface],
            x_i, x_k,
            diagK, bdiagK, batch_start_idx,
        )
    return r0, diagK, bdiagK


def _s_res_fi(
        r0, f_i, E_F_i,
        f_inb, E_F_inb,
        x_i, x_k,
        diagK, bdiagK, batch_start_idx,
        nb_gi_aln,
):
    """internal faces"""
    batch_in = f_i.shape[0]
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    # get element parameters
    u_nloc = sf_nd_nb.disp_func_space.element.nloc
    r0 = r0.view(nele, u_nloc, ndim)
    x_i = x_i.view(nele, u_nloc, ndim)
    x_k = x_k.view(nele, u_nloc, ndim)

    # shape function on this side
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.disp_func_space.element.snlx,
        x_loc=sf_nd_nb.disp_func_space.x_ref_in[E_F_i],
        sweight=sf_nd_nb.disp_func_space.element.sweight,
        nloc=sf_nd_nb.disp_func_space.element.nloc,
        sngi=sf_nd_nb.disp_func_space.element.sngi,
        sn=sf_nd_nb.disp_func_space.element.sn,
    )
    sn = sf_nd_nb.disp_func_space.element.sn[f_i, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_i, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_i, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_i, ...]  # (batch_in, ndim, sngi)

    # shape function on the other side
    snx_nb, _, snormal_nb = sdet_snlx(
        snlx=sf_nd_nb.disp_func_space.element.snlx,
        x_loc=sf_nd_nb.disp_func_space.x_ref_in[E_F_inb],
        sweight=sf_nd_nb.disp_func_space.element.sweight,
        nloc=sf_nd_nb.disp_func_space.element.nloc,
        sngi=sf_nd_nb.disp_func_space.element.sngi,
        sn=sf_nd_nb.disp_func_space.element.sn,
    )
    # get faces we want
    sn_nb = sf_nd_nb.disp_func_space.element.sn[f_inb, ...]  # (batch_in, nloc, sngi)
    snx_nb = snx_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim, nloc, sngi)
    snormal_nb = snormal_nb[dummy_idx, f_inb, ...]  # (batch_in, ndim, sngi)
    # change gaussian points order on other side
    nb_aln = sf_nd_nb.disp_func_space.element.gi_align[nb_gi_aln, :]  # nb_aln for velocity element
    snx_nb = snx_nb[..., nb_aln]
    snormal_nb = snormal_nb[..., nb_aln]
    # don't forget to change gaussian points order on sn_nb!
    sn_nb = sn_nb[..., nb_aln]
    nb_aln = sf_nd_nb.pre_func_space.element.gi_align[nb_gi_aln, :]  # nb_aln for pressure element

    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = config.eta_e / h

    u_ith = x_i[E_F_i, ...]
    u_inb = x_i[E_F_inb, ...]
    u_kth = x_k[E_F_i, ...]
    u_knb = x_k[E_F_inb, ...]

    mu_th = torch.einsum('bng,bn->bg', sn,
                         sf_nd_nb.mesh_material.mu[E_F_i, ...])
    lam_th = torch.einsum('bng,bn->bg', sn,
                          sf_nd_nb.mesh_material.lam[E_F_i, ...])
    mu_nb = torch.einsum('bng,bn->bg', sn,
                         sf_nd_nb.mesh_material.mu[E_F_inb, ...])
    lam_nb = torch.einsum('bng,bn->bg', sn,
                          sf_nd_nb.mesh_material.lam[E_F_inb, ...])
    AA = sf_nd_nb.mesh_material.calc_AA_ave(nx=snx, u=u_kth, nx_nb=snx_nb, u_nb=u_knb, batch_in=batch_in,
                                            mu_gi_th=mu_th, lam_gi_th=lam_th, mu_gi_nb=mu_nb, lam_gi_nb=lam_nb)
    # K block
    K = torch.zeros(batch_in, u_nloc, ndim, u_nloc, ndim, device=dev, dtype=torch.float64)
    # this side
    # [vi nj] {A \nabla u_kl}
    K += torch.einsum(
        'bmg,bjg,bijklg,blng,bg->bmink',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        # AA_th,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        AA,  # use AA_ave, i.e. A({F})
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (-0.5)
    # [ui nj] {A \nabla v_kl}
    K += torch.einsum(
        'bng,bjg,bijklg,blmg,bg->bmkni',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        # AA_th,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        AA,  # use AA_ave, i.e. A({F})
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (-0.5)
    # penalty term
    # \gamma_e [v_i n_j]{A}
    K += torch.einsum(
        'b,bmg,bjg,bijklg,bng,blg,bg->bmink',
        gamma_e,  # (batch_in)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        # 0.5 * (AA_th + AA_nb),  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        AA,  # use AA_ave = A({F}), {F} is average F on the face
        # config.cijkl,
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
    )

    # update residual
    r0[E_F_i, ...] -= torch.einsum('bminj,bnj->bmi', K, u_ith)
    # put diagonal into diagK and bdiagK
    diagK[E_F_i - batch_start_idx, :, :] += torch.diagonal(K.view(batch_in, u_nloc * ndim, u_nloc * ndim),
                                                           dim1=1, dim2=2).view(batch_in, u_nloc, ndim)
    bdiagK[E_F_i - batch_start_idx, :, :] += K

    # other side
    K *= 0
    # [vi nj] {A \nabla u_kl}
    K += torch.einsum(
        'bmg,bjg,bijklg,blng,bg->bmink',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        # AA_nb,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        AA,  # use AA_ave, i.e. A({F})
        snx_nb,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (-0.5)
    # [ui nj] {A \nabla v_kl}
    K += torch.einsum(
        'bng,bjg,bijklg,blmg,bg->bmkni',
        sn_nb,  # (batch_in, nloc, sngi)
        snormal_nb,  # (batch_in, ndim, sngi)
        # AA_th,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        AA,  # use AA_ave, i.e. A({F})
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    ) * (-0.5)
    # penalty term
    # \gamma_e [v_i n_j]{A}
    K += torch.einsum(  # FIXME: here it should be S-= ... ? Please make sure.
        'b,bmg,bjg,bijklg,bng,blg,bg->bmink',
        gamma_e,  # (batch_in)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        # 0.5 * (AA_th + AA_nb),  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        AA,  # use AA_ave, i.e. A({F})
        # config.cijkl,
        sn_nb,  # (batch_in, nloc, sngi)
        snormal_nb,  # (batch_in, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
    )

    # update residual
    r0[E_F_i, ...] -= torch.einsum('bminj,bnj->bmi', K, u_inb)
    return r0, diagK, bdiagK


def _s_res_fb(
        r0, f_b, E_F_b,
        x_i, x_k,
        diagK, bdiagK,
        batch_start_idx,
):
    """boundary faces"""
    batch_in = f_b.shape[0]
    if batch_in < 1:  # nothing to do here.
        return r0, diagK, bdiagK
    dummy_idx = torch.arange(0, batch_in, device=dev, dtype=torch.int64)
    # get element parameters
    u_nloc = sf_nd_nb.disp_func_space.element.nloc
    x_i = x_i.view(nele, u_nloc, ndim)
    x_k = x_k.view(nele, u_nloc, ndim)
    r0 = r0.view(nele, u_nloc, ndim)
    # shape function
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.disp_func_space.element.snlx,
        x_loc=sf_nd_nb.disp_func_space.x_ref_in[E_F_b],
        sweight=sf_nd_nb.disp_func_space.element.sweight,
        nloc=sf_nd_nb.disp_func_space.element.nloc,
        sngi=sf_nd_nb.disp_func_space.element.sngi,
        sn=sf_nd_nb.disp_func_space.element.sn,
    )
    sn = sf_nd_nb.disp_func_space.element.sn[f_b, ...]  # (batch_in, nloc, sngi)
    snx = snx[dummy_idx, f_b, ...]  # (batch_in, ndim, nloc, sngi)
    sdetwei = sdetwei[dummy_idx, f_b, ...]  # (batch_in, sngi)
    snormal = snormal[dummy_idx, f_b, ...]  # (batch_in, ndim, sngi)
    h = torch.sum(sdetwei, -1)
    if ndim == 3:
        h = torch.sqrt(h)
    gamma_e = config.eta_e / h

    # get elasticity tensor at face quadrature points
    # (batch_in, ndim, ndim, ndim, ndim, sngi)
    mu_th = torch.einsum('bng,bn->bg', sn,
                         sf_nd_nb.mesh_material.mu[E_F_b, ...])
    lam_th = torch.einsum('bng,bn->bg', sn,
                          sf_nd_nb.mesh_material.lam[E_F_b, ...])
    AA = sf_nd_nb.mesh_material.calc_AA(nx=snx, u=x_k[E_F_b, ...], batch_in=batch_in,
                                        mu_gi=mu_th, lam_gi=lam_th)

    u_ith = x_i[E_F_b, ...]
    # block K
    K = torch.zeros(batch_in, u_nloc, ndim, u_nloc, ndim,
                    device=dev, dtype=torch.float64)
    # [v_i n_j] {A \nabla u_kl}
    K -= torch.einsum(
        'bmg,bjg,bijklg,blng,bg->bmink',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    )
    # [u_i n_j] {A \nabla v_kl}
    K -= torch.einsum(
        'bng,bjg,bijklg,blmg,bg->bmkni',
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        snx,  # (batch_in, ndim, nloc, sngi)
        sdetwei,  # (batch_in, sngi)
    )
    # penalty term
    # \gamma_e [v_i n_j] {A} [u_k n_l]
    K += torch.einsum(
        'b,bmg,bjg,bijklg,bng,blg,bg->bmink',
        gamma_e,  # (batch_in)
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        AA,  # (batch_in, ndim, ndim, ndim, ndim, sngi)
        # config.cijkl,
        sn,  # (batch_in, nloc, sngi)
        snormal,  # (batch_in, ndim, sngi)
        sdetwei,  # (batch_in, sngi)
    )

    # update residual
    r0[E_F_b, ...] -= torch.einsum('bminj,bnj->bmi', K, u_ith)
    # put in diagonal
    diagK[E_F_b - batch_start_idx, :, :] += torch.diagonal(K.view(batch_in, u_nloc * ndim, u_nloc * ndim),
                                                           dim1=-2, dim2=-1).view(batch_in, u_nloc, ndim)
    bdiagK[E_F_b - batch_start_idx, ...] += K
    return r0, diagK, bdiagK


def _get_RAR_and_sfc_data_Um(x_k):
    """
    get RAR and coarser grid operator for mesh displacement
    """
    print('=== get RAR and sfc_data for Um ===')
    I_fc = sf_nd_nb.sparse_f.I_fc
    I_cf = sf_nd_nb.sparse_f.I_cf
    whichc = sf_nd_nb.sparse_f.whichc
    ncolor = sf_nd_nb.sparse_f.ncolor
    fina = sf_nd_nb.sparse_f.fina
    cola = sf_nd_nb.sparse_f.cola
    ncola = sf_nd_nb.sparse_f.ncola
    cg_nonods = sf_nd_nb.sparse_f.cg_nonods

    RARvalues = _calc_RAR_mf_color(
        x_k,
        I_fc, I_cf,
        whichc, ncolor,
        fina, cola, ncola,
    )
    from scipy.sparse import bsr_matrix

    if config.mg_opt_Um == 1:
        RAR = bsr_matrix((RARvalues.cpu().numpy(), cola, fina),
                         shape=((ndim) * cg_nonods, (ndim) * cg_nonods))
        sf_nd_nb.set_data(RARmat_Um=RAR.tocsr())
    elif config.mg_opt_Um == 3:
        RAR = bsr_matrix((RARvalues.cpu().numpy(), cola, fina),
                         shape=((ndim) * cg_nonods, (ndim) * cg_nonods))
        RAR = bsr_matrix((RARvalues.cpu().numpy(), cola, fina),
                         shape=((ndim) * cg_nonods, (ndim) * cg_nonods))
        # RAR_ml = pyamg.ruge_stuben_solver(RAR.tocsr())
        RAR_ml = config.pyAMGsmoother(RAR.tocsr())
        sf_nd_nb.set_data(RARmat_Um=RAR_ml)
    else:
        RARvalues = torch.permute(RARvalues, (1, 2, 0)).contiguous()  # (ndim, ndim, ncola)
        # get SFC, coarse grid and operators on coarse grid. Store them to save computational time?
        space_filling_curve_numbering, variables_sfc, nlevel, nodes_per_level = \
            mg.mg_on_P1CG_prep(fina, cola, RARvalues, sparse_in=sf_nd_nb.sparse_f)
        sf_nd_nb.sfc_data_Um.set_data(
            space_filling_curve_numbering=space_filling_curve_numbering,
            variables_sfc=variables_sfc,
            nlevel=nlevel,
            nodes_per_level=nodes_per_level
        )

    return 0


def _calc_RAR_mf_color(
        x_k,
        I_fc, I_cf,
        whichc, ncolor,
        fina, cola, ncola,
):
    """
    get operator on P1CG grid, i.e. RAR
    where R is prolongator/restrictor
    via coloring method.
    """
    cg_nonods = sf_nd_nb.sparse_f.cg_nonods
    u_nloc = sf_nd_nb.disp_func_space.element.nloc
    nonods = nele * u_nloc

    value = torch.zeros(ncola, ndim, ndim, device=dev, dtype=torch.float64)  # NNZ entry values
    dummy = torch.zeros(nonods, ndim, device=dev, dtype=torch.float64)  # dummy variable of same length as PnDG
    Rm = torch.zeros_like(dummy, device=dev, dtype=torch.float64)
    ARm = torch.zeros_like(dummy, device=dev, dtype=torch.float64)
    RARm = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
    mask = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)  # color vec
    for color in tqdm(range(1, ncolor + 1), disable=config.disabletqdm):
        # print('color: ', color)
        for jdim in range(ndim):
            mask *= 0
            mask[:, jdim] += torch.tensor((whichc == color),
                                          device=dev,
                                          dtype=torch.float64)  # 1 if true; 0 if false
            Rm *= 0
            Rm = Rm.view(nele, u_nloc, ndim)
            for idim in range(ndim):
                Rm[0:nele_f, :, idim] += \
                    (mg.vel_p1dg_to_pndg_prolongator(
                        torch.mv(I_fc, mask[:, idim].view(-1))
                    )).view(nele_f, u_nloc)  # (p3dg_nonods, ndim)
            ARm *= 0
            ARm, _ = get_residual_or_smooth(
                r0=ARm,
                x_i=Rm,
                x_rhs=dummy,
                x_k=x_k,
                do_smooth=False,
            )
            ARm *= -1.  # (p3dg_nonods, ndim)
            ARm = ARm.view(nele, u_nloc, ndim)
            RARm *= 0
            for idim in range(ndim):
                RARm[:, idim] += torch.mv(
                    I_cf,
                    mg.vel_pndg_to_p1dg_restrictor(ARm[0:nele_f, :, idim].view(-1))
                )  # (cg_nonods, ndim)
            for idim in range(ndim):
                # add to value
                for i in range(RARm.shape[0]):
                    for count in range(fina[i], fina[i + 1]):
                        j = cola[count]
                        value[count, idim, jdim] += RARm[i, idim] * mask[j, jdim]
        # print('finishing (another) one color, time comsumed: ', time.time() - start_time)
    return value


def _gmres_mg_solver(
        x_i, x_rhs, x_k, tol
):
    u_nloc = sf_nd_nb.disp_func_space.element.nloc
    total_nonods = nele * u_nloc
    real_nonods = nele_f * u_nloc

    m = 80  # config.gmres_m  # TODO: maybe we can use smaller number for this
    v_m = torch.zeros(m + 1, real_nonods * ndim, device=dev, dtype=torch.float64)
    v_m_j = torch.zeros(total_nonods * ndim, device=dev, dtype=torch.float64)
    h_m = torch.zeros(m + 1, m, device=dev, dtype=torch.float64)
    r0 = torch.zeros(total_nonods * ndim, device=dev, dtype=torch.float64)

    x_dummy = torch.zeros_like(r0, device=dev, dtype=torch.float64)

    r0l2 = 1.
    sf_nd_nb.its = 0
    e_1 = torch.zeros(m + 1, device=dev, dtype=torch.float64)
    e_1[0] += 1

    while r0l2 > tol and sf_nd_nb.its < 80:  # config.gmres_its:  # TODO: maybe we can use smaller number for this
        h_m *= 0
        v_m *= 0
        r0 *= 0
        # get residual
        r0, _ = get_residual_or_smooth(
            r0, x_i, x_rhs, x_k,
            do_smooth=False)
        # apply left preconditioner
        x_dummy *= 0
        x_dummy = _um_left_precond(x_dummy, r0, x_k)
        r0 *= 0
        r0 += x_dummy.view(r0.shape)

        beta = torch.linalg.norm(r0.view(-1))
        r0 = r0.view(nele, u_nloc, ndim)
        v_m[0, :] += r0[0:nele_f, :, :].view(-1) / beta
        w = r0  # this should place w in the same memory as r0 so that we don't take two nonods memory space
        for j in tqdm(range(0, m), disable=config.disabletqdm):
            w *= 0
            # w = A * v_m[j]
            v_m_j *= 0
            v_m_j = v_m_j.view(nele, u_nloc, ndim)
            v_m_j[0:nele_f, :, :] += v_m[j, :].view(nele_f, u_nloc, ndim)
            w, _ = get_residual_or_smooth(
                r0=w,
                x_i=v_m_j,
                x_rhs=0,
                x_k=x_k,
                do_smooth=False)
            w *= -1.  # providing rhs=0, b-Ax is -Ax
            # apply preconditioner
            x_dummy *= 0
            x_dummy = _um_left_precond(x_dummy, w, x_k)
            w *= 0
            w += x_dummy.view(w.shape)
            w = w.view(nele, u_nloc, ndim)
            for i in range(0, j + 1):
                h_m[i, j] = torch.linalg.vecdot(w[0:nele_f, :, :].view(-1),
                                                v_m[i, :])
                w[0:nele_f, :, :] -= (h_m[i, j] * v_m[i, :]).view(nele_f, u_nloc, ndim)

            h_m[j + 1, j] = torch.linalg.norm(w[0:nele_f, :, :].view(-1))
            v_m[j + 1, :] += w[0:nele_f, :, :].view(-1) / h_m[j + 1, j]
            sf_nd_nb.its += 1
        # solve least-square problem
        q, r = torch.linalg.qr(h_m, mode='complete')  # h_m: (m+1)xm, q: (m+1)x(m+1), r: (m+1)xm
        e_1[0] = 0
        e_1[0] += beta
        y_m = torch.linalg.solve(r[0:m, 0:m], q[0:m + 1, 0:m].T @ e_1)  # y_m: m
        # update c_i and get residual
        dx_i = torch.einsum('ji,j->i', v_m[0:m, :], y_m)
        x_i = x_i.view(nele, u_nloc, ndim)
        x_i[0:nele_f, :, :] += dx_i.view(nele_f, u_nloc, ndim)

        # r0l2 = torch.linalg.norm(q[:, m:m+1].T @ e_1)
        r0 *= 0
        # get residual
        r0, _ = get_residual_or_smooth(
            r0, x_i, x_rhs, x_k,
            do_smooth=False)
        r0 = r0.view(nele, u_nloc, ndim)
        r0l2 = torch.linalg.norm(r0[0:nele_f, :, :].view(-1))
        print('its=', sf_nd_nb.its, 'fine grid rel residual l2 norm=', r0l2.cpu().numpy())
    return x_i, sf_nd_nb.its


def _um_left_precond(x_i, x_rhs, x_k):
    """
    do one multi-grid V cycle as left preconditioner
    """
    u_nloc = sf_nd_nb.disp_func_space.element.nloc
    total_no_dofs = nele * u_nloc * ndim
    real_no_dofs = nele_f * u_nloc * ndim

    r0 = torch.zeros(nele, u_nloc, ndim, device=dev, dtype=torch.float64)

    cg_nonods = sf_nd_nb.sparse_f.cg_nonods

    # pre smooth
    for its1 in range(config.pre_smooth_its):
        r0 *= 0
        r0, x_i = get_residual_or_smooth(
            r0, x_i, x_rhs, x_k,
            do_smooth=True
        )
    # get residual on PnDG
    r0 *= 0
    r0, _ = get_residual_or_smooth(
        r0, x_i, x_rhs, x_k,
        do_smooth=False
    )

    if not config.is_pmg:  # PnDG to P1CG
        r1 = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
        r0 = r0.view(nele, u_nloc, ndim)
        for idim in range(ndim):
            r1[:, idim] += torch.mv(
                sf_nd_nb.sparse_f.I_cf,
                mg.vel_pndg_to_p1dg_restrictor(r0[0:nele_f, :, idim]).view(-1)
            )
    else:  # PnDG down one order each time, eventually go to P1CG
        raise NotImplementedError('pmg not implemented!')

    if config.mg_opt_Um == 1:  # two-grid method
        e_i = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
        e_direct = sp.sparse.linalg.spsolve(
            sf_nd_nb.RARmat_Um,
            r1.contiguous().view(-1).cpu().numpy())
        e_direct = np.reshape(e_direct, (cg_nonods, ndim))
        e_i += torch.tensor(e_direct, device=dev, dtype=torch.float64)
    elif config.mg_opt_Um == 3:
        e_i = torch.zeros(cg_nonods, ndim, device=dev, dtype=torch.float64)
        e_direct = sf_nd_nb.RARmat_Um.solve(
            r1.contiguous().view(-1).cpu().numpy(),
            maxiter=1,
            tol=1e-10)
        e_direct = np.reshape(e_direct, (cg_nonods, ndim))
        e_i += torch.tensor(e_direct, device=dev, dtype=torch.float64)
    else:  # multi-grid method
        ncurve = 1  # always use 1 sfc
        N = len(sf_nd_nb.sfc_data_Um.space_filling_curve_numbering)
        inverse_numbering = np.zeros((N, ncurve), dtype=int)
        inverse_numbering[:, 0] = np.argsort(sf_nd_nb.sfc_data_Um.space_filling_curve_numbering[:, 0])
        r1_sfc = r1[inverse_numbering[:, 0], :].view(cg_nonods, ndim)

        # go to SFC coarse grid levels and do 1 mg cycles there
        e_i = mg.mg_on_P1CG(
            r1_sfc.view(cg_nonods, ndim),
            sf_nd_nb.sfc_data_Um.variables_sfc,
            sf_nd_nb.sfc_data_Um.nlevel,
            sf_nd_nb.sfc_data_Um.nodes_per_level,
            cg_nonods
        )
        # reverse to original order
        e_i = e_i[sf_nd_nb.sfc_data_Um.space_filling_curve_numbering[:, 0] - 1, :].view(cg_nonods, ndim)
    if not config.is_pmg:  # from P1CG to P3DG
        # prolongate error to fine grid
        e_i0 = torch.zeros(nele_f * u_nloc, ndim, device=dev, dtype=torch.float64)
        for idim in range(ndim):
            e_i0[:, idim] += mg.vel_p1dg_to_pndg_prolongator(torch.mv(
                sf_nd_nb.sparse_f.I_fc,
                e_i[:, idim]))
    else:  # from P1CG to P1DG, then go one order up each time while also do post smoothing
        raise Exception('pmg not implemented')
    # correct fine grid solution
    x_i = x_i.view(nele, u_nloc, ndim)
    x_i[0:nele_f, ...] += e_i0.view(nele_f, u_nloc, ndim)

    # post smooth
    for its1 in range(config.pre_smooth_its):
        r0 *= 0
        r0, x_i = get_residual_or_smooth(
            r0, x_i, x_rhs, x_k,
            do_smooth=True
        )
    # r0l2 = torch.linalg.norm(r0.view(-1), dim=0) / r0_init  # fNorm

    return x_i
