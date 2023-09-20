"""
Assemble 3D navier-stokes problem for direct solver
"""
import numpy as np
import scipy as sp
from tqdm import tqdm
import config
from config import sf_nd_nb
if config.ndim == 2:
    from shape_function import get_det_nlx as get_det_nlx
    from shape_function import sdet_snlx as sdet_snlx
else:
    from shape_function import get_det_nlx_3d as get_det_nlx
    from shape_function import sdet_snlx_3d as sdet_snlx


def assemble(u_bc_in, f, indices, values, use_fict_dt_in_vel_precond=False):
    u_nonods = sf_nd_nb.vel_func_space.nonods
    p_nonods = sf_nd_nb.pre_func_space.nonods
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    ndim = config.ndim
    nface = ndim + 1
    dev = config.dev
    nele = config.nele

    # rhs
    rhs = np.zeros(u_nonods*ndim+p_nonods)
    # input
    u_bc = u_bc_in[0].view(-1).cpu().numpy()  # dirichlet bc
    if len(u_bc_in) > 1:
        print('has neumann bc data.')
        u_bc_n = u_bc_in[1].view(-1).cpu().numpy()  # neumann bc

    # # pressure bc's
    # p_bc = np.zeros(p_nonods)
    # px_all = sf_nd_nb.pre_func_space.x_all  # (p_nonods, ndim)
    # for ele in range(nele):
    #     for iloc in range(p_nloc):
    #         glb_iloc = ele*p_nloc + iloc
    #         xi = px_all[glb_iloc, 0]
    #         p_bc[glb_iloc] += np.sin(xi)

    # get shape functions
    n = sf_nd_nb.vel_func_space.element.n.cpu().numpy()
    q = sf_nd_nb.pre_func_space.element.n.cpu().numpy()
    sn = sf_nd_nb.vel_func_space.element.sn.cpu().numpy()
    sq = sf_nd_nb.pre_func_space.element.sn.cpu().numpy()
    nx, detwei = get_det_nlx(
        nlx=sf_nd_nb.vel_func_space.element.nlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in,
        weight=sf_nd_nb.vel_func_space.element.weight,
        nloc=u_nloc,
        ngi=sf_nd_nb.vel_func_space.element.ngi,
    )
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi,
    )
    nx = nx.cpu().numpy()
    detwei = detwei.cpu().numpy()
    snx = snx.cpu().numpy()
    sdetwei = sdetwei.cpu().numpy()
    snormal = snormal.cpu().numpy()

    # indices = []
    # values = []

    # volume integral
    nxnx = np.einsum('bkmg,bkng,bg->bmn', nx, nx, detwei) * config.mu
    nn = np.einsum('mg,ng,bg->bmn', n, n, detwei)
    qnx = np.einsum('mg,bing,bg->bmni', q, nx, detwei)
    nxq = np.einsum('bimg,ng,bg->bmin', nx, q, detwei)

    f = f.view(nele, u_nloc, ndim).cpu().numpy()
    rhs[0:u_nonods*ndim] += np.einsum('bmn, bni->bmi', nn, f).reshape(u_nonods*ndim)
    for ele in tqdm(range(nele), disable=config.disabletqdm):
        # K and G
        for iloc in range(u_nloc):
            for idim in range(ndim):
                glb_iloc = ele*u_nloc*ndim + iloc * ndim + idim
                # print('glb_iloc', glb_iloc)
                # K
                for jloc in range(u_nloc):
                    jdim = idim
                    glb_jloc = ele*u_nloc*ndim + jloc * ndim + jdim
                    # print('     glb_jloc', glb_jloc)
                    value = nxnx[ele, iloc, jloc]
                    if sf_nd_nb.isTransient or use_fict_dt_in_vel_precond:
                        value += nn[ele, iloc, jloc] * config.rho / sf_nd_nb.dt * sf_nd_nb.bdfscm.gamma
                    # elif add_mass_to_precond:
                    #     value += nn[ele, iloc, jloc] * sf_nd_nb.fict_mass_coeff
                    indices.append([glb_iloc, glb_jloc])
                    values.append(value)
                # G
                for jloc in range(p_nloc):
                    glb_jloc = nele*u_nloc*ndim + p_nloc*ele + jloc
                    # print('     glb_jloc', glb_jloc)
                    indices.append([glb_iloc, glb_jloc])
                    values.append(-nxq[ele, iloc, idim, jloc])
        # G^T
        for iloc in range(p_nloc):
            glb_iloc = nele*u_nloc*ndim + p_nloc*ele + iloc
            # print('glb_iloc', glb_iloc)
            for jloc in range(u_nloc):
                for jdim in range(ndim):
                    glb_jloc = ele*u_nloc*ndim + jloc * ndim + jdim
                    # print('     glb_jloc', glb_jloc)
                    indices.append([glb_iloc, glb_jloc])
                    values.append(-qnx[ele, iloc, jloc, jdim])

    # surface integral
    eta_e = config.eta_e
    nbele = sf_nd_nb.vel_func_space.nbele.cpu().numpy()
    nbf = sf_nd_nb.vel_func_space.nbf.cpu().numpy()
    alnmt = sf_nd_nb.vel_func_space.alnmt.cpu().numpy()
    glb_bcface_type = sf_nd_nb.vel_func_space.glb_bcface_type.cpu().numpy()
    # print('glb bcface type', glb_bcface_type)
    # print('alnmt', alnmt)
    u_gi_align = sf_nd_nb.vel_func_space.element.gi_align.cpu().numpy()
    q_gi_align = sf_nd_nb.pre_func_space.element.gi_align.cpu().numpy()
    if True:  # use python to assemble surface integral term (would be slow)
        for ele in tqdm(range(nele), disable=config.disabletqdm):
            for iface in range(nface):
                glb_iface = ele*nface + iface
                glb_iface_type = glb_bcface_type[glb_iface]
                if glb_iface_type == 0:
                # if alnmt[glb_iface] < 0:
                    # this is boundary face
                    # K and G
                    # mu_e = eta_e/np.power(np.sum(detwei[ele, :]), 1./ndim)
                    if ndim == 3:
                        mu_e = eta_e / np.sqrt(np.sum(sdetwei[ele, iface, :]))
                    else:
                        mu_e = eta_e / np.sum(sdetwei[ele, iface, :])
                    # mu_e = eta_e / .5  # TOOO: temporarily change to 1 to debug (compare to other code)
                    # print('ele, iface, mu_e', ele, iface, mu_e)
                    for inod in range(u_nloc):
                        for idim in range(ndim):
                            glb_inod = ele*u_nloc*ndim + inod*ndim + idim
                            # K
                            for jnod in range(u_nloc):
                                jdim = idim
                                glb_jnod = ele*u_nloc*ndim + jnod*ndim + jdim
                                vnux = 0  # [v_i n_k][du_i / dx_k]
                                vxun = 0  # [dv_i/ dx_k][u_i n_k]
                                nn = 0
                                for kdim in range(ndim):
                                    vnux += np.sum(sn[iface, inod, :] * snormal[ele, iface, kdim] *
                                                   snx[ele, iface, kdim, jnod, :] * sdetwei[ele, iface, :])
                                    vxun += np.sum(snx[ele, iface, kdim, inod, :] * sn[iface, jnod, :] *
                                                   snormal[ele, iface, kdim] * sdetwei[ele, iface, :])
                                nn += np.sum(sn[iface, inod, :] * sn[iface, jnod, :] *
                                             sdetwei[ele, iface, :]) * mu_e
                                # print('    inod, idim, jnod, jdim, glbi, glbj', inod, idim, jnod, jdim, glb_inod, glb_jnod)
                                indices.append([glb_inod, glb_jnod])
                                values.append((-vnux - vxun + nn) * config.mu)
                                # add boundary contribution to rhs
                                rhs[glb_inod] += u_bc[glb_jnod] * (-vxun + nn) * config.mu
                            # G
                            for jnod in range(p_nloc):
                                glb_jnod = nele*u_nloc*ndim + ele*p_nloc + jnod
                                # print('ele, iface, inod, idim, jnod, glbinod, glbjnod',
                                #       ele, iface, inod, idim, jnod, glb_inod, glb_jnod)
                                vnq = 0  # [vi ni] {q}
                                vnq += np.sum(sn[iface, inod, :] * snormal[ele, iface, idim] *
                                              sq[iface, jnod, :] * sdetwei[ele, iface, :])
                                # print('    inod, idim, jnod, glbi, glbj', inod, idim, jnod, glb_inod, glb_jnod)
                                indices.append([glb_inod, glb_jnod])
                                values.append(vnq)
                    # G^T
                    for inod in range(p_nloc):
                        glb_inod = nele*u_nloc*ndim + ele*p_nloc + inod
                        for jnod in range(u_nloc):
                            for jdim in range(ndim):
                                glb_jnod = ele*u_nloc*ndim + jnod*ndim + jdim
                                qun = np.sum(sq[iface, inod, :] * sn[iface, jnod, :] *
                                             snormal[ele, iface, jdim] * sdetwei[ele, iface, :])
                                # print('    inod, jnod, jdim, glbi, glbj', inod, jnod, jdim, glb_inod, glb_jnod)
                                indices.append([glb_inod, glb_jnod])
                                values.append(qun)
                                # add boundary contribution to rhs
                                rhs[glb_inod] += u_bc[glb_jnod] * qun
                elif glb_iface_type == 1:
                    # neumann boundary
                    for inod in range(u_nloc):
                        for idim in range(ndim):
                            glb_inod = ele*u_nloc*ndim + inod*ndim + idim
                            for jnod in range(u_nloc):
                                jdim = idim
                                glb_jnod = ele*u_nloc*ndim + jnod*ndim + jdim
                                snsn = np.sum(sn[iface, inod, :] * sn[iface, jnod, :] *
                                              sdetwei[ele, iface, :])
                                rhs[glb_inod] += snsn * u_bc_n[glb_jnod]
                else:
                    # this is interior / internal face
                    ele2 = nbele[glb_iface]
                    # mu_e = 2. * eta_e / (np.power(np.sum(detwei[ele, :]), 1./ndim) +
                    #                      np.power(np.sum(detwei[ele2, :]), 1./ndim))
                    if ndim == 3:
                        h = np.sqrt(np.sum(sdetwei[ele, iface, :]))
                    else:
                        h = np.sum(sdetwei[ele, iface, :])
                    mu_e = eta_e / h
                    # mu_e = eta_e / .5  # TOOO: temporarily change to 1 to debug (compare to other code)
                    # print('ele, iface, mu_e, ele2', ele, iface, mu_e, ele2)
                    glb_iface2 = nbf[glb_iface]
                    iface2 = glb_iface2 % nface
                    # K and G
                    for inod in range(u_nloc):
                        for idim in range(ndim):
                            glb_inod = ele*u_nloc*ndim + inod*ndim + idim
                            # this side
                            # K
                            for jnod in range(u_nloc):
                                jdim = idim
                                glb_jnod = ele*u_nloc*ndim + jnod*ndim + jdim
                                vnux = 0  # [v_i n_k][du_i / dx_k]
                                vxun = 0  # [dv_i/ dx_k][u_i n_k]
                                nn = 0
                                vxux = 0
                                for kdim in range(ndim):
                                    vnux += np.sum(sn[iface, inod, :] * snormal[ele, iface, kdim] *
                                                   snx[ele, iface, kdim, jnod, :] * sdetwei[ele, iface, :])
                                    vxun += np.sum(snx[ele, iface, kdim, inod, :] * sn[iface, jnod, :] *
                                                   snormal[ele, iface, kdim] * sdetwei[ele, iface, :])
                                    vxux += np.sum(snx[ele, iface, kdim, jnod, :] *
                                                   snx[ele, iface, kdim, inod, :] *
                                                   sdetwei[ele, iface, :]) * \
                                            h**2 * config.gammaES * 0.5
                                nn += np.sum(sn[iface, inod, :] * sn[iface, jnod, :] *
                                             sdetwei[ele, iface, :]) * mu_e
                                # print('    inod, idim, jnod, jdim, glbi, glbj', inod, idim, jnod, jdim, glb_inod, glb_jnod)
                                indices.append([glb_inod, glb_jnod])
                                values.append((-0.5*vnux - 0.5*vxun + nn) * config.mu + vxux * sf_nd_nb.isES)
                            # G
                            for jnod in range(p_nloc):
                                glb_jnod = nele * u_nloc * ndim + ele * p_nloc + jnod
                                vnq = 0  # [vi ni] {q}
                                vnq += np.sum(sn[iface, inod, :] * snormal[ele, iface, idim] *
                                              sq[iface, jnod, :] * sdetwei[ele, iface, :])
                                # print('    inod, idim, jnod, glbi, glbj', inod, idim, jnod, glb_inod, glb_jnod)
                                indices.append([glb_inod, glb_jnod])
                                values.append(0.5*vnq)
                            # other side
                            # K
                            for jnod2 in range(u_nloc):
                                jdim2 = idim
                                glb_jnod2 = ele2 * u_nloc * ndim + jnod2 * ndim + jdim2
                                vnux = 0  # [v_i n_k][du_i / dx_k]
                                vxun = 0  # [dv_i/ dx_k][u_i n_k]
                                nn = 0
                                vxux = 0
                                for kdim in range(ndim):
                                    vnux += np.sum(sn[iface, inod, :] * snormal[ele, iface, kdim] *
                                                   snx[ele2, iface2, kdim, jnod2, u_gi_align[alnmt[glb_iface]]] *
                                                   sdetwei[ele, iface, :])
                                    vxun += np.sum(snx[ele, iface, kdim, inod, :] *
                                                   sn[iface2, jnod2, u_gi_align[alnmt[glb_iface]]] *
                                                   snormal[ele2, iface2, kdim] * sdetwei[ele, iface, :])
                                    vxux += np.sum(snx[ele2, iface2, kdim, jnod2, u_gi_align[alnmt[glb_iface]]] *
                                                   snx[ele, iface, kdim, inod, :] *
                                                   sdetwei[ele, iface, :]) * \
                                            h ** 2 * config.gammaES * (-0.5)
                                nn += np.sum(sn[iface, inod, :] * sn[iface2, jnod2, u_gi_align[alnmt[glb_iface]]] *
                                             sdetwei[ele, iface, :]) * mu_e * (-1.)
                                # print('    inod, idim, jnod, jdim2, glbi, glbj', inod, idim, jnod2, jdim2, glb_inod, glb_jnod2)
                                indices.append([glb_inod, glb_jnod2])
                                values.append((-0.5 * vnux - 0.5 * vxun + nn) * config.mu + vxux * sf_nd_nb.isES)
                            # G
                            for jnod2 in range(p_nloc):
                                glb_jnod2 = nele * u_nloc * ndim + ele2 * p_nloc + jnod2
                                vnq = 0  # [vi ni] {q}
                                vnq += np.sum(sn[iface, inod, :] * snormal[ele, iface, idim] *
                                              sq[iface2, jnod2, u_gi_align[alnmt[glb_iface]]] *
                                              sdetwei[ele, iface, :])
                                # print('    inod, idim, jnod, glbi, glbj', inod, idim, jnod2, glb_inod, glb_jnod2)
                                indices.append([glb_inod, glb_jnod2])
                                values.append(0.5 * vnq)
                    # G^T
                    for inod in range(p_nloc):
                        glb_inod = nele*u_nloc*ndim + ele*p_nloc + inod
                        # this side
                        for jnod in range(u_nloc):
                            for jdim in range(ndim):
                                glb_jnod = ele*u_nloc*ndim + jnod*ndim + jdim
                                qun = np.sum(sq[iface, inod, :] * sn[iface, jnod, :] *
                                             snormal[ele, iface, jdim] * sdetwei[ele, iface, :])
                                # print('    inod, jnod, jdim, glbi, glbj', inod, jnod, jdim, glb_inod, glb_jnod)
                                indices.append([glb_inod, glb_jnod])
                                values.append(0.5*qun)
                        # other side
                        for jnod2 in range(u_nloc):
                            for jdim2 in range(ndim):
                                glb_jnod2 = ele2*u_nloc*ndim + jnod2*ndim + jdim2
                                qun = np.sum(sq[iface, inod, :] *
                                             sn[iface2, jnod2, u_gi_align[alnmt[glb_iface]]] *
                                             snormal[ele2, iface2, jdim2] * sdetwei[ele, iface, :])
                                # print('    inod, jnod2, jdim2, glbi, glbj', inod, jnod2, jdim2, glb_inod, glb_jnod2)
                                indices.append([glb_inod, glb_jnod2])
                                values.append(0.5 * qun)
                    if config.is_pressure_stablise:
                        # S (pressure penalty term), as per Di Pietro & Ern 2012 pg. 252.
                        # In theory, we can use same order for both vel and pre if we have this term.
                        # this term only integrates on interior faces,
                        # thus does not introduce additional pressure bc contri
                        for inod in range(p_nloc):
                            glb_inod = nele*u_nloc*ndim + ele*p_nloc + inod
                            # this side
                            for jnod in range(p_nloc):
                                glb_jnod = nele*u_nloc*ndim + ele*p_nloc + jnod
                                qq = np.sum(sq[iface, inod, :] * sq[iface, jnod, :] *
                                            sdetwei[ele, iface, :]) * config.eta_e / mu_e  # mu_e = eta_e/h
                                indices.append([glb_inod, glb_jnod])
                                values.append(qq)  # positive because n1.n1 = 1
                            # other side
                            for jnod2 in range(p_nloc):
                                glb_jnod2 = nele*u_nloc*ndim + ele2*p_nloc + jnod2
                                qq = np.sum(sq[iface, inod, :] *
                                            sq[iface2, jnod2, u_gi_align[alnmt[glb_iface]]] *
                                            sdetwei[ele, iface, :]) * config.eta_e / mu_e
                                indices.append([glb_inod, glb_jnod2])
                                values.append(-qq)  # negative because n1.n2 = -1
    else:  # use fortran to assemble surface integral term
        from stokes_assemble_fortran import stokes_asssemble_fortran
        values_f, indices_, ndix, rhs_ = stokes_asssemble_fortran(
            sn=sn, snx=snx, sdetwei=sdetwei, snormal=snormal,
            sq=sq, detwei=detwei,
            u_bc=u_bc.reshape((nele, u_nloc, ndim)),
            nbele=nbele, nbf=nbf, alnmt=alnmt, gi_align=u_gi_align,
            mx_nidx=nele*(u_nloc*ndim+p_nloc)*100,
            eta_e=eta_e
        )

    # remove null space
    if config.hasNullSpace:
        # # indices.append([nele*u_nloc*ndim + nele*p_nloc - 1, nele*u_nloc*ndim + nele*p_nloc - 1])
        # indices.append([nele*u_nloc*ndim, nele*u_nloc*ndim])
        # values.append(1.)
    # else:  # enforce average pressure over whole domain is 0
        for ele in range(nele):
            glb_inod = u_nonods*ndim
            for jnod in range(p_nloc):
                glb_jnod = u_nonods*ndim + ele*p_nloc + jnod
                int_q = np.sum(q[jnod, :] * detwei[ele, :])
                indices.append([glb_inod, glb_jnod])
                values.append(int_q)
    # # convert to np csr sparse mat
    # values = np.asarray(values)
    # indices = np.transpose(np.asarray(indices))
    # Amat = sp.sparse.coo_matrix((values, (indices[0, :], indices[1, :])),
    #                             shape=(nele*u_nloc*ndim + nele*p_nloc,
    #                                    nele*u_nloc*ndim + nele*p_nloc))
    # Amat = Amat.tocsr()
    return rhs, indices, values


def assemble_adv(u_n_in, u_bc_in, indices, values):
    """this function assemble the advection term and its rhs
    at current non-linear step. thus, input is
    u_n -> velocity at lsat non-linear step
    u_D -> dirichlet bc data"""
    u_nonods = sf_nd_nb.vel_func_space.nonods
    p_nonods = sf_nd_nb.pre_func_space.nonods
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    ndim = config.ndim
    nface = ndim + 1
    dev = config.dev
    nele = config.nele

    # rhs
    rhs = np.zeros(u_nonods * ndim + p_nonods)
    # input
    if type(u_n_in) is not np.ndarray:
        u_n_in = u_n_in.cpu().numpy()
    u_n = u_n_in.reshape(-1)[0:u_nonods*ndim]  # u at last non-linear step
    u_bc = u_bc_in[0].view(nele, u_nloc, ndim).cpu().numpy()  # dirichlet bc

    # get shape functions
    n = sf_nd_nb.vel_func_space.element.n.cpu().numpy()
    q = sf_nd_nb.pre_func_space.element.n.cpu().numpy()
    sn = sf_nd_nb.vel_func_space.element.sn.cpu().numpy()
    sq = sf_nd_nb.pre_func_space.element.sn.cpu().numpy()
    nx, detwei = get_det_nlx(
        nlx=sf_nd_nb.vel_func_space.element.nlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in,
        weight=sf_nd_nb.vel_func_space.element.weight,
        nloc=u_nloc,
        ngi=sf_nd_nb.vel_func_space.element.ngi,
    )
    snx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.vel_func_space.element.snlx,
        sweight=sf_nd_nb.vel_func_space.element.sweight,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in,
        nloc=sf_nd_nb.vel_func_space.element.nloc,
        sngi=sf_nd_nb.vel_func_space.element.sngi,
    )
    nx = nx.cpu().numpy()
    detwei = detwei.cpu().numpy()
    snx = snx.cpu().numpy()
    sdetwei = sdetwei.cpu().numpy()
    snormal = snormal.cpu().numpy()

    isTemam = False  # add skew-symmetric term
    print('isTemam?', isTemam)
    isNewton = False  # Newton linearisation or Picard linearisation
    u_n = u_n.reshape([nele, u_nloc, ndim])
    # volume integral
    wduxv = np.einsum('lg,bli,bing,mg,bg->bmn', n, u_n, nx, n, detwei)
    wxduv = 0.5*np.einsum('bilg,bli,mg,ng,bg->bmn', nx, u_n, n, n, detwei)
    if not isTemam:
        wxduv *= 0
    for ele in tqdm(range(nele), disable=config.disabletqdm):
        for iloc in range(u_nloc):
            for idim in range(ndim):
                glb_iloc = ele*u_nloc*ndim + iloc * ndim + idim
                for jloc in range(u_nloc):
                    jdim = idim
                    glb_jloc = ele*u_nloc*ndim + jloc * ndim + jdim
                    value = 0
                    value += wduxv[ele, iloc, jloc]
                    value += wxduv[ele, iloc, jloc]

                    indices.append([glb_iloc, glb_jloc])
                    values.append(value)

    # surface integral
    nbele = sf_nd_nb.vel_func_space.nbele.cpu().numpy()
    nbf = sf_nd_nb.vel_func_space.nbf.cpu().numpy()
    alnmt = sf_nd_nb.vel_func_space.alnmt.cpu().numpy()
    glb_bcface_type = sf_nd_nb.vel_func_space.glb_bcface_type.cpu().numpy()
    # print('glb bcface type', glb_bcface_type)
    # print('alnmt', alnmt)
    u_gi_align = sf_nd_nb.vel_func_space.element.gi_align.cpu().numpy()
    for ele in tqdm(range(nele), disable=config.disabletqdm):
        for iface in range(nface):
            glb_iface = ele*nface + iface
            glb_iface_type = glb_bcface_type[glb_iface]
            if glb_iface_type == 0:
                # print(iface)
                # this is Dirichlet boundary face
                for inod in range(u_nloc):
                    for idim in range(ndim):
                        glb_inod = ele * u_nloc * ndim + inod * ndim + idim

                        wknk_ave = np.einsum(
                            'mg,mi,i->g',
                            sn[iface, :, :],
                            u_bc[ele, :, :],
                            snormal[ele, iface, :]
                        )
                        wknk_upwd = 0.5 * (wknk_ave - np.abs(wknk_ave))
                        wknk_jump = np.einsum(
                            'mg,mi,i->g',
                            sn[iface, :, :],
                            u_n[ele, :, :],
                            snormal[ele, iface, :]
                        )

                        for jnod in range(u_nloc):
                            jdim = idim
                            glb_jnod = ele*u_nloc*ndim + jnod*ndim + jdim
                            wnduv = 0
                            wnduv += -np.sum(
                                0.5 * wknk_ave  # using Gauger2019 : (g_D . n)(u . v)
                                * sn[iface, jnod, :]
                                * sn[iface, inod, :]
                                * sdetwei[ele, iface, :]
                            )  # Temam
                            wnduv_upwd = 0
                            wnduv_upwd += -np.sum(
                                wknk_upwd
                                * sn[iface, jnod, :]
                                * sn[iface, inod, :]
                                * sdetwei[ele, iface, :]
                            )  # upwind flux at boundary faces
                            indices.append([glb_inod, glb_jnod])
                            values.append(wnduv * isTemam + wnduv_upwd)
                            rhs[glb_inod] += (wnduv * isTemam + wnduv_upwd) * u_bc.reshape(-1)[glb_jnod]
            elif glb_iface_type == 1:
                # print(f'iface {glb_iface} is a neumann boundary face')
                # this is Neumann boundary face
                continue  # nothing to do here
            else:  # interior face
                # this is interior face
                ele2 = nbele[glb_iface]
                glb_iface2 = nbf[glb_iface]
                iface2 = glb_iface2 % nface
                # # fix outer normal vector direction
                # snormal_fix = snormal[ele, iface, :]
                # if iface2 > iface:
                #     snormal_fix *= -1
                for inod in range(u_nloc):
                    for idim in range(ndim):
                        glb_inod = ele * u_nloc * ndim + inod * ndim + idim

                        wknk_ave = np.einsum(
                            'mg,mi,i->g',
                            sn[iface, :, :],
                            u_n[ele, :, :],
                            snormal[ele, iface, :]
                        ) * 0.5 + np.einsum(
                            'gm,mi,i->g',
                            sn[iface2, :, u_gi_align[alnmt[glb_iface]]],
                            u_n[ele2, :, :],
                            snormal[ele, iface, :]
                        ) * 0.5
                        wknk_upwd = 0.5*(wknk_ave - np.abs(wknk_ave))
                        wknk_jump = np.einsum(
                            'mg,mi,i->g',
                            sn[iface, :, :],
                            u_n[ele, :, :],
                            snormal[ele, iface, :]
                            # snormal_fix
                        ) + np.einsum(
                            'gm,mi,i->g',
                            sn[iface2, :, u_gi_align[alnmt[glb_iface]]],
                            u_n[ele2, :, :],
                            snormal[ele2, iface2, :]
                            # -snormal_fix
                        )
                        # this side
                        for jnod in range(u_nloc):
                            jdim = idim
                            glb_jnod = ele * u_nloc * ndim + jnod * ndim + jdim
                            wnduv_upwd = 0  # upwind term
                            wnduv = 0  # skew-symmetric term

                            wnduv_upwd += -np.sum(
                                wknk_upwd
                                * sn[iface, jnod, :]
                                * sn[iface, inod, :]
                                * sdetwei[ele, iface, :]
                            )
                            wnduv += -0.5 * np.sum(
                                wknk_jump
                                * sn[iface, jnod, :]
                                * sn[iface, inod, :]
                                * sdetwei[ele, iface, :]
                            ) * 0.5  # this is the 1/2 in ave operator

                            indices.append([glb_inod, glb_jnod])
                            values.append(wnduv_upwd + wnduv*isTemam)
                        # other side
                        for jnod2 in range(u_nloc):
                            jdim2 = idim
                            glb_jnod2 = ele2 * u_nloc * ndim + jnod2 * ndim + jdim2
                            wnduv_upwd = 0  # upwind term
                            wnduv = 0  # skew-symmetric term

                            wnduv_upwd += -np.sum(
                                wknk_upwd
                                * sn[iface2, jnod2, u_gi_align[alnmt[glb_iface]]]
                                * sn[iface, inod, :]
                                * sdetwei[ele, iface, :]
                            ) * (-1.)
                            # wnduv += -0.5 * np.sum(
                            #     wknk_jump
                            #     * sn[iface2, jnod2, u_gi_align[alnmt[glb_iface]]]
                            #     * sn[iface, inod, :]
                            #     * sdetwei[ele, iface, :]
                            # ) * 0.5  # this is the 1/2 in ave operator

                            indices.append([glb_inod, glb_jnod2])
                            values.append(wnduv_upwd + wnduv*isTemam)
    # # convert to np csr sparse mat
    # values = np.asarray(values)
    # indices = np.transpose(np.asarray(indices))
    # Cmat = sp.sparse.coo_matrix((values, (indices[0, :], indices[1, :])),
    #                             shape=(nele * u_nloc * ndim + nele * p_nloc,
    #                                    nele * u_nloc * ndim + nele * p_nloc))
    # Cmat = Cmat.tocsr()
    return rhs, indices, values


def get_ave_pressure(pre):
    """
    input: pressure field
    output: its average
    """
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    ndim = config.ndim
    nele = config.nele
    p_nonods = nele * p_nloc
    n = sf_nd_nb.vel_func_space.element.n.cpu().numpy()
    q = sf_nd_nb.pre_func_space.element.n.cpu().numpy()
    sn = sf_nd_nb.vel_func_space.element.sn.cpu().numpy()
    sq = sf_nd_nb.pre_func_space.element.sn.cpu().numpy()
    _, detwei = get_det_nlx(
        nlx=sf_nd_nb.vel_func_space.element.nlx,
        x_loc=sf_nd_nb.vel_func_space.x_ref_in,
        weight=sf_nd_nb.vel_func_space.element.weight,
        nloc=u_nloc,
        ngi=sf_nd_nb.vel_func_space.element.ngi,
    )
    detwei = detwei.cpu().numpy()
    pre = pre.reshape([nele, p_nloc])
    int_pre = np.sum(np.einsum('mg,ng,bg,bn', q, q, detwei, pre))
    int_vol = np.sum(np.einsum('mg,ng,bg', q, q, detwei))
    print('total p, total vol', int_pre, int_vol)
    return int_pre / int_vol


def assemble_csr_mat(indices, values, shape=None):
    # convert to np csr sparse mat
    nele = config.nele
    u_nloc = sf_nd_nb.vel_func_space.element.nloc
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    ndim = config.ndim
    values = np.asarray(values)
    indices = np.transpose(np.asarray(indices))
    if shape is None:  # by default, assemble the whole matrix
        Amat = sp.sparse.coo_matrix((values, (indices[0, :], indices[1, :])),
                                    shape=(nele * u_nloc * ndim + nele * p_nloc,
                                           nele * u_nloc * ndim + nele * p_nloc))
    else:
        Amat = sp.sparse.coo_matrix((values, (indices[0, :], indices[1, :])),
                                    shape=shape)
    Amat = Amat.tocsr()
    return Amat


def pressure_laplacian_assemble(indices, values):
    p_nonods = sf_nd_nb.pre_func_space.nonods
    p_nloc = sf_nd_nb.pre_func_space.element.nloc
    ndim = config.ndim
    nface = ndim + 1
    dev = config.dev
    nele = config.nele

    # get shape functions
    q = sf_nd_nb.pre_func_space.element.n.cpu().numpy()
    sq = sf_nd_nb.pre_func_space.element.sn.cpu().numpy()
    qx, detwei = get_det_nlx(
        nlx=sf_nd_nb.pre_func_space.element.nlx,
        x_loc=sf_nd_nb.pre_func_space.x_ref_in,
        weight=sf_nd_nb.pre_func_space.element.weight,
        nloc=p_nloc,
        ngi=sf_nd_nb.pre_func_space.element.ngi,
    )
    sqx, sdetwei, snormal = sdet_snlx(
        snlx=sf_nd_nb.pre_func_space.element.snlx,
        sweight=sf_nd_nb.pre_func_space.element.sweight,
        x_loc=sf_nd_nb.pre_func_space.x_ref_in,
        nloc=sf_nd_nb.pre_func_space.element.nloc,
        sngi=sf_nd_nb.pre_func_space.element.sngi,
    )
    qx = qx.cpu().numpy()
    detwei = detwei.cpu().numpy()
    sqx = sqx.cpu().numpy()
    sdetwei = sdetwei.cpu().numpy()
    snormal = snormal.cpu().numpy()

    # indices = []
    # values = []

    # volume integral
    qxqx = np.einsum('bkmg,bkng,bg->bmn', qx, qx, detwei) * config.mu

    for ele in tqdm(range(nele), disable=config.disabletqdm):
        # K and G
        for iloc in range(p_nloc):
            glb_iloc = ele*p_nloc + iloc
            # print('glb_iloc', glb_iloc)
            # K
            for jloc in range(p_nloc):
                glb_jloc = ele*p_nloc + jloc
                # print('     glb_jloc', glb_jloc)
                indices.append([glb_iloc, glb_jloc])
                values.append(qxqx[ele, iloc, jloc])

    # surface integral
    eta_e = config.eta_e
    nbele = sf_nd_nb.vel_func_space.nbele.cpu().numpy()
    nbf = sf_nd_nb.vel_func_space.nbf.cpu().numpy()
    alnmt = sf_nd_nb.vel_func_space.alnmt.cpu().numpy()
    glb_bcface_type = sf_nd_nb.vel_func_space.glb_bcface_type.cpu().numpy()
    # print('glb bcface type', glb_bcface_type)
    # print('alnmt', alnmt)
    q_gi_align = sf_nd_nb.pre_func_space.element.gi_align.cpu().numpy()
    if True:  # use python to assemble surface integral term (would be slow)
        for ele in tqdm(range(nele), disable=config.disabletqdm):
            for iface in range(nface):
                glb_iface = ele*nface + iface
                glb_iface_type = glb_bcface_type[glb_iface]
                if glb_iface_type == 0:
                # if alnmt[glb_iface] < 0:
                    # this is boundary face
                    # K and G
                    # mu_e = eta_e/np.power(np.sum(detwei[ele, :]), 1./ndim)
                    if ndim == 3:
                        mu_e = eta_e / np.sqrt(np.sum(sdetwei[ele, iface, :]))
                    else:
                        mu_e = eta_e / np.sum(sdetwei[ele, iface, :])
                    # mu_e = eta_e / .5  # TOOO: temporarily change to 1 to debug (compare to other code)
                    # print('ele, iface, mu_e', ele, iface, mu_e)
                    for inod in range(p_nloc):
                        glb_inod = ele*p_nloc + inod
                        # K
                        for jnod in range(p_nloc):
                            glb_jnod = ele*p_nloc + jnod
                            vnux = 0  # [v_i n_k][du_i / dx_k]
                            vxun = 0  # [dv_i/ dx_k][u_i n_k]
                            nn = 0
                            for kdim in range(ndim):
                                vnux += np.sum(sq[iface, inod, :] * snormal[ele, iface, kdim] *
                                               sqx[ele, iface, kdim, jnod, :] * sdetwei[ele, iface, :])
                                vxun += np.sum(sqx[ele, iface, kdim, inod, :] * sq[iface, jnod, :] *
                                               snormal[ele, iface, kdim] * sdetwei[ele, iface, :])
                            nn += np.sum(sq[iface, inod, :] * sq[iface, jnod, :] *
                                         sdetwei[ele, iface, :]) * mu_e
                            # print('    inod, idim, jnod, jdim, glbi, glbj', inod, idim, jnod, jdim, glb_inod, glb_jnod)
                            indices.append([glb_inod, glb_jnod])
                            values.append((-vnux - vxun + nn) * config.mu)
                    # print('ele iface values final', ele, iface, values[-1])
                elif glb_iface_type == 1:
                    # neumann boundary
                    continue
                else:
                    # this is interior / internal face
                    ele2 = nbele[glb_iface]
                    # mu_e = 2. * eta_e / (np.power(np.sum(detwei[ele, :]), 1./ndim) +
                    #                      np.power(np.sum(detwei[ele2, :]), 1./ndim))
                    if ndim == 3:
                        h = np.sqrt(np.sum(sdetwei[ele, iface, :]))
                    else:
                        h = np.sum(sdetwei[ele, iface, :])
                    mu_e = eta_e / h
                    # mu_e = eta_e / .5  # TOOO: temporarily change to 1 to debug (compare to other code)
                    # print('ele, iface, mu_e, ele2', ele, iface, mu_e, ele2)
                    glb_iface2 = nbf[glb_iface]
                    iface2 = glb_iface2 % nface
                    # K and G
                    for inod in range(p_nloc):
                        glb_inod = ele*p_nloc + inod
                        # this side
                        # K
                        for jnod in range(p_nloc):
                            glb_jnod = ele*p_nloc + jnod
                            vnux = 0  # [v_i n_k][du_i / dx_k]
                            vxun = 0  # [dv_i/ dx_k][u_i n_k]
                            nn = 0
                            vxux = 0
                            for kdim in range(ndim):
                                vnux += np.sum(sq[iface, inod, :] * snormal[ele, iface, kdim] *
                                               sqx[ele, iface, kdim, jnod, :] * sdetwei[ele, iface, :])
                                vxun += np.sum(sqx[ele, iface, kdim, inod, :] * sq[iface, jnod, :] *
                                               snormal[ele, iface, kdim] * sdetwei[ele, iface, :])
                                vxux += np.sum(sqx[ele, iface, kdim, jnod, :] *
                                               sqx[ele, iface, kdim, inod, :] *
                                               sdetwei[ele, iface, :]) * \
                                        h**2 * config.gammaES * 0.5
                            nn += np.sum(sq[iface, inod, :] * sq[iface, jnod, :] *
                                         sdetwei[ele, iface, :]) * mu_e
                            # print('    inod, idim, jnod, jdim, glbi, glbj', inod, idim, jnod, jdim, glb_inod, glb_jnod)
                            indices.append([glb_inod, glb_jnod])
                            values.append((-0.5*vnux - 0.5*vxun + nn) * config.mu + vxux * sf_nd_nb.isES)

                        # other side
                        # K
                        for jnod2 in range(p_nloc):
                            glb_jnod2 = ele2 * p_nloc + jnod2
                            vnux = 0  # [v_i n_k][du_i / dx_k]
                            vxun = 0  # [dv_i/ dx_k][u_i n_k]
                            nn = 0
                            vxux = 0
                            for kdim in range(ndim):
                                vnux += np.sum(sq[iface, inod, :] * snormal[ele, iface, kdim] *
                                               sqx[ele2, iface2, kdim, jnod2, q_gi_align[alnmt[glb_iface]]] *
                                               sdetwei[ele, iface, :])
                                vxun += np.sum(sqx[ele, iface, kdim, inod, :] *
                                               sq[iface2, jnod2, q_gi_align[alnmt[glb_iface]]] *
                                               snormal[ele2, iface2, kdim] * sdetwei[ele, iface, :])
                                vxux += np.sum(sqx[ele2, iface2, kdim, jnod2, q_gi_align[alnmt[glb_iface]]] *
                                               sqx[ele, iface, kdim, inod, :] *
                                               sdetwei[ele, iface, :]) * \
                                        h ** 2 * config.gammaES * (-0.5)
                            nn += np.sum(sq[iface, inod, :] * sq[iface2, jnod2, q_gi_align[alnmt[glb_iface]]] *
                                         sdetwei[ele, iface, :]) * mu_e * (-1.)
                            # print('    inod, idim, jnod, jdim2, glbi, glbj', inod, idim, jnod2, jdim2, glb_inod, glb_jnod2)
                            indices.append([glb_inod, glb_jnod2])
                            values.append((-0.5 * vnux - 0.5 * vxun + nn) * config.mu + vxux * sf_nd_nb.isES)

    return indices, values
