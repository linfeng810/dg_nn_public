"""
Assemble 3D stokes problem for direct solver
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


def assemble(u_bc, f):
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
    u_bc = u_bc.view(-1).cpu().numpy()
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

    indices = []
    values = []

    # volume integral
    nxnx = np.einsum('bkmg,bkng,bg->bmn', nx, nx, detwei)
    nn = np.einsum('mg,ng,bg->bmn', n, n, detwei)
    qnx = np.einsum('mg,bing,bg->bmni', q, nx, detwei)
    nxq = np.einsum('bimg,ng,bg->bmin', nx, q, detwei)

    f = f.view(nele, u_nloc, ndim).cpu().numpy()
    rhs[0:u_nonods*ndim] += np.einsum('bmn, bni->bmi', nn, f).reshape(u_nonods*ndim)
    for ele in tqdm(range(nele)):
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
                    indices.append([glb_iloc, glb_jloc])
                    values.append(nxnx[ele, iloc, jloc])
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
    u_gi_align = sf_nd_nb.vel_func_space.element.gi_align.cpu().numpy()
    q_gi_align = sf_nd_nb.pre_func_space.element.gi_align.cpu().numpy()
    if True:  # use python to assemble surface integral term (would be slow)
        for ele in tqdm(range(nele)):
            for iface in range(nface):
                glb_iface = ele*nface + iface
                if alnmt[glb_iface] < 0:
                    # this is boundary face
                    # K and G
                    mu_e = eta_e/np.power(np.sum(detwei[ele, :]), 1./ndim)
                    # mu_e = 1.  # TOOO: temporarily change to 1 to debug (compare to other code)
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
                                values.append(-vnux - vxun + nn)
                                # add boundary contribution to rhs
                                rhs[glb_inod] += u_bc[glb_jnod] * (-vxun + nn)
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
                                # # add pressure bc as well since we integrate by once
                                # rhs[glb_inod] += p_bc[glb_jnod - nele*u_nloc*ndim] * vnq
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
                else:
                    # this is interior / internal face
                    ele2 = nbele[glb_iface]
                    mu_e = 2. * eta_e / (np.power(np.sum(detwei[ele, :]), 1./ndim) +
                                         np.power(np.sum(detwei[ele2, :]), 1./ndim))
                    # mu_e = 1  # TOOO: temporarily change to 1 to debug (compare to other code)
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
                                for kdim in range(ndim):
                                    vnux += np.sum(sn[iface, inod, :] * snormal[ele, iface, kdim] *
                                                   snx[ele, iface, kdim, jnod, :] * sdetwei[ele, iface, :])
                                    vxun += np.sum(snx[ele, iface, kdim, inod, :] * sn[iface, jnod, :] *
                                                   snormal[ele, iface, kdim] * sdetwei[ele, iface, :])
                                nn += np.sum(sn[iface, inod, :] * sn[iface, jnod, :] *
                                             sdetwei[ele, iface, :]) * mu_e
                                # print('    inod, idim, jnod, jdim, glbi, glbj', inod, idim, jnod, jdim, glb_inod, glb_jnod)
                                indices.append([glb_inod, glb_jnod])
                                values.append(-0.5*vnux - 0.5*vxun + nn)
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
                                for kdim in range(ndim):
                                    vnux += np.sum(sn[iface, inod, :] * snormal[ele, iface, kdim] *
                                                   snx[ele2, iface2, kdim, jnod2, u_gi_align[alnmt[glb_iface]]] *
                                                   sdetwei[ele, iface, :])
                                    vxun += np.sum(snx[ele, iface, kdim, inod, :] *
                                                   sn[iface2, jnod2, u_gi_align[alnmt[glb_iface]]] *
                                                   snormal[ele2, iface2, kdim] * sdetwei[ele, iface, :])
                                nn += np.sum(sn[iface, inod, :] * sn[iface2, jnod2, u_gi_align[alnmt[glb_iface]]] *
                                             sdetwei[ele, iface, :]) * mu_e * (-1.)
                                # print('    inod, idim, jnod, jdim2, glbi, glbj', inod, idim, jnod2, jdim2, glb_inod, glb_jnod2)
                                indices.append([glb_inod, glb_jnod2])
                                values.append(-0.5 * vnux - 0.5 * vxun + nn)
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
                    # S (pressure penalty term, as per Di Pietro & Ern 2012 pg. 252. In theory, we can use same order
                    # for both vel and pre if we have this term.
                    # this term only integrates on interior faces, thus does not introduce additional pressure bc contri
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
    if False:
        # indices.append([nele*u_nloc*ndim + nele*p_nloc - 1, nele*u_nloc*ndim + nele*p_nloc - 1])
        indices.append([nele*u_nloc*ndim, nele*u_nloc*ndim])
        values.append(1.)
    else:  # enforce average pressure over whole domain is 0
        for ele in range(nele):
            glb_inod = u_nonods*ndim
            for jnod in range(p_nloc):
                glb_jnod = u_nonods*ndim + ele*p_nloc + jnod
                int_q = np.sum(q[jnod, :] * detwei[ele, :])
                indices.append([glb_inod, glb_jnod])
                values.append(int_q)
    # convert to np csr sparse mat
    values = np.asarray(values)
    indices = np.transpose(np.asarray(indices))
    Amat = sp.sparse.coo_matrix((values, (indices[0, :], indices[1, :])),
                                shape=(nele*u_nloc*ndim + nele*p_nloc,
                                       nele*u_nloc*ndim + nele*p_nloc))
    Amat = Amat.tocsr()
    return Amat, rhs
