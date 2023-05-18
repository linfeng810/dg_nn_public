"""
assemble 3D diffusion problem for direct solver
"""
import numpy as np
import scipy as sp
from tqdm import tqdm
import config, shape_function
from config import sf_nd_nb


def assemble(c_bc, f):
    # output rhs
    rhs = np.zeros(config.nonods)
    # input
    c_bc = c_bc.view(-1).cpu().numpy()
    f = f.view(-1).cpu().numpy()

    nele = config.nele
    nloc = config.nloc
    ndim = config.ndim
    nface = config.nface
    n = sf_nd_nb.n.cpu().numpy()
    sn = sf_nd_nb.sn.cpu().numpy()
    nx, detwei = shape_function.get_det_nlx_3d(
        nlx=sf_nd_nb.nlx,
        x_loc=sf_nd_nb.x_ref_in,
        weight=sf_nd_nb.weight
    )
    snx, sdetwei, snormal = shape_function.sdet_snlx_3d(
        snlx=sf_nd_nb.snlx,
        sweight=sf_nd_nb.sweight,
        x_loc=sf_nd_nb.x_ref_in
    )
    nx = nx.cpu().numpy()
    detwei = detwei.cpu().numpy()
    snx = snx.cpu().numpy()
    sdetwei = sdetwei.cpu().numpy()
    snormal = snormal.cpu().numpy()

    # volume integral
    indices = []
    values = []
    for ele in tqdm(range(nele)):
        for iloc in range(nloc):
            glob_iloc = ele*nloc + iloc
            for jloc in range(nloc):
                glob_jloc = ele*nloc + jloc
                nxnx = 0
                for idim in range(ndim):
                    nxnx += np.sum(nx[ele,idim,iloc,:] * nx[ele,idim,jloc,:] * detwei[ele,:])
                indices.append([glob_iloc, glob_jloc])
                values.append(nxnx)
                # account for rhs source f
                nn = np.sum(n[iloc,:] * n[jloc,:] * detwei[ele,:])
                rhs[glob_iloc] += nn * f[glob_jloc]
    # surface integral
    eta_e = config.eta_e
    nbele = sf_nd_nb.nbele
    nbf = sf_nd_nb.nbf
    alnmt = sf_nd_nb.alnmt
    gi_align = sf_nd_nb.gi_align
    gi_align = gi_align.cpu().numpy()
    for ele in tqdm(range(nele)):
        for iface in range(config.nface):
            mu_e = eta_e/np.sum(sdetwei[ele,iface,:])
            glb_iface = ele*nface + iface
            if alnmt[glb_iface] < 0:
                # this is boundary face
                for inod in range(nloc):
                    glb_inod = ele * nloc + inod
                    # this side
                    for jnod in range(nloc):
                        glb_jnod = ele * nloc + jnod
                        nnx = 0
                        nxn = 0
                        nn = 0
                        for idim in range(ndim):
                            nnx += np.sum(sn[iface, jnod, :] * snx[ele, iface, idim, inod, :] *
                                          snormal[ele, iface, idim] * sdetwei[ele, iface, :])
                            nxn += np.sum(snx[ele, iface, idim, jnod, :] * sn[iface, inod, :] *
                                          snormal[ele, iface, idim] * sdetwei[ele, iface, :])
                        nn += np.sum(sn[iface, jnod, :] * sn[iface, inod, :] *
                                     sdetwei[ele, iface, :])
                        indices.append([glb_inod, glb_jnod])
                        values.append(-nnx - nxn + mu_e*nn)
                        # account for boundary conditions c_bc
                        rhs[glb_inod] += c_bc[glb_jnod] * (-nnx + mu_e*nn)
            else:
                # this is interior face
                ele2 = nbele[glb_iface]
                glb_iface2 = nbf[glb_iface]
                iface2 = glb_iface2 % nface
                for inod in range(nloc):
                    glb_inod = ele*nloc + inod
                    # this side
                    for jnod in range(nloc):
                        glb_jnod = ele*nloc + jnod
                        nnx = 0
                        nxn = 0
                        nn = 0
                        for idim in range(ndim):
                            nnx += np.sum(sn[iface,jnod,:] * snx[ele,iface,idim,inod,:] *
                                          snormal[ele,iface,idim] * sdetwei[ele,iface,:])
                            nxn += np.sum(snx[ele,iface,idim,jnod,:] * sn[iface,inod,:] *
                                          snormal[ele,iface,idim] * sdetwei[ele,iface,:])
                        nn += np.sum(sn[iface,jnod,:] * sn[iface,inod,:] *
                                     sdetwei[ele,iface,:])
                        indices.append([glb_inod, glb_jnod])
                        values.append(-0.5*nnx - 0.5*nxn + mu_e*nn)
                    # other side
                    for jnod2 in range(nloc):
                        glb_jnod2 = ele2*nloc + jnod2
                        nnx = 0
                        nxn = 0
                        nn = 0
                        for idim in range(ndim):
                            nnx += np.sum(sn[iface2,jnod2,gi_align[alnmt[glb_iface]]] *
                                          snx[ele,iface,idim,inod,:] *
                                          snormal[ele2,iface2,idim] * sdetwei[ele,iface,:])
                            nxn += np.sum(snx[ele2,iface2,idim,jnod2,gi_align[alnmt[glb_iface]]] *
                                          sn[iface,inod,:] * snormal[ele,iface,idim] *
                                          sdetwei[ele,iface,:])
                        nn += (-1.) * np.sum(sn[iface2, jnod2, gi_align[alnmt[glb_iface2]]] *
                                             sn[iface, inod, :] * sdetwei[ele,iface,:])
                        indices.append([glb_inod, glb_jnod2])
                        values.append(-0.5*nnx - 0.5*nxn + mu_e*nn)
    values = np.asarray(values)
    indices = np.transpose(np.asarray(indices))
    Amat = sp.sparse.coo_matrix((values, (indices[0,:], indices[1,:])),
                                shape=(config.nonods, config.nonods))
    Amat = Amat.tocsr()
    return Amat, rhs


