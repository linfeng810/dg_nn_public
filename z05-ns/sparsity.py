""" get P1CG sparsity for subdomain """
import meshio

import color
from cmmn_data import Sparsity
import config
import numpy as np
from scipy.sparse import coo_matrix, bsr_matrix, lil_matrix
import torch
from get_nb import getfin_p1cg
import time

ndim = config.ndim
dev = config.dev


def get_subdomain_sparsity(cg_ndglno, nele_f, nele_s, cg_nonods):
    """
    input: meshio object

    output:
    """
    nele = nele_f + nele_s

    # fluid
    fluid_spar = Sparsity(name='fluid')

    cg_ndglno_f = cg_ndglno.reshape((nele, -1))[0:nele_f, 0:ndim + 1].reshape((nele_f * (ndim+1)))
    cg_node_order_f = np.unique(cg_ndglno_f)
    idx_dict_f = {old: new for new, old in enumerate(cg_node_order_f)}
    cg_nonods_f = cg_node_order_f.shape[0]
    if cg_nonods_f > 0: # if there is fluid subdomain
        cg_ndglno_f_new = np.vectorize(idx_dict_f.get)(cg_ndglno_f)
    else: # no fluid subdomain
        cg_ndglno_f_new = np.array([], dtype=np.int64)

    fina, cola, ncola = p1cg_sparsity_from_ndglno(cg_ndglno_f_new, nele_f, cg_nonods_f, nloc=ndim+1)
    whichc, ncolor = color.color2(fina, cola, cg_nonods_f)
    I_fc, I_cf = get_cd_dc_prolongator_restrictor(
        cg_ndglno_f_new,
        cg_nonods_f,
        p1dg_nonods=nele_f * (ndim + 1),
    )

    fluid_spar.set_data(
        fina=fina,
        cola=cola,
        ncola=ncola,
        whichc=whichc,
        ncolor=ncolor,
        I_fc=I_fc,
        I_cf=I_cf,
        cg_nonods=cg_nonods_f,
        p1dg_nonods=nele_f * (ndim+1),
    )

    # solid
    solid_spar = Sparsity(name='solid')

    cg_ndglno_s = cg_ndglno.reshape((nele, -1))[nele_f:nele, 0:ndim + 1].reshape((nele_s * (ndim + 1)))
    cg_node_order_s = np.unique(cg_ndglno_s)
    idx_dict_s = {old: new for new, old in enumerate(cg_node_order_s)}
    cg_nonods_s = cg_node_order_s.shape[0]
    if cg_nonods_s > 0:
        cg_ndglno_s_new = np.vectorize(idx_dict_s.get)(cg_ndglno_s)
    else:  # no solid subdomain
        cg_ndglno_s_new = np.array([], dtype=np.int64)

    fina, cola, ncola = p1cg_sparsity_from_ndglno(cg_ndglno_s_new, nele_s, cg_nonods_s, nloc=ndim + 1)
    whichc, ncolor = color.color2(fina, cola, cg_nonods_s)
    I_fc, I_cf = get_cd_dc_prolongator_restrictor(
        cg_ndglno_s_new,
        cg_nonods_s,
        p1dg_nonods=nele_s * (ndim + 1),
    )

    solid_spar.set_data(
        fina=fina,
        cola=cola,
        ncola=ncola,
        whichc=whichc,
        ncolor=ncolor,
        I_fc=I_fc,
        I_cf=I_cf,
        cg_nonods=cg_nonods_s,
        p1dg_nonods=nele_s * (ndim + 1),
    )

    return fluid_spar, solid_spar


def p1cg_sparsity_from_ndglno(cg_ndglbno, nele, cg_nonods, nloc):
    '''
    get P1CG sparsity from node-global-number list
    '''
    import time

    start_time = time.time()
    print('im in get p1cg sparsity, time:', time.time()-start_time)
    # nele = config.nele
    p1dg_nonods = nele * nloc
    idx = []
    val = []
    import scipy.sparse as sp
    idx, n_idx = getfin_p1cg(cg_ndglbno+1, nele, nloc, p1dg_nonods)
    print('im back from fortran, time:', time.time()-start_time)
    idx = np.asarray(idx)-1
    val = np.ones(n_idx)
    spmat = sp.coo_matrix((val,(idx[0,:],idx[1,:])), shape=(cg_nonods, cg_nonods))
    spmat = spmat.tocsr()
    print('ive finished, time:', time.time()-start_time)
    return spmat.indptr, spmat.indices, spmat.nnz


def get_cd_dc_prolongator_restrictor(cg_ndglno, cg_nonods, p1dg_nonods):
    # p1dg_nonods = vel_func_space.p1dg_nonods
    # cg_ndglno = vel_func_space.cg_ndglno
    # cg_nonods = vel_func_space.cg_nonods
    I_fc_colx = np.arange(0, p1dg_nonods)
    I_fc_coly = cg_ndglno
    I_fc_val = np.ones(p1dg_nonods)
    I_cf = coo_matrix((I_fc_val, (I_fc_coly, I_fc_colx)),
                      shape=(cg_nonods, p1dg_nonods))  # fine to coarse == DG to CG
    I_cf = I_cf.tocsr()
    # print('don assemble I_cf I_fc', time.time() - starttime)
    no_dgnodes = I_cf.sum(axis=1)
    # print('done suming', time.time() - starttime)
    # weight by 1 / overlapping dg nodes number
    for i in range(p1dg_nonods):
        I_fc_val[i] /= no_dgnodes[I_fc_coly[i]]
    # print('done weighting', time.time() - starttime)
    # for i in tqdm(range(cg_nonods)):
    #     # no_dgnodes = np.sum(I_cf[i,:])
    #     I_cf[i,:] /= no_dgnodes[i]
    #     I_fc[:,i] /= no_dgnodes[i]
    # print('done weighting', time.time()-starttime)
    I_cf = coo_matrix((I_fc_val, (I_fc_coly, I_fc_colx)),
                      shape=(cg_nonods, p1dg_nonods))  # fine to coarse == DG to CG
    I_cf = I_cf.tocsr()
    I_fc = coo_matrix((I_fc_val, (I_fc_colx, I_fc_coly)),
                      shape=(p1dg_nonods, cg_nonods))  # coarse to fine == CG to DG
    I_fc = I_fc.tocsr()
    # print('done transform to csr', time.time() - starttime)
    # transfer to torch device
    I_fc = torch.sparse_csr_tensor(crow_indices=torch.tensor(I_fc.indptr),
                                   col_indices=torch.tensor(I_fc.indices),
                                   values=I_fc.data,
                                   size=(p1dg_nonods, cg_nonods),
                                   device=dev)
    I_cf = torch.sparse_csr_tensor(crow_indices=torch.tensor(I_cf.indptr),
                                   col_indices=torch.tensor(I_cf.indices),
                                   values=I_cf.data,
                                   size=(cg_nonods, p1dg_nonods),
                                   device=dev)
    return I_fc, I_cf


def get_fluid_pndg_sparsity(func_space):
    """
    get PnDG mesh sparsity. Input is a function space object,
    practically, we will only use the fluid subdomain,
    so we will only get the PnDG sparsity in the fluid subdomain:
    (0:nele_f) elements.
    """
    nloc = func_space.element.nloc
    # Get the unique points in the mesh
    pndg_x_all = func_space.x_all.reshape((-1, ndim))[0:config.nele_f * nloc, :]
    tolerance = 1e-10
    pndg_x_all_rounded = np.round(pndg_x_all / tolerance) * tolerance
    pncg_x_all, pndg_ndglbno = np.unique(pndg_x_all_rounded, axis=0, return_inverse=True)

    # create a sparse matrix I_fc
    I_fc_colx = np.arange(0, len(pndg_x_all_rounded))
    I_fc_coly = pndg_ndglbno
    I_fc_val = np.ones(len(pndg_x_all))
    I_fc = coo_matrix((I_fc_val, (I_fc_colx, I_fc_coly)),
                      shape=(len(pndg_x_all), len(pncg_x_all)))  # coarse to fine == CG to DG
    # # Create a sparse matrix where each row corresponds to a point in the old list,
    # # and columns represent whether that point matches a unique point or not
    # is_matching = lil_matrix((len(pndg_x_all), len(pncg_x_all)))
    # print('going to get matching_indices...')
    # starttime = time.time()
    # matching_indices = np.all(pndg_x_all_rounded[:, None, :] == pncg_x_all[None, :, :], axis=-1)
    # # Fill the sparse matrix efficiently
    # is_matching[:, :] = matching_indices
    #
    # # for i, pncg_point in enumerate(pncg_x_all):
    # #     is_matching[:, i] = np.all(pndg_x_all_rounded == pncg_point, axis=1)
    # print('finishing getting matching_indices... time comsumed:', time.time() - starttime)
    return I_fc


def get_pndg_sparsity(func_space):
    """
    get PnDG mesh sparsity. Input is a function space object,
    practically, we will only use the fluid subdomain,
    so we will only get the PnDG sparsity in the fluid subdomain:
    (0:nele_f) elements.
    """
    nloc = func_space.element.nloc
    # Get the unique points in the mesh
    pndg_x_all = func_space.x_all.reshape((-1, ndim))
    tolerance = 1e-10
    pndg_x_all_rounded = np.round(pndg_x_all / tolerance) * tolerance
    pncg_x_all, pndg_ndglbno = np.unique(pndg_x_all_rounded, axis=0, return_inverse=True)

    # create a sparse matrix I_fc
    I_fc_colx = np.arange(0, len(pndg_x_all_rounded))
    I_fc_coly = pndg_ndglbno
    I_fc_val = np.ones(len(pndg_x_all))
    I_fc = coo_matrix((I_fc_val, (I_fc_colx, I_fc_coly)),
                      shape=(len(pndg_x_all), len(pncg_x_all)))  # coarse to fine == CG to DG

    # transform to lil format so that we can use indexing (to use weight in fluid/solid subdomain)
    I_fc = I_fc.tolil()
    # # Create a sparse matrix where each row corresponds to a point in the old list,
    # # and columns represent whether that point matches a unique point or not
    # is_matching = lil_matrix((len(pndg_x_all), len(pncg_x_all)))
    #
    # for i, pncg_point in enumerate(pncg_x_all):
    #     is_matching[:, i] = np.all(pndg_x_all_rounded == pncg_point, axis=1)
    return I_fc
