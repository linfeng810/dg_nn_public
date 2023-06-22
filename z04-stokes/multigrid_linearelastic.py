#!/usr/bin/env python3
'''
This file has functions related  to 
multi-grid cycle on SFC level 
coarsened grids for Stokes problems
(where operators are (ndim+1, ndim+1) tensors)
'''
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import config
# import volume_mf_linear_elastic
from config import sf_nd_nb
import sfc as sf # to be compiled ...
import map_to_sfc_matrix as map_sfc
import shape_function
import time, os.path
from scipy.sparse import bsr_matrix, linalg

ndim = config.ndim


def mg_smooth_one_level(level, e_i, b, variables_sfc):
    """
    do one smooth step on mg level = level
    """
    e_i = e_i.view(ndim+1,variables_sfc[level][2])
    rr_i = torch.zeros_like(e_i, device=config.dev, dtype=torch.float64)
    a_sfc_sparse, diagA, _ = variables_sfc[level]
    for idim in range(ndim+1):
        for jdim in range(ndim+1):
            rr_i[idim,:] += torch.mv(a_sfc_sparse[idim][jdim], e_i[jdim,:])
    rr_i *= -1
    rr_i += b.view(ndim+1, -1)
    e_i += rr_i / diagA.view(ndim+1,-1) * config.jac_wei
    e_i = e_i.view(ndim+1,1,-1)
    rr_i = rr_i.view(ndim+1,1,-1)
    return e_i, rr_i


def get_a_diaga(level,
            fin_sfc_nonods,
            fina_sfc_all_un,
            cola_sfc_all_un,
            a_sfc):
    '''
    extract operators and its diagonals on one level of SFC
    coarse grid

    # Input
    level : int
        level-th grid to extract
    fin_sfc_nonods : numpy array (integer) (nlevel+1)
        a list containing the starting index in a_sfc of each level
        actual length will be longer than nlevel+1, extras are 0
    fina_sfc_all_un : numpy array (float)
        fina on all levels
    cola_sfc_all_un : numpy array (float)
        cola on all levels
    a_sfc : numpy/torch array (ndim, ndim, :)
        a_sfc values on all levels

    # Output
    a_sfc_level_sparse : a 2-d list of torch coo sparse tensor
        list dimension (ndim, ndim)
        coo sparse tensor dimension (nonods, nonods)
        this stores operator A as a block matrix 
        in SFC ordering for the specified level
    diagonal : torch tensor, (ndim,nonods)
        Diagonal values in a_sfc_level_sparse
    nonods: scalar
        Number of nodes in the *specified* level. N.B. this is *NOT*
        the global number of nodes.

    '''
    # level 0 is the highest level;
    # level nlevel-1 is the lowest level;
    # subtracting 1 because fortran indexes from 1 but python indexes from 0
    
    start_index = fin_sfc_nonods[level] - 1
    end_index = fin_sfc_nonods[level + 1] - 1
    nonods = end_index - start_index

    diagonal = torch.zeros((config.ndim+1, nonods), dtype=torch.float64, device=config.dev)
    a_indices = []
    a_values = []
    for i in range(start_index, end_index):
        for j in range(fina_sfc_all_un[i]-1, fina_sfc_all_un[i+1]-1):
            a_indices.append([i-start_index, cola_sfc_all_un[j]-1-start_index])
            a_values.append(a_sfc[:,:,j])
    a_indices = np.asarray(a_indices).transpose()
    a_values = np.asarray(a_values) # now a_values has shape (nonods, ndim+1, ndim+1)
    a_values = torch.tensor(a_values, dtype=torch.float64, device=config.dev)
    # convert to sparse
    a_sfc_level_sparse = [[] for _ in range(ndim+1)]
        # see https://stackoverflow.com/questions/240178/list-of-lists-changes-reflected-across-sublists-unexpectedly 
        # about the reason why we cannot use list([[]*ndim]*ndim)
    for idim in range(ndim+1):
        for jdim in range(ndim+1):
            a_sfc_level_sparse[idim].append( torch.sparse_coo_tensor(
                a_indices,
                a_values[:,idim,jdim],
                (nonods, nonods),
                dtype=torch.float64,
                device=config.dev).to_sparse_csr() )
    # find diagonal
    for i in range(nonods):
        i += start_index
        for j in range(fina_sfc_all_un[i]-1, fina_sfc_all_un[i+1]-1):
            if (i==cola_sfc_all_un[j]-1):
                for idim in range(ndim+1):
                    diagonal[idim,i-start_index] = a_sfc[idim,idim,j]
    
    return a_sfc_level_sparse, diagonal, nonods 


def mg_on_P1CG(r0, variables_sfc, nlevel, nodes_per_level):
    """
    Multi-grid cycle on P1CG mesh.
    Takes in residaul on P1CG mesh (restricted from residual on PnDG mesh),
    do 1 mg cycle on the SFC levels, then spit out an error correction on
    P1CG mesh (add it to the input e_i0).
    """
    cg_nonods = sf_nd_nb.vel_func_space.cg_nonods
    # get residual on each level
    sfc_restrictor = torch.nn.Conv1d(in_channels=1,
                                     out_channels=1, kernel_size=2,
                                     stride=2, padding='valid', bias=False)
    sfc_restrictor.weight.data = torch.tensor([[1., 1.]],
                                              dtype=torch.float64,
                                              device=config.dev).view(1, 1, 2)

    smooth_start_level = config.smooth_start_level
    r0 = r0.view(cg_nonods, ndim+1).transpose(dim0=0, dim1=1).view(ndim+1, 1, cg_nonods)
    # r = r0.view(ndim, 1, cg_nonods)  # residual r in error equation, Ae=r
    r_s = [r0]  # collection of r
    e_s = [torch.zeros(ndim+1, 1, cg_nonods, device=config.dev, dtype=torch.float64)]  # collec. of e
    for level in range(0,smooth_start_level):
        # pre-smooth
        for its1 in range(config.pre_smooth_its):
            e_s[level], _ = mg_smooth_one_level(
                level=level,
                e_i=e_s[level],
                b=r_s[level],
                variables_sfc=variables_sfc)
        # get residual on this level
        _, rr = mg_smooth_one_level(
            level=level,
            e_i=e_s[level],
            b=r_s[level],
            variables_sfc=variables_sfc)
        # restriction
        rr = F.pad(rr.view(ndim+1,1,-1), (0,1), "constant", 0)
        e_i = F.pad(e_s[level].view(ndim+1,1,-1), (0,1), "constant", 0)
        with torch.no_grad():
            rr = sfc_restrictor(rr)
            r_s.append(rr.view(ndim+1,1,-1))
            e_i = sfc_restrictor(e_i)
            e_s.append(e_i.view(ndim+1,1,-1))
    for level in range(smooth_start_level, -1, -1):
        if level == smooth_start_level:
            # direct solve on smooth_start_level
            a_on_l = variables_sfc[level][0]
            e_i_direct = linalg.spsolve(a_on_l.tocsr(),
                                        r_s[level].view(ndim+1,-1).transpose(0,1).view(-1).cpu().numpy())
            e_s[level] = torch.tensor(e_i_direct, device=config.dev, dtype=torch.float64)
        else:  # smooth
            # prolongation
            CNN1D_prol_odd = nn.Upsample(scale_factor=nodes_per_level[level]/nodes_per_level[level+1])
            e_s[level] += CNN1D_prol_odd(e_s[level+1].view(ndim+1,1,-1))
            # post smooth
            for its1 in range(config.post_smooth_its):
                e_s[level], _ = mg_smooth_one_level(
                    level=level,
                    e_i=e_s[level],
                    b=r_s[level],
                    variables_sfc=variables_sfc)
    return e_s[0].view(ndim+1, cg_nonods).transpose(0,1).contiguous()


def mg_on_P0DG_prep(fina, cola, RARvalues):
    '''
    # Prepare for Multi-grid cycle on P0DG mesh

    This function forms space filling curve. Then form
    a series of coarse grid and operators thereon.

    # Input 
    fina : torch tensor, (nele+1)
        sparsity - start of rows - of RAR matrix
    cola : torch tensor, (ncola)
        sparsity - column indices - of RAR matrix
    RARvalues : torch tensor, (ncola, ndim, ndim)
        ncola (ndim, ndim) block of RAR matrix

    # Output
    sfc : numpy list (nele)
        space filling curve index for ele.
    variables_sfc : list (nlevel)
        a list of all ingredients one needs to perform a smoothing
        step on level-th grid. Each list member is a list of the 
        following member:
        [0] a_sfc_sparse : a 2-D list of torch coo sparse tensor, 
            list shape (ndim, ndim)
            coo sparse tensor shape (nonods_level, nonods_level)
            coarse level grid operator
        [1] diag_weights : torch tensor, (ndim, nonods_level)
            diagonal of coarse grid operator
        [2] nonods : integer
            number of nodes on level-th grid
    nlevel : scalar, int
        number of SFC coarse grid levels
    nodes_per_level : list of int, (nlevel)
        number of nodes (DOFs) on each level
    '''

    cg_nonods = sf_nd_nb.vel_func_space.cg_nonods
    dummy = np.zeros((config.ndim+1, cg_nonods))

    starting_node = 1  # setting according to BY
    graph_trim = -10  # ''
    ncurve = 1        # '' 
    # nele = config.nele # this is the shape of RAR # or should we use RAR.shape[0] for clarity?
    ncola = cola.shape[0]
    start_time = time.time()
    # print('to get space filling curve...', time.time()-start_time)
    if os.path.isfile(config.filename[:-4] + '_sfc.npy'):
        # print('pre-calculated sfc exists. readin from file...')
        sfc = np.load(config.filename[:-4] + '_sfc.npy')
    else:
        _, sfc = \
            sf.ncurve_python_subdomain_space_filling_curve( \
            cola+1, fina+1, starting_node, graph_trim, ncurve, \
            ) # note that fortran array index start from 1, so cola and fina should +1.
        np.save(config.filename[:-4] + '_sfc.npy', sfc)
    # print('to get sfc operators...', time.time()-start_time)
    
    # get coarse grid info
    max_nlevel = sf.calculate_nlevel_sfc(config.nele) + 1
    max_nonods_sfc_all_grids = 5 * config.nele
    max_ncola_sfc_all_un = 10*ncola
    a_sfc, fina_sfc_all_un, cola_sfc_all_un, ncola_sfc_all_un, b_sfc, \
        ml_sfc, fin_sfc_nonods, nonods_sfc_all_grids, nlevel = \
        map_sfc.vector_best_sfc_mapping_to_sfc_matrix_unstructured(
            vec_a=RARvalues.cpu().numpy(),
            vec_b=dummy,
            ml=dummy[0,:],
            fina=fina+1,
            cola=cola+1,
            sfc_node_ordering=sfc[:,0],
            max_nonods_sfc_all_grids=max_nonods_sfc_all_grids,
            max_ncola_sfc_all_un=max_ncola_sfc_all_un,
            max_nlevel=max_nlevel,
            ndim=config.ndim+1, ncola=ncola,nonods=cg_nonods)
    # print('back from sfc operator fortran subroutine,', time.time() - start_time)
    nodes_per_level = [fin_sfc_nonods[i] - fin_sfc_nonods[i-1] for i in range(1, nlevel+1)]
    # print(fin_sfc_nonods.shape)
    a_sfc = a_sfc[: ,: , :ncola_sfc_all_un]
    del b_sfc, ml_sfc 
    # choose a level to directly solve on. then we'll iterate from there and levels up
    if config.smooth_start_level < 0:
        # for level in range(1,nlevel):
        #     if nodes_per_level[level] < 2:
        #         config.smooth_start_level = level
        #         break
        config.smooth_start_level += nlevel
    # print('start_level: ', config.smooth_start_level)
    variables_sfc = []
    for level in range(config.smooth_start_level+1):
        variables_sfc.append(get_a_diaga(
            level,
            fin_sfc_nonods,
            fina_sfc_all_un,
            cola_sfc_all_un,
            a_sfc
        ))
    # build A on smooth_start_level. all levels before this are smooth levels,
    # on smooth_start_level, we use direct solve. Therefore, A is replaced with a
    # scipy bsr_matrix.
    level = config.smooth_start_level
    a_sfc_l = variables_sfc[level][0]  # this is a ndim+1 x ndim+1 list of torch csr tensors.
    cola = a_sfc_l[0][0].col_indices().detach().clone().cpu().numpy()
    fina = a_sfc_l[0][0].crow_indices().detach().clone().cpu().numpy()
    vals = np.zeros((cola.shape[0], ndim+1, ndim+1), dtype=np.float64)
    for idim in range(ndim+1):
        for jdim in range(ndim+1):
            vals[:, idim, jdim] += a_sfc_l[idim][jdim].values().detach().clone().cpu().numpy()
    a_on_l = bsr_matrix((vals, cola, fina),
                        shape=(nodes_per_level[level] * (ndim+1), nodes_per_level[level] * (ndim+1)))
    variables_sfc[level] = (a_on_l, 0, nodes_per_level[level])
    return sfc, variables_sfc, nlevel, nodes_per_level


def vel_pndg_to_p1dg_restrictor(x):
    y = torch.einsum('ij,kj->ki',
                     sf_nd_nb.vel_func_space.restrictor_to_p1dg,
                     x.view(config.nele, sf_nd_nb.vel_func_space.element.nloc)).contiguous().view(-1)
    return y


def vel_p1dg_to_pndg_prolongator(x):
    y = torch.einsum('ij,kj->ki',
                     sf_nd_nb.vel_func_space.prolongator_from_p1dg,
                     x.view(config.nele, p_nloc(1))).contiguous().view(-1)
    return y


def pre_pndg_to_p1dg_restrictor(x):
    y = torch.einsum('ij,kj->ki',
                     sf_nd_nb.pre_func_space.restrictor_to_p1dg,
                     x.view(config.nele, sf_nd_nb.pre_func_space.element.nloc)).contiguous().view(-1)
    return y


def pre_p1dg_to_pndg_prolongator(x):
    y = torch.einsum('ij,kj->ki',
                     sf_nd_nb.pre_func_space.prolongator_from_p1dg,
                     x.view(config.nele, p_nloc(1))).contiguous().view(-1)
    return y



def get_p1cg_lumped_mass(x_ref_in):
    '''
    this function spiltes out lumped mass matrix on P1CG.
    the lumped mass is used for mass-weighting SFC coarsened
    grid operators.
    '''
    x_ref_in = x_ref_in.view(-1, config.ndim, config.nloc)
    cg_n, cg_nlx, cg_wt, cg_sn, cg_snlx, cg_swt = \
        shape_function.SHATRInew(nloc=3, ngi=3, ndim=2, snloc=2, sngi=2)
    cg_n = torch.tensor(cg_n, device=config.dev, dtype=torch.float64)
    cg_nlx = torch.tensor(cg_nlx, device=config.dev, dtype=torch.float64)
    cg_wt = torch.tensor(cg_wt, device=config.dev, dtype=torch.float64)
    ml = torch.zeros(sf_nd_nb.cg_nonods, device=config.dev, dtype=torch.float64)
    for ele in range(config.nele):
        nx, detwei = shape_function.get_det_nlx(cg_nlx,
                                                x_ref_in[ele,:,0:3].view(-1,config.ndim,3),
                                                cg_wt,
                                                nloc=3, ngi=3)
        for iloc in range(3):
            glb_iloc = config.cg_ndglno[ele*3+iloc]
            for jloc in range(3):
                # glb_jloc = config.cg_ndglno[ele*3+jloc]
                ninj = torch.sum(cg_n[iloc,:] * cg_n[jloc,:] * detwei[0,:])
                ml[glb_iloc] += ninj
    return ml


def p_mg_pre(r0):
    """
    do p-multigrid pre-smooth.
    input is residual on finest (highest p) mesh,
    output is a series of error and residual on
    coarser (with lower order p) mesh.
    The geometry of the mesh is the same for these
    p multi-grids.
    """
    ele_p = config.ele_p
    r_p = [r0]  # store residaul on each p level
    e_p = [torch.zeros_like(r0, device=config.dev, dtype=torch.float64)]  # store error on each p level
    rr_i = r0  # residual of error equation Ae=r. rr_i := r_p - A e_i
    for p in range(ele_p-1, 0, -1):
        ilevel = ele_p - p
        # restrict r and e
        r_i = torch.zeros(config.nele*p_nloc(p), ndim, device=config.dev, dtype=torch.float64)
        e_i = torch.zeros(config.nele*p_nloc(p), ndim, device=config.dev, dtype=torch.float64)
        for idim in range(ndim):
            r_i[:, idim] += p_restrict(rr_i[:, idim], p+1, p)
            e_i[:, idim] += p_restrict(e_p[ilevel-1][:, idim], p+1, p)
        r_p.append(r_i)
        e_p.append(e_i)
        # pre-smooth
        for its1 in range(config.pre_smooth_its):
            _, e_p[ilevel] = volume_mf_linear_elastic.pmg_get_residual_and_smooth_once(
                r_p[ilevel], e_p[ilevel], p)
        # get residual on this level
        rr_i = volume_mf_linear_elastic.pmg_get_residual_only(r_p[ilevel], e_p[ilevel], p)
    return r_p, e_p


def p_mg_post(e_p, r_p):
    """
    do p-multigrid post-smooth
    """
    ele_p = config.ele_p
    for p in range(1, ele_p):
        ilevel = ele_p - p
        # post smooth
        for its1 in range(config.post_smooth_its):
            _, e_p[ilevel] = volume_mf_linear_elastic.pmg_get_residual_and_smooth_once(
                r_p[ilevel], e_p[ilevel], p)
        # prolongation and correct error
        for idim in range(ndim):
            e_p[ilevel-1][:,idim] += p_prolongate(e_p[ilevel][:,idim], p, p+1)
    return r_p, e_p


def p_prolongate(x, p_in, p_out):
    """
    prolongate a scalar field x
    from p_in order mesh to p_out order mesh
    """
    y = x
    for p in range(p_in, p_out):
        y, _ = _p_prolongate_1level(y, p)
    return y


def p_restrict(x, p_in, p_out):
    """
    restrict a scalar field x
    from p_in order mesh to p_out order mesh
    """
    y = x
    for p in range(p_in, p_out, -1):
        y, _ = _p_restrict_1level(y, p)
    return y


def p_prolongator(p_in, p_out):
    """
    return an element-wise prolongator
    from p_in order grid to p_out order grid
    """
    I = _p_prolongator_1level(p_in)
    for p in range(p_in+1, p_out):
        I = torch.matmul(_p_prolongator_1level(p), I)
    return I


def p_restrictor(p_in, p_out):
    """
    return an element-wise restrictor
    from p_in order grid to p_out order grid
    """
    I = _p_restrictor_1level(p_in)
    for p in range(p_in-1, p_out, -1):
        I = torch.matmul(_p_restrictor_1level(p), I)
    return I


def _p_prolongate_1level(x, p_in):
    """
    prolongate to next p level.
    input is on p_in order mesh,
    output is on p_in+1 order mesh.
    input should be in shape (nele, nloc(p_in))
    """
    y = torch.einsum('ij,kj->ki',
                     _p_prolongator_1level(p_in),
                     x.view(config.nele, p_nloc(p_in))).contiguous().view(-1)
    return y, p_in+1


def _p_restrict_1level(x, p_in):
    """
    restrict to next p level.
    input is on p_in order mesh,
    output is on p_in-1 order mesh.
    input should be in shape (nele, nloc(p_in))
    """
    y = torch.einsum('ij,kj->ki',
                     _p_restrictor_1level(p_in),
                     x.view(config.nele, p_nloc(p_in))).contiguous().view(-1)
    return y, p_in-1


def _p_prolongator_1level(p_in):
    """
    retrun element-wise prolongator from p_in level to p_in+1 level
    """
    if config.ndim == 3:
        if p_in == 2:
            I = torch.tensor([
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [2 / 9, 0, -1 / 9, 0, 0, 8 / 9, 0, 0, 0, 0],
                [-1 / 9, 0, 2 / 9, 0, 0, 8 / 9, 0, 0, 0, 0],
                [2 / 9, -1 / 9, 0, 0, 0, 0, 8 / 9, 0, 0, 0],
                [-1 / 9, 2 / 9, 0, 0, 0, 0, 8 / 9, 0, 0, 0],
                [0, 2 / 9, -1 / 9, 0, 8 / 9, 0, 0, 0, 0, 0],
                [0, -1 / 9, 2 / 9, 0, 8 / 9, 0, 0, 0, 0, 0],
                [2 / 9, 0, 0, -1 / 9, 0, 0, 0, 8 / 9, 0, 0],
                [-1 / 9, 0, 0, 2 / 9, 0, 0, 0, 8 / 9, 0, 0],
                [0, 2 / 9, 0, -1 / 9, 0, 0, 0, 0, 8 / 9, 0],
                [0, -1 / 9, 0, 2 / 9, 0, 0, 0, 0, 8 / 9, 0],
                [0, 0, 2 / 9, -1 / 9, 0, 0, 0, 0, 0, 8 / 9],
                [0, 0, -1 / 9, 2 / 9, 0, 0, 0, 0, 0, 8 / 9],
                [0, -1 / 9, -1 / 9, -1 / 9, 4 / 9, 0, 0, 0, 4 / 9, 4 / 9],
                [-1 / 9, -1 / 9, -1 / 9, 0, 4 / 9, 4 / 9, 4 / 9, 0, 0, 0],
                [-1 / 9, 0, -1 / 9, -1 / 9, 0, 4 / 9, 0, 4 / 9, 0, 4 / 9],
                [-1 / 9, -1 / 9, 0, -1 / 9, 0, 0, 4 / 9, 4 / 9, 4 / 9, 0],
            ], device=config.dev, dtype=torch.float64)  # P2DG to P3DG, element-wise prolongation operator
        elif p_in == 1:
            I = torch.tensor([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 1 / 2, 1 / 2, 0],
                [1 / 2, 0, 1 / 2, 0],
                [1 / 2, 1 / 2, 0, 0],
                [1 / 2, 0, 0, 1 / 2],
                [0, 1 / 2, 0, 1 / 2],
                [0, 0, 1 / 2, 1 / 2],
            ], device=config.dev, dtype=torch.float64)  # P1DG to P2DG, element-wise prolongation operator
        else:
            raise Exception('input order for prolongator should be 1 or 2!')
    # otherwise its 2D
    else:
        if p_in == 2:
            I = torch.tensor([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [2 / 9, -1 / 9, 0, 8 / 9, 0, 0],
                [-1 / 9, 2 / 9, 0, 8 / 9, 0, 0],
                [0, 2 / 9, -1 / 9, 0, 0, 8 / 9],
                [0, -1 / 9, 2 / 9, 0, 0, 8 / 9],
                [-1 / 9, 0, 2 / 9, 0, 8 / 9, 0],
                [2 / 9, 0, -1 / 9, 0, 8 / 9, 0],
                [-1 / 9, -1 / 9, -1 / 9, 4 / 9, 4 / 9, 4 / 9]
            ], device=config.dev, dtype=torch.float64)  # P2DG to P3DG, element-wise prolongation operator
        elif p_in == 1:
            I = torch.tensor([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1 / 2, 1 / 2, 0],
                [1 / 2, 0, 1 / 2],
                [0, 1 / 2, 1 / 2]
            ], device=config.dev, dtype=torch.float64)  # P1DG to P2DG, element-wise prolongation operator
        else:
            raise Exception('input order for prolongator should be 1 or 2!')
    return I


def _p_restrictor_1level(p_in):
    """
    return element-wise restrictor from p_in level to p_in-1 level
    """
    if config.ndim == 3:
        if p_in == 3:
            I = torch.tensor([
                [1, 0, 0, 0, 2 / 9, -1 / 9, 2 / 9, -1 / 9, 0, 0, 2 / 9, -1 / 9, 0, 0, 0, 0, 0, -1 / 9, -1 / 9, -1 / 9],
                [0, 1, 0, 0, 0, 0, -1 / 9, 2 / 9, 2 / 9, -1 / 9, 0, 0, 2 / 9, -1 / 9, 0, 0, -1 / 9, -1 / 9, 0, -1 / 9],
                [0, 0, 1, 0, -1 / 9, 2 / 9, 0, 0, -1 / 9, 2 / 9, 0, 0, 0, 0, 2 / 9, -1 / 9, -1 / 9, -1 / 9, -1 / 9, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1 / 9, 2 / 9, -1 / 9, 2 / 9, -1 / 9, 2 / 9, -1 / 9, 0, -1 / 9, -1 / 9],
                [0, 0, 0, 0, 0, 0, 0, 0, 8 / 9, 8 / 9, 0, 0, 0, 0, 0, 0, 4 / 9, 4 / 9, 0, 0],
                [0, 0, 0, 0, 8 / 9, 8 / 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4 / 9, 4 / 9, 0],
                [0, 0, 0, 0, 0, 0, 8 / 9, 8 / 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4 / 9, 0, 4 / 9],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8 / 9, 8 / 9, 0, 0, 0, 0, 0, 0, 4 / 9, 4 / 9],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8 / 9, 8 / 9, 0, 0, 4 / 9, 0, 0, 4 / 9],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8 / 9, 8 / 9, 4 / 9, 0, 4 / 9, 0],
            ], device=config.dev, dtype=torch.float64)  # P3DG to P2DG, element-wise restriction operator
        elif p_in == 2:
            I = torch.tensor([
                [1, 0, 0, 0, 0, 1 / 2, 1 / 2, 1 / 2, 0, 0],
                [0, 1, 0, 0, 1 / 2, 0, 1 / 2, 0, 1 / 2, 0],
                [0, 0, 1, 0, 1 / 2, 1 / 2, 0, 0, 0, 1 / 2],
                [0, 0, 0, 1, 0, 0, 0, 1 / 2, 1 / 2, 1 / 2],
            ], device=config.dev, dtype=torch.float64)  # P2DG to P1DG, element-wise restriction operator
        else:
            raise Exception('input order for restrictor should be 3 or 2!')
    else:  # otherwise its 2D
        if p_in == 3:
            I = torch.tensor([
                [1, 0, 0, 2 / 9, -1 / 9, 0, 0, -1 / 9, 2 / 9, -1 / 9],
                [0, 1, 0, -1 / 9, 2 / 9, 2 / 9, -1 / 9, 0, 0, -1 / 9],
                [0, 0, 1, 0, 0, -1 / 9, 2 / 9, 2 / 9, -1 / 9, -1 / 9],
                [0, 0, 0, 8 / 9, 8 / 9, 0, 0, 0, 0, 4 / 9],
                [0, 0, 0, 0, 0, 0, 0, 8 / 9, 8 / 9, 4 / 9],
                [0, 0, 0, 0, 0, 8 / 9, 8 / 9, 0, 0, 4 / 9]
            ], device=config.dev, dtype=torch.float64)  # P3DG to P2DG, element-wise restriction operator
        elif p_in == 2:
            I = torch.tensor([
                [1, 0, 0, 1 / 2, 1 / 2, 0],
                [0, 1, 0, 1 / 2, 0, 1 / 2],
                [0, 0, 1, 0, 1 / 2, 1 / 2]
            ], device=config.dev, dtype=torch.float64)  # P2DG to P1DG, element-wise restriction operator
        else:
            raise Exception('input order for restrictor should be 3 or 2!')
    return I


def p_nloc(p):
    """
    return nloc in p order grid
    """
    if config.ndim == 3:
        nloc = [1, 4, 10, 20]
        return nloc[p]
    else:  # its 2D
        nloc = [1, 3, 6, 10]
        return nloc[p]
