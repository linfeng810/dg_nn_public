#!/usr/bin/env python3
import os.path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import config
from config import sf_nd_nb
import sfc as sf # to be compiled ...
import map_to_sfc_matrix as map_sfc
import volume_integral
import shape_function


def get_a_ml_b(level, \
    fin_sfc_nonods, \
    fina_sfc_all_un, \
    cola_sfc_all_un, \
    a_sfc, b_sfc=1, ml_sfc=1):
    '''Function to get a, b ml and ml inverse in sfc ordering for each level as sparse matrices
    
        Input
        -----
        level: Level to get sfc ordering starting from 0 as the first
        fin_sfc_nonods: List containing the starting index in a_sfc of each level
        fina_sfc_all_un: fina on every level 
        cola_sfc_all_un: cola on every level
        a_sfc = list containing a_sfcs in different SFC ordering
        b_sfc = list containing b_sfcs in different SFC ordering
        ml_sfc = list containing ml_sfcs in different SFC ordering
        
        Output
        ------
        a_sfc_level_sparse : torch coo sparse tensor, (nonods, nonods)
            a matrix in SFC ordering for specified level
        b_sfc_level : torch tensor, (nonods)
            b vector in SFC ordering for specified level. (DEPRECATED)
        diagonal : torch tensor, (1,nonods)
            Diagonal values in a_sfc_level_sparse
        nonods: scalar, 
            Number of nodes in specified level
    '''
    # level 0 is the highest level
    # level nlevel-1 is the lowest level
    # subtracting 1 because fortran indexes from 1 but python indexes from 0
    
    start_index = fin_sfc_nonods[level] - 1
    end_index = fin_sfc_nonods[level + 1] - 1
    nonods = end_index - start_index
    
    diagonal = torch.zeros((1,nonods), dtype=config.dtype, device=config.dev)
    a_indices = []
    a_values = []
    #---------------------------- get a_sfc_level-------------------------------#
    for i in range(start_index,end_index):
        for j in range(fina_sfc_all_un[i]-1,fina_sfc_all_un[i+1]-1):
            a_indices.append([i-start_index, cola_sfc_all_un[j]-1-start_index])
            a_values.append(a_sfc[j])
            
    a_indices = np.asarray(a_indices).transpose()
    #----------- convert a_sfc_level to sparse --------------#
    a_sfc_level_sparse = torch.sparse_coo_tensor(
        a_indices, 
        a_values, 
        (nonods, nonods), 
        dtype=config.dtype,
        device=config.dev)
    a_sfc_level_sparse = a_sfc_level_sparse.to_sparse_csr()

    # find the diag index
    #print(diagonal.shape)
    # print("=======")
    for i in range(nonods):
        i = i + start_index
        # print('i=',i, 'j_start:', fina_sfc_all_un[i]-1, 'j_end:', fina_sfc_all_un[i+1]-1)
        # print('---------')
        for j in range(fina_sfc_all_un[i]-1,fina_sfc_all_un[i+1]-1):
            temp = cola_sfc_all_un[j]-1
            # print('j=',j,i-start_index,temp)
            if (i==temp):
                # print(i,j)
                diagonal[0][i-start_index] = a_sfc[j]
        # print('---------')
    if (False):
        #-----------get ml_sfc_level and its inverse and convert  to sparse ----------#
        ml_sfc_level = ml_sfc[start_index:end_index]
        #indices
        ml_sfc_level_indices = torch.zeros((2,nonods), dtype=config.dtype)
        ml_sfc_level_indices[0] = torch.Tensor([i for i in range(nonods)])
        ml_sfc_level_indices[1] = ml_sfc_level_indices[0]
        ml_sfc_level_indices = ml_sfc_level_indices.int()
        # convert to sparse
        ml_sfc_level_sparse = torch.sparse_coo_tensor(ml_sfc_level_indices, ml_sfc_level, (nonods, nonods))
        # inverse
        ml_sfc_level_sparse_inv = torch.sparse_coo_tensor(ml_sfc_level_indices, 1/ml_sfc_level, (nonods, nonods))
        #---------------------------- get b_sfc -----------------------------------#
        b_sfc_level = b_sfc[start_index:end_index]
        #----------------------- Divide each one by ml_sfc_level ------------------#
        # if scale_matrices == True: 
        #     a_sfc_level_sparse = torch.sparse.mm(ml_sfc_level_sparse_inv, a_sfc_level_sparse)
        #     b_sfc_level = b_sfc_level/ml_sfc_level
        #  ml_sfc_level_sparse, ml_sfc_level_sparse_inv,
        np.savetxt('a_sfc_'+str(level)+'.txt', a_sfc_level_sparse.to_dense().cpu().numpy(), delimiter=',')
    if type(ml_sfc) == int:
        ml_sfc_level = 1
    else:
        ml_sfc_level = torch.tensor(ml_sfc[start_index:end_index],
                                    device=config.dev,
                                    dtype=config.dtype)
    return a_sfc_level_sparse, ml_sfc_level, diagonal, nonods


def mg_smooth(level, e_i, b, variables_sfc):
    '''
    Do one smooth step on *level*-th grid.

    # Input
    level : integer
        level of grid
    e_i : torch tensor (nonods_level[level])
        error on level-th grid at i-th smooth step.
    b : torch tensor (nonods_level[level])
        residual on level-th grid
    variable_sfc : list (level)
        a list of all ingredients one needs to perform a smoothing
        step on level-th grid. Each list member is a list of the 
        following member:
        a_sfc_sparse : torch coo sparse tensor, (nonods_level, nonods_level)
            coarse level grid operator
        b_sfc_sparse : torch tensor, (nonods_level)
            should be residual but didn't use here.
        diag_weights : torch tensor, (nonods_level)
            diagonal of coarse grid operator
        nonods : integer
            number of nodes on level-th grid

    # Output
    e_i : torch tensor (nonods_level[level])
        error after smooth (at i+1-th smooth step)
    res_this_level : torch tensor (nonods_level[level])
        residual on this level (at i+1-th smooth step)
    '''
    # get a_sfc and ml_sfc for this level and curve
    a_sfc_sparse, _, diag_weights, nonods_level = variables_sfc[level]
    # np.savetxt('a_sfc_'+str(level)+'.txt', a_sfc_sparse.to_dense().cpu().numpy(), delimiter=',')
    res_this_level = torch.sparse.mm(a_sfc_sparse, e_i.view(-1,1)).view(-1)
    res_this_level *= -1.
    res_this_level += b.view(-1)
    # print(diag_weights.view(-1).min(), diag_weights.view(-1).max())
    e_i = e_i.view(-1)
    e_i += config.jac_wei * res_this_level / diag_weights.view(-1)
    # print('a ', a_sfc_sparse.to_dense())
    # print('b ', b)
    # print('diag ', diag_weights)
    # print('e_i ', e_i)
    return e_i, res_this_level

def mg_on_P1CG_prep(RAR):
    '''
    # Prepare for Multi-grid cycle on P0DG mesh

    This function forms space filling curve. Then form
    a series of coarse grid and operators thereon.

    # Input 
    RAR : torch csr sparse matrix, (nele, nele)
        operator on level-1 grid (P0DG). One can get the 
        sparsity via native torch csr tensor members.

        e.g. : 

        fina = S.crow_indices().cpu().numpy()

        cola = S.col_indices().cpu().numpy()

        vals = S.values().cpu().numpy()

    # output

    sfc : numpy list (nele)
        space filling curve index for ele.
    variables_sfc : list (nlevel)
        a list of all ingredients one needs to perform a smoothing
        step on level-th grid. Each list member is a list of the 
        following member:
        a_sfc_sparse : torch coo sparse tensor, (nonods_level, nonods_level)
            coarse level grid operator
        b_sfc_sparse : torch tensor, (nonods_level)
            should be residual but didn't use here. (DEPRECATED)
        diag_weights : torch tensor, (nonods_level)
            diagonal of coarse grid operator
        nonods : integer
            number of nodes on level-th grid
    nlevel : scalar, int
        number of SFC coarse grid levels
    nodes_per_level : list of int, (nlevel)
        number of nodes (DOFs) on each level
    '''

    # np.savetxt('RAR.txt', RAR.to_dense().cpu().numpy(), delimiter=',')
    ## get SFC
    cola = RAR.col_indices().detach().clone().cpu().numpy()
    fina = RAR.crow_indices().detach().clone().cpu().numpy()
    vals = RAR.values().detach().clone().cpu().numpy()
    nonods = RAR.shape[0] # this is the shape of RAR # or should we use RAR.shape[0] for clarity?
    starting_node = 1 # setting according to BY
    graph_trim = -10  # ''
    ncurve = 1        # ''
    ncola = cola.shape[0]
    dummy_vec = np.zeros(nonods)
    import time
    start_time = time.time()
    print('to get space filling curve...', time.time()-start_time)
    # if os.path.isfile(config.filename[:-4] + '_sfc.npy'):
    if False:
        print('pre-calculated sfc exists. readin from file...')
        sfc = np.load(config.filename[:-4] + '_sfc.npy')
    else:
        _, sfc = \
            sf.ncurve_python_subdomain_space_filling_curve( \
            cola+1, fina+1, starting_node, graph_trim, ncurve, \
            ) # note that fortran array index start from 1, so cola and fina should +1.
        np.save(config.filename[:-4] + '_sfc.npy', sfc)
    print('to get sfc operators...', time.time()-start_time)
    ## get coarse grid info
    max_nlevel = sf.calculate_nlevel_sfc(nonods) + 1
    max_nonods_sfc_all_grids = 5*nonods
    max_ncola_sfc_all_un = 10*ncola

    if config.is_mass_weighted:
        ml = get_p1cg_lumped_mass(sf_nd_nb.x_ref_in)
        ml = ml.cpu().numpy()
    else:
        ml = dummy_vec
    a_sfc, fina_sfc_all_un, cola_sfc_all_un, ncola_sfc_all_un, b_sfc, \
        ml_sfc, fin_sfc_nonods, nonods_sfc_all_grids, nlevel = \
        map_sfc.best_sfc_mapping_to_sfc_matrix_unstructured(\
            a=vals,b=dummy_vec, ml=ml,\
            fina=fina+1,cola=cola+1, \
            sfc_node_ordering=sfc[:,0], \
            max_nonods_sfc_all_grids=max_nonods_sfc_all_grids, \
            max_ncola_sfc_all_un=max_ncola_sfc_all_un, \
            max_nlevel=max_nlevel, \
            ismasswt=config.is_mass_weighted)
    print('back from sfc operator fortran subroutine,', time.time()-start_time)
    nodes_per_level = [fin_sfc_nonods[i] - fin_sfc_nonods[i-1] for i in range(1, nlevel+1)]
    a_sfc = torch.from_numpy(a_sfc[:ncola_sfc_all_un]).to(device=config.dev)

    variables_sfc = []
    for level in range(nlevel):
        variables_sfc.append(get_a_ml_b(level,
                                        fin_sfc_nonods,
                                        fina_sfc_all_un,
                                        cola_sfc_all_un,
                                        a_sfc,
                                        ml_sfc=ml_sfc))
    print('after forming sfc operator torch csr tensors', time.time()-start_time)
    # choose a level to directly solve on. then we'll iterate from there and levels up
    if config.smooth_start_level < 0:
        # for level in range(1,nlevel):
        #     if nodes_per_level[level] < 2:
        #         config.smooth_start_level = level
        #         break
        config.smooth_start_level += nlevel
    print('start_level: ', config.smooth_start_level)
    return sfc, variables_sfc, nlevel, nodes_per_level

def mg_on_P1CG(r0, rr0, e_i0, sfc, variables_sfc, nlevel, nodes_per_level):
    '''
    # Multi-grid cycle on P0DG mesh

    This function takes in residual on P0DG mesh, forms 
    a series of coarse grid via space-filling-curve, then 
    smooths the residual on each level.
    '''

    # restrict residual
    sfc_restrictor = torch.nn.Conv1d(in_channels=1,
                                     out_channels=1, kernel_size=2,
                                     stride=2, padding='valid', bias=False)
    sfc_restrictor.weight.data = \
        torch.tensor([[1., 1.]],
                     dtype=config.dtype,
                     device=config.dev).view(1, 1, 2)

    # ## ordering node according to SFC
    # ncurve = 1 # always use 1 sfc
    # N = len(sfc)
    # inverse_numbering = np.zeros((N, ncurve), dtype=int)
    # inverse_numbering[:, 0] = np.argsort(sfc[:, 0])
    #
    # r = r1[inverse_numbering[:,0],0].view(1,1,config.cg_nonods)

    smooth_start_level = config.smooth_start_level
    r = r0.view(1,1,sf_nd_nb.cg_nonods)  # residual r in error equation, Ae=r
    e_i = e_i0.view(1, 1, sf_nd_nb.cg_nonods)  # error
    r_s = [r]  # collection of r
    e_s = [e_i.view(-1)]  # collec. of e, all stored as 1D tensor
    rr = rr0
    for i in range(1,smooth_start_level):
        # restriction
        if config.is_mass_weighted:
            rr = torch.mul(rr, variables_sfc[i-1][1].view(1,1,-1))
            e_s[i-1] = torch.mul(e_s[i-1], variables_sfc[i-1][1].view(1,1,-1))
        # pad one node with same value as final node so that odd nodes won't be joined
        rr = F.pad(rr, (0,1), "constant", 0)
        e_i = F.pad(e_s[i-1].view(1,1,-1), (0,1), "constant", 0)
        with torch.no_grad():
            rr = sfc_restrictor(rr)
            r_s.append(rr)
            e_i = sfc_restrictor(e_i)
            e_s.append(e_i.view(-1))

        # pre-smooth
        for its in range(config.pre_smooth_its):
            e_s[i], _ = mg_smooth(
                level=i,
                e_i=e_s[i],
                b=r_s[i],
                variables_sfc=variables_sfc)
        # after presmooth, get residual on this level
        _, rr = mg_smooth(
            level=i,
            e_i=e_s[i],
            b=r_s[i],
            variables_sfc=variables_sfc)
        rr = rr.view(1, 1, nodes_per_level[i])
    if smooth_start_level > 0:
        if config.is_mass_weighted:
            rr = torch.mul(rr, variables_sfc[smooth_start_level-1][1].view(1,1,-1))
        # restrict residual to smooth_start_level
        rr = F.pad(rr, (0, 1), "constant", 0)
        with torch.no_grad():
            rr = sfc_restrictor(rr).view(1, 1, nodes_per_level[smooth_start_level])
            r_s.append(rr)
        e_s.append(torch.zeros_like(rr.view(-1), device=config.dev, dtype=config.dtype))  # 占个坑
    # mg sweep on SFC levels (level up)
    for level1 in reversed(range(0,1)):  # if bunny-net cycle, use range(0,smooth_start_level)
        for level in reversed(range(level1,smooth_start_level+1)):
            if level == smooth_start_level:  # direct solve on start_level
                # get operator
                a_sfc_l = variables_sfc[level][0]
                cola = a_sfc_l.col_indices().detach().clone().cpu().numpy()
                fina = a_sfc_l.crow_indices().detach().clone().cpu().numpy()
                vals = a_sfc_l.values().detach().clone().cpu().numpy()
                # direct solve on level1 sfc coarse grid
                from scipy.sparse import csr_matrix, linalg
                # np.savetxt('a_sfc_l0.txt', variables_sfc[0][0].to_dense().cpu().numpy(), delimiter=',')
                # np.savetxt('a_sfc_l1.txt', variables_sfc[1][0].to_dense().cpu().numpy(), delimiter=',')
                a_on_l = csr_matrix((vals, cola, fina), shape=(nodes_per_level[level], nodes_per_level[level]))
                # np.savetxt('a_sfc_l1_sp.txt', a_on_l1.todense(), delimiter=',')
                e_i_direct = linalg.spsolve(a_on_l, r_s[level].view(-1).cpu().numpy())
                # prolongation
                e_s[level] = torch.tensor(e_i_direct, device=config.dev, dtype=config.dtype)
            else:  # smooth
                CNN1D_prol_odd = nn.Upsample(scale_factor=nodes_per_level[level] / nodes_per_level[level+1])
                if config.is_mass_weighted:
                    e_s[level] += torch.mul(CNN1D_prol_odd(e_s[level + 1].view(1, 1, -1)).view(-1),
                                            variables_sfc[level][1].view(-1))
                else:
                    e_s[level] += CNN1D_prol_odd(e_s[level+1].view(1,1,-1)).view(-1)
                for its in range(config.post_smooth_its):  # sfc level is the level+1 overall level.
                    # (since we have a level0 = PnDG)
                    e_s[level], _ = mg_smooth(
                        level=level,
                        e_i=e_s[level],
                        b=r_s[level],
                        variables_sfc=variables_sfc)
                    # print('  sfc level ', level, 'its ', its_l, 'residual l2norm', torch.linalg.norm(r_i.view(-1),dim=0))
        if level1 == 0:
            break
        # get residual on level1 and restrict to lower level(s)
        _, rr = mg_smooth(
            level=level1,
            e_i=e_s[level],
            b=r_s[level1],
            variables_sfc=variables_sfc)
        rr = rr.view(1, 1, nodes_per_level[level1])
        r_s[level1] = rr
        for i in range(level1, smooth_start_level):
            if config.is_mass_weighted:
                rr = torch.mul(rr, variables_sfc[i - 1][1].view(1, 1, -1))
            # pad one node with same value as final node so that odd nodes won't be joined
            rr = F.pad(rr, (0, 1), "constant", 0)
            # r[0,0,r.shape[2]-1] = r[0,0,r.shape[2]-2]
            with torch.no_grad():
                rr = sfc_restrictor(rr)
            r_s[i+1] = rr

    # # correct r1 residual
    # e_i1p1 = e_i1.view(-1) + e_i.view(-1)

    return e_s[0].view(-1,1)


def p3dg_to_p1dg_restrictor(x):
    '''
    takes in a vector on p3dg, do restriction and spit out
    its projection on p1dg.
    '''
    y = torch.einsum('ij,kj->ki', sf_nd_nb.I_13, x.view(config.nele, config.nloc)).contiguous().view(-1)
    return y


def p1dg_to_p3dg_prolongator(x):
    '''
    takes in a vector on p1dg, do prolongation and spit out
    its projection on p3dg
    '''
    y = torch.einsum('ij,kj->ki', sf_nd_nb.I_31, x.view(config.nele, config.p1cg_nloc)).contiguous().view(-1)
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
    cg_n = torch.tensor(cg_n, device=config.dev, dtype=config.dtype)
    cg_nlx = torch.tensor(cg_nlx, device=config.dev, dtype=config.dtype)
    cg_wt = torch.tensor(cg_wt, device=config.dev, dtype=config.dtype)
    ml = torch.zeros(sf_nd_nb.cg_nonods, device=config.dev, dtype=config.dtype)
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
    e_p = [torch.zeros_like(r0, device=config.dev, dtype=config.dtype)]  # store error on each p level
    rr_i = r0  # residual of error equation Ae=r. rr_i := r_p - A e_i
    for p in range(ele_p-1, 0, -1):
        ilevel = ele_p - p
        # restrict r and e
        r_i = p_restrict(rr_i, p+1, p)
        e_i = p_restrict(e_p[ilevel-1], p+1, p)
        r_p.append(r_i)
        e_p.append(e_i)
        # pre-smooth
        for its1 in range(config.pre_smooth_its):
            _, e_p[ilevel] = volume_integral.pmg_get_residual_and_smooth_once(
                r_p[ilevel], e_p[ilevel], p)
        # get residual on this level
        rr_i = volume_integral.pmg_get_residual_only(r_p[ilevel], e_p[ilevel], p)
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
            _, e_p[ilevel] = volume_integral.pmg_get_residual_and_smooth_once(
                r_p[ilevel], e_p[ilevel], p)
        # prolongation and correct error
        e_p[ilevel-1] += p_prolongate(e_p[ilevel], p, p+1)
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
            ], device=config.dev, dtype=config.dtype)  # P2DG to P3DG, element-wise prolongation operator
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
            ], device=config.dev, dtype=config.dtype)  # P1DG to P2DG, element-wise prolongation operator
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
            ], device=config.dev, dtype=config.dtype)  # P2DG to P3DG, element-wise prolongation operator
        elif p_in == 1:
            I = torch.tensor([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1 / 2, 1 / 2, 0],
                [1 / 2, 0, 1 / 2],
                [0, 1 / 2, 1 / 2]
            ], device=config.dev, dtype=config.dtype)  # P1DG to P2DG, element-wise prolongation operator
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
            ], device=config.dev, dtype=config.dtype)  # P3DG to P2DG, element-wise restriction operator
        elif p_in == 2:
            I = torch.tensor([
                [1, 0, 0, 0, 0, 1 / 2, 1 / 2, 1 / 2, 0, 0],
                [0, 1, 0, 0, 1 / 2, 0, 1 / 2, 0, 1 / 2, 0],
                [0, 0, 1, 0, 1 / 2, 1 / 2, 0, 0, 0, 1 / 2],
                [0, 0, 0, 1, 0, 0, 0, 1 / 2, 1 / 2, 1 / 2],
            ], device=config.dev, dtype=config.dtype)  # P2DG to P1DG, element-wise restriction operator
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
            ], device=config.dev, dtype=config.dtype)  # P3DG to P2DG, element-wise restriction operator
        elif p_in == 2:
            I = torch.tensor([
                [1, 0, 0, 1 / 2, 1 / 2, 0],
                [0, 1, 0, 1 / 2, 0, 1 / 2],
                [0, 0, 1, 0, 1 / 2, 1 / 2]
            ], device=config.dev, dtype=config.dtype)  # P2DG to P1DG, element-wise restriction operator
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
