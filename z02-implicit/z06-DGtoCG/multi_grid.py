#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import config
import sfc as sf # to be compiled ...
import map_to_sfc_matrix as map_sfc
# import space_filling_decomp as sfc

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
    
    diagonal = torch.zeros((1,nonods), dtype=torch.float64, device=config.dev)
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
        dtype=torch.float64,
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
        ml_sfc_level_indices = torch.zeros((2,nonods), dtype=torch.float64)
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
    return a_sfc_level_sparse, 1, diagonal, nonods


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
    e_ip1 : torch tensor (nonods_level[level])
        error after smooth (at i+1-th smooth step)
    '''
    # get a_sfc and ml_sfc for this level and curve
    a_sfc_sparse, _, diag_weights, nonods_level = variables_sfc[level]
    # np.savetxt('a_sfc_'+str(level)+'.txt', a_sfc_sparse.to_dense().cpu().numpy(), delimiter=',')
    e_ip1 = torch.sparse.mm(a_sfc_sparse, e_i.view(nonods_level,1))
    e_ip1 = b.view(-1) - e_ip1.view(-1)
    # print(diag_weights.view(-1).min(), diag_weights.view(-1).max())
    e_ip1 = e_i.view(-1) + config.jac_wei * e_ip1 / diag_weights.view(-1).max()
    # print('a ', a_sfc_sparse.to_dense())
    # print('b ', b)
    # print('diag ', diag_weights)
    # print('e_i ', e_i)
    # print('e_ip1 ', e_ip1)
    return e_ip1

def mg_on_P0DG_prep(RAR):
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
    starting_node = 1 # setting according to BY
    graph_trim = -10  # ''
    ncurve = 1        # ''
    nonods = RAR.shape[0] # this is the shape of RAR # or should we use RAR.shape[0] for clarity?
    ncola = cola.shape[0]
    dummy_vec = np.zeros(nonods)

    whichd, sfc = \
        sf.ncurve_python_subdomain_space_filling_curve( \
        cola+1, fina+1, starting_node, graph_trim, ncurve, \
        ) # note that fortran array index start from 1, so cola and fina should +1.
    # np.savetxt('sfc.txt', sfc[:,0], delimiter=',')
    ## get coarse grid info
    max_nlevel = sf.calculate_nlevel_sfc(nonods) + 1
    max_nonods_sfc_all_grids = 5*nonods
    max_ncola_sfc_all_un = 10*ncola

    a_sfc, fina_sfc_all_un, cola_sfc_all_un, ncola_sfc_all_un, b_sfc, \
        ml_sfc, fin_sfc_nonods, nonods_sfc_all_grids, nlevel = \
        map_sfc.best_sfc_mapping_to_sfc_matrix_unstructured(\
            a=vals,b=dummy_vec, ml=dummy_vec,\
            fina=fina+1,cola=cola+1, \
            sfc_node_ordering=sfc[:,0], \
            max_nonods_sfc_all_grids=max_nonods_sfc_all_grids, \
            max_ncola_sfc_all_un=max_ncola_sfc_all_un, \
            max_nlevel=max_nlevel)
    nodes_per_level = [fin_sfc_nonods[i] - fin_sfc_nonods[i-1] for i in range(1, nlevel+1)]
    a_sfc = torch.from_numpy(a_sfc[:ncola_sfc_all_un]).to(device=config.dev)
    del b_sfc, ml_sfc
    variables_sfc = []
    for level in range(nlevel):
        variables_sfc.append(get_a_ml_b(level,
            fin_sfc_nonods,
            fina_sfc_all_un,
            cola_sfc_all_un,
            a_sfc))
    return sfc, variables_sfc, nlevel, nodes_per_level

def mg_on_P0DG(r1, e_i1, sfc, variables_sfc, nlevel, nodes_per_level):
    '''
    # Multi-grid cycle on P0DG mesh

    This function takes in residual on P0DG mesh, forms 
    a series of coarse grid via space-filling-curve, then 
    smooths the residual on each level until it gets a
    correction :math:`e_1`. Finally, it returns 
    :math:`r_1 <- r_1 + e_1`.

    # Input 

    r1 : torch tensor, (nele, 1)
        residual passed from PnDG mesh via restrictor, 

    e_i1 : torch tensor, (nele, 1)
        previous computed error on P0DG (level-1 grid).

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

    # output

    e_i1p1 : torch tensor, (nele, 1)
        corrected error on P0DG. To be passed to P0DG 
        smoother.
    '''

    ## get residual on each level
    sfc_restrictor = torch.nn.Conv1d(in_channels=1, \
        out_channels=1, kernel_size=2, \
        stride=2, padding='valid', bias=False)
    sfc_restrictor.weight.data = \
        torch.tensor([[1., 1.]], \
        dtype=torch.float64, \
        device=config.dev).view(1, 1, 2)

    ## ordering node according to SFC
    ncurve = 1 # always use 1 sfc
    N = len(sfc)
    inverse_numbering = np.zeros((N, ncurve), dtype=int)
    inverse_numbering[:, 0] = np.argsort(sfc[:, 0])

    r = r1[inverse_numbering[:,0],0].view(1,1,config.nele)
    r_s = []
    r_s.append(r)
    for i in range(1,nlevel):
        # pad one node with same value as final node so that odd nodes won't be joined
        r = F.pad(r, (0,1), "constant", 0)
        # r[0,0,r.shape[2]-1] = r[0,0,r.shape[2]-2]
        with torch.no_grad():
            r = sfc_restrictor(r)
        r_s.append(r)

    ## on 1DOF level
    e_i = r_s[-1]/variables_sfc[-1][2]
    CNN1D_prol_odd = nn.Upsample(scale_factor=nodes_per_level[-2]/nodes_per_level[-1])
    e_i = CNN1D_prol_odd(e_i.view(1,1,-1))
    # e_i = torch.zeros([1,1,2], device=config.dev, dtype=torch.float64)
    ## mg sweep
    for level in reversed(range(1,nlevel-1)):
        for _ in range(config.mg_smooth_its):
            e_i = mg_smooth(level=level, 
                e_i=e_i, 
                b=r_s[level], 
                variables_sfc=variables_sfc)
        CNN1D_prol_odd = nn.Upsample(scale_factor=nodes_per_level[level-1]/nodes_per_level[level])
        e_i = CNN1D_prol_odd(e_i.view(1,1,-1))
    
    # Map e_i to original order
    e_i = e_i[0,0,sfc[:,0]-1]
    # correct r1 residual
    e_i1p1 = e_i1.view(-1) + e_i.view(-1)

    return e_i1p1.view(-1,1)