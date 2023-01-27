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
    a_sfc, b_sfc, ml_sfc):
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
            b vector in SFC ordering for specified level
        diagonal : torch tensor, (1,nonods)
            Diagonal values in a_sfc_level_sparse
        nonods: scalar, 
            Number of nodes in specified level
    '''
    # level 0 is the highest level
    # level nlevel-1 is the lowest level
    # subtracting 1 because fortran indexes from 1 but python indexes from 0
    # print(level)
    start_index = fin_sfc_nonods[level] - 1
    end_index = fin_sfc_nonods[level + 1] - 1
    # print(start_index,end_index)
    nonods = end_index - start_index
    a_sfc_level = torch.zeros((nonods,nonods), dtype=torch.float64, device=config.dev)
    diagonal = torch.zeros((1,nonods), dtype=torch.float64, device=config.dev)
    #---------------------------- get a_sfc_level-------------------------------#
    for i in range(start_index,end_index):
    #    print(i,fina_sfc_all_un[i],fina_sfc_all_un[i+1]-1)
    #    print("-------")
        for j in range(fina_sfc_all_un[i]-1,fina_sfc_all_un[i+1]-1):
            # print(j,i-start_index,cola_sfc_all_un[j]-1-start_index)
            a_sfc_level[i-start_index][cola_sfc_all_un[j]-1-start_index] = a_sfc[j]
    #    print("-------")
    
    # find the diag index
    # ---------------------------- get a_sfc_level-------------------------------#

    #print(diagonal.shape)
    #print("=======")
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
    #----------- convert a_sfc_level to sparse --------------#
    # get indices
    a_sfc_level_indices = a_sfc_level.nonzero().t()
    # get values
    a_sfc_level_values = a_sfc_level[a_sfc_level_indices[0], a_sfc_level_indices[1]]
    a_sfc_level_sparse = torch.sparse_coo_tensor(
        a_sfc_level_indices, 
        a_sfc_level_values, 
        (nonods, nonods), 
        dtype=torch.float64,
        device=config.dev)
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
    return a_sfc_level_sparse, b_sfc_level, diagonal, nonods

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
    
    e_ip1 = torch.sparse.mm(a_sfc_sparse, e_i.view(nonods_level,1))
    e_ip1 = b.view(-1) - e_ip1.view(-1)
    e_ip1 = e_i.view(-1) + config.jac_wei * e_ip1 / diag_weights.view(-1)
        
    return e_ip1


def mg_on_P0DG(r1, RAR, diagRAR):
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

    RAR : torch csr sparse matrix, (nele, nele)
        operator on level-1 grid (P0DG). One can get the 
        sparsity via native torch csr tensor members.

        e.g. : 

        fina = S.crow_indices().cpu().numpy()

        cola = S.col_indices().cpu().numpy()

        vals = S.values().cpu().numpy()

    diagRAR : torch tensor, (nele, 1)
        diagonal of RAR matrix

    # output

    r1 : torch tensor, (nele, 1)
        corrected residual on P0DG. To be passed to P0DG 
        smoother.
    '''

    ## transfer data to cpu as well as numpy array
    r1_np = r1.detach().clone().cpu().numpy()

    ## get SFC
    cola = RAR.col_indices().detach().clone().cpu().numpy()
    fina = RAR.crow_indices().detach().clone().cpu().numpy()
    vals = RAR.values().detach().clone().cpu().numpy()
    starting_node = 1 # setting according to BY
    graph_trim = -10  # ''
    ncurve = 1        # '' 
    nele = config.nele # this is the shape of RAR # or should we use RAR.shape[0] for clarity? 
    ncola = cola.shape[0]

    whichd, space_filling_curve_numbering = \
        sf.ncurve_python_subdomain_space_filling_curve( \
        cola+1, fina+1, starting_node, graph_trim, ncurve, \
        [nele, ncola]) # note that fortran array index start from 1, so cola and fina should +1.
    np.savetxt('sfc.txt', space_filling_curve_numbering, delimiter=',')
    ## ordering node according to SFC
    N = len(space_filling_curve_numbering)
    inverse_numbering = np.zeros((N, ncurve), dtype=int)
    inverse_numbering[:, 0] = np.argsort(space_filling_curve_numbering[:, 0])

    ## get coarse grid info
    max_nlevel = sf.calculate_nlevel_sfc(nele) + 1
    max_nonods_sfc_all_grids = 5*config.nele 
    max_ncola_sfc_all_un = 10*ncola

    a_sfc, fina_sfc_all_un, cola_sfc_all_un, ncola_sfc_all_un, b_sfc, \
        ml_sfc, fin_sfc_nonods, nonods_sfc_all_grids, nlevel = \
        map_sfc.best_sfc_mapping_to_sfc_matrix_unstructured(\
            a=vals,b=r1_np[:,0], ml=r1_np[:,0],\
            fina=fina+1,cola=cola+1, \
            sfc_node_ordering=space_filling_curve_numbering[:,0], \
            max_nonods_sfc_all_grids=max_nonods_sfc_all_grids, \
            max_ncola_sfc_all_un=max_ncola_sfc_all_un, \
            max_nlevel=max_nlevel)
    nodes_per_level = [fin_sfc_nonods[i] - fin_sfc_nonods[i-1] for i in range(1, nlevel+1)]
    
    a_sfc = torch.from_numpy(a_sfc).to(device=config.dev)
    b_sfc = torch.from_numpy(b_sfc)
    ml_sfc = torch.from_numpy(ml_sfc)
    variables_sfc = []
    for level in range(nlevel):
        variables_sfc.append(get_a_ml_b(level, \
            fin_sfc_nonods, \
            fina_sfc_all_un, \
            cola_sfc_all_un, \
            a_sfc, b_sfc, ml_sfc))

    ## get residual on each level
    sfc_restrictor = torch.nn.Conv1d(in_channels=1, \
        out_channels=1, kernel_size=2, \
        stride=2, padding='valid', bias=False)
    sfc_restrictor.weight.data = \
        torch.tensor([[0.5, 0.5]], \
        dtype=torch.float64, \
        device=config.dev).view(1, 1, 2)

    e_i = torch.zeros([1,1,2], device=config.dev, dtype=torch.float64)
    r = r1[inverse_numbering[:,0],0].view(1,1,nele)
    r_s = []
    r_s.append(r)
    for i in range(1,nlevel-1):
        # pad one node with same value as final node so that odd nodes won't be joined
        r = F.pad(r, (0,1), "constant", 0)
        r[0,0,r.shape[2]-1] = r[0,0,r.shape[2]-2]
        with torch.no_grad():
            r = sfc_restrictor(r)
        r_s.append(r)
    
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
    e_i = e_i[0,0,space_filling_curve_numbering[:,0]-1]
    # correct r1 residual
    r1 = e_i.view(-1) + torch.tensor(r1_np,device=config.dev).view(-1)

    return r1.view(-1,1)