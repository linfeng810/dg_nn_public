#!/usr/bin/env python3

import numpy as np 
import torch
import sfc 
import map_to_sfc_matrix as map_sfc
import config 

def get_a_ml_b(level, fin_sfc_nonods, a_sfc, b_sfc, ml_sfc):
    '''Function to get a, b ml and ml inverse in sfc ordering for each level as sparse matrices
    
        Input
        ---------------------------------------------------------------------------------------------
        level: Level to get sfc ordering starting from 0 as the first
        curve: the sfc curve to consider
        fin_sfc_nonods: List containing the starting index in a_sfc of each level
        a_sfc = list containing a_sfcs in different SFC ordering
        b_sfc = list containing b_sfcs in different SFC ordering
        ml_sfc = list containing ml_sfcs in different SFC ordering
        
        Output
        ---------------------------------------------------------------------------------------------
        a_sfc_level_sparse: a matrix in SFC ordering for specified level
        b_sfc_level: b vector in SFC ordering for specified level
        diagonal: Diagonal values in a_sfc_level_sparse
        nonods: Number of nodes in specified level
    '''
    # level 0 is the highest level
    # level nlevel-1 is the lowest level
    # subtracting 1 because fortran indexes from 1 but python indexes from 0
    # print(level)
    start_index = fin_sfc_nonods[level] - 1
    end_index = fin_sfc_nonods[level + 1] - 1
    # print(start_index,end_index)
    nonods = end_index - start_index
    a_sfc_level = torch.zeros((nonods,nonods), dtype=torch.float64)
    diagonal = torch.zeros((1,nonods), dtype=torch.float64)
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
    a_sfc_level_sparse = torch.sparse_coo_tensor(a_sfc_level_indices, a_sfc_level_values, (nonods, nonods), dtype=torch.float64)
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


if __name__=='__main__':
    fina = np.genfromtxt('fina.txt', delimiter=',')
    cola = np.genfromtxt('cola.txt', delimiter=',')
    vals = np.genfromtxt('vals.txt', delimiter=',')
    space_filling_curve_numbering = np.genfromtxt('sfc.txt', delimiter=',')
    r1_np = np.genfromtxt('r1.txt', delimiter=',')
    ml_np = np.genfromtxt('r1.txt', delimiter=',')

    ncola = cola.shape[0]
    ## get coarse grid info
    max_nlevel = sfc.calculate_nlevel_sfc(config.nele) + 1
    max_nonods_sfc_all_grids = 5*config.nele 
    max_ncola_sfc_all_un = 10*ncola

    a_sfc, fina_sfc_all_un, cola_sfc_all_un, ncola_sfc_all_un, b_sfc, \
        ml_sfc, fin_sfc_nonods, nonods_sfc_all_grids, nlevel = \
        map_sfc.best_sfc_mapping_to_sfc_matrix_unstructured(a=vals,b=r1_np, ml=ml_np,\
            fina=fina+1,cola=cola+1, \
            sfc_node_ordering=space_filling_curve_numbering, \
            max_nonods_sfc_all_grids=max_nonods_sfc_all_grids, \
            max_ncola_sfc_all_un=max_ncola_sfc_all_un, \
            max_nlevel=max_nlevel)

    print('a_sfc', a_sfc.dtype)
    print('fina_sfc_all_un', fina_sfc_all_un)
    print("cola_sfc_all_un", cola_sfc_all_un)
    print('ncola_sfc_all_un', ncola_sfc_all_un)
    print('b_sfc', b_sfc)
    print('ml_sfc', ml_sfc)
    print('fin_sfc_nonods', fin_sfc_nonods)
    print('nonods_sfc_all_grids', nonods_sfc_all_grids)
    print('nlevel', nlevel)

    print('what is going wrong...')
    print('========================================')

    print(a_sfc.dtype)
    a_sfc = torch.from_numpy(a_sfc)
    print(a_sfc.dtype)
    b_sfc = torch.Tensor(b_sfc)
    ml_sfc = torch.Tensor(ml_sfc)

    levels = [level for level in range(nlevel)]
    print(levels)
    # ------------------------------- FIRST CURVE -----------------------------#
    # variables_sfc = list(map(get_a_ml_b, levels, [fin_sfc_nonods], [a_sfc], [b_sfc], [ml_sfc]))
    variables_sfc = []
    for level in range(nlevel):
        variables_sfc.append(get_a_ml_b(level, fin_sfc_nonods, a_sfc, b_sfc, ml_sfc))

    print(variables_sfc[0][0].to_dense())
    np.savetxt('variable_sfc00.txt', variables_sfc[0][0].to_dense().cpu().detach(), delimiter=',')
