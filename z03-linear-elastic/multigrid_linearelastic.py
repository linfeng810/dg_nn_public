#!/usr/bin/env python3
'''
This file has functions related  to 
multi-grid cycle on SFC level 
coarsened grids for linear elastic problems
(where operators are (ndim, ndim) tensors)
'''
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import config
import sfc as sf # to be compiled ...
import map_to_sfc_matrix as map_sfc

ndim = config.ndim

# mg on p0dg smooth
def mg_smooth(r1, sfc, variables_sfc, nlevel, nodes_per_level):
    '''
    # Multi-grid cycle on P0DG mesh

    This function takes in residual on P0DG mesh, forms 
    a series of coarse grid via space-filling-curve, then 
    smooths the residual on each level until it gets a
    correction :math:`e_1`. Finally, it returns 
    :math:`r_1 <- r_1 + e_1`.

    # Input 

    r1 : torch tensor, (ndim, nele)
        residual passed from PnDG mesh via restrictor, 

    ~~e_i1 : torch tensor, (ndim, nele)~~
        ~~previous computed error on P0DG (level-1 grid).~~

    sfc : numpy list (nele)
        space filling curve index for ele.
    variables_sfc : list (nlevel)
        a list of all ingredients one needs to perform a smoothing
        step on level-th grid. Each list member is a list of the 
        following member:
        a_sfc_sparse : list of torch coo sparse tensor, len= ndim x ndim
            each member is the (idim,jdim) block of a_sfc_sparse --
            coarse level grid operator, (nonods_level, nonods_level)
        diag_weights : torch tensor, (ndim, nonods_level)
            diagonal of coarse grid operator
        nonods : integer
            number of nodes on level-th grid
    nlevel : scalar, int
        number of SFC coarse grid levels
    nodes_per_level : list of int, (nlevel)
        number of nodes (DOFs) on each level

    # output

    e_i1p1 : torch tensor, (ndim, nele)
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
    # ordering node according to SFC
    ncurve = 1 # always use 1 sfc
    N = len(sfc)
    inverse_numbering = np.zeros((N, ncurve), dtype=int)
    inverse_numbering[:, 0] = np.argsort(sfc[:, 0])
    
    r = r1[:,inverse_numbering[:,0]].view(ndim,1,config.nele)
    r_s = []
    r_s.append(r)
    for i in range(1,nlevel):
        # pad one node with 0 as final node so that odd nodes won't be joined
        r = F.pad(r, (0,1), "constant", 0)
        with torch.no_grad():
            r = sfc_restrictor(r)
        r_s.append(r)

    ## on 1DOF level
    # note that from here on, e_i, e_ip1 have a singleton dimension 
    # between batch-shape (ndim) and node.number (nodes_per_level[level])
    # this singleton dimension is the "channel_in" for 1D filter (prolongator)
    e_i = r_s[-1].view(ndim,-1)/variables_sfc[-1][1].view(ndim,-1)
    CNN1D_prol_odd = nn.Upsample(scale_factor=nodes_per_level[-2]/nodes_per_level[-1])
    e_i = CNN1D_prol_odd(e_i.view(ndim,1,-1)) # prolongate
    ## mg sweep
    for level in reversed(range(1,nlevel-1)):
        for _ in range(config.mg_smooth_its):
            a_sfc_sparse, diagA, _ = variables_sfc[level]
            e_ip1 = torch.zeros_like(e_i, device=config.dev, dtype=torch.float64) # error after smooth (at i+1-th smooth step)
            for idim in range(ndim):
                for jdim in range(ndim):
                    e_ip1[idim,:] += torch.matmul(a_sfc_sparse[idim][jdim], e_i[jdim,:,:].view(-1))
            e_ip1 -= r_s[level].view(ndim,1,-1)
            e_ip1 *= -1.0 # now its b-Ax
            e_ip1 = e_ip1 / diagA.view(ndim,1,-1) * config.jac_wei + e_i.view(ndim,1,-1)
            CNN1D_prol_odd = nn.Upsample(scale_factor=nodes_per_level[level-1]/nodes_per_level[level])
            e_i = CNN1D_prol_odd(e_ip1.view(ndim,1,-1))
    ## map e_i to original order
    e_i = e_i[:,0,sfc[:,0]-1]

    return e_i

# get a and diag a from best_sfc_map
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

    diagonal = torch.zeros((config.ndim,nonods), dtype=torch.float64, device=config.dev)
    a_indices = []
    a_values = []
    for i in range(start_index, end_index):
        for j in range(fina_sfc_all_un[i]-1, fina_sfc_all_un[i+1]-1):
            a_indices.append([i-start_index, cola_sfc_all_un[j]-1-start_index])
            a_values.append(a_sfc[:,:,j])
    a_indices = np.asarray(a_indices).transpose()
    a_values = np.asarray(a_values) # now a_values has shape (nonods, ndim, ndim)
    a_values = torch.tensor(a_values, dtype=torch.float64, device=config.dev)
    # convert to sparse
    a_sfc_level_sparse = [[] for _ in range(ndim)]
        # see https://stackoverflow.com/questions/240178/list-of-lists-changes-reflected-across-sublists-unexpectedly 
        # about the reason why we cannot use list([[]*ndim]*ndim)
    for idim in range(ndim):
        for jdim in range(ndim):
            a_sfc_level_sparse[idim].append( torch.sparse_coo_tensor(
                a_indices,
                a_values[:,idim,jdim],
                (nonods, nonods),
                dtype=torch.float64,
                device=config.dev) )
    
    # find diagonal
    for i in range(nonods):
        i += start_index
        for j in range(fina_sfc_all_un[i]-1, fina_sfc_all_un[i+1]-1):
            if (i==cola_sfc_all_un[j]-1):
                for idim in range(ndim):
                    diagonal[idim,i-start_index] = a_sfc[idim,idim,j]
    
    return a_sfc_level_sparse, diagonal, nonods 

# mg on sfc - prep
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

    dummy = np.zeros((config.ndim,config.nele))

    starting_node = 1 # setting according to BY
    graph_trim = -10  # ''
    ncurve = 1        # '' 
    nele = config.nele # this is the shape of RAR # or should we use RAR.shape[0] for clarity? 
    ncola = cola.shape[0]

    whichd, sfc = \
        sf.ncurve_python_subdomain_space_filling_curve( \
        cola+1, fina+1, starting_node, graph_trim, ncurve, \
        ) # note that fortran array index start from 1, so cola and fina should +1.
    
    ## get coarse grid info
    max_nlevel = sf.calculate_nlevel_sfc(nele) + 1
    max_nonods_sfc_all_grids = 5*config.nele 
    max_ncola_sfc_all_un = 10*ncola
    a_sfc, fina_sfc_all_un, cola_sfc_all_un, ncola_sfc_all_un, b_sfc, \
        ml_sfc, fin_sfc_nonods, nonods_sfc_all_grids, nlevel = \
        map_sfc.vector_best_sfc_mapping_to_sfc_matrix_unstructured(\
            vec_a=RARvalues.cpu().numpy(),\
            vec_b=dummy, \
            ml=dummy[0,:],\
            fina=fina+1, \
            cola=cola+1, \
            sfc_node_ordering=sfc[:,0], \
            max_nonods_sfc_all_grids=max_nonods_sfc_all_grids, \
            max_ncola_sfc_all_un=max_ncola_sfc_all_un, \
            max_nlevel=max_nlevel,
            ndim=config.ndim, ncola=ncola,nonods=nele)
    nodes_per_level = [fin_sfc_nonods[i] - fin_sfc_nonods[i-1] for i in range(1, nlevel+1)]
    # print(fin_sfc_nonods.shape)
    a_sfc = a_sfc[:,:,:ncola_sfc_all_un]
    del b_sfc, ml_sfc 
    variables_sfc = []
    for level in range(nlevel):
        variables_sfc.append(get_a_diaga(
            level, 
            fin_sfc_nonods,
            fina_sfc_all_un,
            cola_sfc_all_un,
            a_sfc
        ))

    return sfc, variables_sfc, nlevel, nodes_per_level
