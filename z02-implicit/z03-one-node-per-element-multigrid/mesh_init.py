# mesh manipulation
import numpy as np
import torch
import config 

def init():
    # initiate mesh ...
    # output:
    # x_all, coordinates of all nodes, numpy array, (nonods, nloc)
    nele = config.nele 
    mesh = config.mesh 
    nonods = config.nonods 
    nloc = config.nloc
    

    # create faces
    faces=[]
    for ele in range(nele):
        element = mesh.cells[0][1][ele]
        for iloc in range(3):
            faces.append([element[iloc],element[(iloc+1)%3]])

    # neighbouring faces (global indices)
    # nbface = nbf(iface)
    # input: a global face index
    # output: the global index of the input face's neighbouring face
    #         sign denotes face node numbering orientation
    #         np.nan denotes no neighbouring found (indicating this is a boundary face)
    #         !! output type is real, convert to int before use as index !!
    nbf=np.empty(len(faces))
    nbf[:]=np.nan
    color=np.zeros(len(faces))
    for iface in range(len(faces)):
        if color[iface]==1 :
            continue 
        for jface in range(iface+1,len(faces)):
            if (color[jface]==1):
                continue 
            elif (set(faces[jface])==set(faces[iface])):
                # print(faces[jface],'|',faces[iface])
                if faces[jface][0]==faces[iface][0]:
                    nbf[iface]=jface 
                    nbf[jface]=iface
                else:
                    nbf[iface]=-jface 
                    nbf[jface]=-iface
                color[iface]=1
                color[jface]=1
                continue

    # find neighbouring elements associated with each face
    # via neighbouring faces
    # and store in nbele
    # nb_ele = nbele(iface)
    # input: a global face index
    # output type: float, sign denotes face node numbering orientation
    #              nan denotes non found (input is boundary element)
    #              !! convert to positive int before use as index !!
    nbele=np.empty(len(faces))
    nbele[:]=np.nan
    for iface in range(len(nbf)):
        nbele[iface] = np.sign(nbf[iface])*(np.abs(nbf[iface])//3)

    
    # generate cubic nodes from element vertices
    x_all = []
    for ele in range(nele):
        # vertex nodes global index
        idx = mesh.cells[0][1][ele]
        # vertex nodes coordinate 
        x_loc=[]
        for id in idx:
            x_loc.append(mesh.points[id])
            # print(x_loc)
        # ! a reference cubic element looks like this:
        # !  y
        # !  | 
        # !  2
        # !  | \
        # !  6  5
        # !  |   \
        # !  7 10 4
        # !  |     \
        # !  3-8-9--1--x
        # nodes 1-3
        x_all.append([x_loc[0][0], x_loc[0][1]])
        x_all.append([x_loc[1][0], x_loc[1][1]])
        x_all.append([x_loc[2][0], x_loc[2][1]])
        # nodes 4,5
        x_all.append([x_loc[0][0]*2./3.+x_loc[1][0]*1./3., x_loc[0][1]*2./3.+x_loc[1][1]*1./3.])
        x_all.append([x_loc[0][0]*1./3.+x_loc[1][0]*2./3., x_loc[0][1]*1./3.+x_loc[1][1]*2./3.])
        # nodes 6,7
        x_all.append([x_loc[1][0]*2./3.+x_loc[2][0]*1./3., x_loc[1][1]*2./3.+x_loc[2][1]*1./3.])
        x_all.append([x_loc[1][0]*1./3.+x_loc[2][0]*2./3., x_loc[1][1]*1./3.+x_loc[2][1]*2./3.])
        # nodes 8,9
        x_all.append([x_loc[2][0]*2./3.+x_loc[0][0]*1./3., x_loc[2][1]*2./3.+x_loc[0][1]*1./3.])
        x_all.append([x_loc[2][0]*1./3.+x_loc[0][0]*2./3., x_loc[2][1]*1./3.+x_loc[0][1]*2./3.])
        # node 10
        x_all.append([(x_loc[0][0]+x_loc[1][0]+x_loc[2][0])/3.,(x_loc[0][1]+x_loc[1][1]+x_loc[2][1])/3.])

    x_all = np.asarray(x_all, dtype=np.float64)
    # print('x_all shape: ', x_all.shape)

    # mark boundary nodes
    # bc1: y=0
    bc1=[]
    for inod in range(nonods):
        if x_all[inod,1]<1e-8 :
            bc1.append(inod)
    # bc2: x=0
    bc2=[]
    for inod in range(nonods):
        if x_all[inod,0]<1e-8 :
            bc2.append(inod)
    # bc3: y=1
    bc3=[]
    for inod in range(nonods):
        if x_all[inod,1]>1.-1e-8 :
            bc3.append(inod)
    # bc4: x=1
    bc4=[]
    for inod in range(nonods):
        if x_all[inod,0]>1.-1e-8 :
            bc4.append(inod)
    # print(bc1)
    # print(bc2)
    # print(bc3)
    # print(bc4)

    return x_all, nbf, nbele, bc1,bc2,bc3,bc4 

def connectivity(nbele):
    '''
    # generate element connectivity matrix
    # adjacency_matrix[nele, nele]
    # its sparsity: fina, cola, ncola is nnz
    '''

    nele = config.nele
    nloc = config.nloc
    adjacency_matrix = torch.eye(nele)
    for ele in range(nele):
        for iface in range(3):
            glb_iface = ele*3 + iface 
            ele2 = nbele[glb_iface]
            if np.isnan(ele2) :  # boundary face, no neighbour
                continue
            ele2 = int(abs(nbele[glb_iface]))
            if adjacency_matrix[ele, ele2] !=1 :
                adjacency_matrix[ele, ele2] = 1
    cola = adjacency_matrix.nonzero().t()[1] 
    fina = np.zeros(nele+1)
    fina_value = 0
    i = 0 
    for occurrence in torch.bincount(adjacency_matrix.nonzero().t()[0]):
        fina[i] = fina_value
        fina_value += occurrence.item()
        i += 1
    fina[-1] = fina_value
    fina = fina.astype(int)
    ncola=cola.shape[0]

    return adjacency_matrix, fina, cola, ncola


# local nodes number on a face
def face_iloc(iface):
    # return local nodes number on a face
    # !      y
    # !      | 
    # !      2
    # !  f2  | \   ┌
    # !   |  6  5   \   
    # !   |  |   \   \  f1
    # !  \|/ 7 10 4   \
    # !      |     \
    # !      3-8-9--1--x
    #         ---->
    #           f3
    match iface:
        case 0:
            return [0,3,4,1]
        case 1:
            return [1,5,6,2]
        case 2:
            return [2,7,8,0]
        case _:
            return []
def face_iloc2(iface):
    # return local nodes number on the other side of a face
    # in reverse order 
    iloc_list=face_iloc(iface)
    iloc_list.reverse()
    return iloc_list