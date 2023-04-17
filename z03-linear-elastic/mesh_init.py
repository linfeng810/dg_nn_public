# mesh manipulation
import numpy as np
import torch
import config, time
from config import sf_nd_nb
from get_nb import getfinele, getfin_p1cg

def init():
    # initiate mesh ...
    # output:
    # x_all, coordinates of all nodes, numpy array, (nonods, nloc)
    nele = config.nele 
    mesh = config.mesh 
    nonods = config.nonods 
    nloc = config.nloc
    

    # check and make sure triangle vertices are ordered anti-clockwisely
    for ele in range(nele):
        # vertex nodes global index
        idx = mesh.cells[0][1][ele]
        # vertex nodes coordinate 
        x_loc=[]
        for id in idx:
            x_loc.append(mesh.points[id])
        x_loc = np.asarray(x_loc)
        x_loc[:,-1]=1.
        det = np.linalg.det(x_loc)
        if (det<0) :
            # print(mesh.cells[0][1][ele])
            mesh.cells[0][1][ele] = [idx[0], idx[2], idx[1]]
            # print(mesh.cells[0][1][ele])
            # print('clockise')
        

    # create faces
    faces=[]
    for ele in range(nele):
        element = mesh.cells[0][1][ele]
        for iloc in range(3):
            faces.append([element[iloc],element[(iloc+1)%3]])

    cg_ndglno = np.zeros(nele*3, dtype=np.int64)
    for ele in range(nele):
        for iloc in range(3):
            cg_ndglno[ele*3+iloc] = mesh.cells[0][1][ele][iloc]
    config.sf_nd_nb.set_data(cg_ndglno = cg_ndglno)
    # np.savetxt('cg_ndglno.txt', cg_ndglno, delimiter=',')
    starttime = time.time()
    # element connectivity matrix
    ncolele,finele,colele,_ = getfinele(
        nele,nloc=3,snloc=2,nonods=mesh.points.shape[0],
        ndglno=cg_ndglno+1,mx_nface_p1=4,mxnele=5*nele)
    finele=finele-1 
    colele=colele-1
    # pytorch is very strict about the index order of cola
    # let's do a sorting
    for row in range(finele.shape[0]-1):
        col_this = colele[finele[row]:finele[row+1]]
        col_this = np.sort(col_this)
        colele[finele[row]:finele[row+1]] = col_this 
    colele = colele[:ncolele] # cut off extras at the end of colele

    # neighbouring faces (global indices)
    # nbface = nbf(iface)
    # input: a global face index
    # output: the global index of the input face's neighbouring face
    #         sign denotes face node numbering orientation
    #         np.nan denotes no neighbouring found (indicating this is a boundary face)
    #         !! output type is real, convert to int before use as index !!
    nbf=np.empty(len(faces))
    nbf[:]=np.nan
    found=np.empty(len(faces))
    found[:]=False
    for ele in range(config.nele):
        for iface in range(3):
            glb_iface = ele*3+iface 
            if(found[glb_iface]):
                continue
            for idx in range(finele[ele],finele[ele+1]):
                ele2 = colele[idx]
                if (ele==ele2):
                    continue
                for iface2 in range(3):
                    glb_iface2 = ele2*3+iface2 
                    if (set(faces[glb_iface])==set(faces[glb_iface2])):
                        if faces[glb_iface][0]==faces[glb_iface2][0]:
                            nbf[glb_iface]=glb_iface2
                            nbf[glb_iface2]=glb_iface
                        else:
                            nbf[glb_iface]=-glb_iface2
                            nbf[glb_iface2]=-glb_iface
                        found[glb_iface]=True 
                        found[glb_iface2]=True 
    endtime = time.time()
    # print('nbf: ', nbf)
    print('time consumed in finding neighbouring:', endtime-starttime,' s')
    #### old method very slow to be deleted in the future ####
    # # neighbouring faces (global indices)
    # # nbface = nbf(iface)
    # # input: a global face index
    # # output: the global index of the input face's neighbouring face
    # #         sign denotes face node numbering orientation
    # #         np.nan denotes no neighbouring found (indicating this is a boundary face)
    # #         !! output type is real, convert to int before use as index !!
    # starttime = time.time()
    # nbf=np.empty(len(faces))
    # nbf[:]=np.nan
    # color=np.zeros(len(faces))
    # for iface in range(len(faces)):
    #     if color[iface]==1 :
    #         continue 
    #     for jface in range(iface+1,len(faces)):
    #         if (color[jface]==1):
    #             continue 
    #         elif (set(faces[jface])==set(faces[iface])):
    #             # print(faces[jface],'|',faces[iface])
    #             if faces[jface][0]==faces[iface][0]:
    #                 nbf[iface]=jface 
    #                 nbf[jface]=iface
    #             else:
    #                 nbf[iface]=-jface 
    #                 nbf[jface]=-iface
    #             color[iface]=1
    #             color[jface]=1
    #             continue
    # endtime = time.time()
    # print('time consumed in finding neighbouring:', endtime-starttime,' s')
    # print('nbf:',nbf)
    #### end of old method ####

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

    if nloc == 10:
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
    elif nloc == 3:  # linear element
        x_all = []
        for ele in range(nele):
            # vertex nodes global index
            idx = mesh.cells[0][1][ele]
            # vertex nodes coordinate
            x_loc=[]
            for id in idx:
                x_loc.append(mesh.points[id])
            # print(x_loc)
            # ! a reference linear element looks like this:
            # !  y
            # !  |
            # !  2
            # !  | \
            # !  |  \
            # !  |   \
            # !  |    \
            # !  |     \
            # !  3------1--x
            # nodes 1-3
            x_all.append([x_loc[0][0], x_loc[0][1]])
            x_all.append([x_loc[1][0], x_loc[1][1]])
            x_all.append([x_loc[2][0], x_loc[2][1]])
    cg_nonods = mesh.points.shape[0]
    np.savetxt('x_all_cg.txt', mesh.points, delimiter=',')
    config.sf_nd_nb.set_data(cg_nonods = cg_nonods)
    x_all = np.asarray(x_all, dtype=np.float64)
    # print('x_all shape: ', x_all.shape)

    # mark boundary nodes
    #      bc4
    #   ┌--------┐
    #   |        |
    #bc1|        |bc3
    #   |        |
    #   └--------┘
    #      bc2
    # bc1: x=0
    bc1=[]
    for inod in range(nonods):
        if x_all[inod,0]<1e-8 :
            bc1.append(inod)
    # bc2: y=0
    bc2=[]
    for inod in range(nonods):
        if x_all[inod,1]<1e-8 :
            bc2.append(inod)
    # bc3: x=1
    bc3=[]
    for inod in range(nonods):
        if x_all[inod,0]>1.-1e-8 :
            bc3.append(inod)
    # bc4: y=1
    bc4=[]
    for inod in range(nonods):
        if x_all[inod,1]>1.-1e-8 :
            bc4.append(inod)

    # cg bc
    cg_bc1 = []
    for inod in range(cg_nonods):
        if mesh.points[inod, 0] < 1e-8:
            cg_bc1.append(inod)
    # bc2: y=0
    cg_bc2 = []
    for inod in range(cg_nonods):
        if mesh.points[inod, 1] < 1e-8:
            cg_bc2.append(inod)
    # bc3: x=1
    cg_bc3 = []
    for inod in range(cg_nonods):
        if mesh.points[inod, 0] > 1. - 1e-8:
            cg_bc3.append(inod)
    # bc4: y=1
    cg_bc4 = []
    for inod in range(cg_nonods):
        if mesh.points[inod, 1] > 1. - 1e-8:
            cg_bc4.append(inod)
    cg_bc = [cg_bc1, cg_bc2, cg_bc3, cg_bc4]
    return x_all, nbf, nbele, finele, colele, ncolele, bc1,bc2,bc3,bc4 , cg_ndglno, cg_nonods, cg_bc

def connectivity(nbele):
    '''
    !! extremely memory hungry
    !! to be deprecated
    generate element connectivity matrix
    adjacency_matrix[nele, nele] 
    its sparsity: fina, cola, ncola is nnz
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

    return fina, cola, ncola


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

def sgi2(sgi):
    # return gaussian pnts index on the other side
    if config.sngi == 4:  # cubic element
        order_on_other_side = [3,2,1,0]
    elif config.sngi == 2:  # linear element
        order_on_other_side = [1,0]
    else:
        raise Exception(f"config.sngi is not accepted in sgi2 (find gaussian "
                        f"points on the other side)")
    return order_on_other_side[sgi]


def p1cg_sparsity(cg_ndglbno):
    '''
    get P1CG sparsity from node-global-number list
    '''
    import time
    start_time = time.time()
    print('im in get p1cg sparsity, time:', time.time()-start_time)
    nele = config.nele
    cg_nonods = sf_nd_nb.cg_nonods
    p1dg_nonods = config.p1dg_nonods
    nloc = 3
    idx = []
    val = []
    import scipy.sparse as sp
    # for ele in range(nele):
    #     for inod in range(nloc):
    #         glbi = cg_ndglbno[ele*nloc+inod]
    #         for jnod in range(nloc):
    #             glbj = cg_ndglbno[ele*nloc+jnod]
    #             idx.append([glbi,glbj])
    #             val.append(0)
    idx, n_idx = getfin_p1cg(cg_ndglbno+1, nele, nloc, p1dg_nonods)
    print('im back from fortran, time:', time.time()-start_time)
    idx = np.asarray(idx)-1
    val = np.ones(n_idx)
    spmat = sp.coo_matrix((val,(idx[0,:],idx[1,:])),
                          shape=(cg_nonods, cg_nonods))
    spmat = spmat.tocsr()
    print('ive finished, time:', time.time()-start_time)
    return spmat.indptr, spmat.indices, spmat.nnz
