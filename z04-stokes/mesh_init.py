# mesh manipulation
import numpy as np
import torch
import config , time
from config import sf_nd_nb
from get_nb import getfinele, getfin_p1cg


def init(mesh, nele, nonods, nloc):
    # initiate mesh ...
    # output:
    # x_all, coordinates of all nodes, numpy array, (nonods, nloc)

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
    sf_nd_nb.set_data(cg_ndglno=cg_ndglno)
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

    # ** new nbf definition **
    # neighbouring faces
    # nbface = nbf(iface)
    # input: a global face index
    # output: the global index of the input face's neighbouring face, if negative, then
    #   no neighbours found, indicating boundary nodes.
    # the alignment of gaussian points is determined by alnmt
    #      0 - same alignment (impossible if all element nodes are ordered anti-clockwisely)
    #      1 - reverse alignment
    #     -1 - no neighbour
    nbf = np.zeros(len(faces), dtype=np.int64) - 1
    alnmt = np.ones(len(faces), dtype=np.int64) * -1.
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
                        nbf[glb_iface] = glb_iface2
                        nbf[glb_iface2] = glb_iface
                        if faces[glb_iface][0]==faces[glb_iface2][0]:
                            alnmt[glb_iface] = 0
                            alnmt[glb_iface2] = 0
                        else:
                            alnmt[glb_iface] = 1
                            alnmt[glb_iface2] = 1
                        found[glb_iface]=True 
                        found[glb_iface2]=True 
    endtime = time.time()
    print('nbf: ', nbf)
    print('time consumed in finding neighbouring:', endtime-starttime,' s')

    # find neighbouring elements associated with each face
    # via neighbouring faces
    # and store in nbele
    # nb_ele = nbele(iface)
    # input: a global face index
    # output type: float, sign denotes face node numbering orientation
    #              nan denotes non found (input is boundary element)
    #              !! convert to positive int before use as index !!
    nbele = nbf // config.nface

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
    else:
        raise Exception('nloc %d is not accepted in mesh init' % nloc)
    cg_nonods = mesh.points.shape[0]
    np.savetxt('x_all_cg.txt', mesh.points, delimiter=',')
    sf_nd_nb.set_data(cg_nonods=cg_nonods)
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
    bc = [bc1, bc2, bc3, bc4]

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

    # store P3DG from/to P1DG restrictor/prolongator
    sf_nd_nb.set_data(I_31=torch.tensor([
        [1., 0, 0],
        [0, 1., 0],
        [0, 0, 1.],
        [2. / 3, 1. / 3, 0],
        [1. / 3, 2. / 3, 0],
        [0, 2. / 3, 1. / 3],
        [0, 1. / 3, 2. / 3],
        [1. / 3, 0, 2. / 3],
        [2. / 3, 0, 1. / 3],
        [1. / 3, 1. / 3, 1. / 3]
    ], device=config.dev, dtype=torch.float64))  # P1DG to P3DG, element-wise prolongation operator)
    if nloc == 3:  # linear element
        sf_nd_nb.set_data(I_31=torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], device=config.dev, dtype=torch.float64))

    return x_all, nbf, nbele, alnmt, finele, colele, ncolele, bc, cg_ndglno, cg_nonods, cg_bc


def init_3d(mesh, nele, nonods, nloc, nface):
    # initiate mesh ...
    # output:
    # x_all, coordinates of all nodes, numpy array, (nonods, nloc)

    # check and make sure tetrahedron are ordered left-handed
    for ele in range(nele):
        idx = mesh.cells[0][1][ele]
        x_loc = mesh.points[idx]
        det = np.linalg.det(x_loc[1:4, :] - x_loc[0, :])
        if det > 0:  # it's right-handed, flip two nodes to flip hand.
            mesh.cells[0][1][ele] = [idx[0], idx[2], idx[1], idx[3]]
        # x_loc = mesh.points[idx]
        # det = np.linalg.det(x_loc[1:4, :] - x_loc[0, :])
        # print(det > 0)
    # create faces
    faces = []
    for ele in range(nele):
        element = mesh.cells[0][1][ele]
        # for iface in range(nface):
        #     faces.append([element[iface],
        #                   element[(iface + 1) % nface],
        #                   element[(iface + 2) % nface]])
        # iface = 0, face nodes (2,1,3) -- this is triangle 3-2-4
        faces.append([element[2],
                      element[1],
                      element[3]])
        # iface = 1, face nodes (0,2,3) -- this is triangle 1-3-4
        faces.append([element[0],
                      element[2],
                      element[3]])
        # iface = 2, face nodes (1,0,3) -- this is triangle 2-1-4
        faces.append([element[1],
                      element[0],
                      element[3]])
        # iface = 3, face nodes (0,1,2) -- this is triangel 1-2-3
        faces.append([element[0],
                      element[1],
                      element[2]])
    # print(mesh.cells[0][1]+1)
    # cg_ndglno = np.zeros(nele * 4, dtype=np.int64)
    # for ele in range(nele):
    #     for iloc in range(4):
    #         cg_ndglno[ele * 4 + iloc] = mesh.cells[0][1][ele][iloc]
    cg_ndglno = mesh.cells[0][1].reshape((nele * 4))
    # sf_nd_nb.set_data(cg_ndglno=cg_ndglno)
    np.savetxt('cg_ndglno.txt', cg_ndglno, delimiter=',')
    starttime = time.time()
    # element connectivity matrix
    ncolele, finele, colele, _ = getfinele(
        nele, nloc=4, snloc=3, nonods=mesh.points.shape[0],
        ndglno=cg_ndglno + 1, mx_nface_p1=5, mxnele=6 * nele)
    finele = finele - 1
    colele = colele - 1
    # pytorch is very strict about the index order of cola
    # let's do a sorting
    for row in range(finele.shape[0] - 1):
        col_this = colele[finele[row]:finele[row + 1]]
        col_this = np.sort(col_this)
        colele[finele[row]:finele[row + 1]] = col_this
    colele = colele[:ncolele]  # cut off extras at the end of colele

    # neighbouring faces (global indices)
    # nbface = nbf(iface)
    # input: a global face index. Int
    # output: the global index of the input face's neighbouring face. Int64
    # alnmt(iface) : alignment of neighbour face. Int64
    #       -1 is none, denotes boundary face without neighbour
    #       0  is the same alignment
    #       1  rotate once (e.g. self is 1-2-3, nb is 2-3-1
    #       2  rotate twice (e.g. self is 1-2-3, nb is 3-1-2
    nbf = np.zeros(len(faces), dtype=np.int64) - 1
    alnmt = np.ones(len(faces), dtype=np.int64) * - 1
    found = np.zeros(len(faces), dtype=np.bool)
    for ele in range(nele):
        for iface in range(nface):
            glb_iface = ele * nface + iface
            if found[glb_iface]:
                continue
            for idx in range(finele[ele], finele[ele + 1]):
                if found[glb_iface]:
                    continue
                ele2 = colele[idx]
                if ele == ele2:
                    continue
                for iface2 in range(nface):
                    if found[glb_iface]:
                        continue
                    glb_iface2 = ele2 * nface + iface2
                    if set(faces[glb_iface]) == set(faces[glb_iface2]):
                        nbf[glb_iface] = glb_iface2
                        nbf[glb_iface2] = glb_iface
                        if faces[glb_iface][0] == faces[glb_iface2][0]:
                            alnmt[glb_iface] = 0
                            alnmt[glb_iface2] = 0
                        elif faces[glb_iface][1] == faces[glb_iface2][0]:
                            alnmt[glb_iface] = 1
                            alnmt[glb_iface2] = 1
                        else:
                            alnmt[glb_iface] = 2
                            alnmt[glb_iface2] = 2
                        found[glb_iface] = True
                        found[glb_iface2] = True
                        continue
    endtime = time.time()
    print('nbf: ', nbf)
    print('time consumed in finding neighbouring:', endtime - starttime, ' s')

    # find neighbouring elements associated with each face
    # via neighbouring faces
    # and store in nbele
    # nb_ele = nbele(iface)
    # input: a global face index
    # output type: int64
    nbele = nbf // nface

    if nloc == 20:
        # generate cubic nodes from element vertices
        x_all = []
        ref_node_order = [
            2, 3, 0, 1, 14,
            15, 11, 10, 4, 5,
            9, 8, 13, 12, 6,
            7, 16, 19, 17, 18,
        ]  # node order in vtk tetrahedron. will use this in outputing to vtk.
        # sf_nd_nb.set_data(ref_node_order=ref_node_order)
        for ele in range(nele):
            # vertex nodes global index
            idx = mesh.cells[0][1][ele]
            # vertex nodes coordinate
            x_loc = []
            for id in idx:
                x_loc.append(mesh.points[id])
            # print(x_loc)
            # ! a reference cubic element looks like this:
            # Tetrahedron:
            #                    z
            #                  .
            #                ,/
            #               /
            #            2
            #          ,/|`\
            #        14  |  `9
            #      ,/    '.   `\
            #    15       5     `8
            #  ,/         |       `\
            # 3-----13----'.--12----1 --> y
            #  `\.         |      ,/
            #     11.      4    ,7
            #        `10   '. ,6
            #           `\. |/
            #              `0
            #                 `\.
            #                    ` x
            # face central nodes:
            # 16 - on face 2-1-3
            # 17 - on face 0-1-2
            # 18 - on face 0-2-3
            # 19 - on face 1-0-3
            # corner  0, 1, 2, 3
            x_all.append([x_loc[0][0], x_loc[0][1], x_loc[0][2]])
            x_all.append([x_loc[1][0], x_loc[1][1], x_loc[1][2]])
            x_all.append([x_loc[2][0], x_loc[2][1], x_loc[2][2]])
            x_all.append([x_loc[3][0], x_loc[3][1], x_loc[3][2]])
            # edge nodes 4, 5 (on edge 0-2)
            x_all.append([x_loc[0][0] * 2. / 3. + x_loc[2][0] * 1. / 3.,
                          x_loc[0][1] * 2. / 3. + x_loc[2][1] * 1. / 3.,
                          x_loc[0][2] * 2. / 3. + x_loc[2][2] * 1. / 3.])
            x_all.append([x_loc[0][0] * 1. / 3. + x_loc[2][0] * 2. / 3.,
                          x_loc[0][1] * 1. / 3. + x_loc[2][1] * 2. / 3.,
                          x_loc[0][2] * 1. / 3. + x_loc[2][2] * 2. / 3.])
            # edge nodes 6, 7 (on edge 0-1)
            x_all.append([x_loc[0][0] * 2. / 3. + x_loc[1][0] * 1. / 3.,
                          x_loc[0][1] * 2. / 3. + x_loc[1][1] * 1. / 3.,
                          x_loc[0][2] * 2. / 3. + x_loc[1][2] * 1. / 3.])
            x_all.append([x_loc[0][0] * 1. / 3. + x_loc[1][0] * 2. / 3.,
                          x_loc[0][1] * 1. / 3. + x_loc[1][1] * 2. / 3.,
                          x_loc[0][2] * 1. / 3. + x_loc[1][2] * 2. / 3.])
            # edge nodes 8, 9 (on edge 1-2)
            x_all.append([x_loc[1][0] * 2. / 3. + x_loc[2][0] * 1. / 3.,
                          x_loc[1][1] * 2. / 3. + x_loc[2][1] * 1. / 3.,
                          x_loc[1][2] * 2. / 3. + x_loc[2][2] * 1. / 3.])
            x_all.append([x_loc[1][0] * 1. / 3. + x_loc[2][0] * 2. / 3.,
                          x_loc[1][1] * 1. / 3. + x_loc[2][1] * 2. / 3.,
                          x_loc[1][2] * 1. / 3. + x_loc[2][2] * 2. / 3.])
            # edge nodes 10, 11 (on edge 0-3)
            x_all.append([x_loc[0][0] * 2. / 3. + x_loc[3][0] * 1. / 3.,
                          x_loc[0][1] * 2. / 3. + x_loc[3][1] * 1. / 3.,
                          x_loc[0][2] * 2. / 3. + x_loc[3][2] * 1. / 3.])
            x_all.append([x_loc[0][0] * 1. / 3. + x_loc[3][0] * 2. / 3.,
                          x_loc[0][1] * 1. / 3. + x_loc[3][1] * 2. / 3.,
                          x_loc[0][2] * 1. / 3. + x_loc[3][2] * 2. / 3.])
            # edge nodes 12, 13 (on edge 1-3)
            x_all.append([x_loc[1][0] * 2. / 3. + x_loc[3][0] * 1. / 3.,
                          x_loc[1][1] * 2. / 3. + x_loc[3][1] * 1. / 3.,
                          x_loc[1][2] * 2. / 3. + x_loc[3][2] * 1. / 3.])
            x_all.append([x_loc[1][0] * 1. / 3. + x_loc[3][0] * 2. / 3.,
                          x_loc[1][1] * 1. / 3. + x_loc[3][1] * 2. / 3.,
                          x_loc[1][2] * 1. / 3. + x_loc[3][2] * 2. / 3.])
            # edge nodes 14, 15 (on edge 2-3)
            x_all.append([x_loc[2][0] * 2. / 3. + x_loc[3][0] * 1. / 3.,
                          x_loc[2][1] * 2. / 3. + x_loc[3][1] * 1. / 3.,
                          x_loc[2][2] * 2. / 3. + x_loc[3][2] * 1. / 3.])
            x_all.append([x_loc[2][0] * 1. / 3. + x_loc[3][0] * 2. / 3.,
                          x_loc[2][1] * 1. / 3. + x_loc[3][1] * 2. / 3.,
                          x_loc[2][2] * 1. / 3. + x_loc[3][2] * 2. / 3.])
            # face node 16 - on face 213
            x_all.append(
                [(x_loc[2][0] + x_loc[1][0] + x_loc[3][0]) / 3.,
                 (x_loc[2][1] + x_loc[1][1] + x_loc[3][1]) / 3.,
                 (x_loc[2][2] + x_loc[1][2] + x_loc[3][2]) / 3.])
            # face node 17 - on face 012
            x_all.append(
                [(x_loc[0][0] + x_loc[1][0] + x_loc[2][0]) / 3.,
                 (x_loc[0][1] + x_loc[1][1] + x_loc[2][1]) / 3.,
                 (x_loc[0][2] + x_loc[1][2] + x_loc[2][2]) / 3.])
            # face node 18 - on face 023
            x_all.append(
                [(x_loc[0][0] + x_loc[2][0] + x_loc[3][0]) / 3.,
                 (x_loc[0][1] + x_loc[2][1] + x_loc[3][1]) / 3.,
                 (x_loc[0][2] + x_loc[2][2] + x_loc[3][2]) / 3.])
            # face node 19 - on face 103
            x_all.append(
                [(x_loc[1][0] + x_loc[0][0] + x_loc[3][0]) / 3.,
                 (x_loc[1][1] + x_loc[0][1] + x_loc[3][1]) / 3.,
                 (x_loc[1][2] + x_loc[0][2] + x_loc[3][2]) / 3.])
    elif nloc == 10:  # quadratic element
        x_all = []
        ref_node_order = [
            2, 3, 0, 1, 9,
            7, 5, 4, 8, 6,
        ]  # node order in vtk tetrahedron. will use this in outputing to vtk.
        # sf_nd_nb.set_data(ref_node_order=ref_node_order)
        for ele in range(nele):
            # vertex nodes global index
            idx = mesh.cells[0][1][ele]
            # vertex nodes coordinate
            x_loc = []
            for id in idx:
                x_loc.append(mesh.points[id])
            # print(x_loc)
            # ! a reference quadratic element looks like this:
            # Tetrahedron:
            #                    z
            #                  .
            #                ,/
            #               /
            #            2
            #          ,/|`\
            #        ,/  |  `\
            #      ,9    '.   `4
            #    ,/       5     `\
            #  ,/         |       `\
            # 3--------8--'.--------1---y
            #  `\.         |      ,/
            #     `\.      |    ,6
            #        `7.   '. ,/
            #           `\. |/
            #              `0
            #                 `\.
            #                    ` x
            # corner  0, 1, 2, 3
            x_all.append([x_loc[0][0], x_loc[0][1], x_loc[0][2]])
            x_all.append([x_loc[1][0], x_loc[1][1], x_loc[1][2]])
            x_all.append([x_loc[2][0], x_loc[2][1], x_loc[2][2]])
            x_all.append([x_loc[3][0], x_loc[3][1], x_loc[3][2]])
            # edge node 4 on edge 1-2
            x_all.append([x_loc[1][0] * 1. / 2. + x_loc[2][0] * 1. / 2.,
                          x_loc[1][1] * 1. / 2. + x_loc[2][1] * 1. / 2.,
                          x_loc[1][2] * 1. / 2. + x_loc[2][2] * 1. / 2.])
            # edge node 5 on edge 0-2
            x_all.append([x_loc[0][0] * 1. / 2. + x_loc[2][0] * 1. / 2.,
                          x_loc[0][1] * 1. / 2. + x_loc[2][1] * 1. / 2.,
                          x_loc[0][2] * 1. / 2. + x_loc[2][2] * 1. / 2.])
            # edge node 6 on edge 0-1
            x_all.append([x_loc[0][0] * 1. / 2. + x_loc[1][0] * 1. / 2.,
                          x_loc[0][1] * 1. / 2. + x_loc[1][1] * 1. / 2.,
                          x_loc[0][2] * 1. / 2. + x_loc[1][2] * 1. / 2.])
            # edge node 7 on edge 0-3
            x_all.append([x_loc[0][0] * 1. / 2. + x_loc[3][0] * 1. / 2.,
                          x_loc[0][1] * 1. / 2. + x_loc[3][1] * 1. / 2.,
                          x_loc[0][2] * 1. / 2. + x_loc[3][2] * 1. / 2.])
            # edge node 8 on edge 1-3
            x_all.append([x_loc[1][0] * 1. / 2. + x_loc[3][0] * 1. / 2.,
                          x_loc[1][1] * 1. / 2. + x_loc[3][1] * 1. / 2.,
                          x_loc[1][2] * 1. / 2. + x_loc[3][2] * 1. / 2.])
            # edge node 9 on edge 2-3
            x_all.append([x_loc[2][0] * 1. / 2. + x_loc[3][0] * 1. / 2.,
                          x_loc[2][1] * 1. / 2. + x_loc[3][1] * 1. / 2.,
                          x_loc[2][2] * 1. / 2. + x_loc[3][2] * 1. / 2.])
    elif nloc == 4:  # linear element
        x_all = []
        ref_node_order = [
            2, 3, 0, 1,
        ]  # node order in vtk tetrahedron. will use this in outputing to vtk.
        # sf_nd_nb.set_data(ref_node_order=ref_node_order)
        for ele in range(nele):
            # vertex nodes global index
            idx = mesh.cells[0][1][ele]
            # vertex nodes coordinate
            x_loc = []
            for id in idx:
                x_loc.append(mesh.points[id])
                # corner  0, 1, 2, 3
            x_all.append([x_loc[0][0], x_loc[0][1], x_loc[0][2]])
            x_all.append([x_loc[1][0], x_loc[1][1], x_loc[1][2]])
            x_all.append([x_loc[2][0], x_loc[2][1], x_loc[2][2]])
            x_all.append([x_loc[3][0], x_loc[3][1], x_loc[3][2]])
    else:
        raise Exception('nloc %d is not accepted in mesh init' % nloc)
    cg_nonods = mesh.points.shape[0]
    np.savetxt('x_all_cg.txt', mesh.points, delimiter=',')
    # sf_nd_nb.set_data(cg_nonods=cg_nonods)
    x_all = np.asarray(x_all, dtype=np.float64)
    # print('x_all shape: ', x_all.shape)

    # mark boundary nodes of a cube
    bc1 = []  # x=0
    for inod in range(nonods):
        if x_all[inod, 0] < 1e-8:
            bc1.append(inod)
    bc2 = []  # y=0
    for inod in range(nonods):
        if x_all[inod, 1] < 1e-8:
            bc2.append(inod)
    bc3 = []  # z=0
    for inod in range(nonods):
        if x_all[inod, 2] < 1e-8:
            bc3.append(inod)
    bc4 = []  # x=1
    for inod in range(nonods):
        if x_all[inod, 0] > 1. - 1e-8:
            bc4.append(inod)
    bc5 = []  # y=0
    for inod in range(nonods):
        if x_all[inod, 1] > 1. - 1e-8:
            bc5.append(inod)
    bc6 = []  # z=0
    for inod in range(nonods):
        if x_all[inod, 2] > 1. - 1e-8:
            bc6.append(inod)
    # mark boundary nodes
    bc = [bc1, bc2, bc3, bc4, bc5, bc6]

    # store P3DG from/to P1DG restrictor/prolongator
    sf_nd_nb.set_data(vel_I_prol=torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [2 / 3, 0, 1 / 3, 0],
        [1 / 3, 0, 2 / 3, 0],
        [2 / 3, 1 / 3, 0, 0],
        [1 / 3, 2 / 3, 0, 0],
        [0, 2 / 3, 1 / 3, 0],
        [0, 1 / 3, 2 / 3, 0],
        [2 / 3, 0, 0, 1 / 3],
        [1 / 3, 0, 0, 2 / 3],
        [0, 2 / 3, 0, 1 / 3],
        [0, 1 / 3, 0, 2 / 3],
        [0, 0, 2 / 3, 1 / 3],
        [0, 0, 1 / 3, 2 / 3],
        [0, 1 / 3, 1 / 3, 1 / 3],
        [1 / 3, 1 / 3, 1 / 3, 0],
        [1 / 3, 0, 1 / 3, 1 / 3],
        [1 / 3, 1 / 3, 0, 1 / 3]
    ], device=config.dev, dtype=torch.float64))  # P1DG to P3DG, element-wise prolongation operator
    sf_nd_nb.set_data(pre_I_prol=torch.tensor([
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
    ], device=config.dev, dtype=torch.float64))  # P2DG to P1DG, element-wise prolongation operator
    if nloc == 4:  # linear element
        sf_nd_nb.set_data(vel_I_prol=torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], device=config.dev, dtype=torch.float64))  # P1DG to P1DG, element-wise prolongation operator
    elif nloc == 10:  # quadratic element
        sf_nd_nb.set_data(vel_I_prol=torch.tensor([
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
        ], device=config.dev, dtype=torch.float64))  # P2DG to P1DG, element-wise prolongation operator
        sf_nd_nb.set_data(pre_I_prol=torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], device=config.dev, dtype=torch.float64))  # P1DG to P1DG, element-wise prolongation operator
        # sf_nd_nb.I_13 = torch.tensor([
        #     [1 / 5, -2 / 15, -2 / 15, -2 / 15, -2 / 15, 8 / 15, 8 / 15, 8 / 15, -2 / 15, -2 / 15],
        #     [-2 / 15, 1 / 5, -2 / 15, -2 / 15, 8 / 15, -2 / 15, 8 / 15, -2 / 15, 8 / 15, -2 / 15],
        #     [-2 / 15, -2 / 15, 1 / 5, -2 / 15, 8 / 15, 8 / 15, -2 / 15, -2 / 15, -2 / 15, 8 / 15],
        #     [-2 / 15, -2 / 15, -2 / 15, 1 / 5, -2 / 15, -2 / 15, -2 / 15, 8 / 15, 8 / 15, 8 / 15],
        # ], device=config.dev, dtype=torch.float64)
        # sf_nd_nb.set_data(I_31=torch.tensor([
        #     [1, 0, 0, 0],
        #     [0, 1, 0, 0],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1],
        #     [0, 0, 0, 0],
        #     [0, 0, 0, 0],
        #     [0, 0, 0, 0],
        #     [0, 0, 0, 0],
        #     [0, 0, 0, 0],
        #     [0, 0, 0, 0],
        # ], device=config.dev, dtype=torch.float64))  # P2DG to P1DG, element-wise prolongation operator
    return x_all, nbf, nbele, alnmt, finele, colele, ncolele, bc, cg_ndglno, cg_nonods, ref_node_order


def connectivity(nbele):
    '''
    !! extremely memory hungry
    !! TODO: to be deprecated
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


def p1cg_sparsity(vel_func_space):
    '''
    get P1CG sparsity from node-global-number list
    '''
    import time
    cg_ndglbno = vel_func_space.cg_ndglno
    nele = vel_func_space.nele
    start_time = time.time()
    print('im in get p1cg sparsity, time:', time.time()-start_time)
    # nele = config.nele
    cg_nonods = vel_func_space.cg_nonods
    p1dg_nonods = vel_func_space.p1dg_nonods
    nloc = vel_func_space.p1cg_nloc
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
    spmat = sp.coo_matrix((val,(idx[0,:],idx[1,:])), shape=(cg_nonods, cg_nonods))
    spmat = spmat.tocsr()
    print('ive finished, time:', time.time()-start_time)
    return spmat.indptr, spmat.indices, spmat.nnz
