# mesh manipulation
import numpy as np
import torch
import config , time
from config import sf_nd_nb
from get_nb import getfinele, getfin_p1cg


def init_2d(mesh, nele, nonods, nloc, nface):
    # initiate mesh ...
    # output:
    # x_all, coordinates of all nodes, numpy array, (nonods, nloc)
    cg_ndglno = np.zeros((nele, 3), dtype=np.int64)
    cg_ndglno += mesh.cells_dict['triangle']

    # check and make sure triangle vertices are ordered anti-clockwisely
    for ele in range(nele):
        # vertex nodes global index
        idx = cg_ndglno[ele]  # cells[0][1][ele]
        # vertex nodes coordinate 
        x_loc=[]
        for id in idx:
            x_loc.append(mesh.points[id])
        x_loc = np.asarray(x_loc)
        x_loc[:,-1]=1.
        det = np.linalg.det(x_loc)
        if det<0:
            # print(mesh.cells[0][1][ele])
            cg_ndglno[ele] = [idx[0], idx[2], idx[1]]
            # print(mesh.cells[0][1][ele])
            # print('clockise')

    # create faces
    faces=[]
    for ele in range(nele):
        element = cg_ndglno[ele]
        for iloc in range(3):
            faces.append([element[iloc],element[(iloc+1)%3]])

    # sf_nd_nb.set_data(cg_ndglno=cg_ndglno)
    # np.savetxt('cg_ndglno.txt', cg_ndglno, delimiter=',')
    starttime = time.time()
    # element connectivity matrix
    ncolele,finele,colele,_ = getfinele(
        nele,nloc=3,snloc=2,nonods=mesh.points.shape[0],
        ndglno=cg_ndglno.reshape((nele*3))+1,mx_nface_p1=4,mxnele=5*nele)
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
    alnmt = np.ones(len(faces), dtype=np.int64) * -1
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
    # print('nbf: ', nbf)
    print('time consumed in finding neighbouring:', endtime-starttime,' s')

    # find neighbouring elements associated with each face
    # via neighbouring faces
    # and store in nbele
    # nb_ele = nbele(iface)
    # input: a global face index
    # output type: float, sign denotes face node numbering orientation
    #              nan denotes non found (input is boundary element)
    #              !! convert to positive int before use as index !!
    nbele = nbf // nface

    if nloc == 10:
        # generate cubic nodes from element vertices
        x_all = []
        for ele in range(nele):
            # vertex nodes global index
            idx = cg_ndglno[ele]
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
        ref_node_order = [1, 2, 0, 5, 6, 7, 8, 3, 4, 9]
    elif nloc == 6:  # quadratic element
        x_all = []
        for ele in range(nele):
            # vertex nodes global index
            idx = cg_ndglno[ele]
            # vertex nodes coordinate
            x_loc = []
            for id in idx:
                x_loc.append(mesh.points[id])
            # reference quadratic triangle
            # !  y
            # !  |
            # !  2
            # !  | \
            # !  |  \
            # !  5   4
            # !  |    \
            # !  |     \
            # !  3--6---1--x
            # nodes 1-3
            x_all.append([x_loc[0][0], x_loc[0][1]])
            x_all.append([x_loc[1][0], x_loc[1][1]])
            x_all.append([x_loc[2][0], x_loc[2][1]])
            # node 4-6
            x_all.append([x_loc[0][0] * 0.5 + x_loc[1][0] * 0.5, x_loc[0][1] * 0.5 + x_loc[1][1] * 0.5])
            x_all.append([x_loc[1][0] * 0.5 + x_loc[2][0] * 0.5, x_loc[1][1] * 0.5 + x_loc[2][1] * 0.5])
            x_all.append([x_loc[2][0] * 0.5 + x_loc[0][0] * 0.5, x_loc[2][1] * 0.5 + x_loc[0][1] * 0.5])
        ref_node_order = [1, 2, 0, 4, 5, 3]
    elif nloc == 3:  # linear element
        x_all = []
        for ele in range(nele):
            # vertex nodes global index
            idx = cg_ndglno[ele]
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
        ref_node_order = [1, 2, 0]
    else:
        raise Exception('nloc %d is not accepted in mesh init' % nloc)
    cg_nonods = mesh.points.shape[0]
    np.savetxt('x_all_cg.txt', mesh.points, delimiter=',')
    # sf_nd_nb.set_data(cg_nonods=cg_nonods)
    x_all = np.asarray(x_all, dtype=np.float64)
    # print('x_all shape: ', x_all.shape)

    # find boundary nodes and mark boundary faces
    if not config.isFSI:
        bc, glb_bcface_type = get_bc_node(mesh, faces, alnmt, nloc, x_all)
    else:
        bc, glb_bcface_type = get_bc_node_fsi(mesh, faces, alnmt, nbf, nloc, x_all)

    # store P3DG from/to P1DG restrictor/prolongator
    prolongator_from_p1dg = torch.tensor([
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
    ], device=config.dev, dtype=torch.float64)  # P1DG to P3DG, element-wise prolongation operator)
    if nloc == 3:  # linear element
        prolongator_from_p1dg = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], device=config.dev, dtype=torch.float64)
    elif nloc == 6:  # quadratic element
        prolongator_from_p1dg = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1 / 2, 1 / 2, 0],
            [0, 1 / 2, 1 / 2],
            [1 / 2, 0, 1 / 2],
        ], device=config.dev, dtype=torch.float64)

    cg_ndglno = cg_ndglno.reshape((nele * 3))

    return x_all, nbf, nbele, alnmt, glb_bcface_type, \
        finele, colele, ncolele, \
        bc, cg_ndglno, cg_nonods, ref_node_order, \
        prolongator_from_p1dg


def init_3d(mesh, nele, nonods, nloc, nface):
    # initiate mesh ...
    # output:
    # x_all, coordinates of all nodes, numpy array, (nonods, nloc)
    cg_ndglno = np.zeros((nele, 4), dtype=np.int64)
    cg_ndglno += mesh.cells_dict['tetra']

    # check and make sure tetrahedron are ordered left-handed
    for ele in range(nele):
        idx = cg_ndglno[ele]
        x_loc = mesh.points[idx]
        det = np.linalg.det(x_loc[1:4, :] - x_loc[0, :])
        if det > 0:  # it's right-handed, flip two nodes to flip hand.
            # TODO: change back!!!!
            # mesh.cells[-1].data[ele] = [idx[0], idx[2], idx[1], idx[3]]
            # temporarily change order
            cg_ndglno[ele, :] = np.array([idx[3], idx[1], idx[2], idx[0]])
        # else:
        #     raise Warning('ele %d is not right-handed. I didnt flip two nodes. \n '
        #                   'Boundary nodes might not been correctly found.\n' % ele)
        # idx = cg_ndglno[ele]
        # x_loc = mesh.points[idx]
        # det = np.linalg.det(x_loc[1:4, :] - x_loc[0, :])
        # print(det > 0)
    # create faces
    faces = []
    for ele in range(nele):
        element = cg_ndglno[ele]
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

    # sf_nd_nb.set_data(cg_ndglno=cg_ndglno)
    np.savetxt('cg_ndglno.txt', cg_ndglno, delimiter=',')
    starttime = time.time()
    # element connectivity matrix
    ncolele, finele, colele, _ = getfinele(
        nele, nloc=4, snloc=3, nonods=mesh.points.shape[0],
        ndglno=cg_ndglno.reshape((nele*4)) + 1, mx_nface_p1=5, mxnele=6 * nele)
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
    found = np.zeros(len(faces), dtype=bool)
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
    # print('nbf: ', nbf)
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
            idx = cg_ndglno[ele]
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
            idx = cg_ndglno[ele]
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
            idx = cg_ndglno[ele]
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

    # mark boundary nodes and boundary face types
    if not config.isFSI:
        bc, glb_bcface_type = get_bc_node(mesh, faces, alnmt, nloc, x_all)
    else:
        bc, glb_bcface_type = get_bc_node_fsi(mesh, faces, alnmt, nbf, nloc, x_all)

    # store P3DG from/to P1DG restrictor/prolongator
    prolongator_from_p1dg = torch.tensor([
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
    ], device=config.dev, dtype=torch.float64)  # P1DG to P3DG, element-wise prolongation operator
    if nloc == 4:  # linear element
        prolongator_from_p1dg = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], device=config.dev, dtype=torch.float64)  # P1DG to P1DG, element-wise prolongation operator
    elif nloc == 10:  # quadratic element
        prolongator_from_p1dg = torch.tensor([
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
        ], device=config.dev, dtype=torch.float64)  # P2DG to P1DG, element-wise prolongation operator

    cg_ndglno = cg_ndglno.reshape((nele * 4))

    return x_all, nbf, nbele, alnmt, glb_bcface_type, \
        finele, colele, ncolele, \
        bc, cg_ndglno, cg_nonods, ref_node_order,\
        prolongator_from_p1dg


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


def get_bc_node(mesh, faces, alnmt, nloc, x_all=0):
    """this is to find boundary nodes and
    log them in a list. In the list, positive
    number denotes bc 'physical label' as defined
    in gmsh. negative number denotes non-bc nodes."""

    # input mesh is meshio object. it contains boundary element physical tag info.
    # we're going to use that to mark nodes in x_all with physical tag.

    # no_entity = len(mesh.cells)  # among these entities, except for the last one is volume, the rest are boundaries
    nele = config.nele
    nface = config.ndim + 1
    nonods = nloc * nele
    face_iloc_list = face_iloc_array(config.ndim, nloc)
    entity_labels = [key for key in mesh.cell_sets_dict]
    no_labels = len(entity_labels) - 1
    bc_list = [np.zeros(nonods, dtype=bool) for _ in range(no_labels-1)]  # a list of markers
    found = np.zeros(nele*nface, dtype=bool)
    bc_face_list = np.where(alnmt < 0)[0]
    glb_bcface_type = np.ones(nele*nface, dtype=np.int64) * -1
    if config.ndim == 2:
        face_ele_key = 'line'
    else:
        face_ele_key = 'triangle'
    sttime = time.time()
    print('start getting bc nodes... ')
    for ent_id in range(no_labels-1):
        ent_lab = entity_labels[ent_id]
        # print('ent_id, ent_lab', ent_id, ent_lab)
        for fele in mesh.cell_sets_dict[ent_lab][face_ele_key]:
            bc_face_nodes = mesh.cells_dict[face_ele_key][fele]
            for glb_iface in bc_face_list:
                if found[glb_iface]:
                    continue
                if set(faces[glb_iface]) == set(bc_face_nodes):
                    # this is boundary face we're looking for!
                    found[glb_iface] = True
                    glb_bcface_type[glb_iface] = ent_id  # use ent_id to mark bc face's entity id.
                    iface = glb_iface % nface  # local face idx
                    ele = glb_iface // nface  # which element this face is in
                    nod_list_in_this_ele = ele*nloc + face_iloc_list[iface]
                    # print('ele iface nod_list', ele, '|', iface, '|', nod_list_in_this_ele)
                    bc_list[ent_id][nod_list_in_this_ele] = True  # marked!
    print('finishing getting bc nodes... time consumed: %f s' % (time.time() - sttime))
    return bc_list, glb_bcface_type


def get_bc_node_fsi(mesh, faces, alnmt, nbf, nloc, x_all=0):
    """this is to find boundary nodes and
    log them in a list.

    output:
    bc_list: nodes on bc (exclude interface nodes since we
        don't need to apply bc there, they are still treated
        as internal nodes.)
    glb_bcface_type: a list. In the list, positive
    number denotes bc 'physical label' as defined
    in gmsh. negative number denotes non-bc nodes.
    """

    # input mesh is meshio object. it contains boundary element physical tag info.
    # we're going to use that to mark nodes in x_all with physical tag.

    # no_entity = len(mesh.cells)  # among these entities, except for the last one is volume, the rest are boundaries
    nele = config.nele
    nele_f = config.nele_f
    nele_s = config.nele_s
    nface = config.ndim + 1
    nonods = nloc * nele
    face_iloc_list = face_iloc_array(config.ndim, nloc)

    if config.ndim == 2:
        face_ele_key = 'line'
    else:
        face_ele_key = 'triangle'

    entity_labels = [key for key in mesh.cell_sets_dict]
    no_labels = 0  # number of bc labels
    for entity in entity_labels:
        if face_ele_key in mesh.cell_sets_dict[entity]:
            no_labels += 1
    no_labels -= 1  # exclude 'gmsh:bounding_entities'
    # no_labels = len(entity_labels) - 1  # 6 type of bc: diri_f, neu_f, diri_s, neu_s, interface_f, interface_s
    #     # if neu_s is not presented, we have 5.
    #     # but below we will only read in the first 4 or 3 (i.e. no interface_f/interface_s is recorded here)
    bc_list = [np.zeros(nonods, dtype=bool) for _ in range(no_labels)]  # a list of markers
    found = np.zeros(nele*nface, dtype=bool)
    bc_face_list = np.where(alnmt < 0)[0]
    glb_bcface_type = np.ones(nele*nface, dtype=np.int64) * -1
    sttime = time.time()
    print('start getting bc nodes... ')
    for ent_id in range(no_labels):
        ent_lab = entity_labels[ent_id]
        # print('ent_id, ent_lab', ent_id, ent_lab)
        for fele in mesh.cell_sets_dict[ent_lab][face_ele_key]:
            bc_face_nodes = mesh.cells_dict[face_ele_key][fele]
            for glb_iface in bc_face_list:
                if found[glb_iface]:
                    continue
                if set(faces[glb_iface]) == set(bc_face_nodes):
                    # this is boundary face we're looking for!
                    found[glb_iface] = True
                    glb_bcface_type[glb_iface] = ent_id  # use ent_id to mark bc face's entity id.
                    iface = glb_iface % nface  # local face idx
                    ele = glb_iface // nface  # which element this face is in
                    nod_list_in_this_ele = ele*nloc + face_iloc_list[iface]
                    # print('ele iface nod_list', ele, '|', iface, '|', nod_list_in_this_ele)
                    bc_list[ent_id][nod_list_in_this_ele] = True  # marked!
    # now mark faces on interface.
    interior_face_list = np.where(alnmt >= 0)[0]  # interior face
    for glb_iface in interior_face_list:
        glb_jface = nbf[glb_iface]
        glb_iele = glb_iface // nface
        glb_jele = glb_jface // nface
        if glb_iele < nele_f <= glb_jele:
            glb_bcface_type[glb_iface] = 4  # mark as interface_f
        elif glb_iele >= nele_f > glb_jele:
            glb_bcface_type[glb_iface] = 5  # mark as interface_s

    print('finishing getting bc nodes... time consumed: %f s' % (time.time() - sttime))
    return bc_list, glb_bcface_type


# local nodes number on a face
def face_iloc_array(ndim, nloc):
    # return local nodes number on a face
    if ndim == 2:
        if nloc == 10:  # cubic element
            arr = np.asarray([
                0,3,4,1,
                1,5,6,2,
                2,7,8,0,
            ], dtype=np.int64).reshape((3, 4))
        elif nloc == 6:  # quadratic element
            arr = np.asarray([
                0,3,1,
                1,4,2,
                2,5,0,
            ], dtype=np.int64).reshape((3, 3))
        elif nloc == 3:  # linear element
            arr = np.asarray([
                0,1,
                1,2,
                2,0,
            ], dtype=np.int64).reshape((3, 2))
        else:
            raise ValueError('cannot find face nodes list for nloc = %d' % nloc)
    elif ndim == 3:
        if nloc == 20:  # cubic element
            arr = np.asarray([
                1, 2, 3, 8, 9, 14, 15, 12, 13, 16,
                0, 2, 3, 4, 5, 10, 11, 14, 15, 18,
                0, 1, 3, 6, 7, 10, 11, 12, 13, 19,
                0, 1, 2, 4, 5, 6,  7,  8,  9,  17,
            ], dtype=np.int64).reshape((4, 10))
        elif nloc == 10:  # quadratic element
            arr = np.asarray([
                1,2,3,4,8,9,
                0,2,3,5,7,9,
                0,1,3,6,7,8,
                0,1,2,4,5,6,
            ], dtype=np.int64).reshape((4, 6))
        elif nloc == 4:  # linear element
            arr = np.asarray([
                1,2,3,
                0,2,3,
                0,1,3,
                0,1,2,
            ], dtype=np.int64).reshape((4, 3))
        else:
            raise ValueError('cannot find face nodes list for nloc = %d' % nloc)
    else:
        raise ValueError('ndim not correct.')
    return arr


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
