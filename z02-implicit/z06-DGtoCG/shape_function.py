# shape functions on a reference element


import numpy as np 
import config
from config import sf_nd_nb
import torch
from torch.nn import Conv1d,Sequential,Module

nele = config.nele 
mesh = config.mesh 
nonods = config.nonods
ndim = config.ndim
dev = config.dev


def _gi_pnts_tetra(ngi):
    """
    output gaussian points and weights in tetrahedron
    reference tetrahedron:
    (1,0,0) - (0,1,0) - (0,0,1) - (0,0,0)
    """
    if ngi == 24:
        # 24 pnts tetronhedron quadrature rule, 6 degree precision
        # c.f. https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
        # keast7
        L = [0.3561913862225449, 0.2146028712591517, 0.2146028712591517,
             0.2146028712591517, 0.2146028712591517, 0.2146028712591517,
             0.2146028712591517, 0.2146028712591517, 0.3561913862225449,
             0.2146028712591517, 0.3561913862225449, 0.2146028712591517,
             0.8779781243961660, 0.0406739585346113, 0.0406739585346113,
             0.0406739585346113, 0.0406739585346113, 0.0406739585346113,
             0.0406739585346113, 0.0406739585346113, 0.8779781243961660,
             0.0406739585346113, 0.8779781243961660, 0.0406739585346113,
             0.0329863295731731, 0.3223378901422757, 0.3223378901422757,
             0.3223378901422757, 0.3223378901422757, 0.3223378901422757,
             0.3223378901422757, 0.3223378901422757, 0.0329863295731731,
             0.3223378901422757, 0.0329863295731731, 0.3223378901422757,
             0.2696723314583159, 0.0636610018750175, 0.0636610018750175,
             0.0636610018750175, 0.2696723314583159, 0.0636610018750175,
             0.0636610018750175, 0.0636610018750175, 0.2696723314583159,
             0.6030056647916491, 0.0636610018750175, 0.0636610018750175,
             0.0636610018750175, 0.6030056647916491, 0.0636610018750175,
             0.0636610018750175, 0.0636610018750175, 0.6030056647916491,
             0.0636610018750175, 0.2696723314583159, 0.6030056647916491,
             0.2696723314583159, 0.6030056647916491, 0.0636610018750175,
             0.6030056647916491, 0.0636610018750175, 0.2696723314583159,
             0.0636610018750175, 0.6030056647916491, 0.2696723314583159,
             0.2696723314583159, 0.0636610018750175, 0.6030056647916491,
             0.6030056647916491, 0.2696723314583159, 0.0636610018750175]
        L = np.asarray(L, dtype=np.float64)
        L = np.reshape(L, (ngi, 3))
        weight = [0.0399227502581679, 0.0399227502581679, 0.0399227502581679,
                  0.0399227502581679, 0.0100772110553207, 0.0100772110553207,
                  0.0100772110553207, 0.0100772110553207, 0.0553571815436544,
                  0.0553571815436544, 0.0553571815436544, 0.0553571815436544,
                  0.0482142857142857, 0.0482142857142857, 0.0482142857142857,
                  0.0482142857142857, 0.0482142857142857, 0.0482142857142857,
                  0.0482142857142857, 0.0482142857142857, 0.0482142857142857,
                  0.0482142857142857, 0.0482142857142857, 0.0482142857142857]
        weight = np.asarray(weight, dtype=np.float64)
        weight *= 1. / 6.
    elif ngi == 11:
        # keast4 quadrature, 11 points, order 4
        # cf https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
        L = [
            0.2500000000000000, 0.2500000000000000, 0.2500000000000000,
            0.7857142857142857, 0.0714285714285714, 0.0714285714285714,
            0.0714285714285714, 0.0714285714285714, 0.0714285714285714,
            0.0714285714285714, 0.0714285714285714, 0.7857142857142857,
            0.0714285714285714, 0.7857142857142857, 0.0714285714285714,
            0.1005964238332008, 0.3994035761667992, 0.3994035761667992,
            0.3994035761667992, 0.1005964238332008, 0.3994035761667992,
            0.3994035761667992, 0.3994035761667992, 0.1005964238332008,
            0.3994035761667992, 0.1005964238332008, 0.1005964238332008,
            0.1005964238332008, 0.3994035761667992, 0.1005964238332008,
            0.1005964238332008, 0.1005964238332008, 0.3994035761667992,
        ]
        L = np.asarray(L, dtype=np.float64)
        L = np.reshape(L, (ngi, 3))
        weight = [
            -0.0789333333333333, 0.0457333333333333, 0.0457333333333333,
            0.0457333333333333, 0.0457333333333333, 0.1493333333333333, 0.1493333333333333,
            0.1493333333333333, 0.1493333333333333, 0.1493333333333333, 0.1493333333333333,
        ]
        weight = np.asarray(weight, dtype=np.float64)
        weight *= 1. / 6.
    elif ngi == 4:
        L = [0.1381966011250105, 0.1381966011250105, 0.1381966011250105,
             0.1381966011250105, 0.1381966011250105, 0.5854101966249685,
             0.1381966011250105, 0.5854101966249685, 0.1381966011250105,
             0.5854101966249685, 0.1381966011250105, 0.1381966011250105, ]
        L = np.asarray(L, dtype=np.float64)
        L = np.reshape(L, (ngi, 3))
        weight = [0.25, 0.25, 0.25, 0.25]
        weight = np.asarray(weight, dtype=np.float64)
        weight *= 1. / 6.
    else:
        raise Exception('ngi ', ngi, 'for tetrahedron is not implemented or existed!')
    return L, weight


def _gi_pnts_tri(sngi):
    """
    output gaussian points and weights in triangle
    also output neighbour face gaussian points order in 3 different
    alignment.
    reference triangle:
    (1,0) - (0,1) - (0,0)
    neighbour face: note that nodes in neighbours are in opposite order.
    alignment 0:
    (1,0) - (0,0) - (0,1)
    alignment 1:
    (0,1) - (1,0) - (0,0)
    alignment 2:
    (0,0) - (0,1) - (1,0)
    """
    if sngi == 9:
        # 9 pnts triangle quadrature rule, 6 degree precision
        # c.f. https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tri/quadrature_rules_tri.html
        # strang8
        a = 0.437525248383384
        b = 0.124949503233232
        c = 0.797112651860071
        d = 0.165409927389841
        e = 0.037477420750088
        pnts = np.asarray(
            [b, a, a,
             a, b, a,
             a, a, b,
             c, e, d,
             c, d, e,
             d, c, e,
             e, c, d,
             e, d, c,
             d, e, c,
             ],
            dtype=np.float64).reshape((sngi, 3))
        sweight = [0.205950504760887, 0.205950504760887, 0.205950504760887,
                   0.063691414286223, 0.063691414286223, 0.063691414286223,
                   0.063691414286223, 0.063691414286223, 0.063691414286223]
        # alignment = [1, 3, 2, 5, 4, 7, 6, 9, 8,
        #              2, 1, 3, 6, 8, 4, 9, 5, 7,
        #              3, 2, 1, 9, 7, 8, 5, 6, 4]
        alignment = [0, 2, 1, 4, 3, 8, 7, 6, 5,
                     1, 0, 2, 6, 5, 4, 3, 8, 7,
                     2, 1, 0, 8, 7, 6, 5, 4, 3]
        sweight = np.asarray(sweight, dtype=np.float64)
        sweight *= 0.5
    elif sngi == 12:
        # strang9, 12 points, degree of precision 6
        pnts = np.asarray([
            0.873821971016996, 0.063089014491502,
            0.063089014491502, 0.873821971016996,
            0.063089014491502, 0.063089014491502,
            0.501426509658179, 0.249286745170910,
            0.249286745170910, 0.501426509658179,
            0.249286745170910, 0.249286745170910,
            0.636502499121399, 0.310352451033785,
            0.636502499121399, 0.053145049844816,
            0.310352451033785, 0.636502499121399,
            0.310352451033785, 0.053145049844816,
            0.053145049844816, 0.636502499121399,
            0.053145049844816, 0.310352451033785,
        ], dtype=np.float64).reshape((sngi, 2))
        pnts3 = 1 - np.sum(pnts, axis=1)
        pnts = np.concatenate((pnts, pnts3.reshape(sngi, 1)), axis=1)
        sweight = np.asarray([
            0.050844906370207,
            0.050844906370207,
            0.050844906370207,
            0.116786275726379,
            0.116786275726379,
            0.116786275726379,
            0.082851075618374,
            0.082851075618374,
            0.082851075618374,
            0.082851075618374,
            0.082851075618374,
            0.082851075618374,
        ], dtype=np.float64)
        sweight *= 0.5
        alignment = np.asarray([
            1, 3, 2, 4, 6, 5, 8, 7, 10, 9, 12, 11,
            2, 1, 3, 5, 4, 6, 9, 11, 7, 12, 8, 10,
            3, 2, 1, 6, 5, 4, 12, 10, 11, 8, 9, 7,
        ]) - 1
    elif sngi == 6:
        # strang5 6 pnts quadrature rule, order 4 precision
        # cf https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tri/quadrature_rules_tri.html
        a = 0.816847572980459
        b = 0.091576213509771
        c = 0.108103018168070
        d = 0.445948490915965
        pnts = np.asarray([
            a, b, b,
            b, a, b,
            b, b, a,
            c, d, d,
            d, c, d,
            d, d, c,
        ], dtype=np.float64).reshape((sngi, 3))
        sweight = [0.109951743655322, 0.109951743655322, 0.109951743655322,
                   0.223381589678011, 0.223381589678011, 0.223381589678011]
        alignment = [
            0, 2, 1, 3, 5, 4,
            1, 0, 2, 4, 3, 5,
            2, 1, 0, 5, 4, 3,
        ]
        sweight = np.asarray(sweight, dtype=np.float64)
        sweight *= 0.5
    elif sngi == 3:
        # 3 pnts triangle quadrature rule, 2 degree precision
        # strang2
        pnts = np.asarray([0.5, 0, 0.5,
                           0.5, 0.5, 0,
                           0, 0.5, 0.5], dtype=np.float64).reshape((sngi, 3))
        sweight = [1. / 3., 1. / 3., 1. / 3.]
        alignment = [
            1, 0, 2,
            2, 1, 0,
            0, 2, 1,
        ]
        sweight = np.asarray(sweight, dtype=np.float64)
        sweight *= 0.5
    else:
        raise Exception('ngi ', sngi, 'for triangle is not implemented or existed!')
    return pnts, sweight, alignment


def SHATRInew(nloc,ngi,ndim, snloc, sngi):
    '''
    shape functions on a reference element

    input: 
    
    nloc, ngi, ndim

    surface input: snloc, sngi

    output: 

    n, shape functions on a reference element at quadrature points, 
          numpy array (nloc, ngi)

    nlx, shape function deriatives on a reference element at quad pnts,
          numpy array (ndim, nloc, ngi)

    weight, quad pnts weights, np array (ngi)

    sn, surface shape functions on a reference element, numpy array (nface,nloc,sngi)

    snlx_all, shape function derivatives on surface, on a reference element, 
           numpy array (nface, ndim, nloc, sngi)

    sweight, quad pnts weights on face, np array (sngi)
    '''

    if ndim == 3 :
        # 3D tetrahedron
        # volume shape function
        if nloc == 20:  # cubic elements
            # if ngi != 24:
            #     raise Exception('ngi ', ngi, 'is not compatible with nloc ', nloc)
            L, weight = _gi_pnts_tetra(ngi)
            nface = ndim + 1

            n = np.zeros((nloc, ngi))
            nlx = np.zeros((nloc, ngi))
            nly = np.zeros((nloc, ngi))
            nlz = np.zeros((nloc, ngi))
            for gi in range(ngi):
                l1 = L[gi,0]
                l2 = L[gi,1]
                l3 = L[gi,2]
                l4 = 1. - l1 - l2 - l3
                # corner nodes
                n[0, gi] = 1. / 2. * (3. * l1 - 1) * (3 * l1 - 2) * l1
                n[1, gi] = 1. / 2. * (3. * l2 - 1) * (3 * l2 - 2) * l2
                n[2, gi] = 1. / 2. * (3. * l3 - 1) * (3 * l3 - 2) * l3
                n[3, gi] = 1. / 2. * (3. * l4 - 1) * (3 * l4 - 2) * l4
                # edge nodes
                n[4, gi] = 9 / 2 * (3 * l1 - 1) * l1 * l3
                n[5, gi] = 9 / 2 * (3 * l3 - 1) * l1 * l3
                n[6, gi] = 9 / 2 * (3 * l1 - 1) * l1 * l2
                n[7, gi] = 9 / 2 * (3 * l2 - 1) * l1 * l2
                n[8, gi] = 9 / 2 * (3 * l2 - 1) * l2 * l3
                n[9, gi] = 9 / 2 * (3 * l3 - 1) * l2 * l3
                n[10, gi] = 9 / 2 * (3 * l1 - 1) * l1 * l4
                n[11, gi] = 9 / 2 * (3 * l4 - 1) * l1 * l4
                n[12, gi] = 9 / 2 * (3 * l2 - 1) * l2 * l4
                n[13, gi] = 9 / 2 * (3 * l4 - 1) * l2 * l4
                n[14, gi] = 9 / 2 * (3 * l3 - 1) * l3 * l4
                n[15, gi] = 9 / 2 * (3 * l4 - 1) * l3 * l4
                # centre surface nodes
                n[16, gi] = 27 * l2 * l3 * l4
                n[17, gi] = 27 * l1 * l2 * l3
                n[18, gi] = 27 * l1 * l3 * l4
                n[19, gi] = 27 * l1 * l2 * l4
                # x - derivative
                nlx[0, gi] = (3 * l1 - 2) * ((3 * l1) / 2 - 1 / 2) + (3 * l1 * (3 * l1 - 2)) / 2 + 3 * l1 * (
                            (3 * l1) / 2 - 1 / 2)
                nlx[1, gi] = 0
                nlx[2, gi] = 0
                nlx[3, gi] = - (3 * (l1 + l2 + l3 - 1) * (3 * l1 + 3 * l2 + 3 * l3 - 1)) / 2 - 3 * (
                            l1 + l2 + l3 - 1) * ((3 * l1) / 2 + (3 * l2) / 2 + (3 * l3) / 2 - 1) - (
                                         3 * l1 + 3 * l2 + 3 * l3 - 1) * (
                                         (3 * l1) / 2 + (3 * l2) / 2 + (3 * l3) / 2 - 1)
                nlx[4, gi] = (27 * l1 * l3) / 2 + l3 * ((27 * l1) / 2 - 9 / 2)
                nlx[5, gi] = l3 * ((27 * l3) / 2 - 9 / 2)
                nlx[6, gi] = (27 * l1 * l2) / 2 + l2 * ((27 * l1) / 2 - 9 / 2)
                nlx[7, gi] = l2 * ((27 * l2) / 2 - 9 / 2)
                nlx[8, gi] = 0
                nlx[9, gi] = 0
                nlx[10, gi] = - ((27 * l1) / 2 - 9 / 2) * (l1 + l2 + l3 - 1) - (
                            27 * l1 * (l1 + l2 + l3 - 1)) / 2 - l1 * ((27 * l1) / 2 - 9 / 2)
                nlx[11, gi] = (27 * l1 * (l1 + l2 + l3 - 1)) / 2 + l1 * (
                            (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9) + (l1 + l2 + l3 - 1) * (
                                          (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9)
                nlx[12, gi] = -l2 * ((27 * l2) / 2 - 9 / 2)
                nlx[13, gi] = (27 * l2 * (l1 + l2 + l3 - 1)) / 2 + l2 * (
                            (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9)
                nlx[14, gi] = -l3 * ((27 * l3) / 2 - 9 / 2)
                nlx[15, gi] = (27 * l3 * (l1 + l2 + l3 - 1)) / 2 + l3 * (
                            (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9)
                nlx[16, gi] = -27 * l2 * l3
                nlx[17, gi] = 27 * l2 * l3
                nlx[18, gi] = - 27 * l1 * l3 - 27 * l3 * (l1 + l2 + l3 - 1)
                nlx[19, gi] = - 27 * l1 * l2 - 27 * l2 * (l1 + l2 + l3 - 1)
                # y - derivative
                nly[0, gi] = 0
                nly[1, gi] = (3 * l2 - 2) * ((3 * l2) / 2 - 1 / 2) + (3 * l2 * (3 * l2 - 2)) / 2 + 3 * l2 * (
                            (3 * l2) / 2 - 1 / 2)
                nly[2, gi] = 0
                nly[3, gi] = - (3 * (l1 + l2 + l3 - 1) * (3 * l1 + 3 * l2 + 3 * l3 - 1)) / 2 - 3 * (
                            l1 + l2 + l3 - 1) * ((3 * l1) / 2 + (3 * l2) / 2 + (3 * l3) / 2 - 1) - (
                                         3 * l1 + 3 * l2 + 3 * l3 - 1) * (
                                         (3 * l1) / 2 + (3 * l2) / 2 + (3 * l3) / 2 - 1)
                nly[4, gi] = 0
                nly[5, gi] = 0
                nly[6, gi] = l1 * ((27 * l1) / 2 - 9 / 2)
                nly[7, gi] = (27 * l1 * l2) / 2 + l1 * ((27 * l2) / 2 - 9 / 2)
                nly[8, gi] = (27 * l2 * l3) / 2 + l3 * ((27 * l2) / 2 - 9 / 2)
                nly[9, gi] = l3 * ((27 * l3) / 2 - 9 / 2)
                nly[10, gi] = -l1 * ((27 * l1) / 2 - 9 / 2)
                nly[11, gi] = (27 * l1 * (l1 + l2 + l3 - 1)) / 2 + l1 * (
                            (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9)
                nly[12, gi] = - ((27 * l2) / 2 - 9 / 2) * (l1 + l2 + l3 - 1) - (
                            27 * l2 * (l1 + l2 + l3 - 1)) / 2 - l2 * ((27 * l2) / 2 - 9 / 2)
                nly[13, gi] = (27 * l2 * (l1 + l2 + l3 - 1)) / 2 + l2 * (
                            (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9) + (l1 + l2 + l3 - 1) * (
                                          (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9)
                nly[14, gi] = -l3 * ((27 * l3) / 2 - 9 / 2)
                nly[15, gi] = (27 * l3 * (l1 + l2 + l3 - 1)) / 2 + l3 * (
                            (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9)
                nly[16, gi] = - 27 * l2 * l3 - 27 * l3 * (l1 + l2 + l3 - 1)
                nly[17, gi] = 27 * l1 * l3
                nly[18, gi] = -27 * l1 * l3
                nly[19, gi] = - 27 * l1 * l2 - 27 * l1 * (l1 + l2 + l3 - 1)
                # z - derivative
                nlz[0, gi] = 0
                nlz[1, gi] = 0
                nlz[2, gi] = (3 * l3 - 2) * ((3 * l3) / 2 - 1 / 2) + (3 * l3 * (3 * l3 - 2)) / 2 + 3 * l3 * (
                            (3 * l3) / 2 - 1 / 2)
                nlz[3, gi] = - (3 * (l1 + l2 + l3 - 1) * (3 * l1 + 3 * l2 + 3 * l3 - 1)) / 2 - 3 * (
                            l1 + l2 + l3 - 1) * ((3 * l1) / 2 + (3 * l2) / 2 + (3 * l3) / 2 - 1) - (
                                         3 * l1 + 3 * l2 + 3 * l3 - 1) * (
                                         (3 * l1) / 2 + (3 * l2) / 2 + (3 * l3) / 2 - 1)
                nlz[4, gi] = l1 * ((27 * l1) / 2 - 9 / 2)
                nlz[5, gi] = (27 * l1 * l3) / 2 + l1 * ((27 * l3) / 2 - 9 / 2)
                nlz[6, gi] = 0
                nlz[7, gi] = 0
                nlz[8, gi] = l2 * ((27 * l2) / 2 - 9 / 2)
                nlz[9, gi] = (27 * l2 * l3) / 2 + l2 * ((27 * l3) / 2 - 9 / 2)
                nlz[10, gi] = -l1 * ((27 * l1) / 2 - 9 / 2)
                nlz[11, gi] = (27 * l1 * (l1 + l2 + l3 - 1)) / 2 + l1 * (
                            (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9)
                nlz[12, gi] = -l2 * ((27 * l2) / 2 - 9 / 2)
                nlz[13, gi] = (27 * l2 * (l1 + l2 + l3 - 1)) / 2 + l2 * (
                            (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9)
                nlz[14, gi] = - ((27 * l3) / 2 - 9 / 2) * (l1 + l2 + l3 - 1) - (
                            27 * l3 * (l1 + l2 + l3 - 1)) / 2 - l3 * ((27 * l3) / 2 - 9 / 2)
                nlz[15, gi] = (27 * l3 * (l1 + l2 + l3 - 1)) / 2 + l3 * (
                            (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9) + (l1 + l2 + l3 - 1) * (
                                          (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9)
                nlz[16, gi] = - 27 * l2 * l3 - 27 * l2 * (l1 + l2 + l3 - 1)
                nlz[17, gi] = 27 * l1 * l2
                nlz[18, gi] = - 27 * l1 * l3 - 27 * l1 * (l1 + l2 + l3 - 1)
                nlz[19, gi] = -27 * l1 * l2
        elif nloc == 10:  # quadratic element
            # if ngi != 11:
            #     raise Exception('ngi ', ngi, 'is not compatible iwth nloc ', nloc)
            L, weight = _gi_pnts_tetra(ngi)
            nface = ndim + 1

            n = np.zeros((nloc, ngi))
            nlx = np.zeros((nloc, ngi))
            nly = np.zeros((nloc, ngi))
            nlz = np.zeros((nloc, ngi))
            for gi in range(ngi):
                l1 = L[gi, 0]
                l2 = L[gi, 1]
                l3 = L[gi, 2]
                l4 = 1. - l1 - l2 - l3
                # corner nodes
                n[0, gi] = (2 * l1 - 1) * l1
                n[1, gi] = (2 * l2 - 1) * l2
                n[2, gi] = (2 * l3 - 1) * l3
                n[3, gi] = (2 * l4 - 1) * l4
                # edge nodes
                n[4, gi] = 4 * l2 * l3
                n[5, gi] = 4 * l1 * l3
                n[6, gi] = 4 * l1 * l2
                n[7, gi] = 4 * l1 * l4
                n[8, gi] = 4 * l2 * l4
                n[9, gi] = 4 * l3 * l4
                # x-derivative
                nlx[0, gi] = 4. * l1 - 1.
                nlx[1, gi] = 0
                nlx[2, gi] = 0
                nlx[3, gi] = 4. * l1 + 4. * l2 + 4. * l3 - 3.
                nlx[4, gi] = 0
                nlx[5, gi] = 4. * l3
                nlx[6, gi] = 4. * l2
                nlx[7, gi] = 4. - 4. * l2 - 4. * l3 - 8. * l1
                nlx[8, gi] = -4. * l2
                nlx[9, gi] = -4. * l3
                # y-derivative
                nly[0, gi] = 0
                nly[1, gi] = 4. * l2 - 1.
                nly[2, gi] = 0
                nly[3, gi] = 4. * l1 + 4. * l2 + 4. * l3 - 3.
                nly[4, gi] = 4. * l3
                nly[5, gi] = 0
                nly[6, gi] = 4. * l1
                nly[7, gi] = -4. * l1
                nly[8, gi] = 4. - 8. * l2 - 4. * l3 - 4. * l1
                nly[9, gi] = -4. * l3
                # z-derivative
                nlz[0, gi] = 0
                nlz[1, gi] = 0
                nlz[2, gi] = 4. * l3 - 1.
                nlz[3, gi] = 4. * l1 + 4. * l2 + 4. * l3 - 3.
                nlz[4, gi] = 4. * l2
                nlz[5, gi] = 4. * l1
                nlz[6, gi] = 0
                nlz[7, gi] = -4. * l1
                nlz[8, gi] = -4. * l2
                nlz[9, gi] = 4. - 4. * l2 - 8. * l3 - 4. * l1
        elif nloc == 4:  # linear element
            # if ngi != 4:
            #     raise Exception('ngi ', ngi, 'is not compatible with nloc ', nloc)
            L, weight = _gi_pnts_tetra(ngi)
            nface = ndim + 1

            n = np.zeros((nloc, ngi))
            nlx = np.zeros((nloc, ngi))
            nly = np.zeros((nloc, ngi))
            nlz = np.zeros((nloc, ngi))
            for gi in range(ngi):
                l1 = L[gi, 0]
                l2 = L[gi, 1]
                l3 = L[gi, 2]
                l4 = 1. - l1 - l2 - l3
                # corner nodes
                n[0, gi] = l1
                n[1, gi] = l2
                n[2, gi] = l3
                n[3, gi] = l4
                # x-derivative
                nlx[0, gi] = 1.
                nlx[1, gi] = 0
                nlx[2, gi] = 0
                nlx[3, gi] = -1.
                # y-derivative
                nly[0, gi] = 0
                nly[1, gi] = 1.
                nly[2, gi] = 0
                nly[3, gi] = -1.
                # z-derivative
                nlz[0, gi] = 0
                nlz[1, gi] = 0
                nlz[2, gi] = 1.
                nlz[3, gi] = -1.
        else:  # nloc
            raise Exception('nloc %d is not accepted in 3D' % nloc)

        # face shape function
        if snloc == 10:  # cubic face shape functions
            # if sngi != 9:
            #     raise Exception('sngi ', sngi, 'is not compatible with snloc ', snloc)
            pnts, sweight, alignment = _gi_pnts_tri(sngi)
            sf_nd_nb.set_data(gi_align=torch.tensor(alignment, device=dev, dtype=torch.int64).view(ndim, sngi))
            SL = np.zeros((nface, sngi, 4), dtype=np.float64)
            # face1  triangle 3-2-4, l1 = 0
            SL[0, :, 0] = 0
            SL[0, :, 1] = pnts[:, 1]
            SL[0, :, 2] = pnts[:, 0]
            SL[0, :, 3] = pnts[:, 2]
            # face2  triangle 1-3-4, l2 = 0
            SL[1, :, 0] = pnts[:, 0]
            SL[1, :, 1] = 0
            SL[1, :, 2] = pnts[:, 1]
            SL[1, :, 3] = pnts[:, 2]
            # face3  triangle 2-1-4, l3 = 0
            SL[2, :, 0] = pnts[:, 1]
            SL[2, :, 1] = pnts[:, 0]
            SL[2, :, 2] = 0
            SL[2, :, 3] = pnts[:, 2]
            # face4  triangle 1-2-3, l4 = 0
            SL[3, :, 0] = pnts[:, 0]
            SL[3, :, 1] = pnts[:, 1]
            SL[3, :, 2] = pnts[:, 2]
            SL[3, :, 3] = 0

            sn = np.zeros((nface, nloc, sngi))
            snlx = np.zeros((nface, nloc, sngi))
            snly = np.zeros((nface, nloc, sngi))
            snlz = np.zeros((nface, nloc, sngi))
            for iface in range(nface):
                for gi in range(sngi):
                    l1 = SL[iface, gi, 0]
                    l2 = SL[iface, gi, 1]
                    l3 = SL[iface, gi, 2]
                    l4 = SL[iface, gi, 3]
                    # corner nodes
                    sn[iface, 0, gi] = 1. / 2. * (3. * l1 - 1) * (3 * l1 - 2) * l1
                    sn[iface, 1, gi] = 1. / 2. * (3. * l2 - 1) * (3 * l2 - 2) * l2
                    sn[iface, 2, gi] = 1. / 2. * (3. * l3 - 1) * (3 * l3 - 2) * l3
                    sn[iface, 3, gi] = 1. / 2. * (3. * l4 - 1) * (3 * l4 - 2) * l4
                    # edge nodes
                    sn[iface, 4, gi] = 9 / 2 * (3 * l1 - 1) * l1 * l3
                    sn[iface, 5, gi] = 9 / 2 * (3 * l3 - 1) * l1 * l3
                    sn[iface, 6, gi] = 9 / 2 * (3 * l1 - 1) * l1 * l2
                    sn[iface, 7, gi] = 9 / 2 * (3 * l2 - 1) * l1 * l2
                    sn[iface, 8, gi] = 9 / 2 * (3 * l2 - 1) * l2 * l3
                    sn[iface, 9, gi] = 9 / 2 * (3 * l3 - 1) * l2 * l3
                    sn[iface, 10, gi] = 9 / 2 * (3 * l1 - 1) * l1 * l4
                    sn[iface, 11, gi] = 9 / 2 * (3 * l4 - 1) * l1 * l4
                    sn[iface, 12, gi] = 9 / 2 * (3 * l2 - 1) * l2 * l4
                    sn[iface, 13, gi] = 9 / 2 * (3 * l4 - 1) * l2 * l4
                    sn[iface, 14, gi] = 9 / 2 * (3 * l3 - 1) * l3 * l4
                    sn[iface, 15, gi] = 9 / 2 * (3 * l4 - 1) * l3 * l4
                    # centre surface nodes
                    sn[iface, 16, gi] = 27 * l2 * l3 * l4
                    sn[iface, 17, gi] = 27 * l1 * l2 * l3
                    sn[iface, 18, gi] = 27 * l1 * l3 * l4
                    sn[iface, 19, gi] = 27 * l1 * l2 * l4
                    # x - derivative
                    snlx[iface, 0, gi] = (3 * l1 - 2) * ((3 * l1) / 2 - 1 / 2) + (3 * l1 * (3 * l1 - 2)) / 2 + 3 * l1 * (
                            (3 * l1) / 2 - 1 / 2)
                    snlx[iface, 1, gi] = 0
                    snlx[iface, 2, gi] = 0
                    snlx[iface, 3, gi] = - (3 * (l1 + l2 + l3 - 1) * (3 * l1 + 3 * l2 + 3 * l3 - 1)) / 2 - 3 * (
                            l1 + l2 + l3 - 1) * ((3 * l1) / 2 + (3 * l2) / 2 + (3 * l3) / 2 - 1) - (
                                         3 * l1 + 3 * l2 + 3 * l3 - 1) * (
                                         (3 * l1) / 2 + (3 * l2) / 2 + (3 * l3) / 2 - 1)
                    snlx[iface, 4, gi] = (27 * l1 * l3) / 2 + l3 * ((27 * l1) / 2 - 9 / 2)
                    snlx[iface, 5, gi] = l3 * ((27 * l3) / 2 - 9 / 2)
                    snlx[iface, 6, gi] = (27 * l1 * l2) / 2 + l2 * ((27 * l1) / 2 - 9 / 2)
                    snlx[iface, 7, gi] = l2 * ((27 * l2) / 2 - 9 / 2)
                    snlx[iface, 8, gi] = 0
                    snlx[iface, 9, gi] = 0
                    snlx[iface, 10, gi] = - ((27 * l1) / 2 - 9 / 2) * (l1 + l2 + l3 - 1) - (
                            27 * l1 * (l1 + l2 + l3 - 1)) / 2 - l1 * ((27 * l1) / 2 - 9 / 2)
                    snlx[iface, 11, gi] = (27 * l1 * (l1 + l2 + l3 - 1)) / 2 + l1 * (
                            (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9) + (l1 + l2 + l3 - 1) * (
                                          (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9)
                    snlx[iface, 12, gi] = -l2 * ((27 * l2) / 2 - 9 / 2)
                    snlx[iface, 13, gi] = (27 * l2 * (l1 + l2 + l3 - 1)) / 2 + l2 * (
                            (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9)
                    snlx[iface, 14, gi] = -l3 * ((27 * l3) / 2 - 9 / 2)
                    snlx[iface, 15, gi] = (27 * l3 * (l1 + l2 + l3 - 1)) / 2 + l3 * (
                            (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9)
                    snlx[iface, 16, gi] = -27 * l2 * l3
                    snlx[iface, 17, gi] = 27 * l2 * l3
                    snlx[iface, 18, gi] = - 27 * l1 * l3 - 27 * l3 * (l1 + l2 + l3 - 1)
                    snlx[iface, 19, gi] = - 27 * l1 * l2 - 27 * l2 * (l1 + l2 + l3 - 1)
                    # y - derivative
                    snly[iface, 0, gi] = 0
                    snly[iface, 1, gi] = (3 * l2 - 2) * ((3 * l2) / 2 - 1 / 2) + (3 * l2 * (3 * l2 - 2)) / 2 + 3 * l2 * (
                            (3 * l2) / 2 - 1 / 2)
                    snly[iface, 2, gi] = 0
                    snly[iface, 3, gi] = - (3 * (l1 + l2 + l3 - 1) * (3 * l1 + 3 * l2 + 3 * l3 - 1)) / 2 - 3 * (
                            l1 + l2 + l3 - 1) * ((3 * l1) / 2 + (3 * l2) / 2 + (3 * l3) / 2 - 1) - (
                                         3 * l1 + 3 * l2 + 3 * l3 - 1) * (
                                         (3 * l1) / 2 + (3 * l2) / 2 + (3 * l3) / 2 - 1)
                    snly[iface, 4, gi] = 0
                    snly[iface, 5, gi] = 0
                    snly[iface, 6, gi] = l1 * ((27 * l1) / 2 - 9 / 2)
                    snly[iface, 7, gi] = (27 * l1 * l2) / 2 + l1 * ((27 * l2) / 2 - 9 / 2)
                    snly[iface, 8, gi] = (27 * l2 * l3) / 2 + l3 * ((27 * l2) / 2 - 9 / 2)
                    snly[iface, 9, gi] = l3 * ((27 * l3) / 2 - 9 / 2)
                    snly[iface, 10, gi] = -l1 * ((27 * l1) / 2 - 9 / 2)
                    snly[iface, 11, gi] = (27 * l1 * (l1 + l2 + l3 - 1)) / 2 + l1 * (
                            (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9)
                    snly[iface, 12, gi] = - ((27 * l2) / 2 - 9 / 2) * (l1 + l2 + l3 - 1) - (
                            27 * l2 * (l1 + l2 + l3 - 1)) / 2 - l2 * ((27 * l2) / 2 - 9 / 2)
                    snly[iface, 13, gi] = (27 * l2 * (l1 + l2 + l3 - 1)) / 2 + l2 * (
                            (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9) + (l1 + l2 + l3 - 1) * (
                                          (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9)
                    snly[iface, 14, gi] = -l3 * ((27 * l3) / 2 - 9 / 2)
                    snly[iface, 15, gi] = (27 * l3 * (l1 + l2 + l3 - 1)) / 2 + l3 * (
                            (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9)
                    snly[iface, 16, gi] = - 27 * l2 * l3 - 27 * l3 * (l1 + l2 + l3 - 1)
                    snly[iface, 17, gi] = 27 * l1 * l3
                    snly[iface, 18, gi] = -27 * l1 * l3
                    snly[iface, 19, gi] = - 27 * l1 * l2 - 27 * l1 * (l1 + l2 + l3 - 1)
                    # z - derivative
                    snlz[iface, 0, gi] = 0
                    snlz[iface, 1, gi] = 0
                    snlz[iface, 2, gi] = (3 * l3 - 2) * ((3 * l3) / 2 - 1 / 2) + (3 * l3 * (3 * l3 - 2)) / 2 + 3 * l3 * (
                            (3 * l3) / 2 - 1 / 2)
                    snlz[iface, 3, gi] = - (3 * (l1 + l2 + l3 - 1) * (3 * l1 + 3 * l2 + 3 * l3 - 1)) / 2 - 3 * (
                            l1 + l2 + l3 - 1) * ((3 * l1) / 2 + (3 * l2) / 2 + (3 * l3) / 2 - 1) - (
                                         3 * l1 + 3 * l2 + 3 * l3 - 1) * (
                                         (3 * l1) / 2 + (3 * l2) / 2 + (3 * l3) / 2 - 1)
                    snlz[iface, 4, gi] = l1 * ((27 * l1) / 2 - 9 / 2)
                    snlz[iface, 5, gi] = (27 * l1 * l3) / 2 + l1 * ((27 * l3) / 2 - 9 / 2)
                    snlz[iface, 6, gi] = 0
                    snlz[iface, 7, gi] = 0
                    snlz[iface, 8, gi] = l2 * ((27 * l2) / 2 - 9 / 2)
                    snlz[iface, 9, gi] = (27 * l2 * l3) / 2 + l2 * ((27 * l3) / 2 - 9 / 2)
                    snlz[iface, 10, gi] = -l1 * ((27 * l1) / 2 - 9 / 2)
                    snlz[iface, 11, gi] = (27 * l1 * (l1 + l2 + l3 - 1)) / 2 + l1 * (
                            (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9)
                    snlz[iface, 12, gi] = -l2 * ((27 * l2) / 2 - 9 / 2)
                    snlz[iface, 13, gi] = (27 * l2 * (l1 + l2 + l3 - 1)) / 2 + l2 * (
                            (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9)
                    snlz[iface, 14, gi] = - ((27 * l3) / 2 - 9 / 2) * (l1 + l2 + l3 - 1) - (
                            27 * l3 * (l1 + l2 + l3 - 1)) / 2 - l3 * ((27 * l3) / 2 - 9 / 2)
                    snlz[iface, 15, gi] = (27 * l3 * (l1 + l2 + l3 - 1)) / 2 + l3 * (
                            (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9) + (l1 + l2 + l3 - 1) * (
                                          (27 * l1) / 2 + (27 * l2) / 2 + (27 * l3) / 2 - 9)
                    snlz[iface, 16, gi] = - 27 * l2 * l3 - 27 * l2 * (l1 + l2 + l3 - 1)
                    snlz[iface, 17, gi] = 27 * l1 * l2
                    snlz[iface, 18, gi] = - 27 * l1 * l3 - 27 * l1 * (l1 + l2 + l3 - 1)
                    snlz[iface, 19, gi] = -27 * l1 * l2
        elif snloc == 6:  # quadratic element
            # if sngi != 6:
            #     raise Exception('sngi ', sngi, 'is not compatible with snloc ', snloc)
            pnts, sweight, alignment = _gi_pnts_tri(sngi)

            sf_nd_nb.set_data(gi_align=torch.tensor(alignment,
                                                    device=dev,
                                                    dtype=torch.int64).view(ndim, sngi))
            SL = np.zeros((nface, sngi, 4), dtype=np.float64)
            # face1  triangle 2-1-3, l1 = 0
            SL[0, :, 0] = 0
            SL[0, :, 1] = pnts[:, 1]
            SL[0, :, 2] = pnts[:, 0]
            SL[0, :, 3] = pnts[:, 2]
            # face2  triangle 0-2-3, l2 = 0
            SL[1, :, 0] = pnts[:, 0]
            SL[1, :, 1] = 0
            SL[1, :, 2] = pnts[:, 1]
            SL[1, :, 3] = pnts[:, 2]
            # face3  triangle 1-0-3, l3 = 0
            SL[2, :, 0] = pnts[:, 1]
            SL[2, :, 1] = pnts[:, 0]
            SL[2, :, 2] = 0
            SL[2, :, 3] = pnts[:, 2]
            # face4  triangle 0-1-2, l4 = 0
            SL[3, :, 0] = pnts[:, 0]
            SL[3, :, 1] = pnts[:, 1]
            SL[3, :, 2] = pnts[:, 2]
            SL[3, :, 3] = 0

            sn = np.zeros((nface, nloc, sngi))
            snlx = np.zeros((nface, nloc, sngi))
            snly = np.zeros((nface, nloc, sngi))
            snlz = np.zeros((nface, nloc, sngi))
            for iface in range(nface):
                for gi in range(sngi):
                    l1 = SL[iface, gi, 0]
                    l2 = SL[iface, gi, 1]
                    l3 = SL[iface, gi, 2]
                    l4 = SL[iface, gi, 3]
                    # cornor nodes
                    sn[iface, 0, gi] = (2 * l1 - 1) * l1
                    sn[iface, 1, gi] = (2 * l2 - 1) * l2
                    sn[iface, 2, gi] = (2 * l3 - 1) * l3
                    sn[iface, 3, gi] = (2 * l4 - 1) * l4
                    # edge nodes
                    sn[iface, 4, gi] = 4 * l2 * l3
                    sn[iface, 5, gi] = 4 * l1 * l3
                    sn[iface, 6, gi] = 4 * l1 * l2
                    sn[iface, 7, gi] = 4 * l1 * l4
                    sn[iface, 8, gi] = 4 * l2 * l4
                    sn[iface, 9, gi] = 4 * l3 * l4
                    # x-derivative
                    snlx[iface, 0, gi] = 4. * l1 - 1.
                    snlx[iface, 1, gi] = 0
                    snlx[iface, 2, gi] = 0
                    snlx[iface, 3, gi] = 4. * l1 + 4. * l2 + 4. * l3 - 3.
                    snlx[iface, 4, gi] = 0
                    snlx[iface, 5, gi] = 4. * l3
                    snlx[iface, 6, gi] = 4. * l2
                    snlx[iface, 7, gi] = 4. - 4. * l2 - 4. * l3 - 8. * l1
                    snlx[iface, 8, gi] = -4. * l2
                    snlx[iface, 9, gi] = -4. * l3
                    # y-derivative
                    snly[iface, 0, gi] = 0
                    snly[iface, 1, gi] = 4. * l2 - 1.
                    snly[iface, 2, gi] = 0
                    snly[iface, 3, gi] = 4. * l1 + 4. * l2 + 4. * l3 - 3.
                    snly[iface, 4, gi] = 4. * l3
                    snly[iface, 5, gi] = 0
                    snly[iface, 6, gi] = 4. * l1
                    snly[iface, 7, gi] = -4. * l1
                    snly[iface, 8, gi] = 4. - 8. * l2 - 4. * l3 - 4. * l1
                    snly[iface, 9, gi] = -4. * l3
                    # z-derivative
                    snlz[iface, 0, gi] = 0
                    snlz[iface, 1, gi] = 0
                    snlz[iface, 2, gi] = 4. * l3 - 1.
                    snlz[iface, 3, gi] = 4. * l1 + 4. * l2 + 4. * l3 - 3.
                    snlz[iface, 4, gi] = 4. * l2
                    snlz[iface, 5, gi] = 4. * l1
                    snlz[iface, 6, gi] = 0
                    snlz[iface, 7, gi] = -4. * l1
                    snlz[iface, 8, gi] = -4. * l2
                    snlz[iface, 9, gi] = 4. - 4. * l2 - 8. * l3 - 4. * l1
        elif snloc == 3:  # linear element
            # if sngi != 3:
            #     raise Exception('sngi ', sngi, 'is not compatible with snloc ', snloc)
            pnts, sweight, alignment = _gi_pnts_tri(sngi)
            sf_nd_nb.set_data(gi_align=torch.tensor(alignment,
                                                    device=dev,
                                                    dtype=torch.int64).view(ndim, sngi))
            SL = np.zeros((nface, sngi, 4), dtype=np.float64)
            # face1  triangle 2-1-3, l1 = 0
            SL[0, :, 0] = 0
            SL[0, :, 1] = pnts[:, 1]
            SL[0, :, 2] = pnts[:, 0]
            SL[0, :, 3] = pnts[:, 2]
            # face2  triangle 0-2-3, l2 = 0
            SL[1, :, 0] = pnts[:, 0]
            SL[1, :, 1] = 0
            SL[1, :, 2] = pnts[:, 1]
            SL[1, :, 3] = pnts[:, 2]
            # face3  triangle 1-0-3, l3 = 0
            SL[2, :, 0] = pnts[:, 1]
            SL[2, :, 1] = pnts[:, 0]
            SL[2, :, 2] = 0
            SL[2, :, 3] = pnts[:, 2]
            # face4  triangle 0-1-2, l4 = 0
            SL[3, :, 0] = pnts[:, 0]
            SL[3, :, 1] = pnts[:, 1]
            SL[3, :, 2] = pnts[:, 2]
            SL[3, :, 3] = 0

            sn = np.zeros((nface, nloc, sngi))
            snlx = np.zeros((nface, nloc, sngi))
            snly = np.zeros((nface, nloc, sngi))
            snlz = np.zeros((nface, nloc, sngi))
            for iface in range(nface):
                for gi in range(sngi):
                    l1 = SL[iface, gi, 0]
                    l2 = SL[iface, gi, 1]
                    l3 = SL[iface, gi, 2]
                    l4 = SL[iface, gi, 3]
                    # cornor nodes
                    sn[iface, 0, gi] = l1
                    sn[iface, 1, gi] = l2
                    sn[iface, 2, gi] = l3
                    sn[iface, 3, gi] = l4
                    # x-derivative
                    snlx[iface, 0, gi] = 1.
                    snlx[iface, 1, gi] = 0
                    snlx[iface, 2, gi] = 0
                    snlx[iface, 3, gi] = -1.
                    # y-derivative
                    snly[iface, 0, gi] = 0
                    snly[iface, 1, gi] = 1.
                    snly[iface, 2, gi] = 0
                    snly[iface, 3, gi] = -1.
                    # z-derivative
                    snlz[iface, 0, gi] = 0
                    snlz[iface, 1, gi] = 0
                    snlz[iface, 2, gi] = 1.
                    snlz[iface, 3, gi] = -1.
        else:  # snloc
            raise Exception('snloc %d is not accpted in 3D' % snloc)
        nlx_all = np.stack([nlx, nly, nlz], axis=0)
        snlx_all = np.stack([snlx, snly, snlz], axis=1)
        return n, nlx_all, weight, sn, snlx_all, sweight
    # ================== FROM here on, its 2D shape functions.==============================
    l1=np.zeros(ngi)
    l2=np.zeros(ngi)
    l3=np.zeros(ngi)
    weight=np.zeros(ngi)

    nface = ndim + 1
    sl1 = np.zeros((nface, sngi))
    sl2 = np.zeros((nface, sngi))
    sl3 = np.zeros((nface, sngi))
    sweight = np.zeros(sngi)

    if nloc==10:  # cubic elements
        alpha = -0.149570044467682
        beta = 0.333333333333333
        alpha1 = 0.175615257433208
        beta1 = 0.479308067841920
        gamma1 = 0.260345966079040
        alpha2 = 0.053347235608838
        beta2 = 0.869739794195568
        gamma2 = 0.065130102902216
        alpha3 = 0.077113760890257
        beta3 = 0.048690315425316
        gamma3 = 0.312865496004874
        gamma4 = 0.638444188569810
        # ! get wild
        weight[0] = alpha;   l1[0] = beta ;  l2[0] = beta;     l3[0] = beta
        weight[1] = alpha1;  l1[1] = beta1;  l2[1] = gamma1;   l3[1] = gamma1
        weight[2] = alpha1;  l1[2] = gamma1; l2[2] = beta1;    l3[2] = gamma1 
        weight[3] = alpha1;  l1[3] = gamma1; l2[3] = gamma1;   l3[3] = beta1 
        weight[4] = alpha2;  l1[4] = beta2;  l2[4] = gamma2;   l3[4] = gamma2 
        weight[5] = alpha2;  l1[5] = gamma2; l2[5] = beta2;    l3[5] = gamma2 
        weight[6] = alpha2;  l1[6] = gamma2; l2[6] = gamma2;   l3[6] = beta2 
        weight[7] = alpha3;  l1[7] = beta3;  l2[7] = gamma3;   l3[7] = gamma4 
        weight[8] = alpha3;  l1[8] = beta3;  l2[8] = gamma4;   l3[8] = gamma3
        weight[9] = alpha3;  l1[9]= gamma3;  l2[9]= gamma4;    l3[9]= beta3 
        weight[10] = alpha3; l1[10]= gamma3; l2[10]= beta3;    l3[10]= gamma4 
        weight[11] = alpha3; l1[11]= gamma4; l2[11]= beta3;    l3[11]= gamma3 
        weight[12] = alpha3; l1[12]= gamma4; l2[12]= gamma3;   l3[12]= beta3
        # print('sum of weights', np.sum(weight))
    elif nloc==3:  # linear elements
        weight[:] = 1./3.
        l1[0] = 0.5;    l2[0] = 0.5;    l3[0] = 0
        l1[1] = 0;      l2[1] = 0.5;    l3[1] = 0.5
        l1[2] = 0.5;    l2[2] = 0;      l3[2] = 0.5

    weight = weight*0.5

    if snloc == 4:  # cubic element
        ## 4pnt gaussian quadrature in 1D
        a = 0.339981043584856
        b = 0.861136311594053
        w1 = 0.652145154862546
        w2 = 0.347854845137454
        ## transfer to [0,1]
        a1 = 0.5 + 0.5*a 
        a2 = 0.5 - 0.5*a
        b1 = 0.5 + 0.5*b 
        b2 = 0.5 - 0.5*b
        # 
        sweight[0] = w2
        sweight[1] = w1
        sweight[2] = w1
        sweight[3] = w2
        alignment = [
            [0, 1, 2, 3],
            [3, 2, 1, 0],
        ]
        sf_nd_nb.set_data(gi_align=torch.tensor(alignment,
                                                device=dev,
                                                dtype=torch.int64).view(ndim, sngi))
        # face 1
        sl1[0,:] = np.asarray([b2,a2,a1,b1])
        sl2[0,:] = 1-sl1[0,:]
        sl3[0,:] = 0.
        # face 2
        sl1[1,:] = 0.
        sl2[1,:] = np.asarray([b2,a2,a1,b1])
        sl3[1,:] = 1-sl2[1,:]
        # face 3 
        sl2[2,:] = 0.
        sl3[2,:] = np.asarray([b2,a2,a1,b1])
        sl1[2,:] = 1-sl3[2,:]
    elif snloc == 2:  # linear element
        # 2 pnt gaussian quadrature in 1D
        a = 0.5 + 0.5/np.sqrt(3.)
        b = 0.5 - 0.5/np.sqrt(3.)
        sweight = 1.
        alignment = [
            [0, 1],
            [1, 0],
        ]
        sf_nd_nb.set_data(gi_align=torch.tensor(alignment,
                                                device=dev,
                                                dtype=torch.int64).view(ndim, sngi))
        # face 1
        sl1[0, :] = np.asarray([b, a])
        sl2[0, :] = 1 - sl1[0, :]
        sl3[0, :] = 0.
        # face 2
        sl1[1, :] = 0.
        sl2[1, :] = np.asarray([b, a])
        sl3[1, :] = 1 - sl2[1, :]
        # face 3
        sl2[2, :] = 0.
        sl3[2, :] = np.asarray([b, a])
        sl1[2, :] = 1 - sl3[2, :]
    
    sweight = sweight/2.

    n = np.zeros((nloc,ngi))
    nlx = np.zeros((nloc,ngi))
    nly = np.zeros((nloc,ngi))
    if (nloc==10) :
        for gi in range(ngi):
            # corner nodes...
            n[ 0, gi ] = 0.5*( 3. * l1[ gi ] - 1. ) * (3. * l1[ gi ]   -2.) *  l1[ gi ]
            n[ 1, gi ] = 0.5*( 3. * l2[ gi ] - 1. ) * (3. * l2[ gi ]   -2.) *  l2[ gi ]
            n[ 2, gi ] = 0.5*( 3. * l3[ gi ] - 1. ) * (3. * l3[ gi ]   -2.) *  l3[ gi ]
            # mid side nodes...
            n[ 3, gi ] = (9./2.)*l1[ gi ]*l2[ gi ]*( 3. * l1[ gi ] - 1. )
            n[ 4, gi ] = (9./2.)*l2[ gi ]*l1[ gi ]*( 3. * l2[ gi ] - 1. )

            n[ 5, gi ] = (9./2.)*l2[ gi ]*l3[ gi ]*( 3. * l2[ gi ] - 1. )
            n[ 6, gi ] = (9./2.)*l3[ gi ]*l2[ gi ]*( 3. * l3[ gi ] - 1. )

            n[ 7, gi ] = (9./2.)*l3[ gi ]*l1[ gi ]*( 3. * l3[ gi ] - 1. )
            n[ 8, gi ] = (9./2.)*l1[ gi ]*l3[ gi ]*( 3. * l1[ gi ] - 1. )
            # central node...
            n[ 9, gi ] = 27.*l1[ gi ]*l2[ gi ]*l3[ gi ]

            # x-derivative (nb. l1 + l2 + l3  = 1 )
            # corner nodes...
            nlx[ 0, gi ] = 0.5*( 27. * l1[ gi ]**2  - 18. *  l1[ gi ] + 2. )
            nlx[ 1, gi ] = 0.0
            nlx[ 2, gi ] = 0.5*( 27. * l3[ gi ]**2  - 18. *  l3[ gi ] + 2. )   *  (-1.0)
            # mid side nodes...
            nlx[ 3, gi ] = (9./2.)*(6.*l1[ gi ]*l2[ gi ]  - l2[ gi ] )
            nlx[ 4, gi ] = (9./2.)*l2[ gi ]*( 3. * l2[ gi ] - 1. )

            nlx[ 5, gi ] = - (9./2.)*l2[ gi ]*( 3. * l2[ gi ] - 1. )
            nlx[ 6, gi ] = (9./2.)*(   -l2[gi]*( 6.*l3[gi] -1. )    )

            nlx[ 7, gi ] = (9./2.)*( l1[ gi ]*(-6.*l3[gi]+1.) + l3[gi]*(3.*l3[gi]-1.)  )
            nlx[ 8, gi ] = (9./2.)*(  l3[gi]*(6.*l1[gi]-1.) -l1[gi]*(3.*l1[gi]-1.)  )
            # central node...
            nlx[ 9, gi ] = 27.*l2[ gi ]*( 1. - 2.*l1[gi]  - l2[ gi ] )

            # y-derivative (nb. l1 + l2 + l3  = 1 )
            # corner nodes...
            nly[ 0, gi ] = 0.0
            nly[ 1, gi ] = 0.5*( 27. * l2[ gi ]**2  - 18. *  l2[ gi ] + 2.  )
            nly[ 2, gi ] = 0.5*( 27. * l3[ gi ]**2  - 18. *  l3[ gi ] + 2.  )   *  (-1.0)
            # mid side nodes...
            nly[ 3, gi ] = (9./2.)*l1[ gi ]*( 3. * l1[ gi ] - 1. )
            nly[ 4, gi ] = (9./2.)*l1[ gi ]*( 6. * l2[ gi ] - 1. )

            nly[ 5, gi ] = (9./2.)*( l3[ gi ]*( 6. * l2[ gi ] - 1. ) -l2[gi]*( 3.*l2[gi]-1. )  )
            nly[ 6, gi ] = (9./2.)*( -l2[ gi ]*( 6. * l3[ gi ] - 1. ) +l3[gi]*(3.*l3[gi]-1.)  )

            nly[ 7, gi ] = -(9./2.)*l1[ gi ]*( 6. * l3[ gi ] - 1. )
            nly[ 8, gi ] = -(9./2.)*l1[ gi ]*( 3. * l1[ gi ] - 1. )
            # central node...
            nly[ 9, gi ] = 27.*l1[ gi ]*( 1. - 2.*l2[gi]  - l1[ gi ] )
    elif nloc == 3:  # linear element
        for gi in range(ngi):
            n[0, gi] = l1[gi]
            n[1, gi] = l2[gi]
            n[2, gi] = l3[gi]
            # x derivative
            nlx[0, gi] = 1.0
            nlx[1, gi] = 0.0
            nlx[2, gi] = -1.0
            # y derivative
            nly[0, gi] = 0.0
            nly[1, gi] = 1.0
            nly[2, gi] = -1.0

    nlx_all=np.stack([nlx,nly],axis=0)

    ## shape function at surface gaussian quadrature points
    sn = np.zeros((nface, nloc, sngi))
    snlx = np.zeros((nface, nloc, sngi))
    snly = np.zeros((nface, nloc, sngi))
    if (snloc==4) :
        for iface in range(nface):
            for gi in range(sngi):
                # corner nodes...
                sn[iface, 0, gi ] = 0.5*( 3. * sl1[iface, gi ] - 1. ) * (3. * sl1[iface, gi ]   -2.) *  sl1[iface, gi ]
                sn[iface, 1, gi ] = 0.5*( 3. * sl2[iface, gi ] - 1. ) * (3. * sl2[iface, gi ]   -2.) *  sl2[iface, gi ]
                sn[iface, 2, gi ] = 0.5*( 3. * sl3[iface, gi ] - 1. ) * (3. * sl3[iface, gi ]   -2.) *  sl3[iface, gi ]
                # mid side nodes...
                sn[iface, 3, gi ] = (9./2.)*sl1[iface, gi ]*sl2[iface, gi ]*( 3. * sl1[iface, gi ] - 1. )
                sn[iface, 4, gi ] = (9./2.)*sl2[iface, gi ]*sl1[iface, gi ]*( 3. * sl2[iface, gi ] - 1. )

                sn[iface, 5, gi ] = (9./2.)*sl2[iface, gi ]*sl3[iface, gi ]*( 3. * sl2[iface, gi ] - 1. )
                sn[iface, 6, gi ] = (9./2.)*sl3[iface, gi ]*sl2[iface, gi ]*( 3. * sl3[iface, gi ] - 1. )

                sn[iface, 7, gi ] = (9./2.)*sl3[iface, gi ]*sl1[iface, gi ]*( 3. * sl3[iface, gi ] - 1. )
                sn[iface, 8, gi ] = (9./2.)*sl1[iface, gi ]*sl3[iface, gi ]*( 3. * sl1[iface, gi ] - 1. )
                # central node...
                sn[iface, 9, gi ] = 27.*sl1[iface, gi ]*sl2[iface, gi ]*sl3[iface, gi ]

                # x-derivative (nb. sl1 + sl2 + sl3  = 1 )
                # corner nodes...
                snlx[iface, 0, gi ] = 0.5*( 27. * sl1[iface, gi ]**2  - 18. *  sl1[iface, gi ] + 2. )
                snlx[iface, 1, gi ] = 0.0
                snlx[iface, 2, gi ] = 0.5*( 27. * sl3[iface, gi ]**2  - 18. *  sl3[iface, gi ] + 2. )   *  (-1.0)
                # mid side nodes...
                snlx[iface, 3, gi ] = (9./2.)*(6.*sl1[iface, gi ]*sl2[iface, gi ]  - sl2[iface, gi ] )
                snlx[iface, 4, gi ] = (9./2.)*sl2[iface, gi ]*( 3. * sl2[iface, gi ] - 1. )

                snlx[iface, 5, gi ] = - (9./2.)*sl2[iface, gi ]*( 3. * sl2[iface, gi ] - 1. )
                snlx[iface, 6, gi ] = (9./2.)*(   -sl2[iface,gi]*( 6.*sl3[iface,gi] -1. )    )

                snlx[iface, 7, gi ] = (9./2.)*( sl1[iface, gi ]*(-6.*sl3[iface,gi]+1.) + sl3[iface,gi]*(3.*sl3[iface,gi]-1.)  )
                snlx[iface, 8, gi ] = (9./2.)*(  sl3[iface,gi]*(6.*sl1[iface,gi]-1.) -sl1[iface,gi]*(3.*sl1[iface,gi]-1.)  )
                # central node...
                snlx[iface, 9, gi ] = 27.*sl2[iface, gi ]*( 1. - 2.*sl1[iface,gi]  - sl2[iface, gi ] )

                # y-derivative (nb. sl1 + sl2 + sl3  = 1 )
                # corner nodes...
                snly[iface, 0, gi ] = 0.0
                snly[iface, 1, gi ] = 0.5*( 27. * sl2[iface, gi ]**2  - 18. *  sl2[iface, gi ] + 2.  )
                snly[iface, 2, gi ] = 0.5*( 27. * sl3[iface, gi ]**2  - 18. *  sl3[iface, gi ] + 2.  )   *  (-1.0)
                # mid side nodes...
                snly[iface, 3, gi ] = (9./2.)*sl1[iface, gi ]*( 3. * sl1[iface, gi ] - 1. )
                snly[iface, 4, gi ] = (9./2.)*sl1[iface, gi ]*( 6. * sl2[iface, gi ] - 1. )

                snly[iface, 5, gi ] = (9./2.)*( sl3[iface, gi ]*( 6. * sl2[iface, gi ] - 1. ) -sl2[iface,gi]*( 3.*sl2[iface,gi]-1. )  )
                snly[iface, 6, gi ] = (9./2.)*( -sl2[iface, gi ]*( 6. * sl3[iface, gi ] - 1. ) +sl3[iface,gi]*(3.*sl3[iface,gi]-1.)  )

                snly[iface, 7, gi ] = -(9./2.)*sl1[iface, gi ]*( 6. * sl3[iface, gi ] - 1. )
                snly[iface, 8, gi ] = -(9./2.)*sl1[iface, gi ]*( 3. * sl1[iface, gi ] - 1. )
                # central node...
                snly[iface, 9, gi ] = 27.*sl1[iface, gi ]*( 1. - 2.*sl2[iface,gi]  - sl1[iface, gi ] )
    elif nloc == 3:  # linear element
        for iface in range(nface):
            for gi in range(sngi):
                sn[iface, 0, gi] = sl1[iface, gi]
                sn[iface, 1, gi] = sl2[iface, gi]
                sn[iface, 2, gi] = sl3[iface, gi]
                # x-derivative
                snlx[iface, 0, gi] = 1.0
                snlx[iface, 1, gi] = 0.0
                snlx[iface, 2, gi] = -1.0
                # y-derivative
                snly[iface, 0, gi] = 0.0
                snly[iface, 1, gi] = 1.0
                snly[iface, 2, gi] = -1.0

    snlx_all=np.stack([snlx,snly],axis=1)

    return n, nlx_all, weight, sn, snlx_all, sweight


# local shape function
# can pass in multiple elements in a batch
# 
# input: 
# ~~n, shape function on a reference element, ~~
# nlx, derivatives of shape function on a ref. ele., 
#      provide this when create the det_nlx object
#      (ndim, nloc, ngi), numpy array, on cpu by default
#      will be moved to gpu if dev='gpu'
# ~~ngi, number of quadrature points~~
# ~~ndim, number of dimension~~
# x_loc, local nodes coordinates
#        provide this when call det_nlx.forward
#        (batch_size, ndim, nloc), torch tensor, on dev
# ~~1nloc, number of local nodes ~~
# weight, weights of quadrature points
#         provide this when call det_nlx.forward
#         (ngi), numpy array, on cpu by default
#         will be moved to gpu if dev='gpu'
#
# output: 
# nx, derivatives of shape functions on local element(s)
#     torch tensor (batch_in, ndim, nloc, ngi) on dev
# detwei, weights * determinant |J|, 
#         torch tensor (batch_in, ngi) on dev
class det_nlx(Module):
    """
    # local shape function
    can pass in multiple elements in a batch
    
    # input: 
    :~~n, shape function on a reference element, ~~  
    :nlx, derivatives of shape function on a ref. ele., 
         provide this when create the det_nlx object
         (ndim, nloc, ngi), numpy array, on cpu by default
         will be moved to gpu if dev='gpu'
    :~~ngi, number of quadrature points~~
    :~~ndim, number of dimension~~
    :x_loc, local nodes coordinates
           provide this when call det_nlx.forward
           (batch_size, ndim, nloc), torch tensor, on dev
    :~~nloc, number of local nodes ~~
    :weight, weights of quadrature points
            provide this when call det_nlx.forward
            (ngi), numpy array, on cpu by default
            will be moved to gpu if dev='gpu'
    
    # output: 
    :nx, derivatives of shape functions on local element(s)
        torch tensor (batch_in, ndim, nloc, ngi) on dev
    :detwei, weights * determinant |J|, 
            torch tensor (batch_in, ngi) on dev
    """
    def __init__(self, nlx, nloc=config.nloc, ngi=config.ngi):
        super(det_nlx, self).__init__()

        # calculate jacobian
        self.calc_j11 = Conv1d(in_channels=1, \
            out_channels=ngi, \
            kernel_size=nloc, \
            bias=False)
        self.calc_j12 = Conv1d(in_channels=1, \
            out_channels=ngi, \
            kernel_size=nloc, \
            bias=False)
        self.calc_j21 = Conv1d(in_channels=1, \
            out_channels=ngi, \
            kernel_size=nloc, \
            bias=False)
        self.calc_j22 = Conv1d(in_channels=1, \
            out_channels=ngi, \
            kernel_size=nloc, \
            bias=False)

        # stack jacobian to ngi* (ndim*ndim)
        # determinant of jacobian
        # no need to matrix multiplication
        # do it directly
        # self.calc_det = Conv1d(in_channels=ngi, \
        #     out_channels=ngi, \
        #     kernel_size=ndim*ndim,\
        #     bias=False)

        # inverse of jacobian
        # no need to matrix multiplication
        # do it directly
        # self.calc_invjac = Conv1d(in_channels=ngi, \
        #     out_channels=ndim*ndim*ngi, \
        #     kernel_size=ndim*ndim, \
        #     bias=False)

        # stack inverse jacobian to ngi* (ndim*ndim)
        # nx at local element
        # output: (batch_size, ngi, ndim*nloc)
        self.calc_nx = Conv1d(in_channels=ngi, \
            out_channels=ndim*ngi,
            kernel_size=ndim*ndim,\
            bias=False)
        
        self.nlx = nlx
        
    def forward(self, x_loc, weight, nloc=config.nloc, ngi=config.ngi):
        '''
        
        # input 
        
        x_loc - (batch_size , ndim, nloc), coordinate info of local nodes
            reference coordinate: (xi, eta)
            physical coordinate: (x, y)
        weight  -        np array (ngi)

        # output
        nx - shape function derivatives Nix & Niy, 
            (batch_size, ndim, nloc, ngi)
        detwei - determinant times GI weight, (batch_size, ngi)
        '''

        batch_in = x_loc.shape[0]
        # print(x_loc.is_cuda)
        x = x_loc[:,0,:].view(batch_in,1,nloc)
        y = x_loc[:,1,:].view(batch_in,1,nloc)
        # print('x',x,'\ny',y)
        # print(torch.cuda.memory_summary())
        # first we calculate jacobian matrix (J^T) = [j11,j12;
        #                                             j21,j22]
        # [ d x/d xi,   dy/d xi ;
        #   d x/d eta,  dy/d eta]
        # output: each component of jacobi
        # (batch_size , ngi)
        j11 = self.calc_j11(x).view(batch_in, ngi)
        j12 = self.calc_j12(y).view(batch_in, ngi)
        j21 = self.calc_j21(x).view(batch_in, ngi)
        j22 = self.calc_j22(y).view(batch_in, ngi)
        # print('j11', j11)
        # print('j12', j12)
        # print('j21', j21)
        # print('j22', j22)
        # print(torch.cuda.memory_summary())
        # calculate determinant of jacobian
        det = torch.mul(j11,j22)-torch.mul(j21,j12)
        det = det.view(batch_in, ngi)
        invdet = torch.div(1.0,det)
        det = abs( det )
        # print('det', det)
        # print('invdet', invdet)
        det = torch.mul(det, torch.tensor(weight, device=dev).unsqueeze(0).expand(det.shape[0],ngi)) # detwei
        del j11, j12, j21, j22
        ####### 
        # calculate and store inv jacobian...
        # inverse of jacobian
        # print(torch.cuda.memory_summary())
        # calculate nx
        # input: invjac (batch_size, ngi, ndim*ndim)
        # output: nx (batch_size, ngi, ndim, nloc)
        # nx = self.calc_nx(invjac)
        nlx1 = self.nlx[0,:,:].expand(batch_in,-1,-1)
        nlx2 = self.nlx[1,:,:].expand(batch_in,-1,-1)
        j12 = self.calc_j12(y).view(batch_in, ngi)
        j22 = self.calc_j22(y).view(batch_in, ngi)
        invj11 = torch.mul(j22,invdet).view(batch_in,-1)
        invj12 = torch.mul(j12,invdet).view(batch_in,-1)*(-1.0)
        del j22 
        del j12
        invj11 = invj11.unsqueeze(1).expand(-1,nloc,-1)
        invj12 = invj12.unsqueeze(1).expand(-1,nloc,-1)
        # print('invj11', invj11)
        # print('invj12', invj12)
        nx1 = torch.mul(invj11, nlx1) \
            + torch.mul(invj12, nlx2)
        del invj11 
        del invj12 

        # print('nlx1', nlx1)
        # print('nlx2', nlx2)
        j21 = self.calc_j21(x).view(batch_in, ngi)
        j11 = self.calc_j11(x).view(batch_in, ngi)
        invj21 = torch.mul(j21,invdet).view(batch_in,-1)*(-1.0)
        invj22 = torch.mul(j11,invdet).view(batch_in,-1)
        del j21
        del j11
        invj21 = invj21.unsqueeze(1).expand(-1,nloc,-1)
        invj22 = invj22.unsqueeze(1).expand(-1,nloc,-1)
        del invdet 
        # print('invj21', invj21)
        # print('invj22', invj22)
        # print('invj11expand', invj22)
        # print(invj11.shape, nlx1.shape)
        nx2 = torch.mul(invj21, nlx1) \
            + torch.mul(invj22, nlx2)
        del invj21 
        del invj22 

        #######
        # do not store inv jacobian but calculate on the fly!
        # calculate nx
        # print(torch.cuda.memory_summary())
        # nlx1 = torch.tensor(np.transpose(nlx[0,:,:]), device=dev)
        # nlx1 = nlx1.expand(batch_in,ngi,nloc)
        # nlx2 = torch.tensor(np.transpose(nlx[1,:,:]), device=dev)
        # nlx2 = nlx2.expand(batch_in,ngi,nloc)
        # nx1 = torch.mul(torch.mul(j11,invdet).view(batch_in,-1).unsqueeze(-1).expand(batch_in,ngi,nloc), nlx1) \
        #     - torch.mul(torch.mul(j21,invdet).view(batch_in,-1).unsqueeze(-1).expand(batch_in,ngi,nloc), nlx2)
        # nx2 =-torch.mul(torch.mul(j12,invdet).view(batch_in,-1).unsqueeze(-1).expand(batch_in,ngi,nloc), nlx1) \
        #     + torch.mul(torch.mul(j22,invdet).view(batch_in,-1).unsqueeze(-1).expand(batch_in,ngi,nloc), nlx2)
        # # print('nx1', nx1)
        nx = torch.stack((nx1,nx2),dim=1)
        # print(torch.cuda.memory_summary())
        return nx, det

# local shape function at surface(s)
# can pass in multiple elements in a batch
# 
def sdet_snlx(snlx, x_loc, sweight, nloc=config.nloc, sngi=config.sngi):
    """
    # local shape function on element face
    can pass in multiple elements in a batch
    
    # input: 
    
    :~~nlx, derivatives of shape function on a ref. ele., 
         (ndim, nloc, sngi), numpy array, on cpu by default
         will be moved to gpu if dev='gpu'~~
    
    :snlx, derivatives of shape function on a ref. ele., 
        at surface quadratures
        (nface, ndim, nloc, sngi), numpy array, on cpu by default
        will be moved to gpu if dev='gpu'

    :x_loc, local nodes coordinates
        (batch_in, ndim, nloc), torch tensor, on dev

    :sweight, weights of surface quadrature points
            provide this when call det_nlx.forward
            (sngi), numpy array, on cpu by default
            will be moved to gpu if dev='gpu'
    
    # output: 

    :snx, derivatives of shape functions on local element(s)
        torch tensor (batch_in, nface, ndim, nloc, sngi) on dev
    
    :sdetwei, weights * determinant |J|, 
            torch tensor (batch_in, nface, sngi) on dev
    """

    nface=config.nface
    # input : x_loc
    # (batch_size , ndim, nloc), coordinate info of local nodes
    # reference coordinate: (xi, eta)
    # physical coordinate: (x, y)
    # input : weight
    # np array (ngi)
    # print('x_loc size', x_loc.shape)
    # print('x size', x_loc[:,0,:].shape)
    batch_in = x_loc.shape[0]
    # print(x_loc.is_cuda)
    x = x_loc[:,0,:].view(batch_in,1,nloc)
    y = x_loc[:,1,:].view(batch_in,1,nloc)

    snlx = torch.tensor(snlx, device=dev)
    # print('x',x,'\ny',y)
    # print(torch.cuda.memory_summary())
    # first we calculate jacobian matrix (J^T) = [j11,j12;
    #                                             j21,j22]
    # [ d x/d xi,   dy/d xi ;
    #   d x/d eta,  dy/d eta]
    # output: each component of jacobi
    # (nface, sngi, batch_in)
    j11 = torch.tensordot(snlx[:,0,:,:], x, dims=([1],[2])).view(nface, sngi, batch_in)
    j12 = torch.tensordot(snlx[:,0,:,:], y, dims=([1],[2])).view(nface, sngi, batch_in)
    j21 = torch.tensordot(snlx[:,1,:,:], x, dims=([1],[2])).view(nface, sngi, batch_in)
    j22 = torch.tensordot(snlx[:,1,:,:], y, dims=([1],[2])).view(nface, sngi, batch_in)
    
    # print('j11', j11)
    # print('j12', j12)
    # print('j21', j21)
    # print('j22', j22)
    # print(torch.cuda.memory_summary())
    # calculate determinant of jacobian
    # (nface, sngi, batch_in)
    det = torch.mul(j11,j22)-torch.mul(j21,j12)
    invdet = torch.div(1.0,det)

    # print('det', det)
    # print('invdet', invdet)
    del det # this is the final use of volume det
    
    # del j11, j12, j21, j22
    ####### 
    # calculate and store inv jacobian...
    # inverse of jacobian
    # print(torch.cuda.memory_summary())

    invj11 = torch.mul(j22,invdet)
    invj12 = torch.mul(j12,invdet)*(-1.0)
    del j22
    del j12
    # operands
    # invj11 (nface, sngi, batch_in)
    # snlx (nface, ndim, nloc, sngi)
    # result
    # snx1 (nface, nloc, sngi, batch_in) # will stack & transpose dimensions later
    snx1 = torch.mul(invj11.unsqueeze(1).expand(nface,nloc,sngi,batch_in), \
        snlx[:,0,:,:].unsqueeze(-1).expand(nface,nloc,sngi,batch_in)) \
        + torch.mul(invj12.unsqueeze(1).expand(nface,nloc,sngi,batch_in), \
        snlx[:,1,:,:].unsqueeze(-1).expand(nface,nloc,sngi,batch_in)) 
    # print('invj11', invj11)
    # print('invj12', invj12)
    del invj11 
    del invj12 

    invj21 = torch.mul(j21,invdet)*(-1.0)
    invj22 = torch.mul(j11,invdet)
    del j21
    del j11 
    del invdet 
    snx2 = torch.mul(invj21.unsqueeze(1).expand(nface,nloc,sngi,batch_in), \
        snlx[:,0,:,:].unsqueeze(-1).expand(nface,nloc,sngi,batch_in)) \
        + torch.mul(invj22.unsqueeze(1).expand(nface,nloc,sngi,batch_in), \
        snlx[:,1,:,:].unsqueeze(-1).expand(nface,nloc,sngi,batch_in)) 
    # print('invj21', invj21)
    # print('invj22', invj22)
    del invj21 
    del invj22 

    snx = torch.stack((snx1,snx2),dim=1)
    
    # now we calculate surface det
    # IMPORTANT: we are assuming straight edges
    sdet = torch.zeros(batch_in,nface,device=dev, dtype=torch.float64)
    sdet[:,0] = torch.linalg.vector_norm(x_loc[:,:,0]-x_loc[:,:,1], dim=1) # # face 0, local node 0 and 1
    sdet[:,1] = torch.linalg.vector_norm(x_loc[:,:,1]-x_loc[:,:,2], dim=1) # # face 1, local node 1 and 2
    sdet[:,2] = torch.linalg.vector_norm(x_loc[:,:,2]-x_loc[:,:,0], dim=1) # # face 2, local node 2 and 0
    # print(x_loc)
    # print(x_loc[:,:,0]-x_loc[:,:,1])
    # print(torch.linalg.vector_norm(x_loc[:,:,0]-x_loc[:,:,1], dim=1))
    # print(sdet)
    
    # # face 1, local node 1 and 2
    # sdetwei
    sdetwei = torch.mul(sdet.unsqueeze(-1).expand(batch_in,nface,sngi), \
        torch.tensor(sweight, device=dev).unsqueeze(0).unsqueeze(1).expand(batch_in,nface,sngi)) # sdetwei

    # surface normal
    snormal = torch.zeros(batch_in, nface, ndim, device=dev, dtype=torch.float64)
    # face 0 
    iface = 0
    idim = 0; snormal[:,iface,idim] = y[:,0,1] - y[:,0,0]
    idim = 1; snormal[:,iface,idim] = x[:,0,0] - x[:,0,1]
    # face 1
    iface = 1
    idim = 0; snormal[:,iface,idim] = y[:,0,2] - y[:,0,1]
    idim = 1; snormal[:,iface,idim] = x[:,0,1] - x[:,0,2]
    # face 3
    iface = 2
    idim = 0; snormal[:,iface,idim] = y[:,0,0] - y[:,0,2]
    idim = 1; snormal[:,iface,idim] = x[:,0,2] - x[:,0,0]
    # normalise
    snormal = snormal/sdet.unsqueeze(-1).expand(batch_in,nface,ndim)

    ## permute dimensions
    snx = torch.permute(snx, (4,0,1,2,3)) # (batch_in, nface, ndim, nloc, sngi)
    snx = snx.contiguous() # change memory storage so that we can view it in the future.
    
    # print(snx.shape, sdetwei.shape)

    return snx, sdetwei, snormal


def get_det_nlx(nlx, x_loc, weight, nloc=config.nloc, ngi=config.ngi):
    '''
    take in element nodes coordinates, spit out
    detwei and local shape function derivative

    # Input
    nlx : torch tensor (ndim, nloc, ngi)
        shape function derivative on reference element
    x_loc : torch tensor (batch_in, ndim, nloc)
        nodes coordinates
    weight : torch tensor (ngi,)
        quadrature points weight

    # Output
    nx : torch tensor (batch_in, ndim, nloc, ngi)
        local shpae function derivatives
    detwei : torch tensor (batch_in, ngi)
        determinant x quadrature weight
    '''
    batch_in = x_loc.shape[0]
    # print(x_loc.is_cuda)
    x = x_loc[:, 0, :].view(batch_in, nloc)
    y = x_loc[:, 1, :].view(batch_in, nloc)
    # print('x',x,'\ny',y)
    # print(torch.cuda.memory_summary())
    # first we calculate jacobian matrix (J^T) = [j11,j12;
    #                                             j21,j22]
    # [ d x/d xi,   dy/d xi ;
    #   d x/d eta,  dy/d eta]
    # output: each component of jacobi
    # (batch_size , ngi)
    j11 = torch.einsum('ij,ki->kj', nlx[0, :, :], x).view(batch_in, ngi)
    j12 = torch.einsum('ij,ki->kj', nlx[0, :, :], y).view(batch_in, ngi)
    j21 = torch.einsum('ij,ki->kj', nlx[1, :, :], x).view(batch_in, ngi)
    j22 = torch.einsum('ij,ki->kj', nlx[1, :, :], y).view(batch_in, ngi)
    # j11 = torch.tensordot(nlx[0,:,:], x, dims=([0],[2])).view(ngi, batch_in)
    # j12 = torch.tensordot(nlx[0,:,:], y, dims=([0],[2])).view(ngi, batch_in)
    # j21 = torch.tensordot(nlx[1,:,:], x, dims=([0],[2])).view(ngi, batch_in)
    # j22 = torch.tensordot(nlx[1,:,:], y, dims=([0],[2])).view(ngi, batch_in)
    # calculate determinant of jacobian
    det = torch.mul(j11, j22) - torch.mul(j21, j12)
    det = det.view(batch_in, ngi)
    invdet = torch.div(1.0, det)
    det = abs(det)
    # print('det', det)
    # print('invdet', invdet)
    detwei = torch.mul(det, torch.tensor(weight, device=dev).unsqueeze(0).expand(det.shape[0], ngi))  # detwei
    del det
    #######
    # calculate and store inv jacobian...
    # inverse of jacobian
    # print(torch.cuda.memory_summary())
    # calculate nx
    nlx1 = nlx[0, :, :].expand(batch_in, -1, -1)
    nlx2 = nlx[1, :, :].expand(batch_in, -1, -1)
    invj11 = torch.mul(j22, invdet).view(batch_in, -1)
    invj12 = torch.mul(j12, invdet).view(batch_in, -1) * (-1.0)
    del j22
    del j12
    invj11 = invj11.unsqueeze(1).expand(-1, nloc, -1)
    invj12 = invj12.unsqueeze(1).expand(-1, nloc, -1)
    # print('invj11', invj11)
    # print('invj12', invj12)
    nx1 = torch.mul(invj11, nlx1) + torch.mul(invj12, nlx2)
    del invj11
    del invj12

    invj21 = torch.mul(j21, invdet).view(batch_in, -1) * (-1.0)
    invj22 = torch.mul(j11, invdet).view(batch_in, -1)
    del j21
    del j11
    invj21 = invj21.unsqueeze(1).expand(-1, nloc, -1)
    invj22 = invj22.unsqueeze(1).expand(-1, nloc, -1)
    del invdet
    # print('invj21', invj21)
    # print('invj22', invj22)
    # print('invj11expand', invj22)
    # print(invj11.shape, nlx1.shape)
    nx2 = torch.mul(invj21, nlx1) + torch.mul(invj22, nlx2)
    del invj21
    del invj22

    nx = torch.stack((nx1, nx2), dim=1)

    return nx, detwei


def get_det_nlx_3d(nlx, x_loc, weight, nloc=config.nloc, ngi=config.ngi):
    """
    take in element nodes coordinates, spit out
    detwei and local shape function derivative

    # Input
    nlx : torch tensor (ndim, nloc, ngi)
        shape function derivative on reference element
    x_loc : torch tensor (batch_in, ndim, nloc)
        nodes coordinates
    weight : torch tensor (ngi,)
        quadrature points weight

    # Output
    nx : torch tensor (batch_in, ndim, nloc, ngi)
        local shpae function derivatives
    detwei : torch tensor (batch_in, ngi)
        determinant x quadrature weight
    """
    batch_in = x_loc.shape[0]
    # x = x_loc[:, 0, :].view(batch_in, nloc)
    # y = x_loc[:, 1, :].view(batch_in, nloc)
    # z = x_loc[:, 2, :].view(batch_in, nloc)
    # first we calculate jacobian matrix (J^T) = [j11,j12,j13;
    #                                             j21,j22,j23;
    #                                             j31,j32,j33]
    # [ d x/d xi,   dy/d xi,    dz/d xi;
    #   d x/d eta,  dy/d eta,   dz/d eta;
    #   d x/d chi,  dy/d chi,   dz/d chi]
    # output: each component of jacobi
    # (batch_size , ngi)
    j = torch.zeros(batch_in, ngi, ndim, ndim, device=dev, dtype=torch.float64)
    for idim in range(ndim):
        for jdim in range(ndim):
            j[..., idim, jdim] = torch.einsum('ij,ki->kj',
                                              nlx[idim, :, :],
                                              x_loc[:, jdim, :]).view(batch_in, ngi)
    # j11 = torch.einsum('ij,ki->kj', nlx[0, :, :], x).view(batch_in, ngi)
    # j12 = torch.einsum('ij,ki->kj', nlx[0, :, :], y).view(batch_in, ngi)
    # j21 = torch.einsum('ij,ki->kj', nlx[1, :, :], x).view(batch_in, ngi)
    # j22 = torch.einsum('ij,ki->kj', nlx[1, :, :], y).view(batch_in, ngi)
    # calculate determinant of jacobian
    det = torch.det(j)
    # det = torch.mul(j11, j22) - torch.mul(j21, j12)
    # calculate inv jacobian
    invj = torch.linalg.inv(j)
    # calculate detwei
    det = det.view(batch_in, ngi)
    # invdet = torch.div(1.0, det)
    det = abs(det)
    detwei = torch.mul(det, weight.unsqueeze(0).expand(det.shape[0], ngi))  # detwei
    del det
    # calculate nx
    nx = torch.zeros(batch_in, ndim, nloc, ngi, device=dev, dtype=torch.float64)
    for idim in range(ndim):
        nx[:, idim, :, :] = torch.einsum('bgj,jng->bng', invj[:, :, idim, :], nlx[:, :, :])
    # nlx1 = nlx[0, :, :].expand(batch_in, -1, -1)
    # nlx2 = nlx[1, :, :].expand(batch_in, -1, -1)
    # invj11 = torch.mul(j22, invdet).view(batch_in, -1)
    # invj12 = torch.mul(j12, invdet).view(batch_in, -1) * (-1.0)
    # del j22
    # del j12
    # invj11 = invj11.unsqueeze(1).expand(-1, nloc, -1)
    # invj12 = invj12.unsqueeze(1).expand(-1, nloc, -1)
    # # print('invj11', invj11)
    # # print('invj12', invj12)
    # nx1 = torch.mul(invj11, nlx1) + torch.mul(invj12, nlx2)
    # del invj11
    # del invj12
    #
    # invj21 = torch.mul(j21, invdet).view(batch_in, -1) * (-1.0)
    # invj22 = torch.mul(j11, invdet).view(batch_in, -1)
    # del j21
    # del j11
    # invj21 = invj21.unsqueeze(1).expand(-1, nloc, -1)
    # invj22 = invj22.unsqueeze(1).expand(-1, nloc, -1)
    # del invdet
    # # print('invj21', invj21)
    # # print('invj22', invj22)
    # # print('invj11expand', invj22)
    # # print(invj11.shape, nlx1.shape)
    # nx2 = torch.mul(invj21, nlx1) + torch.mul(invj22, nlx2)
    # del invj21
    # del invj22

    # nx = torch.stack((nx1, nx2), dim=1)

    return nx, detwei


def sdet_snlx_3d(snlx, x_loc, sweight, nloc=config.nloc, sngi=config.sngi):
    """ TODO: not implemented!
    # local shape function on element face
    can pass in multiple elements in a batch

    # input:

    :~~nlx, derivatives of shape function on a ref. ele.,
         (ndim, nloc, sngi), numpy array, on cpu by default
         will be moved to gpu if dev='gpu'~~

    :snlx, derivatives of shape function on a ref. ele.,
        at surface quadratures
        (nface, ndim, nloc, sngi), numpy array, on cpu by default
        will be moved to gpu if dev='gpu'

    :x_loc, local nodes coordinates
        (batch_in, ndim, nloc), torch tensor, on dev

    :sweight, weights of surface quadrature points
            provide this when call det_nlx.forward
            (sngi), numpy array, on cpu by default
            will be moved to gpu if dev='gpu'

    # output:

    :snx, derivatives of shape functions on local element(s)
        torch tensor (batch_in, nface, ndim, nloc, sngi) on dev

    :sdetwei, weights * determinant |J|,
            torch tensor (batch_in, nface, sngi) on dev
    """

    nface = config.nface
    # input : x_loc
    # (batch_size , ndim, nloc), coordinate info of local nodes
    # reference coordinate: (xi, eta)
    # physical coordinate: (x, y)
    # input : weight
    # np array (ngi)
    # print('x_loc size', x_loc.shape)
    # print('x size', x_loc[:,0,:].shape)
    batch_in = x_loc.shape[0]
    # print(x_loc.is_cuda)
    # x = x_loc[:, 0, :].view(batch_in, 1, nloc)
    # y = x_loc[:, 1, :].view(batch_in, 1, nloc)

    # snlx = torch.tensor(snlx, device=dev)

    # first we calculate jacobian matrix (J^T) = [j11,j12,j13;
    #                                             j21,j22,j23;
    #                                             j31,j32,j33]
    # [ d x/d xi,   dy/d xi,    dz/d xi;
    #   d x/d eta,  dy/d eta,   dz/d eta;
    #   d x/d chi,  dy/d chi,   dz/d chi]
    # output: each component of jacobi
    # (nface, sngi, batch_in)
    j = torch.zeros(batch_in, nface, sngi, ndim, ndim, device=dev, dtype=torch.float64)
    for idim in range(ndim):
        for jdim in range(ndim):
            j[..., idim, jdim] = torch.einsum('fig,bi->bfg',  # f: nface, b: batch_in, g: sngi, i: inod
                                              snlx[:, idim, :, :],
                                              x_loc[:, jdim, :])
    # j11 = torch.tensordot(snlx[:, 0, :, :], x, dims=([1], [2])).view(nface, sngi, batch_in)
    # j12 = torch.tensordot(snlx[:, 0, :, :], y, dims=([1], [2])).view(nface, sngi, batch_in)
    # j21 = torch.tensordot(snlx[:, 1, :, :], x, dims=([1], [2])).view(nface, sngi, batch_in)
    # j22 = torch.tensordot(snlx[:, 1, :, :], y, dims=([1], [2])).view(nface, sngi, batch_in)

    # calculate determinant of jacobian
    # (nface, sngi, batch_in)
    # det = torch.det(j)
    # det = torch.mul(j11, j22) - torch.mul(j21, j12)
    # calculate inverse of jacobian
    invj = torch.linalg.inv(j)
    # invdet = torch.div(1.0, det)
    # del det  # this is the final use of volume det
    # calculate snx
    snx = torch.einsum('bfgij,fjng->bfing',  # b-batch_in, f-nface, g-sngi, i-ndim, j-ndim, n-nloc
                       invj,
                       snlx)

    # now we calculate surface det
    # IMPORTANT: we are assuming straight edges
    sdet = torch.zeros(batch_in, nface, device=dev, dtype=torch.float64)
    # face 0 node 2-1-3
    sdet[:, 0] = torch.linalg.vector_norm(torch.linalg.cross(
        x_loc[..., 1] - x_loc[..., 2], x_loc[..., 3] - x_loc[..., 2]
    ), dim=1)
    # face 1 node 0-2-3
    sdet[:, 1] = torch.linalg.vector_norm(torch.linalg.cross(
        x_loc[..., 2] - x_loc[..., 0], x_loc[..., 3] - x_loc[..., 0]
    ), dim=1)
    # face 2 node 1-0-3
    sdet[:, 2] = torch.linalg.vector_norm(torch.linalg.cross(
        x_loc[..., 0] - x_loc[..., 1], x_loc[..., 3] - x_loc[..., 1]
    ), dim=1)
    # face 3 node 0-1-2
    sdet[:, 3] = torch.linalg.vector_norm(torch.linalg.cross(
        x_loc[..., 1] - x_loc[..., 0], x_loc[..., 2] - x_loc[..., 0]
    ), dim=1)
    # sdet = torch.zeros(batch_in, nface, device=dev, dtype=torch.float64)
    # sdet[:, 0] = torch.linalg.vector_norm(x_loc[:, :, 0] - x_loc[:, :, 1], dim=1)  # # face 0, local node 0 and 1
    # sdet[:, 1] = torch.linalg.vector_norm(x_loc[:, :, 1] - x_loc[:, :, 2], dim=1)  # # face 1, local node 1 and 2
    # sdet[:, 2] = torch.linalg.vector_norm(x_loc[:, :, 2] - x_loc[:, :, 0], dim=1)  # # face 2, local node 2 and 0

    # # face 1, local node 1 and 2
    # sdetwei
    # sdetwei = torch.mul(sdet.unsqueeze(-1).expand(batch_in, nface, sngi), \
    #                     torch.tensor(sweight, device=dev).unsqueeze(0).unsqueeze(1).expand(batch_in, nface,
    #                                                                                        sngi))  # sdetwei
    sdetwei = torch.einsum('bf,g->bfg', sdet, sweight)
    # surface normal
    snormal = torch.zeros(batch_in, nface, ndim, device=dev, dtype=torch.float64)
    # face 0 node 2-1-3
    snormal[:, 0, :] = torch.linalg.cross(
        x_loc[..., 1] - x_loc[..., 2], x_loc[..., 3] - x_loc[..., 2]
    ) / sdet[:, 0].view(batch_in, 1)
    # face 1 node 0-2-3
    snormal[:, 1, :] = torch.linalg.cross(
        x_loc[..., 2] - x_loc[..., 0], x_loc[..., 3] - x_loc[..., 0]
    ) / sdet[:, 1].view(batch_in, 1)
    # face 2 node 1-0-3
    snormal[:, 2, :] = torch.linalg.cross(
        x_loc[..., 0] - x_loc[..., 1], x_loc[..., 3] - x_loc[..., 1]
    ) / sdet[:, 2].view(batch_in, 1)
    # face 3 node 0-1-2
    snormal[:, 3, :] = torch.linalg.cross(
        x_loc[..., 1] - x_loc[..., 0], x_loc[..., 2] - x_loc[..., 0]
    ) / sdet[:, 3].view(batch_in, 1)

    return snx, sdetwei, snormal
