# shape functions on a reference element


import numpy as np 
import config 
import torch
from torch.nn import Conv1d,Sequential,Module

nele = config.nele 
mesh = config.mesh 
nonods = config.nonods 
ngi = config.ngi
ndim = config.ndim
nloc = config.nloc 
dt = config.dt 
tend = config.tend 
tstart = config.tstart
dev = config.dev
sngi = config.sngi

def SHATRInew(nloc,ngi,ndim, snloc=4, sngi=4):
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

    l1=np.zeros(ngi)
    l2=np.zeros(ngi)
    l3=np.zeros(ngi)
    weight=np.zeros(ngi)

    nface = config.nface
    sl1 = np.zeros((nface, sngi))
    sl2 = np.zeros((nface, sngi))
    sl3 = np.zeros((nface, sngi))
    sweight = np.zeros(sngi)

    if (ndim!=2) :
        raise Exception('Input dimension should be 2')
        
    if (ngi==13):
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

    weight = weight*0.5

    if (sngi==4):
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
    def __init__(self, nlx):
        super(det_nlx, self).__init__()

        # calculate jacobian
        self.calc_j11 = Conv1d(in_channels=1, \
            out_channels=ngi, \
            kernel_size=10, \
            bias=False)
        self.calc_j12 = Conv1d(in_channels=1, \
            out_channels=ngi, \
            kernel_size=10, \
            bias=False)
        self.calc_j21 = Conv1d(in_channels=1, \
            out_channels=ngi, \
            kernel_size=10, \
            bias=False)
        self.calc_j22 = Conv1d(in_channels=1, \
            out_channels=ngi, \
            kernel_size=10, \
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
        
    def forward(self, x_loc, weight):
        '''
        
        # input 
        
        x_loc - (batch_size , ndim, nloc), coordinate info of local nodes
            reference coordinate: (xi, eta)
            physical coordinate: (x, y)
        weight  -        np array (ngi)

        # output
        nx - shape function derivatives Nix & Niy, 
            (batch_size, ndim, ngi, nloc)
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
        nlx1 = torch.tensor(np.transpose(self.nlx[0,:,:]), device=dev)
        nlx1 = nlx1.expand(batch_in,ngi,nloc)
        nlx2 = torch.tensor(np.transpose(self.nlx[1,:,:]), device=dev)
        nlx2 = nlx2.expand(batch_in,ngi,nloc)
        j12 = self.calc_j12(y).view(batch_in, ngi)
        j22 = self.calc_j22(y).view(batch_in, ngi)
        invj11 = torch.mul(j22,invdet).view(batch_in,-1)
        invj12 = torch.mul(j12,invdet).view(batch_in,-1)*(-1.0)
        del j22 
        del j12
        invj11 = invj11.unsqueeze(-1).expand(batch_in,ngi,nloc)
        invj12 = invj12.unsqueeze(-1).expand(batch_in,ngi,nloc)
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
        invj21 = invj21.unsqueeze(-1).expand(batch_in,ngi,nloc)
        invj22 = invj22.unsqueeze(-1).expand(batch_in,ngi,nloc)
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
def sdet_snlx(snlx, x_loc, sweight):
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
    
    # print(snx.shape, sdetwei.shape)

    return snx, sdetwei, snormal
