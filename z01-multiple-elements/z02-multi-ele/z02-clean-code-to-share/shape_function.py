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

def SHATRInew(nloc,ngi,ndim):
    '''
    shape functions on a reference element
    input: nloc, ngi, ndim
    output: 
    n, shape functions on a reference element at quadrature points, 
          numpy array (nloc, ngi)
    nlx, shape function deriatives on a reference element at quad pnts,
          numpy array (ndim, nloc, ngi)
    weight, quad pnts weights, np array (ngi)
    '''

    l1=np.zeros(ngi)
    l2=np.zeros(ngi)
    l3=np.zeros(ngi)
    weight=np.zeros(ngi)

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
    return n, nlx_all, weight


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
        det = abs( det )
        invdet = torch.div(1.0,det)
        det = torch.mul(det, torch.tensor(weight, device=dev).unsqueeze(0).expand(det.shape[0],ngi)) # detwei
        # print('det', det)
        # print('invdet', invdet)
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
        j11 = self.calc_j11(x).view(batch_in, ngi)
        j21 = self.calc_j21(x).view(batch_in, ngi)
        invj11 = torch.mul(j11,invdet).view(batch_in,-1)
        invj12 = torch.mul(j21,invdet).view(batch_in,-1)*(-1.0)
        del j11 
        del j21
        invj11 = invj11.unsqueeze(-1).expand(batch_in,ngi,nloc)
        invj12 = invj12.unsqueeze(-1).expand(batch_in,ngi,nloc)
        nx1 = torch.mul(invj11, nlx1) \
            + torch.mul(invj12, nlx2)
        del invj11 
        del invj12 
        # print('invj11', invj11)
        # print('invj12', invj12)
        # print('invj21', invj21)
        # print('invj22', invj22)

        # print('nlx1', nlx1)
        # print('nlx2', nlx2)
        j12 = self.calc_j12(y).view(batch_in, ngi)
        j22 = self.calc_j22(y).view(batch_in, ngi)
        invj21 = torch.mul(j12,invdet).view(batch_in,-1)*(-1.0)
        invj22 = torch.mul(j22,invdet).view(batch_in,-1)
        del j12
        del j22 
        invj21 = invj21.unsqueeze(-1).expand(batch_in,ngi,nloc)
        invj22 = invj22.unsqueeze(-1).expand(batch_in,ngi,nloc)
        del invdet 
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
