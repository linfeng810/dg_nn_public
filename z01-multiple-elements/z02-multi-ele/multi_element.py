#!/usr/bin/env python3

# import 
import toughio 
import numpy as np
import torch
from torch.nn import Conv1d,Sequential,Module
import time

filename='/data2/linfeng/aicfd/z01-chrisemail/z01-multiple-elements/z01-geo/square.msh'
mesh = toughio.read_mesh(filename)

# mesh info
nele = mesh.n_cells
cubic = True
if (cubic) :
    nloc = 10 
    ngi = 13
nonods = nloc*nele 
ndim = 2
ndglno=np.arange(0,nonods)
# print(mesh.cells[0][1]) # cell node list
# mesh.points

# create faces
faces=[]

for ele in range(nele):
    element = mesh.cells[0][1][ele]
    for iloc in range(3):
        faces.append([element[iloc],element[(iloc+1)%3]])

# neighbouring elements
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
nbele=np.empty(len(faces))
nbele[:]=np.nan
for iface in range(len(nbf)):
    nbele[iface] = np.sign(nbf[iface])*(np.abs(nbf[iface])//3)



# generate cubic nodes
x_all = []
for ele in range(16):
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


# local nodes number on a face
def face_iloc(iface):
    # return local nodes number on a face
    match iface:
        case 0:
            return [0,3,4,1]
        case 1:
            return [1,5,6,2]
        case 2:
            return [0,8,7,2]
        case _:
            return []
def face_iloc2(iface):
    # return local nodes number on the other side of a face
    # in reverse order 
    iloc_list=face_iloc(iface)
    iloc_list.reverse()
    return iloc_list

# shape functions on a reference element
# input: nloc, ngi, ndim
# n, nlx, weight
def SHATRInew(nloc,ngi,ndim):
    l1=np.zeros(ngi)
    l2=np.zeros(ngi)
    l3=np.zeros(ngi)
    weight=np.zeros(ngi)

    if (ndim!=2) :
        raise Exception('Input dimension should be 2')
        
    if (ngi==13):
        alpha = -0.149570044467682
        print(type(alpha))
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
    



# test shape function on reference node
# [n, nlx, weight] = SHATRInew(nloc, 13, 2)
# nn=np.zeros((nloc,nloc))
# nxnx=np.zeros((nloc,nloc))
# for iloc in range(nloc):
#     for jloc in range(nloc):
#         nn[iloc,jloc]=np.sum(n[iloc,:]*n[jloc,:]*weight)
#         nxnx[iloc,jloc]=np.sum(nlx[0,iloc,:]*nlx[0,jloc,:]*weight)+np.sum(nlx[1,iloc,:]*nlx[1,jloc,:]*weight)
# np.set_printoptions(suppress=True)
# np.savetxt("nn.txt", nn, delimiter=',')
# np.savetxt("nxnx.txt", nxnx, delimiter=',')
# # print(n)
# print(nlx) 
# print(weight)
# print('sum of weight', np.sum(weight))
# print('nn')
# print(nn)
# print('nxnx')
# print(nxnx)
#======================================================================
# test passed. 


# for pretty print out torch tensor
torch.set_printoptions(sci_mode=False)
# local shape function
# input: n, nlx, ngi, ndim, x_loc, nloc, weight
# output: nx, detwei, inv_jac
# maybe we should write this on a layer of NN?
# def det_nlx(n,nlx,ngi,ndim,x_loc,nloc,weight):
#     return nx, detwei, inv_jac
class det_nlx(Module):
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
        
    def forward(self, x_loc):
        # input : x_loc
        # (batch_size , ndim, nloc), coordinate info of local nodes
        # reference coordinate: (xi, eta)
        # physical coordinate: (x, y)
        print('x_loc size', x_loc.shape)
        print('x size', x_loc[:,0,:].shape)
        batch_in = x_loc.shape[0]
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
        # print('det', det)
        invdet = torch.div(1.0,det)
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
        nlx1 = torch.tensor(np.transpose(nlx[0,:,:]), device=dev)
        nlx1 = nlx1.expand(batch_in,ngi,nloc)
        nlx2 = torch.tensor(np.transpose(nlx[1,:,:]), device=dev)
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
        return nx, abs( det )

# test det_nlx shape
# [n, nlx, weight] = SHATRInew(nloc, ngi, ndim)
# Det_nlx = det_nlx(nlx)
# inputx = torch.randn(5,ndim,nloc, requires_grad=False)
# with torch.no_grad():
#     output, detwei = Det_nlx(inputx)
# ================================
# test passed

dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dev="cpu"

## set weights in det_nlx
[n, nlx, weight] = SHATRInew(nloc, ngi, ndim)
Det_nlx = det_nlx(nlx)
Det_nlx.to(dev)

# filter for calc jacobian
calc_j11_j12_filter = np.transpose(nlx[0,:,:]) # dN/dx
calc_j11_j12_filter = torch.tensor(calc_j11_j12_filter, device=dev).unsqueeze(1) # (ngi, 1, nloc)
calc_j21_j22_filter = np.transpose(nlx[1,:,:]) # dN/dy
calc_j21_j22_filter = torch.tensor(calc_j21_j22_filter, device=dev).unsqueeze(1) # (ngi, 1, nloc)
# print(Det_nlx.calc_j11.weight.shape)
# print(nlx.shape)
# print(calc_j21_j22_filter.shape)
Det_nlx.calc_j11.weight.data = calc_j11_j12_filter
Det_nlx.calc_j12.weight.data = calc_j11_j12_filter
Det_nlx.calc_j21.weight.data = calc_j21_j22_filter
Det_nlx.calc_j22.weight.data = calc_j21_j22_filter
# print(Det_nlx.calc_j11.weight.shape)
# print(Det_nlx.calc_j11.weight.data)

#####################################
# test two elements input
#
x_ref_in1 = np.asarray([ 1.0, 0.0, \
            0.0, 1.0, \
            0.0, 0.0, \
            2./3., 1./3., \
            1./3., 2./3., \
            0., 2./3., \
            0., 1./3., \
            1./3., 0., \
            2./3., 0., \
            1./3., 1./3.])
x_ref_in1 = x_ref_in1.reshape((nloc,ndim))

ele=15
x_all=np.asarray(x_all)
toplt = np.arange((ele)*nloc,(ele)*nloc+nloc)

x_ref_in2 = x_all[toplt,:]

x_ref_in1 = np.transpose(x_ref_in1)
x_ref_in2 = np.transpose(x_ref_in2)
x_ref_in = np.stack((x_ref_in1,x_ref_in2), axis=0)
x_ref_in = torch.tensor(x_ref_in, requires_grad=False, device=dev)
# x_ref_in = torch.transpose(x_ref_in, 0,1).unsqueeze(0)
print('xin size', x_ref_in.shape)
print(x_ref_in)
# x_ref_in = x_ref_in.repeat(1500000,1,1)
# print(torch.cuda.memory_summary())
# # np.savetxt('x_ref_in.txt',x_ref_in,delimiter=',')
# # x_ref_in = torch.tensor(x_ref_in,requires_grad=False)#.unsqueeze(0)
# # print(x_ref_in.shape)
# start = time.time()
with torch.no_grad():
    nx, det = Det_nlx.forward(x_ref_in)
detwei = torch.mul(det, torch.tensor(weight).unsqueeze(0).expand(det.shape[0],ngi))
print('det size', det.shape)
print('weight size', torch.tensor(weight).unsqueeze(0).expand(det.shape[0],ngi).shape)
# end = time.time()
# print('time: ', end-start, 'on ', dev)
# print(torch.cuda.memory_summary())
print('nx', nx)
print('det', detwei)
np.savetxt('nx.txt', np.squeeze(nx[0,0,:,:]), delimiter=',')
np.savetxt('detwei.txt', detwei, delimiter=',')

# # print('j11_filter', calc_j11_j12_filter)
# # print('x_fen_in', torch.squeeze(x_ref_in[:,0,:]))
# # print('j11-outsidenn', torch.matmul(calc_j11_j12_filter,torch.squeeze(x_ref_in[:,0,:])))

# test passed
################################################

# assemble mass matrix and stiffness matrix
# solve for intermediate c^(n*):
# 
# 1/dt M( c^(n*) - c^(n-1) ) + K c^(n-1) = 0
# ==>
# M c^(n*) = (M-dt*K) c^(n-1)
# to form a Jacobi iteration, here we calculate 
# M and b = (M-dt*K) c^(n-1)
class mk(Module):
    def __init__(self):
        super(mk, self).__init__()

    def forward(self, c, k, dt, n, nx, detwei):
        ### input
        # c  - node values at last timestep, (batch_size, 1, nloc)
        # k  - diffusion coefficient at node, right now we simplify it using constant k. (1)
        # dt - timestep. (1)
        # n  - shape function Ni, (ngi, nloc)
        # nx - shape function derivatives Nix & Niy, (batch_size, ndim, ngi, nloc)
        # detwei - determinant times GI weight, (batch_size, ngi)
        ### output
        # nn - mass matrix (consistent), (batch_size, nloc, nloc)
        # b  - node values at next (future) timestep, (batch_size, 1, nloc)

        batch_in = c.shape[0]
        # mass matrix
        nn = torch.mul(n.unsqueeze(0).expand(batch_in, ngi, nloc), \
            detwei.unsqueeze(-1).expand(batch_in, ngi, nloc))   # (batch_in, ngi, nloc)
        nn = torch.bmm(torch.transpose(n,0,1).unsqueeze(0).expand(batch_in, nloc, ngi), \
            nn) # (batch_in, nloc, nloc)
        
        # stiffness matrix 
        nx1nx1 = torch.mul(nx[:,0,:,:].view(batch_in, ngi, nloc), \
            detwei.unsqueeze(-1).expand(batch_in, ngi, nloc)) # (batch_in, ngi, nloc)
        nx1nx1 = torch.bmm(torch.transpose(nx[:,0,:,:].view(batch_in, ngi, nloc), 1,2), \
            nx1nx1) # (batch_in, nloc, nloc)
        print('nx1nx1',nx1nx1)
        nx2nx2 = torch.mul(nx[:,1,:,:].view(batch_in, ngi, nloc), \
            detwei.unsqueeze(-1).expand(batch_in, ngi, nloc)) # (batch_in, ngi, nloc)
        nx2nx2 = torch.bmm(torch.transpose(nx[:,1,:,:].view(batch_in, ngi, nloc), 1,2), \
            nx2nx2) # (batch_in, nloc, nloc)
        nxnx = (nx1nx1+nx2nx2)*k # scalar multiplication, (batch_in, nloc, nloc)
        del nx1nx1 , nx2nx2 
        print('stiffness matrix\n', nxnx)
        nxnx = nn - nxnx*dt # this is (M-dt K), (batch_in, nloc, nloc)
        b = torch.matmul(nxnx,torch.transpose(c,1,2)) # batch matrix-vector multiplication, 
            # input1: (batch_in, nloc, nloc)
            # input2: (batch_in, nloc, 1)
            # broadcast over batch
            # output: (batch_in, nloc, 1)
        b = torch.transpose(b,1,2) # return to (batch_in, 1, nloc)
        
        return nn, b

Mk = mk()
c = np.arange(1,21).reshape(2,nloc)
c = torch.tensor(c, dtype=torch.float64).view(-1,1,nloc)
print(c.shape)
print(c)
n = torch.transpose(torch.tensor(n),0,1)
print(type(n), n.shape)
print(type(nx), nx.shape)
# print('nx transpose',torch.transpose(nx[:,0,:,:].view(1, ngi, nloc), 1,2))
print(type(detwei), detwei.shape)
[M,b] = Mk.forward(c,1,1e-3,n,nx,detwei)
print('mass matrix\n', torch.squeeze(M))
# cc = torch.tensor(np.arange(1,14), dtype=torch.float64).view(1,ngi)
# print(cc.unsqueeze(-1).expand(1, ngi, nloc))
########################
# Pass test!
########################