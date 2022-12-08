#!/usr/bin/env python3

# import 
import toughio 
import numpy as np
import torch
from torch.nn import Conv1d,Sequential,Module
import time
from scipy.sparse import coo_matrix 
from tqdm import tqdm

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
dt = 1e-4 # timestep
tstart=0
tend=10
print(mesh.cells[0][1]) # cell node list
print(mesh.points)

# create faces
faces=[]

for ele in range(nele):
    element = mesh.cells[0][1][ele]
    for iloc in range(3):
        faces.append([element[iloc],element[(iloc+1)%3]])

# neighbouring faces (global indices)
# input: a global face index
# output: the global index of the input face's neighbouring face
#         sign denotes face node numbering orientation
#         np.nan denotes no neighbouring found (implicating this is a boundary face)
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
# output type: float, sign denotes face node numbering orientation
#              nan denotes non found (input is boundary element)
#              !! convert to positive int before use as index !!
nbele=np.empty(len(faces))
nbele[:]=np.nan
for iface in range(len(nbf)):
    nbele[iface] = np.sign(nbf[iface])*(np.abs(nbf[iface])//3)



# generate cubic nodes
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
print('x_all shape: ', x_all.shape)

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
print(bc1)
print(bc2)
print(bc3)
print(bc4)

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
        return nx, det

# test det_nlx shape
# [n, nlx, weight] = SHATRInew(nloc, ngi, ndim)
# Det_nlx = det_nlx(nlx)
# inputx = torch.randn(5,ndim,nloc, requires_grad=False)
# with torch.no_grad():
#     output, detwei = Det_nlx(inputx)
# ================================
# test passed

dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dev="cpu"
print('computation on ',dev)

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
# x_ref_in1 = np.asarray([ 1.0, 0.0, \
#             0.0, 1.0, \
#             0.0, 0.0, \
#             2./3., 1./3., \
#             1./3., 2./3., \
#             0., 2./3., \
#             0., 1./3., \
#             1./3., 0., \
#             2./3., 0., \
#             1./3., 1./3.])
# x_ref_in1 = x_ref_in1.reshape((nloc,ndim))

# ele=15
# x_all=np.asarray(x_all)
# toplt = np.arange((ele)*nloc,(ele)*nloc+nloc)

# x_ref_in2 = x_all[toplt,:]

# x_ref_in1 = np.transpose(x_ref_in1)
# x_ref_in2 = np.transpose(x_ref_in2)
# x_ref_in = np.stack((x_ref_in1,x_ref_in2), axis=0)
# x_ref_in = torch.tensor(x_ref_in1, requires_grad=False, device=dev).view(1,2,nloc)
# x_ref_in = torch.transpose(x_ref_in, 0,1).unsqueeze(0)
# print('xin size', x_ref_in.shape)
# print(x_ref_in)
# x_ref_in = x_ref_in.repeat(1000000,1,1)
# print(torch.cuda.memory_summary())
# # np.savetxt('x_ref_in.txt',x_ref_in,delimiter=',')
# # x_ref_in = torch.tensor(x_ref_in,requires_grad=False)#.unsqueeze(0)
# # print(x_ref_in.shape)
# start = time.time()


# calculate shape functions from element nodes coordinate
x_ref_in = np.empty((nele, ndim, nloc))
for ele in range(nele):
    for iloc in range(nloc):
        glb_iloc = ele*nloc+iloc
        x_ref_in[ele,0,iloc] = x_all[glb_iloc,0]
        x_ref_in[ele,1,iloc] = x_all[glb_iloc,1]
x_ref_in = torch.tensor(x_ref_in, device=dev, requires_grad=False)
# print(x_ref_in.shape)

with torch.no_grad():
    nx, detwei = Det_nlx.forward(x_ref_in, weight)
# print('det size', det.shape)
# print('weight size', torch.tensor(weight).unsqueeze(0).expand(det.shape[0],ngi).shape)
# end = time.time()
# print('time: ', end-start, 'on ', dev)
# print(torch.cuda.memory_summary())
# print('nx fresh', nx)
# print('det', det)
# print('detwei', detwei)
# np.savetxt('nx.txt', np.squeeze(nx[0,0,:,:]), delimiter=',')
# np.savetxt('detwei.txt', detwei, delimiter=',')

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
        # b  - rhs of Mc=b, (batch_size, 1, nloc)

        batch_in = c.shape[0]
        # stiffness matrix 
        nx1nx1 = torch.mul(nx[:,0,:,:].view(batch_in, ngi, nloc), \
            detwei.unsqueeze(-1).expand(batch_in, ngi, nloc)) # (batch_in, ngi, nloc)
        nx1nx1 = torch.bmm(torch.transpose(nx[:,0,:,:].view(batch_in, ngi, nloc), 1,2), \
            nx1nx1) # (batch_in, nloc, nloc)
        # print('nx1nx1',nx1nx1)
        nx2nx2 = torch.mul(nx[:,1,:,:].view(batch_in, ngi, nloc), \
            detwei.unsqueeze(-1).expand(batch_in, ngi, nloc)) # (batch_in, ngi, nloc)
        nx2nx2 = torch.bmm(torch.transpose(nx[:,1,:,:].view(batch_in, ngi, nloc), 1,2), \
            nx2nx2) # (batch_in, nloc, nloc)
        del nx
        nxnx = (nx1nx1+nx2nx2)*k # scalar multiplication, (batch_in, nloc, nloc)
        del nx1nx1 , nx2nx2 
        # for ele in range(batch_in):
        #     np.savetxt('nxnx'+str(ele)+'.txt',nxnx[ele,:,:].view(nloc,nloc),delimiter=',')
        
        # print('nxnx', nxnx)
        # mass matrix
        nn = torch.mul(n.unsqueeze(0).expand(batch_in, ngi, nloc), \
            detwei.unsqueeze(-1).expand(batch_in, ngi, nloc))   # (batch_in, ngi, nloc)
        nn = torch.bmm(torch.transpose(n,0,1).unsqueeze(0).expand(batch_in, nloc, ngi), \
            nn) # (batch_in, nloc, nloc)
        
        nxnx = nn - nxnx*dt # this is (M-dt K), (batch_in, nloc, nloc)
        b = torch.matmul(nxnx,torch.transpose(c,1,2)) # batch matrix-vector multiplication, 
            # input1: (batch_in, nloc, nloc)
            # input2: (batch_in, nloc, 1)
            # broadcast over batch
            # output: (batch_in, nloc, 1)
        b = torch.transpose(b,1,2) # return to (batch_in, 1, nloc)
        
        return nn, b

Mk = mk()
Mk.to(device=dev)
c = np.arange(1,nonods+1).reshape(nele,nloc)
c = torch.tensor(c, dtype=torch.float64, device=dev).view(-1,1,nloc)
# c = c.repeat(1000000,1,1)
# print(c.shape)
# print(c)
n = torch.transpose(torch.tensor(n, device=dev),0,1)
# print(type(n), n.shape)
# print(type(nx), nx.shape)
# print('nx',nx[:,0,:,:].view(1, ngi, nloc))
# print(type(detwei), detwei.shape)
# start = time.time()
with torch.no_grad():
    [M,b] = Mk.forward(c,k=1,dt=1e-3,n=n,nx=nx,detwei=detwei)
# end = time.time()
# print('time consumed with dev = ', dev , end-start)
# print('mass matrix\n', torch.squeeze(M))
# cc = torch.tensor(np.arange(1,14), dtype=torch.float64).view(1,ngi)
# print(cc.unsqueeze(-1).expand(1, ngi, nloc))
########################
# Pass test!
########################



###
# jacobi iteration to solve for c^(n*)
###
# class jac_it(Module):
#     def __init__(self):
#         super(jac_it, self).__init__()

#     def forward(self, c, M, b):
#         # do one jacobi iteration
#         # input:
#         # c - node values at last iteration, (batch_size, 1, nloc)
#         # M - consistent mass matrix of Mx=b, (batch_size, nloc, nloc)
#         # b - rhs vector of Mx=b, (batch_size, 1, nloc)
#         batch_in = c.shape[0]
#         res = b - torch.matmul(M,torch.transpose(c,1,2)).view(batch_in, 1, nloc) # (batchin, 1, nloc)
#         diagM = torch.clone(torch.diagonal(M,offset=0,dim1=1,dim2=2)) # (batchin,nloc)
#         diagM = torch.div(1.0, diagM) # 1/Diag(M), (batchin,nloc)
#         l2res = torch.linalg.norm(res.view(batch_in,nloc), ord=2, dim=-1)
#         # print('l2res size', l2res.shape)
#         # print('res ', res)
#         print('diagm inv', diagM)
#         c = c + torch.mul(diagM, res) # (batch_in, 1, nloc)

#         return c , l2res

# # Jac_it = jac_it()
# # Jac_it.to(device=dev)
# # print('c',c)
# # print('M',M)
# # print('b',b)
# # M_np = M.view(nloc,nloc).numpy() # convert to numpy 
# # np.savetxt('Mnp.txt' , M_np, delimiter=',')
# # b_np = b.view(nloc).numpy() 
# # np.savetxt('b.txt', b_np, delimiter=',')
# # with torch.no_grad():
# #     for its in range(10):
# #         print('c before', c)
# #         [c,l2res]=Jac_it.forward(c,M,b)
# #         print('============its: ',its,'=============')
# #         print('c after', c)
# #         print('l2res', l2res)

# Jac_it = jac_it()
# Jac_it.to(device=dev)
# nloc=4 # change temporarily to test jacobi iteration
# Ajac = [[2.52, 0.95, 1.25, -.85],\
#     [0.39, 1.69, -.45, .49],\
#     [.55, -1.25, 1.96, -.98], \
#     [.23, -1.15, -.45, 2.31]]
# Ajac = torch.tensor(Ajac, dtype=torch.float64,device=dev).view(1,4,4)
# print('Ajac', Ajac)
# bjac = torch.tensor([1.38,-.34,.67,1.52], dtype=torch.float64,device=dev).view(1,1,4)
# c = torch.tensor([0,0,0,0],dtype=torch.float64,device=dev).view(1,1,4)
# with torch.no_grad():
#     for its in range(100):
#         print('c before', c)
#         [c,l2res]=Jac_it.forward(c,Ajac,bjac)
#         print('============its: ',its,'=============')
#         print('c after', c)
#         print('l2res', l2res)
#############################
# jacobi iteration works for small 4x4 matrix (example taken from Numerical Analysis (in CN) pg. 113 3.1)
# then it's the mass matrix that doesn't work.
# maybe we should use a relaxed jacobi iteration?...
# may be we don't use jacobi iteration at all!
# use torch.linalg.inv() instead. Input A(*,nloc,nloc)
#############################

# inv mass matrix
# input: M (batch_in, nloc, nloc)
# output: Minv (batch_in, nloc, nloc)
Minv = torch.linalg.inv(M)
del M # we probably don't need M anymore



#####################################
# now we assemble S (surface integral)
#####################################

# first let's define surface shape functions
# note that we're using mass lumping and
# only one value per node is required
# that is either 1/3 or 1/6 multiplied by 
# the curve length

#####
# build sparsity
#####
# let's build in coo format first - easier to construct
# then transform to csr format - more efficient to do linear algebra operations
def S_Minv_sparse(Minv):
    # input:
    # Minv - torch tensor (nele,nloc,nloc)
    # output:
    # S - surface integral matrix, torch.sparse_csr_tensor
    # Minv - inverse of mass matrix, torch.sparse_csr_tensor

    # S matrix 
    indices=[] # indices of entries, a list of lists
    values=[]  # values to be add to S
    for ele in range(nele):
        # loop over surfaces
        for iface in range(3):
            glb_iface = ele*3+iface 
            ele2 = nbele[glb_iface]
            if (np.isnan(ele2)):
                # this is a boundary face without neighbouring element
                continue 
            ele2 = int(abs(ele2))
            glb_iface2 = int(abs(nbf[glb_iface]))
            iface2 = glb_iface2 % 3
            dx=np.linalg.norm(x_all[ele*nloc+9,:]-x_all[int(ele2)*nloc+9,:])/4. # dx/(order+1)
            # print(ele, iface, dx,x_all[ele*nloc+9,:], x_all[int(ele2)*nloc+9,:])
            farea=np.linalg.norm(x_all[ele*nloc+iface,:]-x_all[ele*nloc+(iface+1)%3,:])
            # print(ele, iface, farea)
            # print(ele, ele2, '|', iface, iface2, '|', face_iloc(iface), face_iloc2(iface2))
            for iloc,iloc2 in zip(face_iloc(iface), face_iloc2(iface2)):
                glb_iloc = ele*nloc+iloc 
                glb_iloc2 = int(abs(ele2*nloc+iloc2))
                # print(ele, ele2, '|', iface, iface2, '|', iloc, iloc2, '|', glb_iloc, glb_iloc2)
                # print('\t',x_all[glb_iloc]-x_all[glb_iloc2])
                            
                indices.append([glb_iloc, glb_iloc])
                indices.append([glb_iloc, glb_iloc2])
            # S matrix value                  face node   0--1--2--3
            values.append(1./6.*farea/dx)   # 0     diag  
            values.append(-1./6.*farea/dx)  # 0     off-diag
            values.append(1./3.*farea/dx)   # 1     diag
            values.append(-1./3.*farea/dx)  # 1     off-diag
            values.append(1./3.*farea/dx)   # 2     diag
            values.append(-1./3.*farea/dx)  # 2     off-diag
            values.append(1./6.*farea/dx)   # 3     diag
            values.append(-1./6.*farea/dx)  # 3     off-diag

    values = torch.tensor(values)
    # print(values)
    indices = torch.transpose(torch.tensor(indices),0,1)



    S_scipy = coo_matrix((values, (indices[0,:].numpy(), indices[1,:].numpy()) ), shape=(nonods, nonods))
    S_scipy = S_scipy.tocsr()  # this transformation will altomatically add entries at same position, perfect for assembling
    S = torch.sparse_csr_tensor(crow_indices=torch.tensor(S_scipy.indptr), \
        col_indices=torch.tensor(S_scipy.indices), \
        values=S_scipy.data, \
        size=(nonods, nonods), \
        device=dev)
    # np.savetxt('indices.txt',S.to_dense().numpy(),delimiter=',')

    # inverse of mass matrix Minv_sparse
    indices=[]
    values=[]
    for ele in range(nele):
        for iloc in range(nloc):
            for jloc in range(nloc):
                glb_iloc = int( ele*nloc+iloc )
                glb_jloc = int( ele*nloc+jloc )
                indices.append([glb_iloc, glb_jloc])
                values.append(Minv[ele,iloc,jloc])
    values = torch.tensor(values)
    indices = torch.transpose(torch.tensor(indices),0,1)
    Minv_scipy = coo_matrix((values,(indices[0,:].numpy(), indices[1,:].numpy())), shape=(nonods, nonods))
    Minv_scipy = Minv_scipy.tocsr() 
    Minv = torch.sparse_csr_tensor( crow_indices=torch.tensor(Minv_scipy.indptr), \
        col_indices=torch.tensor(Minv_scipy.indices), \
        values=Minv_scipy.data, \
        size=(nonods, nonods) , \
        device=dev)

    return S, Minv


######### 
# time loop
###########
x_ref_in = np.empty((nele, ndim, nloc))
for ele in range(nele):
    for iloc in range(nloc):
        glb_iloc = ele*nloc+iloc
        x_ref_in[ele,0,iloc] = x_all[glb_iloc,0]
        x_ref_in[ele,1,iloc] = x_all[glb_iloc,1]
x_ref_in = torch.tensor(x_ref_in, device=dev, requires_grad=False)

# initical condition
c = np.zeros(nonods)
# c = np.asarray([1.000000000000000000e+00,2.000000000000000000e+00,1.493838904965597569e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.449966370388888537e+00,1.469033186016506010e+00,1.347141914183232148e+00,1.152750854595107066e+00,1.160140167446554349e+00,2.000000000000000000e+00,2.000000000000000000e+00,1.604796987540639153e+00,2.000000000000000000e+00,2.000000000000000000e+00,1.924636785854314303e+00,1.777579115307477586e+00,1.598073528854923264e+00,1.739355725613122461e+00,1.880373604087656636e+00,2.000000000000000000e+00,1.754399305660250041e+00,1.860698503999662634e+00,1.912555293209513341e+00,1.832992867485747412e+00,1.793603819785748232e+00,1.828777781554110415e+00,1.904292029262314001e+00,1.950644311037632583e+00,1.870696672554985618e+00,2.000000000000000000e+00,2.000000000000000000e+00,1.875466519341594918e+00,2.000000000000000000e+00,2.000000000000000000e+00,1.970084772125898498e+00,1.921416927406050412e+00,1.909446495780714681e+00,1.959869059419381498e+00,1.966051717905866525e+00,1.708788247347079237e+00,1.590406631659132985e+00,1.598421738447025842e+00,1.645239968676932341e+00,1.606731501002102158e+00,1.592745392397802329e+00,1.597020424780011716e+00,1.635550814925486929e+00,1.669762327596505669e+00,1.621135676584691820e+00,1.588358489223376413e+00,1.453113046928694585e+00,1.584281940384167919e+00,1.571851118994411767e+00,1.530738882663677636e+00,1.503649385313970965e+00,1.544286115996887920e+00,1.582328190551923974e+00,1.585873858125989688e+00,1.557301076290084296e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.224463585125032283e+00,1.000000000000000000e+00,1.000000000000000000e+00,1.076461232701819482e+00,1.168821855317328406e+00,1.139537447109792545e+00,1.052369945083597269e+00,1.061829941916374143e+00,1.391892842440358713e+00,1.000000000000000000e+00,1.251255031108101878e+00,1.278956361726842017e+00,1.149832667989081125e+00,1.085427114030233176e+00,1.167440717592010957e+00,1.293160324816911677e+00,1.340071754548867178e+00,1.220546144132276023e+00,1.000000000000000000e+00,1.465468887236520246e+00,1.266761006261882372e+00,1.151271503687611775e+00,1.305470291792105275e+00,1.360533856633150629e+00,1.302941410124508259e+00,1.228825867503321900e+00,1.145876673913813315e+00,1.257219388930236326e+00,1.737056362497977124e+00,1.658859748710394433e+00,1.839779815233241633e+00,1.726003219039905501e+00,1.700235873718550117e+00,1.722120836381548115e+00,1.781083135021837682e+00,1.801759549431894536e+00,1.769540568893342769e+00,1.752270443868931826e+00,1.642824389936767826e+00,1.722402891657959900e+00,1.601318832181879204e+00,1.678888388057656833e+00,1.706738930968299872e+00,1.683622891114131548e+00,1.643092127937530122e+00,1.610131826480696304e+00,1.623806846994243491e+00,1.659712817768047133e+00,2.000000000000000000e+00,1.658795287023820730e+00,1.597301394522557372e+00,1.846920713034494455e+00,1.734544945483164735e+00,1.659963175674846303e+00,1.645024360694456789e+00,1.766236358058662415e+00,1.894559238560293846e+00,1.750396877319061684e+00,1.668209330932912682e+00,2.000000000000000000e+00,1.853874193600455245e+00,1.751994352722610504e+00,1.857143353739321601e+00,1.927765923022346950e+00,1.882183199279316987e+00,1.798558452053737211e+00,1.734623621035235930e+00,1.823910141926509132e+00,1.610450060843217734e+00,1.416911105440552099e+00,1.295550763684644391e+00,1.506742415673864466e+00,1.442391602695812791e+00,1.376821805580491720e+00,1.341504056475049333e+00,1.414273724536526444e+00,1.518900368834024350e+00,1.426227949409357487e+00,1.435834199812773937e+00,1.618397721589056815e+00,1.583313378420909023e+00,1.467038039402340344e+00,1.524616777838820658e+00,1.590381285395748634e+00,1.582518822908904665e+00,1.538800727920351941e+00,1.488600745986405682e+00,1.528243591423988024e+00,1.501419304636960828e+00,1.628227026803016120e+00,1.282718202701798704e+00,1.571271161698147845e+00,1.611625878276715662e+00,1.535405207842922559e+00,1.404060315890601762e+00,1.325333598289352599e+00,1.391144332285839891e+00,1.475470254153124605e+00])
# apply boundary conditions (two Dirichlet bcs)
for inod in bc1:
    c[inod]=1.
for inod in bc2:
    c[inod]=2.

tstep=int(np.ceil((tend-tstart)/dt))
# print(c)
c = c.reshape(nele,nloc)
# print(c)
c = torch.tensor(c, dtype=torch.float64, device=dev).view(-1,1,nloc)
c_all=np.empty([tstep,nonods])
c_all[0,:]=c.view(-1).cpu().numpy()[:]
#
for itime in tqdm(range(1,tstep)):
    c = c.view(-1,1,nloc)
    # calculate shape functions from element nodes coordinate
    # print(x_ref_in.shape)
    with torch.no_grad():
        nx, detwei = Det_nlx.forward(x_ref_in, weight)

    # mass matrix and rhs
    with torch.no_grad():
        [M,b] = Mk.forward(c,k=1,dt=dt,n=n,nx=nx,detwei=detwei)
    
    # inverse mass matrix
    Minv = torch.linalg.inv(M)
    del M # we probably don't need M anymore
    # print(b)
    # np.savetxt('b.txt', b.view(-1).numpy(), delimiter=',')
    # np.savetxt('cbefore.txt', c.view(-1).numpy(), delimiter=',')
    # next step 1 (inner element)
    cn_inele = Minv @ torch.transpose(b,1,2)
    cn_inele = cn_inele.view(-1)
    # np.savetxt('cafter.txt', cn_inele.view(-1).numpy(), delimiter=',')
    
    
    # surface integral 
    [S, Minv] = S_Minv_sparse(Minv)
    # print('device S Minvm c', S.is_cuda, Minv.is_cuda, c.is_cuda)
    cn_surf = torch.sparse.mm(Minv, torch.sparse.mm(S, c.view(nonods,1)))*dt
    cn_surf = cn_surf.view(-1)
    # if itime==1:
        # print(Minv.to_dense().shape)
        # np.savetxt('Minv.txt',Minv.to_dense().numpy(),delimiter=',')
    # np.savetxt('cn_inele.txt', cn_inele.view(-1).numpy(), delimiter=',')
    # np.savetxt('cn_surf.txt', cn_surf.view(-1).numpy(), delimiter=',')
    c = cn_inele-cn_surf 
    # np.savetxt('c.txt', c.view(-1).numpy(), delimiter=',')

    # apply boundary conditions (two Dirichlet bcs)
    for inod in bc1:
        c[inod]=1.
    for inod in bc2:
        c[inod]=2.

    # toughio.write_time_series(filename='output.vtk', \
    #     points=mesh.points, \
    #     cells=mesh.cells, \
    #     point_data=,\
    #     time_steps=itime )
    # if (itime%100==0):
    # c_all[itime,:]=c.view(-1).cpu().numpy()[:]

####
# Minv * b passed test
####
# c_all = np.asarray(c_all)[::100,:]
# np.savetxt('c_all.txt', c_all, delimiter=',')
# np.savetxt('x_all.txt', x_all, delimiter=',')