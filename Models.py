import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter


#SAGEConv:
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size
from torch import Tensor
from torch.nn import Linear
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
# SGConv:
from typing import Optional
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
# Graph-UNet:
from torch_sparse import spspmm
from torch_geometric.nn import TopKPooling, GCNConv
from torch_geometric.utils import (sort_edge_index, dropout_adj)
from torch_geometric.utils.repeat import repeat
# GAE, VGAE:
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import (negative_sampling, remove_self_loops, add_self_loops)
from torch_geometric.nn.inits import reset  # ..inits means a folder "inits"
# AGE:
from torch.nn.modules.module import Module
import numpy as np
# TGCN (https://github.com/cassianobecker/tgcn):
#from torch.nn import Parameter, init
from torch_geometric.utils import degree#, remove_self_loops
from torch_scatter import scatter_add
# import autograd.numpy as npa
# GAT
from torch_geometric.nn import GCNConv, ChebConv, GATConv
from torch_geometric.nn import TransformerConv


# STGCN model: ---------------------------------------------------------------------------------

class align(nn.Module):
    def __init__(self, c_in, c_out):
        super(align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x

class temporal_conv_layer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(temporal_conv_layer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)

class spatio_conv_layer(nn.Module):
    def __init__(self, ks, c, Lk):
        super(spatio_conv_layer, self).__init__()
        self.Lk = Lk
        self.theta = nn.Parameter(torch.FloatTensor(c, c, ks))  # c for input dimension and for output dimension, for this layer
        self.b = nn.Parameter(torch.FloatTensor(1, c, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        x_c = torch.einsum("knm,bitm->bitkn", self.Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
        return torch.relu(x_gc + x)

class st_conv_block(nn.Module):
    def __init__(self, ks, kt, n, c, p, Lk):
        super(st_conv_block, self).__init__()
        self.tconv1 = temporal_conv_layer(kt, c[0], c[1], "GLU")
        self.sconv = spatio_conv_layer(ks, c[1], Lk)
        self.tconv2 = temporal_conv_layer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)   # c[2] is output channels and input for next block (either ST or output block).

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_s = self.sconv(x_t1)
        x_t2 = self.tconv2(x_s)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)

class fully_conv_layer(nn.Module):
    def __init__(self, c):
        super(fully_conv_layer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)
        # c input channels, 1 output channel (could be more if we'd want several features to be predicted), and conv 1X1 kernel.

    def forward(self, x):
        return self.conv(x)

class output_layer(nn.Module):
    def __init__(self, c, T, n):
        super(output_layer, self).__init__()
        self.tconv1 = temporal_conv_layer(T, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = temporal_conv_layer(1, c, c, "sigmoid")
        self.fc = fully_conv_layer(c)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        return self.fc(x_t2)

class STGCN(nn.Module):
# ks=kernel of 1D-conv for temporal, ks=2D-conv for spatial, T=#past steps input, n=#nodes.
    def __init__(self, ks, kt, bs, T, n, Lk, p):
        super(STGCN, self).__init__()
        self.st_conv1 = st_conv_block(ks, kt, n, bs[0], p, Lk)
        self.st_conv2 = st_conv_block(ks, kt, n, bs[1], p, Lk)
        self.output = output_layer(bs[1][2], T - 4 * (kt - 1), n)   # bs[1][2] is #output channels from 2nd ST-conv block.

    def forward(self, x):
        x_st1 = self.st_conv1(x)
        x_st2 = self.st_conv2(x_st1)
        return self.output(x_st2)


class STGCNb(nn.Module):
# ks=kernel of 1D-conv for temporal, ks=2D-conv for spatial, T=#past steps input, n=#nodes.
    def __init__(self, ks, kt, bs, T, n, Lk, p):
        super(STGCNb, self).__init__()
        bss = bs.copy()
        self.input_channels = bss[0][0]
        bss[0][0] = 1
        st_conv1 = []
        st_conv2 = []
        output = []
        for ii in range(self.input_channels):
            st_conv1.append(st_conv_block(ks, kt, n, bss[0], p, Lk))
            st_conv2.append(st_conv_block(ks, kt, n, bss[1], p, Lk))
            output.append(output_layer(bss[1][2], T - 4 * (kt - 1), n))   # bs[1][2] is #output channels from 2nd ST-conv block.
        output.append(nn.Conv2d(self.input_channels, 1, 1))     #output_layer(self.input_channels, 1, n))
        self.parms1 = nn.ModuleList(st_conv1)
        self.parms2 = nn.ModuleList(st_conv2)
        self.parms3 = nn.ModuleList(output)
        bss[0][0] = self.input_channels

    def forward(self, x):
        # x's shape: [data slots, channels, time steps, nodes]
        x_st1 = []
        x_st2 = []
        xs = list(x.size())
        xs[1] = self.input_channels
        xs[2] = 1
        x_st3 = torch.empty(xs) #np.empty(xs)
        xs = list(x.size())
        for ii in range(self.input_channels):
            xx = torch.reshape(x[:,ii,:,:], (xs[0], 1, xs[2], xs[3]))
            x_st1.append(self.parms1[ii](xx))
            x_st2.append(self.parms2[ii](x_st1[ii]))
            x_st3[:,ii,:,:] = torch.reshape(self.parms3[ii](x_st2[ii]), (xs[0], 1, xs[3]))
        yy = self.parms3[self.input_channels](x_st3.cuda())
        return yy


class STGCNc(nn.Module):
# ks=kernel of 1D-conv for temporal, ks=2D-conv for spatial, T=#past steps input, n=#nodes.
    def __init__(self, ks, kt, bs, T, n, Lk, p, input_diff):
        super(STGCNc, self).__init__()
        bss = bs.copy()
        self.input_channels = bss[0][0]
        self.nuniques = len(np.unique(np.array(input_diff)))
        self.input_diff = input_diff
        st_conv1 = []
        st_conv2 = []
        output = []
        for ii in range(self.nuniques):
            bss[0][0] = input_diff.count(ii)
            st_conv1.append(st_conv_block(ks, kt, n, bss[0], p, Lk))
            st_conv2.append(st_conv_block(ks, kt, n, bss[1], p, Lk))
            output.append(output_layer(bss[1][2], T - 4 * (kt - 1), n))   # bs[1][2] is #output channels from 2nd ST-conv block.
        output.append(output_layer(self.nuniques, 1, n))   # self.input_channels
        self.parms1 = nn.ModuleList(st_conv1)
        self.parms2 = nn.ModuleList(st_conv2)
        self.parms3 = nn.ModuleList(output)
        bss[0][0] = self.input_channels

    def forward(self, x):
        # x's shape: [data slots, channels, time steps, nodes]
        x_st1 = []
        x_st2 = []
        xs = list(x.size())
        xs[1] = self.nuniques  # self.input_channels    is the number of input channels to the ending output layer, fusing all previous
        xs[2] = 1   # number of time steps after each seperate STGCN
        x_st3 = torch.empty(xs) #np.empty(xs)
        xs = list(x.size())
        for ii in range(self.nuniques):
            rr = np.squeeze(np.where(np.array(self.input_diff) == ii))
            rt = self.input_diff.count(ii)
            xx = torch.reshape(x[:,rr,:,:], (xs[0], rt, xs[2], xs[3]))
            x_st1.append(self.parms1[ii](xx))
            x_st2.append(self.parms2[ii](x_st1[ii]))
            x_st3[:,ii,:,:] = torch.reshape(self.parms3[ii](x_st2[ii]), (xs[0], 1, xs[3]))
        yy = self.parms3[self.nuniques](x_st3.cuda())
        return yy

class STGCNd(nn.Module):
# ks=kernel of 1D-conv for temporal, ks=2D-conv for spatial, T=#past steps input, n=#nodes.
    def __init__(self, ks, kt, bs, T, n, Lk, p, input_diff):
        super(STGCNd, self).__init__()
        bss = bs.copy()
        self.input_channels = bss[0][0]
        self.nuniques = len(np.unique(np.array(input_diff)))
        self.input_diff = input_diff
        #st_conv1 = []
        #st_conv2 = []
        output = []
        for ii in range(self.nuniques):
            bss[0][0] = input_diff.count(ii)
            output.append(STGCN(ks,kt,bs,T,n,Lk,p))   # bs[1][2] is #output channels from 2nd ST-conv block.
        output.append(nn.Conv2d(self.nuniques, 1, 1)) #output_layer(self.nuniques, 1, n))   # self.input_channels
        self.parms1 = STGCN(ks,kt,[[1,32,64],[64,32,64]],T,n,Lk,p)#nn.ModuleList(st_conv1)
        self.parms2 = STGCN(ks,kt,[[6,32,64],[64,32,64]],T,n,Lk,p)#nn.ModuleList(st_conv2)
        self.parms3 = nn.Linear(self.nuniques, 1) # nn.Conv2d(self.nuniques, 1, 1) #nn.ModuleList(output)
        bss[0][0] = self.input_channels

    def forward(self, x):
        # x's shape: [data slots, channels, time steps, nodes]
        xs = list(x.size())
        x_st1 = self.parms1(x[:,0,:,:].view(xs[0],1,xs[2],xs[3]))
        x_st2 = self.parms2(x[:,1:,:,:])
        xr = torch.cat((x_st1,x_st2),1)
        #if xr.nelement() == 6156:
        #    xr = xr
        yy = self.parms3(xr.reshape(xs[0],1,171,2)).cuda()
        yy = yy.reshape(xs[0],1,1,171)
        '''
        #x_st1 = []
        #x_st2 = []
        xs = list(x.size())
        xs[1] = self.nuniques  # self.input_channels    is the number of input channels to the ending output layer, fusing all previous
        xs[2] = 1   # number of time steps after each seperate STGCN
        x_st3 = [] #torch.empty(xs) #np.empty(xs)
        xs = list(x.size())
        for ii in range(self.nuniques):
            rr = np.squeeze(np.where(np.array(self.input_diff) == ii))
            rt = self.input_diff.count(ii)
            xx = torch.reshape(x[:,rr,:,:], (xs[0], rt, xs[2], xs[3]))
            #x_st1.append(self.parms1[ii](xx))
            #x_st2.append(self.parms2[ii](x_st1[ii]))
            #x_st3[:,ii,:,:] = torch.reshape(self.parms3[ii](xx), (xs[0], 1, xs[3]))
            x_st3.append(torch.reshape(self.parms3[ii](xx), (xs[0], 1, 1, xs[3])))
        yy = self.parms3[self.nuniques](torch.cat(x_st3, dim=1).cuda())
        '''
        return yy






# STGCN2 model: ---------------------------------------------------------------------------------


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN2(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, A_hat, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN2, self).__init__()
        self.A_hat = A_hat
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
                               num_timesteps_output)

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.block1(X, self.A_hat)
        out2 = self.block2(out1, self.A_hat)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4





# Graph_Convolutional_LSTM model: ---------------------------------------------------------------------------------

# from py files:
class RNN_from_pyfile(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_from_pyfile, self).__init__()
        
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
#         print(combined)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden
    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            return Variable(torch.zeros(batch_size, self.hidden_size).cuda())
        else:
            return Variable(torch.zeros(batch_size, self.hidden_size))
        
        
class LSTM_from_pyfile(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size):
        """
        cell_size is the size of cell_state.
        hidden_size is the size of hidden_state, or say the output_state of each step
        """
        super(LSTM, self).__init__()
        
        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)
        
    def forward(self, input, Hidden_State, Cell_State):
        combined = torch.cat((input, Hidden_State), 1)
        f = F.sigmoid(self.fl(combined))
        i = F.sigmoid(self.il(combined))
        o = F.sigmoid(self.ol(combined))
        C = F.tanh(self.Cl(combined))
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * F.tanh(Cell_State)
        
        return Hidden_State, Cell_State
    
    def loop(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)
        for i in range(time_step):
            Hidden_State, Cell_State = self.forward(torch.squeeze(inputs[:,i:i+1,:]), Hidden_State, Cell_State)
        return Hidden_State, Cell_State
    
    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State
        
        
class GraphConvolutionalLSTM_from_pyfile(nn.Module):
    
    def __init__(self, K, A, FFR, feature_size, Clamp_A=True):
        '''
        Args:
            K: K-hop graph
            A: adjacency matrix
            FFR: free-flow reachability matrix
            feature_size: the dimension of features
            Clamp_A: Boolean value, clamping all elements of A between 0. to 1.
        '''
        super(GraphConvolutionalLSTM, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size
        
        self.K = K
        
        self.A_list = [] # Adjacency Matrix List
        A = torch.FloatTensor(A)
        A_temp = torch.eye(feature_size,feature_size)
        for i in range(K):
            A_temp = torch.matmul(A_temp, torch.Tensor(A))
            if Clamp_A:
                # confine elements of A 
                A_temp = torch.clamp(A_temp, max = 1.) 
            self.A_list.append(torch.mul(A_temp, torch.Tensor(FFR)))
#             self.A_list.append(A_temp)
        
        # a length adjustable Module List for hosting all graph convolutions
        #self.gc_list = nn.ModuleList([FilterLinear(feature_size, feature_size, self.A_list[i], bias=False) for i in range(K)])
        
        hidden_size = self.feature_size
        input_size = self.feature_size * K

        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)
        
        # initialize the neighbor weight for the cell state
        self.Neighbor_weight = Parameter(torch.FloatTensor(feature_size))
        stdv = 1. / math.sqrt(feature_size)
        self.Neighbor_weight.data.uniform_(-stdv, stdv)
        
    def forward(self, input, Hidden_State, Cell_State):
        
        x = input

        gc = self.gc_list[0](x)
        for i in range(1, self.K):
            gc = torch.cat((gc, self.gc_list[i](x)), 1)
            
        combined = torch.cat((gc, Hidden_State), 1)
        f = F.sigmoid(self.fl(combined))
        i = F.sigmoid(self.il(combined))
        o = F.sigmoid(self.ol(combined))
        C = F.tanh(self.Cl(combined))

        NC = torch.mul(Cell_State,  torch.mv(Variable(self.A_list[-1], requires_grad=False).cuda(), self.Neighbor_weight))
        Cell_State = f * NC + i * C
        Hidden_State = o * F.tanh(Cell_State)
        
        return Hidden_State, Cell_State, gc
    
    def Bi_torch(self, a):
        a[a < 0] = 0
        a[a > 0] = 1
        return a


# from_notebook:
class LSTM(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size, output_last = True):
        """
        cell_size is the size of cell_state.
        hidden_size is the size of hidden_state, or say the output_state of each step
        """
        super(LSTM, self).__init__()
        
        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)
        
        self.output_last = output_last
        
    def step(self, input, Hidden_State, Cell_State):
        combined = torch.cat((input, Hidden_State), 1)
        f = F.sigmoid(self.fl(combined))
        i = F.sigmoid(self.il(combined))
        o = F.sigmoid(self.ol(combined))
        C = F.tanh(self.Cl(combined))
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * F.tanh(Cell_State)
        
        return Hidden_State, Cell_State
    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(2)
        Hidden_State, Cell_State = self.initHidden(batch_size)
        
        if self.output_last:
            for i in range(time_step):
                Hidden_State, Cell_State = self.step(torch.squeeze(inputs[:,0,i:i+1,:]), Hidden_State, Cell_State)
            return Hidden_State
        else:
            outputs = None
            for i in range(time_step):
                Hidden_State, Cell_State = self.step(torch.squeeze(inputs[:,0,i:i+1,:]), Hidden_State, Cell_State)
                if outputs is None:
                    outputs = Hidden_State.unsqueeze(1)
                else:
                    outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)
            return outputs
    
    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State


class ConvLSTM(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size, output_last = True):
        """
        cell_size is the size of cell_state.
        hidden_size is the size of hidden_state, or say the output_state of each step
        """
        super(ConvLSTM, self).__init__()
        
        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)
        
        self.conv = nn.Conv1d(1, hidden_size, hidden_size)   # in_channels, out_channels, kernel_sizeXkernel_size
        
        self.output_last = output_last
        
    def step(self, input, Hidden_State, Cell_State):
        
        conv = torch.squeeze(self.conv(input))  # input=[50,1,171]->[50,171,1]->[50,171] to fit size of Hidden_State, to combine it in next command
        
        combined = torch.cat((conv, Hidden_State), 1)
        f = F.sigmoid(self.fl(combined))
        i = F.sigmoid(self.il(combined))
        o = F.sigmoid(self.ol(combined))
        C = F.tanh(self.Cl(combined))
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * F.tanh(Cell_State)
        
        return Hidden_State, Cell_State
    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(2)
        Hidden_State, Cell_State = self.initHidden(batch_size)
        
        if self.output_last:
            for i in range(time_step):
                # Here we keep the channel dimension, for Conv1D operation that come next, only remove the time dimension
                Hidden_State, Cell_State = self.step(torch.squeeze(inputs[:,:,i:i+1,:],2), Hidden_State, Cell_State)
            return Hidden_State
        else:
            outputs = None
            for i in range(time_step):
                Hidden_State, Cell_State = self.step(torch.squeeze(inputs[:,:,i:i+1,:]), Hidden_State, Cell_State)
                if outputs is None:
                    outputs = Hidden_State.unsqueeze(1)
                else:
                    outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)
            return outputs
    
    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State


class LocalizedSpectralGraphConvolution(nn.Module):
    def __init__(self, A, K):
        
        super(LocalizedSpectralGraphConvolution, self).__init__()
        
        
        self.K = K
        self.A = A#.cuda()
        feature_size = A.shape[0]
        self.D = torch.diag(torch.sum(self.A, dim=0)+torch.sum(self.A, dim=1))#.cuda()
        
        I = torch.eye(feature_size,feature_size)#.cuda()
        inverseD = torch.inverse(torch.sqrt(self.D))
        self.L = I - inverseD.matmul(self.A).matmul(inverseD)
        
        L_temp = I
        for i in range(K):
            L_temp = torch.matmul(L_temp, self.L)
            if i == 0:
                self.L_tensor = torch.unsqueeze(L_temp, 2)
            else:
                self.L_tensor = torch.cat((self.L_tensor, torch.unsqueeze(L_temp, 2)), 2)
            
        self.L_tensor = Variable(self.L_tensor#.cuda()
                                , requires_grad=False)

        self.params = Parameter(torch.FloatTensor(K))#.cuda())
        
        stdv = 1. / math.sqrt(K)
        for i in range(K):
            self.params[i].data.uniform_(-stdv, stdv)

    def forward(self, input):
        x = input

        conv = x.matmul( torch.sum(self.params.expand_as(self.L_tensor) * self.L_tensor, 2) )

        return conv
        
        
class LocalizedSpectralGraphConvolutionalLSTM(nn.Module):
    
    def __init__(self, K, A, feature_size, Clamp_A=True, output_last = True):
        '''
        Args:
            K: K-hop graph
            A: adjacency matrix
            FFR: free-flow reachability matrix
            feature_size: the dimension of features
            Clamp_A: Boolean value, clamping all elements of A between 0. to 1.
        '''
        super(LocalizedSpectralGraphConvolutionalLSTM, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size
        
        self.K = K
        self.A = A
        self.gconv = LocalizedSpectralGraphConvolution(A, K)
    
        hidden_size = self.feature_size
        input_size = self.feature_size + hidden_size

        self.fl = nn.Linear(input_size, hidden_size)
        self.il = nn.Linear(input_size, hidden_size)
        self.ol = nn.Linear(input_size, hidden_size)
        self.Cl = nn.Linear(input_size, hidden_size)
        
        self.output_last = output_last
        
    def step(self, input, Hidden_State, Cell_State):
        
#         conv_sample_start = time.time()  
        conv = F.relu(self.gconv(input))
#         conv_sample_end = time.time()  
#         print('conv_sample:', (conv_sample_end - conv_sample_start))
        combined = torch.cat((conv, Hidden_State), 1)
        f = F.sigmoid(self.fl(combined))
        i = F.sigmoid(self.il(combined))
        o = F.sigmoid(self.ol(combined))
        C = F.tanh(self.Cl(combined))
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * F.tanh(Cell_State)
        
        return Hidden_State, Cell_State
    
    def Bi_torch(self, a):
        a[a < 0] = 0
        a[a > 0] = 1
        return a
    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(2)
        Hidden_State, Cell_State = self.initHidden(batch_size)
        
        outputs = None
        
        for i in range(time_step):
            Hidden_State, Cell_State = self.step(torch.squeeze(inputs[:,0,i:i+1,:]), Hidden_State, Cell_State)

            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)
#         print(type(outputs))
        
        if self.output_last:
            return outputs[:,-1,:]
        else:
            return outputs
    
    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State
    def reinitHidden(self, batch_size, Hidden_State_data, Cell_State_data):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(Hidden_State_data.cuda(), requires_grad=True)
            Cell_State = Variable(Cell_State_data.cuda(), requires_grad=True)
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(Hidden_State_data, requires_grad=True)
            Cell_State = Variable(Cell_State_data, requires_grad=True)
            return Hidden_State, Cell_State


class SpectralGraphConvolution(nn.Module):
    def __init__(self, A):
        
        super(SpectralGraphConvolution, self).__init__()
        
        feature_size = A.shape[0]
        
        self.A = A
        self.D = torch.diag(torch.sum(self.A, dim=0))
        self.L = self.D - A
        self.param = Parameter(torch.FloatTensor(feature_size))#.cuda())
        stdv = 1. / math.sqrt(feature_size)
        self.param.data.uniform_(-stdv, stdv)
        
        # self.e, self.v = torch.eig(self.L, eigenvectors=True)   # L_
        self.e, self.v = torch.linalg.eigh(self.L)
        self.vt = torch.t(self.v)
        self.v = Variable(self.v, requires_grad=False)   # self.v.cuda()
        self.vt = Variable(self.vt, requires_grad=False)  # self.vt.cuda()
    
    def forward(self, input):
        x = input
        #conv_sample_start = time.time()
        conv = x.matmul(self.v.matmul(torch.diag(self.param)).matmul(self.vt))
        #conv_sample_end = time.time()
        #print('conv_sample:', (conv_sample_end - conv_sample_start))
        return conv
        

class SpectralGraphConvolutionalLSTM(nn.Module):
    
    def __init__(self, K, A, feature_size, Clamp_A=True, output_last = True):
        '''
        Args:
            K: K-hop graph
            A: adjacency matrix
            FFR: free-flow reachability matrix
            feature_size: the dimension of features
            Clamp_A: Boolean value, clamping all elements of A between 0. to 1.
        '''
        super(SpectralGraphConvolutionalLSTM, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size
        
        self.K = K
        self.A = A
        self.gconv = SpectralGraphConvolution(A)
    
        hidden_size = self.feature_size
        input_size = self.feature_size + hidden_size

        self.fl = nn.Linear(input_size, hidden_size)
        self.il = nn.Linear(input_size, hidden_size)
        self.ol = nn.Linear(input_size, hidden_size)
        self.Cl = nn.Linear(input_size, hidden_size)
        
        self.output_last = output_last
        
    def step(self, input, Hidden_State, Cell_State):
        #conv_sample_start = time.time()
        conv = self.gconv(input)
        #conv_sample_end = time.time()
        #print('conv_sample:', (conv_sample_end - conv_sample_start))
        combined = torch.cat((conv, Hidden_State), 1)
        f = F.sigmoid(self.fl(combined))
        i = F.sigmoid(self.il(combined))
        o = F.sigmoid(self.ol(combined))
        C = F.tanh(self.Cl(combined))
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * F.tanh(Cell_State)
        
        return Hidden_State, Cell_State
    
    def Bi_torch(self, a):
        a[a < 0] = 0
        a[a > 0] = 1
        return a
    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(2)
        Hidden_State, Cell_State = self.initHidden(batch_size)
        
        outputs = None
        
        #train_sample_start = time.time()
        
        for i in range(time_step):
            Hidden_State, Cell_State = self.step(torch.squeeze(inputs[:,0,i:i+1,:]), Hidden_State, Cell_State)

            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)
        
        #train_sample_end = time.time()
        #print('train sample:' , (train_sample_end - train_sample_start))
        if self.output_last:
            return outputs[:,-1,:]
        else:
            return outputs
    
    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State
    def reinitHidden(self, batch_size, Hidden_State_data, Cell_State_data):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(Hidden_State_data.cuda(), requires_grad=True)
            Cell_State = Variable(Cell_State_data.cuda(), requires_grad=True)
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(Hidden_State_data, requires_grad=True)
            Cell_State = Variable(Cell_State_data, requires_grad=True)
            return Hidden_State, Cell_State


class GraphConvolutionalLSTM(nn.Module):
    
    def __init__(self, K, A, FFR, feature_size, Clamp_A=True, output_last = True):
        '''
        Args:
            K: K-hop graph
            A: adjacency matrix
            FFR: free-flow reachability matrix
            feature_size: the dimension of features
            Clamp_A: Boolean value, clamping all elements of A between 0. to 1.
        '''
        super(GraphConvolutionalLSTM, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size
        
        self.K = K
        
        self.A_list = [] # Adjacency Matrix List
        A = torch.FloatTensor(A)
        A_temp = torch.eye(feature_size,feature_size)
        for i in range(K):
            A_temp = torch.matmul(A_temp, torch.Tensor(A))
            if Clamp_A:
                # confine elements of A 
                A_temp = torch.clamp(A_temp, max = 1.) 
            self.A_list.append(torch.mul(A_temp, torch.Tensor(FFR)))
#             self.A_list.append(A_temp)
        
        # a length adjustable Module List for hosting all graph convolutions
        #self.gc_list = nn.ModuleList([FilterLinear(feature_size, feature_size, self.A_list[i], bias=False) for i in range(K)])
        
        hidden_size = self.feature_size
        input_size = self.feature_size * K

        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)
        
        # initialize the neighbor weight for the cell state
        self.Neighbor_weight = Parameter(torch.FloatTensor(feature_size))
        stdv = 1. / math.sqrt(feature_size)
        self.Neighbor_weight.data.uniform_(-stdv, stdv)
        
        self.output_last = output_last
        
    def step(self, input, Hidden_State, Cell_State):
        
        x = input

        gc = self.gc_list[0](x)
        for i in range(1, self.K):
            gc = torch.cat((gc, self.gc_list[i](x)), 1)
            
        combined = torch.cat((gc, Hidden_State), 1)
        f = F.sigmoid(self.fl(combined))
        i = F.sigmoid(self.il(combined))
        o = F.sigmoid(self.ol(combined))
        C = F.tanh(self.Cl(combined))

        NC = torch.mul(Cell_State,  torch.mv(Variable(self.A_list[-1], requires_grad=False).cuda(), self.Neighbor_weight))
        Cell_State = f * NC + i * C
        Hidden_State = o * F.tanh(Cell_State)

        return Hidden_State, Cell_State, gc
    
    def Bi_torch(self, a):
        a[a < 0] = 0
        a[a > 0] = 1
        return a
    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(2)
        Hidden_State, Cell_State = self.initHidden(batch_size)
        
        outputs = None
        
        for i in range(time_step):
            Hidden_State, Cell_State, gc = self.step(torch.squeeze(inputs[:,0,i:i+1,:]), Hidden_State, Cell_State)

            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)
        
        if self.output_last:
            return outputs[:,-1,:]
        else:
            return outputs
    
    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State
    def reinitHidden(self, batch_size, Hidden_State_data, Cell_State_data):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(Hidden_State_data.cuda(), requires_grad=True)
            Cell_State = Variable(Cell_State_data.cuda(), requires_grad=True)
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(Hidden_State_data, requires_grad=True)
            Cell_State = Variable(Cell_State_data, requires_grad=True)
            return Hidden_State, Cell_State



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_last = True):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.output_last = output_last

    def step(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def forward(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(2)
        Hidden_State = self.initHidden(batch_size)

        if self.output_last:
            for i in range(time_step):
                _ , Hidden_State = self.step(torch.squeeze(inputs[:, 0, i:i + 1, :]), Hidden_State)
            return Hidden_State
        else:
            outputs = None
            for i in range(time_step):
                _ , Hidden_State = self.step(torch.squeeze(inputs[:, 0, i:i + 1, :]), Hidden_State)
                if outputs is None:
                    outputs = Hidden_State.unsqueeze(1)
                else:
                    outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)
            return outputs

    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            return Variable(torch.zeros(batch_size, self.hidden_size).cuda())
        else:
            return Variable(torch.zeros(batch_size, self.hidden_size))




# GGCN model: ---------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, num_vetex, act=F.relu, dropout=0.5, bias=True):
        super(GraphConvolution, self).__init__()

        self.alpha = 1.

        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim)).to(device)
        if bias:
            self.bias = nn.Parameter(torch.randn(output_dim)).to(device)
        else:
            self.bias = None

        for w in [self.weight]:
            nn.init.xavier_normal_(w)

    def normalize(self, m):
        rowsum = torch.sum(m, 0)
        r_inv = torch.pow(rowsum, -0.5)
        r_mat_inv = torch.diag(r_inv).float()

        m_norm = torch.mm(r_mat_inv, m)
        m_norm = torch.mm(m_norm, r_mat_inv)

        return m_norm

    def forward(self, adj, x):

        x = self.dropout(x)

        # K-ordered Chebyshev polynomial
        adj_norm = self.normalize(adj)
        sqr_norm = self.normalize(torch.mm(adj, adj))
        m_norm = self.alpha * adj_norm + (1. - self.alpha) * sqr_norm

        x_tmp = torch.einsum('abcd,de->abce', x, self.weight)
        x_out = torch.einsum('ij,abid->abjd', m_norm, x_tmp)
        if self.bias is not None:
            x_out += self.bias

        x_out = self.act(x_out)

        return x_out


class StandConvolution(nn.Module):
    def __init__(self, dims, num_classes, dropout):
        super(StandConvolution, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=5, stride=2),
            nn.InstanceNorm2d(dims[1]),
            nn.ReLU(inplace=True),
            # nn.AvgPool2d(3, stride=2),
            nn.Conv2d(dims[1], dims[2], kernel_size=5, stride=2),
            nn.InstanceNorm2d(dims[2]),
            nn.ReLU(inplace=True),
            # nn.AvgPool2d(3, stride=2),
            nn.Conv2d(dims[2], dims[3], kernel_size=5, stride=2),
            nn.InstanceNorm2d(dims[3]),
            nn.ReLU(inplace=True),
            # nn.AvgPool2d(3, stride=2)
        ).to(device)

        self.fc = nn.Linear(dims[3] * 3, num_classes).to(device)

    def forward(self, x):
        x = self.dropout(x.permute(0, 3, 1, 2))
        x_tmp = self.conv(x)
        x_out = self.fc(x_tmp.view(x.size(0), -1))

        return x_out


class StandRecurrent(nn.Module):
    def __init__(self, dims, num_classes, dropout):
        super(StandRecurrent, self).__init__()

        self.lstm = nn.LSTM(dims[0] * 45, dims[1], batch_first=True,
                            dropout=0).to(device)
        self.fc = nn.Linear(dims[1], num_classes).to(device)

    def forward(self, x):
        x_tmp, _ = self.lstm(x.contiguous().view(x.size(0), x.size(1), -1))
        x_out = self.fc(x_tmp[:, -1])

        return x_out


class GGCN(nn.Module):
    def __init__(self, adj, num_v, num_classes, gc_dims, sc_dims, feat_dims, dropout=0.5):
        super(GGCN, self).__init__()

        terminal_cnt = 5
        actor_cnt = 1
        adj = adj + torch.eye(adj.size(0)).to(adj).detach()
        ident = torch.eye(adj.size(0)).to(adj)
        zeros = torch.zeros(adj.size(0), adj.size(1)).to(adj)
        self.adj = torch.cat([torch.cat([adj, ident, zeros], 1),
                              torch.cat([ident, adj, ident], 1),
                              torch.cat([zeros, ident, adj], 1)], 0).float()
        self.terminal = nn.Parameter(torch.randn(terminal_cnt, actor_cnt, feat_dims))

        self.gcl = GraphConvolution(gc_dims[0] + feat_dims, gc_dims[1], num_v, dropout=dropout)
        self.conv = StandConvolution(sc_dims, num_classes, dropout=dropout)

        nn.init.xavier_normal_(self.terminal)

    def forward(self, x):
        head_la = F.interpolate(torch.stack([self.terminal[0], self.terminal[1]], 2), 6)
        head_ra = F.interpolate(torch.stack([self.terminal[0], self.terminal[2]], 2), 6)
        lw_ra = F.interpolate(torch.stack([self.terminal[3], self.terminal[4]], 2), 6)
        node_features = torch.cat([
            (head_la[:, :, :3] + head_ra[:, :, :3]) / 2,
            torch.stack((lw_ra[:, :, 2], lw_ra[:, :, 1], lw_ra[:, :, 0]), 2),
            lw_ra[:, :, 3:], head_la[:, :, 3:], head_ra[:, :, 3:]], 2).to(x)
        x = torch.cat((x, node_features.permute(0, 2, 1).unsqueeze(1).repeat(1, 32, 1, 1)), 3)

        concat_seq = torch.cat([x[:, :-2], x[:, 1:-1], x[:, 2:]], 2)  # 1, 30, 45, 3
        multi_conv = self.gcl(self.adj, concat_seq)
        logit = self.conv(multi_conv)

        return logit



# Graph Auto-Encoder models (VGAE, ARGA): ---------------------------------------------------------------------------------

EPS = 1e-15
MAX_LOGSTD = 10


class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper
    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})
    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    def forward(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class GAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.
    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder, decoder=None):
        super(GAE, self).__init__()
        self.encoder = encoder      # some custom defined function
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)

        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1)) # size is like shape, You can also write tensor.size(i) to access a single dimension, which is equivalent to but preferred over tensor.sizes()[i]
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)  # is 1,1..1,0,0...0 as the GT of all edges

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


class VGAE(GAE):
    r"""The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.
    Args:
        encoder (Module): The encoder module to compute :math:`\mu` and
            :math:`\log\sigma^2`.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder, decoder=None):
        super(VGAE, self).__init__(encoder, decoder)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs):
        """"""
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu=None, logstd=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.
        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))


class ARGA(GAE):
    r"""The Adversarially Regularized Graph Auto-Encoder model from the
    `"Adversarially Regularized Graph Autoencoder for Graph Embedding"
    <https://arxiv.org/abs/1802.04407>`_ paper.
    paper.
    Args:
        encoder (Module): The encoder module.
        discriminator (Module): The discriminator module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder, discriminator, decoder=None):
        super(ARGA, self).__init__(encoder, decoder)
        self.discriminator = discriminator
        reset(self.discriminator)

    def reset_parameters(self):
        super(ARGA, self).reset_parameters()
        reset(self.discriminator)

    def reg_loss(self, z):
        r"""Computes the regularization loss of the encoder.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
        """
        real = torch.sigmoid(self.discriminator(z))
        real_loss = -torch.log(real + EPS).mean()
        return real_loss

    def discriminator_loss(self, z):
        r"""Computes the loss of the discriminator.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
        """
        real = torch.sigmoid(self.discriminator(torch.randn_like(z)))
        fake = torch.sigmoid(self.discriminator(z.detach()))
        real_loss = -torch.log(real + EPS).mean()
        fake_loss = -torch.log(1 - fake + EPS).mean()
        return real_loss + fake_loss


class ARGVA(ARGA):
    r"""The Adversarially Regularized Variational Graph Auto-Encoder model from
    the `"Adversarially Regularized Graph Autoencoder for Graph Embedding"
    <https://arxiv.org/abs/1802.04407>`_ paper.
    paper.
    Args:
        encoder (Module): The encoder module to compute :math:`\mu` and
            :math:`\log\sigma^2`.
        discriminator (Module): The discriminator module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder, discriminator, decoder=None):
        super(ARGVA, self).__init__(encoder, discriminator, decoder)
        self.VGAE = VGAE(encoder, decoder)

    @property
    def __mu__(self):
        return self.VGAE.__mu__

    @property
    def __logstd__(self):
        return self.VGAE.__logstd__

    def reparametrize(self, mu, logstd):
        return self.VGAE.reparametrize(mu, logstd)

    def encode(self, *args, **kwargs):
        """"""
        return self.VGAE.encode(*args, **kwargs)

    def kl_loss(self, mu=None, logstd=None):
        return self.VGAE.kl_loss(mu, logstd)


# Encoder for GAE or VGAE
class GAE_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(GAE_Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.name = name
        if name=='GAE':
            self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)
        elif name=='VGAE':
            self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
            self.conv_logstd = GCNConv(2 * out_channels, out_channels,
                                       cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        if self.name=='GAE':
            return self.conv2(x, edge_index)
        elif self.name=='VGAE':
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)




# Graph U-Net model: ----------------------------------------------------------------------------------


class GUNet(torch.nn.Module):
    def __init__(self, edge_index, n_route, n_his, n_pred_seq):
        super(GUNet, self).__init__()
        pool_ratios = [0.75, 0.5]
        self.unet = GraphUNet(n_his, 32, 1, depth=3, pool_ratios=pool_ratios)
        self.edge_index = edge_index
        self.n_route = n_route
    def forward(self, x):
        xx = x.squeeze()
        if xx.ndim == 1:
            xx = xx.reshape((xx.shape[0],1))
        edge_index1, _ = dropout_adj(self.edge_index, p=0.2, force_undirected=True, num_nodes=self.n_route, training=self.training)
        if xx.shape[0] > 1:
            xx = F.dropout(xx, p=0.22, training=self.training)  # p=0.92 , 0.22
        xx = self.unet(xx, edge_index1)
        return xx.transpose(0,1) #F.log_softmax(xx, dim=1)


class GraphUNet(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """

    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        super(GraphUNet, self).__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = act
        self.sum_res = sum_res

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels, improved=True))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(GCNConv(in_channels, channels, improved=True))
        self.up_convs.append(GCNConv(in_channels, out_channels, improved=True))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        return x

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)



# SAGEConv layer: ---------------------------------------------------------------------------------

class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 bias: bool = True, **kwargs):  # yapf: disable
        super(SAGEConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out


    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


# SGConv layer: ---------------------------------------------------------------------------------

class SGConv(MessagePassing):
    r"""The simple graph convolutional operator from the `"Simplifying Graph
    Convolutional Networks" <https://arxiv.org/abs/1902.07153>`_ paper

    .. math::
        \mathbf{X}^{\prime} = {\left(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int, optional): Number of hops :math:`K`. (default: :obj:`1`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X}` on
            first execution, and will use the cached version for further
            executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_x: Optional[Tensor]

    def __init__(self, in_channels: int, out_channels: int, K: int = 1,
                 cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        super(SGConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._cached_x = None

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self._cached_x = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        cache = self._cached_x
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)

            for k in range(self.K):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
                if self.cached:
                    self._cached_x = x
        else:
            x = cache

        return self.lin(x)


    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.K)



# for AGE model: -----------------------------------------------------------------------

# AGE layers:
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SampleDecoder(Module):
    def __init__(self, act=torch.sigmoid):
        super(SampleDecoder, self).__init__()
        self.act = act

    def forward(self, zx, zy):
        sim = (zx * zy).sum(1)
        sim = self.act(sim)
    
        return sim

# AGE model:

class LinTrans(nn.Module):
    def __init__(self, layers, dims):
        super(LinTrans, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        self.dcs = SampleDecoder(act=lambda x: x)

    def scale(self, z):
        
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
    
        return z_scaled

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.scale(out)
        out = F.normalize(out)
        return out

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)


    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret





# TGCN model (https://github.com/cassianobecker/tgcn): -------------------------------------------------------------------


class TGCNCheb(torch.nn.Module):

    def __init__(self, L, in_channels, out_channels, filter_order, bias=True):
        super(TGCNCheb, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(filter_order, in_channels, out_channels)) # tensor of dimensions k x f x g

        # ADD LAPLACIAN AS A MEMBER VARIABLE
        self.L = L
        self.filter_order = filter_order

        if bias:
            self.bias = Parameter(torch.Tensor(1, L[0].shape[0], out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)


    def forward(self, x):
        """"""
        # Perform filter operation recurrently.

        xc = self._time_chebyshev(x)
        out = torch.einsum("kqnf,kfg->qng", xc, self.weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, filter_order={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))


    def _time_chebyshev(self, X):
        """Return T_k X where T_k are the Chebyshev polynomials of order up to filter_order.
        Complexity is O(KMN).
        self.L: m x n laplacian
        X: q (# examples) x n (vertex count of graph) x f (number of input filters)
        Xt: tensor of dims k (order of chebyshev polynomials) x q x n x f
        """

        #if len(list(X.shape)) == 2:
        #    X = X.unsqueeze(2)

        dims = list(X.shape)
        dims = tuple([self.filter_order] + dims)

        Xt = torch.empty(dims)

        Xt[0, ...] = X

        # Xt_1 = T_1 X = L X.
        if self.filter_order > 1:
            X = torch.einsum("nm,qmf->qnf", self.L, X.float())
            Xt[1, ...] = X
        # Xt_k = 2 L Xt_k-1 - Xt_k-2.
        for k in range(2, self.filter_order):
            #X = Xt[k - 1, ...]
            X = torch.einsum("nm,qmf->qnf", self.L, X.float())
            Xt[k, ...] = 2 * X - Xt[k - 2, ...]
        return Xt


class TGCNCheb_H(torch.nn.Module):

    def __init__(self, L, in_channels, out_channels, filter_order, horizon, bias=True):  # horizon=prediction horizon
        super(TGCNCheb_H, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(filter_order, horizon, in_channels, out_channels)) # tensor of dimensions k x f x g

        # ADD LAPLACIAN AS A MEMBER VARIABLE
        self.L = L
        self.filter_order = filter_order

        if bias:
            self.bias = Parameter(torch.Tensor(1, L[0].shape[0], out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)


    def forward(self, x):
        """"""
        # Perform filter operation recurrently.

        xc = self._time_chebyshev(x)
        out = torch.einsum("kqnhf,khfg->qng", xc, self.weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, filter_order={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))


    def _time_chebyshev(self, X):
        """Return T_k X where T_k are the Chebyshev polynomials of order up to filter_order.
        Complexity is O(KMN).
        self.L: m x n Laplacian
        X: q (# examples) x n (vertex count of graph) x f (number of input filters)
        Xt: tensor of dims k (order of chebyshev polynomials) x q x n x f
        """

        if len(list(X.shape)) == 3:
            X = X.unsqueeze(3)

        dims = list(X.shape)
        dims = tuple([self.filter_order] + dims)

        Xt = torch.empty(dims).to(X.device)
        L = self.L.to(X.device)
        L = L.double()

        Xt[0, ...] = X

        # Xt_1 = T_1 X = L X.
        if self.filter_order > 1:
            X = torch.einsum("nm,qmhf->qnhf", L, X)
            Xt[1, ...] = X
        # Xt_k = 2 L Xt_k-1 - Xt_k-2.
        for k in range(2, self.filter_order):
            #X = Xt[k - 1, ...]
            X = torch.einsum("nm,qmhf->qnhf", L, X)
            Xt[k, ...] = 2 * X - Xt[k - 2, ...]
        return Xt



class GCNCheb(torch.nn.Module):

    def __init__(self, L, in_channels, out_channels, filter_order, bias=True):
        super(GCNCheb, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(filter_order, in_channels, out_channels)) # tensor of dimensions k x f x g

        # ADD LAPLACIAN AS A MEMBER VARIABLE
        self.L = L
        self.filter_order = filter_order

        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)
        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #     #bound = 1 / math.sqrt(fan_in)
        #     #init.uniform_(self.bias, -bound, bound)
        #     uniform(fan_in, self.bias)


    def forward(self, x):
        """"""
        # Perform filter operation recurrently.

        xc = self._chebyshev(x)
        out = torch.einsum("kqnf,kfg->qng", xc, self.weight)
        #b = self.bias.cpu().detach().numpy()

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, filter_order={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))


    def _chebyshev(self, X):
        """Return T_k X where T_k are the Chebyshev polynomials of order up to filter_order.
        Complexity is O(KMN).
        self.L: m x n laplacian
        X: q (# examples) x n (vertex count of graph) x f (number of input filters)
        Xt: tensor of dims k (order of chebyshev polynomials) x q x n x f
        """

        if len(list(X.shape)) == 2:
            X = X.unsqueeze(2)

        dims = list(X.shape)
        dims = tuple([self.filter_order] + dims)

        Xt = torch.empty(dims, dtype=torch.float).to(X.device)
        L = self.L.to(X.device)

        Xt[0, ...] = X

        # Xt_1 = T_1 X = L X.
        # L = torch.Tensor(self.L)
        if self.filter_order > 1:
            X = torch.einsum("nm,qmf->qnf", L, X)
            Xt[1, ...] = X
        # Xt_k = 2 L Xt_k-1 - Xt_k-2.
        for k in range(2, self.filter_order):
            #X = Xt[k - 1, ...]
            X = torch.einsum("nm,qmf->qnf", L, X)
            Xt[k, ...] = 2 * X - Xt[k - 2, ...]
        return Xt


def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def gcn_pool(x):
    x = torch.reshape(x, [x.shape[0], int(x.shape[1] / 2), 2, x.shape[2]])
    x = torch.max(x, dim=2)[0]
    return x


def gcn_pool_4(x):
    x = torch.reshape(x, [x.shape[0], int(x.shape[1] / 4), 4, x.shape[2]])
    x = torch.max(x, dim=2)[0]
    return x


def spmm(index, value, m, matrix):
    """Matrix product of sparse matrix with dense matrix.
    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of sparse matrix.
        matrix (:class:`Tensor`): The dense matrix.
    :rtype: :class:`Tensor`
    """

    row, col = index
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix[col]
    out = out.permute(-1, 0) * value
    out = out.permute(-1, 0)
    out = scatter_add(out, row, dim=0, dim_size=m)

    return out


def spmm_batch_2(index, value, m, matrix):
    """Matrix product of sparse matrix with dense matrix.
    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of sparse matrix.
        matrix (:class:`Tensor`): The dense matrix.
    :rtype: :class:`Tensor`
    """

    row, col = index
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix[:, col]
    try:
        sh = out.shape[2]
    except:
        out = out.unsqueeze(-1)
        sh = 1

    #out = out.permute(1, 2, 0)
    #out = torch.mul(out, value.repeat(-1, sh))
    #out = out.permute(1, 2, 0)
    temp = value.expand(sh, value.shape[0]).permute(1, 0)
    out = torch.einsum("abc,bc->abc", out, temp)
    out = scatter_add(out, row, dim=1, dim_size=m)

    return out


def spmm_batch_3(index, value, m, matrix):
    """Matrix product of sparse matrix with dense matrix.
    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of sparse matrix.
        matrix (:class:`Tensor`): The dense matrix.
    :rtype: :class:`Tensor`
    """

    row, col = index
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix[:, col]
    try:
        sh = out.shape[3:]
        sh = matrix.shape[2:]
    except:
        out = out.unsqueeze(-1)
        sh = matrix.shape[2:]

    #out = out.permute(1, 2, 0)
    #out = torch.mul(out, value.repeat(-1, sh))
    #out = out.permute(1, 2, 0)
    sh = sh + (value.shape[0],)
    temp = value.expand(sh)
    temp = temp.permute(2, 0, 1)
    out = torch.einsum("abcd,bcd->abcd", out, temp)
    out = scatter_add(out, row, dim=1, dim_size=m)

    return out


class ChebConv(torch.nn.Module):
    r"""The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper
    .. math::
        \mathbf{X}^{\prime} = \sum_{k=0}^{K-1} \mathbf{\hat{X}}_k \cdot
        \mathbf{\Theta}_k
    where :math:`\mathbf{\hat{X}}_k` is computed recursively by
    .. math::
        \mathbf{\hat{X}}_0 &= \mathbf{X}
        \mathbf{\hat{X}}_1 &= \mathbf{\hat{L}} \cdot \mathbf{X}
        \mathbf{\hat{X}}_k &= 2 \cdot \mathbf{\hat{L}} \cdot
        \mathbf{\hat{X}}_{k-1} - \mathbf{\hat{X}}_{k-2}
    and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size, *i.e.* number of hops.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, K, bias=True):
        super(ChebConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        row, col = edge_index
        num_nodes, num_edges, K = x.size(1), row.size(0), self.weight.size(0)

        if edge_weight is None:
            edge_weight = x.new_ones((num_edges, ))
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        deg = degree(row, num_nodes, dtype=x.dtype)

        # Compute normalized and rescaled Laplacian.
        deg = deg.pow(-0.5)
        deg[deg == float('inf')] = 0
        lap = -deg[row] * edge_weight * deg[col]

        # Perform filter operation recurrently.
        if len(x.shape) < 3:
            Tx_0 = x.unsqueeze(-1)
        else:
            Tx_0 = x
        out = torch.matmul(Tx_0, self.weight[0])

        if K > 1:
            Tx_1 = spmm_batch_2(edge_index, lap, num_nodes, Tx_0)
            #Tx_1 = torch.stack([spmm(edge_index, lap, num_nodes, Tx_0[i]) for i in range(x.shape[0])])
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, K):
            temp = spmm_batch_2(edge_index, lap, num_nodes, Tx_1)
            #temp = torch.stack([spmm(edge_index, lap, num_nodes, Tx_1[i]) for i in range(x.shape[0])])
            Tx_2 = 2 * temp - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))


class ChebTimeConv(torch.nn.Module):
    r"""The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper
    .. math::
        \mathbf{X}^{\prime} = \sum_{k=0}^{K-1} \mathbf{\hat{X}}_k \cdot
        \mathbf{\Theta}_k
    where :math:`\mathbf{\hat{X}}_k` is computed recursively by
    .. math::
        \mathbf{\hat{X}}_0 &= \mathbf{X}
        \mathbf{\hat{X}}_1 &= \mathbf{\hat{L}} \cdot \mathbf{X}
        \mathbf{\hat{X}}_k &= 2 \cdot \mathbf{\hat{L}} \cdot
        \mathbf{\hat{X}}_{k-1} - \mathbf{\hat{X}}_{k-2}
    and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size, *i.e.* number of hops.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, K, H, bias=True):
        super(ChebTimeConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(K, H, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        row, col = edge_index
        num_nodes, num_edges, K = x.size(1), row.size(0), self.weight.size(0)

        if edge_weight is None:
            edge_weight = x.new_ones((num_edges, ))
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        deg = degree(row, num_nodes, dtype=x.dtype)

        # Compute normalized and rescaled Laplacian.
        deg = deg.pow(-0.5)
        deg[deg == float('inf')] = 0
        lap = -deg[row] * edge_weight * deg[col]

        # Perform filter operation recurrently.
        if len(x.shape) < 4:
            Tx_0 = x.unsqueeze(-1)
        else:
            Tx_0 = x
        #out = torch.matmul(Tx_0, self.weight[0])

        out = torch.einsum("qnhf,hfg->qng", Tx_0, self.weight[0])
        if K > 1:
            Tx_1 = spmm_batch_3(edge_index, lap, num_nodes, Tx_0)
            out = out + torch.einsum("qnhf,hfg->qng", Tx_1, self.weight[1])

        for k in range(2, K):
            temp = spmm_batch_3(edge_index, lap, num_nodes, Tx_1)
            Tx_2 = 2 * temp - Tx_0
            out = out + torch.einsum("qnhf,hfg->qng", Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.weight.size(0))

# GCNs:
class NetGCN1(nn.Module):  # NetGCNBasic

    def __init__(self, L):

        # f: number of input filters
        # g: number of output filters
        # k: order of chebyshev polynomial
        # c: number of classes
        # n: number of vertices at coarsening level
        super(NetGCN1, self).__init__()

        f1, g1, k1 = 1, 10, 25
        self.gcn1 = GCNCheb(L[0], f1, g1, k1)   # originally L[0], but that since there calculated different graphs coarsening..

        n1 = L[0].shape[0]  # originally L[0], same reason above..
        d = 10
        self.fc1 = nn.Linear(n1 * g1, d)

    def forward(self, x):

        x = self.gcn1(x)
        x = F.relu(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return x #F.log_softmax(x, dim=1)


class NetGCN2(nn.Module):  # NetGCN2Layer

    def __init__(self, L):

        # f: number of input filters
        # g: number of output filters
        # k: order of chebyshev polynomial
        # c: number of classes
        # n: number of vertices at coarsening level
        super(NetGCN2, self).__init__()

        f1, g1, k1 = 1, 32, 25
        self.gcn1 = GCNCheb(L[0], f1, g1, k1)

        f2, g2, k2 = g1, 64, 25
        self.gcn2 = GCNCheb(L[2], f2, g2, k2)

        n2 = L[2].shape[0]
        d = 512
        self.fc1 = nn.Linear(int(n2 * g2 /4), d)

        #self.drop = nn.Dropout(0)

        c = 10
        self.fc2 = nn.Linear(d, c)

    def forward(self, x):

        x = self.gcn1(x)
        x = F.relu(x)
        x = gcn_pool_4(x)

        x = self.gcn2(x)
        x = F.relu(x)
        x = gcn_pool_4(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.drop(x)
        x = self.fc2(x)

        return x #F.log_softmax(x, dim=1)


class NetGCN3(nn.Module):   # NetGCN3Layer

    def __init__(self, L):

        # f: number of input filters
        # g: number of output filters
        # k: order of chebyshev polynomial
        # c: number of classes
        # n: number of vertices at coarsening level
        super(NetGCN3, self).__init__()

        f1, g1, k1 = 1, 32, 25
        self.gcn1 = GCNCheb(L[0], f1, g1, k1)

        self.drop1 = nn.Dropout(0.1)

        f2, g2, k2 = g1, 64, 25
        self.gcn2 = GCNCheb(L[2], f2, g2, k2)
        self.dense1_bn = nn.BatchNorm1d(50)

        f3, g3, k3 = g2, 64, 25
        self.gcn3 = GCNCheb(L[4], f3, g3, k3)
        self.dense1_bn = nn.BatchNorm1d(50)

        n3 = L[2].shape[0]
        d = 512
        self.fc1 = nn.Linear(int(n3 * g3 /4), d)

        self.dense1_bn = nn.BatchNorm1d(d)
        self.drop2 = nn.Dropout(0.5)

        c = 10
        self.fc2 = nn.Linear(d, c)


    def forward(self, x):

        x = self.gcn1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = gcn_pool_4(x)
        x = self.gcn2(x)
        x = F.relu(x)
        x = gcn_pool_4(x)
        x = self.gcn3(x)
        x = F.relu(x)
        #x = gcn_pool_4(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.dense1_bn(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc2(x)

        return x #F.log_softmax(x, dim=1)


class NetGCN4(nn.Module):  # NetGCN3

    def __init__(self, L):

        # f: number of input filters
        # g: number of output filters
        # k: order of chebyshev polynomial
        # c: number of classes
        # n: number of vertices at coarsening level

        super(NetGCN4, self).__init__()

        f1, g1, k1 = 1, 30, 25
        self.gcn1 = GCNCheb(L[0], f1, g1, k1)

        f2, g2, k2 = g1, 20, 25
        self.gcn2 = GCNCheb(L[0], f2, g2, k2)

        f3, g3, k3 = g2, 10, 25
        self.gcn3 = GCNCheb(L[0], f3, g3, k2)

        n3 = L[0].shape[0]
        d = 500
        self.fc1 = nn.Linear(n3 * g3, d)

        c = 10
        self.fc2 = nn.Linear(d, c)



    def forward(self, x):

        x = self.gcn1(x)
        x = F.relu(x)
        # x = gcn_pool(x)
        x = self.gcn2(x)
        x = F.relu(x)
        x = self.gcn3(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x #F.log_softmax(x, dim=1)


# TGCNs:
class NetTGCN1(nn.Module):

    def __init__(self, L, n_route, f1 = 1, g1 = 15, k1 = 10, h1 = 12):
        super(NetTGCN1, self).__init__()

        # f: number of input filters
        # g: number of output layers
        # k: order of chebyshev polynomials
        # c: number of classes
        # n: number of vertices at coarsening level


        #f1, g1, k1, h1 = 1, 15, 10, 12
        self.tgcn1 = TGCNCheb_H(L, f1, g1, k1, h1) # originally L[0], but that since there calculated different graphs coarsening..

        n1 = L.shape[0] # originally L[0], same reason above..
        c = n_route #10
        self.fc1 = nn.Linear(n1 * g1, c)


    def forward(self, x):
        x = torch.tensor(npa.real(npa.fft.fft(x.to('cpu').numpy(), axis=2))).to('cuda')
        x = self.tgcn1(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x #F.log_softmax(x, dim=1)



class NetTGCN2(nn.Module):

    def __init__(self, L, n_route):
        super(NetTGCN2, self).__init__()
        # f: number of input filters
        # g: number of output layers
        # k: order of chebyshev polynomials
        # c: number of classes
        # n: number of vertices at coarsening level

        f1, g1, k1, h1 = 1, 32, 10, 15
        self.tgcn1 = TGCNCheb_H(L[0], f1, g1, k1, h1)

        self.drop1 = nn.Dropout(0.1)

        g2, k2 = 64, 10
        self.gcn2 = GCNCheb(L[2], g1, g2, k2)

        #g3, k3 = 32, 10
        #self.gcn3 = GCNCheb(L[4], g2, g3, k3)

        self.dense1_bn = nn.BatchNorm1d(50)
        # n1 = L[0].shape[0]
        # # n2 = L[0].shape[0]
        # c = 6
        # self.fc1 = nn.Linear(n1 * g1, c)

        n2 = L[2].shape[0]
        c = 200
        self.fc1 = nn.Linear(int(n2 * g2 / 4), c)

        self.dense1_bn = nn.BatchNorm1d(c)
        self.drop2 = nn.Dropout(0.5)

        d = n_route  # 6
        self.fc2 = nn.Linear(c, d)


    def forward(self, x):
        #x = torch.tensor(npa.real(npa.fft.fft(x.to('cpu').numpy(), axis=2))).to('cuda')
        x = torch.rfft(x, signal_ndim=1, onesided=False)[:, :, :, 0].to('cuda')
        x = self.tgcn1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = gcn_pool_4(x)

        x = self.gcn2(x)
        x = F.relu(x)
        x = gcn_pool_4(x)

        #x = self.gcn3(x)
        #x = F.relu(x)

        # x = self.dense1_bn(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        x = self.dense1_bn(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc2(x)

        return x #F.log_softmax(x, dim=1)


class NetTGCN3(torch.nn.Module):
    def __init__(self, graphs, coos, n_route):
        super(NetTGCN3, self).__init__()

        f1, g1, k1, h1 = 1, 32, 25, 15
        self.conv1 = ChebTimeConv(f1, g1, K=k1, H=h1)

        #self.drop1 = nn.Dropout(0.1)

        g2, k2 = 64, 25
        self.conv2 = ChebConv(g1, g2, K=k2)

        n2 = graphs[0].shape[0]

        c = 512
        self.fc1 = torch.nn.Linear(int(n2 * g2), c)

        #self.dense1_bn = nn.BatchNorm1d(d)
        #self.drop2 = nn.Dropout(0.5)

        d = n_route # 6
        self.fc2 = torch.nn.Linear(c, d)

        self.coos = coos

    def forward(self, x):
        x = torch.tensor(npa.real(npa.fft.fft(x.to('cpu').numpy(), axis=2))).to('cuda')
        x, edge_index = x, self.coos[0].to(x.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = gcn_pool_4(x)

        #x = self.drop1(x)

        edge_index = self.coos[0].to(x.device)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = gcn_pool_4(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        #x = self.dense1_bn(x)
        #x = F.relu(x)
        #x = self.drop2(x)

        x = self.fc2(x)
        return x #F.log_softmax(x, dim=1)



class NetTGCN4(torch.nn.Module):
    def __init__(self, graphs, coos, n_route):
        super(NetTGCN4, self).__init__()

        f1, g1, k1, h1 = 1, 32, 25, 15
        self.conv1 = ChebTimeConv(f1, g1, K=k1, H=h1)

        #self.drop1 = nn.Dropout(0.1)

        #g2, k2 = 64, 10
        #self.conv2 = ChebConv(g1, g2, K=k2)

        n2 = graphs[0].shape[0]

        c = 512
        self.fc1 = torch.nn.Linear(int(n2 * g1), c)

        #self.dense1_bn = nn.BatchNorm1d(d)
        #self.drop2 = nn.Dropout(0.5)

        d = n_route # 6
        self.fc2 = torch.nn.Linear(c, d)

        self.coos = coos

    def forward(self, x):
        x = torch.tensor(npa.real(npa.fft.fft(x.to('cpu').numpy(), axis=2))).to('cuda')
        x, edge_index = x, self.coos[0].to(x.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = gcn_pool_4(x)

        #x = self.drop1(x)

        #edge_index = self.coos[0].to(x.device)
        #x = self.conv2(x, edge_index)
        #x = F.relu(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        #x = self.dense1_bn(x)
        #x = F.relu(x)
        #x = self.drop2(x)

        x = self.fc2(x)
        return x #F.log_softmax(x, dim=1)



# GAT and GCN models: ------------------------------------------------------------------------------------------

class GATNet(torch.nn.Module):
    def __init__(self, n_his, edge_index):
        super(GATNet, self).__init__()
        self.edge_index = edge_index
        self.conv1 = GATConv(n_his, 8, heads=8, dropout=0.6)   # n_his act here as a #features since it has no spati o-temporal input
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, 1, heads=1, concat=False, dropout=0.6)
    def forward(self, x):
        x = x.squeeze() # to remove #features dimension, leave only [batch=1, inputs=#nodes, channels=time steps]
        xx = F.dropout(x, p=0.6, training=self.training)
        xx = F.elu(self.conv1(xx, self.edge_index))
        xx = F.dropout(xx, p=0.6, training=self.training)
        xx = self.conv2(xx, self.edge_index)
        # transpose since it sees channels in different index
        return xx.transpose(0,1) #F.log_softmax(x, dim=1)


class GCNNet(torch.nn.Module):
    def __init__(self, n_his, edge_index, edge_weight):
        super(GCNNet, self).__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.conv1 = GCNConv(n_his, 16, cached=True, normalize=True)   # n_his act here as a #features since it has no spati o-temporal input
        self.conv2 = GCNConv(16, 1, cached=True, normalize=True)
        # self.conv1 = ChebConv(n_his, 16, K=2)
        # self.conv2 = ChebConv(16, n_his, K=2)
    def forward(self, x):
        x = x.squeeze()
        xx = F.relu(self.conv1(x, self.edge_index, self.edge_weight))
        xx = F.dropout(xx, training=self.training)
        xx = self.conv2(xx, self.edge_index, self.edge_weight)
        return xx.transpose(0,1) #F.log_softmax(x, dim=1)






#------------------- Graph Transformer ----------------------------------


class GraphTransformerNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, edge_attr, heads=2, num_layers=2, dropout=0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout, edge_dim=1))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout, edge_dim=1))
        self.convs.append(TransformerConv(hidden_channels, out_channels, heads=1, concat=False, edge_dim=1))  # final layer
        self.edge_index, self.edge_attr = edge_index, edge_attr

    def forward(self, x):#, edge_index, edge_weight=None):
        # x shape: [batch_size, 1, in_channels, num_nodes]
        x = x.squeeze(1)  # -> [batch_size, in_channels, num_nodes]
        x = x.permute(0, 2, 1)  # -> [batch_size, num_nodes, in_channels]
        x = x.reshape(-1, x.shape[-1])  # -> [num_nodes * batch_size, in_channels]
        for conv in self.convs[:-1]:
            x = conv(x, self.edge_index, edge_attr=self.edge_attr)
            x = F.relu(x)
        x = self.convs[-1](x, self.edge_index, edge_attr=self.edge_attr)
        return x