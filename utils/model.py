#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

from .attentions import *


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GraphOperate(nn.Module):

    def __init__(self, in_d, out_d, bias=True, node_n=48, paraminit='xavier', J=1):
        super(GraphOperate, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.weight = Parameter(torch.FloatTensor(J, in_d, out_d))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(J, out_d))
        else:
            self.register_parameter('bias', None)
        
        if paraminit=='uniform':
            self.reset_parameters_uniform()
        elif paraminit=='xavier':
            self.reset_parameters_xavier()
        elif paraminit=='kaiming':
            self.reset_parameters_kaiming()
        else:
            raise ValueError('No such initialization')

    def reset_parameters_kaiming(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def reset_parameters_uniform(self):
        bound = 1. / math.sqrt(self.weight.size(1) * self.weight.size(2))
        self.weight.data.uniform_(-bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def reset_parameters_xavier(self):
        bound = 1. / math.sqrt(self.weight.size(1) * self.weight.size(2))
        nn.init.xavier_normal_(self.weight, gain=1)
        if self.bias is not None:
            self.bias.data.uniform_(-bound, bound)

    def forward(self, x, adj):
        support = torch.einsum('bkwd,jdf->bkwjf', (x, self.weight))
        if self.bias is not None:
            support = torch.add(support, self.bias)
        support = support.transpose(2,3)
        b, k, j, v, d = support.shape
        output = torch.einsum('jvw,bkjwf->bkjvf', (adj, support))
        output = output.contiguous().view(b,-1,v,d)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_d) + ' -> ' \
               + str(self.out_d) + ')'
        

class GraphConvScatter(nn.Module):

    def __init__(self, in_d, out_d, bias=True, node_n=48, J=1):
        super(GraphConvScatter, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.J = J

        self.feat_gcn = GraphOperate(in_d, out_d, bias=bias, paraminit='xavier', J=1)
        self.feat_sc1 = GraphOperate(in_d, out_d, bias=bias, paraminit='xavier', J=J)
        self.feat_sc2 = GraphOperate(in_d, out_d, bias=bias, paraminit='xavier', J=J)
        
        self.f_coef = nn.Parameter(torch.rand(J*2)*1e-4)

        self.act = nn.Tanh()

    #def diffWavelets(self, W, J, _node_n):
     #   I = torch.eye(_node_n).cuda()
     #   A = W / (torch.norm(W, p='fro') + 1e-3).detach()
     #   P = 1/2 * (I + A)
     #   I = I.detach()
     #   H = (I - P).reshape(1, _node_n, _node_n)
     #   for j in range(1, J):
     #       if j==1:
     #           powerP = P
     #       else:
     #           powerP = torch.matmul(powerP, powerP)
     #       thisH = torch.matmul(powerP, I-powerP)
     #       H = torch.cat((H, thisH.reshape(1, _node_n, _node_n)), dim=0)
            
     #   return H
      
    def diffWavelets(self, W, J, _node_n):
        I = torch.eye(_node_n).cuda()
        A = W / (torch.norm(W, p='fro') + 1e-3).detach()
        P = 1/2 * (I + A)
        I = I.detach()
        H = ((1+self.f_coef[0])*I - (1-self.f_coef[1])*P).reshape(1, _node_n, _node_n)
        for j in range(1, J):
            if j==1:
                powerP = P
            else:
                powerP = torch.matmul(powerP, powerP)
            thisH = torch.matmul(powerP, (1+self.f_coef[2*j])*I-(1-self.f_coef[2*j+1])*powerP)
            H = torch.cat((H, thisH.reshape(1, _node_n, _node_n)), dim=0)
            
        return H
      
    def graphLap(self, A, normalize=True):
        # A: [N, N]
        d = torch.sum(A, dim=-1)
        D = torch.diag(d)
        L = D - A
        if normalize:
            d_ = 1/torch.sqrt(d+1e-4)
            D_ = torch.diag(d_)
            L = torch.matmul(D_, torch.matmul(L, D_)) 
        return L
        
        

    def forward(self, x, att, mask=None):
        #self.att = att
        self.att = torch.mul(att, mask)
        _node_n = self.att.shape[0]
        self.sca = self.diffWavelets(self.att, self.J, _node_n)
        
        L = self.graphLap(self.att)
        
        print(x.shape, self.att.shape, L.shape)

        x = x.unsqueeze(1)
        y_conv = self.feat_gcn(x, self.att.unsqueeze(0))

        y_scat = self.feat_sc1(x, self.sca)
        y_scat = self.act(y_scat)
        y_scat = self.feat_sc2(y_scat, self.sca)

        y = torch.cat((y_conv, y_scat), dim=1)

        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_d) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvScatter_Forest(nn.Module):

    def __init__(self, in_d, out_d, bias=True, node_n=48, J=1, part_sep=None, pg=True, W_pg=0.2, W_p=0.5, edge_prob=0.4):
        super(GraphConvScatter_Forest, self).__init__()
        
        self.dim_up, self.dim_down, n_up, n_down = part_sep
        
        stdv = 1. / math.sqrt(out_d)
        self.att1 = Parameter(torch.FloatTensor(node_n, node_n))
        self.att1.data.uniform_(-stdv, stdv)
        self.att_up1 = Parameter(torch.FloatTensor(n_up, n_up))
        self.att_up1.data.uniform_(-stdv, stdv)
        self.att_down1 = Parameter(torch.FloatTensor(n_down, n_down))
        self.att_down1.data.uniform_(-stdv, stdv)
        
        self.m = Bernoulli(torch.ones_like(self.att1.data)*edge_prob)
        self.m_up = Bernoulli(torch.ones_like(self.att_up1.data)*edge_prob)
        self.m_down = Bernoulli(torch.ones_like(self.att_down1.data)*edge_prob)

        # self.att2 = Parameter(torch.FloatTensor(node_n, node_n))
        # self.att2.data.uniform_(-stdv, stdv)
        # self.att_up2 = Parameter(torch.FloatTensor(n_up, n_up))
        # self.att_up2.data.uniform_(-stdv, stdv)
        # self.att_down2 = Parameter(torch.FloatTensor(n_down, n_down))
        # self.att_down2.data.uniform_(-stdv, stdv)

        self.scat_tree = GraphConvScatter(in_d, out_d, bias, node_n, J)
        #self.scat_tree_up = GraphConvScatter(in_d, out_d, bias, node_n, J)
        #self.scat_tree_down = GraphConvScatter(in_d, out_d, bias, node_n, J)
        # self.m = Bernoulli(torch.ones_like(self.att.data)*edge_prob)
        
        self.pg = pg
        
        if self.pg:
            self.partfuse = PartFuse(out_d)
            self.W_pg = W_pg
        self.W_p = W_p

    def forward(self, x):
      
        mask = self.m.sample().to('cuda')
        mask_up = self.m_up.sample().to('cuda')
        mask_down = self.m_down.sample().to('cuda') 
        
        y = self.scat_tree(x, self.att1, mask)
        y_up = self.scat_tree(x[:,self.dim_up], self.att_up1, mask_up)
        y_down = self.scat_tree(x[:,self.dim_down], self.att_down1, mask_down)
        
        if self.pg:
            Wup2down, Wdown2up = self.partfuse(y_up, y_down)
            y_down = y_down + torch.einsum('nwv,njvd->njwd', (Wup2down, y_up))
            y_up = y_up + torch.einsum('nvw,njwd->njvd',(Wdown2up, y_down))
        
        y_updown = torch.zeros_like(y)
        
        y_updown[:,:,self.dim_up] = y_up
        y_updown[:,:,self.dim_down] = y_down
        
        y = y + y_updown
        
        return y



class GC_Block(nn.Module):

    def __init__(self, in_d, p_dropout, bias=True, node_n=48, J=1, part_sep=None, pg=True, W_pg=0.2, W_p=0.5, edge_prob=0.4):
        super(GC_Block, self).__init__()
        self.in_d = in_d
        self.J = J
        
        stdv = 1. / math.sqrt(in_d)
        self.att1 = Parameter(torch.FloatTensor(node_n, node_n))
        self.att1.data.uniform_(-stdv, stdv)
        

        self.gc1 = GraphConvScatter_Forest(in_d, in_d, bias, node_n, J, part_sep, pg=pg, W_pg=W_pg, W_p=W_p)
        self.bn1_conv = nn.BatchNorm1d(node_n * in_d)
        self.bn1_scat = nn.BatchNorm1d(node_n * in_d)
        
        self.gc2 = GraphOperate(in_d, in_d, bias=bias, paraminit='xavier', J=1)
        #self.gc2 = GraphConvScatter(in_d, in_d, bias, node_n, J)
        #self.gc2 = GraphConvScatter_Forest(in_d, in_d, bias, node_n, J, part_sep, pg=pg, W_pg=W_pg, W_p=W_p)
        self.bn2_conv = nn.BatchNorm1d(node_n * in_d)
        #self.bn2_scat = nn.BatchNorm1d(node_n * in_d)
        self.m = Bernoulli(torch.ones_like(self.att1.data)*edge_prob)

        self.att_Sigmoid1 = KernelAttention_Sigmoid(in_d, in_d)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        N, J, V, d = y.shape

        y_conv = self.bn1_conv(y[:,0].contiguous().view(N, -1)).view(N, 1, V, d)
        y_scat = self.bn1_scat(y[:,1:].contiguous().view(N, self.J * self.J, -1).transpose(1,2)).transpose(1,2).view(N, self.J * self.J, V, d)
        y = torch.cat((y_conv, y_scat), dim=1)
        y = self.act_f(y)                         # y: [16, 5, 66, 256]
        att = self.att_Sigmoid1(y)
        y = y + 0.1 * y * att
        y = y.mean(1)
        y = self.do(y)

        #mask = self.m.sample().to('cuda')
        #att1 = torch.mul(self.att1, mask)
        att1 = self.att1
        y = self.gc2(y.unsqueeze(1), att1.unsqueeze(0))
        #y = self.gc2(y, self.att1, mask)
        y = self.bn2_conv(y.contiguous().view(N, -1)).view(N, V, d)
        #y_conv = self.bn2_conv(y[:,0].contiguous().view(N, -1)).view(N, 1, V, d)
        #y_scat = self.bn2_scat(y[:,1:].contiguous().view(N, self.J * self.J, -1).transpose(1,2)).transpose(1,2).view(N, self.J * self.J, V, d)
        #y = torch.cat((y_conv, y_scat), dim=1)
        y = F.relu(y)
        #y = torch.tanh(y)
        #y = y.mean(1)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_d) + ' -> ' \
               + str(self.in_d) + ')'


class GCN(nn.Module):
    def __init__(self, in_d, hid_d, p_dropout, num_stage=1, node_n=48, J=1, part_sep=None, pg=True, W_pg=0.2, W_p=0.2):

        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(in_d, hid_d, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hid_d)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hid_d, p_dropout=p_dropout, node_n=node_n, J=J, part_sep=part_sep, pg=pg, W_pg=W_pg, W_p=W_p))
        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hid_d, in_d, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        y = y + x

        return y