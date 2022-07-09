#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F




class FeatLinearTrans(nn.Module):

    __constants__ = ['bias', 'in_d', 'out_d']

    def __init__(self, in_d, out_d, bias=False, paraminit='uniform'):
        super(FeatLinearTrans, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.weight = Parameter(torch.FloatTensor(out_d, in_d))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_d))
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
        bound = 1. / math.sqrt(self.weight.size(0) * self.weight.size(1))
        self.weight.data.uniform_(-bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def reset_parameters_xavier(self):
        bound = 1. / math.sqrt(self.weight.size(0) * self.weight.size(1))
        nn.init.xavier_normal_(self.weight, gain=1)
        if self.bias is not None:
            self.bias.data.uniform_(-bound, bound)

    def forward(self, inputs):
        return F.linear(inputs, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_d, self.out_d, self.bias is not None)


class KernelAttention_Sigmoid(nn.Module):

    def __init__(self, in_d, out_d, paraminit='xavier'):
        super(KernelAttention_Sigmoid, self).__init__()
        self.fc1 = FeatLinearTrans(in_d, out_d, bias=False, paraminit=paraminit)
        self.fc2 = FeatLinearTrans(out_d, 1, bias=True, paraminit=paraminit)
        self.w_h = nn.Parameter(torch.FloatTensor(in_d, out_d))
        self.w_q = nn.Parameter(torch.FloatTensor(in_d, out_d))
        self.bs = nn.Parameter(torch.FloatTensor(out_d))

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.reset_parameters_xavier()

    def reset_parameters_xavier(self):
        bound = 1. / math.sqrt(self.w_h.size(1))
        nn.init.xavier_normal_(self.w_h, gain=1)
        nn.init.xavier_normal_(self.w_q, gain=1)
        self.bs.data.uniform_(-bound, bound)

    def forward(self, x):
        x = self.fc1(x)          # 16xJx66x256 -> 16xJx66x256  
        q = self.relu(x.mean(1))  # 16xJx66x256 -> 16x66x256
        h = torch.matmul(x, self.w_h) + torch.matmul(q, self.w_q).unsqueeze(1) + self.bs
        h = self.tanh(h)
        h = self.fc2(h)
        a = self.softmax(h)
        return a


class KernelAttention_Softmax(nn.Module):

    def __init__(self, in_d, out_d, paraminit='xavier'):
        super(KernelAttention_Softmax, self).__init__()
        self.fc1 = FeatLinearTrans(in_d, out_d, bias=False, paraminit=paraminit)
        self.fc2 = FeatLinearTrans(out_d, 1, bias=True, paraminit=paraminit)
        self.w_h = nn.Parameter(torch.FloatTensor(in_d, out_d))
        self.w_q = nn.Parameter(torch.FloatTensor(in_d, out_d))
        self.bs = nn.Parameter(torch.FloatTensor(out_d))

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.reset_parameters_xavier()

    def reset_parameters_xavier(self):
        bound = 1. / math.sqrt(self.w_h.size(1))
        nn.init.xavier_normal_(self.w_h, gain=1)
        nn.init.xavier_normal_(self.w_q, gain=1)
        self.bs.data.uniform_(-bound, bound)

    def forward(self, x):
        x = self.fc1(x)          # 16xJx66x256 -> 16xJx66x256  
        q = self.relu(x.mean(1))  # 16xJx66x256 -> 16x66x256 -> 16x66x256x1
        q = torch.matmul(q, self.w_q).unsqueeze(-1) # 16x66x256 -> 16x66x256x1
        h = torch.matmul(x, self.w_h)
        k = h.transpose(1,2)    # 16xJx66x256 -> 16x66xJx256
        a = torch.matmul(k,q).transpose(1,2)   # 16x66xJx256, 16x66x256x1 -> 16x66xJx1 -> 16xJx66x1
        a = self.softmax(a)
        return a


class PartFuse(nn.Module):

    def __init__(self, in_d, out_d=64, paraminit='xavier'):
        super(PartFuse, self).__init__()
        self.fc_up1 = FeatLinearTrans(in_d, out_d, bias=False, paraminit=paraminit)
        self.fc_up2 = FeatLinearTrans(out_d, out_d, bias=False, paraminit=paraminit)
        self.fc_down1 = FeatLinearTrans(in_d, out_d, bias=False, paraminit=paraminit)
        self.fc_down2 = FeatLinearTrans(out_d, out_d, bias=False, paraminit=paraminit)

    def forward(self, x_up, x_down):
        x_up = x_up.mean(1)
        x_down = x_down.mean(1)
        f_up1 = F.relu(self.fc_up1(x_up))          # 16x36x256 -> 16x36x64
        f_down1 = F.relu(self.fc_down1(x_down))    # 16x30x256 -> 16x30x64
        f_up2 = self.fc_up2(f_up1)                 # 16x36x64
        f_down2 = self.fc_down2(f_down1)           # 16x30x64

        f_upT = f_up2.transpose(-1,-2)             # 16x36x64 -> 16x64x36
        f_downT = f_down2.transpose(-1,-2)         # 16x30x64 -> 16x64x30

        Wdown2up = torch.softmax(torch.matmul(f_up2, f_downT), dim=-1)  # 16x36x30
        Wup2down = torch.softmax(torch.matmul(f_down2, f_upT), dim=-1)  # 16x30x36 
        
        return Wup2down, Wdown2up

      
      
class KernelAttention_Transformer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False, device=None, dtype=None):
        super(KernelAttention_Transformer, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        N, J, V, D = src.shape
        src = src.transpose(0,1).reshape(J, -1, D)      # src: [16, 5, 66, 256] -> [5, 16, 66, 256] -> [5, 990, 256]
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2) * 0.2
        # src = self.norm1(src)
        # src = self.linear1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        src = src.reshape(J, N, V, D).transpose(0,1)    # src:[5, 990, 256] -> [5, 16, 66, 256] -> [16, 5, 66, 256]
        return src
      
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)