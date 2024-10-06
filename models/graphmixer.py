import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.nn.conv.gcn_conv import gcn_norm, Linear

from torch_geometric.utils import add_remaining_self_loops,  softmax, coalesce
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from torch_geometric.typing import (
    Adj,
    OptTensor,
    Size,
)
import math
import numpy as np
from functools import partial

class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, dim=100):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim, bias=False)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float64))).reshape(self.dim, -1))
        self.w.weight.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear, bias=True):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim, bias=bias),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim, bias=bias),
        nn.Dropout(dropout)
    )


class MixerBlock(torch.nn.Module):
    def __init__(self,  n_vertex, n_channel, dropout=0, vertex_factor=0.5, channel_factor=4):
        super(MixerBlock, self).__init__()
        self.n_vertex = n_vertex
        self.n_channel = n_channel
        self.vertex_mix = PreNormResidual(n_channel, FeedForward(n_vertex, vertex_factor, dropout, dense=partial(nn.Conv1d, kernel_size = 1)))
        self.channel_mix = PreNormResidual(n_channel, FeedForward(n_channel, channel_factor, dropout))
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x):
        x = self.vertex_mix(x)
        x = self.channel_mix(x)
        return x

class FeatEncode(nn.Module):
    """
    Return [raw_edge_feat | TimeEncode(edge_time_stamp)]
    """
    def __init__(self, time_dims, feat_dims, out_dims):
        super().__init__()
        
        self.time_encoder = TimeEncode(time_dims)
        self.feat_encoder = nn.Linear(time_dims + feat_dims, out_dims) 
        
    def forward(self, edge_ts, edge_feats=None):
        x = self.time_encoder(edge_ts)
        if edge_feats is not None:
            x = torch.cat([edge_feats, x], dim=1)
        return self.feat_encoder(x)



class MLPMixer(nn.Module):
    """
    Input : [ batch_size, graph_size, edge_dims+time_dims]
    Output: [ batch_size, graph_size, output_dims]
    """
    def __init__(self, n_vertex, hidden_channels=100, out_channels=100):
        super().__init__()
        self.n_vertex = n_vertex
        self.num_layers = 1
        # input & output classifer
        
        self.layernorm = nn.LayerNorm(hidden_channels)
        self.mlp_head = nn.Linear(hidden_channels, out_channels, bias=False)
        
        # inner layers
        self.mixer_blocks = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.mixer_blocks.append(
                MixerBlock(n_vertex, hidden_channels)
            )
            
        # init
        self.reset_parameters()

    def reset_parameters(self):    
        self.layernorm.reset_parameters()
        self.mlp_head.reset_parameters()

    def forward(self, x):
        # apply to original feats
        for i in range(self.num_layers):
            x = self.mixer_blocks[i](x)
        x = self.layernorm(x)
        x = torch.mean(x, dim=1)
        x = self.mlp_head(x)
        return x
    


from models.base import BaseLPModel
from models.predictor import LinkPredictor

class GraphMixer(BaseLPModel):
    """
    Wrapper of MLPMixer and LinkPredictor
    """
    def __init__(self, n_node, n_feat, n_edge, n_hidden, dropout=0, dim_in_time=100, n_history=10):
        super(GraphMixer, self).__init__()
        self.n_node = n_node
        self.n_edge = n_edge
        self.n_history = n_history
        
        self.feat_encoder = FeatEncode(dim_in_time, n_edge, n_hidden)
        self.mixer = MLPMixer(n_history, n_hidden, n_hidden)
        self.input_fc = torch.nn.Linear(n_hidden, n_hidden, bias=False)
        self.predictor = LinkPredictor(n_hidden, n_hidden, 1, 2, dropout)

        
    def forward(self, data):
        edge_time_feats = self.feat_encoder(
            data.edge_attr[:, -1],
            data.edge_attr
        )

        x = torch.zeros((self.n_node * self.n_history,  edge_time_feats.size(1)), device=edge_time_feats.device)
        x[data.mapping] = edge_time_feats     
        x = torch.stack(torch.split(x, self.n_history))  # (N * H, D) -> (N, H, D)
        x = self.mixer(x)

        # x = torch.cat([x, F.one_hot(torch.arange(self.n_node, device=x.device), num_classes=self.n_node )], dim=1)
        x = self.input_fc(x)
        return x
