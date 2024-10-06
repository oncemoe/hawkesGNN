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

from models.base import BaseLPModel
from models.predictor import LinkPredictor
from einops import rearrange


def similarity(a, b):
    return - torch.sum((a - b) ** 2, dim=-1)
        
        

# https://github.com/rootlu/MMDNE/blob/master/code/MMDNE.py
class M2DNE(BaseLPModel):
    """
    Wrapper of MLPMixer and LinkPredictor
    """
    def __init__(self, n_node, n_feat, n_edge, n_hidden, dropout=0.1, bias=False, dim_in_time=100, n_history=10):
        super(M2DNE, self).__init__()
        self.n_node = n_node
        self.n_edge = n_edge
        self.n_history = n_history

        self.node_emb = nn.Embedding(n_node, 1)
        self.zeta     = nn.Parameter(torch.ones(1))
        self.gamma    = nn.Parameter(torch.ones(1))
        self.theta    = nn.Parameter(torch.ones(1))
        self.delta_s  = nn.Parameter(torch.ones(n_node, 1))
        self.delta_t  = nn.Parameter(torch.ones(n_node, 1))
        self.global_attn = nn.Linear(n_hidden, 1)

        self.W = nn.Linear(1, n_hidden, bias=False)
        self.a = nn.Parameter(torch.zeros(1, 1, n_hidden * 2))
        self.leakyrelu = 0.2

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb.weight, gain=1.414)
        
    def forward(self, data, the_edge):
        u_i = self.node_emb(the_edge[0])
        u_j = self.node_emb(the_edge[1])
        lamb_base = similarity(u_i, u_j).view(-1, 1)

        node_emb = self.W(data.x)
        neighbor_emb = node_emb[data.edge_index[1]]
        hist_emb = torch.zeros((self.n_node * self.n_history,  neighbor_emb.size(1)), device=neighbor_emb.device)
        hist_time = torch.zeros((self.n_node * self.n_history,  1), device=neighbor_emb.device)
        mask = torch.zeros((self.n_node * self.n_history,  1), device=neighbor_emb.device)
        
        hist_emb[data.mapping] = neighbor_emb
        hist_time[data.mapping] = data.edge_attr[:, -1:]
        mask[data.mapping] = 1

        hist_emb = rearrange(hist_emb, '(n h) d -> n h d', h=self.n_history) # (N * H, D) -> (N, H, D)
        hist_time = rearrange(hist_time, '(n h) d -> n h d', h=self.n_history).squeeze(-1)
        mask = rearrange(mask, '(n h) d -> n h d', h=self.n_history).squeeze(-1)

        a_input = torch.cat([
            node_emb.unsqueeze(1).expand(-1, self.n_history, -1),
            hist_emb
        ], dim=-1)
        a_score = (a_input * self.a).sum(-1)  # (N, H)
        decay_score_s = F.leaky_relu(torch.exp( - self.delta_s * hist_time) * a_score, negative_slope=self.leakyrelu)
        decay_score_t = F.leaky_relu(torch.exp( - self.delta_t * hist_time) * a_score, negative_slope=self.leakyrelu)
        decay_score_s.masked_fill_(mask == torch.tensor(0), float("-inf"))
        decay_score_t.masked_fill_(mask == torch.tensor(0), float("-inf"))
        hist_attn_s = torch.nan_to_num(torch.softmax(decay_score_s, dim=-1), 0)
        hist_attn_t = torch.nan_to_num(torch.softmax(decay_score_t, dim=-1), 0)

        inter_s = (hist_attn_s.unsqueeze(-1) * hist_emb).sum(dim=1)
        inter_t = (hist_attn_t.unsqueeze(-1) * hist_emb).sum(dim=1)
        global_inter = self.global_attn(torch.stack([inter_s, inter_t], dim=1)).squeeze(-1) # N, 2
        global_attn = torch.softmax(torch.tanh(global_inter), dim=-1)

        global_attn_i = global_attn[:, 0].view(-1, 1)[the_edge[0]]
        global_attn_j = global_attn[:, 1].view(-1, 1)[the_edge[1]]


        lamb_i = similarity(hist_emb[the_edge[0]], node_emb[the_edge[1]].unsqueeze(1)) * torch.exp(self.delta_s[the_edge[0]] * hist_time[the_edge[0]]) * mask[the_edge[0]]
        lamb_j = similarity(hist_emb[the_edge[1]], node_emb[the_edge[0]].unsqueeze(1)) * torch.exp(self.delta_t[the_edge[1]] * hist_time[the_edge[1]]) * mask[the_edge[1]]
        lamb = lamb_base + global_attn_i * lamb_i.sum(dim=1, keepdim=True) + global_attn_j * lamb_j.sum(dim=1, keepdim=True)
        
        return torch.sigmoid(lamb)