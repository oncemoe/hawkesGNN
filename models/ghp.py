import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, ChebConv
from torch_geometric.nn.inits import glorot
from models.predictor import LinkPredictor
from models.base import BaseLPModel
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



class GHP(BaseLPModel):
    def __init__(self, n_node, n_feat, n_hidden, n_layers=1, dropout=0.1, bias=False, name='gru'):  # lstm cell has bad performance
        super().__init__()

        n_param = 2
        self.gcn = ChebConv(n_param, n_param, 2, bias=bias,)
        self.params = nn.Parameter(torch.zeros((n_node, n_param)))
        self.rnn = nn.LSTMCell(input_size=n_param, hidden_size=n_param)


    def predict(self, h, edge_index, bs=1024*1024):
        h, hist = h[0], h[1]
        h, a = h[:, 0:1], h[:, 1:2]
        
        out = []
        for edge in torch.split(edge_index, bs, dim=1):
            base = h[edge[0]]
            excite = a[edge[0]]
            weight = hist[edge[0], edge[1]]
            lamb = base + excite * weight.view(-1, 1)
            out.append(torch.sigmoid(lamb))
        return torch.cat(out)
        

    def forward(self, batch_list, alpha=0.5):
        x = self.params
        hx = torch.zeros_like(x)
        cx = torch.zeros_like(x)
        hist = None
        T = len(batch_list)
        for t, snap in enumerate(batch_list):
            x = self.gcn(x, snap.edge_index)
            hx, cx = self.rnn(x, (hx, cx))
            x = x + hx
            ###
            val = (1 + alpha) * math.exp((t - T)/alpha) - alpha
            adj = torch.sparse_coo_tensor(snap.edge_index, torch.ones(snap.edge_index.size(1), device=x.device) * val, (x.size(0), x.size(0)))
            if hist is None:
                hist = adj
            else: 
                hist = hist + adj
        return F.relu(x), hist.to_dense()  # 

    def loss(self):
        return 0.1 * self.params.square().sum()

