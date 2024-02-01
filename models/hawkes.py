import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.nn.conv.gcn_conv import gcn_norm, Linear

from torch_geometric.utils import add_remaining_self_loops,  softmax, coalesce
from torch_geometric.utils.coalesce import maybe_num_nodes
from torch_scatter import scatter_add
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from torch_geometric.typing import (
    Adj,
    OptTensor,
    Size,
)
import math


# return 1 dimensional norm
def hawkes_norm(edge_index, shrink_edge_index, num_nodes=None, flow="source_to_target", dtype=None):
    assert flow in ["source_to_target", "target_to_source"]

    edge_count = torch.ones((shrink_edge_index.size(1), ), dtype=dtype,
                                    device=shrink_edge_index.device)
    row, col = shrink_edge_index[0], shrink_edge_index[1]
    idx = col if flow == "source_to_target" else row
    deg = scatter_add(edge_count, idx, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-1)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    
    row = edge_index[0]
    return deg_inv_sqrt[row]


def simple_coalesce(
    edge_index: Tensor,
    num_nodes: Optional[int] = None,
    sort_by_row: bool = True,
):
    nnz = edge_index.size(1)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    idx = edge_index.new_empty(nnz + 1)
    idx[0] = -1
    idx[1:] = edge_index[1 - int(sort_by_row)]
    idx[1:].mul_(num_nodes).add_(edge_index[int(sort_by_row)])

    idx[1:], perm = idx[1:].sort()
    edge_index = edge_index[:, perm]
    mask = idx[1:] > idx[:-1]
    
    if mask.all():
        shrink_edge_index = edge_index
    else:
        shrink_edge_index = edge_index[:, mask]
        
    idx = torch.arange(0, nnz, device=edge_index.device)
    idx.sub_(mask.logical_not_().cumsum(dim=0))
    
    return edge_index, shrink_edge_index, perm, idx


class HGCNConv(MessagePassing):
    def __init__(self, n_node: int, in_channels: int, out_channels: int, dropout=0,
                bias: bool = False, skip=False, normalize: bool = False, no_decay=False, **kwargs):
        """
        no_decay = False -> no hawkes process
        """
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.n_node = n_node
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.do_skip = skip
        self.dropout = dropout

        if no_decay:
            self.decay = Parameter(torch.zeros(n_node, 1), requires_grad=False)
        else:
            self.decay = Parameter(torch.ones(n_node, 1))

        if skip:
            self.skip = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)


    def forward(self, x: Tensor, edge_index: Tensor, edge_age: Tensor, node_id=None) -> Tensor:
        # hawkes excitation entity
        if node_id is not None:
            C = torch.exp(-F.relu(edge_age.view(-1, 1) * self.decay[node_id][edge_index[0]]))
        else:
            C = torch.exp(-F.relu(edge_age.view(-1, 1) * self.decay[edge_index[0]]))
        
        if self.training and self.dropout > 0:
            edge_mask = torch.rand(edge_index.size(1), device=edge_index.device) >= self.dropout
            edge_index = edge_index[:, edge_mask]
            C = C[edge_mask] / (1-self.dropout)

        shrink_edge_index = coalesce(edge_index) # A
        norm = hawkes_norm(  # yapf: disable
            edge_index, shrink_edge_index, x.size(self.node_dim),
            self.flow, x.dtype)

        h = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: Tensor)
        out = self.propagate(edge_index, x=h, edge_weight=C * norm.view(-1, 1))

        if self.bias is not None:
            out = out + self.bias
        if self.do_skip:
            out = out + self.skip(x)
        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j








class HGATConv(MessagePassing):
    def __init__(
        self,
        n_node: int,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        bias: bool = False,
        skip = False,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        edge_dim = None,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.do_skip = skip

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')
        self.decay = Parameter(torch.ones(n_node, 1))
        if skip:
            self.skip = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_age: Tensor,
                edge_attr: Tensor = None, node_id=None, size = None,
                return_attention_weights=None):
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.
        
        # if self.training and self.dropout > 0:
        #     edge_mask = torch.rand(edge_index.size(1), device=edge_index.device) >= self.dropout
        #     edge_index = edge_index[:, edge_mask]
        
        if node_id is not None:
            edge_weight = F.relu(edge_age.view(-1, 1) * self.decay[node_id][edge_index[0]])
        else:
            edge_weight = F.relu(edge_age.view(-1, 1) * self.decay[edge_index[0]])
        
        if self.training and self.dropout > 0:
            edge_mask = torch.rand(edge_index.size(1), device=edge_index.device) >= self.dropout
            edge_index = edge_index[:, edge_mask]
            edge_weight = edge_weight[edge_mask] / (1-self.dropout)

        H, C = self.heads, self.out_channels
        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
        x_src = x_dst = self.lin_src(x).view(-1, H, C)

        h = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)
        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr, edge_weight=edge_weight)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=h, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.do_skip:
            out = out + self.skip(x)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            return out, (edge_index, alpha)
        else:
            return out


    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, edge_weight: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge
        
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        #alpha = torch.exp(-edge_age.view(-1, 1) * alpha)
        alpha = torch.exp(-edge_weight.view(-1, 1) * alpha)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')



class HGATConv(MessagePassing):
    def __init__(
        self,
        n_node: int,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        bias: bool = False,
        skip = False,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        edge_dim = None,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.do_skip = skip

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')
        self.decay = Parameter(torch.ones(n_node, 1))
        if skip:
            self.skip = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_age: Tensor, node_id=None, size = None,
                return_attention_weights=None):
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """        
        edge_index, shrink_edge_index, perm, idx = simple_coalesce(edge_index)
        
        H, C = self.heads, self.out_channels
        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
        x_src = x_dst = self.lin_src(x).view(-1, H, C)
        h = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        delta_src = (x_src * self.att_src).sum(dim=-1)
        delta_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        delta = (delta_src, delta_dst)
        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        delta = self.edge_updater(shrink_edge_index, delta=delta)     # sensitivity between two nodes, softmax guarantee positive
        
        alpha = torch.exp(- edge_age[perm].view(-1, 1) * delta[idx])     # 
        norm = hawkes_norm(  # yapf: disable
            edge_index, shrink_edge_index, x.size(self.node_dim), self.flow, x.dtype)
        
        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=h, alpha=alpha * norm.view(-1, 1), size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.do_skip:
            out = out + self.skip(x)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            return out, (edge_index, alpha)
        else:
            return out


    def edge_update(self, delta_j: Tensor, delta_i: OptTensor,
                    index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        delta = delta_j if delta_i is None else delta_j + delta_i
        if index.numel() == 0:
            return delta
        
        delta = F.leaky_relu(delta, self.negative_slope)
        delta = softmax(delta, index, ptr, size_i)
        delta = F.dropout(delta, p=self.dropout, training=self.training)
        return delta

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


from models.predictor import LinkPredictor, JodieLinkPredictor, CosinePredictor
from models.hawkes import HGCNConv, HGATConv
from torch_geometric.nn import GraphConv, GATConv, GCNConv, TransformerConv
from models.base import BaseLPModel
import numpy as np

class HGNNLP(BaseLPModel):
    def __init__(self, n_node, n_feat, n_edge, n_hidden, dropout=0.1, bias = False, layer_skip=False, 
                         name='gcn', layers=2, heads=1, time_encoder=False, jodie=False, batch_norm=False, initial_skip=False):
        super().__init__()
        assert name in ['gcn', 'gat', 'hgcn', 'hgat']
        self.n_node = n_node
        self.name = name
        self.skip = initial_skip
        self.input_fc = torch.nn.Linear(n_feat, n_hidden, bias=False)

        if name == 'gcn':
            self.model = torch.nn.ModuleList([GCNConv(n_hidden, n_hidden, add_self_loops=False, bias=bias) for _ in range(layers)])
        elif name == 'gat':
            self.model = torch.nn.ModuleList([GATConv(n_hidden, n_hidden // heads, add_self_loops=False, bias=bias, heads=heads, edge_dim=n_edge) for _ in range(layers)])
        elif name == 'hgcn':
            self.model = torch.nn.ModuleList([HGCNConv(n_node, n_hidden, n_hidden, dropout, bias, layer_skip) for _ in range(layers)])
        elif name == 'hgat':
            self.model = torch.nn.ModuleList([HGATConv(n_node, n_hidden, n_hidden // heads, dropout, bias, layer_skip, heads=heads, edge_dim=n_edge) for _ in range(layers)])
        self.act = torch.nn.ReLU()
        if batch_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(layers)])
        else:
            self.register_buffer('batch_norms', None)
        self.drop = torch.nn.Dropout(dropout)
        n_out = n_hidden + n_hidden if initial_skip else n_hidden
        self.predictor = LinkPredictor(n_out, n_hidden, 1, 2, dropout)


    def forward(self, data):
        h = self.input_fc(data.x.to_dense())
        h_stack = [h]
        
        edge_attr = data.edge_attr
            
        for layer, net in enumerate(self.model):    
            if self.name == 'gcn':
                h = net(h_stack[-1], data.edge_index)
            elif self.name == 'gat':
                h = net(h_stack[-1], data.edge_index, edge_attr)
            elif self.name == 'hgcn':
                if hasattr(data, 'n_id'):  # speical case for mini batch
                    h = net(h_stack[-1], data.edge_index, data.edge_attr[:, -1:], data.n_id)
                else:
                    h = net(h_stack[-1], data.edge_index, data.edge_attr[:, -1:])
            elif self.name == 'hgat':
                h = net(h_stack[-1], data.edge_index, data.edge_attr[:, -1:])
            
            if self.batch_norms:
                h = self.batch_norms[layer](h)
    
            if layer < len(self.model)-1:
                h = self.act(h)
                h = self.drop(h)
            h_stack.append(h)

        h = h_stack[-1]
        if self.skip:
            h = torch.cat([h_stack[0], h], dim=1)
        return h