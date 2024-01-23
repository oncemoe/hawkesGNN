import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.nn.conv.gcn_conv import gcn_norm, Linear

from torch_geometric.utils import add_remaining_self_loops,  softmax
from torch_scatter import scatter_add
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from torch_geometric.typing import (
    Adj,
    OptTensor,
    Size,
)
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax

class HGCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}^{\top} \sum_{j \in
        \mathcal{N}(v) \cup \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j
        \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """


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


    def forward(self, x: Tensor, edge_index: Tensor, edge_age: Tensor) -> Tensor:
        edge_weight = torch.exp(-F.relu(edge_age.view(-1, 1) * self.decay[edge_index[0]]))
        
        if self.training and self.dropout > 0:
            edge_mask = torch.rand(edge_index.size(1), device=edge_index.device) >= self.dropout
            edge_index = edge_index[:, edge_mask]
            edge_weight = edge_weight[edge_mask] / (1-self.dropout)

        #print(edge_weight.sum(), self.decay.sum())
        #print(self.decay.min().item(), self.decay.mean().item(), self.decay.max().item())
        #print(edge_weight.min().item(), edge_weight.mean().item(), edge_weight.max().item())

        # gcn norm is bad for hawkes
        # if self.normalize:
        #     edge_index, edge_weight = gcn_norm(  # yapf: disable
        #         edge_index, edge_weight, x.size(self.node_dim),
        #         False, False, self.flow, x.dtype)

        h = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: Tensor)
        out = self.propagate(edge_index, x=h, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out = out + self.bias
        if self.do_skip:
            out = out + self.skip(x)
        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j








class HGATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j
        \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,j}]\right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k
        \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,k}]\right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out}))` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """
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
        # self.scale = Linear(1, 1, bias=False, weight_initializer='glorot')
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
                edge_attr: Tensor = None, size = None,
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
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr, edge_age=edge_age)# * edge_weight

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
                    edge_attr: OptTensor, edge_age: OptTensor, index: Tensor, ptr: OptTensor,
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
        alpha = torch.exp(-edge_age.view(-1, 1) * alpha)
        #alpha = torch.exp(-F.relu(self.scale(edge_age.view(-1, 1)) * alpha))
        return alpha

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
                         name='gcn', layers=2, heads=1, time_encoder=False, jodie=False, initial_skip=False):
        super().__init__()
        assert name in ['gcn', 'gat', 'hgcn', 'hgat']
        self.name = name
        self.skip = initial_skip
        self.input_fc = torch.nn.Linear(n_feat, n_hidden, bias=bias)

        if name == 'gcn':
            self.model = torch.nn.ModuleList([GCNConv(n_hidden, n_hidden, add_self_loops=False, bias=bias) for _ in range(layers)])
        elif name == 'gat':
            self.model = torch.nn.ModuleList([GATConv(n_hidden, n_hidden // heads, add_self_loops=False, bias=bias, heads=heads, edge_dim=n_edge) for _ in range(layers)])
        elif name == 'hgcn':
            self.model = torch.nn.ModuleList([HGCNConv(n_node, n_hidden, n_hidden, dropout, bias, layer_skip) for _ in range(layers)])
        elif name == 'hgat':
            self.model = torch.nn.ModuleList([HGATConv(n_node, n_hidden, n_hidden // heads, dropout, bias, layer_skip, heads=heads, edge_dim=n_edge) for _ in range(layers)])
        self.act = torch.nn.ReLU()
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
                h = net(h_stack[-1], data.edge_index, data.edge_attr[:, -1:])
            elif self.name == 'hgat':
                h = net(h_stack[-1], data.edge_index, data.edge_attr[:, -1:], edge_attr)
    
            if layer < len(self.model)-1:
                h = self.act(h)
                h = self.drop(h)
            h_stack.append(h)

        h = h_stack[-1]
        if self.skip:
            h = torch.cat([h_stack[0], h], dim=1)
        return h