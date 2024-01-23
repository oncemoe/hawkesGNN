import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
from models.predictor import LinkPredictor
import torch.nn.functional as F
from models.base import BaseLPModel

def init_weights(m):
    """Performs weight initialization."""
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data = nn.init.xavier_uniform_(m.weight.data,
                                                gain=nn.init.calculate_gain(
                                                    'relu'))
        if m.bias is not None:
            m.bias.data.zero_()


class ResidualEdgeConvLayer(MessagePassing):
    r"""General GNN layer, with arbitrary edge features.
    from https://github.com/snap-stanford/roland/blob/master/graphgym/contrib/layer/residual_edge_conv.py
    """

    def __init__(self, in_channels, out_channels, edge_channels, improved=False, cached=False,
                 bias=True, aggr='add', normalize_adj=False, msg_direction='both', skip_connection='affine', **kwargs):
        super(ResidualEdgeConvLayer, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize_adj
        self.msg_direction = msg_direction
        self.skip_connection = skip_connection

        if self.msg_direction == 'single':
            self.linear_msg = nn.Linear(in_channels + edge_channels,
                                        out_channels, bias=False)
        elif self.msg_direction == 'both':
            self.linear_msg = nn.Linear(in_channels * 2 + edge_channels,
                                        out_channels, bias=False)
        else:
            raise ValueError

        if skip_connection == 'affine':
            self.linear_skip = nn.Linear(in_channels, out_channels, bias=True)
        elif skip_connection == 'identity':
            assert self.in_channels == self.out_channels

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None, edge_feature=None):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                             edge_weight, self.improved, x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        if self.skip_connection == 'affine':
            skip_x = self.linear_skip(x)
        elif self.skip_connection == 'identity':
            skip_x = x
        else:
            skip_x = 0
        return self.propagate(edge_index, x=x, norm=norm, edge_feature=edge_feature) + skip_x

    def message(self, x_i, x_j, norm, edge_feature):
        if self.msg_direction == 'both':
            x_j = torch.cat((x_i, x_j, edge_feature), dim=-1)
        elif self.msg_direction == 'single':
            x_j = torch.cat((x_j, edge_feature), dim=-1)
        else:
            raise ValueError
        x_j = self.linear_msg(x_j)
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {},{})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels, self.edge_channels)


class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0, has_act=True, has_bn=True, has_l2norm=False):
        super().__init__()
        self.has_l2norm = has_l2norm
    
        layer_wrapper = [nn.Linear(dim_in, dim_out, bias=not has_bn)]
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(dim_out))
        if dropout > 0:
            layer_wrapper.append(nn.Dropout(dropout))
        if has_act:
            layer_wrapper.append(nn.PReLU())
        self.layer = nn.Sequential(*layer_wrapper)

    def forward(self, x):
        x = self.layer(x)
        if self.has_l2norm:
            x = F.normalize(x, p=2, dim=1)
        return x


class RolandLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels*2, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


# ~~from https://github.com/snap-stanford/roland/blob/master/graphgym/models/gnn.py~~
#  oh shit! it should from:
# https://github.com/snap-stanford/roland/blob/master/graphgym/contrib/network/gnn_recurrent.py
# refactoring this project frustrates me  :<
class RolandGNN(BaseLPModel):
    r"""The General GNN model"""
    def __init__(self, n_node, n_feat, n_edge, n_hidden, n_mlp_layer=2, n_mp_layer=2, 
                        dropout=0.1, updater='gru'):
        super(RolandGNN, self).__init__()
        assert updater in ['gru', 'ma', 'mlp', 'gru-ma']
        self.edge_encoder = nn.Linear(n_edge, n_hidden)
        self.edge_encoder_bn = nn.BatchNorm1d(n_hidden)
    
        
        self.pre_mp = nn.Sequential(
            *[LinearLayer(n_feat if i ==0 else n_hidden, n_hidden) for i in range(n_mlp_layer)]
        )
    
        if n_mp_layer > 0:
            self.mp = nn.ModuleList([
                ResidualEdgeConvLayer(n_hidden, n_hidden, n_hidden) for _ in range(n_mp_layer)
            ])
            self.bn = nn.ModuleList([ nn.BatchNorm1d(n_hidden) for _ in range(n_mp_layer) ])
            self.act = nn.ModuleList([ nn.PReLU() for _ in range(n_mp_layer) ])
        else:
            self.register_parameter('mp', None)
    
        if updater in ['gru', 'gru-ma']:
            self.updater = GRUUpdater(n_hidden, n_hidden, updater)
        elif updater == 'mlp':
            self.updater = MLPUpdater(n_hidden, n_hidden)
        else:
            self.updater = MAUpdater()
        self.predictor = LinkPredictor(n_hidden, n_hidden, 1, 2, 0)
        self.apply(init_weights)

    def forward(self, data, H_prev):
        if data.x.is_sparse:
            x = data.x.to_dense()
        else:
            x = data.x
        h = self.pre_mp(x)
        
        e = self.edge_encoder(data.edge_attr.double())
        e = self.edge_encoder_bn(e)
        h_stack = []
        for i, conv in enumerate(self.mp):
            h = conv(h, data.edge_index, edge_feature=e)
            h = self.bn[i](h)
            h = self.act[i](h)
            if H_prev is not None:
                h = self.updater(h, H_prev[i], data.keep_ratio)
            h_stack.append(h.detach())
        return h, h_stack


# from https://github.com/snap-stanford/roland/blob/master/graphgym/models/layer.py
class GRUUpdater(nn.Module):
    """
    Node embedding update block using standard GRU and variations of it.
    """
    def __init__(self, dim_in, dim_out, method='gru'):
        # dim_in (dim of X): dimension of input node_feature.
        # dim_out (dim of H): dimension of previous and current hidden states.
        # forward(X, H) --> H.
        super(GRUUpdater, self).__init__()
        assert method in ['gru', 'gru-ma']
        self.method = method
        self.GRU_Z = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Sigmoid())
        # reset gate.
        self.GRU_R = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Sigmoid())
        # new embedding gate.
        self.GRU_H_Tilde = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Tanh())
    
    def forward(self, X, H_prev, keep_ratio):
        Z = self.GRU_Z(torch.cat([X, H_prev], dim=1))
        R = self.GRU_R(torch.cat([X, H_prev], dim=1))
        H_tilde = self.GRU_H_Tilde(torch.cat([X, R * H_prev], dim=1))
        H_gru = Z * H_prev + (1 - Z) * H_tilde

        if self.method == 'gru-ma':
            # Only update for active nodes, using moving average with output from GRU.
            H_out = H_prev * keep_ratio + H_gru * (1 - keep_ratio)
        elif self.method== 'gru':
            # Update all nodes' embedding using output from GRU.
            H_out = H_gru
        return H_out


class MLPUpdater(nn.Module):
    """
    Node embedding update block using simple MLP.
    embedding_new <- MLP([embedding_old, node_feature_new])
    fix: 2 layer mlp (all configs has 2 layers except as733 has 3 layers)
    """
    def __init__(self, dim_in, dim_out):
        super(MLPUpdater, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_in),
            nn.PReLU(),
            nn.Linear(dim_in, dim_out)
        )

    def forward(self, X, H_prev, keep_ratio):
        concat = torch.cat((H_prev, X), axis=1)
        H_new = self.mlp(concat)
        return H_new


class MAUpdater(nn.Module):
    """Moving Average updater"""
    def __init__(self):
        super().__init__()
    
    def forward(self, X, H_prev, keep_ratio):
        return H_prev * keep_ratio + X * (1 - keep_ratio)