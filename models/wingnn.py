import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm, OptTensor
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
import torch.nn.functional as F


# from https://github.com/thudm/WinGNN
class GCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)

        # skip_connection  will always be affine
        self.linear_skip_weight = nn.Parameter(torch.ones(size=(out_channels, in_channels)))
        self.linear_skip_bias = nn.Parameter(torch.ones(size=(out_channels, )))
    
        self.linear_msg_weight = nn.Parameter(torch.ones(size=(out_channels, in_channels)))
        self.linear_msg_bias = nn.Parameter(torch.ones(size=(out_channels, )))

        self.activate = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear_skip_weight, gain=gain)
        nn.init.xavier_normal_(self.linear_msg_weight, gain=gain)

        nn.init.constant_(self.linear_skip_bias, 0)
        nn.init.constant_(self.linear_msg_bias, 0)

    def norm(self, edge_index):
        row = edge_index[0]
        edge_weight = torch.ones((row.size(0),), device=row.device)
        deg = scatter_add(edge_weight, row, dim=0, dim_size=graph.num_nodes())
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt

    def message_fun(self, edges):
        return {'m': edges.src['h'] * 0.1}

    def forward(self, x: Tensor, edge_index, edge_weight=None, fast_weights=None):
        if fast_weights:
            linear_skip_weight = fast_weights[0]
            linear_skip_bias = fast_weights[1]
            linear_msg_weight = fast_weights[2]
            linear_msg_bias = fast_weights[3]
        else:
            linear_skip_weight = self.linear_skip_weight
            linear_skip_bias = self.linear_skip_bias
            linear_msg_weight = self.linear_msg_weight
            linear_msg_bias = self.linear_msg_bias

        edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(0))

        rst = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        rst_ = F.linear(rst, linear_msg_weight, linear_msg_bias)
        skip_x = F.linear(x, linear_skip_weight, linear_skip_bias)
        return rst_ + skip_x

    def message(self, x_j: Tensor, edge_weight: OptTensor):
        return (x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j) * 0.1

from models.base import BaseLPModel
class WinGNN(BaseLPModel):
    def __init__(self, n_node, n_feat, n_edge, n_hidden, dropout=0.1, edge_decoding='dot'):
        super(WinGNN, self).__init__()
        self.dropout = dropout
        
        self.weight1 = nn.Parameter(torch.ones(size=(n_hidden, n_hidden)))
        self.weight2 = nn.Parameter(torch.ones(size=(1, n_hidden)))

        self.gnn = nn.ModuleList()
        self.gnn.append(GCNLayer(n_feat, n_hidden))
        self.gnn.append(GCNLayer(n_hidden, n_hidden))

        if edge_decoding == 'dot':
            self.decode_module = lambda v1, v2: torch.sum(v1 * v2, dim=-1)
        elif edge_decoding == 'cosine_similarity':
            self.decode_module = nn.CosineSimilarity(dim=-1)
        else:
            raise ValueError('Unknown edge decoding {}.'.format(edge_decoding))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.weight1, gain=gain)
        nn.init.xavier_normal_(self.weight2, gain=gain)

    def forward(self, data, fast_weights=None):
        if data.x.is_sparse:
            x = data.x.to_dense()
        else:
            x = data.x

        for i, conv in enumerate(self.gnn):
            if fast_weights is None:
                x = conv(x, data.edge_index)
            else:
                x = conv(x, data.edge_index, fast_weights=fast_weights[2+i*4: 6+i*4])

        if fast_weights:
            weight1 = fast_weights[0]
            weight2 = fast_weights[1]
        else:
            weight1 = self.weight1
            weight2 = self.weight2

        x = F.normalize(x)
        pred = F.dropout(x, self.dropout)
        pred = F.relu(F.linear(pred, weight1))
        pred = F.dropout(pred, self.dropout)
        pred = F.sigmoid(F.linear(pred, weight2))
        return pred

    def predict(self, h, edge_index, bs=1024*64):
        out = []
        for edge in torch.split(edge_index, bs, dim=1):
            out.append(self.decode_module(h[edge[0]], h[edge[1]]))
        return torch.cat(out)

if __name__ == "__main__":
    model = WinGNN(5881, 1, 2, 64)
    print(model)
    for p in model.parameters():
        print(p.shape)
    #print(list(model.parameters()))