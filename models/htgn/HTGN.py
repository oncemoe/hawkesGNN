import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, kaiming_uniform
from models.htgn.hyplayers import HGCNConv, HypGRU, HGATConv
from models.htgn.manifolds import PoincareBall
from models.htgn.BaseModel import BaseModel


# use default args from https://github.com/marlin-codes/HTGN/blob/main/script/config.py
class HTGN(BaseModel):
    def __init__(self, n_node, n_feat, n_edge, n_hidden, dropout=0.1, bias = False, window=10, aggregation='deg', heads=1, layer_skip=False, ):
        super().__init__(n_node, n_feat, n_hidden, window)
        self.manifold_name = 'PoincareBall'
        self.manifold = PoincareBall()

        curvature = 1.0
        fixed_curvature = False
        use_hta = True
        assert aggregation in ['deg', 'att']
        
        self.c = Parameter(torch.ones(3, 1) * curvature, requires_grad=not fixed_curvature)
        self.feat = Parameter((torch.ones(n_node, n_feat)), requires_grad=True)
        self.linear = nn.Linear(n_feat, n_hidden)
        self.hidden_initial = torch.ones(n_node, n_hidden)
        self.use_hta = use_hta
        if aggregation == 'deg':
            self.layer1 = HGCNConv(self.manifold, 2 * n_hidden, 2 * n_hidden, self.c[0], self.c[1],
                                   dropout=dropout)
            self.layer2 = HGCNConv(self.manifold, 2 * n_hidden, n_hidden, self.c[1], self.c[2], dropout=dropout)
        if aggregation == 'att':
            self.layer1 = HGATConv(self.manifold, 2 * n_hidden, 2 * n_hidden, self.c[0], self.c[1],
                                   heads=heads, dropout=dropout, att_dropout=dropout, concat=True)
            self.layer2 = HGATConv(self.manifold, 2 * n_hidden * heads, n_hidden, self.c[1], self.c[2],
                                   heads=heads, dropout=dropout, att_dropout=dropout, concat=False)
        self.gru = nn.GRUCell(n_hidden, n_hidden)

        self.nhid = n_hidden
        self.nout = n_hidden
        self.cat = True
        self.Q = Parameter(torch.ones((n_hidden, n_hidden)), requires_grad=True)
        self.r = Parameter(torch.ones((n_hidden, 1)), requires_grad=True)
        self.num_window = window
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.Q)
        glorot(self.r)
        glorot(self.feat)
        glorot(self.linear.weight)
        glorot(self.hidden_initial)

    def init_hiddens(self):
        self.hiddens = [self.initHyperX(self.hidden_initial)] * self.num_window
        return self.hiddens

    def weighted_hiddens(self, hidden_window):
        if self.use_hta == 0:
            return self.manifold.proj_tan0(self.manifold.logmap0(self.hiddens[-1], c=self.c[2]), c=self.c[2])
        # temporal self-attention
        e = torch.matmul(torch.tanh(torch.matmul(hidden_window.to(self.Q.device), self.Q)), self.r)
        e_reshaped = torch.reshape(e, (self.num_window, -1))
        a = F.softmax(e_reshaped, dim=0).unsqueeze(2)
        hidden_window_new = torch.reshape(hidden_window, [self.num_window, -1, self.nout])
        s = torch.mean(a * hidden_window_new, dim=0) # torch.sum is also applicable
        return s

    def initHyperX(self, x, c=1.0):
        if self.manifold_name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        return self.toHyperX(x, c)

    def toHyperX(self, x, c=1.0):
        x_tan = self.manifold.proj_tan0(x, c)
        x_hyp = self.manifold.expmap0(x_tan, c)
        x_hyp = self.manifold.proj(x_hyp, c)
        return x_hyp

    def toTangentX(self, x, c=1.0):
        x = self.manifold.proj_tan0(self.manifold.logmap0(x, c), c)
        return x

    def htc(self, x):
        x = self.manifold.proj(x, self.c[2])
        h = self.manifold.proj(self.hiddens[-1], self.c[2])

        return self.manifold.sqdist(x, h, self.c[2]).mean()

    def forward(self, edge_index, x=None, weight=None):
        if x is None:  # using trainable feat matrix
            x = self.initHyperX(self.linear(self.feat), self.c[0])
        else:
            x = self.initHyperX(self.linear(x), self.c[0])
        if self.cat:
            x = torch.cat([x, self.hiddens[-1].to(x.device)], dim=1)

        # layer 1
        x = self.manifold.proj(x, self.c[0])
        x = self.layer1(x, edge_index)

        # layer 2
        x = self.manifold.proj(x, self.c[1])
        x = self.layer2(x, edge_index)

        # GRU layer
        x = self.toTangentX(x, self.c[2])  # to tangent space
        hlist = self.manifold.proj_tan0(
            torch.cat([self.manifold.logmap0(hidden, c=self.c[2]).to(x.device) for hidden in self.hiddens], dim=0), c=self.c[2])
        h = self.weighted_hiddens(hlist)
        x = self.gru(x, h)  # can also utilize HypGRU
        x = self.toHyperX(x, self.c[2])  # to hyper space
        return x
