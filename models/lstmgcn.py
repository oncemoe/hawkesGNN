import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot
from models.predictor import LinkPredictor
from models.base import BaseLPModel


# from Dyted
# https://github.com/Kaike-Zhang/DyTed/blob/main/model/LSTMGCN/LSTM_GCN.py
class LSTMGCN(BaseLPModel):
    def __init__(self, n_node, n_feat, n_hidden, n_layers=2, dropout=0.1, bias=False):
        super().__init__()
        self.input_fc = torch.nn.Linear(n_feat, n_hidden)

        self.dropout = nn.Dropout(dropout)
        self.gcn_layers = nn.ModuleList([GCNConv(n_hidden, n_hidden, bias=bias) for i in range(n_layers)])
        self.rnn = nn.GRU(input_size=n_hidden, hidden_size=n_hidden)
        self.rnn2 = nn.GRU(input_size=n_hidden, hidden_size=n_hidden)

        self.predictor = LinkPredictor(n_hidden, n_hidden, 1, 2, dropout)

    def forward(self, batch_list):
        struct_out = []
        for snap in batch_list:
            x =self.input_fc( snap.x.to_dense())
            for gcn in self.gcn_layers:
                x = gcn.forward(x, snap.edge_index)
            struct_out.append(x[None, :, :])  # N x dim - len(T)

        x = torch.cat(struct_out, dim=0)
        output, hn = self.rnn(x)
        output, hn = self.rnn2(output)

        return output.transpose(0, 1).contiguous()

