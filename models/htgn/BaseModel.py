import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
from models.base import BaseLPModel

class BaseModel(BaseLPModel):
    def __init__(self, n_node, n_feat, n_hidden, window, use_gru=True):
        super(BaseModel, self).__init__()
        if use_gru:
            self.gru = nn.GRUCell(n_hidden, n_hidden)
        else:
            self.gru = lambda x, h: x

        self.feat = Parameter((torch.ones(n_node, n_feat)), requires_grad=True)
        self.linear = nn.Linear(n_feat, n_hidden)
        self.hidden_initial = torch.ones(n_node, n_hidden)

        self.model_type = 'HTGN' #args.model[:3]  # GRU or Dyn
        self.num_window = window
        self.Q = Parameter(torch.ones((n_hidden, n_hidden)), requires_grad=True)
        self.r = Parameter(torch.ones((n_hidden, 1)), requires_grad=True)
        self.nhid = n_hidden
        self.reset_parameter()

    def reset_parameter(self):
        glorot(self.Q)
        glorot(self.r)
        glorot(self.feat)
        glorot(self.linear.weight)
        glorot(self.hidden_initial)

    def init_hiddens(self):
        self.hiddens = [self.hidden_initial] * self.num_window
        return self.hiddens

    def weighted_hiddens(self, hidden_window):
        e = torch.matmul(torch.tanh(torch.matmul(hidden_window, self.Q)), self.r)
        e_reshaped = torch.reshape(e, (self.num_window, -1))
        a = F.softmax(e_reshaped, dim=0).unsqueeze(2)
        hidden_window_new = torch.reshape(hidden_window, [self.num_window, -1, self.nhid])
        s = torch.mean(a * hidden_window_new, dim=0)
        return s

    def save_hiddens(self, data_name, path):
        torch.save(self.hiddens, path + '{}_embeddings.pt'.format(data_name))

    def load_hiddens(self, data_name, path):
        self.hiddens = [torch.load(path + '{}_embeddings.pt'.format(data_name))[-1]]
        return self.hiddens[-1]

    def htc(self, x):
        h = self.hiddens[-1]
        return (x - h).pow(2).sum(-1).mean()

    # replace all nodes
    def update_hiddens_all_with(self, z_t):
        self.hiddens.pop(0)  # [element0, element1, element2] remove the first element0
        self.hiddens.append(z_t.clone().detach().requires_grad_(False))  # [element1, element2, z_t]
        return z_t

    # replace current nodes state
    def update_hiddens_with(self, z_t, nodes):
        last_z = self.hiddens[-1].detach_().clone().requires_grad_(False)
        last_z[nodes, :] = z_t[nodes, :].detach_().clone().requires_grad_(False)
        self.hiddens.pop(0)  # [element0, element1, element2] remove the first element0
        self.hiddens.append(last_z)  # [element1, element2, z_t]
        return last_z

    def continuous_encode(self, edge_index, x=None, weight=None):
        x = torch.cat([x, self.hiddens[-1]], dim=1)
        x = F.dropout(x, p=self.dropout1, training=self.training)
        x = self.layer1(x, edge_index)
        x = self.act(x)
        x = F.dropout(x, self.dropout2, training=self.training)
        x = self.layer2(x, edge_index)
        return x

    def gru_encode(self, edge_index, x=None, weight=None):
        x = torch.cat([x, self.hiddens[-1]], dim=1)
        x = F.dropout(x, p=self.dropout1, training=self.training)
        x = self.layer1(x, edge_index)
        x = self.act(x)
        x = F.dropout(x, self.dropout2, training=self.training)
        x = self.layer2(x, edge_index)
        h = self.weighted_hiddens(torch.cat(self.hiddens, dim=0))
        x = self.gru(x, h)
        return x

    def forward(self, edge_index, x=None, weight=None):
        if x is None:
            x = self.linear(self.feat)
        else:
            x = self.linear(x)
        if self.model_type == 'Dyn':
            x = self.continuous_encode(edge_index, x, weight)
        if self.model_type == 'GRU':
            x = self.gru_encode(edge_index, x, weight)
        return x
