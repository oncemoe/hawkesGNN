import torch
import torch.nn.functional as F

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j, raw=False):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        if raw:
            return x, torch.sigmoid(x)
        else:
            return torch.sigmoid(x)


# Jodie is bipartie graph, the source and destination are different
class JodieLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = torch.nn.Linear(in_channels, in_channels)
        self.lin_dst = torch.nn.Linear(in_channels, in_channels)
        self.lin_final = torch.nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) * self.lin_dst(z_dst)
        h = F.relu(h)
        h = self.lin_final(h)
        return torch.sigmoid(h)

# Bad performance
class CosinePredictor(torch.nn.Module):
    def __init__(self, in_channels, l2_norm=True):
        super(CosinePredictor, self).__init__()
        self.lin_src = torch.nn.Linear(in_channels, in_channels)
        self.lin_dst = torch.nn.Linear(in_channels, in_channels)
        self.l2_norm = l2_norm

    def forward(self, x_i, x_j):
        x_i = F.relu(self.lin_src(x_i))
        x_j = F.relu(self.lin_dst(x_j))
        if self.l2_norm:
            x_i = F.normalize(x_i)
            x_j = F.normalize(x_j)
        x = x_i * x_j
        x = torch.sum(x, dim=1, keepdim=True)
        return torch.sigmoid(x)