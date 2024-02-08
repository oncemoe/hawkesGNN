import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from models.htgn.manifolds import PoincareBall


EPS = 1e-15
MAX_LOGVAR = 10


class ReconLoss(nn.Module):
    def __init__(self):
        super(ReconLoss, self).__init__()
        self.r = 2.0
        self.t = 1.0
        self.sigmoid = True
        self.manifold = PoincareBall()
        self.use_hyperdecoder = True #args.use_hyperdecoder and args.model == 'HTGN'

    @staticmethod
    def maybe_num_nodes(index, num_nodes=None):
        return index.max().item() + 1 if num_nodes is None else num_nodes

    def decoder(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def hyperdeoder(self, z, edge_index):
        def FermiDirac(dist):
            probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
            return probs

        edge_i = edge_index[0]
        edge_j = edge_index[1]
        z_i = torch.nn.functional.embedding(edge_i, z)
        z_j = torch.nn.functional.embedding(edge_j, z)
        dist = self.manifold.sqdist(z_i, z_j, c=1.0)
        return FermiDirac(dist)

    def forward(self, z, pos_edge_index, neg_edge_index=None):
        decoder = self.hyperdeoder if self.use_hyperdecoder else self.decoder
        pos_loss = -torch.log(
            decoder(z, pos_edge_index) + EPS).mean()
        neg_loss = -torch.log(1 - decoder(z, neg_edge_index) + EPS).mean()
        alpha = neg_edge_index.size(1) / pos_edge_index.size(1)
        return pos_loss * alpha + neg_loss

    def predict(self, z, pos_edge_index, neg_edge_index):
        decoder = self.hyperdeoder if self.use_hyperdecoder else self.decoder
        pos_pred = decoder(z, pos_edge_index)
        neg_pred = decoder(z, neg_edge_index)
        return pos_pred, neg_pred


class VGAEloss(ReconLoss):
    def __init__(self):
        super(VGAEloss, self).__init__()

    def kl_loss(self, mu=None, logvar=None):
        mu = self.__mu__ if mu is None else mu
        logvar = self.__logvar__ if logvar is None else logvar.clamp(
            max=MAX_LOGVAR)
        return -0.5 * torch.mean(
            torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))

    def forward(self, x, pos_edge_index, neg_edge_index):
        z, mu, logvar = x
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
        reconloss = pos_loss + neg_loss
        klloss = (1 / z.size(0)) * self.kl_loss(mu=mu, logvar=logvar)

        return reconloss + klloss