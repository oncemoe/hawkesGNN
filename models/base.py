import torch
import torch.nn.functional as F
from models.predictor import LinkPredictor
from torch_geometric.data import Data

class BaseLPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.predictor = LinkPredictor(n_out, n_hidden, 1, 2, dropout)

    
    def predict(self, h, edge_index, bs=1024*1024):
        out = []
        for edge in torch.split(edge_index, bs, dim=1):
            out.append(self.predictor(h[edge[0]], h[edge[1]]))
        return torch.cat(out)
    
    
    # predict next snap edges using data from previous snaps
    def train_step(self, data, pos_edge, neg_edge):
        if isinstance(data, list) or isinstance(data, Data):
            h = self(data)
        else:
            h = data

        pos_out = self.predict(h, pos_edge)
        pos_loss = -torch.log(pos_out + 1e-15)
    
        neg_out = self.predict(h, neg_edge)
        neg_loss = -torch.log(1 - neg_out + 1e-15)
        
        alpha =  len(neg_loss)/len(pos_loss)
        loss = torch.cat([pos_loss * alpha, neg_loss]).mean()
        return loss
    
    # re_idx, recover edges from torch.unique (in case tooo may negative samplings)
    def test_step(self, data, pos_edge, neg_edge, re_idx=None):
        if isinstance(data, list) or isinstance(data, Data):
            h = self(data)
        else:
            h = data

        pos_out = self.predict(h, pos_edge)
        pos_loss = -torch.log(pos_out + 1e-15)
    
        neg_out = self.predict(h, neg_edge)
        if re_idx is not None:
            neg_out = neg_out[re_idx]
        neg_loss = -torch.log(1 - neg_out + 1e-15)
        
        alpha =  len(neg_loss)/len(pos_loss)
        loss = torch.cat([pos_loss * alpha, neg_loss]).mean()
        return h, pos_out, neg_out, loss