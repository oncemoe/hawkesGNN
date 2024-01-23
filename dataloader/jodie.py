from typing import List, Optional, Tuple, Union

import numpy as np
from torch.utils import data as td
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data.dataset import Dataset, IndexType
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import JODIEDataset

import torch_geometric.utils as tgu
from tqdm import tqdm
from dataloader.utils import EvolveGCNDS, GraphPairDS


# @torch.no_grad()
# def jodie_negative_sampling(target, src_max, num_neg_samples=1):
#     """ Sampling exactlly num_neg_samples for every positive edge"""
#     num_dest = target.num_nodes - src_max
#     start = target.edge_index[0]
#     return torch.stack([
#         start.view(-1, 1).repeat(1, num_neg_samples).flatten(), 
#         torch.randint(num_dest, (len(start) * num_neg_samples, )) + src_max
#     ])

@torch.no_grad()
def jodie_negative_sampling(num_nodes, edge_index, src_max=0, num_neg_samples=1):
    """ Sampling exactlly num_neg_samples for every positive edge"""
    num_dest = num_nodes - src_max
    return torch.stack([
        edge_index[0].view(-1, 1).repeat(1, num_neg_samples).flatten(), 
        torch.randint(num_dest, (len(edge_index[0]) * num_neg_samples, )) + src_max
    ])

class JodieLoaderFactory:
    def __init__(self, root, name, node_feat_type='onehot-id',  negative_sampling=1):
        dataset = JODIEDataset(root, name=name)
        data = dataset[0]
        assert data.src.max().item() < data.dst.min().item()

        self.frequency=24*3600*7 if name == 'lastfm' else 1 * 3600
        self.data = data
        self._num_nodes = data.dst.max().item() + 1
        self._src_max = data.src.max().item()
        self._num_neg = negative_sampling

        stamp, count = (data.t / self.frequency).floor().unique(return_counts=True)
        self.ruler = torch.cat([torch.zeros(1), torch.cumsum(count, 0)]).long()
        self.n_snaps = len(stamp)

        # we use last column in msg for time stamp
        

        self._edge_feats_dim = data.msg.shape[1] + 1 # the last is timestamp, [keep same with bitcoin...]
        if node_feat_type == 'onehot-id':
            self.x = torch.eye(self._num_nodes).to_sparse()
            self._node_feats_dim = self._num_nodes
        elif node_feat_type == 'dummy':
            self.x = torch.ones((self._num_nodes, 1))
            self._node_feats_dim = 1
        else:
            raise f"feature type {node_feat_type} not supported! "

    def generate_graph(self, a, b, bidirection=True):
        g = Data()
        g.x = self.x
        s, t = self.ruler[a], self.ruler[b]

        if bidirection:  # jodie src nodes are seperable from dst nodes
            indices = torch.stack([self.data.src[s:t], self.data.dst[s:t], self.data.dst[s:t], self.data.src[s:t]]).view(2, -1)
            ts = (self.frequency * b - torch.cat([self.data.t[s:t], self.data.t[s:t]])) / self.frequency
            attr = torch.cat([self.data.msg[s:t], self.data.msg[s:t]])
            attr = torch.cat([attr, ts.view(-1, 1)], axis=1)
            stamp = torch.cat([self.data.t[s:t], self.data.t[s:t]])
            stamp = (stamp.max() - stamp)/stamp.max()
            g.raw_edge_index = torch.stack([self.data.src[s:t], self.data.dst[s:t]])
            g.neg_edge_index = jodie_negative_sampling(self.num_nodes, g.raw_edge_index, self._src_max+1, self._num_neg)
        else:
            indices = torch.stack([self.data.src[s:t], self.data.dst[s:t]])
            ts = (self.frequency * b - self.data.t[s: t]) / self.frequency
            attr = torch.cat([self.data.msg[s:t], ts.view(-1, 1)], dim=1)
            stamp = self.data.t[s:t]
            g.neg_edge_index = jodie_negative_sampling(self.num_nodes, indices, self._src_max+1, self._num_neg)

        g.adj = torch.sparse.FloatTensor(indices, torch.ones(len(indices[0])), (self.num_nodes, self.num_nodes))
        g.edge_index = indices
        g.edge_attr = attr
        g.t = stamp
        g.num_nodes = self.num_nodes        
        return g
        
    def get_list_dataloader(self, window=5, split=[0.7, 0.85, 1], device='cpu', **kwargs):
        split = [int(self.n_snaps * split[i]) for i in range(3)]
        assert window < split[0]

        ds = []
        for i in tqdm(range(split[2]), desc='Generating'):
            delta = self.generate_graph(i, i+1)
            ds.append(delta)
        train_ds = EvolveGCNDS(ds, window, split[0]-window, window)
        val_ds = EvolveGCNDS(ds, split[0], split[1]-split[0], window)
        test_ds = EvolveGCNDS(ds, split[1], split[2]-split[1], window)
        print(f"train/ val/ test = {len(train_ds)}/ {len(val_ds)}/ {len(test_ds)}")

        # https://github.com/pytorch/pytorch/issues/20248
        # sparse tensor in DataSet reqiure num_workers=0
        return {
            'train': DataLoader(train_ds, batch_size=1, num_workers=0, shuffle=True, **kwargs),
            'val': DataLoader(val_ds, batch_size=1, num_workers=0, **kwargs),
            'test': DataLoader(test_ds, batch_size=1, num_workers=0, **kwargs),
        }
    


    def get_pair_dataloader(self, window=5, split=[0.7, 0.85, 1], device='cpu', **kwargs):
        split = [int(self.n_snaps * split[i]) for i in range(3)]
        assert window < split[0]
        ds = []
        for i in tqdm(range(window, split[-1]), desc='Generating'):
            target = self.generate_graph(i, i+1, bidirection=False)
            data = self.generate_graph(i-window, i)
            ds.append((data, target))
        train_ds = GraphPairDS(ds[:split[0]-window])
        val_ds = GraphPairDS(ds[split[0]-window: split[1]-window])
        test_ds = GraphPairDS(ds[split[1]-window: split[2]-window])
        print(f"train/ val/ test = {len(train_ds)}/ {len(val_ds)}/ {len(test_ds)}")
        
        # https://github.com/pytorch/pytorch/issues/20248
        # sparse tensor in DataSet reqiure num_workers=0
        return {
            'train': DataLoader(train_ds, batch_size=1, num_workers=0, shuffle=True, **kwargs),
            'val': DataLoader(val_ds, batch_size=1, num_workers=0, **kwargs),
            'test': DataLoader(test_ds, batch_size=1, num_workers=0, **kwargs),
        }

    
    def get_roland_snaps(self, split=[0.7, 0.85, 1], device='cpu'):
        split = [int(self.n_snaps * split[i]) for i in range(3)]
        ds = []  # dataset is constant, cannot change in place...
        num_nodes = self.num_nodes
        dp = None
        deg = 0
        for i in tqdm(range(split[2]), desc='Generating'):
            data = self.generate_graph(i, i+1, bidirection=True)            
            if dp is not None:
                d1 = tgu.degree(dp.edge_index[0], num_nodes)
                d2 = tgu.degree(data.edge_index[0], num_nodes)
                deg += d1
                data.keep_ratio = (deg / (deg + d2 + 1e-6)).unsqueeze(-1)  # equation 3 in paper
            ds.append(data)
            dp = data
        return ds, split

    @property
    def num_nodes(self) -> int:
        return self._num_nodes
    
    @property
    def node_feats_dim(self) -> int:
        if hasattr(self, '_node_feats_dim'):
            return self._node_feats_dim
        else:
            return None
        
    @property
    def edge_feats_dim(self) -> int:
        if hasattr(self, '_edge_feats_dim'):
            return self._edge_feats_dim
        else:
            return None