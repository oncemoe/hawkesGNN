from typing import List, Optional, Tuple, Union

import numpy as np
from torch.utils import data as td
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data.dataset import Dataset, IndexType
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling, degree, to_undirected

import torch_geometric.utils as tgu
from tqdm import tqdm
from dataloader.utils import safe_negative_sampling, EvolveGCNDS, GraphPairDS
from dataloader.utils import LRUUpdater


class BitcoinLoaderFactory:
    def __init__(self, ds: Dataset, node_feat_type='onehot-id', negative_sampling=1000):
        """
        node_feat_type: onehot-id is prefered. Actually, node degree should not be used for link prediction task. 
        """
        self.ds = ds
        self._num_nodes = ds[0].num_nodes
        self._num_neg = negative_sampling

        if node_feat_type == 'onehot-id':
            self.x = torch.eye(self._num_nodes).to_sparse()
            self._node_feats_dim = self._num_nodes
        elif node_feat_type == 'dummy':
            self.x = torch.ones((self._num_nodes, 1))
            self._node_feats_dim = 1
        else:
            raise f"feature type {node_feat_type} not supported! "

    # 
    def get_list_dataloader(self, window=10, split=[95, 95+14, 95+14+28], device='cpu', **kwargs):
        split = np.array(split)
        assert window < split[0]
        ds = []
        for i, delta in tqdm(enumerate(self.ds[:split[2]]), desc='Generating'):
            delta.x = self.x
            indices, attr = tgu.to_undirected(delta.edge_index, delta.edge_attr)
            values = torch.ones(len(indices[0]))
            adj = torch.sparse.FloatTensor(indices, values, delta.size()).coalesce()
            delta.adj = adj
            delta.raw_edge_index = delta.edge_index
            delta.neg_edge_index = safe_negative_sampling(delta, self._num_neg, device=device)  # neg sampling before undirected
            delta.edge_index = indices
            delta.edge_attr = attr
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
    
    #
    def get_pair_dataloader(self, window=10, split=[95, 95+14, 95+14+28], device='cpu', **kwargs):
        split = np.array(split)
        assert window < split[0] and len(self.ds) >= split[-1]
        ds = []
        for i in tqdm(range(window, split[-1]), desc='Generating'):
            data = Data()
            delta = self.ds[i]
            data.x = self.x
            if window > 1:
                indices = torch.cat([self.ds[i-j].edge_index for j in range(1, window)], dim=1)
                attr = torch.cat([self.ds[i-j].edge_attr for j in range(1, window)], dim=0).double()
            else:
                indices = self.ds[i-1].edge_index
                attr = self.ds[i-1].edge_attr
            #attr = torch.cat([torch.ones_like(self.ds[i-j].edge_attr) * j for j in range(1, window)], dim=0).double()
            #indices, attr = tgu.coalesce(indices, attr, self.num_nodes, reduce='max')
            indices, attr = tgu.to_undirected(indices, attr, self.num_nodes, reduce='none')
            data.adj = torch.sparse.FloatTensor(indices, torch.ones(len(indices[0])), delta.size()).coalesce()
            data.edge_index = indices
            #attr[:, -1] = (attr[:, -1].max() - attr[:, -1]) / self.ds.frequency
            attr[:, -1] = (self.ds.frequency*(i+1) - attr[:, -1]) / self.ds.frequency
            data.edge_attr = attr

            target = Data()
            target.x = self.x
            indices = delta.edge_index
            values = torch.ones(len(indices[0]))
            target.adj = torch.sparse.FloatTensor(indices, values, delta.size()).coalesce()
            target.edge_index = indices
            target.neg_edge_index = safe_negative_sampling(target, self._num_neg, device=device)
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


    def get_roland_snaps(self, split, device='cpu'):
        ds = []  # dataset is constant, cannot change in place...
        num_nodes = self.num_nodes
        x = self.x
        dp = None
        deg = 0
        for i in tqdm(range(split[2]), desc='Generating'):
            data = self.ds[i]
            data.x = x
            
            data.raw_edge_index = data.edge_index
            data.neg_edge_index = safe_negative_sampling(data, self._num_neg, device=device)
            
            indices = data.edge_index
            attr = data.edge_attr
            attr[:, -1] = (self.ds.frequency*(i+1) - attr[:, -1]) / self.ds.frequency
            indices, attr = tgu.to_undirected(indices, attr, self.num_nodes, reduce='none')
            data.adj = torch.sparse.FloatTensor(indices, torch.ones(len(indices[0])), data.size()).coalesce()
            
            data.edge_index = indices
            data.edge_attr = attr
            if dp is not None:
                d1 = degree(dp.edge_index[0], num_nodes)
                d2 = degree(data.edge_index[0], num_nodes)
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
        if hasattr(self, 'ds'):
            return self.ds[0].edge_attr.shape[1]
        else:
            return None
    
    
    # this should not be used
    def one_hot_degree_feature(self, data_list: Union[Dataset, List[Data]]):
        degree_list = []
        max_degree = 0
        for data in data_list:
            degree = tgu.degree(data.edge_index[0], data.num_nodes, torch.long)
            degree_list.append(degree)
            max_degree = max(max_degree, degree.max().item())
        feat_list = []
        self._node_feats_dim = max_degree+1
        for degree in degree_list:
            idx = torch.cat([torch.arange(self.num_nodes).view(1, -1), degree.view(1, -1)])
            val = torch.ones(self._num_nodes)
            sp_feat = torch.sparse.FloatTensor(idx, val, [self._num_nodes, max_degree+1])
            feat_list.append(sp_feat)
        return feat_list