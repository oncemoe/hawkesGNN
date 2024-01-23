from typing import List, Optional, Tuple, Union

import numpy as np
from torch.utils import data as td
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data.dataset import Dataset, IndexType
from torch_geometric.loader import DataLoader

import torch_geometric.utils as tgu


class StandardScaler:
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
    
class TrafficPredictionDS(td.Dataset):
    def __init__(self, ds: Data, start, end, len_x, len_y) -> None:
        super().__init__()
        self.ds = ds
        self.start = start
        self.end = end
        self.len_x = len_x
        self.len_y = len_y
    
    def __len__(self):
        return self.end - self.start + 1
    
    def __getitem__(self, index):
        data = Data()
        x_u, x_v = self.start + index - self.len_x + 1, self.start + index + 1
        data.id = self.start + index
        data.x = torch.cat([
            self.ds.y[:, x_u : x_v].unsqueeze(0),  # D_y, N, L
            self.ds.tx[:, x_u : x_v].unsqueeze(1).repeat(1, self.ds.num_nodes, 1) # D_x, N, L
        ]).permute(2, 1, 0).unsqueeze(0) # B, L, N ,D
        data.y = self.ds.y[:, self.start + index + 1: self.start + index + self.len_y + 1].permute(1, 0).unsqueeze(0)  # B, L, N
        data.mask = self.ds.mask[:, self.start + index + 1: self.start + index + self.len_y + 1].permute(1, 0).unsqueeze(0)
        return data


class TrafficPredictionLocTimeDS(TrafficPredictionDS):
    def __init__(self, ds: Data, start, end, len_x, len_y) -> None:
        super().__init__(ds, start, end, len_x, len_y)
        lat, lon = ds.loc[:, 0], ds.loc[:, 1]
        x = torch.cos(lat) * torch.cos(lon)
        y = torch.cos(lat) * torch.sin(lon)
        z = torch.sin(lat)
        self.loc = torch.stack([x, y, z]).unsqueeze(-1).repeat(1, 1, len_x)
        sd = torch.arange(ds.num_times, dtype=torch.long) % 288 / 288
        sw = torch.arange(ds.num_times, dtype=torch.long) % (288 * 7) // 288 / 7
        self.step = torch.stack([sd, sw])
        self.seq = torch.arange(ds.num_times, dtype=torch.long) % 288 // 12
    
    def __getitem__(self, index):
        data = Data()
        x_u, x_v = self.start + index - self.len_x + 1, self.start + index + 1
        y_u, y_v = self.start + index + 1, self.start + index + self.len_y + 1
        data.id = self.start + index
        data.x = torch.cat([
            self.ds.y[:, x_u : x_v].unsqueeze(0),  # D_y, N, L
            self.step[:, x_u: x_v].view(2, 1, -1).repeat(1, self.ds.num_nodes, 1) # D_tx, N, L
            #self.loc
        ]).permute(2, 1, 0).unsqueeze(0) # B, L, N ,D
        data.y = self.ds.y[:,y_u: y_v].permute(1, 0).unsqueeze(0)  # B, L, N
        data.mask = self.ds.mask[:, y_u: y_v].permute(1, 0).unsqueeze(0)
        data.xmask = self.ds.mask[:, x_u: x_v].permute(1, 0).unsqueeze(0) < 0.5
        data.seq = self.seq[x_u: x_v].view(1, -1)
        return data


class MetrLaLoaderFactory:
    def __init__(self, dataset: Dataset, len_x = 12, len_y = 12) -> None:
        self.ds = dataset[0]
        self._num_nodes = self.ds.num_nodes
        self._num_times = self.ds.num_times
        self.len_x = len_x
        self.len_y = len_y
        self.split = self.split_dataset()

    def split_dataset(self):
        num_samples = self.num_times - self.len_x - self.len_y + 1
        num_test = round(num_samples* 0.2)
        num_train = round(num_samples * 0.7)
        num_val = num_samples - num_test - num_train
        assert num_test == 6850

        split = [
            num_train+self.len_x-1, 
            num_train + num_val +self.len_x-1, 
            num_train + num_val + num_test + self.len_x-2, 
        ]
        self.train_mean = self.ds.y[:, :split[0]-1].mean().item()
        self.train_std = self.ds.y[:, :split[0]-1].std().item()
        self.scaler = StandardScaler(self.train_mean, self.train_std)
        self.ds.y = self.scaler.transform(self.ds.y)
        return split

    def get_dcrnn_dataloader(self, **kwargs):
        split = self.split
        self._node_feats_dim = 2
        self._edge_feats_dim = 0
        train_ds = TrafficPredictionDS(self.ds, self.len_x-1, split[0]-1, self.len_x, self.len_y)
        val_ds = TrafficPredictionDS(self.ds, split[0], split[1]-1, self.len_x, self.len_y)
        test_ds = TrafficPredictionDS(self.ds, split[1], split[2], self.len_x, self.len_y)
        assert len(test_ds) == 6850
        return {
            'train': DataLoader(train_ds, shuffle=True, **kwargs),
            'val': DataLoader(val_ds, **kwargs),
            'test': DataLoader(test_ds, **kwargs),
        }
    
    def get_stmixer_dataloader(self, **kwargs):
        split = self.split
        self._node_feats_dim = 3
        self._edge_feats_dim = 0
        train_ds = TrafficPredictionLocTimeDS(self.ds, self.len_x-1, split[0]-1, self.len_x, self.len_y)
        val_ds = TrafficPredictionLocTimeDS(self.ds, split[0], split[1]-1, self.len_x, self.len_y)
        test_ds = TrafficPredictionLocTimeDS(self.ds, split[1], split[2], self.len_x, self.len_y)
        assert len(test_ds) == 6850
        return {
            'train': DataLoader(train_ds, shuffle=True, **kwargs),
            'val': DataLoader(val_ds, **kwargs),
            'test': DataLoader(test_ds, **kwargs),
        }
    
    
    @property
    def num_nodes(self) -> int:
        return self._num_nodes
    
    @property
    def num_times(self) -> int:
        return self._num_times
    
    
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
    