import datetime
import os
import os.path as osp
from typing import Callable, Optional
import pandas as pd
import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_gz,
)
from datasets.utils import shrink_edge_index


class StackOverflow(InMemoryDataset):
    r"""The Bitcoin-OTC dataset from the `"EvolveGCN: Evolving Graph
    Convolutional Networks for Dynamic Graphs"
    <https://arxiv.org/abs/1902.10191>`_ paper, consisting of 138
    who-trusts-whom networks of sequential time steps.

    Args:
        root (str): Root directory where the dataset should be saved.
        edge_window_size (int, optional): The window size for the existence of
            an edge in the graph sequence since its initial creation.
            (default: :obj:`10`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #classes
        * - 138
          - 6,005
          - 63497050
          - 0
          - 0
    """

    url = 'https://snap.stanford.edu/data/sx-stackoverflow.txt.gz'

    def __init__(self, root: str, edge_window_size: int = 10,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self._freq = 1200000
        self.edge_window_size = edge_window_size
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def frequency(self):
        return self._freq

    @property
    def raw_file_names(self) -> str:
        return 'sx-stackoverflow.txt'

    @property
    def processed_file_names(self) -> str:
        return 'stackoverflow.pt'

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'so_processed')
    
    @property
    def num_nodes(self) -> int:
        return self._data.edge_index.max().item() + 1

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_gz(path, self.raw_dir)
        os.unlink(path)

    def shrink_edge_index(self, df):
        _, new_edge_index = np.unique(df[['x', 'y']].values, return_inverse=True)
        new_edge_index = new_edge_index.reshape(len(df), 2)
        df['x'] = new_edge_index[:, 0]
        df['y'] = new_edge_index[:, 1]

    def process(self):
        df = pd.read_csv(self.raw_paths[0], sep=' ',  header=None)
        df.columns = ['x', 'y', 't_']
        shrink_edge_index(df)
        tmin = df.t_.min()
        df['t'] = df['t_'].apply(lambda x: int((x - tmin)//self._freq))   # same with EvolveGCN
        df['t_'] = df['t_'].apply(lambda x: int(x - tmin))
        num_nodes = max(df.x.max(), df.y.max())+1
        stamps = df.t.unique()
        stamps.sort()
        
        data_list = []
        for t in stamps:
            sf = df[df['t'] == t]
            data = Data()
            data.num_nodes = num_nodes
            data.edge_index = torch.tensor(sf[['x', 'y']].values.T, dtype=torch.long)
            data.edge_attr = torch.tensor(sf[['t_']].values, dtype=torch.long)
            data_list.append(data)
            
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
