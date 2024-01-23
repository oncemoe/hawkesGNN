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
    extract_bz2,
)
import tarfile
from datasets.utils import shrink_edge_index


class AS733(InMemoryDataset):
    r"""The UCIMessage dataset from the `"EvolveGCN: Evolving Graph
    Convolutional Networks for Dynamic Graphs"
    <https://arxiv.org/abs/1902.10191>`_ paper, consisting of 733 
    communication networks of routers that exchange traffic flows with peers. 
    This data set may be used to forecast message exchanges in the future.

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
        * - 733
          - 1,899
          - 59835
          - 0
          - 0
    """

    url = 'https://snap.stanford.edu/data/as-733.tar.gz'

    def __init__(self, root: str, edge_window_size: int = 10,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.edge_window_size = edge_window_size
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self._freq = 1

    @property
    def frequency(self):
        return self._freq

    @property
    def raw_file_names(self) -> str:
        return 'as-733.tar.gz'

    @property
    def processed_file_names(self) -> str:
        return 'as733.pt'

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')
    
    @property
    def num_nodes(self) -> int:
        return self._data.edge_index.max().item() + 1

    def download(self):
        path = download_url(self.url, self.raw_dir)
        # extract_gz(path, self.raw_dir)
        # os.unlink(path)

    def process(self):
        dfs = []
        with tarfile.open(self.raw_paths[0], 'r:gz') as tf:
            for fn in tf.getnames():
                f = tf.extractfile(fn)
                df = pd.read_csv(f, sep='\t', header=None, skiprows=4)
                df.columns = ['x', 'y']
                df['t_'] = int(fn[2:-4])
                dfs.append(df)

        df = pd.concat(dfs)
        shrink_edge_index(df)
        tmin = df.t_.min()
        df['t'] = df['t_'].apply(lambda x: (x - tmin))   # same with EvolveGCN
        num_nodes = max(df.x.max(), df.y.max())+1
        stamps = df.t.unique()
        stamps.sort()
        
        data_list = []
        for i, t in enumerate(stamps):  # as733 is sampled nonuniformally from 1971, 1980, 1990, 2000
            sf = df[df['t'] == t]
            data = Data()
            data.num_nodes = num_nodes
            data.edge_index = torch.tensor(sf[['x', 'y']].values.T, dtype=torch.long)
            data.edge_attr = torch.ones((len(sf), 1), dtype=torch.long) * i #torch.tensor(sf[['t']].values, dtype=torch.long)
            data_list.append(data)
            
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
