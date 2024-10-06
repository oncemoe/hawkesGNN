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
    extract_tar
)
from datasets.utils import shrink_edge_index


class DBLP(InMemoryDataset):
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
        * - 50
          - 1000
          - 4870863
          - 0
          - 0
    """

    url1 = 'https://raw.githubusercontent.com/rootlu/MMDNE/master/data/dblp/dblp.txt'
    url2 = 'https://raw.githubusercontent.com/rootlu/MMDNE/master/data/dblp/node2label.txt'

    def __init__(self, root: str, 
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])
        #self.load(self.processed_paths[0], data_cls=Data)

    @property
    def raw_file_names(self) -> str:
        return 'dblp.txt', 'node2label.txt', 'dblp.npy'

    @property
    def processed_file_names(self) -> str:
        return 'dblp.pt'

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'dblp_processed')
    
    @property
    def num_nodes(self) -> int:
        return self._data.edge_index.max().item() + 1

    def download(self):
        path = download_url(self.url1, self.raw_dir)
        path = download_url(self.url2, self.raw_dir)

    def shrink_edge_index(self, df):
        _, new_edge_index = np.unique(df[['x', 'y']].values, return_inverse=True)
        new_edge_index = new_edge_index.reshape(len(df), 2)
        df['x'] = new_edge_index[:, 0]
        df['y'] = new_edge_index[:, 1]

    def process(self):
        df = pd.read_csv(self.raw_paths[0], header=None, sep=' ')
        df.columns = ['x', 'y', 't']
        num_nodes = max(df.x.max(), df.y.max())+1
        
        data = Data()
        data.num_nodes = int(num_nodes)
        data.edge_index = torch.tensor(df[['x', 'y']].values.T, dtype=torch.long)
        data.edge_attr = torch.tensor(df[[ 't']].values, dtype=torch.float)
        
        yf = pd.read_csv(self.raw_paths[1], header=None, sep=' ')
        yf.columns = ['id', 'y']
        data.y = torch.tensor(yf[[ 'y']].values, dtype=torch.long)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save((data), self.processed_paths[0])


        # data, slices = self.collate(data)
        # torch.save((data, slices), self.processed_paths[0])
