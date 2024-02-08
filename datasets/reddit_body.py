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
from datetime import datetime

class RedditBody(InMemoryDataset):

    url = 'https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv'

    def __init__(self, root: str, edge_window_size: int = 10,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self._freq = 7 # 7day, weekly
        self.edge_window_size = edge_window_size
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def frequency(self):
        return self._freq

    @property
    def raw_file_names(self) -> str:
        return 'soc-redditHyperlinks-body.tsv'

    @property
    def processed_file_names(self) -> str:
        return 'reddit_body.pt'

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'reddit_processed')
    
    @property
    def num_nodes(self) -> int:
        return self._data.edge_index.max().item() + 1

    def download(self):
        path = download_url(self.url, self.raw_dir)

    def shrink_edge_index(self, df):
        _, new_edge_index = np.unique(df[['x', 'y']].values, return_inverse=True)
        new_edge_index = new_edge_index.reshape(len(df), 2)
        df['x'] = new_edge_index[:, 0]
        df['y'] = new_edge_index[:, 1]

    def process(self):
        df = pd.read_csv(self.raw_paths[0], sep='\t').iloc[:, [0,1,3,4]]
        df.columns = ['x', 'y','t_', 'w']
        shrink_edge_index(df)
        date_format = "%Y-%m-%d"
        tmin = datetime.strptime(df.t_.min()[:10], date_format)
        df['t_'] = df['t_'].apply(lambda x: (datetime.strptime(x[:10], date_format) - tmin).days)
        df['t'] = df['t_'].apply(lambda x: int( x//self._freq ))   # same with EvolveGCN
        
        num_nodes = max(df.x.max(), df.y.max())+1
        stamps = df.t.unique()
        stamps.sort()
        
        data_list = []
        for t in stamps:
            sf = df[df['t'] == t]
            data = Data()
            data.num_nodes = int(num_nodes)
            data.edge_index = torch.tensor(sf[['x', 'y']].values.T, dtype=torch.long)
            data.edge_attr = torch.tensor(sf[['w', 't_']].values, dtype=torch.long)
            data_list.append(data)
            
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
