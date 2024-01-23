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
    extract_zip,
)


class MetrLa(InMemoryDataset):
    r"""A traffic forecasting dataset based on Los Angeles
    Metropolitan traffic conditions. The dataset contains traffic
    readings collected from 207 loop detectors on highways in Los Angeles
    County in aggregated 5 minute intervals for 4 months between March 2012
    to June 2012.

    For further details on the version of the sensor network and
    discretization see: `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_

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
          - ~2,573.2
          - 0
          - 0
    """

    url = 'https://drive.switch.ch/index.php/s/Z8cKHAVyiDqkzaG/download'

    def __init__(self, root: str, normalized_k: int = 0.1, 
                 add_time_in_day=True, add_day_in_week=False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.normalized_k = normalized_k
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return ['metr_la.h5', 'distances_la.csv', 'sensor_locations_la.csv']

    @property
    def processed_file_names(self) -> str:
        return 'metla.pt'

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'metla_processed')
    

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)


    def process(self):
        df = pd.read_hdf(self.raw_paths[0]).reset_index(names='t')
        mapping = {int(v):k for k,v in enumerate(df.columns[1:])}
        dist = pd.read_csv(self.raw_paths[1])
        dist.columns = ['u', 'v', 'w']
        dist = dist[dist['u'].isin(set(mapping.keys())) & dist['v'].isin(set(mapping.keys()))]
        dist['u'] = dist['u'].apply(lambda x: mapping[x])
        dist['v'] = dist['v'].apply(lambda x: mapping[x])
        std = dist['w'].std()
        dist['w_'] = dist['w'].apply(lambda x: np.exp(-(x/std)**2))
        dist = dist[dist['w_'] > self.normalized_k]
        loc = pd.read_csv(self.raw_paths[2])
        loc.columns = ['index', 'id', 'lat', 'lon']
        loc['id'] = loc['id'].apply(lambda x: mapping[x])
        loc = loc.sort_values(by='id', ascending=True)
        assert len(dist) == 1515 + 207
        
        data = Data()
        data.num_nodes = len(mapping)
        data.num_times = len(df)
        data.edge_index = torch.tensor(dist[['u', 'v']].values.T, dtype=torch.long)
        data.edge_attr = torch.tensor(dist[['w', 'w_']].values, dtype=torch.float)
        data.mask = torch.tensor((df.iloc[:, 1:].values.T != 0), dtype=torch.float)
        data.y = torch.tensor(df.iloc[:, 1:].values.T, dtype=torch.float)
        data.loc = torch.tensor(loc[['lat', 'lon']].values, dtype=torch.float)
        
        x = []
        if self.add_time_in_day:
            df['tid'] = (df.t.values - df.t.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            x.append(torch.tensor(df['tid'].values.T, dtype=torch.float).view(1, -1))
        if self.add_day_in_week:
            df['diw'] = df.t.apply(lambda x: x.dayofweek)
            x.append(torch.tensor(df['diw'].values.T, dtype=torch.float).view(1, -1))
        if len(x) > 0:
                data.tx = torch.cat(x, axis=0)
        

        data = data if self.pre_filter is None else self.pre_filter(data)
        data = data if self.pre_transform is None else self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
