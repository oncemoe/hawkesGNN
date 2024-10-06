import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils import data as td
from typing import List, Optional, Tuple, Union
from queue import Queue
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops


def simple_to_undirected(
    edge_index: Tensor,
    edge_attr: Optional[Union[Tensor, List[Tensor]]] = None,
):
    r"""
    same as torch_geometric.utils.to_undirected but no coalesce
    """
    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    if edge_attr is not None and isinstance(edge_attr, torch.Tensor):
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    elif edge_attr is not None:
        edge_attr = [torch.cat([e, e], dim=0) for e in edge_attr]

    return (edge_index, edge_attr)


# @torch.no_grad()
# def safe_negative_sampling(target, num_neg_samples=1000, device='cpu', allow_self=False):
#     if target.num_nodes < 10000:
#         return fast_negative_sampling(target, num_neg_samples, device, allow_self)
#     else:
#         return iterate_negative_sampling(target, num_neg_samples, device, allow_self)

# @torch.no_grad()
# def fast_negative_sampling(target, num_neg_samples=1000, device='cpu', allow_self=False):
#     """ Sampling exactlly num_neg_samples for every positive edge"""
#     t = torch.zeros((target.num_nodes, 1), device=device)
#     exist, appears = target.edge_index.flatten().unique(return_counts=True)
#     start = target.adj.coalesce().indices()[0]  # target.edge_index[0]  # make sure edge are sorted
#     mapping = {x:i for i, x in enumerate(exist.numpy())}
#     t[exist.to(device)] = 1
#     t = t.repeat(1, target.num_nodes)
#     if not allow_self:
#         t = t - target.adj.to(device)
#     t.fill_diagonal_(0)
#     # sorted matrix, negative edge ids at top
#     s = t[start.to(device)].argsort(axis=1, descending=True)
#     del t
#     select = torch.stack([
#         torch.randperm(target.num_nodes-appears[mapping[u]], device=device)[:num_neg_samples] for u in start.numpy()
#     ])
#     return torch.stack([
#         start.view(-1, 1).repeat(1, num_neg_samples).flatten(), 
#         s.gather(1, select).flatten().cpu()
#     ])


@torch.no_grad()
def safe_negative_sampling(target, num_neg_samples=1000, device='cpu', allow_self=False):
    """ Sampling exactlly num_neg_samples for every positive edge"""
    avoid_edge_index = target.edge_index
    if not allow_self:
        avoid_edge_index,_ = add_remaining_self_loops(avoid_edge_index)
    scale = pow(10, len(str(target.num_nodes)))
    avoid = (avoid_edge_index[0] * scale + avoid_edge_index[1]).to(device)

    result = []
    for _ in range(num_neg_samples):
        u = target.edge_index[0].to(device)
        i = torch.arange(len(u), device=device)
        r = []
        while len(u) > 0:
            v = torch.randint(0, target.num_nodes, (len(u),), device=device)
            e = u * scale + v
            m = torch.isin(e, avoid)
            r.append(torch.stack([i[~m], u[~m], v[~m]]))
            u = u[m]
            i = i[m]
        arr = torch.cat(r, dim=1)
        result.append(arr[1:, arr[0].argsort()])
        del u
        del i
    return torch.cat(result, dim=1).cpu()




class EvolveGCNDS(td.Dataset):
    def __init__(self, ds, start, size, window) -> None:
        super().__init__()
        self.ds = ds
        self.start = start
        self.size = size
        self.window = window
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.ds[self.start + index - self.window: self.start + index] # 0, 10


class GraphPairDS(td.Dataset):
    def __init__(self, ds, start=0, end=None) -> None:
        super().__init__()
        self.start = start
        self.end = len(ds) if end is None else end
        self.ds = ds[start:end]
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        return self.ds[index]


class LRUUpdater:
    def __init__(self, dataset, k):
        num_nodes = dataset[0].num_nodes
        self.k = k
        self.buf = [Queue(maxsize=k) for _ in range(num_nodes)]
        self.edge = torch.cat([d.edge_index for d in dataset], dim=1)
        self.attr = torch.cat([d.edge_attr for d in dataset])
        self.offset = 0
        assert self.edge.shape[1] == self.attr.shape[0]

    def update(self, data):
        edge_index = data.edge_index.T
        indices = torch.arange(self.offset, self.offset+len(edge_index), dtype=torch.long, device=edge_index.device)
        edge_index = torch.cat([edge_index, indices.view(-1,1)], axis=1)
        edge_attr = data.edge_attr
        time_seq = edge_attr[:, -1].argsort()
        edge = edge_index[time_seq].cpu().long().numpy()
        
        for idx in range(len(edge)):
            u,v,i = edge[idx]
            q = self.buf[u]
            if q.full():
                q.get()
            q.put(i)
        self.offset += len(edge)
    
    def get(self):
        d = Data()
        d.num_nodes = len(self.buf)
        indices = np.hstack([np.array(q.queue) for q in self.buf])
        indices = torch.from_numpy(indices).long()
        #indices = np.hstack([np.stack([np.ones(q.qsize()) * i, np.array(q.queue)]) for i, q in enumerate(self.buf)])
        d.edge_index = self.edge[:, indices]
        #d.adj = torch.sparse_coo_tensor(d.edge_index, torch.ones(len(indices)), d.size()).coalesce()
        d.edge_attr = self.attr[indices]
        #attr = torch.cat([dataset[t-j].edge_attr.cpu() for j in range(0, window)], dim=0).double()
        #data.edge_attr = attr
        mapping = np.hstack([np.arange(len(q.queue)) + self.k * i for i,q in enumerate(self.buf)])
        d.mapping = torch.from_numpy(mapping)
        return d