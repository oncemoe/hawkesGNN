import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from torch.sparse import FloatTensor as SparseFloatTensor
from  torch_geometric.typing import torch_sparse
from torch_geometric.nn import GATConv
import numpy as np
import math
import random
from sklearn.metrics import roc_auc_score
from evaluator import Evaluator
import datetime

def generate_random_seeds(seed, nums):
    random.seed(seed)
    return [random.randint(1, 999999999) for _ in range(nums)]


def seed_everything(seed=0):
    """
    Fix random process by a seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_negative_adj(target):
    t = torch.zeros((target.num_nodes, 1))
    exist = target.edge_index[0].flatten().unique()
    t[exist] = 1
    t = t.repeat(1, target.num_nodes)
    r = t - target.adj.cpu()
    r.fill_diagonal_(0)
    return r.to_sparse()

def make_full_adj(target):
    num_nodes = target.num_nodes
    t = torch.zeros((num_nodes, num_nodes))
    exist = target.edge_index[0].flatten().unique()
    t[exist] = 1
    t.fill_diagonal_(0)
    return t.to_sparse()

# not used, prone to bugs
def calculate_row_mrr(pos_adj, pos_pred, neg_adj, neg_pred):
    indices = torch.cat([pos_adj.coalesce().indices(), neg_adj.coalesce().indices()], dim=1)
    values = torch.cat([pos_pred, neg_pred]).flatten()
    #print(indices.shape, values.shape)
    mat = torch.sparse_coo_tensor(indices, values, pos_adj.size()).to_dense()
    mrr_list = []
    for i in range(pos_adj.size(0)):
        pos = pos_adj[i].coalesce().indices()
        if pos.shape[1] > 0:
            ranks = mat[i].argsort(descending=True)
            mask = torch.zeros(len(ranks), dtype=torch.bool)
            mask[pos] = 1
            mrr = torch.arange(1, len(ranks)+1)[mask[ranks]]
            mrr = (1 / mrr).sum() / len(mrr)
            mrr_list.append(mrr)
    return np.mean(mrr_list)
    
# keep same with EvolveGCN
# https://github.com/IBM/EvolveGCN/blob/master/logger.py#L206
def calculate_row_mrr(target, full_adj, full_pred, device):
    indices = full_adj.coalesce().indices().to(device)
    values = full_pred.flatten().to(device)
    mat = torch.sparse_coo_tensor(indices, values, full_adj.size()).to_dense()
    exist = target.edge_index[0].flatten().unique()
    mrr_list = []
    for u in exist:
        pos = target.adj[u].coalesce().indices().to(device)
        if pos.shape[1] > 0:
            ranks = mat[u].argsort(descending=True)
            mask = torch.zeros(len(ranks), dtype=torch.bool, device=device)
            mask[pos.flatten()] = 1
            mrr = torch.arange(1, len(ranks)+1, device=device)[mask[ranks]]
            mrr = (1 / mrr).sum() / len(mrr)
            mrr_list.append(mrr.item())
    return np.mean(mrr_list)

# keep pace with ogb
# https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/citation2/gnn.py#L159
def calculate_sample_mrr(pos_pred, neg_pred, num_neg_samples):
    e = Evaluator('EvolveGCN', 'mrr')
    r = e.eval({
        'y_pred_pos': pos_pred.view(-1), 'y_pred_neg': neg_pred.view(-1, num_neg_samples)
    })
    e = Evaluator('EvolveGCN', 'rocauc')

    # ap and auc of neg 1000 is tooo slow for large dataset.(cpu calculation)
    if num_neg_samples == 1:
        a = e.eval({
            'y_pred_pos': pos_pred.view(-1), 'y_pred_neg': neg_pred.view(-1)
        })
    else:
        a = {'rocauc':torch.zeros(1).sum(), 'ap': torch.zeros(1).sum()}
    return r['mrr_list'].mean(), r['hits@1_list'].mean(), r['hits@3_list'].mean(), r['hits@10_list'].mean(), a['rocauc'], a['ap']


def save_result(filename, args, mrr_lists, gpu_usage_list, time_usage_list, log=True):
    metric_lists = np.array(mrr_lists)
    mean = np.around(np.mean(mrr_lists, axis=0), decimals=4)
    std = np.around(np.std(mrr_lists, axis=0), decimals=4)
    
    epoch_time = np.mean(sum(time_usage_list, []))  # flat then mean
    train_times = [np.sum(x) for x in time_usage_list]

    print('Raw Metrics: ', metric_lists)
    from prettytable import PrettyTable
    print('Model:', args.model)
    t = PrettyTable(['', 'Mrr@row', 'Mrr@1000', 'Hits@1', 'Hits@3', 'Hits@10', 'AUC', 'AP'], float_format='.5')
    t.add_row(['Mean'] +  [m for m in mean])
    t.add_row(['Std'] +  [s for s in std])
    print(t)

    if not log:
        return
    fn = 'log/' + filename
    with open(fn, 'a') as f:
        f.write(f"Experiment finished at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        for k, v in args._get_kwargs():
            f.write(f"{k} = {v}\n")
        f.write(f"Average max GPU memory allocated: {np.mean(gpu_usage_list):.2f} MB\n")
        f.write(", ".join([f"{x:.2f}" for x in gpu_usage_list]))
        f.write('\n')
        f.write(f"Average Epoch time usage: {epoch_time:.2f} s\n")
        f.write(f"Average Train time usage: {np.mean(train_times):.2f} s\n")
        f.write(", ".join([f"{x:.2f}" for x in train_times]))
        f.write('\n')

        np.set_printoptions(suppress=True)
        f.write(np.array2string(metric_lists, precision=7, separator=','))
        f.write('\n')
        f.write(t.get_string())
        f.write('\n')


################################
#  below not used
################################
def ogb_eval_hits(y_pred_pos, y_pred_neg, K):
    '''
        compute Hits@K
        For each positive target node, the negative target nodes are the same.

        y_pred_neg is an array.
        rank y_pred_pos[i] against y_pred_neg for each i
    '''
    if len(y_pred_neg) < K:
        return {'hits@{}'.format(K): 1.}

    kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
    hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)
    return {'hits@{}'.format(K): hitsK}


def ogb_eval_mrr(y_pred_pos, y_pred_neg):
        '''
            compute mrr
            y_pred_neg is an array with shape (batch size, num_entities_neg).
            y_pred_pos is an array with shape (batch size, )
        '''
        # calculate ranks
        y_pred_pos = y_pred_pos.view(-1, 1)
        # optimistic rank: "how many negatives have a larger score than the positive?"
        # ~> the positive is ranked first among those with equal score
        optimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
        # pessimistic rank: "how many negatives have at least the positive score?"
        # ~> the positive is ranked last among those with equal score
        pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
        hits1_list = (ranking_list <= 1).to(torch.float)
        hits3_list = (ranking_list <= 3).to(torch.float)
        hits10_list = (ranking_list <= 10).to(torch.float)
        mrr_list = 1./ranking_list.to(torch.float)

        return {'hits@1_list': hits1_list,
                    'hits@3_list': hits3_list,
                    'hits@10_list': hits10_list,
                    'mrr_list': mrr_list}


def ogb_eval_rocauc(y_pred_pos, y_pred_neg):
    '''
        compute rocauc
    '''
    y_pred_pos_numpy = y_pred_pos.cpu().numpy()
    y_pred_neg_numpy = y_pred_neg.cpu().numpy()
    y_true = np.concatenate([np.ones(len(y_pred_pos_numpy)), np.zeros(len(y_pred_neg_numpy))]).astype(np.int32)
    y_pred = np.concatenate([y_pred_pos_numpy, y_pred_neg_numpy])
    rocauc = roc_auc_score(y_true, y_pred)
    return {'rocauc': rocauc}


if __name__ == "__main__":
    from torch_geometric.data import Data

    snap = Data()
    indices = torch.Tensor([[0, 0, 1], [1, 2, 2]]).long()
    snap.edge_index = indices
    snap.num_nodes = 3
    snap.adj = torch.sparse_coo_tensor(indices, torch.ones(len(indices[0]))).coalesce()

    full_adj = make_full_adj(snap)
    full_pred = torch.Tensor([2, 1, 3, 4])

    mrr = calculate_row_mrr(snap, full_adj, full_pred, 'cpu')

    assert mrr == 0.875