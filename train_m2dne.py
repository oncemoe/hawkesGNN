import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
from utils import seed_everything, generate_random_seeds, save_result
from models.base import BaseLPModel
import copy
import time
from utils import make_negative_adj,  calculate_row_mrr, calculate_sample_mrr, make_full_adj
from train import LinkPrediction
from models.htgn.loss import ReconLoss

class M2DNELinkPrediction(LinkPrediction):
    """from https://github.com/marlin-codes/HTGN/blob/main/script/main.py"""

    def __init__(self, args, build_model) -> None:
        super().__init__(args, build_model)

    def train_epoch(self, model, dataset, device, optimizer):
        model.train()
        loss_list = []
        count_list = []
        for batch in dataset:        
            target = batch[-1]
            if len(batch) > 2:
                data = [d.to(device) for d in batch[:-1]]
            else:
                data =batch[0].to(device)

            pos_edges, neg_edges = self.negative_sampling(target, device)

            pos_out = model(data, pos_edges)
            pos_loss = -torch.log(pos_out + 1e-15)
        
            neg_out = model(data, neg_edges)
            neg_loss = -torch.log(1 - neg_out + 1e-15)

            alpha =  len(neg_loss)/len(pos_loss)
            loss = torch.cat([pos_loss * alpha, neg_loss]).mean()

            loss_list.append(loss.item())
            count_list.append(pos_edges.size(1)+neg_edges.size(1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
            optimizer.step()

            del pos_edges
            del neg_edges

        total_loss = np.array(loss_list)
        total_count = np.array(count_list)
        return (total_loss * total_count).sum() / total_count.sum()

    @torch.no_grad()
    def test(self, model, dataset, device, return_full_mrr=False):
        model.eval()
        loss_list = []
        result_list = [[] for _ in range(7)]
        count_list = []
        for batch in dataset:
            target = batch[-1]
            if len(batch) > 2:
                data = [d.to(device) for d in batch[:-1]]
            else:
                data =batch[0].to(device)

            pos_edges, neg_edges, _ = self.prepare_test_edges(target, device)
            pos_out = model(data, pos_edges)        
            neg_out = model(data, neg_edges)

            pos_loss = -torch.log(pos_out + 1e-15)    
            neg_loss = -torch.log(1 - neg_out + 1e-15)
            alpha =  len(neg_loss)/len(pos_loss)
            loss = torch.cat([pos_loss * alpha, neg_loss]).mean()
            loss_list.append(loss.item())

            del pos_edges
            del neg_edges
            del data

            # TODO check pos_out and neg_out is arranged correctly
            res = calculate_sample_mrr(pos_out, neg_out, self.n_neg_test)
            for i, r in enumerate(res):
                result_list[i+1].append(r.item())
            result_list[0].append(0)
            count_list.append(len(pos_out))

        result = np.array(result_list)
        count = np.array(count_list)
        # loss, [mrr@row, mrr@1000, hit@1, hit@3, hit@10, rocauc, ap]
        return np.mean(loss_list), (result * count).sum(1) / count.sum()
        
