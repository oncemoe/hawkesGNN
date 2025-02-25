import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
from utils import seed_everything, generate_random_seeds, save_result, calculate_sample_mrr
from models.base import BaseLPModel
from train import LinkPrediction
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
import random
import time


def generate_random_seed():
    random.seed(int(time.time()))
    return random.randint(1, 999999999)

class MiniBatchLinkPrediction(LinkPrediction):
    """
    MiniBatch: experiment shows 1 negative for train is enough, therefore directly apply minibatch is fine
    """
    def train_epoch(self, model:BaseLPModel, loader, device, optimizer):
        model.train()
        loss_list = []
        count_list = []
        loss_fn = torch.nn.BCELoss(reduction='none')
        for batch in loader:  # snapshots
            assert len(batch) == 2  # scalable only for pair of snaps
            data, target = batch[0], batch[1]

            pos_edges, neg_edges = self.negative_sampling(target)
            alpha =  neg_edges.size(1)/pos_edges.size(1)
            edge_label = torch.cat([
                torch.ones((pos_edges.size(1), 1)), 
                torch.zeros((neg_edges.size(1), 1)), 
            ], dim=0)
            num_edges = edge_label.size(0)
            edge_weight = torch.cat([
                torch.ones((pos_edges.size(1), 1)) * alpha, 
                torch.ones((neg_edges.size(1), 1)), 
            ], dim=0)
            idx = torch.randperm(num_edges)

            loader = LinkNeighborLoader(
                data,
                num_neighbors=[-1] * self.n_layers, # if self.strategy=='all' else [20, 10, 5, 1][:self.n_layers],
                batch_size=self.batch_size,
                edge_label_index=torch.cat([pos_edges, neg_edges], dim=1),
                edge_label = torch.cat([edge_label,edge_weight], dim=1),
                subgraph_type='induced',  # induced keep temporal edges while bidirectional may merge two edges
            )

            losses = []
            optimizer.zero_grad()
            for minibatch in loader:
                minibatch.x.to(device)
                minibatch.edge_index.to(device)
                minibatch.edge_attr.to(device)
                h = model(minibatch.to(device))
                pred = model.predict(h, minibatch.edge_label_index.to(device))
                lw = minibatch.edge_label.to(device)
                label, weight = lw[:, 0].view(-1, 1), lw[:, 1].view(-1, 1)
                loss = loss_fn(pred, label) * weight
                loss = loss.mean() #loss.sum() / num_edges
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
                optimizer.step()
            
            loss_list.append(np.mean(losses))
            count_list.append(pos_edges.size(1)+neg_edges.size(1))
            del pos_edges
            del neg_edges

        total_loss = np.array(loss_list)
        total_count = np.array(count_list)
        return (total_loss * total_count).sum() / total_count.sum()



    @torch.no_grad()
    def test(self, model: BaseLPModel, loader, device, return_full_mrr=True):
        model.eval()
        loss_list = []
        result_list = [[] for _ in range(7)]
        count_list = []
        for batch in loader:
            assert len(batch) == 2  # scalable only for pair of snaps
            data, target = batch[0], batch[1]
            pos_edges = target.edge_index
            neg_edges = target.neg_edge_index

            loader = NeighborLoader(
                data,
                num_neighbors=[-1] * self.n_layers, # if self.strategy=='all' else [20, 10, 5, 1][:self.n_layers],
                batch_size=self.batch_size,
                subgraph_type='induced',
            )
            
            def predict(h, edge_index, bs=1024*1024):
                out = []
                for edge in torch.split(edge_index, bs, dim=1):
                    out.append(model.predictor(h[edge[0]].to(device), h[edge[1]].to(device)).cpu())
                return torch.cat(out)
    
            H = torch.zeros((model.n_node, self.n_hidden), device='cpu')
            for minibatch in loader:
                bs = minibatch.input_id.size(0) # wired bug of pyg NeighborLoader, (bs==1)
                h:torch.Tensor = model(minibatch.to(device))[:bs]
                H[minibatch.n_id[:bs].cpu()] = h[:bs].cpu()

            pos_out = predict(H, pos_edges)
            pos_loss = -torch.log(pos_out + 1e-15)
            neg_out = predict(H, neg_edges)
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
    
    
    

from torch_geometric.data import Data

class SequenceMiniBatchLinkPrediction(LinkPrediction):
    """
    MiniBatch: experiment shows 1 negative for train is enough, therefore directly apply minibatch is fine
    """
    def train_epoch(self, model:BaseLPModel, loader, device, optimizer):
        model.train()
        loss_list = []
        count_list = []
        loss_fn = torch.nn.BCELoss(reduction='none')
        for batch in loader:  # snapshots
            data_list, target = batch[0:-1], batch[-1]
            list_size = len(data_list)
            data = Data()
            data.x = data_list[0].x
            data.edge_index = torch.cat([d.edge_index for d in data_list], dim=1)
            data.edge_attr = torch.cat([torch.cat([d.edge_attr, torch.ones((len(d.edge_attr),1)) * i], dim=1) for i, d in enumerate(data_list)], dim=0)

            pos_edges, neg_edges = self.negative_sampling(target)
            alpha =  neg_edges.size(1)/pos_edges.size(1)
            edge_label = torch.cat([
                torch.ones((pos_edges.size(1), 1)), 
                torch.zeros((neg_edges.size(1), 1)), 
            ], dim=0)
            num_edges = edge_label.size(0)
            edge_weight = torch.cat([
                torch.ones((pos_edges.size(1), 1)) * alpha, 
                torch.ones((neg_edges.size(1), 1)), 
            ], dim=0)

            loader = LinkNeighborLoader(
                data,
                num_neighbors=[-1] * self.n_layers, # if self.strategy=='all' else [20, 10, 5, 1][:self.n_layers],
                batch_size=self.batch_size,
                edge_label_index=torch.cat([pos_edges, neg_edges], dim=1),
                edge_label = torch.cat([edge_label,edge_weight], dim=1),
                subgraph_type='induced',  # induced keep temporal edges while bidirectional may merge two edges
            )

            losses = []
            optimizer.zero_grad()
            for minibatch in loader:
                minibatch.x.to(device)
                minibatch.edge_index.to(device)
                minibatch.edge_attr.to(device)
                mini_list = []
                for i in range(list_size):
                    mask = minibatch.edge_attr[:, -1] == i
                    d = Data()
                    d.x = minibatch.x
                    d.edge_index = minibatch.edge_index[:, mask]
                    d.edge_attr = minibatch.edge_attr[mask, :-1]
                    mini_list.append(d.to(device))
                h = model(mini_list)
                pred = model.predict(h, minibatch.edge_label_index.to(device))
                lw = minibatch.edge_label.to(device)
                label, weight = lw[:, 0].view(-1, 1), lw[:, 1].view(-1, 1)
                loss = loss_fn(pred, label) * weight
                loss = loss.mean() #loss.sum() / num_edges
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
                optimizer.step()
            
            loss_list.append(np.mean(losses))
            count_list.append(pos_edges.size(1)+neg_edges.size(1))
            del pos_edges
            del neg_edges

        total_loss = np.array(loss_list)
        total_count = np.array(count_list)
        return (total_loss * total_count).sum() / total_count.sum()



    @torch.no_grad()
    def test(self, model: BaseLPModel, loader, device, return_full_mrr=True):
        model.eval()
        loss_list = []
        result_list = [[] for _ in range(7)]
        count_list = []
        for batch in loader:
            data_list, target = batch[0:-1], batch[-1]
            list_size = len(data_list)
            data = Data()
            data.x = data_list[0].x
            data.edge_index = torch.cat([d.edge_index for d in data_list], dim=1)
            data.edge_attr = torch.cat([torch.cat([d.edge_attr, torch.ones((len(d.edge_attr),1)) * i], dim=1) for i, d in enumerate(data_list)], dim=0)
            pos_edges = target.raw_edge_index
            neg_edges = target.neg_edge_index

            loader = NeighborLoader(
                data,
                num_neighbors=[-1] * self.n_layers, # if self.strategy=='all' else [20, 10, 5, 1][:self.n_layers],
                batch_size=self.batch_size,
                subgraph_type='induced',
            )
            
            def predict(h, edge_index, bs=1024*1024):
                out = []
                for edge in torch.split(edge_index, bs, dim=1):
                    out.append(model.predictor(h[edge[0]].to(device), h[edge[1]].to(device)).cpu())
                return torch.cat(out)
    
            H = torch.zeros((len(data.x), self.n_hidden), device='cpu')
            for minibatch in loader:
                mini_list = []
                for i in range(list_size):
                    mask = minibatch.edge_attr[:, -1] == i
                    d = Data()
                    d.x = minibatch.x
                    d.edge_index = minibatch.edge_index[:, mask]
                    d.edge_attr = minibatch.edge_attr[mask, :-1]
                    mini_list.append(d.to(device))
                bs = minibatch.input_id.size(0) # wired bug of pyg NeighborLoader, (bs==1)
                h:torch.Tensor = model(mini_list)[:bs]
                H[minibatch.n_id[:bs].cpu()] = h[:bs].cpu()

            pos_out = predict(H, pos_edges)
            pos_loss = -torch.log(pos_out + 1e-15)
            neg_out = predict(H, neg_edges)
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