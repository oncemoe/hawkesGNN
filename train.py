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
from torch_geometric.utils import add_remaining_self_loops


class LinkPrediction:
    def __init__(self, args, build_model) -> None:
        self.n_neg_train = args.n_neg_train        # number of negitive samples to sample
        self.n_neg_test = args.n_neg_test
        self.neg_method = 'sparse'  # dense or sparse
        self.clip_grad_norm = 2.0
        self.build_model = build_model
        self.is_bipart = False
        
        self.batch_size = args.batch_size
        self.n_layers = args.n_layers
        self.n_hidden = args.n_hidden
    
    def set_jodie(self, dst_min, dst_max):
        self.is_bipart = True
        self.min_dst_idx = dst_min
        self.max_dst_idx = dst_max
        
        
    @torch.no_grad()
    def fast_negative_sampling(self, edge_index, num_nodes, num_neg_samples=1000, device='cpu', allow_self=False):
        """ Sampling exactlly num_neg_samples for every positive edge"""
        avoid_edge_index = edge_index
        if not allow_self:
            avoid_edge_index,_ = add_remaining_self_loops(avoid_edge_index)
        scale = pow(10, len(str(num_nodes)))
        avoid = (avoid_edge_index[0] * scale + avoid_edge_index[1]).to(device)

        u = edge_index[0].to(device).repeat(int(num_neg_samples * 1.2))
        v = torch.randint(0, num_nodes, (len(u),), device=device)
        e = u * scale + v
        m = torch.isin(e, avoid)
        return torch.stack([u[~m], v[~m]])


    def negative_sampling(self, target, device=None):
        if self.is_bipart:
            pos_edge = target.edge_index
            neg_dst = torch.randint(self.min_dst_idx, self.max_dst_idx + 1, (pos_edge.size(1), ),
                                dtype=torch.long, device=pos_edge.device)
            return pos_edge, torch.stack([pos_edge[0], neg_dst])
        else:
            if 'raw_edge_index' in target: 
                edges = target.raw_edge_index
            else:
                edges = target.edge_index    
            neg_edges = self.fast_negative_sampling(edges, target.num_nodes, num_neg_samples=self.n_neg_train)
            if device is not None:
                return edges.to(device), neg_edges.to(device)
            return edges, neg_edges
            # return negative_sampling(
            #     edges, 
            #     num_nodes=target.num_nodes,
            #     num_neg_samples=edges.size(1) * self.n_neg_train, 
            #     method=self.neg_method)

    def prepare_test_edges(self, target, device):
        if 'raw_edge_index' in target: 
            # for simple implement for lstm+gcn
            # raw_edge_index = edge_index
            # edge_index = to_undriected(edge_index)
            pos_edges = target.raw_edge_index.to(device)
        else:
            pos_edges = target.edge_index.to(device)
        neg_edges = target.neg_edge_index.to(device)
        
        #TODO: its better to call unique in preprocess, but, i dont have much time...
        if target.num_nodes **2 * 10 < target.neg_edge_index.shape[1]:
            neg_edges, idx = neg_edges.unique(dim=1, return_inverse=True)
        else:
            idx = None
        return pos_edges, neg_edges, idx
    
    def train_epoch(self, model:BaseLPModel, loader, device, optimizer):
        model.train()
        loss_list = []
        count_list = []
        for batch in loader:
            target = batch[-1]
            if len(batch) > 2:
                data = [d.to(device) for d in batch[:-1]]
            else:
                data =batch[0].to(device)

            pos_edges, neg_edges = self.negative_sampling(target, device)
            loss = model.train_step(data, pos_edges, neg_edges)

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
    def test(self, model: BaseLPModel, loader, device, return_full_mrr=True):
        model.eval()
        loss_list = []
        result_list = [[] for _ in range(7)]
        count_list = []
        for batch in loader:
            target = batch[-1]
            if len(batch) > 2:
                data = [d.to(device) for d in batch[:-1]]
            else:
                data =batch[0].to(device)
            
            pos_edges, neg_edges, idx = self.prepare_test_edges(target, device)
            h, pos_out, neg_out, loss = model.test_step(data, pos_edges, neg_edges, idx)
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
        
        # # full_mrr is not fair, unstable, and time consuming, not used any more...
        # if return_full_mrr:   # to accelate training process
        #     full_adj = make_full_adj(target)
        #     full_out = model.predict(h, full_adj.indices().to(device))
        #     mrr = calculate_row_mrr(target, full_adj, full_out, device)
        #     result_list[0].append(mrr)
        # else:
        #     result_list[0].append(0)

        result = np.array(result_list)
        count = np.array(count_list)
        # loss, [mrr@row, mrr@1000, hit@1, hit@3, hit@10, rocauc, ap]
        return np.mean(loss_list), (result * count).sum(1) / count.sum()
        

    # for fix split setting
    def train(self, args, model, loaders):
        # https://github.com/pytorch/pytorch/issues/113758
        optimizer = torch.optim.Adam(params=model.parameters(),weight_decay=args.weight_decay,lr=args.lr, foreach=False)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)

        best_loss = float('inf')
        best_mrr = 0
        wandering = 0
        time_usage_list = []
        pbar = tqdm(total=args.epochs)
        for epoch in range(1, 1 + args.epochs):
            t0 = time.perf_counter()
            train_loss = self.train_epoch(model, loaders['train'], args.device, optimizer)
            t1 = time.perf_counter()
            lr_scheduler.step()

            if epoch % args.eval_steps == 0:
                t2 = time.perf_counter()
                # val_loss, mrr = test(model, loaders['val'], args.device, num_sample=0)
                val_loss, mets = self.test(model, loaders['val'], args.device, False)
                mrr = mets[1]
                t3 = time.perf_counter()

                if best_loss > val_loss:
                    best_loss = val_loss
                    best_mrr = mrr
                    wandering = 0
                    state_dict = copy.deepcopy(model.state_dict())
                else:
                    wandering += 1

                pbar.set_description(f"E({epoch:02d}) loss={train_loss:.4f}/{val_loss:.4f},{mrr:.4f}, time={t1-t0:.2f}s/{t3-t2:.2f}s, best={best_loss:.4f}<{best_mrr:.4f}>|{wandering}")
                pbar.update(1)

                if wandering > args.patiance:
                    break
            time_usage_list.append(t3-t0)
        model.load_state_dict(state_dict)
        return model, time_usage_list


    def main(self, args, factory, loaders):
        mrr_lists = []
        random_seeds = generate_random_seeds(seed=args.seed, nums=args.runs)
        gpu_usage_list = []
        time_usage_list = []
        for run in range(args.runs):
            for b in loaders['test']:
                print(b)
                break
            seed_everything(random_seeds[run])
            model = self.build_model(args, factory)
            model, epoch_time = self.train(args, model, loaders)
            loss, mrr = self.test(model, loaders['test'], args.device, args.row_mrr)
            print(f'Test Metric: {loss}|{mrr[1]:.4f}, {mrr[2]:.4f}, {mrr[3]:.4f}, {mrr[4]:.4f}')
            mrr_lists.append(mrr)
            time_usage_list.append(epoch_time)
            gpu_mem_alloc = torch.cuda.max_memory_allocated(args.device) / 1000000
            gpu_usage_list.append(gpu_mem_alloc)

        m = args.model
        save_result(f"Exp_{m}_{args.dataset}{'_minibatch' if args.minibatch else ''}.log", args, mrr_lists, gpu_usage_list, time_usage_list, not args.no_log)
