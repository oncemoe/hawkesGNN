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

class HTGNLinkPrediction(LinkPrediction):
    """from https://github.com/marlin-codes/HTGN/blob/main/script/main.py"""

    def __init__(self, args, build_model) -> None:
        super().__init__(args, build_model)
        self.loss = ReconLoss()

    def train_epoch(self, model:BaseLPModel, dataset, device, optimizer):
        model.train()
        loss_list = []
        count_list = []
        for data in dataset:        
            data = data.to(device)
            pos_edges, neg_edges = self.negative_sampling(data, device)

            z = model(data.edge_index, data.x)
            loss = self.loss(z, pos_edges, neg_edges)

            loss_list.append(loss.item())
            count_list.append(pos_edges.size(1)+neg_edges.size(1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
            optimizer.step()
            model.update_hiddens_all_with(z)

            del pos_edges
            del neg_edges

        total_loss = np.array(loss_list)
        total_count = np.array(count_list)
        return (total_loss * total_count).sum() / total_count.sum(), z.detach()

    @torch.no_grad()
    def test(self, model: BaseLPModel, dataset, z, device):
        model.eval()
        loss_list = []
        result_list = [[] for _ in range(7)]
        count_list = []
        for data in dataset:
            data = data.to(device)

            pos_edges, neg_edges, _ = self.prepare_test_edges(data, device)
            pos_out, neg_out = self.loss.predict(z, pos_edges, neg_edges)

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
        

    def train(self, args, model, dataset, split):
        result_list = []
        count_list = []
        time_usage_list = []    
        # https://github.com/pytorch/pytorch/issues/113758
        optimizer = torch.optim.Adam(params=model.parameters(),weight_decay=args.weight_decay,lr=args.lr, foreach=False)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)

        best_loss = np.inf# if t == 0 else val_loss
        best_mrr = 0
        wandering = 0
        
        pbar = tqdm(total=args.epochs)
        for epoch in range(1, 1 + args.epochs):
            model.init_hiddens()

            t0 = time.perf_counter()            
            train_loss, z = self.train_epoch(model, dataset[:split[0]], args.device, optimizer)
            t1 = time.perf_counter()
            lr_scheduler.step()

            t2 = time.perf_counter()
            val_loss, mets = self.test(model, dataset[split[0]:split[1]], z, args.device)
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
        pbar.close()
            
        model.load_state_dict(state_dict)
        with torch.no_grad():
            model.init_hiddens()
            for data in dataset[:split[1]]:        
                data = data.to(args.device)
                z = model(data.edge_index, data.x)
        _, result = self.test(model, dataset[split[1]:split[2]], z, args.device)
        
        return model, time_usage_list, result

    def main(self, args, factory, ds_split):
        ds, split = ds_split[0], ds_split[1]
        mrr_lists = []
        random_seeds = generate_random_seeds(seed=args.seed, nums=args.runs)
        gpu_usage_list = []
        time_usage_list = []
        for run in range(args.runs):
            seed_everything(random_seeds[run])
            model = self.build_model(args, factory)
            model, epoch_time, mrr = self.train(args, model, ds, split)
            print(f'Test Metric: {mrr[1]:.4f}, {mrr[2]:.4f}, {mrr[3]:.4f}, {mrr[4]:.4f}')
            mrr_lists.append(mrr)
            time_usage_list.append(epoch_time)
            gpu_mem_alloc = torch.cuda.max_memory_allocated(args.device) / 1000000
            gpu_usage_list.append(gpu_mem_alloc)

        is_meta = "_meta" if args.roland_is_meta else ""
        save_result(f"htgn_{args.roland_updater}_{args.dataset}{is_meta}.log", args, mrr_lists, gpu_usage_list, time_usage_list, not args.no_log)
