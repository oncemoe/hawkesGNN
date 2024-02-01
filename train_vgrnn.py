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

# its very different, so...
class VgrnnLinkPrediction(LinkPrediction):
    def train_epoch(self, model:BaseLPModel, ds, device, optimizer):
        model.train()
        loss_list = []
        count_list = []
        H_prev = None
        for i in range(len(ds)-1):
            data = ds[i].to(device)
            target = ds[i+1]
            
            kld_loss, nll_loss, enc_mean, prior_mean, H_prev = model(data, H_prev)

            pos_edges, neg_edges = self.negative_sampling(target, device)
            loss = model.train_step(enc_mean, pos_edges, neg_edges) + kld_loss + nll_loss
            loss_list.append(loss.item())
            count_list.append(pos_edges.size(1)+neg_edges.size(1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
            optimizer.step()

            del pos_edges
            del neg_edges
            del data

        total_loss = np.array(loss_list)
        total_count = np.array(count_list)
        return (total_loss * total_count).sum() / total_count.sum()


    @torch.no_grad()
    def test(self, model: BaseLPModel, ds, H_prev, device, return_full_mrr=True):
        model.eval()
        loss_list = []
        result_list = [[] for _ in range(7)]
        count_list = []
        for t in range(len(ds)-1):
            data, target = ds[t], ds[t+1]
            _, _, _, prior_mean, H_prev = model(data.to(device), H_prev)
            
            pos_edges, neg_edges, idx = self.prepare_test_edges(target, device)
            h, pos_out, neg_out, loss = model.test_step(prior_mean, pos_edges, neg_edges, idx)
            loss_list.append(loss.item())
            count_list.append(len(pos_out))
            
            # TODO check pos_out and neg_out is arranged correctly
            res = calculate_sample_mrr(pos_out, neg_out, self.n_neg_test)
            for i, r in enumerate(res):
                result_list[i+1].append(r.item()) 
            
            if return_full_mrr:   # to accelate training process
                full_adj = make_full_adj(target)
                full_out = model.predict(prior_mean, full_adj.indices().to(device))
                mrr = calculate_row_mrr(target, full_adj, full_out, device)
                result_list[0].append(mrr)
            else:
                result_list[0].append(0)

            del pos_edges
            del neg_edges

        result = np.array(result_list)
        # loss, [mrr@row, mrr@1000, hit@1, hit@3, hit@10, rocauc, ap]
        return np.mean(loss_list), result.reshape(7, -1).T, np.array(count_list)
    
    def train(self, args, model, dataset, split):
        result_list = []
        H_prev = None

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
            train_loss = self.train_epoch(model, dataset[:split[0]], args.device, optimizer)
            t1 = time.perf_counter()
            lr_scheduler.step()

            if epoch % args.eval_steps == 0:
                t2 = time.perf_counter()
                val_loss, mets, _ = self.test(model, dataset[split[0]-1:split[1]], H_prev, args.device, False)
                mets = mets.mean(0)
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

        test_loss, result_list, count_list = self.test(model, dataset, None, args.device, args.row_mrr)
        test_snap_num = split[2] - split[1]
        result = np.array(result_list)[-test_snap_num:]
        count = np.array(count_list)[-test_snap_num:]
        return model, time_usage_list, (result * count.reshape(-1, 1)).sum(0) / count.sum()

    def main(self, args, factory, ds_split):
        mrr_lists = []
        random_seeds = generate_random_seeds(seed=args.seed, nums=args.runs)
        gpu_usage_list = []
        time_usage_list = []
        print(ds_split[0][-1])
        for run in range(args.runs):
            seed_everything(random_seeds[run])
            model = self.build_model(args, factory)
            model, epoch_time, mrr = self.train(args, model, ds_split[0], ds_split[1])
            print(f'Test Metric: {mrr[1]:.4f}, {mrr[2]:.4f}, {mrr[3]:.4f}, {mrr[4]:.4f}')
            mrr_lists.append(mrr)
            time_usage_list.append(epoch_time)
            gpu_mem_alloc = torch.cuda.max_memory_allocated(args.device) / 1000000
            gpu_usage_list.append(gpu_mem_alloc)

        save_result(f"vrgnn_{args.dataset}.log", args, mrr_lists, gpu_usage_list, time_usage_list, not args.no_log)
