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

class RolandLinkPrediction(LinkPrediction):

    def train_epoch(self, model:BaseLPModel, batch, device, optimizer):
        model.train()

        data, target, H_prev = batch[0], batch[1], batch[2]
        data = data.to(device)
        h, _ = model(data, H_prev)
        
        pos_edges = target.edge_index.to(device)
        neg_edges = self.negative_sampling(target).to(device)
        loss = model.train_step(h, pos_edges, neg_edges)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
        optimizer.step()

        del pos_edges
        del neg_edges
        del data

        return loss.item()


    @torch.no_grad()
    def test(self, model: BaseLPModel, batch, device, return_full_mrr=True):
        model.eval()
        result = [0] * 7

        data, target, H_prev = batch[0], batch[1], batch[2]
        data = data.to(device)
        h, _ = model(data, H_prev)
        
        pos_edges, neg_edges, idx = self.prepare_test_edges(target, device)
        h, pos_out, neg_out, loss = model.test_step(h, pos_edges, neg_edges, idx)
        
        # TODO check pos_out and neg_out is arranged correctly
        res = calculate_sample_mrr(pos_out, neg_out, self.n_neg_test)
        for i, r in enumerate(res):
            result[i+1] = r.item()
        
        if return_full_mrr:   # to accelate training process
            full_adj = make_full_adj(target)
            full_out = model.predict(h, full_adj.indices().to(device))
            mrr = calculate_row_mrr(target, full_adj, full_out, device)
            result[0] = mrr

        del pos_edges
        del neg_edges
        del data

        # loss, [mrr@row, mrr@1000, hit@1, hit@3, hit@10, rocauc, ap]
        return loss.item(), result, len(pos_out)
        

    # from https://github.com/snap-stanford/roland/blob/master/graphgym/contrib/train/train_live_update_fixed_split.py
    def train(self, args, model, dataset, split):
        task_range = split[2] - 1
        result_list = []
        count_list = []
        time_usage_list = []
        H_prev = None
        meta_model = None
        for t in range(task_range):
            if args.roland_is_meta and meta_model is not None:
                model.load_state_dict(copy.deepcopy(meta_model))

            snaps = (dataset[t], dataset[t+1], H_prev)
            val_loss, mets, _cnt = self.test(model, snaps, args.device, args.row_mrr)
            result_list.append(mets)
            count_list.append(_cnt)

            optimizer = torch.optim.Adam(params=model.parameters(),weight_decay=args.weight_decay,lr=args.lr)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)

            best_loss = np.inf# if t == 0 else val_loss
            best_mrr = mets[1]
            wandering = 0
            
            pbar = tqdm(total=args.epochs)
            for epoch in range(1, 1 + args.epochs):
                t0 = time.perf_counter()
                train_loss = self.train_epoch(model, snaps, args.device, optimizer)
                t1 = time.perf_counter()
                lr_scheduler.step()

                t2 = time.perf_counter()
                val_loss, mets, _ = self.test(model, snaps, args.device, False)
                mrr = mets[1]
                t3 = time.perf_counter()

                if best_loss > val_loss:
                    best_loss = val_loss
                    best_mrr = mrr
                    wandering = 0
                    state_dict = copy.deepcopy(model.state_dict())
                else:
                    wandering += 1

                pbar.set_description(f"S({t}), E({epoch:02d}) loss={train_loss:.4f}/{val_loss:.4f},{mrr:.4f}, time={t1-t0:.2f}s/{t3-t2:.2f}s, best={best_loss:.4f}<{best_mrr:.4f}>|{wandering}")
                pbar.update(1)

                if wandering > args.patiance:
                    break
                time_usage_list.append(t3-t0)
            pbar.close()

            model.eval()
            with torch.no_grad():
                data = dataset[t].to(args.device)
                _, H_prev = model(data, H_prev)
                del data
            del snaps
            
            # model.load_state_dict(state_dict)
            if args.roland_is_meta:
                if meta_model is None:
                    meta_model = copy.deepcopy(state_dict)
                else:
                    out = dict()
                    for key in meta_model.keys():
                        param1 = meta_model[key].detach().clone()
                        param2 = state_dict[key].detach().clone()
                        out[key] = (1 - args.roland_alpha) * param1 + args.roland_alpha * param2
                    meta_model = out
            else:
                model.load_state_dict(state_dict)
        test_snap_num = split[2] - split[1]
        result = np.array(result_list)[-test_snap_num:]
        count = np.array(count_list)[-test_snap_num:]
        return model, time_usage_list, (result * count.reshape(-1, 1)).sum(0) / count.sum()



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
        save_result(f"roland_{args.roland_updater}_{args.dataset}{is_meta}.log", args, mrr_lists, gpu_usage_list, time_usage_list, not args.no_log)
