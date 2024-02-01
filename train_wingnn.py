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
import random 

class WinGNNLinkPrediction(LinkPrediction):
    
    @torch.no_grad()
    def test_snap(self, model, fast_weights, snaps, device, return_full_mrr=False):
        loss_list = []
        result = [0] * 7

        data, target = snaps[0], snaps[1]
        h = model(data.to(device), fast_weights)

        pos_edges, neg_edges, idx = self.prepare_test_edges(target, device)
        h, pos_out, neg_out, loss = model.test_step(h, pos_edges, neg_edges, idx)
        loss_list.append(loss.item())
        
        res = calculate_sample_mrr(pos_out, neg_out, self.n_neg_test)
        for i, r in enumerate(res):
            result[i+1] = r.item()

        # if return_full_mrr:   # accelate training process
        #     full_adj = make_full_adj(target)
        #     full_out = model.predict(h, full_adj.indices().to(device))
        #     mrr = calculate_row_mrr(target, full_adj, full_out, device)
        #     result[0] = mrr

        # loss, [mrr@row, mrr@1000, hit@1, hit@3, hit@10]
        return np.mean(loss_list), result
    
    
    
    def train_epoch(self, args, model:BaseLPModel, S_dw, ds_train, device, optimizer):
        model.train()
        beta = 0.89  # keep same with official
        i = 0
        mrr_list = []
        while i < len(ds_train) - args.window:
            if i != 0:
                i = random.randint(i, i + args.window)
            if i >= (len(ds_train) - args.window):
                break 
            ds_window = ds_train[i: i + args.window]
            i+=1

            fast_weights = list(model.parameters())
            losses = 0
            mrr_window_list = []
            for idx, data in enumerate(ds_window[:-2]):
                target = ds_window[idx+1]

                h = model(data.to(device), fast_weights)

                pos_edges, neg_edges = self.negative_sampling(target, device)
                loss = model.train_step(h, pos_edges, neg_edges)

                grad = torch.autograd.grad(loss, fast_weights)
                S_dw = list(map(lambda p: beta * p[1] + (1 - beta) * p[0] * p[0], zip(grad, S_dw)))
                fast_weights = list(
                        map(lambda p: p[1] - args.wingnn_maml_lr / (torch.sqrt(p[2]) + 1e-8) * p[0], zip(grad, fast_weights, S_dw)))

                # official implementaion is wrong! 
                # we are not to predict the input edge_index
                # but the future one
                snaps = (ds_window[idx+1], ds_window[idx+2]) 
                data, target = snaps
                h = model(data.to(device), fast_weights)
                pos_edges, neg_edges = self.negative_sampling(target, device)
                val_loss = model.train_step(h, pos_edges, neg_edges)
                _, mets = self.test_snap(model, fast_weights, snaps, device, False)
                
                xx = random.random() 
                if xx > args.wingnn_drop_snap:
                    losses += val_loss  #!!!!! so wired, sgd using val loss
                    mrr_window_list.append(mets[1])

            if len(mrr_window_list) > 0:
                losses /= len(mrr_window_list)
                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
                optimizer.step()
                mrr_list.append(np.mean(mrr_window_list))

        return np.mean(mrr_list), S_dw
    

    def test(self, args, model, S_dw, dataset, split):
        ds_test = dataset[split[1]-2:]
        fast_weights = list(model.parameters())
        loss_list = []
        result_list = []
        count_list = []

        for idx, data in enumerate(ds_test[:-2]):
            target = ds_test[idx+1]
            model.train()
            h = model(data.to(args.device), fast_weights)
            pos_edges, neg_edges, reidx = self.prepare_test_edges(target, args.device)
            h, pos_out, neg_out, loss = model.test_step(h, pos_edges, neg_edges, reidx)

            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - args.wingnn_maml_lr / (torch.sqrt(p[2]) + 1e-8) * p[0], zip(grad, fast_weights, S_dw)))

            # official implementaion is wrong! links to predict must be next snap!
            snaps = (ds_test[idx+1], ds_test[idx+2]) 
            model.eval()
            test_loss, mets = self.test_snap(model, fast_weights, snaps, args.device, self.n_neg_test)
            loss_list.append(test_loss.item())
            result_list.append(mets)
            count_list.append(len(pos_out))
        
        result = np.array(result_list)
        count = np.array(count_list)
        return np.mean(loss_list), (result * count.reshape(-1, 1)).sum(0) / count.sum()
    
        

    def train(self, args, model, dataset, split):
        S_dw = [0] * len(list(model.parameters()))
        
        # https://github.com/pytorch/pytorch/issues/113758
        optimizer = torch.optim.Adam(params=model.parameters(),weight_decay=args.weight_decay,lr=args.lr, foreach=False)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)

        best_mrr = 0
        wandering = 0
        state_dict = None
        time_usage_list = []
        for epoch in tqdm(range(1, 1 + args.epochs)):
            t0 = time.perf_counter()
            valid_mrr, S_dw = self.train_epoch(args, model, S_dw, dataset[:split[1]], args.device, optimizer)
            t1 = time.perf_counter()
            lr_scheduler.step()

            if valid_mrr > best_mrr:
                best_mrr = valid_mrr
                wandering = 0
                state_dict = copy.deepcopy(model.state_dict())
            else:
                wandering += 1
            
            if wandering > args.patiance:
                break
            time_usage_list.append(t1-t0)
        
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return model, S_dw, time_usage_list



    def main(self, args, factory, ds_split):
        ds, split = ds_split[0], ds_split[1]
        mrr_lists = []
        random_seeds = generate_random_seeds(seed=args.seed, nums=args.runs)
        gpu_usage_list = []
        time_usage_list = []
        for run in range(args.runs):
            seed_everything(random_seeds[run])
            model = self.build_model(args, factory)
            model, S_dw, epoch_time = self.train(args, model, ds, split)
            test_loss, mrr = self.test(args, model, S_dw, ds, split)
            print(f'Test Metric: {mrr[1]:.4f}, {mrr[2]:.4f}, {mrr[3]:.4f}, {mrr[4]:.4f}')
            mrr_lists.append(mrr)
            time_usage_list.append(epoch_time)
            gpu_mem_alloc = torch.cuda.max_memory_allocated(args.device) / 1000000
            gpu_usage_list.append(gpu_mem_alloc)

        save_result(f"wingnn_{args.dataset}.log", args, mrr_lists, gpu_usage_list, time_usage_list, not args.no_log)