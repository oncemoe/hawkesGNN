import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
from utils import seed_everything, generate_random_seeds, save_result
from models.base import BaseLPModel
from train import LinkPrediction
from torch_geometric.loader import NeighborLoader

class ScalableLinkPrediction(LinkPrediction):

    def train_epoch(self, model:BaseLPModel, loader, device, optimizer):
        model.train()
        loss_list = []
        count_list = []
        for batch in loader:
            assert len(batch) == 2  # scalable only for pair of snaps
            data, target = batch[0], batch[1]
            
            pos_edges, neg_edges = self.negative_sampling(target, device)
            alpha = neg_edges.size(1) / pos_edges.size(1)
            count = pos_edges.size(1)+neg_edges.size(1)
            loader = NeighborLoader(
                data,
                num_neighbors=[-1] * self.n_layers,
                batch_size=self.batch_size
            )
            
            H = torch.zeros((model.n_node, self.n_hidden), device='cpu')
            #H = torch.zeros((model.n_node, self.n_hidden), device=device, requires_grad=True)
            H_grad = torch.zeros_like(H)
            losses = []

            # Step 1      
            with torch.no_grad():
                for minibatch in loader:
                    bs = minibatch.batch_size
                    h = model(minibatch.to(device))
                    H[minibatch.n_id[:bs]] = h[:bs].cpu()

            # loss = model.train_step(H, pos_edges, neg_edges)
            # optimizer.zero_grad()
            # loss.backward()
            # H_grad_real = H.grad.cpu()
            # H = H.cpu()
            # H_grad = torch.zeros_like(H)

            # Step 2
            optimizer.zero_grad()
            for i, edges in enumerate([pos_edges, neg_edges]):
                for edge in torch.split(edges, self.batch_size*8, dim=1):
                    h1 = Variable(H[edge[0]], requires_grad=True).to(device)
                    h2 = Variable(H[edge[1]], requires_grad=True).to(device)
                    z, y_hat = model.predictor(h1, h2, raw=True)
                    if i == 0:
                        dLdz = (1-y_hat).detach() * alpha
                        loss = -torch.log(y_hat + 1e-15) * alpha
                    else:
                        dLdz = y_hat.detach()
                        loss = -torch.log(1 - y_hat + 1e-15)
                    h1.retain_grad()
                    h2.retain_grad()
                    (loss.sum()/count).backward()
                    #z.backward(gradient=dLdz/count)
                    H_grad.index_add_(0, edge[0], h1.grad.cpu())
                    H_grad.index_add_(0, edge[1], h2.grad.cpu())
                    # H_grad[edge[0]] += h1.grad.cpu()
                    # H_grad[edge[1]] += h2.grad.cpu()
                    losses.append(loss.sum().item()/count)  # for visualize
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
            optimizer.step()
            loss_list.append(np.sum(losses))
            count_list.append(count)
            # print("h_grad diff", (H_grad - H_grad_real).abs().sum())
            # print(H_grad[:10])
            # print(H_grad_real[:10])

            # Step 3
            optimizer.zero_grad()
            for minibatch in loader:
                bs = minibatch.batch_size
                h:torch.Tensor = model(minibatch.to(device))[:bs]
                h.backward(gradient=H_grad[minibatch.n_id[:bs].cpu()].to(device))
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
            optimizer.step()

            del pos_edges
            del neg_edges

        total_loss = np.array(loss_list)
        total_count = np.array(count_list)
        return (total_loss * total_count).sum() / total_count.sum()
