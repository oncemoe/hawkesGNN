import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
from utils import seed_everything, generate_random_seeds, save_result
from train import LinkPrediction



def build_model(args, factory):
    if args.model in ['gcn', 'gat', 'hgcn', 'hgat']:
        from models.hawkes import HGNNLP
        model = HGNNLP(factory.num_nodes, factory.node_feats_dim, factory.edge_feats_dim,
                            args.n_hidden, dropout=args.dropout, bias=args.bias, name=args.model,
                            layers=args.n_layers, heads=args.heads, batch_norm=args.bn, norm=args.norm_type).to(args.device)
    elif args.model == 'dysat':
        from models.dysat import DySAT
        model = DySAT(factory.num_nodes, factory.node_feats_dim, args.n_hidden, window=args.window-1, # input len = args.window - 1
                            spatial_drop=args.dropout).to(args.device) 
    elif args.model in ['evolve-o', 'evolve-h']:
        from models.evolvegcnh import EvolveGCNLP
        model = EvolveGCNLP(factory.num_nodes, factory.node_feats_dim, args.n_hidden, dropout=args.dropout, name=args.model).to(args.device)
    elif args.model in ['lstmgcn']:
        from models.lstmgcn import LSTMGCN
        model = LSTMGCN(factory.num_nodes, factory.node_feats_dim, args.n_hidden, dropout=args.dropout, bias=args.bias).to(args.device)
    elif args.model == 'vgrnn':
        from models.vgrnn import VGRNN
        model = VGRNN(factory.num_nodes, factory.node_feats_dim, factory.edge_feats_dim, args.n_hidden, args.n_hidden, 
                        dropout=args.dropout).to(args.device)
    elif args.model == 'roland':
        from models.roland import RolandGNN
        model = RolandGNN(factory.num_nodes, factory.node_feats_dim, factory.edge_feats_dim, args.n_hidden, 
                            dropout=args.dropout, updater=args.roland_updater).to(args.device)
    elif args.model == 'wingnn':
        from models.wingnn import WinGNN
        model = WinGNN(factory.num_nodes, factory.node_feats_dim, factory.edge_feats_dim, args.n_hidden, 
                            dropout=args.dropout).to(args.device)
    elif args.model == 'htgn':
        from models.htgn.HTGN import HTGN
        model = HTGN(factory.num_nodes, factory.node_feats_dim, factory.edge_feats_dim,
                            args.n_hidden, dropout=args.dropout, window=args.window-1).to(args.device)
    elif args.model == 'graphmixer':
        from models.graphmixer import GraphMixer
        model = GraphMixer(factory.num_nodes, factory.node_feats_dim, factory.edge_feats_dim,
                            args.n_hidden, dropout=args.dropout).to(args.device)
    elif args.model == 'm2dne':
        from models.m2dne import M2DNE
        model = M2DNE(factory.num_nodes, factory.node_feats_dim, factory.edge_feats_dim,
                            args.n_hidden, dropout=args.dropout).to(args.device)
    elif args.model == 'ghp':
        from models.ghp import GHP
        model = GHP(factory.num_nodes, factory.node_feats_dim, factory.edge_feats_dim,
                            args.n_hidden, dropout=args.dropout).to(args.device)
    return model



if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.set_default_dtype(torch.float64)
    parser = argparse.ArgumentParser(description='')
    # run configuration
    parser.add_argument('--dataset', type=str, default='bitcoinotc')
    parser.add_argument('--model', type=str, default='hgat', choices=['gcn', 'gat', 
                        'hgcn', 'hgat', 'dysat', 'evolve-o', 'evolve-h', 'lstmgcn', 'wdgcn',
                        'vgrnn', 'roland', 'wingnn', 'htgn', 'graphmixer', 'm2dne', 'ghp']) # 
    parser.add_argument('--node_feat', type=str, default='dummy', choices=['onehot-id', 'dummy'])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--patiance', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--no_log', action="store_true")
    parser.add_argument('--row_mrr', action="store_true")
    
    # general model configuration
    parser.add_argument('--n_neg_train', type=int, default=1)
    parser.add_argument('--n_neg_test', type=int, default=100)
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    
    # hawkes gnn
    parser.add_argument('--bias', action="store_true") # bias in layer
    parser.add_argument('--bn', action="store_true")
    parser.add_argument('--time_encoder', action="store_true") # bias in layer
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--norm_type', type=str, default='snorm', choices=['snorm', 'dnorm', 'hnorm'])
    
    # roland
    parser.add_argument('--roland_updater', type=str, default='ma', choices=['gru', 'mlp', 'ma', 'gru-ma']) 
    parser.add_argument('--roland_is_meta', action="store_true")
    parser.add_argument('--roland_alpha', type=float, default=0.9)

    # wingnn
    parser.add_argument('--wingnn_maml_lr', type=float, default=0.008)
    parser.add_argument('--wingnn_drop_snap', type=float, default=0.1)

    # test
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--minibatch', action="store_true")
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--exp_name', type=str, default='')

    args = parser.parse_args()
    print(args)
    seed_everything(args.seed)     # for init negative_sampling

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    import torch_geometric.transforms as T
    from datasets import BitcoinOTC, BitcoinAlpha, UCIMessage, AS733, SBM, StackOverflow, RedditTitle, RedditBody
    transform = T.Compose([T.RemoveDuplicatedEdges(reduce='max')]) # for usi-message

    DS = {'bitcoinotc': BitcoinOTC, 'bitcoinalpha': BitcoinAlpha, 'redt': RedditTitle, 'redb': RedditBody,
            'uci': UCIMessage, 'as733': AS733, 'as733_full': AS733, 'sbm': SBM, 'stackoverflow': StackOverflow}
    split = {
        'bitcoinotc': [95, 95+14, 95+14+28], 'bitcoinalpha': [95, 95+13, 95+13+28], 
        'uci': [35,40,50], 'as733': [70, 70+10, 70+10+20], 
        'uci': [61, 61+9, 61+9+17],
        'redt': [122, 122+35, 122+35+17],
        'redb': [122, 122+35, 122+35+17],
        'sbm': [35,40,50],
        'stackoverflow': [70, 70+10, 70+10+20],
        'as733_full': [int(733*0.7), int(733*0.8), int(733*1)], 
        'wikipedia': [0.7, 0.85, 1], 
        'mooc': [0.7, 0.85, 1], 
        'lastfm': [0.7, 0.85, 1], 
        'reddit': [0.7, 0.85, 1]
    }
    if args.test:
        args.window=3
        for k in split.keys():
            split[k] = [10, 12, 14]
        split['uci'] = [21, 21 + 3, 21 + 3 + 6]
        #split['as733'] = [int(733*0.7), int (733*0.8), 733]

    root = {'bitcoinotc': './data/bitcoin', 'bitcoinalpha': './data/bitcoin', 'redt': './data/reddit', 'redb': './data/reddit',
        'uci': './data/uci-msg', 'as733': './data/as-733', 'sbm': './data/sbm', 'as733_full': './data/as-733',
        'stackoverflow': './data/stackoverflow'}
    
    if args.dataset in ['bitcoinotc', 'bitcoinalpha', 'uci', 'as733', 'as733_full', 'redt', 'redb', 'sbm', 'stackoverflow']:
        from dataloader import BitcoinLoaderFactory
        dataset = DS[args.dataset](root[args.dataset], transform=transform)
        factory = BitcoinLoaderFactory(dataset, 
            node_feat_type=args.node_feat, negative_sampling=args.n_neg_test)
    elif args.dataset in ['wikipedia', 'mooc', 'lastfm', 'reddit']:
        from torch_geometric.datasets import JODIEDataset
        from dataloader import JodieLoaderFactory
        import os
        args.n_neg_test = 1  # args.n_neg_test should == 1
        fo = os.path.dirname(os.path.realpath(__file__))
        root = os.path.join(fo, 'data', 'jodie')
        print(root)
        factory = JodieLoaderFactory(root, args.dataset, negative_sampling=args.n_neg_test) 
    
    if args.model in ['gcn', 'gat', 'hgcn', 'hgat']:
        do_coalesce = True if args.model in ['gcn', 'gat'] else False
        loaders = factory.get_pair_dataloader(split=split[args.dataset], device=device, window=args.window, coalesce=do_coalesce)
    elif args.model in ['dysat', 'evolve-o', 'evolve-h', 'lstmgcn', 'wdgcn', 'ghp']:
        loaders = factory.get_list_dataloader(split=split[args.dataset], device=device, window=args.window)
    elif args.model in ['roland', 'vgrnn', 'wingnn', 'htgn']:
        loaders = factory.get_roland_snaps(split=split[args.dataset], device=device)
    elif args.model in ['graphmixer', 'm2dne']:
        loaders = factory.get_lru_dataloader(split=split[args.dataset], device=device)

    # sorry for sooo many train logic
    # it tooo hard to merge them into one
    if args.model == 'vgrnn':
        from train_vgrnn import VgrnnLinkPrediction
        lp = VgrnnLinkPrediction(args, build_model)
    elif args.model == 'roland':
        from train_roland import RolandLinkPrediction
        lp = RolandLinkPrediction(args, build_model)
    elif args.model == 'wingnn':
        from train_wingnn import WinGNNLinkPrediction
        lp = WinGNNLinkPrediction(args, build_model)
    elif args.model == 'htgn':
        from train_htgn import HTGNLinkPrediction
        lp = HTGNLinkPrediction(args, build_model)
    elif args.model == 'm2dne':
        from train_m2dne import M2DNELinkPrediction
        lp = M2DNELinkPrediction(args, build_model)
    else:
        if args.minibatch:
            #from train_scalable import ScalableLinkPrediction
            from train_minibatch import MiniBatchLinkPrediction
            lp = MiniBatchLinkPrediction(args, build_model)
            if args.dataset in ['as733', 'sbm']:
                lp.strategy = 'all'
                lp.strategy = 'random'
            else:
                lp.strategy = 'random'
        # elif args.live_update:
        #     from train_live import LiveLinkPrediction
        #     lp = LiveLinkPrediction(args, build_model)
        else:
            lp = LinkPrediction(args, build_model)
    
    if args.dataset in ['wikipedia', 'mooc', 'lastfm', 'reddit']:
        min_dst_idx, max_dst_idx = int(factory.data.dst.min()), int(factory.data.dst.max())
        lp.set_jodie(min_dst_idx, max_dst_idx)
    lp.main(args,  factory,  loaders)
