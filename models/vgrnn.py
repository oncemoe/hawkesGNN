import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch_geometric.nn import GATConv
from torch_scatter import scatter
from torch.autograd import Variable
from torch_geometric.nn.conv import GCNConv, SAGEConv, GINConv

from models.base import BaseLPModel
from models.predictor import LinkPredictor


# from https://github.com/VGraphRNN/VGRNN
class graph_gru_gcn(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, bias=True, conv=GCNConv):
        super(graph_gru_gcn, self).__init__()

        self.hidden_size = hidden_size
        self.n_layer = n_layer
        
        # gru weights
        self.weight_xz = nn.ModuleList()
        self.weight_hz = nn.ModuleList()
        self.weight_xr = nn.ModuleList()
        self.weight_hr = nn.ModuleList()
        self.weight_xh = nn.ModuleList()
        self.weight_hh = nn.ModuleList()
        
        for i in range(self.n_layer):
            if i==0:
                self.weight_xz.append(conv(input_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_hz.append(conv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_xr.append(conv(input_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_hr.append(conv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_xh.append(conv(input_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_hh.append(conv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
            else:
                self.weight_xz.append(conv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_hz.append(conv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_xr.append(conv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_hr.append(conv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_xh.append(conv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_hh.append(conv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
    
    def forward(self, inp, edgidx, h):
        h_out = torch.zeros(h.size())
        for i in range(self.n_layer):
            if i==0:
                z_g = torch.sigmoid(self.weight_xz[i](inp, edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](inp, edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](inp, edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
        #         out = self.decoder(h_t.view(1,-1))
            else:
                z_g = torch.sigmoid(self.weight_xz[i](h_out[i-1], edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](h_out[i-1], edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](h_out[i-1], edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
        #         out = self.decoder(h_t.view(1,-1))
        
        out = h_out
        return out, h_out



class InnerProductDecoder(nn.Module):
    def __init__(self, act=torch.sigmoid, dropout=0.):
        super(InnerProductDecoder, self).__init__()
        
        self.act = act
        self.dropout = dropout
    
    def forward(self, inp):
        inp = F.dropout(inp, self.dropout, training=self.training)
        x = torch.transpose(inp, dim0=0, dim1=1)
        x = torch.mm(inp, x)
        return self.act(x)



class VGRNN(BaseLPModel):
    def __init__(self, n_node, n_feat, n_edge, h_dim, z_dim, dropout=0, n_layers=1, eps=1e-8, conv='GCN', bias=False):
        super(VGRNN, self).__init__()
        self.n_node = n_node
        self.eps = eps
        self.h_dim = h_dim
        self.n_layers = n_layers
        
        self.phi_x = nn.Sequential(nn.Linear(n_feat, h_dim), nn.ReLU())
        self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())

        self.prior = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
        self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim))
        self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())

        if conv == 'GCN':         
            self.enc = GCNConv(h_dim + h_dim, h_dim)            
            self.enc_mean = GCNConv(h_dim, z_dim)
            self.enc_std = GCNConv(h_dim, z_dim)
        
            self.rnn = graph_gru_gcn(h_dim + h_dim, h_dim, n_layers, bias)

            
        elif conv == 'GIN':
            self.enc = GINConv(nn.Sequential(nn.Linear(h_dim + h_dim, h_dim), nn.ReLU()))            
            self.enc_mean = GINConv(nn.Sequential(nn.Linear(h_dim, z_dim)))
            self.enc_std = GINConv(nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus()))
            
            self.rnn = graph_gru_gcn(h_dim + h_dim, h_dim, n_layers, bias)  
        
        self.predictor = LinkPredictor(h_dim, h_dim, 1, 2, dropout)
    
    def forward(self, data, hidden_in=None):        
        if data.x.is_sparse:
            x = data.x.to_dense()
        else:
            x = data.x
        edge_index = data.edge_index
        adj = data.adj
        
        if hidden_in is None:
            h = Variable(torch.zeros(self.n_layers, x.size(0), self.h_dim)).to(x.device)
        else:
            h = Variable(hidden_in).to(x.device)
        
        phi_x_t = self.phi_x(x)
        
        #encoder
        enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1), edge_index)
        enc_mean_t = self.enc_mean(enc_t, edge_index)
        enc_std_t = F.softplus(self.enc_std(enc_t, edge_index))
        
        #prior
        prior_t = self.prior(h[-1])
        prior_mean_t = self.prior_mean(prior_t)
        prior_std_t = self.prior_std(prior_t)
        
        #sampling and reparameterization
        z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
        phi_z_t = self.phi_z(z_t)
        
        #decoder
        dec_t = self.dec(z_t)
        
        #recurrence
        _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1), edge_index, h)
        
        #computing losses
#             kld_loss += self._kld_gauss_zu(enc_mean_t, enc_std_t)
        kld_loss = self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
        nll_loss = self._nll_bernoulli(dec_t, adj.to_dense())
        
        return kld_loss, nll_loss, enc_mean_t, prior_mean_t, h.detach()
    
    def dec(self, z):
        outputs = InnerProductDecoder(act=lambda x:x)(z)
        return outputs
    
    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)
     
    def _init_weights(self, stdv):
        pass
    
    def _reparameterized_sample(self, mean, std):
        eps1 = torch.FloatTensor(std.size()).normal_()
        eps1 = Variable(eps1).to(std.device)
        return eps1.mul(std).add_(mean)
    
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        num_nodes = mean_1.size()[0]
        kld_element =  (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) +
                        (torch.pow(std_1 + self.eps ,2) + torch.pow(mean_1 - mean_2, 2)) / 
                        torch.pow(std_2 + self.eps ,2) - 1)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)
    
    def _kld_gauss_zu(self, mean_in, std_in):
        num_nodes = mean_in.size()[0]
        std_log = torch.log(std_in + self.eps)
        kld_element =  torch.mean(torch.sum(1 + 2 * std_log - mean_in.pow(2) -
                                            torch.pow(torch.exp(std_log), 2), 1))
        return (-0.5 / num_nodes) * kld_element
    
    def _nll_bernoulli(self, logits, target_adj_dense):
        temp_size = target_adj_dense.size()[0]
        temp_sum = target_adj_dense.sum()
        posw = float(temp_size * temp_size - temp_sum) / temp_sum
        norm = temp_size * temp_size / float((temp_size * temp_size - temp_sum) * 2)
        nll_loss_mat = F.binary_cross_entropy_with_logits(input=logits
                                                          , target=target_adj_dense
                                                          , pos_weight=posw
                                                          , reduction='none')
        nll_loss = -1 * norm * torch.mean(nll_loss_mat, dim=[0,1])
        return - nll_loss