import torch
from torch.nn import GRU
from torch_geometric.nn import TopKPooling

from models.base import BaseLPModel
from models.predictor import LinkPredictor

from .evolvegcno import glorot, GCNConv_Fixed_W, EvolveGCNO


class EvolveGCNH(torch.nn.Module):
    r"""An implementation of the Evolving Graph Convolutional Hidden Layer.
    For details see this paper: `"EvolveGCN: Evolving Graph Convolutional
    Networks for Dynamic Graph." <https://arxiv.org/abs/1902.10191>`_

    Args:
        num_of_nodes (int): Number of vertices.
        in_channels (int): Number of filters.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
    """

    def __init__(
        self,
        num_of_nodes: int,
        in_channels: int,
        improved: bool = False,
        cached: bool = False,
        normalize: bool = True,
        add_self_loops: bool = True,
    ):
        super(EvolveGCNH, self).__init__()

        self.num_of_nodes = num_of_nodes
        self.in_channels = in_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.weight = None
        self.initial_weight = torch.nn.Parameter(torch.Tensor(1, in_channels, in_channels))
        self._create_layers()
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.initial_weight)

    def reinitialize_weight(self):
        self.weight = None

    def _create_layers(self):

        self.ratio = self.in_channels / self.num_of_nodes

        self.pooling_layer = TopKPooling(self.in_channels, self.ratio)

        self.recurrent_layer = GRU(
            input_size=self.in_channels, hidden_size=self.in_channels, num_layers=1
        )

        self.conv_layer = GCNConv_Fixed_W(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            improved=self.improved,
            cached=self.cached,
            normalize=self.normalize,
            add_self_loops=self.add_self_loops
        )

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node embedding.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Float Tensor, optional)* - Edge weight vector.

        Return types:
            * **X** *(PyTorch Float Tensor)* - Output matrix for all nodes.
        """
        X_tilde = self.pooling_layer(X, edge_index)
        X_tilde = X_tilde[0][None, :, :]
        if self.weight is None:
            _, self.weight = self.recurrent_layer(X_tilde, self.initial_weight)
        else:
            _, self.weight = self.recurrent_layer(X_tilde, self.weight)
        X = self.conv_layer(self.weight.squeeze(dim=0), X, edge_index, edge_weight)
        return X



class EvolveGCNLP(BaseLPModel):
    def __init__(self, n_node, n_feat, n_hidden, dropout=0.1, sampling_ratio=1, initial_skip=False, name='evolve-o'):
        super().__init__()
        self.skip = initial_skip
        self.input_fc = torch.nn.Linear(n_feat, n_hidden)
        if name == 'evolve-o':
            self.model = torch.nn.ModuleList([
                EvolveGCNO(n_hidden, n_hidden) for _ in range(2)
            ])
        else:
            self.model = torch.nn.ModuleList([
                EvolveGCNH(n_node, n_hidden) for _ in range(2)
            ])
        self.act = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(dropout)
        n_out = n_hidden + n_feat if initial_skip else n_hidden
        self.predictor = LinkPredictor(n_out, n_hidden, 1, 2, dropout)
        self.sampling_ratio = sampling_ratio


    def forward(self, batch_list):
        h_stack = []
        for batch in batch_list:
            h_stack.append(self.input_fc(batch.x.to_dense()))
        seq_len = len(h_stack)

        for layer, net in enumerate(self.model):
            net.reinitialize_weight()
            for t, data in enumerate(batch_list):
                h = net(h_stack[layer*seq_len + t], data.edge_index)
                if layer < len(self.model)-1:
                    h = self.act(h)
                    h = self.drop(h)
                h_stack.append(h)
        
        h = h_stack[-1]
        if self.skip:
            h = torch.cat([h_stack[seq_len-1], h], dim=1)
        return h
