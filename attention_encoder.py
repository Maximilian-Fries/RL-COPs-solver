import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv

torch.set_printoptions(profile="full", linewidth=200)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            concat=True
    ):  
        super(MultiHeadAttentionLayer, self).__init__()

        self.concat = concat
        self.single_head_dim = int(embed_dim / n_heads) if self.concat else embed_dim

        self.MHA = GATConv(in_channels=embed_dim, out_channels=self.single_head_dim, heads=n_heads,
                           concat=concat, add_self_loops=False, bias=False)

        self.batch_layer = nn.BatchNorm1d(embed_dim)

    def forward(self, features, edge_index):
        x = features + self.MHA(features, edge_index)
        x = self.batch_layer(x)

        return x


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads=4,
            embed_dim=64,
            n_layers=3,
            node_dim=1,
    ):
        super(GraphAttentionEncoder, self).__init__()

        # map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim)

        self.layers = nn.ModuleList([
            MultiHeadAttentionLayer(n_heads=n_heads, embed_dim=embed_dim)
            for _ in range(n_layers - 1)
        ])
        self.final_layer = MultiHeadAttentionLayer(n_heads=n_heads, embed_dim=embed_dim, concat=False)

    def forward(self, data):
        h, edge_index = data.x, data.edge_index

        h = self.init_embed(h)

        for attention_layer in self.layers:
            h = attention_layer(h, edge_index)

        h = self.final_layer(h, edge_index)

        return h


class GCNEncoder(nn.Module):
    def __init__(self, node_dim=1, n_layers=3, embed_dim=64):
        super(GCNEncoder, self).__init__()
        self.first_layer = GCNConv(node_dim, embed_dim)
        self.layers = nn.ModuleList([GCNConv(embed_dim, embed_dim) for _ in range(n_layers-1)])

    def forward(self, data):
        h, edge_index = data.x, data.edge_index
        h = self.first_layer(h, edge_index)
        for gcn_layer in self.layers:
            h = gcn_layer(h, edge_index)
        return h