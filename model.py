import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, RGCNConv, GCNConv, to_hetero

class GNNEncoder(nn.Module):
    def __init__(self, hidden_channels, dropout_rate=0.5):
        super().__init__()

        #sage convolution : https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv 
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        
        
        # self.conv1 = GCNConv(-1, hidden_channels)
        # self.conv2 = RGCNConv((-1, -1), out_channels, num_relations= 10)
        #attention based graph, closer to the first paper : https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html
        heads = 16
        # self.conv1 = GATConv((-1, -1), hidden_channels, heads=heads, concat=True, add_self_loops=False) # Multi-head
        # self.conv2 = GATConv((-1, -1), out_channels, heads=1, concat=False, add_self_loops=False)  # Single-head for simplicity
        
        

        #check dynconv : https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.DynamicEdgeConv
        #self.conv1 = ...


        # print(dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) 
        if not self.conv1.__class__.__name__ == 'GATConv':
            heads = 1
        self.bn_in = nn.BatchNorm1d(hidden_channels * heads)  # BatchNorm after multi-head. Note the dimension.
        self.bn_out = nn.BatchNorm1d(hidden_channels)  # BatchNorm after single-head
        




    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        #x = self.bn_in(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        # x = self.bn_out(x)
        x = self.dropout(x)
        return x


class EdgeDecoder(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        #print(z_dict['user'].shape, z_dict['movie'].shape)
        z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)
    
class BilinearDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, num_ratings):
        super(BilinearDecoder, self).__init__()
        self.Q = torch.nn.ParameterDict({
            str(r): torch.nn.Parameter(torch.randn(hidden_channels, hidden_channels))
            for r in range(num_ratings)
        })
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        user_embedding = z_dict['user'][row]
        movie_embedding = z_dict['movie'][col]
        logits = []
        for r, Q_r in self.Q.items():
            logit = (user_embedding @ Q_r @ movie_embedding.t()).diag() # u^T Q_r v for each rating r
            logits.append(logit)
        logits = torch.stack(logits, dim=-1) # Stack along the last dimension
        return self.log_softmax(logits) # Apply softmax to get probabilities

# Example usage
# decoder = BilinearDecoder(user_dim=32, movie_dim=32, rating_dim=32, num_ratings=5)

class BilinearEdgeDecoder(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.bilinear = nn.Bilinear(hidden_channels, hidden_channels, 1)
        self.lin = nn.Linear(1, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        user_embedding = z_dict['user'][row]
        movie_embedding = z_dict['movie'][col]
        
        z = self.bilinear(user_embedding, movie_embedding)
        z = self.lin(z).relu()
        return z.view(-1)

class Model(nn.Module):
    def __init__(self, cfg, data):
        super().__init__()
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels
        self.hidden_channels = cfg.out_channels
        self.input_node_embedding = cfg.input_node_embedding

        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, self.hidden_channels)
        self.movie_emb = torch.nn.Embedding(data["movie"].num_nodes, self.hidden_channels)

        self.encoder = GNNEncoder(self.hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')

        # self.decoder = EdgeDecoder(self.out_channels)
        # self.decoder = BilinearDecoder(self.hidden_channels, 10)
        self.decoder = BilinearEdgeDecoder(self.out_channels)

    def forward(self, dict, edge_index_dict, edge_label_index):

        if self.input_node_embedding == "random":

            x_dict = {
            "user": self.user_emb(dict["user"]),
            "movie": self.movie_emb(dict["movie"]),
            }
        elif self.input_node_embedding == "one_hot":
            x_dict = dict
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

