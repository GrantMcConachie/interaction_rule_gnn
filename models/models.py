"""
Script with all models to train
"""

import pickle as pkl

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import GATConv, GPSConv, GINEConv

### Learned simulator MPNN aproach ###
class MPNN(MessagePassing):
    """
    MPNN with distinct node and edge updates
    """

    def __init__(
            self,
            in_channels,
            edge_dim,
            dropout,
            mlp_hidden_dim=128,
            out_channels=128
    ):
        super().__init__(aggr='add')
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, out_channels)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + edge_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, out_channels)
        )
    
    def forward(self, x, edge_attr, edge_index, **kwargs):
        """
        Forward pass of the model

        :param x: node vectors
        :param edge_attr: edge vectors
        :param edge_index: edge index, dictating where edges are
        """
        # Update the edges with the surrounding node information
        edge_attr = self.edge_updater(edge_index=edge_index, edge_attr=edge_attr, x=x)

        # update the nodes with the surrounding edge information
        x = self.propagate(edge_index=edge_index, edge_attr=edge_attr, x=x)

        return x, edge_attr

    def message(self, x_i, edge_attr):
        """
        Update for the nodes

        :param x_i: Node vector
        :param edge_attr: sum of the incoming edgees
        """
        return x_i + self.node_mlp(torch.cat([x_i, edge_attr], axis=1))

    def edge_update(self, x_i, x_j, edge_attr):
        """
        update for edges

        :param x_i: node vector on one side of edge
        :param x_j: node vector on the other side of the edge
        :param edge_attr: edge vector
        """
        return edge_attr + self.edge_mlp(torch.cat([x_i, x_j, edge_attr], axis=1))


class GAT(MessagePassing):
    """
    GAT with distinct edge updates
    """

    def __init__(
            self,
            in_channels,
            edge_dim,
            dropout,
            heads,
            mlp_hidden_dim=128,
            out_channels=128
    ):
        super().__init__(aggr='add')
        self.node_gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            edge_dim=edge_dim,
            heads=heads,
            concat=False,
            add_self_loops=False
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, out_channels)
        )

    def forward(self, x, edge_attr, edge_index, **kwargs):
        """
        Forward pass of the model
        
        :param x: node vectors
        :param edge_attr: edge vectors
        :param edge_index: edge index, dictating where edges are
        """
        # Update the edges with the surrounding node information
        edge_attr = self.edge_updater(edge_index=edge_index, edge_attr=edge_attr, x=x)

        # update the nodes with the surrounding edge information using a GAT
        x, (pred_adj_mat, attn_weights) = self.node_gat(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            return_attention_weights=True
        )

        # store attn weights
        self.pred_adj_mat = pred_adj_mat
        self.attn_weights = attn_weights

        return x, edge_attr
    
    def edge_update(self, x_i, x_j, edge_attr):
        """
        update for edges
        
        :param x_i: node vector on one side of edge
        :param x_j: node vector on the other side of the edge
        :param edge_attr: edge vector
        """
        return edge_attr + self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=1))


class LearnedSimModel(nn.Module):
    """
    Model that closley follows many of the learned simulator physics models
    (e.g. https://arxiv.org/abs/2010.03409)
    """

    def __init__(
            self,
            graph_edge_dim,
            graph_node_dim,
            config
    ):
        super(LearnedSimModel, self).__init__()

        # unpack config
        edge_encoder_hidden_dim = config['model']['edge_encoder_hidden_dim']
        node_encoder_hidden_dim = config['model']['node_encoder_hidden_dim']
        out_dim = config['model']['out_dim']
        gnn_layers = config['model']['gnn_layers']
        self.noise_std = config['model']['noise_std']
        dropout_prob = config['model']['dropout_prob']
        attn_heads = config['model']['num_heads']

        # edge and node encoders
        self.edge_encoder = nn.Sequential(
            nn.Linear(graph_edge_dim, edge_encoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(edge_encoder_hidden_dim, edge_encoder_hidden_dim)
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(graph_node_dim, node_encoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(node_encoder_hidden_dim, node_encoder_hidden_dim)
        )

        # GNN layers
        self.model_type = config['model']['gnn_type']
        model_list = ["MPNN", "GAT", "GPS"] 
        if self.model_type == 'MPNN':
            self.gnn_layers = nn.ModuleList(
                [
                    MPNN(
                        in_channels=node_encoder_hidden_dim,
                        out_channels=node_encoder_hidden_dim,
                        edge_dim=edge_encoder_hidden_dim, 
                        dropout=dropout_prob,
                    )
                    for _ in range(gnn_layers)
                ]
            )
        
        elif self.model_type == 'GAT':
            self.gnn_layers = nn.ModuleList(
                [
                    GAT(
                        in_channels=node_encoder_hidden_dim,
                        out_channels=node_encoder_hidden_dim,
                        edge_dim=edge_encoder_hidden_dim,
                        dropout=dropout_prob,
                        heads=attn_heads
                    )
                    for _ in range(gnn_layers)
                ]
            )
        
        elif self.model_type == 'GPS':
            self.gnn_layers = nn.ModuleList(
                [
                    GPSConv(
                        channels=node_encoder_hidden_dim,
                        conv=GINEConv(nn.Sequential(
                            nn.Linear(node_encoder_hidden_dim, node_encoder_hidden_dim),
                            nn.ReLU(),
                            nn.Linear(node_encoder_hidden_dim, node_encoder_hidden_dim)
                        )),
                        heads=attn_heads,
                        dropout=dropout_prob
                    )
                    for _ in range(gnn_layers)
                ]
            )

        else:
            assert self.model_type not in model_list, f"{self.model_type} not in {model_list}"

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(node_encoder_hidden_dim, node_encoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(node_encoder_hidden_dim, out_dim)
        )

        # layer norms
        self.x_norm = nn.ModuleList([
            nn.LayerNorm(node_encoder_hidden_dim) for _ in range(gnn_layers)
        ])
        self.e_norm = nn.ModuleList([
            nn.LayerNorm(edge_encoder_hidden_dim) for _ in range(gnn_layers)
        ])

    def forward(self, graph):
        # unpack graph
        x = graph.x
        edge_attr = graph.edge_attr

        # add noise to every input for now
        if self.training:
            node_noise = torch.randn_like(x) * self.noise_std
            edge_noise = torch.randn_like(edge_attr) * self.noise_std
            x = x + node_noise
            edge_attr = edge_attr + edge_noise

        # embed nodes and edges
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # pass through gnn
        if self.model_type == "GPS":
            for gnn, x_norm, _ in zip(self.gnn_layers, self.x_norm, self.e_norm):
                x_gnn = gnn(x, edge_attr=edge_attr, edge_index=graph.edge_index, batch=graph.batch)
                x = x + x_gnn
                x = x_norm(x)

        else:
            for gnn, x_norm, e_norm in zip(self.gnn_layers, self.x_norm, self.e_norm):
                x_gnn, edge_attr_gnn = gnn(x, edge_attr=edge_attr, edge_index=graph.edge_index, return_attention_weights=True)
                x = x + x_gnn
                edge_attr = edge_attr + edge_attr_gnn
                x = x_norm(x)
                edge_attr = e_norm(edge_attr)

        # decode for predicted next position, subtracting added noise
        if self.training:
            x = self.decoder(x) - node_noise[:, :2]
        else:
            x = self.decoder(x)

        return x


### NRI model TODO ###


if __name__ == '__main__':
    g = pkl.load(open('data/fish/processed/8fish/240816f1.pkl', 'rb'))[0] # graph
    model = LearnedSimModel(
        graph_edge_dim=g.edge_attr.shape[1],
        graph_node_dim=g.x.shape[1]
    )
    out = model(g)
    print('done')
