"""
Script with all models to train
"""

import pickle as pkl

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


### Learned simulator MPNN aproach ###
class LearnedSimGNN(MessagePassing):
    """
    MPNN with distinct node and edge updates
    """

    def __init__(
            self,
            node_emb_dim,
            edge_emb_dim,
            dropout_prob,
            mlp_hidden_dim=128,
            mlp_out_dim=128
    ):
        super().__init__(aggr='add')
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_emb_dim + edge_emb_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_hidden_dim, mlp_out_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_emb_dim + edge_emb_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_hidden_dim, mlp_out_dim)
        )
    
    def forward(self, x, edge_attr, edge_index):
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


class LearnedSimModel(nn.Module):
    """
    Model that closley follows many of the learned simulator physics models
    (e.g. https://arxiv.org/abs/2010.03409)
    """

    def __init__(
            self,
            graph_edge_dim,
            graph_node_dim,
            edge_encoder_hidden_dim=128,
            node_encoder_hidden_dim=128,
            out_dim=2,
            gnn_layers=5,
            noise_std=0.01,
            dropout_prob=0.3
    ):
        super(LearnedSimModel, self).__init__()

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
        self.gnn_layers = nn.ModuleList([
            LearnedSimGNN(node_emb_dim=node_encoder_hidden_dim,
            edge_emb_dim=edge_encoder_hidden_dim, dropout_prob=dropout_prob)
            for _ in range(gnn_layers)
        ])

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

        # model variables
        self.noise_std = noise_std
    
    def forward(self, graph):
        # unpack graph
        x = graph.x
        edge_attr = graph.edge_attr

        # add noise to every input for now
        node_noise = torch.randn_like(x) * self.noise_std
        edge_noise = torch.randn_like(edge_attr) * self.noise_std
        x = x + node_noise
        edge_attr = edge_attr + edge_noise

        # embed nodes and edges
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # pass through gnn
        for gnn, x_norm, e_norm in zip(self.gnn_layers, self.x_norm, self.e_norm):
            x_gnn, edge_attr_gnn = gnn(x, edge_attr, graph.edge_index)
            x += x_gnn
            edge_attr += edge_attr_gnn
            x = x_norm(x)
            edge_attr = e_norm(edge_attr)

        # decode for predicted next position, subtracting added noise
        x = self.decoder(x) - node_noise[:, :2]

        return x


### NRI model TODO ###


### GAT TODO ###


if __name__ == '__main__':
    g = pkl.load(open('data/fish/processed/8fish/240816f1.pkl', 'rb'))[0] # graph
    model = LearnedSimModel(
        graph_edge_dim=g.edge_attr.shape[1],
        graph_node_dim=g.x.shape[1]
    )
    out = model(g)
    print('done')
