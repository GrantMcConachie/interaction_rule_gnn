"""
General utility functions for each dataset.
"""

import os
import numpy as np
import pickle as pkl
from tqdm import tqdm

import torch
from torch_geometric.data import Data


def make_fc_edge_idx(num_nodes):
    """
    makes a edge index for a fully connected graph given a number of nodes.
    Ignores self edges
    """
    arr = np.arange(num_nodes)
    edge_idx = np.array(np.meshgrid(arr, arr)).T.reshape(-1, 2).T
    self_edge = np.where(edge_idx[0] == edge_idx[1])

    return torch.tensor(
        np.delete(edge_idx, self_edge, axis=1),
        dtype=torch.long
    )


def generate_graphs(state_vars, wall_thresh=0.3, gt_edges=None):
    """
    Takes position data and generates a pytorch geometric graph

    Input:
        state_vars (dict) - position, velocity, and acceleration data for
          a given dataset.
        wall_thresh (float) - Threshold when to add a distance to wall variable
          in the node feature vector

    Return:
        graphs (list) - list of pytorch gemoetric graphs
    """
    eps = 1e-16
    num_nodes = state_vars['x'].shape[0]

    # edge index for fully connected graph
    edge_index = make_fc_edge_idx(num_nodes)

    # loop through time
    graphs = []
    for i in tqdm(range(state_vars['x'].shape[-1] - 1), desc="generating graphs"):
        # position values and next position
        pos = state_vars['x'][:, :, i]
        pos_next = state_vars['x'][:, :, i+1]
        pos_norm = np.linalg.norm(pos, axis=1)
        vel = state_vars['x_dot'][:, :, i]
        acc = state_vars['x_dot_dot'][:, :, i]
        
        # node features (velocity unit vector and distance to wall with threshold)
        dist_from_wall = np.ones_like(pos_norm) - pos_norm
        dist_from_wall_thresh = np.where(dist_from_wall < wall_thresh, pos_norm, 0.)
        heading = vel / (np.linalg.norm(vel, axis=1, keepdims=True) + eps)
        node_feats = torch.tensor(
            np.append(heading, dist_from_wall_thresh[:,None], axis=1),
            dtype=torch.float
        )

        # making edge features
        rel_pos = pos[edge_index[0]] - pos[edge_index[1]]
        rel_pos_norm = np.linalg.norm(rel_pos, axis=1)
        edge_attr = torch.tensor(
            np.append(rel_pos, rel_pos_norm[:,None], axis=1),
            dtype=torch.float
        )

        # making graph
        graph = Data(
            x=node_feats,
            edge_attr=edge_attr,
            edge_index=edge_index,
            gt_edge_index=gt_edges[:, :, i] if gt_edges is not None else None,
            pos=torch.tensor(pos, dtype=torch.float),
            pos_next=torch.tensor(pos_next, dtype=torch.float),
            vel=torch.tensor(vel, dtype=torch.float),
            acc=torch.tensor(acc, dtype=torch.float),
            t=i
        )
        graphs.append(graph)

    return graphs


def save_graphs(fp, graphs):
    """
    Saving all generated graphs

    fp (str) - parent directory for file
    graphs (list) - All generated graphs from given file
    """
    pkl.dump(graphs, open(fp, 'wb'))
