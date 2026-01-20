"""
Script that preprocesses the raw data

TODO: encode heading angle
"""

import os
import h5py
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm

import torch

from torch_geometric.data import Data


def generate_x_dot(loc, frame_rate=120):
    """
    generates x dot data from location data
    """
    # init
    x = []
    x_dot = []
    x_dot_dot = []

    for i, fish in enumerate(loc):
        # averaging over keypoints
        pos = np.mean(fish, axis=1)

        # linearaly interpolating over nans
        pos = np.array(
            pd.DataFrame(pos).interpolate(axis=1, limit_direction='both')
        )

        # px to mm, center, and normalize
        pos *= (300.0 / 750.0)
        pos -= 150.0
        pos /= 150.0

        # calculating velocity and acceleration
        vel = np.gradient(pos, axis=1) * frame_rate
        acc = np.gradient(vel, axis=1) * frame_rate

        x.append(pos)
        x_dot.append(vel)
        x_dot_dot.append(acc)

    return {
        'x': np.array(x),
        'x_dot': np.array(x_dot),
        'x_dot_dot': np.array(x_dot_dot)
    }


def make_edge_idx(num_nodes):
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


def generate_graphs(state_vars, wall_thresh=0.3):
    """
    Takes position data and generates a pytorch geometric graph

    Input:
        state_vars (dict) - fish position, velocity, and acceleration data for
          a given video. (num_units, x-y position, time)
        wall_thresh (float) - Threshold when to add a distance to wall variable
          in the node feature vector

    Return:
        graphs (list) - list of pytorch gemoetric graphs
    """
    num_nodes = state_vars['x'].shape[0]

    # edge index for fully connected graph
    edge_index = make_edge_idx(num_nodes)

    # loop through time
    graphs = []
    for i in tqdm(range(state_vars['x'].shape[-1] - 64)):
        # position values and next position
        pos = state_vars['x'][:, :, i]
        pos_norm = np.linalg.norm(pos, axis=1)
        vel = state_vars['x_dot'][:, :, i]
        acc = state_vars['x_dot_dot'][:, :, i]

        # next position targets
        pos_1 = state_vars['x'][:, :, i+1]
        pos_2 = state_vars['x'][:, :, i+2]
        pos_4 = state_vars['x'][:, :, i+4]
        pos_8 = state_vars['x'][:, :, i+8]
        pos_16 = state_vars['x'][:, :, i+16]
        pos_32 = state_vars['x'][:, :, i+32]
        pos_64 = state_vars['x'][:, :, i+64]
        
        # node features (velocity unit vector and distance to wall with threshold)
        eps = 1e-8
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
            pos=torch.tensor(pos, dtype=torch.float),
            pos_1=torch.tensor(pos_1, dtype=torch.float),
            pos_2=torch.tensor(pos_2, dtype=torch.float),
            pos_4=torch.tensor(pos_4, dtype=torch.float),
            pos_8=torch.tensor(pos_8, dtype=torch.float),
            pos_16=torch.tensor(pos_16, dtype=torch.float),
            pos_32=torch.tensor(pos_32, dtype=torch.float),
            pos_64=torch.tensor(pos_64, dtype=torch.float),
            vel=torch.tensor(vel, dtype=torch.float),
            acc=torch.tensor(acc, dtype=torch.float)
        )
        graphs.append(graph)

    return graphs


def save_graphs(f, fp, graphs):
    """
    Saving all generated graphs

    f (str) - data file
    fp (str) - parent directory for file
    graphs (list) - All generated graphs from given file
    """
    save_path = os.path.join(fp, f)
    pkl.dump(graphs, open(save_path, 'wb'))


def main(fp, save_fp):
    """
    Gets location data out of the raw files and saves them as pkl files
    """
    # list files
    fish_files = os.listdir(fp)

    # loop though files
    graph_lens = []
    for f in fish_files:
        fish_h5 = h5py.File(os.path.join(fp, f))

        # take out location data
        fish_hdf = fish_h5['tracks']

        # generate state vectors
        state_vectors = generate_x_dot(fish_hdf)

        # generate graphs
        graphs = generate_graphs(state_vectors)

        # saving
        f = f.replace('.h5', '.pkl')
        save_graphs(f, save_fp, graphs)


if __name__ == '__main__':
    # 8 fish
    main(
        fp='data/fish/raw data/8fish',
        save_fp='data/fish/processed/8fish'
    )

    # 10 fish
    main(
        fp='data/fish/raw data/10fish/DATA',
        save_fp='data/fish/processed/10fish'
    )
