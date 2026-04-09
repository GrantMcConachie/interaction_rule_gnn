"""
Utility functions for training script
"""

import os
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj

from dataset_utils import utils


def save_model(model, args):
    """
    Saves model
    """
    dataset = os.path.basename(args.dataset).replace('.pkl', '')
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_path, f'{dataset}.pt'))


def split_with_config(data, config):
    """
    Splits the dataset with dataset and a config file
    """
    dat_train, dat_test = train_test_split(
        data,
        test_size=config['training']['test_split'],
        shuffle=False
    )
    dat_train, dat_val = train_test_split(
        dat_train,
        test_size=config['training']['val_split'] / (1 - config['training']['test_split']),
        shuffle=False
    )

    return dat_train, dat_val, dat_test


def downsample_data(data, step):
    """
    function that downsamples data

    :param data: dataset that you wish to downsample
    :param downsample_step: how many timesteps to downsample
    """
    # calculate new data length
    new_len = len(data) - step

    # calculate new dt
    dt = data[0].dt * step

    # loop through data and adjust next position
    for i in range(new_len):
        data[i].pos_next = data[i + step].pos
        data[i].dt = dt

    return data[:new_len]


def create_windowed_samples(graphs, past_window, future_window):
    """
    Creates windowed training samples from a list of single-timestep graphs.

    Each sample concatenates node and edge features from the last `past_window`
    timesteps as input, and stores the next `future_window` positions and
    accelerations as prediction targets.

    :param graphs: list of single-timestep Data objects
    :param past_window: number of past timesteps to use as input context
    :param future_window: number of future timesteps to predict
    :return: list of windowed Data objects
    """
    windowed = []
    total = len(graphs)
    for t in range(past_window - 1, total - future_window):
        past = graphs[t - past_window + 1: t + 1]  # list of past_window graphs
        curr = graphs[t]

        # Concatenate past node/edge features along the feature dim (oldest→newest)
        x = torch.cat([g.x for g in past], dim=1)
        edge_attr = torch.cat([g.edge_attr for g in past], dim=1)

        # Future targets flattened to [N, future_window * 2]
        pos_future = torch.cat(
            [graphs[t + k + 1].pos for k in range(future_window)], dim=1
        )
        acc_future = torch.cat(
            [graphs[t + k].acc for k in range(future_window)], dim=1
        )

        new_g = Data(
            x=x,
            edge_attr=edge_attr,
            edge_index=curr.edge_index,
            pos=curr.pos,
            vel=curr.vel,
            acc=curr.acc,
            pos_future=pos_future,
            acc_future=acc_future,
            pos_next=graphs[t + 1].pos,  # kept for rollout compatibility
            dt=curr.dt,
            t=curr.t,
        )
        if hasattr(curr, 'gt_edge_index') and curr.gt_edge_index is not None:
            new_g.gt_edge_index = curr.gt_edge_index
            new_g.gt_edge_attr = curr.gt_edge_attr
        windowed.append(new_g)
    return windowed


def build_windowed_input(buf, curr_g, config):
    """
    Builds a single windowed graph from a rolling buffer of recent graphs.
    Used during rollout evaluation to construct model input on-the-fly.

    :param buf: list of the most recent `past_window` Data objects
    :param curr_g: the ground-truth graph at the current timestep (for gt edges)
    :param config: training config dict
    :return: a single Data object with concatenated past features
    """
    x = torch.cat([g.x for g in buf], dim=1)
    edge_attr = torch.cat([g.edge_attr for g in buf], dim=1)
    latest = buf[-1]
    g_in = Data(
        x=x,
        edge_attr=edge_attr,
        edge_index=latest.edge_index,
        pos=latest.pos,
        vel=latest.vel,
        dt=latest.dt,
    )
    if config['training']['gt_edges'] and hasattr(curr_g, 'gt_edge_index'):
        g_in.gt_edge_index = curr_g.gt_edge_index
        g_in.gt_edge_attr = curr_g.gt_edge_attr
    return g_in


def split_and_load_data(config, args):
    """
    Loads in the dataset of interest and splits the data    
    """
    # load data
    with open(args.dataset, 'rb') as f:
        data = pkl.load(f)

    # downsample
    if config['training']['downsample_timestep'] != 1:
        data = downsample_data(data, config['training']['downsample_timestep'])

    # apply context windowing
    past_window = config['training'].get('past_window', 1)
    future_window = config['training'].get('future_window', 1)
    if past_window > 1 or future_window > 1:
        data = create_windowed_samples(data, past_window, future_window)

    # split
    dat_train, dat_val, dat_test = split_with_config(data, config)

    # make into dataloaders
    train_dataloader = DataLoader(
        dat_train,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )

    if config['training']['validate_with_rollout']:
        val_dataloader = DataLoader(
            dat_val,
            batch_size=1,
            shuffle=False
        )

    else:
        val_dataloader = DataLoader(
            dat_val,
            batch_size=config['training']['batch_size'],
            shuffle=False
        )

    test_dataloader = DataLoader(
        dat_test,
        batch_size=1,
        shuffle=False  # don't shuffle for rollout preds
    )

    return train_dataloader, val_dataloader, test_dataloader


def split_and_load_data_NRI(config, args):
    """
    Unique datasplit and dataloader for the NRI model.
    """
    # load data
    with open(args.dataset, 'rb') as f:
        data = pkl.load(f)

    # TODO: make the dataloader from the NRI model


def update_graph_edges(g, new_edge_index):
    """
    Recomputes edge features for a new edge topology using the graph's
    current predicted positions and velocities. Mutates g in place.

    :param g: current graph (with predicted pos/vel)
    :param new_edge_index: new edge index to use
    """
    pos_norm = torch.linalg.norm(g.pos, dim=1)
    _, edge_attr = utils.make_edge_and_nodes(
        g.pos.cpu().detach().numpy(),
        pos_norm.cpu().detach().numpy(),
        g.vel.cpu().detach().numpy(),
        new_edge_index.cpu().detach().numpy()
    )
    g.edge_index = new_edge_index
    g.edge_attr = edge_attr.to(g.pos.device)
    return g


def make_state_graph_acc(model_output, current_graph):
    """
    Makes a single graph from a model prediction
    """
    dt = current_graph.dt
    vel_pred = current_graph.vel + model_output * dt
    pos_pred = current_graph.pos + vel_pred * dt
    pos_norm = torch.linalg.norm(pos_pred, dim=1)
    node_feats, edge_attr = utils.make_edge_and_nodes(
        pos_pred.cpu().detach().numpy(),
        pos_norm.cpu().detach().numpy(),
        vel_pred.cpu().detach().numpy(),
        current_graph.edge_index.cpu().detach().numpy()
    )

    # make new graph
    graph = Data(
        x=node_feats,
        edge_attr=edge_attr,
        edge_index=current_graph.edge_index,
        pos=pos_pred,
        vel=vel_pred,
        dt=dt
    )

    return graph


def make_state_graph_vel(model_output, current_graph):
    """
    Makes a single graph from a model prediction
    """
    pos_pred = model_output * current_graph.dt + current_graph.pos
    pos_norm = torch.linalg.norm(pos_pred, axis=1)
    node_feats, edge_attr = utils.make_edge_and_nodes(
        pos_pred.cpu().detach().numpy(),
        pos_norm.cpu().detach().numpy(),
        model_output.cpu().detach().numpy(),
        current_graph.edge_index.cpu().detach().numpy()
    )

    # make new graph
    graph = Data(
        x=node_feats,
        edge_attr=edge_attr,
        edge_index=current_graph.edge_index,
        pos=pos_pred,
        vel=model_output,
        dt=current_graph.dt
    )

    return graph
