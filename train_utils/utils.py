"""
Utility functions for training script
"""

import os
import pickle as pkl
from sklearn.model_selection import train_test_split

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from dataset_utils import utils


def save_model(model, args):
    """
    Saves model
    """
    dataset = os.path.basename(args.dataset).replace('.pkl', '')
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_path, f'{dataset}.pt'))


def split_and_load_data(config, args):
    """
    Loads in the dataset of interest and splits the data    
    """
    # load data
    with open(args.dataset, 'rb') as f:
        data = pkl.load(f)
    
    # split
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

    # make into dataloaders
    train_dataloader = DataLoader(
        dat_train,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_dataloader = DataLoader(
        dat_val,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    test_dataloader = DataLoader(
        dat_test,
        batch_size=1,
        shuffle=False  # don't shuffle for rollout preds
    )

    return train_dataloader, val_dataloader, test_dataloader


def make_state_graph(model_ouput, current_graph):
    """
    Makes a single graph from a model prediction
    """
    pos_pred = model_ouput + current_graph.pos
    pos_norm = torch.linalg.norm(pos_pred, axis=1)
    vel = current_graph.pos - pos_pred  # magnitude doesn't matter because only heading direction is encoded
    node_feats, edge_attr = utils.make_edge_and_nodes(
        pos_pred.cpu().detach().numpy(),
        pos_norm.cpu().detach().numpy(),
        vel.cpu().detach().numpy(),
        current_graph.edge_index.cpu().detach().numpy()
    )

    # make new graph
    graph = Data(
        x=node_feats,
        edge_attr=edge_attr,
        edge_index=current_graph.edge_index,
        pos=pos_pred,
        vel=vel,
    )

    return graph
