"""
Training script for models
"""

import os
import copy
import yaml
import argparse
import pickle as pkl
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from models import models
from train_utils import utils


def evaluate(model, dataloader, device):
    """
    Evaluate the model on a dataloader without training
    
    :param model: trained model
    :param dataloader: dataloader to evaluate model on
    :param device: device
    """
    model.eval()
    running_loss = []
    for g in dataloader:
        g = g.to(device)
        out = model(g)
        loss = loss_fn(out, g.pos, g.pos_next)
        running_loss.append(loss.item())
    
    return sum(running_loss) / len(running_loss)


def evaluate_rollout(model, dataloader, writer, device):
    """
    Evaluate how well the model does on rollout prediction
    
    :param model: trained model
    :param dataloader: dataloader to evaluate model on
    :param device: device
    """
    model.eval()
    pred_pos = []
    
    # initial graph
    g = next(iter(dataloader))

    # loop through range of dataloader
    for i, g_gt in enumerate(dataloader):
        g_gt = g_gt.to(device)
        g = g.to(device)
        out = model(g)
        loss = loss_fn(out, g.pos, g_gt.pos_next)
        g = utils.make_state_graph(out, g)
        pred_pos.append(g.pos)
        writer.add_scalar("test/rollout_loss", loss.item(), i)

    return pred_pos


def loss_fn(pred, pos, target):
    """
    Euler integration mse loss

    :param pred: model prediction
    :param graph: graph at current time step
    :param graph: time step in the future that needs to be predicted
    """
    pos_pred = pos + pred
    loss = torch.nn.functional.mse_loss(pos_pred, target)

    return loss


def train(args):
    """
    Main training function
    """
    # get device
    device = torch.device(f'cuda' if torch.cuda.is_available() else "cpu")

    # get config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # load data and split
    train_dataloader, val_dataloader, test_dataloader = utils.split_and_load_data(config, args)

    # tensorboard log
    dataset = os.path.basename(args.dataset).replace('.pkl', '')
    writer = SummaryWriter(log_dir=os.path.join(args.log_path, dataset))

    # init model
    model = models.LearnedSimModel(
        graph_edge_dim=next(iter(train_dataloader)).edge_attr.shape[1],
        graph_node_dim=next(iter(train_dataloader)).x.shape[1],
        config=config
    ).to(device)

    # optimizer
    opt = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

    # training loop
    best_val_loss = 1e8
    best_model = copy.deepcopy(model)
    for epoch in tqdm(range(config['training']['epochs']), desc="epoch"):
        model.train()
        running_loss = []
        for g in train_dataloader:
            g = g.to(device)
            opt.zero_grad()
            out = model(g)
            loss = loss_fn(out, g.pos, g.pos_next)
            loss.backward()
            opt.step()
            running_loss.append(loss.item())

        # report to tensorboard
        avg_loss = sum(running_loss) / len(running_loss)
        writer.add_scalar("train/loss", avg_loss, epoch)

        # evaluate on validation data
        with torch.no_grad():
            avg_val_loss = evaluate(model, val_dataloader, device)
            writer.add_scalar("val/loss", avg_val_loss, epoch)

            # saving best performing model
            if avg_val_loss < best_val_loss:
                utils.save_model(model, args)
                best_model = copy.deepcopy(model)
                best_val_loss = avg_val_loss

    # get test loss
    with torch.no_grad():
        one_step_loss = evaluate(best_model, test_dataloader, device)
        writer.add_scalar("test/loss", one_step_loss, 0)
        pred_pos = evaluate_rollout(best_model, test_dataloader, writer, device)

    # save rollout predictions
    with open(os.path.join(args.save_path, f'{dataset}_rollout_preds.pkl'), 'wb') as f:
        pkl.dump(pred_pos, f)

    # close logs
    writer.close()


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', type=str, help='Model to be trained.', required=False, default="MPNN"
    )
    parser.add_argument(
        '-c', '--config', type=str, help='Path to config file.', required=False, default="configs/mpnn.yaml"
    )
    parser.add_argument(
        '-d', '--dataset', type=str, help='Path to dataset.', required=False, default="data/spring_mass/static_graph/graphs/trial_0.pkl"
    )
    parser.add_argument(
        '-sp', '--save_path', type=str, help='model save path. (e.g. path/to/model.pt)', required=False, default="/projectnb/biochemai/Grant/interaction_rule_GNN/results/SpringMass/MPNN/model/"
    )
    parser.add_argument(
        '-lp', '--log_path', type=str, help='model log path.', required=False, default="/projectnb/biochemai/Grant/interaction_rule_GNN/results/SpringMass/MPNN/logs"
    )
    args = parser.parse_args()
    train(args)
