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


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pos_loss_fn(output, g):
    """
    Loss that uses next position as target rather than acceleration

    :param output: model output (acceleration)
    :param g: current graph
    """
    # euler integration for next position
    dt = torch.mean(g.dt)  # to make it a scalar
    pos_pred = output * 1/2 * dt ** 2 + g.vel * dt + g.pos
    loss = torch.nn.functional.mse_loss(pos_pred, g.pos_next)

    return loss


def evaluate(model, dataloader, loss_fn, config, device):
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

        # if using ground truth edges
        if config['training']['gt_edges']:
            g.edge_index = g.gt_edge_index
            g.edge_attr = g.gt_edge_attr
        
        out = model(g)

        if config['training']['pred_pos']:
            loss = loss_fn(out, g)
        else:
            loss = loss_fn(out, g.acc)

        running_loss.append(loss.item())
    
    return sum(running_loss) / len(running_loss)


def evaluate_rollout(model, dataloader, writer, config, dataset, device):
    """
    Evaluate how well the model does on rollout prediction
    
    :param model: trained model
    :param dataloader: dataloader to evaluate model on
    :param device: device
    """
    model.eval()
    pred_pos = []
    loss_roll = []
    step = config['training']['downsample_timestep']

    # initial graph
    dat = list(dataloader)
    g = dat[0]
    pred_pos.append(g.pos.detach().cpu())

    if config['training']['gt_edges']:
        g.edge_index = g.gt_edge_index
        g.edge_attr = g.gt_edge_attr

    # loop through range of dataloader
    i = 0
    while i + step < len(dat):
        g_gt = dat[i].to(device)
        g = g.to(device)

        # if using ground truth edges, update topology and recompute edge features
        # from current predicted state so edge_attr stays consistent with edge_index
        if config['training']['gt_edges']:
            g = utils.update_graph_edges(g, g_gt.gt_edge_index)

        out = model(g)
        g = utils.make_state_graph_acc(out, g)
        loss = torch.nn.functional.mse_loss(g.pos, g_gt.pos_next)
        pred_pos.append(g.pos.detach().cpu())
        loss_roll.append(loss.item())

        if writer is not None and dataset == "test":
            writer.add_scalar("test/rollout_loss", loss.item(), i)

        # advance graph by step
        i += step

    return pred_pos, sum(loss_roll) / len(loss_roll)


def train(args):
    """
    Main training function
    """
    # get device
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    # get config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # load data and split
    if config['model']['gnn_type'] == 'NRI':
        train_dataloader, val_dataloader, test_dataloader = utils.split_and_load_data_NRI(config, args)
    else:
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

    # optimizer and loss
    opt = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

    # changing loss function to either predict acceleration or next poisiton
    if config['training']['pred_pos']:
        loss_fn = pos_loss_fn
    else:
        loss_fn = torch.nn.MSELoss()

    # training loop
    best_val_loss = 1e8
    best_model = copy.deepcopy(model)
    for epoch in tqdm(range(config['training']['epochs']), desc="epoch"):
        model.train()
        running_loss = []
        for g in train_dataloader:
            g = g.to(device)

            if config['training']['gt_edges']:
                g.edge_index = g.gt_edge_index
                g.edge_attr = g.gt_edge_attr

            opt.zero_grad()
            out = model(g)

            if config['training']['pred_pos']:
                loss = loss_fn(out, g)
            else:
                loss = loss_fn(out, g.acc)

            loss.backward()
            opt.step()
            running_loss.append(loss.item())

        # report to tensorboard
        avg_loss = sum(running_loss) / len(running_loss)
        writer.add_scalar("train/loss", avg_loss, epoch)

        # evaluate on validation data
        with torch.no_grad():
            if config['training']['validate_with_rollout']:
                _, avg_val_loss = evaluate_rollout(model, val_dataloader, writer, config, "val", device)
            else:
                avg_val_loss = evaluate(model, val_dataloader, loss_fn, config, device)
            
            writer.add_scalar("val/loss", avg_val_loss, epoch)

            # saving best performing model
            if avg_val_loss < best_val_loss:
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                utils.save_model(best_model, args)
                best_val_loss = avg_val_loss

    print(f"best model epoch: {best_epoch}")

    # get test loss
    with torch.no_grad():
        one_step_loss = evaluate(best_model, test_dataloader, loss_fn, config, device)
        print(f"Test loss (one step): {one_step_loss}")
        writer.add_scalar("test/loss", one_step_loss, 0)
        pred_pos, _ = evaluate_rollout(best_model, test_dataloader, writer, config, "test", device)

    # save rollout predictions
    with open(os.path.join(args.save_path, f'{dataset}_rollout_preds.pkl'), 'wb') as f:
        pkl.dump(pred_pos, f)

    # save config
    with open(os.path.join(args.save_path, f'{dataset}_config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # close logs
    writer.close()


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, help='Path to config file.', required=False, default="configs/mpnn.yaml"
    )
    parser.add_argument(
        '-d', '--dataset', type=str, help='Path to dataset.', required=False, default="data/spring_mass/static_graph/graphs/trial_0.pkl"
    )
    parser.add_argument(
        '-sp', '--save_path', type=str, help='model save path. (e.g. path/to/model.pt)', required=False, default="/projectnb/biochemai/Grant/interaction_rule_GNN/results/SpringMass/MPNN/model/test"
    )
    parser.add_argument(
        '-lp', '--log_path', type=str, help='model log path.', required=False, default="/projectnb/biochemai/Grant/interaction_rule_GNN/results/SpringMass/MPNN/logs/test"
    )
    args = parser.parse_args()
    set_seed()
    train(args)
