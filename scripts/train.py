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
import torch.nn.functional as F
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
    Loss that uses future positions as targets rather than acceleration.
    Supports multi-step prediction via autoregressive Euler integration.

    :param output: model output, shape [N, future_window * 2] (accelerations)
    :param g: current graph with pos_future [N, future_window * 2]
    """
    dt = torch.mean(g.dt)  # scalar
    future_window = output.shape[1] // 2
    acc_pred = output.view(-1, future_window, 2)          # [N, H, 2]
    pos_target = g.pos_future.view(-1, future_window, 2)  # [N, H, 2]

    total_loss = 0.0
    pos_curr = g.pos
    vel_curr = g.vel
    for k in range(future_window):
        a = acc_pred[:, k, :]
        pos_curr = a * 0.5 * dt ** 2 + vel_curr * dt + pos_curr
        vel_curr = vel_curr + a * dt
        total_loss += torch.nn.functional.mse_loss(pos_curr, pos_target[:, k, :])

    return total_loss / future_window


def switching_nri_loss(preds, prob, logits, g, model, beta):
    """
    Combined loss for SwitchingNRIModel.

    :param preds:  [N, T-1, node_step_dim]  within-context next-state predictions
    :param prob:   [E, T, edge_types]        soft edge-type probabilities
    :param logits: [E, T, edge_types]        smoothed logits (for switching cost)
    :param g:      current graph batch
    :param model:  SwitchingNRIModel instance (for dims and lambda_switch)
    :param beta:   KL weight
    :return:       scalar loss, (recon, kl, switch) for logging
    """
    T         = model.past_window
    node_dim  = model.node_step_dim

    # Ground-truth within-context targets: state at t+1 for each t in [0, T-2]
    x_seq  = g.x.view(-1, T, node_dim)   # [N, T, node_dim]
    target = x_seq[:, 1:, :]             # [N, T-1, node_dim]

    recon_loss = F.mse_loss(preds, target)

    # KL(Categorical(prob_t) || Uniform(1/K)), summed over edges and timesteps
    K = model.edge_types
    log_uniform = -torch.log(torch.tensor(K, dtype=torch.float, device=prob.device))
    kl_loss = (prob * (torch.log(prob + 1e-8) + log_uniform)).sum(-1).mean()

    switch_loss = model.switching_loss(logits)

    total = recon_loss + beta * kl_loss + switch_loss
    return total, (recon_loss.item(), kl_loss.item(), switch_loss.item())


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
            loss = loss_fn(out, g.acc_future)

        running_loss.append(loss.item())
    
    return sum(running_loss) / len(running_loss)


def evaluate_rollout(model, dataloader, writer, config, dataset, device):
    """
    Evaluate how well the model does on rollout prediction.
    Maintains a rolling buffer of `past_window` recent graphs to feed as
    context input. Only the first predicted future step is used to advance
    the rollout state.

    :param model: trained model
    :param dataloader: dataloader to evaluate model on
    :param device: device
    """
    model.eval()
    pred_pos = []
    loss_roll = []
    step = config['training']['downsample_timestep']
    past_window = config['training'].get('past_window', 1)
    future_window = config['training'].get('future_window', 1)

    dat = list(dataloader)
    pred_pos.append(dat[0].pos.detach().cpu())

    # Seed the rolling buffer with real data for the first past_window frames
    buf = [dat[j].to(device) for j in range(min(past_window, len(dat)))]
    if config['training']['gt_edges']:
        for g in buf:
            g.edge_index = g.gt_edge_index
            g.edge_attr = g.gt_edge_attr

    i = past_window - 1
    while i + step < len(dat):
        g_gt = dat[i].to(device)

        # Build windowed input from rolling buffer
        g_in = utils.build_windowed_input(buf, g_gt, config).to(device)

        out = model(g_in)

        # Extract only the first future-step acceleration [N, 2]
        acc_first = out[:, :2]

        # Advance state from the last buffer entry
        g_next = utils.make_state_graph_acc(acc_first, buf[-1])
        loss = torch.nn.functional.mse_loss(g_next.pos, g_gt.pos_next)
        pred_pos.append(g_next.pos.detach().cpu())
        loss_roll.append(loss.item())

        if writer is not None and dataset == "test":
            writer.add_scalar("test/rollout_loss", loss.item(), i)

        # Update rolling buffer: drop oldest, append newest predicted graph
        buf.pop(0)
        buf.append(g_next)

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
                loss = loss_fn(out, g.acc_future)

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
