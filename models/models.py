"""
Script with all models to train
"""

import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        return self.node_mlp(torch.cat([x_i, edge_attr], axis=1))

    def edge_update(self, x_i, x_j, edge_attr):
        """
        update for edges

        :param x_i: node vector on one side of edge
        :param x_j: node vector on the other side of the edge
        :param edge_attr: edge vector
        """
        return self.edge_mlp(torch.cat([x_i, x_j, edge_attr], axis=1))


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
        return self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=1))


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
        future_window = config['training'].get('future_window', 1)
        out_dim = future_window * 2  # 2D output per predicted timestep
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

        # decode for predicted future positions, subtracting added noise
        # noise correction applied to each predicted 2D step
        if self.training:
            future_window = self.decoder[-1].out_features // 2
            x = self.decoder(x) - node_noise[:, :2].repeat(1, future_window)
        else:
            x = self.decoder(x)

        return x


### NRI model TODO ###

class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm (NRI model)"""
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class MLPEncoder(nn.Module):
    """
    NRI model specific
    """
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(MLPEncoder, self).__init__()

        self.factor = factor

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        return self.fc_out(x)


class MLPDecoder(nn.Module):
    """MLP decoder module for NRI model."""

    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False):
        super(MLPDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                               pre_msg.size(2), self.msg_out_shape,
                               device=single_timestep_inputs.device)

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()

        sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
                 rel_type.size(2)]
        rel_type = rel_type.unsqueeze(1).expand(sizes)

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                 curr_rel_type)
            preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                 preds[0].size(2), preds[0].size(3)]

        output = torch.zeros(sizes, device=inputs.device)

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()


class NRIModel(nn.Module):
    """
    NRI model from Kipf et. al. 2018 adapted for this work.
    """
    def __init__(self, config):
        super(NRIModel, self).__init__()

        # model config
        self.config = config

        # encoder - decoder
        self.encoder = MLPEncoder(
            n_in=config['timesteps'] * config['dims'],
            n_hid=config['encoder_hidden'],
            n_out=config['edge_types'],
            do_prob=config['encoder_dropout'],
            factor=config['factor']
        )
        self.decoder = MLPDecoder(
            n_in_node=config['dims'],
            edge_types=config['edge_types'],
            msg_hid=config['decoder_hidden'],
            msg_out=config['decoder_hidden'],
            n_hid=config['decoder_hidden'],
            do_prob=config['decoder_dropout'],
            skip_first=config['skip_first']
        )

        # Generate off-diagonal interaction graph and register as buffers
        # so they automatically move to the correct device with the model
        off_diag = torch.ones([config['num_atoms'], config['num_atoms']]) - torch.eye(config['num_atoms'])
        rel_rec = F.one_hot(torch.where(off_diag)[0]).float()
        rel_send = F.one_hot(torch.where(off_diag)[1]).float()
        self.register_buffer('rel_rec', rel_rec)
        self.register_buffer('rel_send', rel_send)

    def forward(self, data):
        # data: [batch, num_atoms, timesteps, dims]
        logits = self.encoder(data, self.rel_rec, self.rel_send)
        edges = F.gumbel_softmax(logits, tau=self.config['temp'], hard=self.config['hard'], dim=-1)
        prob = F.softmax(logits, dim=-1)
        output = self.decoder(data, edges, self.rel_rec, self.rel_send)

        return output, prob

### SwitchingNRI Model ###

class SwitchingEdgeDecoder(MessagePassing):
    """
    Single-timestep decoder that uses edge-type-weighted PyG message passing.

    For each edge, runs one MLP per edge type to produce a message, then
    weights those messages by the inferred edge-type probabilities before
    aggregating at each target node. This replaces the manual rel_rec /
    rel_send matrix-multiply approach used in the original NRI MLPDecoder.

    PyG flow convention (source_to_target):
        edge_index[0] = source  → x_j in message()
        edge_index[1] = target  → x_i in message()
    Messages are aggregated (summed) at the target node.
    """

    def __init__(self, node_dim, edge_types, hidden_dim, dropout=0.0, skip_first=False):
        super().__init__(aggr='add')
        self.edge_types = edge_types
        self.skip_first = skip_first
        self.msg_hidden = hidden_dim

        # One MLP per edge type — same structure as original MLPDecoder
        self.msg_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * node_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            for _ in range(edge_types)
        ])

        # Output MLP: aggregated messages + skip-connected node state → delta
        self.out_net = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(self, x, edge_index, edge_probs):
        """
        :param x:          [N, node_dim]   current node states
        :param edge_index: [2, E]          fully-connected directed graph
        :param edge_probs: [E, edge_types] per-edge type probabilities
        :return:           [N, node_dim]   next-state prediction (current + delta)
        """
        aggr = self.propagate(edge_index, x=x, edge_probs=edge_probs)  # [N, msg_hidden]
        delta = self.out_net(torch.cat([x, aggr], dim=-1))
        return x + delta

    def message(self, x_i, x_j, edge_probs):
        """
        :param x_i:        [E, node_dim]   target node features
        :param x_j:        [E, node_dim]   source node features
        :param edge_probs: [E, edge_types] passed through from propagate
        """
        pre_msg = torch.cat([x_i, x_j], dim=-1)  # [E, 2*node_dim]

        start = 1 if self.skip_first else 0
        all_msgs = torch.zeros(pre_msg.size(0), self.msg_hidden, device=pre_msg.device)
        for k in range(start, self.edge_types):
            msg_k = self.msg_nets[k](pre_msg)                   # [E, msg_hidden]
            all_msgs = all_msgs + msg_k * edge_probs[:, k:k+1]  # weight by prob
        return all_msgs


class SwitchingNRIModel(nn.Module):
    """
    Switching NRI: per-timestep graph inference with a GRU sticky prior.

    Unlike vanilla NRI, which commits to a single static graph for the entire
    context window, SwitchingNRI infers a separate edge-type distribution at
    every timestep. A bidirectional GRU applied over the per-step edge logits
    acts as a 'sticky prior' — it carries the current graph belief forward (and
    backward) in time and only updates when there is strong local evidence of a
    change (e.g. a mass entering another's field-of-view in the dynamic spring
    system).

    Architecture
    ------------
    1. Per-step PyG MPNN encoder (reuses existing MPNN class)
       Input : x [N, T*node_dim], edge_attr [E, T*edge_dim]
       Output: raw logits [E, T, edge_types]

    2. Bidirectional GRU temporal smoother (new, sticky-prior)
       Runs over the T time dimension per edge independently
       Output: smoothed logits [E, T, edge_types]

    3. Gumbel-Softmax — samples a discrete graph per timestep

    4. Per-step PyG decoder (SwitchingEdgeDecoder)
       Teacher-forced within-context next-state predictions
       Output: preds [N, T-1, node_step_dim]

    Loss terms (computed in train.py)
    ----------------------------------
    - Reconstruction MSE  : predicted vs. true next node state (within context)
    - KL divergence       : q(A_t) || Uniform, summed over all steps
    - Switching cost      : penalises ||prob_t - prob_{t-1}||^2  (lambda_switch)

    Config keys read from full config dict
    ---------------------------------------
    model : encoder_hidden, encoder_dropout, factor, gru_layers,
            decoder_hidden, decoder_dropout, skip_first,
            edge_types, temp, hard, lambda_switch
    training : past_window
    """

    def __init__(self, node_in_dim, edge_in_dim, config):
        """
        :param node_in_dim: total windowed node feature dim  (T * node_step_dim)
        :param edge_in_dim: total windowed edge feature dim  (T * edge_step_dim)
        :param config:      full YAML config dict (model + training sections)
        """
        super().__init__()

        mc = config['model']
        tc = config['training']

        hidden_dim          = mc['encoder_hidden']
        edge_types          = mc['edge_types']
        enc_dropout         = mc['encoder_dropout']
        dec_dropout         = mc['decoder_dropout']
        gru_layers          = mc.get('gru_layers', 1)
        self.factor         = mc.get('factor', True)
        self.edge_types     = edge_types
        self.temp           = mc['temp']
        self.hard           = mc['hard']
        self.lambda_switch  = mc.get('lambda_switch', 0.0)
        self.past_window    = tc['past_window']

        # Per-step feature dimensions derived from windowed totals
        self.node_step_dim = node_in_dim // self.past_window
        self.edge_step_dim = edge_in_dim // self.past_window

        # --- Encoder projections ---
        self.node_proj = nn.Sequential(
            nn.Linear(self.node_step_dim, hidden_dim), nn.ReLU()
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(self.edge_step_dim, hidden_dim), nn.ReLU()
        )

        # --- Per-step PyG MPNN encoder (reuses existing MPNN class) ---
        # First pass: initial node+edge update
        self.step_mpnn = MPNN(
            in_channels=hidden_dim,
            edge_dim=hidden_dim,
            dropout=enc_dropout,
            out_channels=hidden_dim,
        )
        # Factor-graph second pass: richer representations via a second round
        if self.factor:
            self.step_mpnn2 = MPNN(
                in_channels=hidden_dim,
                edge_dim=hidden_dim,
                dropout=enc_dropout,
                out_channels=hidden_dim,
            )

        # Linear head: edge embedding → raw per-step edge type logits
        self.edge_logit_head = nn.Linear(hidden_dim, edge_types)

        # --- Bidirectional GRU temporal smoother ---
        # Treats each of the E edges as an independent sequence of length T.
        # Bidirectional so it can detect a graph change mid-context and label
        # all timesteps correctly (not just future ones).
        self.gru = nn.GRU(
            input_size=edge_types,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=enc_dropout if gru_layers > 1 else 0.0,
        )
        self.gru_out = nn.Linear(2 * hidden_dim, edge_types)

        # --- Per-step decoder ---
        self.decoder = SwitchingEdgeDecoder(
            node_dim=self.node_step_dim,
            edge_types=edge_types,
            hidden_dim=mc['decoder_hidden'],
            dropout=dec_dropout,
            skip_first=mc.get('skip_first', False),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def encode(self, x, edge_attr, edge_index):
        """
        Encode the full temporal context into per-step edge type logits.

        :param x:          [N, T*node_step_dim]
        :param edge_attr:  [E, T*edge_step_dim]
        :param edge_index: [2, E]
        :return:           smoothed logits [E, T, edge_types]
        """
        T = self.past_window
        x_seq = x.view(x.size(0), T, self.node_step_dim)               # [N, T, d_n]
        e_seq = edge_attr.view(edge_attr.size(0), T, self.edge_step_dim)  # [E, T, d_e]

        logits_list = []
        for t in range(T):
            x_in = self.node_proj(x_seq[:, t, :])   # [N, hidden]
            e_in = self.edge_proj(e_seq[:, t, :])   # [E, hidden]

            # PyG MPNN: update both node and edge embeddings for timestep t
            x_enc, e_enc = self.step_mpnn(x_in, e_in, edge_index)
            if self.factor:
                x_enc, e_enc = self.step_mpnn2(x_enc, e_enc, edge_index)

            logits_list.append(self.edge_logit_head(e_enc))  # [E, edge_types]

        raw_logits = torch.stack(logits_list, dim=1)       # [E, T, edge_types]

        # GRU smoother: each edge's T-step logit sequence processed independently
        gru_out, _ = self.gru(raw_logits)                  # [E, T, 2*hidden]
        return self.gru_out(gru_out)                       # [E, T, edge_types]

    def decode(self, x, edge_index, edge_probs):
        """
        Predict within-context next node states (teacher-forced).

        :param x:          [N, T*node_step_dim]
        :param edge_index: [2, E]
        :param edge_probs: [E, T, edge_types]   sampled per-step edge types
        :return:           [N, T-1, node_step_dim]
        """
        T = self.past_window
        x_seq = x.view(x.size(0), T, self.node_step_dim)  # [N, T, d_n]

        preds = []
        for t in range(T - 1):
            x_curr  = x_seq[:, t, :].contiguous()  # [N, node_step_dim]
            probs_t = edge_probs[:, t, :]           # [E, edge_types]
            preds.append(self.decoder(x_curr, edge_index, probs_t))

        return torch.stack(preds, dim=1)  # [N, T-1, node_step_dim]

    def switching_loss(self, logits):
        """
        Penalises rapid changes in the inferred graph across consecutive steps.
        Weight lambda_switch=0 disables the term entirely (no extra computation).

        :param logits: [E, T, edge_types]
        :return:       scalar
        """
        if self.lambda_switch == 0.0:
            return torch.tensor(0.0, device=logits.device)
        probs = F.softmax(logits, dim=-1)               # [E, T, edge_types]
        diff  = probs[:, 1:, :] - probs[:, :-1, :]     # [E, T-1, edge_types]
        return self.lambda_switch * (diff ** 2).mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, graph):
        """
        :param graph: PyG Data with windowed x and edge_attr
        :return:
            preds:  [N, T-1, node_step_dim]  within-context next-state predictions
            prob:   [E, T, edge_types]        soft edge-type probabilities
            logits: [E, T, edge_types]        smoothed logits (for switching loss)
        """
        x, edge_attr, edge_index = graph.x, graph.edge_attr, graph.edge_index

        # Step 1 — encode: per-step MPNN + GRU smoother → smoothed logits
        logits = self.encode(x, edge_attr, edge_index)   # [E, T, edge_types]

        # Step 2 — Gumbel-Softmax: sample a discrete graph per timestep
        E, T, K = logits.shape
        edges = F.gumbel_softmax(
            logits.view(E * T, K), tau=self.temp, hard=self.hard, dim=-1
        ).view(E, T, K)                                  # [E, T, edge_types]

        prob = F.softmax(logits, dim=-1)                 # [E, T, edge_types]

        # Step 3 — decode: teacher-forced within-context predictions
        preds = self.decode(x, edge_index, edges)        # [N, T-1, node_step_dim]

        return preds, prob, logits


if __name__ == '__main__':
    g = pkl.load(open('data/fish/processed/8fish/240816f1.pkl', 'rb'))[0] # graph
    model = LearnedSimModel(
        graph_edge_dim=g.edge_attr.shape[1],
        graph_node_dim=g.x.shape[1]
    )
    out = model(g)
    print('done')
