# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import pickle
import torch
import os

import prody as pr
import pytorch_lightning as pl
import torch.nn as nn

from argparse import Namespace
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Any, Dict, List, Literal, Optional, Union

from typeguard import typechecked
from torchtyping import patch_typeguard

patch_typeguard()  # use before @typechecked

MODEL_WATCHING_LOGGERS = [pl.loggers.wandb.WandbLogger]

LR_SCHEDULER_MANUAL_INTERPOLATION_PRIMARY_CONFIG_ITEMS = ["step_size", "start_factor", "total_iters"]
LR_SCHEDULER_MANUAL_INTERPOLATION_HELPER_CONFIG_ITEMS = ["warmup_steps"]

pr.confProDy(verbosity="none")


@typechecked
def watch_model(
    model: pl.LightningModule,
    logger: pl.loggers.logger.Logger,
    log: Optional[Literal["gradients", "parameters", "all"]] = "gradients",
    log_freq: int = 500,
    log_graph: bool = True
):
    logger.watch(model, log=log, log_freq=log_freq, log_graph=log_graph)


@typechecked
def unwatch_model(
    model: pl.LightningModule,
    logger: pl.loggers.logger.Logger
):
    logger.experiment.unwatch(model)


@typechecked
def unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """
    Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # init empty result tensor
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


@typechecked
def unsorted_segment_mean(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """
    Custom PyTorch op to replicate TensorFlow's `unsorted_segment_mean`.
    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules
    """
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # init empty result tensor
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


@typechecked
def get_classifier_model(args: Namespace) -> nn.Module:
    if args.model_name == "egnn":
        model = EGNN(
            in_node_nf=5,
            in_edge_nf=0,
            hidden_nf=args.nf,
            device=args.device,
            n_layers=args.n_layers,
            coords_weight=1.0,
            attention=args.attention,
            node_attr=args.node_attr
        )
    else:
        raise Exception(f"Wrong model name: {args.model_name}")

    return model
    

@typechecked
def get_classifier(
    model_dir: str = "",
    device: Union[torch.device, str] = "cpu"
) -> nn.Module:
    with open(os.path.join(model_dir, "args.pickle"), "rb") as f:
        args_classifier = pickle.load(f)

    args_classifier.device = device
    args_classifier.model_name = "egnn"
    classifier = get_classifier_model(args_classifier)
    classifier_state_dict = torch.load(
        os.path.join(model_dir, "best_checkpoint.npy"),
        map_location=torch.device("cpu")
    )
    classifier.load_state_dict(classifier_state_dict)

    return classifier


@typechecked
def get_classifier_adj_matrix(
    n_nodes: int,
    batch_size: int,
    device: Union[torch.device, str],
    edges_dic: Dict[Any, Any]
) -> List[torch.Tensor]:
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)
    else:
        edges_dic[n_nodes] = {}
        return get_classifier_adj_matrix(n_nodes, batch_size, device, edges_dic=edges_dic)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges


@typechecked
def train_with_property_classifier(
    model: nn.Module,
    epoch: int,
    dataloader: object,
    mean: float,
    mad: float,
    property: str,
    device: Union[torch.device, str],
    partition: str = "train",
    optimizer: Optional[Optimizer] = None,
    lr_scheduler: Optional[_LRScheduler] = None,
    log_interval: int = 20,
    debug_break: bool = False
) -> float:
    loss_l1 = nn.L1Loss()
    if partition == "train":
        lr_scheduler.step()
    res = {"loss": 0, "counter": 0, "loss_arr":[]}
    for i, data in enumerate(dataloader):
        if partition == "train":
            model.train()
            optimizer.zero_grad()

        else:
            model.eval()

        batch_size, n_nodes, _ = data["positions"].size()
        atom_positions = data["positions"].view(batch_size * n_nodes, -1).to(device, torch.float32)
        atom_mask = data["atom_mask"].view(batch_size * n_nodes, -1).to(device, torch.float32)
        edge_mask = data["edge_mask"].to(device, torch.float32)
        nodes = data["one_hot"].to(device, torch.float32)

        nodes = nodes.view(batch_size * n_nodes, -1)
        edges = get_classifier_adj_matrix(n_nodes, batch_size, device, edges_dic={})
        label = data[property].to(device, torch.float32)

        pred = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                     n_nodes=n_nodes)

        if partition == "train":
            loss = loss_l1(pred, (label - mean) / mad)
            loss.backward()
            optimizer.step()
        else:
            loss = loss_l1(mad * pred + mean, label)

        res["loss"] += loss.item() * batch_size
        res["counter"] += batch_size
        res["loss_arr"].append(loss.item())

        prefix = ""
        if partition != "train":
            prefix = ">> %s \t" % partition

        if i % log_interval == 0:
            print(prefix + "Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])))
        if debug_break:
            break

    return res["loss"] / res["counter"]


@typechecked
def test_with_property_classifier(
    model: nn.Module,
    epoch: int,
    dataloader: object,
    mean: float,
    mad: float,
    property: str,
    device: Union[torch.device, str],
    log_interval: int,
    debug_break: bool = False
) -> float:
    return train_with_property_classifier(
        model=model,
        epoch=epoch,
        dataloader=dataloader,
        mean=mean,
        mad=mad,
        property=property,
        device=device,
        partition="test",
        log_interval=log_interval,
        debug_break=debug_break
    )


class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function (`attn` or `sum`).
          temp: Softmax temperature.

    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1


        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)


        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())


    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100)  # this is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord


    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr


class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function (`attn` or `sum`).
          temp: Softmax temperature.
    
    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention)

        del self.coord_mlp
        self.act_fn = act_fn

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, node_mask, edge_mask, edge_attr=None, node_attr=None, n_nodes=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        edge_feat = edge_feat * edge_mask

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
    """
    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules
    """
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device="cpu", act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)
