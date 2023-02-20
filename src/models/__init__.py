# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from functools import partial
from matplotlib.lines import Line2D
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
from wandb.sdk.wandb_run import Run

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from src.models.components import save_xyz_file, visualize_mol_chain

patch_typeguard()  # use before @typechecked

HALT_FILE_EXTENSION = "done"


@typechecked
def get_nonlinearity(nonlinearity: Optional[str] = None, slope: float = 1e-2, return_functional: bool = False) -> Any:
    nonlinearity = nonlinearity if nonlinearity is None else nonlinearity.lower().strip()
    if nonlinearity == "relu":
        return F.relu if return_functional else nn.ReLU()
    elif nonlinearity == "leakyrelu":
        return partial(F.leaky_relu, negative_slope=slope) if return_functional else nn.LeakyReLU(negative_slope=slope)
    elif nonlinearity == "selu":
        return partial(F.selu) if return_functional else nn.SELU()
    elif nonlinearity == "silu":
        return partial(F.silu) if return_functional else nn.SiLU()
    elif nonlinearity == "sigmoid":
        return torch.sigmoid if return_functional else nn.Sigmoid()
    elif nonlinearity is None:
        return nn.Identity()
    else:
        raise NotImplementedError(f"The nonlinearity {nonlinearity} is currently not implemented.")


@typechecked
def compute_mean_mad(
    dataloaders: Dict[str, DataLoader],
    properties: List[str],
    dataset_name: str
) -> Dict[str, Dict[str, torch.Tensor]]:
    if dataset_name == "QM9":
        return compute_mean_mad_from_dataloader(dataloaders["train"], properties)
    elif dataset_name == "QM9_second_half":
        return compute_mean_mad_from_dataloader(dataloaders["valid"], properties)
    else:
        raise Exception("Invalid dataset name was given.")


@typechecked
def compute_mean_mad_from_dataloader(
    dataloader: DataLoader,
    properties: List[str]
) -> Dict[str, Dict[str, torch.Tensor]]:
    property_norms = {}
    for property_key in properties:
        values = dataloader.dataset.data[property_key]
        mean = torch.mean(values)
        ma = torch.abs(values - mean)
        mad = torch.mean(ma)
        property_norms[property_key] = {}
        property_norms[property_key]["mean"] = mean
        property_norms[property_key]["mad"] = mad
    return property_norms


@typechecked
def inflate_batch_array(array: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Inflate the batch array (`array`) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
    axes (i.e., shape (batch_size, 1, ..., 1)) to match the target shape.
    """
    target_shape = (array.shape[0],) + (1,) * (len(target.shape) - 1)
    return array.view(target_shape)


@typechecked
def get_grad_norm(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    norm_type: float = 2.0
) -> torch.Tensor:
    """
    Adapted from: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    norm_type = float(norm_type)

    device = parameters[0].grad.device
    if len(parameters) == 0:
        return torch.tensor(0.0, device=device)

    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type) for p in parameters]
        ),
        p=norm_type
    )
    return total_norm


@typechecked
def batch_tensor_to_list(
    data: torch.Tensor,
    batch_index: TensorType["batch_num_nodes"]
) -> Tuple[torch.Tensor, ...]:
    # note: assumes that `batch_index` is sorted in non-decreasing order
    chunk_sizes = torch.unique(batch_index, return_counts=True)[1].tolist()
    return torch.split(data, chunk_sizes)


@typechecked
def reverse_tensor(x: torch.Tensor) -> torch.Tensor:
    return x[torch.arange(x.size(0) - 1, -1, -1)]


@typechecked
def log_grad_flow_lite(
    named_parameters: Iterator[Tuple[str, nn.Parameter]],
    wandb_run: Optional[Run] = None
):
    """
    Log a lightweight gradient flowing through different layers in a network during training.
    Can be used to check for possible gradient vanishing/exploding problems.

    Usage: Plug this function in the `LightningModule` class after loss.backwards() as 
    `plot_grad_flow(self.model.named_parameters(), wandb_run=run)` to visualize the gradient flow.
    To do so, call this function within its `on_after_backward()` hook.
    """
    if wandb_run is not None:
        avg_grads = []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and p.grad is not None and ("bias" not in n):
                layers.append(n)
                avg_grads.append(p.grad.abs().mean().cpu().item())
        plt.plot(avg_grads, alpha=0.3, color="b")
        plt.hlines(y=0, xmin=0, xmax=len(avg_grads) + 1, linewidth=1, color="k")
        plt.xticks(ticks=range(0, len(avg_grads), 1), labels=layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(avg_grads))
        plt.xlabel("Layers")
        plt.ylabel("Average gradient")
        plt.grid(True)

        wandb_run.log({"Gradient flow": plt})


@typechecked
def log_grad_flow_full(
    named_parameters: Iterator[Tuple[str, nn.Parameter]],
    wandb_run: Optional[Run] = None
):
    """
    Log a full gradient flowing through different layers in a network during training.
    Can be used to check for possible gradient vanishing/exploding problems.

    Usage: Plug this function in the `LightningModule` class after loss.backwards() as 
    `plot_grad_flow(self.model.named_parameters(), wandb_run=run)` to visualize the gradient flow.
    To do so, call this function within its `on_after_backward()` hook.
    """
    if wandb_run is not None:
        avg_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and p.grad is not None and ("bias" not in n):
                layers.append(n)
                avg_grads.append(p.grad.abs().mean().cpu().item())
                max_grads.append(p.grad.abs().max().cpu().item())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.9, lw=2, color="c")
        plt.bar(np.arange(len(avg_grads)), avg_grads, alpha=0.9, lw=2, color="b")
        plt.hlines(y=0, xmin=0, xmax=len(avg_grads) + 1, lw=2, color="k")
        plt.xticks(ticks=range(0, len(avg_grads), 1), labels=layers, rotation="vertical")
        plt.xlim(left=0, right=len(avg_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("Average gradient")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ["max-gradient", "mean-gradient", "zero-gradient"])

        wandb_run.log({"Gradient flow": plt})


@typechecked
def sample_sweep_conditionally(
    model: nn.Module,
    props_distr: object,
    num_nodes: int = 19,
    num_frames: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes_ = torch.tensor([num_nodes] * num_frames, device=model.device)

    context = []
    for key in props_distr.distributions:
        min_val, max_val = props_distr.distributions[key][num_nodes]['params']
        mean, mad = props_distr.normalizer[key]['mean'], props_distr.normalizer[key]['mad']
        min_val = ((min_val - mean) / (mad)).cpu().numpy()
        max_val = ((max_val - mean) / (mad)).cpu().numpy()
        context_row = torch.tensor(np.linspace(min_val, max_val, num_frames)).unsqueeze(1)
        context.append(context_row)
    context = torch.cat(context, dim=-1).float().to(model.device)

    x, one_hot, charges, batch_index = model.sample(
        num_samples=num_frames,
        num_nodes=num_nodes_,
        context=context,
        fix_noise=True
    )
    return x, one_hot, charges, batch_index


@typechecked
def save_and_sample_conditionally(
    cfg: DictConfig,
    model: nn.Module,
    props_distr: object,
    dataset_info: Dict[str, Any],
    epoch: int = 0,
    id_from: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, one_hot, charges, batch_index = sample_sweep_conditionally(
        model=model,
        props_distr=props_distr
    )

    save_xyz_file(
        path=f"outputs/{cfg.experiment_name}/analysis/run{epoch}/",
        positions=x,
        one_hot=one_hot,
        charges=charges,
        dataset_info=dataset_info,
        id_from=id_from,
        name="conditional",
        batch_index=batch_index
    )

    visualize_mol_chain(
        path=f"outputs/{cfg.experiment_name}/analysis/run{epoch}/",
        dataset_info=dataset_info,
        wandb_run=None,
        spheres_3d=True,
        mode="conditional"
    )

    return x, one_hot, charges


class NumNodesDistribution(nn.Module):
    """
    Adapted from: https://github.com/ehoogeboom/e3_diffusion_for_molecules
    """

    def __init__(
        self,
        histogram: Dict[int, int],
        verbose: bool = True,
        eps: float = 1e-30
    ):
        super().__init__()

        self.eps = eps

        num_nodes, self.keys, prob = [], {}, []
        for i, nodes in enumerate(histogram):
            num_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.register_buffer("num_nodes", torch.tensor(num_nodes))

        self.register_buffer("prob", torch.tensor(prob))

        self.prob = self.prob / torch.sum(self.prob)
        self.m = Categorical(self.prob)

        if verbose:
            entropy = torch.sum(self.prob * torch.log(self.prob + eps))
            print("Entropy of n_nodes: H[N]", entropy.item())

    @typechecked
    def sample(self, n_samples: int = 1) -> torch.Tensor:
        idx = self.m.sample((n_samples,))
        return self.num_nodes[idx]

    @typechecked
    def log_prob(self, batch_n_nodes: TensorType["batch_size"]) -> TensorType["batch_size"]:
        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs, device=batch_n_nodes.device)

        log_p = torch.log(self.prob + self.eps)
        log_probs = log_p[idcs]

        return log_probs


class PropertiesDistribution:
    """
    Adapted from: https://github.com/ehoogeboom/e3_diffusion_for_molecules
    """

    def __init__(
        self,
        dataloader: DataLoader,
        properties: List[str],
        device: Union[torch.device, str],
        num_bins: int = 1000,
        normalizer: Optional[Dict[str, Dict[str, float]]] = None
    ):
        self.properties = properties
        self.device = device
        self.num_bins = num_bins
        self.normalizer = normalizer

        self.distributions = {}
        for prop in properties:
            self.distributions[prop] = {}
            # move atom counts and properties to `device` once e.g., at the beginning of training
            self._create_prob_dist(dataloader.dataset.data["num_atoms"].to(device),
                                   dataloader.dataset.data[prop].to(device),
                                   self.distributions[prop])

    @typechecked
    def set_normalizer(self, normalizer: Dict[str, Dict[str, torch.Tensor]]):
        self.normalizer = normalizer

    @typechecked
    def _create_prob_dist(
        self,
        nodes_arr: TensorType["num_examples"],
        values: TensorType["num_examples"],
        distribution: Dict[int, Dict[str, torch.Tensor]]
    ):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {"probs": probs, "params": params}

    @typechecked
    def _create_prob_given_nodes(
        self,
        values: TensorType["num_matched_values"],
        eps: float = 1e-12
    ) -> Tuple[Categorical, Tuple[torch.Tensor, torch.Tensor]]:
        n_bins = self.num_bins  # min(self.num_bins, len(values))
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + eps
        histogram = torch.zeros(n_bins, device=self.device)
        for val in values:
            i = int((val - prop_min)/prop_range * n_bins)
            # Note: Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins - 1).
            # Hence, we move it to bin int(n_bins - 1) if that happens.
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1
        probs = Categorical(histogram / torch.sum(histogram))
        params = (prop_min, prop_max)
        return probs, params

    @typechecked
    def normalize_tensor(self, tensor: torch.Tensor, prop: str) -> torch.Tensor:
        assert self.normalizer is not None, "To normalize properties, `normalizer` must not be null."
        mean = self.normalizer[prop]["mean"]
        mad = self.normalizer[prop]["mad"]
        return (tensor - mean) / mad

    @typechecked
    def sample(self, num_nodes: int = 19) -> TensorType["num_properties"]:
        vals = []
        for prop in self.properties:
            dist = self.distributions[prop][num_nodes]
            idx = dist["probs"].sample((1,))
            val = self._idx2value(idx, dist["params"], len(dist["probs"].probs))
            val = self.normalize_tensor(val, prop)
            vals.append(val)
        vals = torch.cat(vals)
        return vals

    @typechecked
    def sample_batch(
        self,
        num_nodes: TensorType["batch_size"]
    ) -> TensorType["batch_size", "num_properties"]:
        vals = torch.cat([self.sample(n.item()).unsqueeze(0) for n in num_nodes], dim=0)
        return vals

    @typechecked
    def _idx2value(
        self,
        idx: torch.Tensor,
        params: Tuple[torch.Tensor, torch.Tensor],
        num_bins: int
    ) -> torch.Tensor:
        prop_range = params[1] - params[0]
        left = idx / num_bins * prop_range + params[0]
        right = (idx + 1) / num_bins * prop_range + params[0]
        val = torch.rand(1, device=self.device) * (right - left) + left
        return val


class CategoricalDistribution:
    EPS = 1e-10

    def __init__(self, histogram_dict: Dict[int, int], mapping: Dict[str, int]):
        histogram = np.zeros(len(mapping))
        for k, v in histogram_dict.items():
            histogram[k] = v

        # normalize histogram
        self.p = histogram / histogram.sum()
        self.mapping = mapping

    @typechecked
    def kl_divergence(self, other_samples: List[int]) -> float:
        sample_histogram = np.zeros(len(self.mapping))
        for x in other_samples:
            sample_histogram[x] += 1

        # normalize
        q = sample_histogram / sample_histogram.sum()

        return -np.sum(self.p * np.log(q / self.p + self.EPS))


class Queue():
    """
    Adapted from: https://github.com/arneschneuing/DiffSBDD
    """

    def __init__(self, max_len: int = 50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    @typechecked
    def add(self, item: Any):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    @typechecked
    def mean(self) -> Any:
        return np.mean(self.items)

    @typechecked
    def std(self) -> Any:
        return np.std(self.items)
