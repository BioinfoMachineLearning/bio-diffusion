# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig
from random import random
from torch_geometric.data import Batch
from torch_scatter import scatter
from typing import Any, Dict, List, Optional, Tuple, Union

from src.models.components import centralize, num_nodes_to_batch_index
from src.models import NumNodesDistribution, inflate_batch_array

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from src.utils import make_and_save_network_graphviz_plot
from src.utils.pylogger import get_pylogger

patch_typeguard()  # use before @typechecked


log = get_pylogger(__name__)


H_INPUT_TYPE = Union[
    TensorType["batch_num_nodes", "num_atom_types"],
    torch.Tensor  # note: for when `include_charges=False`
]
NODE_FEATURE_DIFFUSION_TARGETS = ["atom_types_and_coords"]


@typechecked
def cosine_beta_schedule(
    num_timesteps: int,
    s: float = 0.008,
    raise_to_power: float = 1
) -> np.ndarray:
    """
    A cosine variance schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ.
    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py
    """
    steps = num_timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


@typechecked
def clip_noise_schedule(
    alphas2: np.ndarray,
    clip_value: float = 0.001
) -> np.ndarray:
    """
    For a noise schedule given by (alpha ^ 2), this clips alpha_t / (alpha_t - 1).
    This may help improve stability during sampling.
    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


@typechecked
def polynomial_schedule(
    num_timesteps: int,
    s: float = 1e-4,
    power: float = 3.0
) -> np.ndarray:
    """
    A noise schedule based on a simple polynomial equation: 1 - (x ^ power).
    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py
    """
    steps = num_timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


class PositiveLinear(nn.Module):
    """
    A linear layer with weights forced to be positive.
    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Union[torch.device, str],
        bias: bool = True,
        weight_init_offset: int = -2
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device))
        else:
            self.register_parameter("bias", None)
        self.weight_init_offset = weight_init_offset

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @typechecked
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        positive_weight = F.softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class GammaNetwork(nn.Module):
    """
    The gamma network models a monotonically-increasing function. Constructed as in the VDM paper.
    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py
    """

    def __init__(self, verbose: bool = True):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = nn.Parameter(torch.tensor([-5.0]))
        self.gamma_1 = nn.Parameter(torch.tensor([10.0]))

        if verbose:
            self.display_schedule()

    @typechecked
    def display_schedule(self, num_steps: int = 50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        log.info(f"Gamma schedule: {gamma.detach().cpu().numpy().reshape(num_steps)}")

    @typechecked
    def gamma_tilde(self, t: TensorType["batch_size", 1]) -> TensorType["batch_size", 1]:
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    @typechecked
    def forward(self, t: TensorType["batch_size", 1]) -> TensorType["batch_size", 1]:
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)

        # note: not very efficient
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # normalize to [0, 1]
        normalized_gamma = (
            (gamma_tilde_t - gamma_tilde_0) / (gamma_tilde_1 - gamma_tilde_0)
        )

        # rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma


class PredefinedNoiseSchedule(nn.Module):
    """
    A predefined noise schedule. Essentially, creates a lookup array for predefined (non-learned) noise schedules.
    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py
    """

    def __init__(
        self,
        noise_schedule: str,
        num_timesteps: int,
        noise_precision: float,
        verbose: bool = True,
        **kwargs
    ):
        super().__init__()

        self.timesteps = num_timesteps

        if noise_schedule == "cosine":
            alphas2 = cosine_beta_schedule(num_timesteps)
        elif "polynomial" in noise_schedule:
            splits = noise_schedule.split("_")
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(num_timesteps, s=noise_precision, power=power)
        else:
            raise ValueError(noise_schedule)

        if verbose:
            log.info(f"alphas2: {alphas2}")

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        if verbose:
            log.info(f"gamma: {-log_alphas2_to_sigmas2}")

        self.gamma = nn.Parameter(
            torch.tensor(-log_alphas2_to_sigmas2).float(),
            requires_grad=False
        )

    @typechecked
    def forward(self, t: TensorType["batch_size", 1]) -> TensorType["batch_size", 1]:
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class EquivariantVariationalDiffusion(nn.Module):
    """
    The Equivariant Variational Diffusion (EVD) Module.
    """

    def __init__(
        self,
        dynamics_network: nn.Module,
        diffusion_cfg: DictConfig,
        dataloader_cfg: DictConfig,
        dataset_info: Dict[str, Any]
    ):
        super().__init__()

        # ensure config arguments are of valid values and structures
        assert diffusion_cfg.parametrization in ["eps"], "Epsilon is currently the only supported parametrization."
        assert diffusion_cfg.loss_type in [
            "vlb", "l2"
        ], "Variational lower-bound and L2 losses are currently the only supported diffusion loss functions."

        if diffusion_cfg.noise_schedule == "learned":
            assert diffusion_cfg.loss_type == "vlb", "A noise schedule can only be learned with a variational lower-bound objective."

        # hyperparameters #
        self.diffusion_cfg = diffusion_cfg
        self.diffusion_target = diffusion_cfg.diffusion_target
        self.num_atom_types = dataloader_cfg.num_atom_types
        self.num_x_dims = dataloader_cfg.num_x_dims
        self.include_charges = dataloader_cfg.include_charges
        self.num_node_scalar_features = dataloader_cfg.num_atom_types + dataloader_cfg.include_charges
        self.T = diffusion_cfg.num_timesteps

        # Forward pass #

        forward_prefix_mapping = {
            "atom_types_and_coords": "atom_types_and_coords_"
        }
        forward_prefix = forward_prefix_mapping[self.diffusion_target]
        self.forward_fn = getattr(self, forward_prefix + "forward")

        # PyTorch modules #

        # network that will predict the noise
        self.dynamics_network = dynamics_network

        # distribution of node counts
        histogram = {int(k): int(v) for k, v in dataset_info["n_nodes"].items()}
        self.num_nodes_distribution = NumNodesDistribution(histogram)

        # noise schedule
        if diffusion_cfg.noise_schedule == "learned":
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(**diffusion_cfg)

        if diffusion_cfg.noise_schedule != "learned":
            self.detect_issues_with_norm_values()

    @staticmethod
    @typechecked
    def sigma(
        gamma: TensorType["batch_size", 1],
        target_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Compute `sigma` given `gamma`."""
        return inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    @staticmethod
    @typechecked
    def alpha(
        gamma: TensorType["batch_size", 1],
        target_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Compute `alpha` given `gamma`."""
        return inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    @staticmethod
    @typechecked
    def SNR(gamma: TensorType["batch_size", 1]) -> TensorType["batch_size", 1]:
        """Compute signal to noise ratio (SNR) (i.e., alpha ^ 2 / sigma ^ 2) given `gamma`."""
        return torch.exp(-gamma)

    @staticmethod
    @typechecked
    def sigma_and_alpha_t_given_s(
        gamma_t: torch.Tensor,
        gamma_s: torch.Tensor,
        target_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute `sigma(t) | s`, using `gamma_t` and `gamma_s`. Used during sampling.
        These are defined as:
            `alpha(t) | s` = `alpha(t)` / `alpha(s)`,
            `sigma(t) | s` = sqrt(1 - (`alpha(t) | s`) ^ 2).
        """
        sigma2_t_given_s = inflate_batch_array(
            -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)),
            target_tensor
        )

        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = inflate_batch_array(alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    @staticmethod
    @typechecked
    def gaussian_KL(
        q_mu_minus_p_mu_squared: TensorType["batch_size"],
        q_sigma: TensorType["batch_size"],
        p_sigma: TensorType["batch_size"],
        d: Union[int, TensorType["batch_size"]]
    ) -> TensorType["batch_size"]:
        """Compute the KL divergence between two normal distributions.
            Args:
                q_mu_minus_p_mu_squared: Squared difference between mean
                    of distribution `q` and distribution `p`: `|| mu_q - mu_p || ^ 2`.
                q_sigma: Standard deviation of distribution `q`.
                p_sigma: Standard deviation of distribution `p`.
                d: Tensor dimension over which to integrate.
            Returns:
                The KL distance.
            """
        return (
            d * torch.log(p_sigma / q_sigma) +
            0.5 * (d * (q_sigma ** 2) + q_mu_minus_p_mu_squared) /
            (p_sigma ** 2) - 0.5 * d
        )

    @staticmethod
    @typechecked
    def cdf_standard_gaussian(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))

    @staticmethod
    @typechecked
    def sample_center_gravity_zero_gaussian_with_mask(
        size: torch.Size,
        batch_index: TensorType["batch_num_nodes"],
        node_mask: TensorType["batch_num_nodes"],
        device: Union[torch.device, str]
    ) -> torch.Tensor:
        assert len(size) == 2
        x = torch.randn(size, device=device)

        x_masked = x * node_mask.float().unsqueeze(-1)

        # note: this projection only works because Gaussians are
        # rotation-invariant around zero and their samples are independent!
        _, x_projected = centralize(
            Batch(x=x_masked),
            "x",
            batch_index=batch_index,
            node_mask=node_mask,
            edm=True
        )
        return x_projected

    @staticmethod
    @typechecked
    def sample_gaussian(
        size: torch.Size,
        device: Union[torch.device, str]
    ) -> torch.Tensor:
        x = torch.randn(size, device=device)
        return x

    @staticmethod
    @typechecked
    def sample_gaussian_with_mask(
        size: torch.Size,
        node_mask: TensorType["batch_num_nodes"],
        device: Union[torch.device, str]
    ) -> torch.Tensor:
        x = torch.randn(size, device=device)
        x_masked = x * node_mask.float().unsqueeze(-1)
        return x_masked

    @staticmethod
    @typechecked
    def assert_correctly_masked(variable: torch.Tensor, node_mask: torch.Tensor):
        assert (node_mask.all()) or (variable[~node_mask].abs().max().item() < 1e-4), "Variables not masked properly."

    @staticmethod
    @typechecked
    def sum_node_features_except_batch(
        values: TensorType["batch_num_nodes", "num_node_features"],
        batch_index: TensorType["batch_num_nodes"]
    ):
        return scatter(values.sum(-1), batch_index, dim=0, reduce="sum")

    @staticmethod
    @typechecked
    def check_mask_correct(variables: torch.Tensor, node_mask: torch.Tensor):
        for variable in variables:
            if len(variable) > 0:
                assert (node_mask.all()) or (variable[~node_mask].abs(
                ).max().item() < 1e-4), "Variables not masked properly."

    @staticmethod
    @typechecked
    def assert_mean_zero_with_mask(
        x: TensorType["batch_num_nodes", 3],
        node_mask: TensorType["batch_num_nodes"],
        eps: float = 1e-10
    ):
        assert (node_mask.all()) or (x[~node_mask].abs().max().item() < 1e-4), "Variables not masked properly."
        largest_value = x.abs().max().item()
        error = torch.sum(x, dim=0, keepdim=True).abs().max().item()
        rel_error = error / (largest_value + eps)
        assert rel_error < 1e-2, f"Mean is not zero, as relative_error {rel_error}"

    @typechecked
    def detect_issues_with_norm_values(self, num_std_dev: int = 8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # detect if (1 / `norm_value`) is still larger than (10 * standard deviation)
        if len(self.diffusion_cfg.norm_values) > 1:
            norm_value = self.diffusion_cfg.norm_values[1]
            if (sigma_0 * num_std_dev) > (1.0 / norm_value):
                raise ValueError(
                    f"Value for normalization value {norm_value} is probably"
                    f" too large with sigma_0={sigma_0:.5f}"
                    f" and (1 / norm_value = {1.0 / norm_value})"
                )

    @typechecked
    def subspace_dimensionality(
        self,
        num_nodes: TensorType["batch_size"]
    ) -> TensorType["batch_size"]:
        """Compute the dimensionality on translation-invariant linear subspace where distributions on `x` are defined."""
        return (num_nodes - 1) * self.num_x_dims

    @typechecked
    def compute_kl_prior(
        self,
        xh: TensorType["batch_num_nodes", "combined_node_feature_dim"],
        batch_index: TensorType["batch_num_nodes"],
        node_mask: TensorType["batch_num_nodes"],
        num_nodes: TensorType["batch_size"],
        device: Union[torch.device, str],
        generate_x_only: bool = False
    ) -> TensorType["batch_size"]:
        """
        Compute the KL divergence between q(z1 | x) and the prior p(z1) = Normal(0, 1).
        This is essentially a lot of work for something that is in practice negligible in the loss.
        However, you compute it so that you see it when you've made a mistake in your noise schedule.
        """
        batch_size = len(num_nodes)

        # compute the last `alpha` value, `alpha_T`
        ones = torch.ones((batch_size, 1), device=device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh)

        # compute means
        mu_T = alpha_T[batch_index] * xh
        mu_T_x, mu_T_h = mu_T[:, :self.num_x_dims], None if generate_x_only else mu_T[:, self.num_x_dims:]

        # compute standard deviations (only batch axis for `x`-part, inflated for `h`-part)
        sigma_T_x = self.sigma(gamma_T, mu_T_x).squeeze()
        sigma_T_h = None if generate_x_only else self.sigma(gamma_T, mu_T_h).squeeze()

        # compute KL divergence for `x`-part
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        mu_norm = self.sum_node_features_except_batch((mu_T_x - zeros) ** 2, batch_index)
        subspace_d = self.subspace_dimensionality(num_nodes)
        kl_distance_x = self.gaussian_KL(
            q_mu_minus_p_mu_squared=mu_norm,
            q_sigma=sigma_T_x,
            p_sigma=ones,
            d=subspace_d
        )

        # bypass calculations for `h`
        if generate_x_only:
            return kl_distance_x

        # compute KL divergence for `h`-part
        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        mu_norm = ((mu_T_h - zeros) ** 2) * node_mask.float().unsqueeze(-1)
        mu_norm = self.sum_node_features_except_batch(mu_norm, batch_index)
        kl_distance_h = self.gaussian_KL(
            q_mu_minus_p_mu_squared=mu_norm,
            q_sigma=sigma_T_h,
            p_sigma=ones,
            d=1
        )

        return kl_distance_x + kl_distance_h

    @typechecked
    def compute_x_pred(
        self,
        zt: TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        net_out: TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        gamma_t: TensorType["batch_size", 1],
        batch_index: TensorType["batch_num_nodes"]
    ) -> torch.Tensor:
        """Compute `x_pred` (i.e., the prediction of `x` that is most likely)."""
        if self.diffusion_cfg.parametrization == "x":
            x_pred = net_out
        elif self.diffusion_cfg.parametrization == "eps":
            sigma_t = self.sigma(gamma_t, target_tensor=net_out)
            alpha_t = self.alpha(gamma_t, target_tensor=net_out)
            eps_t = net_out
            x_pred = 1.0 / alpha_t[batch_index] * (zt - sigma_t[batch_index] * eps_t)
        else:
            raise ValueError(self.diffusion_cfg.parametrization)

        return x_pred

    @typechecked
    def log_constants_p_x_given_z0(
        self,
        num_nodes: TensorType["batch_size"],
        device: Union[torch.device, str]
    ) -> TensorType["batch_size"]:
        """Compute p(x | z0)."""
        batch_size = len(num_nodes)
        degrees_of_freedom_x = self.subspace_dimensionality(num_nodes)

        zeros = torch.zeros((batch_size, 1), device=device)
        gamma_0 = self.gamma(zeros)

        # recall that `sigma_x` = sqrt(`sigma_0` ^ 2 / `alpha_0` ^ 2) = SNR(-0.5 `gamma_0`).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (-log_sigma_x - 0.5 * np.log(2 * np.pi))

    @typechecked
    def log_pxh_given_z0_without_constants(
        self,
        h: Union[Dict[str, H_INPUT_TYPE], H_INPUT_TYPE],
        z_0: TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        eps: TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        net_out: TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        gamma_0: TensorType["batch_size", 1],
        batch_index: TensorType["batch_num_nodes"],
        node_mask: TensorType["batch_num_nodes"],
        device: Union[torch.device, str],
        generate_x_only: bool = False,
        epsilon: float = 1e-10
    ) -> Tuple[
        TensorType["batch_size"],
        Optional[TensorType["batch_size"]]
    ]:
        # take only part over `x`
        eps_x = eps[:, :self.num_x_dims]
        net_x = net_out[:, :self.num_x_dims]

        # compute the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0 / alpha_0 eps_0, sigma_0 / alpha_0),
        # where the weighting in the epsilon parametrization is exactly `1`.
        log_p_x_given_z0_without_constants = (
            -0.5 * self.sum_node_features_except_batch(
                (eps_x - net_x) ** 2,
                batch_index
            )
        )

        # bypass calculations for `h`
        if generate_x_only:
            return log_p_x_given_z0_without_constants, None

        # note: discrete properties are predicted directly from `z_t`
        z_h_cat = z_0[:, self.num_x_dims:-1] if self.include_charges else z_0[:, self.num_x_dims:]
        z_h_int = z_0[:, -1:] if self.include_charges else torch.zeros(0, device=device)

        # `compute `sigma_0` and rescale to the integer scale of the data
        sigma_0 = self.sigma(gamma_0[batch_index], target_tensor=z_0)
        sigma_0_cat = sigma_0 * self.diffusion_cfg.norm_values[1]
        sigma_0_int = sigma_0 * self.diffusion_cfg.norm_values[2]

        # compute delta indicator masks
        h_integer = (
            h["integer"].reshape(h["integer"].shape[0], -1)
            if self.include_charges
            else h["integer"]
        )
        h_integer = torch.round(
            h_integer * self.diffusion_cfg.norm_values[2] + self.diffusion_cfg.norm_biases[2]
        ).long()
        onehot = h["categorical"] * self.diffusion_cfg.norm_values[1] + self.diffusion_cfg.norm_biases[1]

        estimated_h_integer = z_h_int * self.diffusion_cfg.norm_values[2] + self.diffusion_cfg.norm_biases[2]
        estimated_h_cat = z_h_cat * self.diffusion_cfg.norm_values[1] + self.diffusion_cfg.norm_biases[1]

        # compute integer features' likelihood when using integer features for model training and testing
        if self.include_charges:
            assert h_integer.size() == estimated_h_integer.size()
            h_integer_centered = h_integer - estimated_h_integer

            # compute integral from -0.5 to 0.5 of the normal distribution
            # i.e., N(mean = `h_integer_centered`, stdev = `sigma_0_int`)
            log_ph_integer_given_z0 = torch.log(
                self.cdf_standard_gaussian((h_integer_centered + 0.5) / sigma_0_int)
                - self.cdf_standard_gaussian((h_integer_centered - 0.5) / sigma_0_int)
                + epsilon
            )
        else:
            log_ph_integer_given_z0 = torch.zeros(0, device=device)

        # center `h_cat` around 1, since it is one-hot encoded
        centered_h_cat = estimated_h_cat - 1

        # compute integrals from 0.5 to 1.5 of the normal distribution
        # i.e., N(mean = `centered_h_cat`, stdev = `sigma_0_cat`)
        log_ph_cat_proportional = torch.log(
            self.cdf_standard_gaussian((centered_h_cat + 0.5) / sigma_0_cat)
            - self.cdf_standard_gaussian((centered_h_cat - 0.5) / sigma_0_cat)
            + epsilon
        )

        # normalize the distribution over the categories
        log_Z = torch.logsumexp(log_ph_cat_proportional, dim=-1, keepdim=True)
        log_probabilities = log_ph_cat_proportional - log_Z

        # integrate log probability of the current integer property
        log_ph_integer_given_z0 = self.sum_node_features_except_batch(
            log_ph_integer_given_z0 * node_mask.float().unsqueeze(-1),
            batch_index
        )

        # select log probability of the current category using one-hot representation
        log_ph_cat_given_z0 = self.sum_node_features_except_batch(
            log_probabilities * onehot * node_mask.float().unsqueeze(-1),
            batch_index
        )

        # combine categorical and integer log probabilities
        log_ph_given_z0 = log_ph_integer_given_z0 + log_ph_cat_given_z0

        return log_p_x_given_z0_without_constants, log_ph_given_z0

    @typechecked
    def normalize(
        self,
        x: TensorType["batch_num_nodes", 3],
        h: Union[Dict[str, H_INPUT_TYPE], H_INPUT_TYPE],
        node_mask: TensorType["batch_num_nodes"],
        generate_x_only: bool = False
    ) -> Tuple[
        TensorType["batch_num_nodes", 3],
        Union[Dict[str, H_INPUT_TYPE], H_INPUT_TYPE]
    ]:
        x = x / self.diffusion_cfg.norm_values[0]

        # bypass individual normalizations for components of `h`
        if generate_x_only:
            h = (h.float() - self.diffusion_cfg.norm_biases[1]) / self.diffusion_cfg.norm_values[1]
            return x, h

        # cast to float in case `h` still has `long` or `int` type
        h_cat = (h["categorical"].float() - self.diffusion_cfg.norm_biases[1]) / self.diffusion_cfg.norm_values[1]
        h_cat = h_cat * node_mask.float().unsqueeze(-1)
        h_int = (
            (h["integer"].float() - self.diffusion_cfg.norm_biases[2]) / self.diffusion_cfg.norm_values[2]
        )

        if self.include_charges:
            h_int = h_int * node_mask.float()

        # create new `h` dictionary
        h = {"categorical": h_cat, "integer": h_int}

        return x, h

    @typechecked
    def unnormalize(
        self,
        x: TensorType["batch_num_nodes", 3],
        node_mask: TensorType["batch_num_nodes"],
        h_cat: Optional[TensorType["batch_num_nodes", "num_node_categories"]] = None,
        h_int: Optional[H_INPUT_TYPE] = None,
        generate_x_only: bool = False
    ) -> Tuple[
        TensorType["batch_num_nodes", 3],
        Optional[TensorType["batch_num_nodes", "num_node_categories"]],
        Optional[H_INPUT_TYPE]
    ]:
        x = x * self.diffusion_cfg.norm_values[0]

        if generate_x_only:
            return x, None, None

        h_cat = h_cat * self.diffusion_cfg.norm_values[1] + self.diffusion_cfg.norm_biases[1]
        h_cat = h_cat * node_mask.float().unsqueeze(-1)
        h_int = h_int * self.diffusion_cfg.norm_values[2] + self.diffusion_cfg.norm_biases[2]

        if self.include_charges:
            h_int = h_int * node_mask.float().unsqueeze(-1)

        return x, h_cat, h_int

    @typechecked
    def unnormalize_z(
        self,
        z: TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        node_mask: TensorType["batch_num_nodes"],
        generate_x_only: bool = False
    ) -> TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]:
        # parse from `z`
        x = z[:, 0:self.num_x_dims]
        h_cat = (
            None
            if generate_x_only
            else z[:, self.num_x_dims:self.num_x_dims + self.num_atom_types]
        )
        h_int = (
            None
            if generate_x_only
            else z[:, self.num_x_dims + self.num_atom_types:]
        )

        # unnormalize
        if generate_x_only:
            x, _, _ = self.unnormalize(x, node_mask, generate_x_only=True)
            output = x
        else:
            x, h_cat, h_int = self.unnormalize(x, node_mask, h_cat=h_cat, h_int=h_int)
            output = (
                torch.cat([x, h_cat, h_int], dim=-1)
                if self.include_charges
                else torch.cat([x, h_cat], dim=-1)
            )
        return output

    @typechecked
    def sample_combined_position_feature_noise(
        self,
        batch_index: TensorType["batch_num_nodes"],
        node_mask: TensorType["batch_num_nodes"],
        generate_x_only: bool = False
    ) -> TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]:
        """
        Sample mean-centered normal noise for `z_x`, and standard normal noise for `z_h`.
        """
        z_x = self.sample_center_gravity_zero_gaussian_with_mask(
            size=(len(batch_index), self.num_x_dims),
            batch_index=batch_index,
            node_mask=node_mask,
            device=batch_index.device
        )
        if generate_x_only:
            # bypass calculations for `h`
            return z_x
        z_h = self.sample_gaussian_with_mask(
            size=(len(batch_index), self.num_node_scalar_features),
            node_mask=node_mask,
            device=batch_index.device
        )
        z = torch.cat([z_x, z_h], dim=-1)
        return z

    @typechecked
    def sample_normal(
        self,
        mu: TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        sigma: TensorType["batch_size", 1],
        batch_index: TensorType["batch_num_nodes"],
        node_mask: TensorType["batch_num_nodes"],
        fix_noise: bool = False,
        generate_x_only: bool = False
    ) -> TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]:
        """Sample from a Normal distribution."""
        if fix_noise:
            batch_index_ = torch.zeros_like(batch_index)  # broadcast same noise across batch
            eps = self.sample_combined_position_feature_noise(batch_index_, node_mask, generate_x_only=generate_x_only)
        else:
            eps = self.sample_combined_position_feature_noise(batch_index, node_mask, generate_x_only=generate_x_only)
        return mu + sigma[batch_index] * eps

    @typechecked
    def sample_p_xh_given_z0(
        self,
        z_0: TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        batch_index: TensorType["batch_num_nodes"],
        node_mask: TensorType["batch_num_nodes"],
        batch_size: int,
        batch: Optional[Batch] = None,
        context: Optional[TensorType["batch_num_nodes", "num_context_features"]] = None,
        fix_noise: bool = False,
        generate_x_only: bool = False,
        xh_self_cond: Optional[TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]] = None
    ) -> Union[
        Tuple[
            TensorType["batch_num_nodes", 3],
            Dict[
                str,
                torch.Tensor  # note: for when `include_charges=False`
            ]
        ],
        TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]
    ]:
        """Sample `x` ~ p(x | z0)."""
        t_zeros = torch.zeros(size=(batch_size, 1), device=batch_index.device)
        gamma_0 = self.gamma(t_zeros)

        # compute sqrt(`sigma_0` ^ 2 / `alpha_0` ^ 2)
        sigma_x = self.SNR(-0.5 * gamma_0)

        # construct batch input object (e.g., when using a molecule generation DDPM)
        if batch is None:
            batch = Batch(batch=batch_index, mask=node_mask, props_context=context)

        # make network prediction
        _, net_out = self.dynamics_network(
            batch,
            z_0,
            t_zeros[batch_index],
            x_self_cond=xh_self_cond,
            xh_self_cond=xh_self_cond
        )

        # compute `mu` for p(zs | zt)
        mu_x = self.compute_x_pred(z_0, net_out, gamma_0, batch_index)
        xh = self.sample_normal(
            mu=mu_x,
            sigma=sigma_x,
            batch_index=batch_index,
            node_mask=node_mask,
            fix_noise=fix_noise,
            generate_x_only=generate_x_only
        )

        x = xh[:, :self.num_x_dims]

        # bypass scalar predictions for nodes
        if generate_x_only:
            x, _, _ = self.unnormalize(x, node_mask, generate_x_only=generate_x_only)
            return x, {}

        h_cat = xh[:, self.num_x_dims:-1] if self.include_charges else xh[:, self.num_x_dims:]
        h_int = xh[:, -1:] if self.include_charges else torch.zeros(0, device=x.device)
        x, h_cat, h_int = self.unnormalize(x, node_mask, h_cat=h_cat, h_int=h_int)

        h_cat = F.one_hot(torch.argmax(h_cat, dim=-1), self.num_atom_types) * node_mask.long().unsqueeze(-1)
        h_int = torch.round(h_int).long() * node_mask.long().unsqueeze(-1)
        h = {"integer": h_int, "categorical": h_cat}

        return x, h

    @typechecked
    def compute_noised_representation(
        self,
        xh: TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        batch_index: TensorType["batch_num_nodes"],
        node_mask: TensorType["batch_num_nodes"],
        gamma_t: TensorType["batch_size", 1],
        generate_x_only: bool = False
    ) -> Tuple[
        TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]
    ]:
        # compute `alpha_t` and `sigma_t` from gamma
        alpha_t = self.alpha(gamma_t, xh)
        sigma_t = self.sigma(gamma_t, xh)

        # sample `zt` ~ Normal(`alpha_t` `x`, `sigma_t`)
        eps = self.sample_combined_position_feature_noise(batch_index, node_mask, generate_x_only=generate_x_only)

        # sample `z_t` given `x`, `h` for timestep `t`, from q(`z_t` | `x`, `h`)
        z_t = alpha_t[batch_index] * xh + sigma_t[batch_index] * eps

        return z_t, eps

    @typechecked
    def log_pN(self, num_nodes: TensorType["batch_size"]) -> TensorType["batch_size"]:
        """
        Prior on the sample size for computing log p(x, h, N) = log(x, h | N) + log p(N),
        where log p(x, h | N) is a model's output.
        """
        log_pN = self.num_nodes_distribution.log_prob(num_nodes)
        return log_pN

    @typechecked
    def delta_log_px(self, num_nodes: TensorType["batch_size"]) -> TensorType["batch_size"]:
        d = self.subspace_dimensionality(num_nodes)
        return -d * np.log(self.diffusion_cfg.norm_values[0])

    @typechecked
    def forward(self, batch: Batch, return_loss_info: bool = False) -> Tuple[Any, ...]:
        """
        Compute the loss and NLL terms.
        """
        return self.forward_fn(batch, return_loss_info=return_loss_info)

    @typechecked
    def atom_types_and_coords_forward(
        self,
        batch: Batch,
        return_loss_info: bool = False,
        self_conditioning_prob: float = 0.5,
        fix_self_conditioning_noise: bool = False,
        save_dynamics_network_graphviz_plot: bool = False
    ) -> Tuple[Any, ...]:
        """
        Compute the loss and NLL terms for molecule generation (i.e., atom types and coordinates) in 3D.
        """
        # normalize data and take into account volume change in `x`
        x_init, h_init = self.normalize(batch.x, batch.h, node_mask=batch.mask)
        batch.x, batch.h = x_init, h_init

        # retrieve batch properties for simple reference later
        batch_index, batch_size, num_nodes, node_mask = (
            batch.batch, batch.num_graphs, batch.num_nodes_present, batch.mask
        )

        # account for likelihood change due to normalization
        delta_log_px = self.delta_log_px(num_nodes)

        # reset `delta_log_px` if VLB is not the objective.
        if self.training and self.diffusion_cfg.loss_type == "l2":
            delta_log_px = torch.zeros_like(delta_log_px)

        # sample a timestep `t` for each example in batch;
        # at evaluation time, `loss_0` will be computed separately to decrease
        # variance in the estimator (note: this costs two forward passes).
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t,
            self.T + 1,
            size=(batch.num_graphs, 1),
            device=batch.x.device
        )
        s_int = t_int - 1  # previous timestep

        # note: these are important for computing log p(x | z0)
        t_is_zero = (t_int == 0).float()
        t_is_not_zero = 1 - t_is_zero

        # normalize `t` to [0, 1]
        # note: the negative step of `s` will never be used,
        # since then p(x | z0) is computed
        s = s_int / self.T
        t = t_int / self.T

        # compute `gamma_s` and `gamma_t` via the network
        gamma_s = inflate_batch_array(self.gamma(s), batch.x)
        gamma_t = inflate_batch_array(self.gamma(t), batch.x)

        # concatenate `x`, `h`[integer] and `h`[categorical]
        if self.include_charges:
            xh = torch.cat([batch.x, batch.h["categorical"], batch.h["integer"].reshape(-1, 1)], dim=-1)
        else:
            xh = torch.cat([batch.x, batch.h["categorical"]], dim=-1)

        # derive noised representations of nodes
        z_t, eps_t = self.compute_noised_representation(xh, batch_index, batch.mask, gamma_t)

        # self-condition the model's prediction
        self_cond = None
        self_conditioning = (
            self.training and self.diffusion_cfg.self_condition and not (t_int == self.T).any() and random() < self_conditioning_prob
        )
        if self_conditioning:
            with torch.no_grad():
                s_array_self_cond = torch.full((batch_size, 1), fill_value=0, device=t_int.device) / self.T
                t_array_self_cond = (t_int + 1) / self.T
                gamma_t_self_cond = inflate_batch_array(self.gamma(t_array_self_cond), batch.x)
                z_t_self_cond, _ = self.compute_noised_representation(xh, batch_index, batch.mask, gamma_t_self_cond)

                self_cond = self.sample_p_zs_given_zt(
                    s=s_array_self_cond,
                    t=t_array_self_cond,
                    z=z_t_self_cond,
                    batch_index=batch_index,
                    node_mask=node_mask,
                    context=getattr(batch, "props_context", None),
                    fix_noise=fix_self_conditioning_noise,
                    generate_x_only=False,
                    self_condition=True
                ).detach_()

        # make neural network prediction
        _, net_out = self.dynamics_network(batch, z_t, t[batch_index], xh_self_cond=self_cond)

        # plot Graphviz representation of the current computational graph
        if save_dynamics_network_graphviz_plot:
            make_and_save_network_graphviz_plot(
                output_var=net_out,
                params=dict(self.dynamics_network.named_parameters())
            )

        # compute the L2 error
        error_t = self.sum_node_features_except_batch((eps_t - net_out) ** 2, batch_index)

        if self.training and self.diffusion_cfg.loss_type == "l2":
            SNR_weight = torch.ones_like(error_t)
        else:
            # compute weighting with SNR: (SNR(s - t) - 1) for epsilon parametrization
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(-1)
        assert error_t.shape == SNR_weight.shape

        # note: `_constants_` depend on `sigma_0` from
        # cross entropy term `E_q(z0 | x) [log p(x | z0)]`
        neg_log_constants = -self.log_constants_p_x_given_z0(
            num_nodes=num_nodes,
            device=batch.x.device
        )

        # reset constants when training with L2 loss
        if self.training and self.diffusion_cfg.loss_type == "l2":
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # the KL divergence between q(zT | x) and p(zT) = Normal(0, 1);
        # note: should be close to zero - if not, inspect `norm_values`
        kl_prior = self.compute_kl_prior(
            xh,
            batch_index=batch_index,
            node_mask=node_mask,
            num_nodes=num_nodes,
            device=batch.x.device
        )

        if self.training:
            # compute the `L_0` term (even if `gamma_t` is not actually `gamma_0`),
            # as this will later be selected via masking
            log_p_x_given_z0_without_constants, log_ph_given_z0 = (
                self.log_pxh_given_z0_without_constants(
                    h=h_init,
                    z_0=z_t,
                    eps=eps_t,
                    net_out=net_out,
                    gamma_0=gamma_t,
                    batch_index=batch_index,
                    node_mask=node_mask,
                    device=batch.x.device
                )
            )

            loss_0_x = (
                -log_p_x_given_z0_without_constants * t_is_zero.squeeze()
            )
            loss_0_h = (
                -log_ph_given_z0 * t_is_zero.squeeze()
            )

            # apply `t_is_zero` mask
            error_t = error_t * t_is_not_zero.squeeze()

        else:
            # compute noise values for `t = 0`
            t_zeros = torch.zeros_like(s)
            gamma_0 = inflate_batch_array(self.gamma(t_zeros), batch.x)

            # sample `z_0` given `x`, `h` for timestep `t`, from q(`z_t` | `x`, `h`)
            z_0, eps_0 = self.compute_noised_representation(xh, batch_index, batch.mask, gamma_0)

            _, net_out_0 = self.dynamics_network(batch, z_0, t_zeros[batch_index])

            log_p_x_given_z0_without_constants, log_ph_given_z0 = (
                self.log_pxh_given_z0_without_constants(
                    h=h_init,
                    z_0=z_0,
                    eps=eps_0,
                    net_out=net_out_0,
                    gamma_0=gamma_0,
                    batch_index=batch_index,
                    node_mask=node_mask,
                    device=batch.x.device
                )
            )
            loss_0_x = -log_p_x_given_z0_without_constants
            loss_0_h = -log_ph_given_z0

        # sample node counts prior
        log_pN = self.log_pN(num_nodes)

        # assemble loss terms
        loss_terms = (
            delta_log_px, error_t, SNR_weight,
            loss_0_x, loss_0_h, neg_log_constants,
            kl_prior, log_pN, t_int.squeeze()
        )

        if return_loss_info:
            loss_info = {
                "eps_hat_x": scatter(
                    net_out[:, :self.num_x_dims].abs().mean(-1),
                    batch_index,
                    dim=0,
                    reduce="mean"
                ).mean(),
                "eps_hat_h": scatter(
                    net_out[:, self.num_x_dims:].abs().mean(-1),
                    batch_index,
                    dim=0,
                    reduce="mean"
                ).mean()
            }
            return (*loss_terms, loss_info)

        return loss_terms

    @typechecked
    def sample_p_zt_given_zs(
        self,
        zs: TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        batch_index: TensorType["batch_num_nodes"],
        node_mask: TensorType["batch_num_nodes"],
        gamma_t: TensorType["batch_size", 1],
        gamma_s: TensorType["batch_size", 1],
        fix_noise: bool = False,
        generate_x_only: bool = False
    ) -> TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]:
        _, sigma_t_given_s, alpha_t_given_s = (
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zs)
        )

        mu = alpha_t_given_s[node_mask] * zs
        zt = self.sample_normal(
            mu=mu,
            sigma=sigma_t_given_s,
            batch_index=batch_index,
            node_mask=node_mask,
            fix_noise=fix_noise,
            generate_x_only=generate_x_only
        )

        # remove center of mass
        _, zt_x = centralize(
            Batch(x=zt[:, :self.num_x_dims]),
            key="x",
            batch_index=batch_index,
            node_mask=node_mask,
            edm=True
        )
        zt = (
            zt_x
            if generate_x_only
            else torch.cat((zt_x, zt[:, self.num_x_dims:]), dim=-1)
        )

        return zt

    @typechecked
    def sample_p_zs_given_zt(
        self,
        s: TensorType["batch_size", 1],
        t: TensorType["batch_size", 1],
        z: TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        batch_index: TensorType["batch_num_nodes"],
        node_mask: TensorType["batch_num_nodes"],
        batch: Optional[Batch] = None,
        context: Optional[TensorType["batch_num_nodes", "num_context_features"]] = None,
        fix_noise: bool = False,
        generate_x_only: bool = False,
        self_condition: bool = False,
        xh_self_cond: Optional[TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]] = None
    ) -> TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]:
        """Sample from `zs` ~ p(`zs` | `zt`). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = (
            self.sigma_and_alpha_t_given_s(
                gamma_t, gamma_s, z
            )
        )

        sigma_s = self.sigma(gamma_s, target_tensor=z)
        sigma_t = self.sigma(gamma_t, target_tensor=z)

        # organize batch inputs (e.g., for molecule generation DDPMs)
        if batch is None:
            batch = Batch(batch=batch_index, mask=node_mask, props_context=context)

        # make network prediction
        _, eps_t = self.dynamics_network(
            batch,
            z,
            t[batch_index],
            x_self_cond=xh_self_cond,
            xh_self_cond=xh_self_cond
        )

        # compute `mu` for p(zs | zt)
        None if self_condition else self.assert_mean_zero_with_mask(z[:, :self.num_x_dims], node_mask)
        None if self_condition else self.assert_mean_zero_with_mask(eps_t[:, :self.num_x_dims], node_mask)
        mu = (
            z / alpha_t_given_s[batch_index] -
             (sigma2_t_given_s[batch_index] / alpha_t_given_s[batch_index] / sigma_t[batch_index]) * eps_t
        )

        # compute `sigma` for p(zs | zt)
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # sample `zs` given the parameters derived from `zt`
        zs = self.sample_normal(
            mu,
            sigma,
            batch_index,
            node_mask,
            fix_noise=fix_noise,
            generate_x_only=generate_x_only
        )

        # project node positions down to avoid numerical runaway of the center of gravity
        _, zs_x = centralize(
            Batch(x=zs[:, :self.num_x_dims]),
            key="x",
            batch_index=batch_index,
            node_mask=node_mask,
            edm=True
        )
        zs = (
            zs_x
            if generate_x_only
            else torch.cat([zs_x, zs[:, self.num_x_dims:]], dim=-1)
        )
        return zs

    @torch.inference_mode()
    @typechecked
    def mol_gen_sample(
        self,
        num_samples: int,
        num_nodes: TensorType["batch_size"],
        device: Union[torch.device, str],
        return_frames: int = 1,
        num_timesteps: Optional[int] = None,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None,
        context: Optional[TensorType["batch_size", "num_context_features"]] = None,
        fix_noise: bool = False,
        generate_x_only: bool = False,
        fix_self_conditioning_noise: bool = False,
        norm_with_original_timesteps: bool = False,
    ) -> Tuple[
        Union[
            TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
            TensorType["num_timesteps", "batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]
        ],
        TensorType["batch_num_nodes"],
        TensorType["batch_num_nodes"]
    ]:
        """
        Draw samples from the generative model.
        Optionally, return intermediate states for
        visualization purposes.
        """
        num_timesteps = self.T if num_timesteps is None else num_timesteps
        assert 0 < return_frames <= num_timesteps, "Number of frames cannot be greater than number of timesteps."
        assert num_timesteps % return_frames == 0, "Number of frames must be evenly divisible by number of timesteps."

        # derive batch metadata
        batch_index = num_nodes_to_batch_index(num_samples, num_nodes, device=device)
        # note: by default, no nodes are masked
        node_mask = torch.ones_like(batch_index).bool() if node_mask is None else node_mask

        # expand and mask context
        if context is not None:
            context = context[batch_index]
            context = context * node_mask.float().unsqueeze(-1)

        # sample from the noise distribution (i.e., p(z_T))
        if fix_noise:
            batch_index_ = torch.zeros_like(batch_index)  # broadcast same noise across batch
            z = self.sample_combined_position_feature_noise(batch_index_, node_mask, generate_x_only=generate_x_only)
        else:
            z = self.sample_combined_position_feature_noise(batch_index, node_mask, generate_x_only=generate_x_only)

        self.assert_mean_zero_with_mask(z[:, :self.num_x_dims], node_mask)

        # iteratively sample p(z_s | z_t) for `t = 1, ..., T`, with `s = t - 1`.
        self_cond = None
        s_array_self_cond = torch.full((num_samples, 1), fill_value=0, device=device) / (self.T if norm_with_original_timesteps else num_timesteps)
        out = torch.zeros((return_frames,) + z.size(), device=device)
        for s in reversed(range(0, num_timesteps)):
            s_array = torch.full((num_samples, 1), fill_value=s, device=device)
            t_array = s_array + 1
            s_array = s_array / (self.T if norm_with_original_timesteps else num_timesteps)
            t_array = t_array / (self.T if norm_with_original_timesteps else num_timesteps)

            z = self.sample_p_zs_given_zt(
                s=s_array,
                t=t_array,
                z=z,
                batch_index=batch_index,
                node_mask=node_mask,
                context=context,
                fix_noise=fix_noise,
                generate_x_only=generate_x_only,
                xh_self_cond=self_cond
            )

            # save frame
            if (s * return_frames) % num_timesteps == 0:
                idx = (s * return_frames) // num_timesteps
                out[idx] = self.unnormalize_z(
                    z=z,
                    node_mask=node_mask,
                    generate_x_only=generate_x_only
                )

            # self-condition
            if self.diffusion_cfg.self_condition:
                t_array_self_cond = s_array
                self_cond = self.sample_p_zs_given_zt(
                    s=s_array_self_cond,
                    t=t_array_self_cond,
                    z=z,
                    batch_index=batch_index,
                    node_mask=node_mask,
                    context=context,
                    fix_noise=fix_self_conditioning_noise,
                    generate_x_only=False,
                    self_condition=True
                ).detach_()

        # lastly, sample p(x, h | z_0)
        x, h = self.sample_p_xh_given_z0(
            z_0=z,
            batch_index=batch_index,
            node_mask=node_mask,
            batch_size=num_samples,
            context=context,
            fix_noise=fix_self_conditioning_noise if self.diffusion_cfg.self_condition else fix_noise,
            generate_x_only=generate_x_only,
            xh_self_cond=self_cond
        )

        self.assert_mean_zero_with_mask(x, node_mask)

        # correct CoG drift for examples without intermediate states
        if return_frames == 1:
            max_cog = scatter(x, batch_index, dim=0, reduce="sum").abs().max().item()
            if max_cog > 5e-2:
                log.warning(f"CoG drift with error {max_cog:.3f}. Projecting the positions down.")
                _, x = centralize(
                    Batch(x=x),
                    key="x",
                    batch_index=batch_index,
                    node_mask=node_mask,
                    edm=True
                )

        # overwrite last frame with the resulting `x` and `h`
        if generate_x_only:
            out[0] = x
        elif self.include_charges:
            out[0] = torch.cat([x, h["categorical"], h["integer"]], dim=-1)
        else:
            out[0] = torch.cat([x, h["categorical"]], dim=-1)

        return out.squeeze(0), batch_index, node_mask
    
    @torch.inference_mode()
    @typechecked
    def mol_gen_optimize(
        self,
        samples: List[Tuple[torch.Tensor, torch.Tensor]],
        num_nodes: TensorType["batch_size"],
        device: Union[torch.device, str],
        return_frames: int = 1,
        num_timesteps: Optional[int] = None,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None,
        context: Optional[TensorType["batch_size", "num_context_features"]] = None,
        generate_x_only: bool = False,
        norm_with_original_timesteps: bool = False,
    ) -> Tuple[
        Union[
            TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
            TensorType["num_timesteps", "batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]
        ],
        TensorType["batch_num_nodes"],
        TensorType["batch_num_nodes"]
    ]:
        """
        Optimize existing samples using the generative model.
        Optionally, return intermediate states for
        visualization purposes.
        """
        num_timesteps = self.T if num_timesteps is None else num_timesteps
        assert 0 < return_frames <= num_timesteps, "Number of frames cannot be greater than number of timesteps."
        assert num_timesteps % return_frames == 0, "Number of frames must be evenly divisible by number of timesteps."

        # derive batch metadata
        num_samples = len(samples)
        batch_index = num_nodes_to_batch_index(num_samples, num_nodes, device=device)
        # note: by default, no nodes are masked
        node_mask = torch.ones_like(batch_index).bool() if node_mask is None else node_mask

        # expand and mask context
        if context is not None:
            context = context[batch_index]
            context = context * node_mask.float().unsqueeze(-1)

        # combine samples into a single (normalized) feature tensor
        x_init = torch.vstack([sample[0] for sample in samples]).to(device)
        h_init = torch.vstack([sample[1] for sample in samples]).to(device)
        x_init, h_init = self.normalize(
            x=x_init,
            h={"categorical": h_init, "integer": torch.tensor([])},
            node_mask=node_mask,
            generate_x_only=generate_x_only
        )
        z = torch.hstack((x_init, h_init["categorical"]))

        self.assert_mean_zero_with_mask(z[:, :self.num_x_dims], node_mask)

        # iteratively sample p(z_s | z_t) for `t = 1, ..., T`, with `s = t - 1`.
        self_cond = None
        s_array_self_cond = torch.full((num_samples, 1), fill_value=0, device=device) / (self.T if norm_with_original_timesteps else num_timesteps)
        out = torch.zeros((return_frames,) + z.size(), device=device)
        for s in reversed(range(0, num_timesteps)):
            s_array = torch.full((num_samples, 1), fill_value=s, device=device)
            t_array = s_array + 1

            # normalize current timesteps according to the model's total number of training timesteps
            s_array = s_array / (self.T if norm_with_original_timesteps else num_timesteps)
            t_array = t_array / (self.T if norm_with_original_timesteps else num_timesteps)

            z = self.sample_p_zs_given_zt(
                s=s_array,
                t=t_array,
                z=z,
                batch_index=batch_index,
                node_mask=node_mask,
                context=context,
                generate_x_only=generate_x_only,
                xh_self_cond=self_cond
            )

            # save frame
            if (s * return_frames) % num_timesteps == 0:
                idx = (s * return_frames) // num_timesteps
                out[idx] = self.unnormalize_z(
                    z=z,
                    node_mask=node_mask,
                    generate_x_only=generate_x_only
                )

            # self-condition
            if self.diffusion_cfg.self_condition:
                t_array_self_cond = s_array
                self_cond = self.sample_p_zs_given_zt(
                    s=s_array_self_cond,
                    t=t_array_self_cond,
                    z=z,
                    batch_index=batch_index,
                    node_mask=node_mask,
                    context=context,
                    generate_x_only=False,
                    self_condition=True
                ).detach_()

        # lastly, sample p(x, h | z_0)
        x, h = self.sample_p_xh_given_z0(
            z_0=z,
            batch_index=batch_index,
            node_mask=node_mask,
            batch_size=num_samples,
            context=context,
            generate_x_only=generate_x_only,
            xh_self_cond=self_cond
        )

        self.assert_mean_zero_with_mask(x, node_mask)

        # correct CoG drift for examples without intermediate states
        if return_frames == 1:
            max_cog = scatter(x, batch_index, dim=0, reduce="sum").abs().max().item()
            if max_cog > 5e-2:
                log.warning(f"CoG drift with error {max_cog:.3f}. Projecting the positions down.")
                _, x = centralize(
                    Batch(x=x),
                    key="x",
                    batch_index=batch_index,
                    node_mask=node_mask,
                    edm=True
                )

        # overwrite last frame with the resulting `x` and `h`
        if generate_x_only:
            out[0] = x
        else:
            out[0] = torch.cat([x, h["categorical"]], dim=-1)

        return out.squeeze(0), batch_index, node_mask

    @typechecked
    def get_repaint_schedule(
        self,
        resamplings: int,
        jump_length: int,
        num_timesteps: int
    ) -> List[int]:
        """
        Note: Each integer in the schedule list describes how many
        denoising steps need to be applied before jumping back.
        """
        curr_t = 0
        repaint_schedule = []

        while curr_t < num_timesteps:
            if curr_t + jump_length < num_timesteps:
                if len(repaint_schedule) > 0:
                    repaint_schedule[-1] += jump_length
                    repaint_schedule.extend([jump_length] * (resamplings - 1))
                else:
                    repaint_schedule.extend([jump_length] * resamplings)
                curr_t += jump_length
            else:
                residual = (num_timesteps - curr_t)
                if len(repaint_schedule) > 0:
                    repaint_schedule[-1] += residual
                else:
                    repaint_schedule.append(residual)
                curr_t += residual

        return list(reversed(repaint_schedule))

    @torch.inference_mode()
    @typechecked
    def inpaint(
        self,
        molecule: Dict[str, Any],
        node_mask_fixed: TensorType["batch_num_nodes"],
        num_resamplings: int = 1,
        jump_length: int = 1,
        return_frames: int = 1,
        num_timesteps: Optional[int] = None,
        context: Optional[TensorType["batch_size", "num_context_features"]] = None,
        generate_x_only: bool = False
    ) -> Union[
        TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        TensorType["num_timesteps", "batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]
    ]:
        """
        Draw samples from the generative model while fixing parts of the input.
        Optionally, return intermediate states for visualization purposes.
        See:
        Lugmayr, Andreas, et al.
        "Repaint: Inpainting using denoising diffusion probabilistic models."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
        Recognition. 2022.
        """
        num_timesteps = self.T if num_timesteps is None else num_timesteps
        assert 0 < return_frames <= num_timesteps, "Number of frames cannot be greater than number of timesteps."
        assert num_timesteps % return_frames == 0, "Number of frames must be evenly divisible by number of timesteps."
        assert jump_length == 1 or return_frames == 1, "Chain visualization is only implemented for `jump_length=1`"

        # fixed hyperparameters
        num_samples = len(molecule["num_nodes"])

        # expand context
        if context is not None:
            context = context[molecule["batch_index"]]

        if generate_x_only:
            xh0 = molecule["x"]
        elif self.include_charges:
            xh0 = torch.cat([molecule["x"], molecule["one_hot"], molecule["charges"]], dim=-1)
        else:
            xh0 = torch.cat([molecule["x"], molecule["one_hot"]], dim=-1)

        # center initial system by subtracting CoM of known parts
        mean_known = scatter(
            molecule["x"][node_mask_fixed],
            molecule["batch_index"][node_mask_fixed],
            dim=0,
            reduce="mean"
        )
        xh0[:, :self.num_x_dims] = (
            xh0[:, :self.num_x_dims] - mean_known[molecule["batch_index"]]
        )

        # derive noised representation at step `t=T`
        z = self.sample_combined_position_feature_noise(
            batch_index=molecule["batch_index"],
            node_mask=torch.ones_like(node_mask_fixed),
            generate_x_only=generate_x_only
        )

        # craft output tensors
        out = torch.zeros((return_frames,) + z.size(), device=z.device)

        # iteratively sample according to a pre-defined schedule
        s = num_timesteps - 1
        schedule = self.get_repaint_schedule(num_resamplings, jump_length, num_timesteps)

        self_cond = None
        s_array_self_cond = torch.full((num_samples, 1), fill_value=0, device=z.device) / num_denoise_steps
        for i, num_denoise_steps in enumerate(schedule):
            for j in range(num_denoise_steps):
                # denoise one time step: `t` -> `s`
                s_array = torch.full((num_samples, 1), fill_value=s, device=z.device)
                t_array = s_array + 1
                s_array = s_array / num_timesteps
                t_array = t_array / num_timesteps

                # sample known nodes from the input
                gamma_s = inflate_batch_array(self.gamma(s_array), molecule["x"])
                z_known, _ = self.compute_noised_representation(
                    xh=xh0,
                    batch_index=molecule["batch_index"],
                    node_mask=torch.ones_like(node_mask_fixed),
                    gamma_t=gamma_s,
                    generate_x_only=generate_x_only
                )

                # sample inpainted part
                z_unknown = self.sample_p_zs_given_zt(
                    s=s_array,
                    t=t_array,
                    z=z,
                    batch_index=molecule["batch_index"],
                    node_mask=torch.ones_like(node_mask_fixed),
                    context=context,
                    generate_x_only=generate_x_only,
                    xh_self_cond=self_cond
                )

                if self.diffusion_cfg.self_condition:
                    t_array_self_cond = s_array
                    self_cond = self.sample_p_zs_given_zt(
                        s=s_array_self_cond,
                        t=t_array_self_cond,
                        z=z_unknown,
                        batch_index=molecule["batch_index"],
                        node_mask=torch.ones_like(node_mask_fixed),
                        context=context,
                        generate_x_only=generate_x_only,
                        self_condition=True
                    ).detach_()

                # move center of mass of the noised part to the center of mass of the corresponding
                # denoised part before combining them -> the resulting system should be CoM-free
                com_noised = scatter(
                    z_known[:, :self.num_x_dims][node_mask_fixed],
                    molecule["batch_index"][node_mask_fixed],
                    dim=0,
                    reduce="mean"
                )
                com_denoised = scatter(
                    z_unknown[:, :self.num_x_dims][node_mask_fixed],
                    molecule["batch_index"][node_mask_fixed],
                    dim=0,
                    reduce="mean"
                )
                z_known[:, :self.num_x_dims] = (
                    z_known[:, :self.num_x_dims] + (com_denoised - com_noised)[molecule["batch_index"]]
                )

                # combine
                z = z_known * node_mask_fixed.unsqueeze(-1).float() + z_unknown * \
                    (1 - node_mask_fixed.unsqueeze(-1).float())
                self.assert_mean_zero_with_mask(
                    x=z[:, :self.num_x_dims],
                    node_mask=torch.ones_like(node_mask_fixed)
                )

                # save frame at the end of a resample cycle
                if num_denoise_steps > jump_length or i == len(schedule) - 1:
                    if (s * return_frames) % num_timesteps == 0:
                        idx = (s * return_frames) // num_timesteps
                        out[idx] = self.unnormalize_z(
                            z=z,
                            node_mask=torch.ones_like(node_mask_fixed),
                            generate_x_only=generate_x_only
                        )

                # noise combined representation
                if j == num_denoise_steps - 1 and i < len(schedule) - 1:
                    # go back `jump_length` steps
                    t = s + jump_length
                    t_array = torch.full((num_samples, 1), fill_value=t, device=z.device)
                    t_array = t_array / num_timesteps

                    gamma_s = inflate_batch_array(self.gamma(s_array), molecule["x"])
                    gamma_t = inflate_batch_array(self.gamma(t_array), molecule["x"])

                    z = self.sample_p_zt_given_zs(
                        zs=z,
                        batch_index=molecule["batch_index"],
                        node_mask=torch.ones_like(node_mask_fixed),
                        gamma_t=gamma_t,
                        gamma_s=gamma_s,
                        generate_x_only=generate_x_only
                    )

                    s = t

                s -= 1

        # lastly, sample p(`x`, `h` | `z_0`)
        x, h = self.sample_p_xh_given_z0(
            z_0=z,
            batch_index=molecule["batch_index"],
            node_mask=torch.ones_like(node_mask_fixed),
            batch_size=num_samples,
            context=context,
            generate_x_only=generate_x_only,
            xh_self_cond=self_cond
        )
        self.assert_mean_zero_with_mask(
            x=x,
            node_mask=torch.ones_like(node_mask_fixed)
        )

        # correct CoM drift for examples without intermediate states
        if return_frames == 1:
            max_cog = scatter(x, molecule["batch_index"], dim=0, reduce="sum").abs().max().item()
            if max_cog > 5e-2:
                log.warning(f"CoG drift with error {max_cog:.3f}. Projecting the positions down.")
                _, x = centralize(
                    Batch(x=x),
                    key="x",
                    batch_index=molecule["batch_index"],
                    edm=True
                )

        # overwrite last frame with the resulting `x` and `h`
        if generate_x_only:
            out[0] = x
        elif self.include_charges:
            out[0] = torch.cat([x, h["categorical"], h["integer"]], dim=-1)
        else:
            out[0] = torch.cat([x, h["categorical"]], dim=-1)

        # remove frame dimension if only the final molecule is returned
        return out.squeeze(0)
