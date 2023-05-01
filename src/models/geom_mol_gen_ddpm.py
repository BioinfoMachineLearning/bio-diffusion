# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import math
import os
import torch
import torchmetrics

import torch.nn.functional as F

from time import time
from pathlib import Path
from rdkit import Chem
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch_geometric.data import Batch
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from omegaconf import DictConfig
from torch_scatter import scatter

import src.datamodules.components.edm.utils as qm9utils

from src.models.components import centralize, num_nodes_to_batch_index, save_xyz_file, visualize_mol, visualize_mol_chain
from src.datamodules.components.edm.rdkit_functions import BasicMolecularMetrics, build_molecule, process_molecule
from src.datamodules.components.edm.datasets_config import GEOM_NO_H, GEOM_WITH_H
from src.datamodules.components.edm import check_molecular_stability, get_bond_length_arrays
from src.models.components.egnn import EGNNDynamics
from src.models.components.variational_diffusion import EquivariantVariationalDiffusion

from src.models.components.gcpnet import GCPNetDynamics
from src.models import HALT_FILE_EXTENSION, CategoricalDistribution, PropertiesDistribution, Queue, batch_tensor_to_list, compute_mean_mad, get_grad_norm, log_grad_flow_lite, reverse_tensor

from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard

from src.utils.pylogger import get_pylogger

patch_typeguard()  # use before @typechecked


log = get_pylogger(__name__)


class GEOMMoleculeGenerationDDPM(LightningModule):
    """LightningModule for GEOM-Drugs small molecule generation using a DDPM.

    This LightningModule organizes the PyTorch code into 9 sections:
        - Computations (init)
        - Forward (forward)
        - Step (step)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers and LR schedulers (configure_optimizers)
        - Gradient clipping (configure_gradient_clipping)
        - End of model training (on_fit_end)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        model_cfg: DictConfig,
        module_cfg: DictConfig,
        layer_cfg: DictConfig,
        diffusion_cfg: DictConfig,
        dataloader_cfg: DictConfig,
        path_cfg: DictConfig = None,
        **kwargs
    ):
        super().__init__()

        # hyperparameters #

        # prior to saving hyperparameters, adjust number of evaluation samples based on one's conditioning argument(s)
        diffusion_cfg.num_eval_samples = (
            diffusion_cfg.num_eval_samples // 2 if len(module_cfg.conditioning) > 0 else diffusion_cfg.num_eval_samples
        )

        # also prior to saving hyperparameters, adjust the effective number of atom types used
        dataloader_cfg.num_atom_types = (
            dataloader_cfg.num_atom_types - 1
            if dataloader_cfg.remove_h
            else dataloader_cfg.num_atom_types
        )

        # this line allows to access init params with `self.hparams` attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # DDPM
        ddpm_modes = {
            "unconditional": EquivariantVariationalDiffusion,
            "inpainting": EquivariantVariationalDiffusion
        }
        self.ddpm_mode = diffusion_cfg.ddpm_mode
        assert self.ddpm_mode in ddpm_modes, f"Selected DDPM mode {self.ddpm_mode} is currently not supported."

        dynamics_networks = {
            "gcpnet": GCPNetDynamics,
            "egnn": EGNNDynamics
        }
        assert diffusion_cfg.dynamics_network in dynamics_networks, f"Selected dynamics network {diffusion_cfg.dynamics_network} is currently not supported."

        self.T = diffusion_cfg.num_timesteps
        self.loss_type = diffusion_cfg.loss_type
        self.num_atom_types = dataloader_cfg.num_atom_types
        self.num_x_dims = dataloader_cfg.num_x_dims
        self.include_charges = dataloader_cfg.include_charges
        self.condition_on_context = len(module_cfg.conditioning) > 0

        # dataset metadata
        dataset_info_mapping = {"GEOM": GEOM_NO_H if dataloader_cfg.remove_h else GEOM_WITH_H}
        self.dataset_info = dataset_info_mapping[dataloader_cfg.dataset]

        # PyTorch modules #
        dynamics_network = dynamics_networks[diffusion_cfg.dynamics_network](
            model_cfg=model_cfg,
            module_cfg=module_cfg,
            layer_cfg=layer_cfg,
            diffusion_cfg=diffusion_cfg,
            dataloader_cfg=dataloader_cfg
        )

        self.ddpm = ddpm_modes[self.ddpm_mode](
            dynamics_network=dynamics_network,
            diffusion_cfg=diffusion_cfg,
            dataloader_cfg=dataloader_cfg,
            dataset_info=self.dataset_info
        )

        # distributions #
        self.node_type_distribution = CategoricalDistribution(
            self.dataset_info["atom_types"],
            self.dataset_info["atom_encoder"]
        )

        # training #
        if self.hparams.module_cfg.clip_gradients:
            self.gradnorm_queue = Queue()
            self.gradnorm_queue.add(3000)  # add large value that will be flushed

        # metrics #
        self.train_phase, self.val_phase, self.test_phase = "train", "val", "test"
        self.phases = [self.train_phase, self.val_phase, self.test_phase]
        self.metrics_to_monitor = [
            "loss", "loss_t", "SNR_weight", "loss_0",
            "kl_prior", "delta_log_px", "neg_log_const_0", "log_pN",
            "eps_hat_x", "eps_hat_h"
        ]
        self.eval_metrics_to_monitor = self.metrics_to_monitor + ["log_SNR_max", "log_SNR_min"]
        for phase in self.phases:
            metrics_to_monitor = (
                self.metrics_to_monitor
                if phase == self.train_phase
                else self.eval_metrics_to_monitor
            )
            for metric in metrics_to_monitor:
                # note: individual metrics e.g., for averaging loss across batches
                setattr(self, f"{phase}_{metric}", torchmetrics.MeanMetric())

        # sample metrics
        if (dataloader_cfg.smiles_filepath is None) or not os.path.exists(dataloader_cfg.smiles_filepath):
            smiles_list = None
        else:
            with open(dataloader_cfg.smiles_filepath, "r") as f:
                smiles_list = f.read().split("\n")
                smiles_list.remove("")  # remove last line (i.e., an empty entry)
        self.molecular_metrics = BasicMolecularMetrics(
            self.dataset_info,
            data_dir=dataloader_cfg.data_dir,
            dataset_smiles_list=smiles_list
        )

    @typechecked
    def forward(
        self,
        batch: Batch,
        dtype: torch.dtype = torch.float32
    ) -> Tuple[
        torch.Tensor,
        Dict[str, Any]
    ]:
        """
        Compute the loss (type L2 or negative log-likelihood (NLL)) if `training`.
        If `eval`, then always compute NLL.
        """
        # centralize node positions to make them translation-invariant
        _, batch.x = centralize(
            batch,
            key="x",
            batch_index=batch.batch,
            node_mask=batch.mask,
            edm=True
        )

        # construct invariant node features
        batch.h = {"categorical": batch.one_hot, "integer": batch.charges}

        # derive property contexts (i.e., conditionals)
        if self.condition_on_context:
            batch.props_context = qm9utils.prepare_context(
                list(self.hparams.module_cfg.conditioning),
                batch,
                self.props_norms
            ).type(dtype)
        else:
            batch.props_context = None

        # derive node counts per batch
        num_nodes = scatter(batch.mask.int(), batch.batch, dim=0, reduce="sum")
        batch.num_nodes_present = num_nodes

        # note: `L` terms in e.g., the GCDM paper represent log-likelihoods,
        # while our loss terms are negative (!) log-likelihoods
        (
            delta_log_px, error_t, SNR_weight,
            loss_0_x, loss_0_h, neg_log_const_0,
            kl_prior, log_pN, t_int, loss_info
        ) = self.ddpm(batch, return_loss_info=True)

        # support L2 loss training step
        if self.training and self.loss_type == "l2":
            # normalize `loss_t`
            effective_num_nodes = (
                num_nodes.max()
                if self.hparams.diffusion_cfg.norm_training_by_max_nodes
                else num_nodes
            )
            denom = (self.num_x_dims + self.ddpm.num_node_scalar_features) * effective_num_nodes
            error_t = error_t / denom
            loss_t = 0.5 * error_t

            # normalize `loss_0` via `loss_0_x` normalization
            loss_0_x = loss_0_x / denom
            loss_0 = loss_0_x + loss_0_h

        # support VLB objective or evaluation step
        else:
            loss_t = self.T * 0.5 * SNR_weight * error_t
            loss_0 = loss_0_x + loss_0_h
            loss_0 = loss_0 + neg_log_const_0

        # combine loss terms
        nll = loss_t + loss_0 + kl_prior

        # correct for normalization on `x`
        nll = nll - delta_log_px

        # transform conditional `nll` into joint `nll`
        # note: loss = -log p(x,h|N) and log p(x,h,N) = log p(x,h|N) + log p(N);
        # therefore, log p(x,h,N) = -loss + log p(N)
        # => loss_new = -log p(x,h,N) = loss - log p(N)
        nll = nll - log_pN

        # collect all metrics' batch-averaged values
        local_variables = locals()
        for metric in self.metrics_to_monitor:
            if metric in ["eps_hat_x", "eps_hat_h"]:
                continue
            if metric != "loss":
                loss_info[metric] = local_variables[metric].mean(0)

        return nll, loss_info

    def step(self, batch: Batch) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # make a forward pass and score it
        nll, loss_info = self.forward(batch)
        return nll, loss_info

    def on_train_start(self, dtype: torch.dtype = torch.float32):
        # note: by default, Lightning executes validation step sanity checks before training starts,
        # so we need to make sure that val_`metric` doesn't store any values from these checks
        for metric in self.eval_metrics_to_monitor:
            # e.g., averaging loss across batches
            torchmetric = getattr(self, f"{self.val_phase}_{metric}")
            torchmetric.reset()

        # ensure valid bond lengths have been added to each dataset's metadata collection (i.e., `self.dataset_info`)
        if any([
            not getattr(self.dataset_info, "bonds1", None),
            not getattr(self.dataset_info, "bonds2", None),
            not getattr(self.dataset_info, "bonds3", None)
        ]):
            bonds = get_bond_length_arrays(self.dataset_info["atom_encoder"])
            self.dataset_info["bonds1"], self.dataset_info["bonds2"], self.dataset_info["bonds3"] = (
                bonds[0], bonds[1], bonds[2]
            )

        # ensure directory for storing sampling outputs is defined
        if not getattr(self, "sampling_output_dir", None):
            if getattr(self, "logger", None) and getattr(self.logger, "experiment", None) and getattr(self.logger.experiment, "dir", None):
                # handle the case where (e.g., on PowerPC systems, with multiple GPUs) the logger's experiment directory is actually a method
                self.sampling_output_dir = (
                    Path(self.trainer.default_root_dir)
                    if not isinstance(self.logger.experiment.dir, str)
                    else Path(self.logger.experiment.dir)
                )
            else:
                self.sampling_output_dir = Path(self.trainer.default_root_dir)

        # if not already loaded, derive distribution information
        # regarding node counts and possibly properties for model conditioning
        if self.condition_on_context and not getattr(self, "props_distr", None):
            splits = ["train", "valid", "test"]
            dataloaders = [
                self.trainer.datamodule.train_dataloader(),
                self.trainer.datamodule.val_dataloader(),
                self.trainer.datamodule.test_dataloader()
            ]
            dataloaders = {split: dataloader for (split, dataloader) in zip(splits, dataloaders)}
            self.props_norms = compute_mean_mad(
                dataloaders, list(self.hparams.module_cfg.conditioning), self.hparams.dataloader_cfg.dataset
            )
            self.props_distr = PropertiesDistribution(
                self.trainer.datamodule.train_dataloader(), self.hparams.module_cfg.conditioning, device=self.device
            )
            self.props_distr.set_normalizer(self.props_norms)

            # derive number of property context features (i.e., conditionals)
            if self.condition_on_context:
                dummy_data = next(iter(dataloaders["train"]))
                dummy_props_context = qm9utils.prepare_context(
                    list(self.hparams.module_cfg.conditioning),
                    dummy_data,
                    self.props_norms
                ).type(dtype)
                self.num_context_node_feats = dummy_props_context.shape[-1]
            else:
                self.num_context_node_feats = None

    def training_step(self, batch: Batch, batch_idx: int) -> Dict[str, Any]:
        try:
            nll, metrics_dict = self.step(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise(e)
            torch.cuda.empty_cache()
            log.info(f"Skipping training batch with index {batch_idx} due to OOM error...")
            return

        # ensure all intermediate losses to be logged as metrics have their gradients ignored
        metrics_dict = {key: value.detach() for key, value in metrics_dict.items()}

        # calculate standard NLL from forward KL-divergence while preserving its gradients
        metrics_dict["loss"] = nll.mean(0)

        # update metrics
        for metric in self.metrics_to_monitor:
            # e.g., averaging loss across batches
            torchmetric = getattr(self, f"{self.train_phase}_{metric}")
            torchmetric(metrics_dict[metric])

        return metrics_dict

    def training_epoch_end(self, outputs: List[Any]):
        # log metrics
        for metric in self.metrics_to_monitor:
            # e.g., logging loss that has been averaged across batches
            torchmetric = getattr(self, f"{self.train_phase}_{metric}")
            self.log(
                f"{self.train_phase}/{metric}",
                torchmetric,
                prog_bar=False
            )

    def on_validation_start(self, dtype: torch.dtype = torch.float32):
        # ensure directory for storing sampling outputs is defined
        if not getattr(self, "sampling_output_dir", None):
            if getattr(self, "logger", None) and getattr(self.logger, "experiment", None) and getattr(self.logger.experiment, "dir", None):
                # handle the case where (e.g., on PowerPC systems, with multiple GPUs) the logger's experiment directory is actually a method
                self.sampling_output_dir = (
                    Path(self.trainer.default_root_dir)
                    if not isinstance(self.logger.experiment.dir, str)
                    else Path(self.logger.experiment.dir)
                )
            else:
                self.sampling_output_dir = Path(self.trainer.default_root_dir)

        # ensure valid bond lengths have been added to each dataset's metadata collection (i.e., `self.dataset_info`)
        if any([
            not getattr(self.dataset_info, "bonds1", None),
            not getattr(self.dataset_info, "bonds2", None),
            not getattr(self.dataset_info, "bonds3", None)
        ]):
            bonds = get_bond_length_arrays(self.dataset_info["atom_encoder"])
            self.dataset_info["bonds1"], self.dataset_info["bonds2"], self.dataset_info["bonds3"] = (
                bonds[0], bonds[1], bonds[2]
            )

        # if not already loaded, derive distribution information
        # regarding node counts and possibly properties for model conditioning
        if self.condition_on_context and not getattr(self, "props_distr", None):
            splits = ["train", "valid", "test"]
            dataloaders = [
                self.trainer.datamodule.train_dataloader(),
                self.trainer.datamodule.val_dataloader(),
                self.trainer.datamodule.test_dataloader()
            ]
            dataloaders = {split: dataloader for (split, dataloader) in zip(splits, dataloaders)}
            self.props_norms = compute_mean_mad(
                dataloaders, list(self.hparams.module_cfg.conditioning), self.hparams.dataloader_cfg.dataset
            )
            self.props_distr = PropertiesDistribution(
                self.trainer.datamodule.train_dataloader(), self.hparams.module_cfg.conditioning, device=self.device
            )
            self.props_distr.set_normalizer(self.props_norms)

            # derive number of property context features (i.e., conditionals)
            if self.condition_on_context:
                dummy_data = next(iter(dataloaders["valid"]))
                dummy_props_context = qm9utils.prepare_context(
                    list(self.hparams.module_cfg.conditioning),
                    dummy_data,
                    self.props_norms
                ).type(dtype)
                self.num_context_node_feats = dummy_props_context.shape[-1]
            else:
                self.num_context_node_feats = None

    def validation_step(self, batch: Batch, batch_idx: int) -> Dict[str, Any]:
        try:
            nll, metrics_dict = self.step(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise(e)
            torch.cuda.empty_cache()
            log.info(f"Skipping validation batch with index {batch_idx} due to OOM error...")
            return

        # ensure all intermediate losses to be logged as metrics have their gradients ignored
        metrics_dict = {key: value.detach() for key, value in metrics_dict.items()}

        # calculate standard NLL from forward KL-divergence while preserving its gradients
        metrics_dict["loss"] = nll.mean(0)

        # collect additional loss information
        gamma_0 = self.ddpm.gamma(torch.zeros((1, 1), device=self.device)).squeeze()
        gamma_1 = self.ddpm.gamma(torch.ones((1, 1), device=self.device)).squeeze()
        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1
        metrics_dict["log_SNR_max"] = log_SNR_max
        metrics_dict["log_SNR_min"] = log_SNR_min

        # update metrics
        for metric in self.eval_metrics_to_monitor:
            # e.g., averaging loss across batches
            torchmetric = getattr(self, f"{self.val_phase}_{metric}")
            torchmetric(metrics_dict[metric])

        return metrics_dict

    @typechecked
    def log_evaluation_metrics(
        self,
        metrics_dict: Dict[str, Any],
        phase: str,
        batch_size: Optional[int] = None,
        sync_dist: bool = False,
        **kwargs
    ):
        for m, value in metrics_dict.items():
            self.log(
                f"{phase}/{m}",
                value,
                batch_size=batch_size,
                sync_dist=sync_dist,
                **kwargs
            )

    @rank_zero_only
    def evaluate_sampling(self):
        suffix_mapping = {
            "unconditional": "",
            "conditional": "_conditional",
            "simple_conditional": "_simple_conditional"
        }
        suffix = suffix_mapping[self.ddpm_mode]

        if (self.current_epoch + 1) % self.hparams.diffusion_cfg.eval_epochs == 0:
            ticker = time()

            sampler = getattr(self, "sample_and_analyze" + suffix)
            sampling_results = sampler(
                num_samples=self.hparams.diffusion_cfg.num_eval_samples,
                batch_size=self.hparams.diffusion_cfg.eval_batch_size
            )
            self.log_evaluation_metrics(sampling_results, phase="val")

            log.info(f"validation_epoch_end(): Sampling evaluation took {time() - ticker:.2f} seconds")

        if (self.current_epoch + 1) % self.hparams.diffusion_cfg.visualize_sample_epochs == 0:
            ticker = time()
            sampler = getattr(self, "sample_and_save" + suffix)
            sampler(num_samples=self.hparams.diffusion_cfg.num_visualization_samples)
            log.info(f"validation_epoch_end(): Sampling visualization took {time() - ticker:.2f} seconds")

        if (self.current_epoch + 1) % self.hparams.diffusion_cfg.visualize_chain_epochs == 0:
            ticker = time()
            sampler = getattr(self, "sample_chain_and_save" + suffix)
            sampler(keep_frames=self.hparams.diffusion_cfg.keep_frames)
            log.info(f"validation_epoch_end(): Chain visualization took {time() - ticker:.2f} seconds")

    def validation_epoch_end(self, outputs: List[Any]):
        # log metrics
        for metric in self.eval_metrics_to_monitor:
            # e.g., logging loss that has been averaged across batches
            torchmetric = getattr(self, f"{self.val_phase}_{metric}")
            self.log(
                f"{self.val_phase}/{metric}",
                torchmetric,
                prog_bar=False
            )

        # make a backup checkpoint before (potentially) sampling from the model
        self.trainer.save_checkpoint(
            Path(self.trainer.checkpoint_callback.dirpath) / f"model_epoch_{self.trainer.current_epoch}_validation_epoch_end.ckpt"
        )

        # perform sampling evaluation on the first device (i.e., rank zero) only
        intervals = [
            self.hparams.diffusion_cfg.eval_epochs,
            self.hparams.diffusion_cfg.visualize_sample_epochs,
            self.hparams.diffusion_cfg.visualize_chain_epochs
        ]
        time_to_evalute_sampling = (
            self.hparams.diffusion_cfg.sample_during_training
            and any([(self.current_epoch + 1) % interval == 0 for interval in intervals])
        )
        if time_to_evalute_sampling:
            self.evaluate_sampling()

    def test_step(self, batch: Batch, batch_idx: int) -> Dict[str, Any]:
        nll, metrics_dict = self.step(batch)

        # ensure all intermediate losses to be logged as metrics have their gradients ignored
        metrics_dict = {key: value.detach() for key, value in metrics_dict.items()}

        # calculate standard NLL from forward KL-divergence while preserving its gradients
        metrics_dict["loss"] = nll.mean(0)

        # collect additional loss information
        gamma_0 = self.ddpm.gamma(torch.zeros((1, 1), device=self.device)).squeeze()
        gamma_1 = self.ddpm.gamma(torch.ones((1, 1), device=self.device)).squeeze()
        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1
        metrics_dict["log_SNR_max"] = log_SNR_max
        metrics_dict["log_SNR_min"] = log_SNR_min

        # update metrics
        for metric in self.eval_metrics_to_monitor:
            # e.g., averaging loss across batches
            torchmetric = getattr(self, f"{self.test_phase}_{metric}")
            torchmetric(metrics_dict[metric])

        return metrics_dict

    def test_epoch_end(self, outputs: List[Any]):
        # log metrics
        for metric in self.eval_metrics_to_monitor:
            # e.g., logging loss that has been averaged across batches
            torchmetric = getattr(self, f"{self.test_phase}_{metric}")
            self.log(
                f"{self.test_phase}/{metric}",
                torchmetric,
                prog_bar=False
            )

    def on_after_backward(self) -> None:
        # periodically log gradient flow
        if (
            self.trainer.is_global_zero and
            (self.global_step + 1) % self.hparams.module_cfg.log_grad_flow_steps == 0
        ):
            if getattr(self, "logger", None) is not None and getattr(self.logger, "experiment", None) is not None:
                experiment = self.logger.experiment
            else:
                experiment = None
            log_grad_flow_lite(self.named_parameters(), wandb_run=experiment)

    @torch.inference_mode()
    @typechecked
    def sample(
        self,
        num_samples: int,
        num_nodes: Optional[TensorType["batch_size"]] = None,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None,
        context: Optional[TensorType["batch_size", "num_context_features"]] = None,
        fix_noise: bool = False,
        num_timesteps: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # node count-conditioning
        if num_nodes is None:
            num_nodes = self.ddpm.num_nodes_distribution.sample(num_samples)
            max_num_nodes = (
                self.dataset_info["max_n_nodes"]
                if "max_n_nodes" in self.dataset_info
                else num_nodes.max().item()
            )
            assert int(num_nodes.max()) <= max_num_nodes

        # context-conditioning
        context = None

        # sampling
        xh, batch_index, _ = self.ddpm.mol_gen_sample(
            num_samples=num_samples,
            num_nodes=num_nodes,
            node_mask=node_mask,
            context=context,
            fix_noise=fix_noise,
            fix_self_conditioning_noise=fix_noise,
            device=self.device,
            num_timesteps=num_timesteps
        )

        x = xh[:, :self.num_x_dims]
        one_hot = xh[:, self.num_x_dims:-1] if self.include_charges else xh[:, self.num_x_dims:]
        charges = xh[:, -1:] if self.include_charges else torch.zeros(0, device=self.device)

        return x, one_hot, charges, batch_index

    @torch.inference_mode()
    @typechecked
    def sample_and_analyze(
        self,
        num_samples: int,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None,
        context: Optional[TensorType["batch_size", "num_context_features"]] = None,
        batch_size: Optional[int] = None,
        max_num_nodes: Optional[int] = 100
    ) -> Dict[str, Any]:
        log.info(f"Analyzing molecule stability at epoch {self.current_epoch}...")

        max_num_nodes = (
            self.dataset_info["max_n_nodes"]
            if "max_n_nodes" in self.dataset_info
            else max_num_nodes
        )

        batch_size = self.hparams.dataloader_cfg.batch_size if batch_size is None else batch_size
        batch_size = min(batch_size, num_samples)

        # note: each item in `molecules` is a tuple of (`position`, `atom_type_encoded`)
        molecules, atom_types, charges = [], [], []
        for _ in range(math.ceil(num_samples / batch_size)):
            # node count-conditioning
            num_samples_batch = min(batch_size, num_samples - len(molecules))
            num_nodes = self.ddpm.num_nodes_distribution.sample(num_samples_batch)

            assert int(num_nodes.max()) <= max_num_nodes

            # context-conditioning
            context = None

            xh, batch_index, _ = self.ddpm.mol_gen_sample(
                num_samples=num_samples_batch,
                num_nodes=num_nodes,
                node_mask=node_mask,
                context=context,
                device=self.device
            )

            x_ = xh[:, :self.num_x_dims].detach().cpu()
            atom_types_ = (
                xh[:, self.num_x_dims:-1].argmax(-1).detach().cpu()
                if self.include_charges
                else xh[:, self.num_x_dims:].argmax(-1).detach().cpu()
            )
            charges_ = (
                xh[:, -1]
                if self.include_charges
                else torch.zeros(0, device=self.device)
            )

            molecules.extend(
                list(
                    zip(
                        batch_tensor_to_list(x_, batch_index),
                        batch_tensor_to_list(atom_types_, batch_index)
                    )
                )
            )

            atom_types.extend(atom_types_.tolist())
            charges.extend(charges_.tolist())

        return self.analyze_samples(molecules, atom_types, charges)

    @typechecked
    def analyze_samples(
        self,
        molecules: List[Tuple[torch.Tensor, ...]],
        atom_types: List[int],
        charges: List[float]
    ) -> Dict[str, Any]:
        # assess distribution of node types
        kl_div_atom = (
            self.node_type_distribution.kl_divergence(atom_types)
            if self.node_type_distribution is not None
            else -1
        )

        # measure molecular stability
        molecule_stable, nr_stable_bonds, num_atoms = 0, 0, 0
        for pos, atom_type in molecules:
            validity_results = check_molecular_stability(
                positions=pos,
                atom_types=atom_type,
                dataset_info=self.dataset_info
            )
            molecule_stable += int(validity_results[0])
            nr_stable_bonds += int(validity_results[1])
            num_atoms += int(validity_results[2])

        fraction_mol_stable = molecule_stable / float(len(molecules))
        fraction_atm_stable = nr_stable_bonds / float(num_atoms)

        # collect other basic molecular metrics
        metrics = self.molecular_metrics.evaluate(molecules)
        validity, uniqueness, novelty = metrics[0], metrics[1], metrics[2]

        return {
            "kl_div_atom_types": kl_div_atom,
            "mol_stable": fraction_mol_stable,
            "atm_stable": fraction_atm_stable,
            "validity": validity,
            "uniqueness": uniqueness,
            "novelty": novelty
        }

    @torch.inference_mode()
    @typechecked
    def sample_and_save(
        self,
        num_samples: int,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None,
        context: Optional[TensorType["batch_size", "num_context_features"]] = None,
        id_from: int = 0,
        name: str = "molecule"
    ):
        # node count-conditioning
        num_nodes = self.ddpm.num_nodes_distribution.sample(num_samples)
        max_num_nodes = (
            self.dataset_info["max_n_nodes"]
            if "max_n_nodes" in self.dataset_info
            else num_nodes.max().item()
        )
        assert int(num_nodes.max()) <= max_num_nodes

        # context-conditioning
        context = None

        # sampling
        xh, batch_index, _ = self.ddpm.mol_gen_sample(
            num_samples=num_samples,
            num_nodes=num_nodes,
            node_mask=node_mask,
            context=context,
            device=self.device
        )

        x = xh[:, :self.num_x_dims]
        one_hot = xh[:, self.num_x_dims:-1] if self.include_charges else xh[:, self.num_x_dims:]
        charges = xh[:, -1:] if self.include_charges else torch.zeros(0, device=self.device)

        output_dir = Path(self.sampling_output_dir, f"epoch_{self.current_epoch}")
        save_xyz_file(
            path=str(output_dir) + "/",
            positions=x,
            one_hot=one_hot,
            charges=charges,
            dataset_info=self.dataset_info,
            id_from=id_from,
            name=name,
            batch_index=batch_index
        )

        if getattr(self, "logger", None) is not None and getattr(self.logger, "experiment", None) is not None:
            experiment = self.logger.experiment
        else:
            experiment = None

        visualize_mol(str(output_dir), dataset_info=self.dataset_info, wandb_run=experiment)

    @typechecked
    def sample_chain_and_save(
        self,
        keep_frames: int,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None,
        context: Optional[TensorType["batch_size", "num_context_features"]] = None,
        id_from: int = 0,
        num_tries: int = 1,
        name: str = os.sep + "chain",
        verbose: bool = True
    ):
        # fixed hyperparameter(s)
        num_samples = 1

        # node count-conditioning
        if "GEOM" in self.dataset_info["name"]:
            num_nodes = torch.tensor([44], dtype=torch.long, device=self.device)
        else:
            if verbose:
                log.info(f"Sampling `num_nodes` for dataset {self.dataset_info['name']}")
            num_nodes = self.ddpm.num_nodes_distribution.sample(num_samples)
            max_num_nodes = (
                self.dataset_info["max_n_nodes"]
                if "max_n_nodes" in self.dataset_info
                else num_nodes.max().item()
            )
            assert int(num_nodes.max()) <= max_num_nodes

        # context-conditioning
        context = None

        one_hot, x = [None] * 2
        for i in range(num_tries):
            chain, _, _ = self.ddpm.mol_gen_sample(
                num_samples=num_samples,
                num_nodes=num_nodes,
                node_mask=node_mask,
                context=context,
                return_frames=keep_frames,
                device=self.device
            )

            chain = reverse_tensor(chain)

            # repeat last frame to see final sample better
            chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)

            # check stability of the generated molecule
            x_final = chain[-1, :, :self.num_x_dims].cpu().detach()
            one_hot_final = chain[-1, :, self.num_x_dims:-1] if self.include_charges else chain[-1, :, self.num_x_dims:]
            one_hot_final = torch.argmax(one_hot_final, dim=-1).cpu().detach()

            mol_stable = check_molecular_stability(
                positions=x_final,
                atom_types=one_hot_final,
                dataset_info=self.dataset_info
            )[0]

            # prepare entire chain
            x = chain[:, :, :self.num_x_dims]
            one_hot = chain[:, :, self.num_x_dims:-1] if self.include_charges else chain[:, :, self.num_x_dims:]
            one_hot = F.one_hot(
                torch.argmax(one_hot, dim=-1),
                num_classes=self.num_atom_types
            )
            charges = (
                torch.round(chain[:, :, -1:]).long()
                if self.include_charges
                else torch.zeros(0, dtype=torch.long, device=self.device)
            )

            if mol_stable and verbose:
                log.info("Found stable molecule to visualize :)")
                break
            elif i == num_tries - 1 and verbose:
                log.info("Did not find stable molecule :( -> showing last sample")

        # flatten (i.e., treat frame (chain dimension) as batch for visualization)
        x_flat = x.view(-1, x.size(-1))
        one_hot_flat = one_hot.view(-1, one_hot.size(-1))
        charges_flat = charges.view(-1, charges.size(-1)) if charges.numel() > 0 else charges
        batch_index_flat = torch.arange(x.size(0)).repeat_interleave(x.size(1))

        output_dir = Path(self.sampling_output_dir, f"epoch_{self.current_epoch}", "chain")
        save_xyz_file(
            path=str(output_dir),
            positions=x_flat,
            one_hot=one_hot_flat,
            charges=charges_flat,
            dataset_info=self.dataset_info,
            id_from=id_from,
            name=name,
            batch_index=batch_index_flat
        )

        if getattr(self, "logger", None) is not None and getattr(self.logger, "experiment", None) is not None:
            experiment = self.logger.experiment
        else:
            experiment = None

        visualize_mol_chain(str(output_dir), dataset_info=self.dataset_info, wandb_run=experiment)

    @typechecked
    def generate_molecules(
        self,
        ddpm_mode: Literal["unconditional", "inpainting"],
        num_samples: int,
        num_nodes: Optional[TensorType["batch_size"]] = None,
        sanitize: bool = False,
        largest_frag: bool = False,
        add_hydrogens: bool = False,
        relax_iter: int = 0,
        num_timesteps: Optional[int] = None,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None,
        context: Optional[TensorType["batch_size", "num_context_features"]] = None,
        **kwargs
    ) -> List[Chem.Mol]:
        """
        Generate molecules, with inpainting as an option
        Args:
            ddpm_mode: the method by which to generate molecules
            num_samples: number of samples to generate
            num_nodes: number of molecule nodes for each sample; sampled randomly if `None`
            sanitize: whether to sanitize molecules
            largest_frag: whether to return only the largest molecular fragment
            add_hydrogens: whether to include hydrogen atoms in the generated molecule
            relax_iter: number of force field optimization steps
            num_timesteps: number of denoising steps; will use training value instead if `None`
            node_mask: mask indicating which nodes are to be ignored during model generation
                    NOTE: `True` here means to fix a node's type and 3D position when `ddpm_mode=inpainting`;
                          `False` means to ignore a node when `ddpm_mode=unconditional`
            context: a batch of contextual features with which to condition the model's generations
            kwargs: additional e.g., inpainting parameters
        Returns:
            list of generated molecules
        """
        # node count-conditioning
        if num_nodes is None:
            num_nodes = self.ddpm.num_nodes_distribution.sample(num_samples)
            max_num_nodes = (
                self.dataset_info["max_n_nodes"]
                if "max_n_nodes" in self.dataset_info
                else num_nodes.max().item()
            )
            assert int(num_nodes.max()) <= max_num_nodes

        # context-conditioning
        context = None

        # sampling
        if ddpm_mode == "unconditional":
            # sample unconditionally
            xh, batch_index, _ = self.ddpm.mol_gen_sample(
                num_samples=num_samples,
                num_nodes=num_nodes,
                device=self.device,
                num_timesteps=num_timesteps,
                node_mask=node_mask,
                context=context
            )

        elif ddpm_mode == "inpainting":
            # employ inpainting
            batch_index = num_nodes_to_batch_index(
                num_samples=len(num_nodes),
                num_nodes=num_nodes,
                device=self.device
            )

            molecule = {
                "x": torch.zeros(
                    (len(batch_index), self.num_x_dims),
                    device=self.device,
                    dtype=torch.float
                ),
                "one_hot": torch.zeros(
                    (len(batch_index), self.num_atom_types),
                    device=self.device,
                    dtype=torch.float
                ),
                "charges": torch.zeros(
                    (len(batch_index), 1),
                    device=self.device,
                    dtype=torch.float
                ),
                "num_nodes": num_nodes,
                "batch_index": batch_index
            }

            if node_mask is None:
                # largely disable inpainting by sampling for all but the first node
                node_mask = torch.zeros(len(batch_index), dtype=torch.bool, device=self.device)
                node_mask[0] = True  # note: an arbitrary choice of a generation's fixed point
            else:
                # inpaint requested region as specified in `node_mask`
                pass

            # record molecule's original center of mass
            molecule_com_before = scatter(molecule["x"], batch_index, dim=0, reduce="mean")

            xh = self.ddpm.inpaint(
                molecule=molecule,
                node_mask_fixed=node_mask,
                num_timesteps=num_timesteps,
                context=context,
                **kwargs
            )

            # move generated molecule back to its original center of mass position
            molecule_com_after = scatter(xh[:, :self.num_x_dims], batch_index, dim=0, reduce="mean")
            xh[:, :self.num_x_dims] += (molecule_com_before - molecule_com_after)[batch_index]

        else:
            raise NotImplementedError(f"DDPM type {type(self.ddpm)} is currently not implemented.")

        x = xh[:, :self.num_x_dims].detach().cpu()
        atom_types = (
            xh[:, self.num_x_dims:-1].argmax(-1).detach().cpu()
            if self.include_charges
            else xh[:, self.num_x_dims:].argmax(-1).detach().cpu()
        )
        # TODO: incorporate charges in some meaningful way
        charges = (
            xh[:, -1].detach().cpu()
            if self.include_charges
            else torch.zeros(0, device=self.device)
        )

        # build RDKit molecule objects
        molecules = []
        for mol_pc in zip(
            batch_tensor_to_list(x, batch_index),
            batch_tensor_to_list(atom_types, batch_index)
        ):

            mol = build_molecule(
                *mol_pc,
                dataset_info=self.dataset_info,
                add_coords=True
            )
            mol = process_molecule(
                rdmol=mol,
                add_hydrogens=add_hydrogens,
                sanitize=sanitize,
                relax_iter=relax_iter,
                largest_frag=largest_frag
            )
            if mol is not None:
                molecules.append(mol)

        return molecules

    @typechecked
    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    @typechecked
    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        optimizer_idx: int,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        verbose: bool = False
    ):
        if not self.hparams.module_cfg.clip_gradients:
            return

        # allow gradient norm to be 150% + 2 * stdev of recent gradient history
        max_grad_norm = (
            1.5 * self.gradnorm_queue.mean() + 2 * self.gradnorm_queue.std()
        )

        # get current `grad_norm`
        params = [p for g in optimizer.param_groups for p in g["params"]]
        grad_norm = get_grad_norm(params)

        # note: Lightning will then handle the gradient clipping
        self.clip_gradients(
            optimizer,
            gradient_clip_val=max_grad_norm,
            gradient_clip_algorithm="norm"
        )

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if verbose:
            log.info(f"Current gradient norm: {grad_norm}")

        if float(grad_norm) > max_grad_norm:
            log.info(
                f"Clipped gradient with value {grad_norm:.1f}, since the maximum value currently allowed is {max_grad_norm:.1f}")

    def on_fit_end(self):
        """Lightning calls this upon completion of the user's call to `trainer.fit()` for model training.
        For example, Lightning will call this hook upon exceeding `trainer.max_epochs` in model training.
        """
        if self.trainer.is_global_zero:
            path_cfg = self.hparams.path_cfg
            if path_cfg is not None and path_cfg.grid_search_script_dir is not None:
                # uniquely record when model training is concluded
                grid_search_script_dir = self.hparams.path_cfg.grid_search_script_dir
                run_id = self.logger.experiment.id
                fit_end_indicator_filename = f"{run_id}.{HALT_FILE_EXTENSION}"
                fit_end_indicator_filepath = os.path.join(grid_search_script_dir, fit_end_indicator_filename)
                os.makedirs(grid_search_script_dir, exist_ok=True)
                with open(fit_end_indicator_filepath, "w") as f:
                    f.write("`on_fit_end` has been called.")
        return super().on_fit_end()


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "geom_mol_gen_ddpm.yaml")
    _ = hydra.utils.instantiate(cfg)
