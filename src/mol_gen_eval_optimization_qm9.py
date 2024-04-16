# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import hydra
import os
import pyrootutils
import torch

import torch.nn as nn

from omegaconf import DictConfig, open_dict
from pathlib import Path
from pytorch_lightning import LightningDataModule, LightningModule
from typing import Any, Dict, List, Optional, Tuple, Union

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from src.datamodules.components.edm import check_molecular_stability, get_bond_length_arrays
from src.datamodules.components.edm.datasets_config import QM9_WITH_H, QM9_WITHOUT_H
from src.models import PropertiesDistribution, compute_mean_mad
from src.models.components import load_molecule_xyz, save_xyz_file
from src.utils.pylogger import get_pylogger

from src import LR_SCHEDULER_MANUAL_INTERPOLATION_HELPER_CONFIG_ITEMS, LR_SCHEDULER_MANUAL_INTERPOLATION_PRIMARY_CONFIG_ITEMS, get_classifier, test_with_property_classifier, utils

from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard

patch_typeguard()  # use before @typechecked

import lovely_tensors as lt
lt.monkey_patch()


QM9_OPTIMIZATION_NUM_NODES = 19


# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is an optional line at the top of each entry file
# that helps to make the environment more robust and convenient
#
# the main advantages are:
# - allows you to keep all entry files in "src/" without installing project as a package
# - makes paths and scripts always work no matter where is your current work dir
# - automatically loads environment variables from ".env" file if exists
#
# how it works:
# - the line above recursively searches for either ".git" or "pyproject.toml" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
#   to make all paths always relative to the project root
# - loads environment variables from ".env" file in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. simply remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
# 3. always run entry files from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #


log = get_pylogger(__name__)


class OptimizationDiffusionDataLoader:
    def __init__(
        self,
        model: nn.Module,
        sampling_output_dir: str,
        num_nodes: TensorType["num_samples"],
        optim_property: str,
        props_distr: object,
        device: Union[torch.device, str],
        dataset_info: Dict[str, Any],
        iterations: int = 200,
        num_optimization_timesteps: int = 10,
        return_frames: int = 1,
        experiment_name: str = "conditional_diffusion",
        unknown_labels: bool = False,
        save_molecules: bool = False,
    ):
        assert iterations > 0, \
            "Optimization requires at least two iterations, " \
            "one for initial sample scoring and the other for optimization scoring."

        self.model = model
        self.sampling_output_dir = sampling_output_dir
        self.num_nodes = num_nodes
        self.optim_property = optim_property
        self.props_distr = props_distr
        self.device = device
        self.dataset_info = dataset_info
        self.iterations = iterations
        self.experiment_name = experiment_name
        self.num_optimization_timesteps = num_optimization_timesteps
        self.return_frames = return_frames
        self.unknown_labels = unknown_labels
        self.save_molecules = save_molecules

        self.samples = self.load_pregenerated_samples(sampling_output_dir)
        self.context = self.props_distr.sample_batch(num_nodes).to(device)  # fix context so we can compare between iterations
        self.num_samples = len(self.samples)

        self.i = 0  # define iteration counter

    def __iter__(self):
        return self
    
    @typechecked
    def load_pregenerated_samples(
        self,
        samples_dir: str
    ) -> List[
        Tuple[torch.Tensor, torch.Tensor]
    ]:
        samples = [
            load_molecule_xyz(os.path.join(samples_dir, item), self.dataset_info)
            for item in os.listdir(samples_dir)
            if item.endswith(".xyz")
        ]
        return samples

    @typechecked
    def optimize_pregenerated_samples(
        self,
        score_initial_samples: bool = False,
        analyze_initial_samples_stability: bool = True,
        id_from: int = 0
    ) -> Dict[str, Any]:
        if score_initial_samples:
            # evaluate stability of initial molecules
            if analyze_initial_samples_stability:
                num_mols_stable = 0
                for sample in self.samples:
                    x_final, one_hot_final = sample[0], torch.argmax(sample[1], dim=-1) 
                    mol_stable = check_molecular_stability(
                        positions=x_final,
                        atom_types=one_hot_final,
                        dataset_info=self.dataset_info
                    )[0]
                    num_mols_stable = num_mols_stable + 1 if mol_stable else num_mols_stable
                mols_stable_pct = (num_mols_stable / self.num_samples) * 100
                log.info(f"Percentage of initial samples that are stable molecules: {mols_stable_pct}%")
            
            # collate initial molecules
            x, one_hot = (
                torch.vstack([sample[0] for sample in self.samples]).to(self.device),
                torch.vstack([sample[1] for sample in self.samples]).to(self.device)
            )
        else:
            # optimize initial molecules
            x, one_hot, charges, batch_index = self.model.optimize(
                samples=self.samples,
                num_nodes=self.num_nodes,
                context=self.context,
                num_timesteps=self.num_optimization_timesteps,
                sampling_output_dir=self.sampling_output_dir,
                optim_property=self.optim_property,
                iteration_index=self.i - 1,
                return_frames=self.return_frames,
                norm_with_original_timesteps=False, # NOTE: this is important to ensure the samples are "fully optimized" each iteration
            )

            # iteratively update samples with optimized molecule features
            self.samples = [
                (x[(batch_index == sample_idx)].cpu(), one_hot[(batch_index == sample_idx)].cpu())
                for sample_idx in range(self.num_samples)
            ]

            # evaluate stability of optimized molecules
            num_mols_stable = 0
            for sample_idx in range(self.num_samples):
                x_final, one_hot_final = (
                    x[(batch_index == sample_idx)].cpu(),
                    torch.argmax(one_hot[(batch_index == sample_idx)], dim=-1).cpu()
                )
                mol_stable = check_molecular_stability(
                    positions=x_final,
                    atom_types=one_hot_final,
                    dataset_info=self.dataset_info
                )[0]
                num_mols_stable = num_mols_stable + 1 if mol_stable else num_mols_stable
            mols_stable_pct = (num_mols_stable / self.num_samples) * 100
            log.info(f"Percentage of optimized samples that are stable molecules: {mols_stable_pct}%")

            # record optimized molecules as `.xyz` files
            if self.save_molecules:
                save_xyz_file(
                    path=f"outputs/{self.experiment_name}/analysis/run{self.i}/",
                    positions=x,
                    one_hot=one_hot,
                    charges=charges,
                    dataset_info=self.dataset_info,
                    id_from=id_from,
                    name="conditional",
                    batch_index=batch_index
                )

        # build dense node mask, coordinates, and one-hot types
        max_num_nodes = self.num_nodes.max().item()
        node_mask_range_tensor = torch.arange(max_num_nodes, device=self.device).unsqueeze(0)
        node_mask = node_mask_range_tensor < self.num_nodes.unsqueeze(-1)
        dense_x = torch.zeros((self.num_samples, max_num_nodes, x.size(-1)), device=self.device)
        dense_x[node_mask] = x
        dense_one_hot = torch.zeros((self.num_samples, max_num_nodes, one_hot.size(-1)), device=self.device)
        dense_one_hot[node_mask] = one_hot
        context = self.context.squeeze(1)

        # build edge mask
        bs, n_nodes = self.num_samples, max_num_nodes
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        diag_mask = diag_mask.to(self.device)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)

        # build context
        prop_key = self.props_distr.properties[0]
        if self.unknown_labels:
            context[:] = self.props_distr.normalizer[prop_key]["mean"]
        else:
            context = (
                context * self.props_distr.normalizer[prop_key]["mad"] + self.props_distr.normalizer[prop_key]["mean"]
            )
        data = {
            "positions": dense_x.detach(),
            "atom_mask": node_mask.detach(),
            "edge_mask": edge_mask.detach(),
            "one_hot": dense_one_hot.detach(),
            prop_key: context.detach()
        }
        return data

    def __next__(self) -> Optional[Dict[str, Any]]:
        if self.i <= self.iterations:
            self.i += 1
            return self.optimize_pregenerated_samples(
                score_initial_samples=(self.i == 1)
            )
        else:
            self.i = 0
            raise StopIteration

    def __len__(self):
        return self.iterations


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert (
        os.path.exists(cfg.unconditional_generator_model_filepath) and
        os.path.exists(cfg.conditional_generator_model_filepath) and
        os.path.exists(cfg.classifier_model_dir) and
        cfg.property in cfg.conditional_generator_model_filepath and
        cfg.property in cfg.classifier_model_dir
    )

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()

    device = (
        ("cuda" if isinstance(cfg.trainer.devices, int) else f"cuda:{cfg.trainer.devices[0]}")
        if torch.cuda.is_available()
        else "cpu"
    )
    sampling_num_nodes = torch.tensor([QM9_OPTIMIZATION_NUM_NODES for _ in range(cfg.num_samples)], device=device)

    if not cfg.use_pregenerated_molecules:
        log.info(f"Instantiating unconditional generator model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(
            cfg.model,
            model_cfg=hydra.utils.instantiate(cfg.model.model_cfg),
            module_cfg=hydra.utils.instantiate(cfg.model.module_cfg),
            layer_cfg=hydra.utils.instantiate(cfg.model.layer_cfg),
            diffusion_cfg=hydra.utils.instantiate(cfg.model.diffusion_cfg),
            dataloader_cfg=getattr(cfg.datamodule, "dataloader_cfg", None),
            path_cfg=cfg.paths
        )

        log.info("Loading unconditional generator checkpoint!")
        model = model.load_from_checkpoint(
            # allow one to evaluate with an older model using custom hyperparameters
            checkpoint_path=cfg.unconditional_generator_model_filepath,
            map_location=device,
            strict=False,
            model_cfg=hydra.utils.instantiate(cfg.model.model_cfg),
            module_cfg=hydra.utils.instantiate(cfg.model.module_cfg),
            layer_cfg=hydra.utils.instantiate(cfg.model.layer_cfg),
            diffusion_cfg=hydra.utils.instantiate(cfg.model.diffusion_cfg),
            dataloader_cfg=getattr(cfg.datamodule, "dataloader_cfg", None),
            path_cfg=cfg.paths
        )
        model = model.to(device)

        # ensure valid bond lengths have been added to each dataset's metadata collection (i.e., `model.dataset_info`)
        if any([
            not getattr(model.dataset_info, "bonds1", None),
            not getattr(model.dataset_info, "bonds2", None),
            not getattr(model.dataset_info, "bonds3", None)
        ]):
            bonds = get_bond_length_arrays(model.dataset_info["atom_encoder"])
            model.dataset_info["bonds1"], model.dataset_info["bonds2"], model.dataset_info["bonds3"] = (
                bonds[0], bonds[1], bonds[2]
            )

        log.info("Starting sampling with unconditional model!")

        os.makedirs(cfg.sampling_output_dir, exist_ok=True)
        model.sample_and_save(
            num_samples=cfg.num_samples,
            num_nodes=sampling_num_nodes,
            sampling_output_dir=Path(cfg.sampling_output_dir),
            num_timesteps=cfg.num_timesteps,
            norm_with_original_timesteps=False,  # NOTE: this is important to ensure the initial samples are "unoptimized" yet realistic when `num_timesteps << T`
        )

        if cfg.generate_molecules_only:
            log.info(f"Done generating {cfg.num_samples} 3D molecules unconditionally! Exiting...")
            exit(0)

    log.info("Installing conditional model configuration values!")
    with open_dict(cfg):
        cfg.model.module_cfg.conditioning = [cfg.property]
        cfg.model.diffusion_cfg.norm_values = [1.0, 8.0, 1.0]
        cfg.datamodule.dataloader_cfg.include_charges = False
        cfg.datamodule.dataloader_cfg.dataset = "QM9_second_half"

    log.info(f"Instantiating conditional generator model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        model_cfg=hydra.utils.instantiate(cfg.model.model_cfg),
        module_cfg=hydra.utils.instantiate(cfg.model.module_cfg),
        layer_cfg=hydra.utils.instantiate(cfg.model.layer_cfg),
        diffusion_cfg=hydra.utils.instantiate(cfg.model.diffusion_cfg),
        dataloader_cfg=getattr(cfg.datamodule, "dataloader_cfg", None),
        path_cfg=cfg.paths
    )

    log.info("Loading conditional generator checkpoint!")
    model = model.load_from_checkpoint(
        # allow one to evaluate with an older model using custom hyperparameters
        checkpoint_path=cfg.conditional_generator_model_filepath,
        map_location=device,
        strict=False,
        model_cfg=hydra.utils.instantiate(cfg.model.model_cfg),
        module_cfg=hydra.utils.instantiate(cfg.model.module_cfg),
        layer_cfg=hydra.utils.instantiate(cfg.model.layer_cfg),
        diffusion_cfg=hydra.utils.instantiate(cfg.model.diffusion_cfg),
        dataloader_cfg=getattr(cfg.datamodule, "dataloader_cfg", None),
        path_cfg=cfg.paths
    )
    model = model.to(device)

    # ensure valid bond lengths have been added to each dataset's metadata collection (i.e., `model.dataset_info`)
    if any([
        not getattr(model.dataset_info, "bonds1", None),
        not getattr(model.dataset_info, "bonds2", None),
        not getattr(model.dataset_info, "bonds3", None)
    ]):
        bonds = get_bond_length_arrays(model.dataset_info["atom_encoder"])
        model.dataset_info["bonds1"], model.dataset_info["bonds2"], model.dataset_info["bonds3"] = (
            bonds[0], bonds[1], bonds[2]
        )

    splits = ["train", "valid", "test"]
    dataloaders = [
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        datamodule.test_dataloader()
    ]
    dataloaders = {split: dataloader for (split, dataloader) in zip(splits, dataloaders)}
    props_norms = compute_mean_mad(
        dataloaders, list(cfg.model.module_cfg.conditioning), cfg.datamodule.dataloader_cfg.dataset
    )
    mean, mad = props_norms[cfg.property]["mean"].item(), props_norms[cfg.property]["mad"].item()

    props_distr = PropertiesDistribution(
        datamodule.train_dataloader(), list(cfg.model.module_cfg.conditioning), device=device
    )
    props_distr.set_normalizer(props_norms)

    dataset_info_mapping = {
        "QM9": QM9_WITHOUT_H if cfg.datamodule.dataloader_cfg.remove_h else QM9_WITH_H,
        "QM9_first_half": QM9_WITHOUT_H if cfg.datamodule.dataloader_cfg.remove_h else QM9_WITH_H,
        "QM9_second_half": QM9_WITHOUT_H if cfg.datamodule.dataloader_cfg.remove_h else QM9_WITH_H
    }
    dataset_info = dataset_info_mapping[cfg.datamodule.dataloader_cfg.dataset]

    bonds = get_bond_length_arrays(dataset_info["atom_encoder"])
    dataset_info["bonds1"], dataset_info["bonds2"], dataset_info["bonds3"] = (
        bonds[0], bonds[1], bonds[2]
    )

    log.info("Creating dataloader with conditional generator!")

    optimization_diffusion_dataloader = OptimizationDiffusionDataLoader(
        model=model,
        sampling_output_dir=cfg.sampling_output_dir,
        num_nodes=sampling_num_nodes,
        optim_property=cfg.property,
        props_distr=props_distr,
        device=device,
        dataset_info=dataset_info,
        iterations=cfg.iterations,
        num_optimization_timesteps=cfg.num_optimization_timesteps,
        return_frames=cfg.return_frames,
        experiment_name=cfg.experiment_name,
        save_molecules=cfg.save_molecules
    )

    log.info("Loading classifier model!")
    classifier = get_classifier(cfg.classifier_model_dir).to(device)
    
    log.info("Evaluating classifier on conditional generator's optimized samples!")
    with torch.no_grad():
        loss = test_with_property_classifier(
            model=classifier,
            epoch=0,
            dataloader=optimization_diffusion_dataloader,
            mean=mean,
            mad=mad,
            property=cfg.property,
            device=device,
            log_interval=1,
            debug_break=cfg.debug_break
        )
    log.info("Classifier loss (MAE) on conditional generator's optimized samples: %.4f" % loss)

    metric_dict = {}
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model
    }

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="mol_gen_eval_optimization_qm9.yaml")
def main(cfg: DictConfig) -> None:
    # work around Hydra's (current) lack of support for arithmetic expressions with interpolated config variables
    # reference: https://github.com/facebookresearch/hydra/issues/1286
    if cfg.model.get("scheduler") is not None:
        for key in cfg.model.scheduler.keys():
            if key in LR_SCHEDULER_MANUAL_INTERPOLATION_PRIMARY_CONFIG_ITEMS:
                setattr(cfg.model.scheduler, key, eval(cfg.model.scheduler.get(key)))
        # ensure that all requested arithmetic expressions have been performed using interpolated config variables
        lr_scheduler_key_names = [name for name in cfg.model.scheduler.keys()]
        for key in lr_scheduler_key_names:
            if key in LR_SCHEDULER_MANUAL_INTERPOLATION_HELPER_CONFIG_ITEMS:
                delattr(cfg.model.scheduler, key)

    evaluate(cfg)


if __name__ == "__main__":
    main()
