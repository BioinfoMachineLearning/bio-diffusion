# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import hydra
import os
import pyrootutils
import torch

import prody as pr
import torch.nn as nn

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from typing import Any, Dict, Optional, Tuple, Union


from src.datamodules.components.edm import get_bond_length_arrays
from src.datamodules.components.edm.datasets_config import QM9_WITH_H, QM9_WITHOUT_H
from src.models import NumNodesDistribution, PropertiesDistribution, compute_mean_mad
from src.utils.pylogger import get_pylogger

from src import LR_SCHEDULER_MANUAL_INTERPOLATION_HELPER_CONFIG_ITEMS, LR_SCHEDULER_MANUAL_INTERPOLATION_PRIMARY_CONFIG_ITEMS, get_classifier, test_with_property_classifier, utils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typeguard import typechecked
from torchtyping import patch_typeguard

patch_typeguard()  # use before @typechecked


pr.confProDy(verbosity="none")

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


class ConditionalDiffusionDataLoader:
    def __init__(
        self,
        model: nn.Module,
        nodes_distr: nn.Module,
        props_distr: object,
        batch_size: int,
        device: Union[torch.device, str],
        dataset_info: Dict[str, Any],
        iterations: int = 200,
        unknown_labels: bool = False
    ):
        self.model = model
        self.nodes_distr = nodes_distr
        self.props_distr = props_distr
        self.device = device
        self.dataset_info = dataset_info
        self.iterations = iterations
        self.unknown_labels = unknown_labels
        self.num_samples = batch_size
        self.i = 0

    def __iter__(self):
        return self

    @typechecked
    def sample(self) -> Dict[str, Any]:
        num_nodes = self.nodes_distr.sample(self.num_samples).to(self.device)
        context = self.props_distr.sample_batch(num_nodes).to(self.device)
        x, one_hot, _ = self.model.sample(
            num_samples=self.num_samples,
            num_nodes=num_nodes,
            context=context
        )

        # build dense node mask, coordinates, and one-hot types
        max_num_nodes = num_nodes.max().item()
        node_mask_range_tensor = torch.arange(max_num_nodes, device=self.device).unsqueeze(0)
        node_mask = node_mask_range_tensor < num_nodes.unsqueeze(-1)
        dense_x = torch.zeros((self.num_samples, max_num_nodes, x.size(-1)), device=self.device)
        dense_x[node_mask] = x
        dense_one_hot = torch.zeros((self.num_samples, max_num_nodes, one_hot.size(-1)), device=self.device)
        dense_one_hot[node_mask] = one_hot
        context = context.squeeze(1)

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
            return self.sample()
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
        os.path.exists(cfg.generator_model_filepath) and
        os.path.exists(cfg.classifier_model_dir) and
        cfg.property in cfg.generator_model_filepath and
        cfg.property in cfg.classifier_model_dir
    )

    log.info("Loading classifier model!")
    device = f"cuda:{cfg.trainer.devices[0]}" if torch.cuda.is_available() else "cpu"
    classifier = get_classifier(cfg.classifier_model_dir).to(device)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()

    log.info("Installing conditional model configuration values!")
    cfg.model.module_cfg.conditioning = [cfg.property]
    cfg.datamodule.dataloader_cfg.include_charges = False
    cfg.datamodule.dataloader_cfg.dataset = "QM9_second_half"
    cfg.model.diffusion_cfg.norm_values = [1.0, 8.0, 1.0]

    log.info(f"Instantiating generator model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        model_cfg=hydra.utils.instantiate(cfg.model.model_cfg),
        module_cfg=hydra.utils.instantiate(cfg.model.module_cfg),
        layer_cfg=hydra.utils.instantiate(cfg.model.layer_cfg),
        diffusion_cfg=hydra.utils.instantiate(cfg.model.diffusion_cfg),
        dataloader_cfg=getattr(cfg.datamodule, "dataloader_cfg", None),
        path_cfg=cfg.paths
    )

    log.info("Loading generator checkpoint!")
    model = model.load_from_checkpoint(
        # allow one to evaluate with an older model using custom hyperparameters
        checkpoint_path=cfg.generator_model_filepath,
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

    log.info("Creating dataloader with generator!")

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
    histogram = {int(k): int(v) for k, v in dataset_info["n_nodes"].items()}
    nodes_distr = NumNodesDistribution(histogram)

    conditional_diffusion_dataloader = ConditionalDiffusionDataLoader(
        model=model,
        nodes_distr=nodes_distr,
        props_distr=props_distr,
        batch_size=cfg.batch_size,
        device=device,
        dataset_info=dataset_info,
        iterations=cfg.iterations
    )
    
    log.info("Evaluating classifier on generator's samples!")
    loss = test_with_property_classifier(
        model=classifier,
        epoch=0,
        dataloader=conditional_diffusion_dataloader,
        mean=mean,
        mad=mad,
        property=cfg.property,
        device=device,
        log_interval=1,
        debug_break=cfg.debug_break
    )
    log.info("Classifier loss (MAE) on generator's samples: %.4f" % loss)

    metric_dict = {}
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model
    }

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="mol_gen_eval_conditional_qm9.yaml")
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
