# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import torch
import hydra
import os
import pyrootutils

import prody as pr

from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning import LightningModule, Trainer
from omegaconf import DictConfig
from typing import List, Tuple
from datetime import datetime
from pathlib import Path

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from src.utils.pylogger import get_pylogger

from src import LR_SCHEDULER_MANUAL_INTERPOLATION_HELPER_CONFIG_ITEMS, LR_SCHEDULER_MANUAL_INTERPOLATION_PRIMARY_CONFIG_ITEMS, utils
from src.datamodules.components.edm import get_bond_length_arrays
from src.models.components import num_nodes_to_batch_index, write_sdf_file

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


@utils.task_wrapper
def sample(cfg: DictConfig) -> Tuple[dict, dict]:
    """Performs sampling using a given checkpoint.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        model_cfg=hydra.utils.instantiate(cfg.model.model_cfg),
        module_cfg=hydra.utils.instantiate(cfg.model.module_cfg),
        layer_cfg=hydra.utils.instantiate(cfg.model.layer_cfg),
        diffusion_cfg=hydra.utils.instantiate(cfg.model.diffusion_cfg),
        dataloader_cfg=getattr(cfg.datamodule, "dataloader_cfg", None),
        path_cfg=cfg.paths
    )
    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Loading checkpoint!")
    device = (
        ("cuda" if isinstance(cfg.trainer.devices, int) else f"cuda:{cfg.trainer.devices[0]}")
        if torch.cuda.is_available()
        else "cpu"
    )
    model = model.load_from_checkpoint(
        checkpoint_path=cfg.ckpt_path,
        map_location=device,
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

    log.info("Starting sampling!")

    # node count-conditioning
    num_nodes = (
        torch.tensor([cfg.num_nodes for _ in range(cfg.num_samples)], dtype=torch.long, device=device)
        if cfg.num_nodes
        else model.ddpm.num_nodes_distribution.sample(cfg.num_samples)
    )
    max_num_nodes = (
        model.dataset_info["max_n_nodes"]
        if "max_n_nodes" in model.dataset_info
        else num_nodes.max().item()
    )
    assert num_nodes.max().item() <= max_num_nodes

    batch_index = num_nodes_to_batch_index(
        num_samples=len(num_nodes),
        num_nodes=num_nodes,
        device=device
    )

    # TODO: replace values in `node_mask` with one's desired mask values for unconditional and inpainting generation
    if cfg.model.diffusion_cfg.ddpm_mode == "unconditional":
        # note: `False` here means to ignore a node during molecule generation
        node_mask = torch.ones(len(batch_index), dtype=torch.bool, device=device)
    elif cfg.model.diffusion_cfg.ddpm_mode == "inpainting":
        # note: `True` here means to fix an atom's type and 3D position throughout molecule generation
        node_mask = torch.zeros(len(batch_index), dtype=torch.bool, device=device)
        node_mask[0] = True  # note: an arbitrary choice of a generation's fixed point
    else:
        raise NotImplementedError(f"DDPM mode {cfg.model.diffusion_cfg.ddpm_mode} is currently not implemented.")

    # note: in general, one's model should have a sample() function to call like this one       
    molecules = model.generate_molecules(
        ddpm_mode=cfg.model.diffusion_cfg.ddpm_mode,
        num_samples=cfg.num_samples,
        num_nodes=num_nodes,
        sanitize=cfg.sanitize,
        largest_frag=not cfg.all_frags,
        add_hydrogens=False,
        sample_chain=cfg.sample_chain,
        relax_iter=(200 if cfg.relax else 0),
        num_timesteps=cfg.num_timesteps,
        node_mask=node_mask,
        num_resamplings=cfg.num_resamplings,
        jump_length=cfg.jump_length
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d%Y_%H_%M_%S")
    write_sdf_file(Path(cfg.output_dir, f"{timestamp}_mol.sdf"), molecules)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="mol_gen_sample.yaml")
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

    sample(cfg)


if __name__ == "__main__":
    main()
