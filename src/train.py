# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import hydra
import pyrootutils

import prody as pr
import pytorch_lightning as pl

from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from typing import List, Optional, Tuple

from src.utils.pylogger import get_pylogger

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from src import LR_SCHEDULER_MANUAL_INTERPOLATION_HELPER_CONFIG_ITEMS, LR_SCHEDULER_MANUAL_INTERPOLATION_PRIMARY_CONFIG_ITEMS, MODEL_WATCHING_LOGGERS, unwatch_model, utils, watch_model

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
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        model_cfg=hydra.utils.instantiate(cfg.model.model_cfg),
        module_cfg=hydra.utils.instantiate(cfg.model.module_cfg),
        layer_cfg=hydra.utils.instantiate(cfg.model.layer_cfg),
        diffusion_cfg=cfg.model.diffusion_cfg,
        dataloader_cfg=getattr(cfg.datamodule, "dataloader_cfg", None),
        path_cfg=cfg.paths
    )
    if cfg.get("ckpt_path") is not None:
        if os.path.exists(cfg.ckpt_path):
            log.info("Loading checkpoint!")
            model = model.load_from_checkpoint(
                # allow one to resume training with an older model using custom hyperparameters
                cfg.ckpt_path,
                strict=False,
                model_cfg=hydra.utils.instantiate(cfg.model.model_cfg),
                module_cfg=hydra.utils.instantiate(cfg.model.module_cfg),
                layer_cfg=hydra.utils.instantiate(cfg.model.layer_cfg),
                diffusion_cfg=cfg.model.diffusion_cfg,
                dataloader_cfg=getattr(cfg.datamodule, "dataloader_cfg", None),
                path_cfg=cfg.paths
            )
        else:
            log.warning("Requested ckpt not found! Using new weights for training...")

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(
        cfg.get("callbacks")
    )

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(
        cfg.get("logger")
    )

    log.info("Watching model via logger...") if cfg.watch_model else None
    for logger_ in logger:
        if cfg.watch_model and type(logger_) in MODEL_WATCHING_LOGGERS:
            watch_model(model, logger_)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    cfg.trainer.strategy = (
        "ddp"
        if cfg.trainer.strategy == "ddp_find_unused_parameters_false" and
         any([cfg.model.module_cfg.ablate_scalars, cfg.model.module_cfg.ablate_vectors])
        else cfg.trainer.strategy
    )
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        ckpt_path = (
            cfg.get("ckpt_path")
            if cfg.get("ckpt_path") is not None and os.path.exists(cfg.get("ckpt_path"))
            else None
        )
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path
        )

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    # finish watching model after training has concluded
    for logger_ in logger:
        if cfg.watch_model and type(logger_) in MODEL_WATCHING_LOGGERS:
            unwatch_model(model, logger_)

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
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

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
