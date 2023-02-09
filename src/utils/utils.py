# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import hydra
import signal
import time
import warnings

from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

from contextlib import contextmanager
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from src.utils import pylogger, rich_utils

from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked

log = pylogger.get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished
    - Logging the exception if occurs
    - Logging the task total execution time
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):

        # apply extra utilities
        extras(cfg)

        # execute the task
        try:
            start_time = time.time()
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")  # save exception to `.log` file
            raise ex
        finally:
            path = Path(cfg.paths.output_dir, "exec_time.log")
            content = f"'{cfg.task_name}' execution time: {time.time() - start_time} (s)"
            save_file(path, content)  # save task execution time (even if exception occurs)
            close_loggers()  # close loggers (even if exception occurs so multirun won't fail)

        log.info(f"Output dir: {cfg.paths.output_dir}")

        return metric_dict, object_dict

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[LightningLoggerBase]:
    """Instantiates loggers from config."""
    logger: List[LightningLoggerBase] = []

    if not logger_cfg:
        log.warning("Logger config is empty.")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


@typechecked
def replace_dict_str_values_unconditionally(
    obj: Any,
    unconditional_key_value_replacements: Dict[str, str]
) -> Any:
    """Using a replacement mapping, replace all string values found in `obj` recursively."""
    if isinstance(obj, dict):
        return {key: replace_dict_str_values_unconditionally(val, unconditional_key_value_replacements) for key, val in obj.items()}
    elif isinstance(obj, list):
        if len(obj) > 0 and isinstance(obj[0], str):
            new_obj_list = []
            for o in obj:
                filled_obj = o
                for u, r in unconditional_key_value_replacements.items():
                    filled_obj = filled_obj.replace(u, r)
                new_obj_list.append(filled_obj)
            return new_obj_list
        elif len(obj) > 0 and isinstance(obj[0], dict):
            return [
                {
                    key: replace_dict_str_values_unconditionally(val, unconditional_key_value_replacements)
                    for key, val in o.items()
                }
                for o in obj
            ]
        else:
            return obj
    elif isinstance(obj, str):
        filled_obj = obj
        for u, r in unconditional_key_value_replacements.items():
            filled_obj = filled_obj.replace(u, r)
        return filled_obj
    else:
        return obj


def replace_dict_str_values_conditionally_for_dicts(
    obj: dict,
    conditional_key_value_replacements: Dict[str, Union[str, List[str]]]
):
    """Using a replacement mapping for dictionaries, recursively replace all string values found in `obj` that correspond to keys in the mapping."""
    for k in obj.keys():
        if k in conditional_key_value_replacements:
            obj[k] = conditional_key_value_replacements[k]
    return {key: replace_dict_str_values_conditionally(val, conditional_key_value_replacements) for key, val in obj.items()}


@typechecked
def replace_dict_str_values_conditionally(
    obj: Any,
    conditional_key_value_replacements: Dict[str, Union[str, List[str]]]
) -> Any:
    """Using a replacement mapping, recursively replace all string values found in `obj` that correspond to keys in the mapping."""
    if isinstance(obj, dict):
        return replace_dict_str_values_conditionally_for_dicts(
            obj, conditional_key_value_replacements=conditional_key_value_replacements
        )
    elif isinstance(obj, list):
        new_obj_list = []
        for o in obj:
            if isinstance(o, dict):
                new_obj_list.append(
                    replace_dict_str_values_conditionally_for_dicts(
                        o, conditional_key_value_replacements=conditional_key_value_replacements
                    )
                )
            else:
                new_obj_list.append(o)
        return new_obj_list
    else:
        return obj


class TimeoutException(Exception):
    pass


@contextmanager
@typechecked
def time_limit(seconds: int):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
