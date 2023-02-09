# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import ast
import getpass
import json
import os
import wandb
import yaml
from datetime import datetime
from typing import Any, Dict, List, Tuple

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from src.utils.utils import replace_dict_str_values_conditionally, replace_dict_str_values_unconditionally

from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


# define constants #
HIGH_MEMORY = False  # whether to use high-memory (HM) mode
HALT_FILE_EXTENSION = "done"  # TODO: Update `src.models.HALT_FILE_EXTENSION` As Well Upon Making Changes Here!
IMAGE_TAG = "bb558b48"  # TODO: Ensure Is Correct!
USERNAME = getpass.getuser()  # TODO: Ensure Is Correct for Nautilus Before Each Grid Search!
TIMESTAMP = datetime.now().strftime("%m%d%Y_%H_%M")

# choose a base experiment to run
TASK = "qm9_mol_gen_ddpm"  # TODO: Ensure Is Correct Before Each Grid Search!
EXPERIMENT = f"{TASK.lower()}_grid_search"  # TODO: Ensure Is Correct Before Each Grid Search!
TEMPLATE_RUN_NAME = f"{TIMESTAMP}_{TASK.lower()}"
TEMPLATE_COMMAND_STR = f"cd /data/Repositories/Lab_Repositories/bio-diffusion && git pull origin main && /data/Repositories/Lab_Repositories/bio-diffusion/bio-diffusion/bin/python src/train.py experiment={EXPERIMENT}"
TEMPLATE_COMMAND_STR += " trainer.devices=auto"  # TODO: Remove Once No Longer Needed
NUM_RUNS_PER_EXPERIMENT = {"qm9_mol_gen_ddpm": 3, "geom_mol_gen_ddpm": 1}

# establish paths
OUTPUT_SCRIPT_FILENAME_PREFIX = "gpu_job"
SCRIPT_DIR = os.path.join("scripts")
OUTPUT_SCRIPT_DIR = os.path.join(SCRIPT_DIR, f"{TASK}_grid_search_scripts")
TEMPLATE_SCRIPT_FILEPATH = os.path.join(
    SCRIPT_DIR,
    "nautilus",
    "hm_gpu_job_template.yaml"
    if HIGH_MEMORY
    else "gpu_job_template.yaml"
)

assert TASK in NUM_RUNS_PER_EXPERIMENT.keys(), f"The task {TASK} is not currently available."


@typechecked
def build_command_string(
    run: Dict[str, Any],
    items_to_show: List[Tuple[str, Any]],
    command_str: str = TEMPLATE_COMMAND_STR,
    run_name: str = TEMPLATE_RUN_NAME,
    run_id: str = wandb.util.generate_id()
) -> str:
    # substitute latest grid search parameter values into command string of latest script
    command_str += f" tags='[bio-diffusion, qm9_mol_gen_ddpm, grid_search, nautilus]' logger=wandb logger.wandb.id={run_id} logger.wandb.name='{run_name}_GCPv{run['gcp_version']}"

    # install a unique WandB run name
    for s, (key, value) in zip(run["key_names"].split(), items_to_show):
        if s in ["C", "NV", "NB"]:
            # parse individual contexts to use for conditioning
            contexts = ast.literal_eval(value)
            command_str += f"_{s.strip()}:"
            for contextIndex, context in enumerate(contexts):
                command_str += f"{context}"
                command_str = (
                    command_str
                    if contextIndex == len(contexts) - 1
                    else command_str + "-"
                )
        elif s == "N":
            # bypass listing combined nonlinearities due to their redundancy
            pass
        else:
            command_str += f"_{s.strip()}:{value}"
    command_str += "'"  # ensure the WandB name ends in a single quote to avoid Hydra list parsing

    # establish directory in which to store and find checkpoints and other artifacts for run
    run_dir = os.path.join("logs", "train", "runs", run_id)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    command_str += f" hydra.run.dir={run_dir}"
    command_str += f" ckpt_path={ckpt_path}"  # define name of latest checkpoint for resuming model

    # manually specify version of GCP module to use
    command_str += f" model.module_cfg.selected_GCP._target_=src.models.components.gcpnet.GCP{run['gcp_version']}"

    # add each custom grid search argument
    for key, value in items_to_show:
        if key in ["model.module_cfg.conditioning", "model.diffusion_cfg.norm_values", "model.diffusion_cfg.norm_biases"]:
            # ensure that Hydra will be able to parse list of contexts to use for conditioning
            command_str += f" {key}='{value}'"
        elif key == "model.module_cfg.nonlinearities":
            # ensure that Hydra will be able to parse list of nonlinearities to use for training
            parsed_nonlinearities = [
                nonlinearity
                if nonlinearity is not None and len(nonlinearity) > 0
                else "null" for nonlinearity in value
            ]
            command_str += f" {key}='{parsed_nonlinearities}'"
        else:
            command_str += f" {key}={value}"

    return command_str


def main():
    # load search space from storage as JSON file
    search_space_filepath = os.path.join(SCRIPT_DIR, f"{TASK}_grid_search_runs.json")
    assert os.path.exists(
        search_space_filepath
    ), "JSON file describing grid search runs must be generated beforehand using `generate_grid_search_runs.py`"
    with open(search_space_filepath, "r") as f:
        grid_search_runs = json.load(f)

    # curate each grid search run
    grid_search_runs = [run for run in grid_search_runs for _ in range(NUM_RUNS_PER_EXPERIMENT[TASK])]
    for run_index, run in enumerate(grid_search_runs):
        # distinguish items to show in arguments list
        items_to_show = [(key, value) for (key, value) in run.items() if key not in ["gcp_version", "key_names"]]

        # build list of input arguments
        run_id = wandb.util.generate_id()
        cur_script_filename = f"{OUTPUT_SCRIPT_FILENAME_PREFIX}_{run_index}.yaml"
        command_str = build_command_string(run, items_to_show, run_id=run_id)

        # write out latest script as copy of template launcher script
        output_script_filepath = os.path.join(
            OUTPUT_SCRIPT_DIR, cur_script_filename
        )
        with open(TEMPLATE_SCRIPT_FILEPATH, "r") as f:
            yaml_dict = yaml.load(f, Loader)
        unconditional_yaml_dict = replace_dict_str_values_unconditionally(
            yaml_dict,
            unconditional_key_value_replacements={
                "$JOB_INDEX": f"-{run_index}",
                "$IMAGE_TAG": IMAGE_TAG,
                "$USER": USERNAME,
                "$EXPERIMENT": EXPERIMENT
            }
        )
        conditional_yaml_dict = replace_dict_str_values_conditionally(
            unconditional_yaml_dict,
            conditional_key_value_replacements={"command": ["bash", "-c", command_str]}
        )
        with open(output_script_filepath, "w") as f:
            yaml.dump(conditional_yaml_dict, f, Dumper)


if __name__ == "__main__":
    main()
