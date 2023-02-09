# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import getpass
import os
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from src.utils.utils import replace_dict_str_values_unconditionally


# define constants #
JOB_INDEX = ""  # TODO: Ensure Is Correct!
IMAGE_TAG = "bb558b48"  # TODO: Ensure Is Correct!
USERNAME = getpass.getuser()  # TODO: Ensure Is Correct for Nautilus Before Each Grid Search!
EXPERIMENT = "qm9_mol_gen_ddpm"  # TODO: Ensure Is Correct for Nautilus Before Each Grid Search!
SCRIPT_DIR = os.path.join("scripts", "nautilus")
GPU_JOB_TEMPLATE_FILEPATH = os.path.join(SCRIPT_DIR, "gpu_job_template.yaml")
GPU_JOB_OUTPUT_FILEPATH = os.path.join(SCRIPT_DIR, "gpu_job.yaml")


def main():
    with open(GPU_JOB_TEMPLATE_FILEPATH, "r") as f:
        yaml_dict = yaml.load(f, Loader)

    unconditional_yaml_dict = replace_dict_str_values_unconditionally(
        yaml_dict,
        unconditional_key_value_replacements={
            "$JOB_INDEX": JOB_INDEX,
            "$IMAGE_TAG": IMAGE_TAG,
            "$USER": USERNAME,
            "$EXPERIMENT": EXPERIMENT
        }
    )

    with open(GPU_JOB_OUTPUT_FILEPATH, "w") as f:
        yaml.dump(unconditional_yaml_dict, f, Dumper)


if __name__ == "__main__":
    main()
