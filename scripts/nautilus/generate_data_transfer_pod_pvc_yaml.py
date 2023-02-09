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
USERNAME = getpass.getuser()  # TODO: Ensure Is Correct for Nautilus Before Each Grid Search!
SCRIPT_DIR = os.path.join("scripts", "nautilus")
DATA_TRANSFER_POD_PVC_TEMPLATE_FILEPATH = os.path.join(SCRIPT_DIR, "data_transfer_pod_pvc_template.yaml")
DATA_TRANSFER_POD_PVC_OUTPUT_FILEPATH = os.path.join(SCRIPT_DIR, "data_transfer_pod_pvc.yaml")


def main():
    with open(DATA_TRANSFER_POD_PVC_TEMPLATE_FILEPATH, "r") as f:
        yaml_dict = yaml.load(f, Loader)

    unconditional_yaml_dict = replace_dict_str_values_unconditionally(
        yaml_dict,
        unconditional_key_value_replacements={"$USER": USERNAME}
    )

    with open(DATA_TRANSFER_POD_PVC_OUTPUT_FILEPATH, "w") as f:
        yaml.dump(unconditional_yaml_dict, f, Dumper)


if __name__ == "__main__":
    main()
