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

from src.utils.utils import replace_dict_str_values_conditionally, replace_dict_str_values_unconditionally


# define constants #
STORAGE_SIZE = "1000Gi"  # TODO: Ensure Is Correct for Nautilus Before Each Grid Search!
USERNAME = getpass.getuser()  # TODO: Ensure Is Correct for Nautilus Before Each Grid Search!
SCRIPT_DIR = os.path.join("scripts", "nautilus")
PERSISTENT_STORAGE_TEMPLATE_FILEPATH = os.path.join(SCRIPT_DIR, "persistent_storage_template.yaml")
PERSISTENT_STORAGE_OUTPUT_FILEPATH = os.path.join(SCRIPT_DIR, "persistent_storage.yaml")


def main():
    with open(PERSISTENT_STORAGE_TEMPLATE_FILEPATH, "r") as f:
        yaml_dict = yaml.load(f, Loader)

    unconditional_yaml_dict = replace_dict_str_values_unconditionally(
        yaml_dict,
        unconditional_key_value_replacements={"$USER": USERNAME}
    )
    conditional_yaml_dict = replace_dict_str_values_conditionally(
        unconditional_yaml_dict,
        conditional_key_value_replacements={"storage": STORAGE_SIZE}
    )

    with open(PERSISTENT_STORAGE_OUTPUT_FILEPATH, "w") as f:
        yaml.dump(conditional_yaml_dict, f, Dumper)


if __name__ == "__main__":
    main()
