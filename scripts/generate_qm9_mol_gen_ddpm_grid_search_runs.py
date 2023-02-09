# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import itertools
import json


# define constants #
TASK = "qm9_mol_gen_ddpm"  # TODO: Ensure Is Correct Before Each Grid Search!
SCRIPT_DIR = os.path.join("scripts")
SEARCH_SPACE_FILEPATH = os.path.join(SCRIPT_DIR, f"{TASK}_grid_search_runs.json")


def main():
    # TODO: Ensure Is Correct Before Each Grid Search!
    search_space_dict = {
        "gcp_version": [2],
        "key_names": ["NEL NML LR WD DO CHD NT C"],
        "model.model_cfg.num_encoder_layers": [9],
        "model.layer_cfg.mp_cfg.num_message_layers": [4],
        "model.optimizer.lr": [1e-4],
        "model.optimizer.weight_decay": [1e-12],
        "model.model_cfg.dropout": [0.0],
        "model.model_cfg.chi_hidden_dim": [32],
        "model.diffusion_cfg.num_timesteps": [1000],
        "model.module_cfg.conditioning": ["[]"]
    }

    # gather all combinations of hyperparameters while retaining field names for each chosen hyperparameter
    keys, values = zip(*search_space_dict.items())
    hyperparameter_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # save search space to storage as JSON file
    with open(SEARCH_SPACE_FILEPATH, "w") as f:
        f.write(json.dumps(hyperparameter_dicts))


if __name__ == "__main__":
    main()
