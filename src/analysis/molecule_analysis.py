# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import copy
import glob
import hydra
import os
import pyrootutils
import subprocess

import pandas as pd

from omegaconf import DictConfig, open_dict
from pathlib import Path
from posebusters import PoseBusters
from tqdm import tqdm
from typing import Optional

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from src.utils.pylogger import get_pylogger
log = get_pylogger(__name__)


def convert_xyz_to_sdf(input_xyz_filepath: str) -> Optional[str]:
    """Convert an XYZ file to an SDF file using OpenBabel.

    :param input_xyz_filepath: Input XYZ file path.
    :return: Output SDF file path.
    """
    output_sdf_filepath = input_xyz_filepath.replace(".xyz", ".sdf")
    if not os.path.exists(output_sdf_filepath):
        subprocess.run(
            [
                "obabel",
                input_xyz_filepath,
                "-O",
                output_sdf_filepath,
            ],
            check=True,
        )
    return output_sdf_filepath if os.path.exists(output_sdf_filepath) else None


def create_molecule_table(input_molecule_dir: str) -> pd.DataFrame:
    """Create a molecule table from the inference results of a trained model checkpoint.

    :param input_molecule_dir: Directory containing the generated molecules of a trained model checkpoint.
    :return: Molecule table as a Pandas DataFrame.
    """
    inference_xyz_results = [str(item) for item in Path(input_molecule_dir).rglob("*.xyz")]
    inference_sdf_results = [str(item) for item in Path(input_molecule_dir).rglob("*.sdf")]
    if not inference_sdf_results or len(inference_sdf_results) != len(inference_xyz_results):
        inference_sdf_results = [
            convert_xyz_to_sdf(item) for item in tqdm(
                inference_xyz_results, desc="Converting XYZ input files to SDF files"
            )
        ]
    mol_table = pd.DataFrame(
        {
            "mol_pred": [item for item in inference_sdf_results if item is not None],
            "mol_true": None,
            "mol_cond": None,
        }
    )
    return mol_table


def run_unconditional_molecule_analysis(cfg: DictConfig):
    """
    Run molecule analysis for an unconditional method.
    
    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    input_molecule_subdirs = os.listdir(cfg.input_molecule_dir)
    assert cfg.sampling_index is None or (cfg.sampling_index is not None and cfg.sampling_index < len(input_molecule_subdirs)), "The given sampling index is out of range."

    sampling_index = 0
    for item in input_molecule_subdirs:
        sampling_dir = os.path.join(cfg.input_molecule_dir, item)
        if os.path.isdir(sampling_dir):
            if cfg.sampling_index is not None and sampling_index != cfg.sampling_index:
                sampling_index += 1
                continue
            log.info(f"Processing sampling {sampling_index} corresponding to {sampling_dir}...")
            bust_results_filepath = cfg.bust_results_filepath.replace(".csv", f"_{sampling_index}.csv")
            mol_table = create_molecule_table(sampling_dir)
            buster = PoseBusters(config="mol", top_n=None)
            bust_results = buster.bust_table(mol_table, full_report=cfg.full_report)
            bust_results.to_csv(bust_results_filepath, index=False)
            log.info(f"PoseBusters results for sampling {sampling_index} saved to {bust_results_filepath}.")
            sampling_index += 1


def run_conditional_molecule_analysis(cfg: DictConfig):
    """
    Run molecule analysis for a property-conditional method.
    
    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    log.info(f"Processing sampling directory {cfg.input_molecule_dir}...")
    for subdir in glob.glob(os.path.join(cfg.input_molecule_dir, "*")):
        if not os.path.isdir(subdir):
            continue
        log.info(f"Processing sampling directory {subdir}...")
        seed = subdir.split("_")[-1]
        mol_table = create_molecule_table(subdir)
        buster = PoseBusters(config="mol", top_n=None)
        bust_results = buster.bust_table(mol_table, full_report=cfg.full_report)
        bust_results.to_csv(cfg.bust_results_filepath.replace(".csv", f"_seed_{seed}.csv"), index=False)
        log.info(f"PoseBusters results for sampling directory {cfg.input_molecule_dir} saved to {cfg.bust_results_filepath}.")


@hydra.main(
    version_base="1.3",
    config_path="../../configs/analysis",
    config_name="molecule_analysis.yaml",
)
def main(cfg: DictConfig):
    """Analyze the generated molecules from a trained model checkpoint.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    os.makedirs(Path(cfg.bust_results_filepath).parent, exist_ok=True)

    if cfg.model_type == "Unconditional":
        run_unconditional_molecule_analysis(cfg)
    elif cfg.model_type == "Conditional":
        with open_dict(cfg):
            input_molecule_dir = copy.deepcopy(cfg.input_molecule_dir)
            bust_results_filepath = copy.deepcopy(cfg.bust_results_filepath)
            cfg.property = cfg.property.replace("_", "")
            cfg.input_molecule_dir = input_molecule_dir
            cfg.bust_results_filepath = bust_results_filepath
        assert cfg.property in ["alpha", "gap", "homo", "lumo", "mu", "Cv"], "The given property is not supported."
        run_conditional_molecule_analysis(cfg)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model_type}")


if __name__ == "__main__":
    main()
