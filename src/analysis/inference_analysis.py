# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import glob
import hydra
import pyrootutils

import numpy as np
import pandas as pd
import scipy.stats as st

from omegaconf import DictConfig
from typing import Iterable

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from src.utils.pylogger import get_pylogger
log = get_pylogger(__name__)


def calculate_mean_and_conf_int(data: Iterable, alpha: float = 0.95) -> tuple[float, tuple[float, float]]:
    """
    Calculate and report the mean and confidence interval of the data.
    
    :param data: Iterable data to calculate the mean and confidence interval.
    :param alpha: Confidence level (default: 0.95).
    :return: Tuple of the mean and confidence interval.
    """
    conf_int = st.t.interval(
        confidence=alpha,
        df=len(data) - 1,
        loc=np.mean(data),
        scale=st.sem(data),
    )
    return np.mean(data), conf_int


def run_unconditional_inference_analysis(cfg: DictConfig):
    """
    Run inference analysis for an unconditional method.
    
    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    pb_results_filepaths = glob.glob(cfg.bust_results_filepath.replace(".csv", "*.csv"))
    assert pb_results_filepaths, f"PoseBusters bust results file(s) not found: {cfg.bust_results_filepath}"

    atoms_stable_pct_list = [
        # TODO: manually add stable atom percentages from each run here
    ]
    molecules_stable_pct_list = [
        # TODO: manually add stable molecule percentages from each run here
    ]
    valid_molecules_pct_list = [
        # TODO: manually add valid molecule percentages from each run here
    ]
    unique_molecules_pct_list = [
        # TODO: manually add unique molecule percentages from each run here
    ]
    novel_molecules_pct_list = [
        # TODO: manually add novel molecule percentages from each run here
    ]
    nll_list = [
        # TODO: manually add molecule negative log-likelihoods from each run here
    ]
    assert len(pb_results_filepaths) == 5, "Number of PoseBusters result files does not match number of run results."
    assert len(atoms_stable_pct_list) == len(molecules_stable_pct_list) == len(valid_molecules_pct_list) == len(unique_molecules_pct_list) == len(novel_molecules_pct_list), "Number of run results do not match number of PoseBuster result files."

    # accumulate the percentages
    for index in range(len(unique_molecules_pct_list)):
        unique_molecules_pct_list[index] *= valid_molecules_pct_list[index]
    for index in range(len(novel_molecules_pct_list)):
        novel_molecules_pct_list[index] *= unique_molecules_pct_list[index]

    # calculate the corresponding means and confidence intervals
    if atoms_stable_pct_list:
        atoms_stable_mean, atoms_stable_conf_int = calculate_mean_and_conf_int(atoms_stable_pct_list)
    if molecules_stable_pct_list:
        molecules_stable_mean, molecules_stable_conf_int = calculate_mean_and_conf_int(molecules_stable_pct_list)
    if valid_molecules_pct_list:
        valid_molecules_mean, valid_molecules_conf_int = calculate_mean_and_conf_int(valid_molecules_pct_list)
    if unique_molecules_pct_list:
        unique_molecules_mean, unique_molecules_conf_int = calculate_mean_and_conf_int(unique_molecules_pct_list)
    if novel_molecules_pct_list:
        novel_molecules_mean, novel_molecules_conf_int = calculate_mean_and_conf_int(novel_molecules_pct_list)
    if nll_list:
        nll_mean, nll_conf_int = calculate_mean_and_conf_int(nll_list)

    # report the results
    if atoms_stable_pct_list:
        log.info(f"Mean atoms stable percentage: {atoms_stable_mean * 100} % with confidence interval: ±{(atoms_stable_conf_int[1] - atoms_stable_mean) * 100}")
    if molecules_stable_pct_list:
        log.info(f"Mean molecules stable percentage: {molecules_stable_mean * 100} % with confidence interval: ±{(molecules_stable_conf_int[1] - molecules_stable_mean) * 100}")
    if valid_molecules_pct_list:
        log.info(f"Mean valid molecules percentage: {valid_molecules_mean * 100} % with confidence interval: ±{(valid_molecules_conf_int[1] - valid_molecules_mean) * 100}")
    if unique_molecules_pct_list:
        log.info(f"Mean unique molecules percentage: {unique_molecules_mean * 100} % with confidence interval: ±{(unique_molecules_conf_int[1] - unique_molecules_mean) * 100}")
    if novel_molecules_pct_list:
        log.info(f"Mean novel molecules percentage: {novel_molecules_mean * 100} % with confidence interval: ±{(novel_molecules_conf_int[1] - novel_molecules_mean) * 100}")
    if nll_list:
        log.info(f"Mean molecule negative log-likelihood: {nll_mean} with confidence interval: ±{nll_conf_int[1] - nll_mean}")

    # evaluate and report PoseBusters results
    num_pb_valid_molecules = []
    for pb_results_filepath in pb_results_filepaths:
        pb_results = pd.read_csv(pb_results_filepath)
        pb_results["valid"] = (
            pb_results["mol_pred_loaded"].astype(bool)
            & pb_results["sanitization"].astype(bool)
            & pb_results["all_atoms_connected"].astype(bool)
            & pb_results["bond_lengths"].astype(bool)
            & pb_results["bond_angles"].astype(bool)
            & pb_results["internal_steric_clash"].astype(bool)
            & pb_results["aromatic_ring_flatness"].astype(bool)
            & pb_results["double_bond_flatness"].astype(bool)
            & pb_results["internal_energy"].astype(bool)
            & pb_results["passes_valence_checks"].astype(bool)
            & pb_results["passes_kekulization"].astype(bool)
        )
        num_pb_valid_molecules.append(pb_results["valid"].mean())
    num_pb_valid_molecules_mean, num_pb_valid_molecules_conf_int = calculate_mean_and_conf_int(num_pb_valid_molecules)
    log.info(f"Mean percentage of PoseBusters-valid molecules: {num_pb_valid_molecules_mean * 100} % with confidence interval: ±{(num_pb_valid_molecules_conf_int[1] - num_pb_valid_molecules_mean) * 100}")


def run_conditional_inference_analysis(cfg: DictConfig):
    """
    Run inference analysis for a property-conditional method.
    
    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    pb_results_filepaths = glob.glob(cfg.bust_results_filepath.replace(".csv", "*.csv"))
    assert pb_results_filepaths, f"PoseBusters bust results file(s) not found: {cfg.bust_results_filepath}"

    alpha_mae_list = [
        # TODO: manually add alpha MAE from each run here
    ]
    gap_mae_list = [
        # TODO: manually add gap MAE from each run here
    ]
    homo_mae_list = [
        # TODO: manually add homo MAE from each run here
    ]
    lumo_mae_list = [
        # TODO: manually add lumo MAE from each run here
    ]
    mu_mae_list = [
        # TODO: manually add mu MAE from each run here
    ]
    cv_mae_list = [
        # TODO: manually add Cv MAE from each run here
    ]
    assert len(pb_results_filepaths) == 3, "Number of PoseBusters result files does not match number of property result seeds."
    assert len(alpha_mae_list) == len(gap_mae_list) == len(homo_mae_list) == len(lumo_mae_list) == len(mu_mae_list) == len(cv_mae_list), "Number of run results is inconsistent."

    # calculate the corresponding means and confidence intervals
    if alpha_mae_list:
        alpha_mae_mean, alpha_mae_conf_int = calculate_mean_and_conf_int(alpha_mae_list)
    if gap_mae_list:
        gap_mae_mean, gap_mae_conf_int = calculate_mean_and_conf_int(gap_mae_list)
    if homo_mae_list:
        homo_mae_mean, homo_mae_conf_int = calculate_mean_and_conf_int(homo_mae_list)
    if lumo_mae_list:
        lumo_mae_mean, lumo_mae_conf_int = calculate_mean_and_conf_int(lumo_mae_list)
    if mu_mae_list:
        mu_mae_mean, mu_mae_conf_int = calculate_mean_and_conf_int(mu_mae_list)
    if cv_mae_list:
        cv_mae_mean, cv_mae_conf_int = calculate_mean_and_conf_int(cv_mae_list)

    # report the results
    if alpha_mae_list:
        log.info(f"Mean alpha MAE: {alpha_mae_mean} with confidence interval: ±{alpha_mae_conf_int[1] - alpha_mae_mean}")
    if gap_mae_list:
        log.info(f"Mean gap MAE: {gap_mae_mean} with confidence interval: ±{gap_mae_conf_int[1] - gap_mae_mean}")
    if homo_mae_list:
        log.info(f"Mean homo MAE: {homo_mae_mean} with confidence interval: ±{homo_mae_conf_int[1] - homo_mae_mean}")
    if lumo_mae_list:
        log.info(f"Mean lumo MAE: {lumo_mae_mean} with confidence interval: ±{lumo_mae_conf_int[1] - lumo_mae_mean}")
    if mu_mae_list:
        log.info(f"Mean mu MAE: {mu_mae_mean} with confidence interval: ±{mu_mae_conf_int[1] - mu_mae_mean}")
    if cv_mae_list:
        log.info(f"Mean cv MAE: {cv_mae_mean} with confidence interval: ±{cv_mae_conf_int[1] - cv_mae_mean}")

    # evaluate and report PoseBusters results
    num_pb_valid_molecules = []
    for pb_results_filepath in pb_results_filepaths:
        pb_results = pd.read_csv(pb_results_filepath)
        pb_results["valid"] = (
            pb_results["mol_pred_loaded"].astype(bool)
            & pb_results["sanitization"].astype(bool)
            & pb_results["all_atoms_connected"].astype(bool)
            & pb_results["bond_lengths"].astype(bool)
            & pb_results["bond_angles"].astype(bool)
            & pb_results["internal_steric_clash"].astype(bool)
            & pb_results["aromatic_ring_flatness"].astype(bool)
            & pb_results["double_bond_flatness"].astype(bool)
            & pb_results["internal_energy"].astype(bool)
            & pb_results["passes_valence_checks"].astype(bool)
            & pb_results["passes_kekulization"].astype(bool)
        )
        num_pb_valid_molecules.append(pb_results["valid"].mean())
    num_pb_valid_molecules_mean, num_pb_valid_molecules_conf_int = calculate_mean_and_conf_int(num_pb_valid_molecules)
    log.info(f"Mean percentage of PoseBusters-valid molecules for property {cfg.property.replace('_', '')}: {num_pb_valid_molecules_mean * 100} % with confidence interval: ±{(num_pb_valid_molecules_conf_int[1] - num_pb_valid_molecules_mean) * 100}")


@hydra.main(
    version_base="1.3",
    config_path="../../configs/analysis",
    config_name="inference_analysis.yaml",
)
def main(cfg: DictConfig):
    """Analyze the inference results of a trained model checkpoint.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    if cfg.model_type == "Unconditional":
        run_unconditional_inference_analysis(cfg)
    elif cfg.model_type == "Conditional":
        run_conditional_inference_analysis(cfg)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model_type}")

if __name__ == "__main__":
    main()
