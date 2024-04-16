# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import hydra
import pyrootutils

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from omegaconf import DictConfig

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from src.analysis.inference_analysis import calculate_mean_and_conf_int
from src.utils.pylogger import get_pylogger
log = get_pylogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="../../configs/analysis",
    config_name="bust_analysis.yaml",
)
def main(cfg: DictConfig):
    """Compare the bust results of generated molecules from two separate generative model checkpoints.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    method_1_bust_results = pd.read_csv(cfg.method_1_bust_results_filepath)
    method_2_bust_results = pd.read_csv(cfg.method_2_bust_results_filepath)

    assert cfg.bust_column_name in method_1_bust_results.columns, f"{cfg.bust_column_name} not found in {cfg.method_1_bust_results_filepath}"
    assert cfg.bust_column_name in method_2_bust_results.columns, f"{cfg.bust_column_name} not found in {cfg.method_2_bust_results_filepath}"

    # Add a source column to distinguish between datasets
    method_1_bust_results["source"] = cfg.method_1
    method_2_bust_results["source"] = cfg.method_2

    # Select only the requested column as well as the source column
    method_1_data = method_1_bust_results[[cfg.bust_column_name, "source"]]
    method_2_data = method_2_bust_results[[cfg.bust_column_name, "source"]]

    if cfg.verbose:
        method_1_column_mean, method_1_column_conf_int = calculate_mean_and_conf_int(method_1_data[cfg.bust_column_name][~np.isnan(method_1_data[cfg.bust_column_name])])
        method_2_column_mean, method_2_column_conf_int = calculate_mean_and_conf_int(method_2_data[cfg.bust_column_name][~np.isnan(method_2_data[cfg.bust_column_name])])
        log.info(f"Mean of {cfg.bust_column_name} for {cfg.method_1}: {method_1_column_mean} ± {(method_1_column_conf_int[1] - method_1_column_mean)}")
        log.info(f"Mean of {cfg.bust_column_name} for {cfg.method_2}: {method_2_column_mean} ± {(method_2_column_conf_int[1] - method_2_column_mean)}")

    # Combine the data
    combined_data = pd.concat([method_1_data, method_2_data], ignore_index=True)

    # Plotting
    ax = sns.boxplot(x="source", y=cfg.bust_column_name, data=combined_data)
    ax.set_ylim(0, 10)
    plt.xlabel("Method")
    plt.ylabel(f"{cfg.bust_column_name.title()}")
    plt.savefig(cfg.bust_analysis_plot_filepath, dpi=300)

    if cfg.verbose:
        log.info("Bust analysis completed")


if __name__ == "__main__":
    main()
