# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

from typing import Any


def parse_ms_value(ms: Any) -> tuple[float, float]:
    """Parse the MS value and its error, return as a tuple."""
    if isinstance(ms, float) or ms == "N/A":
        return (ms, 0)  # No error for single float values or N/A
    ms_parts = ms.split("±")
    value = float(ms_parts[0].strip())
    error = float(ms_parts[1].strip()) if len(ms_parts) > 1 else 0
    return (value, error)


def format_ms_annotation(value: Any, error: float) -> str:
    """Format the MS annotation based on value and error."""
    if value == "N/A":
        return "N/A"
    lower = value - error
    upper = value + error
    return r"$MS \in " + f"[{lower:.1f}\%, {upper:.1f}\%]" + "$"


def main():
    # Define plot data
    data = {
        "Initial Samples (Moderately Stable)": {
            r"$\alpha\ (Bohr^{3})$": {"value": "4.61 ± 0.2", "MS": 61.7},
            r"$\Delta_{\epsilon}\ (meV)$": {"value": "1.26 ± 0.1", "MS": 61.7},
            r"$\epsilon_{HOMO}\ (meV)$": {"value": "0.53 ± 0.0", "MS": 61.7},
            r"$\epsilon_{LUMO}\ (meV)$": {"value": "1.25 ± 0.0", "MS": 61.7},
            r"$\mu\ (D)$": {"value": "1.35 ± 0.1", "MS": 61.7},
            r"$C_{v}\ (\frac{cal}{mol} K)$": {"value": "2.93 ± 0.1", "MS": 61.7},
        },
        "EDM-Opt (100 steps)": {
            r"$\alpha\ (Bohr^{3})$": {"value": "4.45 ± 0.6", "MS": "77.6 ± 2.1"},
            r"$\Delta_{\epsilon}\ (meV)$": {"value": "0.98 ± 0.1", "MS": "80.0 ± 2.0"},
            r"$\epsilon_{HOMO}\ (meV)$": {"value": "0.45 ± 0.0", "MS": "78.8 ± 1.0"},
            r"$\epsilon_{LUMO}\ (meV)$": {"value": "0.91 ± 0.0", "MS": "83.4 ± 4.6"},
            r"$\mu\ (D)$": {"value": "6e5 ± 6e5", "MS": "78.3 ± 2.9"},
            r"$C_{v}\ (\frac{cal}{mol} K)$": {"value": "2.72 ± 2.6", "MS": "51.0 ± 109.7"},
        },
        "EDM-Opt (250 steps)": {
            r"$\alpha\ (Bohr^{3})$": {"value": "1e2 ± 5e2", "MS": "80.1 ± 2.1"},
            r"$\Delta_{\epsilon}\ (meV)$": {"value": "1e3 ± 6e3", "MS": "83.7 ± 3.8"},
            r"$\epsilon_{HOMO}\ (meV)$": {"value": "0.44 ± 0.0", "MS": "82.5 ± 1.3"},
            r"$\epsilon_{LUMO}\ (meV)$": {"value": "0.91 ± 0.1", "MS": "84.7 ± 1.6"},
            r"$\mu\ (D)$": {"value": "2e5 ± 8e5", "MS": "81.0 ± 5.8"},
            r"$C_{v}\ (\frac{cal}{mol} K)$": {"value": "2.15 ± 0.1", "MS": "78.5 ± 3.4"},
        },
        "GCDM-Opt (100 steps)": {
            r"$\alpha\ (Bohr^{3})$": {"value": "3.29 ± 0.1", "MS": "86.2 ± 1.3"},
            r"$\Delta_{\epsilon}\ (meV)$": {"value": "0.93 ± 0.0", "MS": "89.0 ± 1.9"},
            r"$\epsilon_{HOMO}\ (meV)$": {"value": "0.43 ± 0.0", "MS": "91.6 ± 3.5"},
            r"$\epsilon_{LUMO}\ (meV)$": {"value": "0.86 ± 0.0", "MS": "87.0 ± 1.7"},
            r"$\mu\ (D)$": {"value": "1.08 ± 0.1", "MS": "89.9 ± 4.2"},
            r"$C_{v}\ (\frac{cal}{mol} K)$": {"value": "1.81 ± 0.0", "MS": "87.6 ± 1.1"},
        },
        "GCDM-Opt (250 steps)": {
            r"$\alpha\ (Bohr^{3})$": {"value": "3.24 ± 0.2", "MS": "86.6 ± 1.9"},
            r"$\Delta_{\epsilon}\ (meV)$": {"value": "0.93 ± 0.0", "MS": "89.7 ± 2.2"},
            r"$\epsilon_{HOMO}\ (meV)$": {"value": "0.43 ± 0.0", "MS": "90.7 ± 0.0"},
            r"$\epsilon_{LUMO}\ (meV)$": {"value": "0.85 ± 0.0", "MS": "88.6 ± 3.8"},
            r"$\mu\ (D)$": {"value": "1.04 ± 0.0", "MS": "89.5 ± 2.6"},
            r"$C_{v}\ (\frac{cal}{mol} K)$": {"value": "1.82 ± 0.1", "MS": "87.6 ± 2.3"},
        },
    }

    # Prepare data for plotting
    data_groups = {}
    for k, v in data.items():
        values, errors, ms_values = {}, {}, {}
        for prop in v:
            raw_value = float(v[prop]["value"].split("±")[0].strip())
            ms_value, ms_error = parse_ms_value(v[prop]["MS"])
            if raw_value > 50:
                values[prop], errors[prop], ms_values[prop] = np.nan, 0, ("N/A", "N/A")
            else:
                values[prop] = raw_value
                errors[prop] = float(v[prop]["value"].split("±")[1].strip())
                ms_values[prop] = (ms_value, ms_error)
        data_groups[k] = {"values": values, "errors": errors, "MS": ms_values}

    x_labels = list(next(iter(data_groups.values()))["values"].keys())

    fig, ax = plt.subplots(figsize=(10, 8))

    # Adjustments for improved readability
    width = 0.15
    group_gap = 0.5  # Increased for clearer separation
    n_groups = len(data_groups)
    total_width = n_groups * width + (n_groups - 1) * group_gap
    positions = np.arange(len(x_labels)) * (total_width + group_gap)  # Adjusted calculation

    for i, (group, group_data) in enumerate(data_groups.items()):
        values = list(group_data["values"].values())
        errors = list(group_data["errors"].values())
        ms_values = list(group_data["MS"].values())
        bar_positions = [pos + i * (width + group_gap) for pos in positions]
        bars = ax.barh(bar_positions, values, width, label=group, xerr=errors, capsize=2, alpha=0.8, edgecolor="black")

        for j, value in enumerate(values):
            if np.isnan(value):
                # Correctly place an 'x' symbol for missing values
                # Ensure the 'x' marker is at the correct y-axis position corresponding to the missing value's location
                ax.text(0, bar_positions[j], "x", color="red", va="center", ha="center", fontsize=12, weight="bold")

        for bar, (ms, error) in zip(bars, ms_values):
            if not isinstance(ms, str) or ms != "N/A":
                ms_annotation = format_ms_annotation(ms, error)
                ax.annotate(ms_annotation, (bar.get_width(), bar.get_y() + bar.get_height() / 2 + 0.35),
                            textcoords="offset points", xytext=(5, 0), ha="left",
                            fontsize=8, color="darkblue", weight="black")  # Adjusted for readability

    ax.set_ylabel("Task")
    ax.set_xlabel("Property MAE / Molecule Stability (MS) %")
    ax.set_yticks([pos + total_width / 2 - width / 2 for pos in positions])
    ax.set_yticklabels(x_labels, rotation=45, va="center")
    ax.grid(True, which='both', axis='x', linestyle='-.', linewidth=0.5)

    for pos in positions[1:]:
        ax.axhline(y=pos - group_gap / 2, color="black", linewidth=2)  # Make separation lines clearer

    ax.legend(loc="best")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("qm9_property_optimization_results.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
