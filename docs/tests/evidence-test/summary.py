import os
import re
import json
import argparse
import numpy as np
from typing import Sequence
import matplotlib.pyplot as plt


def extract_data_from_log(log_file_path):
    evidence = None
    evidence_error = None

    with open(log_file_path, "r") as file:
        for line in file:
            match = re.search(r"ln_evidence:\s+(-?\d+\.\d+)\s+\+/-\s+(\d+\.\d+)", line)
            if match:
                evidence_error = match.group(2)

            match_evidence = re.search(r"The evidence is: (-?\d+\.\d+)", line)
            if match_evidence:
                evidence = match_evidence.group(1)

    return evidence, evidence_error


def plot_evidence_test(
    ax: plt.Axes,
    x: Sequence[float],
    sample_vals: Sequence[float],
    sample_errs: Sequence[float],
    sample_mean: float,
    analytical_val: float,
    label: str,
) -> None:

    color_TTD = "#0072C1"
    color_analytical = "k"
    capsize = 5
    capthick = 1.5
    elinewidth = 1.4
    linewidthst = 1.35
    marker_size_TD = 60

    z_err = 0
    ax.errorbar(
        x,
        sample_vals,
        yerr=sample_errs,
        fmt="none",
        capsize=capsize,
        capthick=capthick,
        elinewidth=elinewidth,
        color=color_TTD,
        zorder=z_err,
        alpha=0.6,
    )

    z_b = 1
    ax.scatter(x, sample_vals, s=marker_size_TD, marker="o", c="w", zorder=z_b)

    z_m = 2
    ax.scatter(
        x,
        sample_vals,
        color=color_TTD,
        s=marker_size_TD,
        marker="o",
        facecolors="none",
        edgecolors=color_TTD,
        linewidths=linewidthst,
        zorder=z_m,
        alpha=0.6,
    )

    ax.axhline(
        y=sample_mean,
        color=color_TTD,
        alpha=1,
        ls="--",
        lw=1.5,
        zorder=0,
        label="Sampling",
    )
    ax.axhline(
        y=analytical_val,
        color=color_analytical,
        alpha=1,
        ls="-",
        lw=1,
        zorder=0,
        label="Analytical",
    )

    ax.set_ylabel(r"$\ln \mathcal{Z}$", fontsize=28)

    bbox_props = dict(
        boxstyle="round,pad=0.3",
        edgecolor=(0.8, 0.8, 0.8, 0.8),
        facecolor=(1.0, 1.0, 1.0, 0.8),
        alpha=1,
    )
    ax.text(
        0.955,
        0.09,
        label,
        transform=ax.transAxes,
        fontsize=20,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=bbox_props,
    )
    ax.grid(linestyle="--", linewidth=0.8, alpha=0.6)

    return


def main(lmns):
    base_dir = f"results/evidence-test/Case-{lmns}"
    evidence_data = []
    for sub_dir in os.listdir(base_dir):
        sub_dir_path = os.path.join(base_dir, sub_dir)

        if os.path.isdir(sub_dir_path):
            log_file_path = os.path.join(sub_dir_path, "Gaussian_example.log")

            if os.path.exists(log_file_path):
                evidence, evidence_error = extract_data_from_log(log_file_path)
                if evidence is not None and evidence_error is not None:
                    idx = int(sub_dir.split("-")[1])
                    evidence_data.append((idx, evidence, evidence_error))

    evidence_data_sorted = sorted(evidence_data, key=lambda x: x[0])

    evidence_list = [0 for _ in range(10)]
    evidence_error_list = [0 for _ in range(10)]

    print("Evidence:")
    for i, data in enumerate(evidence_data_sorted):
        print(
            f"Folder: Params-{data[0]}, Evidence: {data[1]}, Evidence Error: {data[2]}"
        )
        evidence_list[i] = eval(data[1])
        evidence_error_list[i] = eval(data[2])

    evidence_mean = np.mean(evidence_list)
    print(f"Mean value of estimated evidence is {evidence_mean}.")

    # Load analytical results
    file_path = f"./results/evidence-test/fisher_matrix/Fisher_matrix_{lmns}.jsonl"
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "Please run `fisher_matrix_calculations.py` first to obtain the Fisher matrix."
        )
    with open(file_path, "r", encoding="utf-8") as f:
        record = json.loads(f.readline().strip())
    analytical_evidence = record["Integral"]
    print(f"Analytical result of evidence is {analytical_evidence}.")

    # Plot
    plt.rcParams["text.latex.preamble"] = r"\usepackage{bm}"
    plt.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": True,
            "axes.labelsize": 28,
            "axes.titlesize": 32,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 18,
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "figure.figsize": (10, 6),
            "figure.autolayout": False,
        }
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    x_labels = [
        r"$1$",
        r"$2$",
        r"$3$",
        r"$4$",
        r"$5$",
        r"$6$",
        r"$7$",
        r"$8$",
        r"$9$",
        r"$10$",
    ]
    x = np.arange(len(x_labels))
    plot_evidence_test(
        ax,
        x,
        evidence_list,
        evidence_error_list,
        evidence_mean,
        analytical_evidence,
        r"$N=1$",
    )

    if lmns in ["221"]:
        ax.set_ylim(-11.4, -10.4)
        ax.set_yticks([-11.2, -11.0, -10.8, -10.6])
    if lmns in ["222"]:
        ax.set_ylim(-19.4, -18.4)
        ax.set_yticks([-19.2, -19.0, -18.8, -18.6])
    if lmns in ["223"]:
        ax.set_ylim(-24.0, -23.0)
        ax.set_yticks([-23.8, -23.6, -23.4, -23.2])

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Runs", fontsize=25)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="center",
        bbox_to_anchor=(0.5, 0.87),
    )

    plt.tight_layout()
    save_path = f"./results/evidence-test/recording/Case-{lmns}"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    figure_path = os.path.join(save_path, "evidence_test.png")
    plt.savefig(figure_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gaussian likelihood example")
    parser.add_argument("--lmns", type=str, required=True)
    args = parser.parse_args()
    main(args.lmns)
