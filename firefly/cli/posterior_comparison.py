import os
import yaml
import bilby
import corner
import argparse
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": True,
        "axes.labelsize": 28,
        "axes.titlesize": 32,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "figure.figsize": (10, 6),
        "figure.autolayout": False,
    }
)


def main(config):

    lmns = config["lmns"]
    file_Firefly = config["firefly_posterior_path"]
    file_Fullparams = config["fullparams_posterior_path"]
    save_path = config["save_path"]
    ranges = config["ranges"]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    lmn_all = [
        "%s%d" % (lmn[:2], n) for lmn in lmns for n in range(int("%s" % lmn[-1]))
    ]
    keys = ["final_mass", "final_spin"]
    for lmn in lmn_all:
        keys.append(f"amp{lmn}")
        keys.append(f"phi{lmn}")
    labels = [r"$M_f\,[M_{\odot}]$", r"$\chi_f$"]
    for lmn in lmn_all:
        labels.append(r"$A_{{{}}}\,[10^{{-20}}]$".format(lmn))
        labels.append(r"$\phi_{{{}}}$".format(lmn))

    data_Firefly = pd.read_csv(file_Firefly)
    Firefly_samples = data_Firefly[keys].values
    result_fullparams = bilby.result.read_in_result(filename=file_Fullparams)
    fullparams_samples = result_fullparams.posterior.values[:, np.arange(len(keys))]

    corner_config = dict(
        show_titles=True,
        title=["" for _ in range(len(labels))],
        title_kwargs={
            "pad": 12,
            "fontsize": 27,
            "color": "#0072C1",
        },
        title_fmt=".1f",
        bins=40,
        smooth=1,
        color="#0072C1",
        truth_color="goldenrod",
        labels=labels,
        levels=[1 - np.exp(-(i**2) / 2.0) for i in [1.0, 1.5, 2.0, 3.0]],
        labelpad=0.1,
        plot_density=False,
        plot_datapoints=False,
        fill_contours=True,
        max_n_ticks=3,
        hist_kwargs={"density": True, "lw": 1.2},
    )
    if ranges not in ["None"]:
        corner_config["range"] = ranges

    plot_fig = corner.corner(
        fullparams_samples,
        **{
            **corner_config,
            **dict(
                plot_datapoints=True,
                hist_kwargs={"density": True, "color": "#0072C1"},
                contour_kwargs={"linewidths": 0.0},
            ),
        },
    )
    plot_fig = corner.corner(
        Firefly_samples,
        fig=plot_fig,
        **{
            **corner_config,
            **dict(
                fill_contours=False,
                no_fill_contours=True,
                show_titles=False,
                color="forestgreen",
                hist_kwargs={"density": True, "color": "forestgreen"},
                contour_kwargs={"linewidths": 1.8, "linestyles": "dashed"},
            ),
        },
    )

    for i, par in enumerate(labels):
        ax = plot_fig.axes[i + i * len(labels)]
        title_text = ax.title.get_text()
        cleaned_title_text = title_text.split("=")
        cleaned_title_text = cleaned_title_text[-1]
        ax.set_title(cleaned_title_text, fontsize=26, pad=10, color="#0072C1")

    plot_fig.text(
        0.55,
        0.9,
        "FIREFLY",
        usetex=False,
        fontname="Arial",
        color="forestgreen",
        fontsize=20,
    )
    plot_fig.text(
        0.55,
        0.85,
        "Full-parameter Sampling",
        usetex=False,
        fontname="Arial",
        color="#0072C1",
        fontsize=20,
    )

    corner_save_path = os.path.join(save_path, "posterior.pdf")
    plt.savefig(corner_save_path, bbox_inches="tight")
    print(f"save to {corner_save_path}.")

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Posterior plot for comparison")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/posterior_comparison.yaml",
        help="The path to the config_file",
    )
    args = parser.parse_args()
    with open(args.config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    main(config)
