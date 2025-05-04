import os
import corner

config = dict(
    show_titles=True,
    title_fmt=".2f",
    bins=30,
    smooth=1,
    color="#0072C1",
    truth_color="tab:orange",
    quantiles=[0.05, 0.5, 0.95],
    levels=[0.5, 0.68, 0.9],
    labelpad=0.1,
    plot_density=False,
    plot_datapoints=True,
    fill_contours=True,
    max_n_ticks=3,
    title_kwargs={"fontsize": 12},
    hist_kwargs={"density": True, "alpha": 0.5},
    contour_kwargs={"linewidths": 1.5},
)


def firefly_corner_plot(
    firefly_posterior, excluded_keys, outdir, file_name, legend_text=None
):
    # plot the corner with firefly_posterior
    keys, samples = firefly_posterior["param_keys"], firefly_posterior["samples"]
    filtered_keys = [key for key in keys if key not in excluded_keys]
    filtered_index = [keys.tolist().index(key) for key in filtered_keys]
    filtered_samples = samples[:, filtered_index]
    figure = corner.corner(filtered_samples, labels=filtered_keys, **config)
    if legend_text:
        figure.text(0.6, 0.9, legend_text, color="#0072C1", fontsize=20)
    figure.savefig(os.path.join(outdir, file_name), dpi=300, bbox_inches="tight")

    return


def bilby_corner_plot(
    bilby_posterior, excluded_keys, outdir, file_name, legend_text=None
):
    # plot the corner with bilby_posterior
    filtered_keys = [key for key in bilby_posterior.keys() if key not in excluded_keys]
    filtered_index = [
        bilby_posterior.keys().tolist().index(key) for key in filtered_keys
    ]
    filtered_samples = bilby_posterior.values[:, filtered_index]
    figure = corner.corner(filtered_samples, labels=filtered_keys, **config)
    if legend_text:
        figure.text(0.6, 0.9, legend_text, color="#0072C1", fontsize=20)
    figure.savefig(os.path.join(outdir, file_name), dpi=300, bbox_inches="tight")
    return
