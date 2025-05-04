import os

os.environ["PYCBC_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import bilby
import argparse
import numpy as np
import scipy.linalg as sl

from gwpy.timeseries import TimeSeries as gtts
from bilby.gw.prior import PriorDict
from bilby.core.prior import Uniform, PowerLaw
from bilby.gw.detector import InterferometerList
from bilby.core.utils import setup_logger, logger

from firefly.utils.load_config import load_config
from firefly.likelihood.fullparams_likelihood import (
    TD_WaveformGenerator,
    Full_params_likelihood,
)
from firefly.utils.ringdown_signal import Pycbc_ringdown_lmn
from firefly.utils.basic import bilby_corner_plot


def main(config):

    print("Fullparams parameter estimation for {} starts.".format(config["label"]))

    # setting the strain and covariance matrix
    ifos = InterferometerList(config["det_name"])
    length_cov = int(config["duration_time"] * config["sampling_frequency"])
    acf = config["acf"]
    strain = config["strain"]
    rd_analysis_time = config["rd_analysis_time"]
    lmn_all = [
        "%s%d" % (lmn[:2], n)
        for lmn in config["lmns"]
        for n in range(int("%s" % lmn[-1]))
    ]
    cov_matrix = {}
    for ifo in ifos:
        # TODO (yiming) : here we add 0.01s for corner case, rethink it.
        strain_rd = strain[ifo.name].time_slice(
            rd_analysis_time[ifo.name]["rd_analysis_start_time"],
            rd_analysis_time[ifo.name]["rd_analysis_end_time"] + 0.01,
        )[:length_cov]
        ifo_data = gtts(
            strain_rd.numpy(),
            sample_rate=strain_rd.get_sample_rate(),
            times=strain_rd.sample_times.numpy(),
        )
        ifo.set_strain_data_from_gwpy_timeseries(ifo_data)
        covariance_cholesky = sl.cholesky(
            sl.toeplitz(acf[ifo.name][:length_cov]), lower=True
        )
        cov_matrix[ifo.name] = sl.solve_triangular(
            covariance_cholesky, np.eye(covariance_cholesky.shape[0]), lower=True
        )

    # recording the information in logging
    setup_logger(outdir=config["save_path"], label=config["label"])
    logger.info("The sampling algorithm is {}".format(config["sampling_type"]))
    logger.info("The strain data are loaded from: {}".format(config["strain_path"]))
    logger.info("The ACFs are loaded from: {}".format(config["acf_path"]))
    logger.info("The analyzed time for strain data: {}".format(rd_analysis_time))
    logger.info("The modes (lmn) considered are: {}".format(lmn_all))
    IMR_paramters = {}
    IMR_paramters["inclination"] = config["inclination"]
    IMR_paramters["phase"] = config["phase"]
    logger.info("The parameters determined by IMR are :{}".format(IMR_paramters))
    logger.info(
        "The estimated final mass is :{} (solar mass)".format(config["final_mass"])
    )

    # setting priors for fullparams inference
    priors = PriorDict()
    priors["geocent_time"] = config["rd_analysis_start_time_geocentric"]
    priors["azimuthal"] = 0.0
    priors["inclination"] = config["inclination"]
    priors["final_mass"] = Uniform(
        name="final_mass",
        minimum=config["prior_setting"]["final_mass"]["min"],
        maximum=config["prior_setting"]["final_mass"]["max"],
        unit="$M_{\\odot}$",
        latex_label="$M_f$",
    )
    priors["final_spin"] = Uniform(
        name="final_spin",
        minimum=config["prior_setting"]["final_spin"]["min"],
        maximum=config["prior_setting"]["final_spin"]["max"],
        latex_label="$\chi_f$",
    )
    for lmn in lmn_all:
        if config["prior_setting"]["amplitude"]["type"] == "quadrature_flat":
            priors["amp%s" % lmn] = PowerLaw(
                name="amp%s" % lmn,
                alpha=1,
                minimum=0.0,
                maximum=config["prior_setting"]["amplitude"]["max"],
                latex_label="$amp%s\,(\\times 10^{-20})$" % lmn,
            )
        elif config["prior_setting"]["amplitude"]["type"] == "flat":
            priors["amp%s" % lmn] = Uniform(
                name="amp%s" % lmn,
                minimum=0.0,
                maximum=config["prior_setting"]["amplitude"]["max"],
                latex_label="$amp%s\,(\\times 10^{-20})$" % lmn,
            )
        else:
            raise ValueError(
                "Prior error: unsupported prior type '{}'. Currently it should be 'quadrature_flat' or 'flat'.".format(
                    config["prior_setting"]["amplitude"]["type"]
                )
            )
        priors["phi%s" % lmn] = Uniform(
            name="phi%s" % lmn,
            minimum=0.0,
            maximum=2 * np.pi,
            latex_label="$\phi_%s$" % lmn,
            boundary="periodic",
        )

    # setting waveform generator
    waveform_arguments = {
        "lmns": config["lmns"],
        "delta_t": 1.0 / config["sampling_frequency"],
        "t_final": config["duration_time"],
    }
    logger.info("the waveform_arguments is: %s" % str(waveform_arguments))
    # TODO (yiming) : check for the duration
    waveform_generator = TD_WaveformGenerator(
        duration=config["duration_time"],
        sampling_frequency=config["sampling_frequency"],
        time_domain_source_model=Pycbc_ringdown_lmn,
        waveform_arguments=waveform_arguments,
    )

    # setting the likelihood
    # TODO (yiming) : currently, we are using sky_average and combined the ET factor here
    likelihood = Full_params_likelihood(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        acfs=cov_matrix,
        priors=priors,
        sky_average=1.5 / np.sqrt(5),
    )

    if config["sampling_algorithm"] in ["nested_sampling"]:
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler=config["nested_sampling"]["sampler"],
            nlive=config["nested_sampling"]["nlive"],
            queue_size=config["nested_sampling"]["queue_size"],
            outdir=config["save_path"],
            label=config["label"],
            dlogz=config["nested_sampling"]["dlogz"],
            resume=config["nested_sampling"]["resume"],
        )
    else:
        # TODO (yiming) : implement mcmc
        raise ValueError("mcmc not implement.")

    # plot the posteriors
    result.plot_corner(
        parameters=["final_mass", "final_spin"],
        filename=os.path.join(
            config["save_path"], "%s_part_corner.png" % config["label"]
        ),
        **{"quantiles": [0.05, 0.95]}
    )

    # plot the full-parameter corner
    bilby_corner_plot(
        result.posterior,
        excluded_keys=[
            "geocent_time",
            "azimuthal",
            "inclination",
            "log_likelihood",
            "log_prior",
        ],
        outdir=config["save_path"],
        file_name=config["label"] + "_fullparams_corner.png",
        legend_text="fullparams",
    )
    logger.info(
        "Corner plot of Full-parameter posteriors is saved in {}".format(
            os.path.join(
                config["save_path"], config["label"] + "_amplitude_flat_prior.png"
            )
        )
    )

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Ringdown analysis with fullparams")
    parser.add_argument(
        "--config_path", type=str, required=True, help="The path of yaml configuration"
    )
    args = parser.parse_args()
    config_path = args.config_path
    config = load_config(config_path, type="fullparams")
    main(config)
