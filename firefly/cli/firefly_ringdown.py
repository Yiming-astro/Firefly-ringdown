import os

os.environ["PYCBC_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import bilby
import datetime
import argparse
import numpy as np
import scipy.linalg as sl

from gwpy.timeseries import TimeSeries as gtts
from bilby.gw.prior import PriorDict
from bilby.core.prior import Uniform
from bilby.gw.detector import InterferometerList
from bilby.core.utils import setup_logger, logger

from firefly.utils.load_config import load_config
from firefly.utils.ringdown_signal import QNMs_lmn
from firefly.likelihood.firefly_likelihood import (
    TD_WaveformGenerator,
    Auxiliary_likelihood,
)
from firefly.utils.resampling import firefly_resampling
from firefly.utils.basic import firefly_corner_plot


def main(config):

    print("Firefly parameter estimation for {} starts.".format(config["label"]))

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
        strain_rd = strain[ifo.name].time_slice(
            rd_analysis_time[ifo.name]["rd_analysis_start_time"],
            rd_analysis_time[ifo.name]["rd_analysis_end_time"],
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

    # setting priors for auxiliary inference
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

    # setting waveform generator
    waveform_arguments = {
        "lmns": config["lmns"],
        "delta_t": 1.0 / config["sampling_frequency"],
        "harmonics": "spherical",
        "model": "pykerr",
    }
    logger.info("The waveform_arguments is: %s" % str(waveform_arguments))
    # TODO (yiming) : check for the duration
    waveform_generator = TD_WaveformGenerator(
        duration=config["duration_time"],
        sampling_frequency=config["sampling_frequency"],
        time_domain_source_model=QNMs_lmn,
        waveform_arguments=waveform_arguments,
    )

    # setting the likelihood
    # TODO (yiming) : currently, we are using sky_average and combined the ET factor here
    likelihood = Auxiliary_likelihood(
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
        **{"quantiles": [0.05, 0.95]},
    )

    # calculation the evidence in quadrature_flat_prior form
    # TODO (yiming) : change the amp_max_list to suit dict input
    amp_prior_max_list = np.array(
        [config["prior_setting"]["amplitude"]["max"]] * len(lmn_all)
    )
    log_evidence_quadrature_flat_prior = result.log_evidence - np.sum(
        [np.log(np.pi * Amp_i_Max**2) for Amp_i_Max in amp_prior_max_list]
    )
    logger.info(
        f"log_evidence under quadrature-flat prior: {log_evidence_quadrature_flat_prior.tolist()}"
    )

    # TODO (yiming) : 如果设定中prior form是正则平，后面应该跳过
    # resampling for prior transfer
    posterior_resampling = firefly_resampling(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        acfs=cov_matrix,
        priors=priors,
        sky_average=1.5 / np.sqrt(5),
        n_MC=config["resampling_setting"]["n_MC"],
        n_w=config["resampling_setting"]["n_w"],
        n_target=config["resampling_setting"]["n_target"],
        n_QNM=config["resampling_setting"]["n_QNM"],
        n_queue=config["resampling_setting"]["n_queue"],
    )

    # calculate evidence in amplitude_flat_prior form
    start_time = datetime.datetime.now()
    (
        mean_log_evidence,
        std_mean_log_evidence,
        full_params_posterior_in_quadrature_flat_prior,
    ) = posterior_resampling.evidence_calculation(
        result.posterior,
        result.log_evidence,
        amp_prior_max_list,
        save_posterior_in_quadrature_flat_prior=config["save_setting"][
            "save_posterior_in_quadrature_flat_prior"
        ],
    )
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    logger.info(
        f"log_evidence under amplitude-flat prior (mean_value // std_in_MC_calculation): {mean_log_evidence} // {std_mean_log_evidence}"
    )
    logger.info(f"Total Evidence Calculation time: {execution_time}")

    if config["save_setting"]["save_posterior_in_quadrature_flat_prior"]:
        posterior_resampling.save_posterior(
            full_params_posterior_in_quadrature_flat_prior,
            outdir=config["save_path"],
            file_name=config["label"] + "_quadrature_flat_prior.csv",
        )
        logger.info(
            "Full-parameter posteriors in quadrature_flat_prior are saved in {}".format(
                os.path.join(
                    config["save_path"], config["label"] + "_quadrature_flat_prior.csv"
                )
            )
        )

        firefly_corner_plot(
            full_params_posterior_in_quadrature_flat_prior,
            excluded_keys=["geocent_time", "azimuthal", "inclination"],
            outdir=config["save_path"],
            file_name=config["label"] + "_quadrature_flat_prior.png",
            legend_text="firefly (quadrature flat prior)",
        )
        logger.info(
            "Corner plot of Full-parameter posteriors in quadrature_flat_prior is saved in {}".format(
                os.path.join(
                    config["save_path"], config["label"] + "_quadrature_flat_prior.png"
                )
            )
        )

    # perform two-step importance sampling to get posterior under amplitude flat prior
    start_time = datetime.datetime.now()
    full_params_posterior_in_amplitude_flat_prior = (
        posterior_resampling.posterior_importance_sampling(result.posterior)
    )
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    logger.info(
        f"Total two-step importance sampling calculation time: {execution_time}"
    )

    posterior_resampling.save_posterior(
        full_params_posterior_in_amplitude_flat_prior,
        outdir=config["save_path"],
        file_name=config["label"] + "_amplitude_flat_prior.csv",
    )

    logger.info(
        "Full-parameter posteriors in amplitude_flat_prior are saved in {}".format(
            os.path.join(
                config["save_path"], config["label"] + "_amplitude_flat_prior.csv"
            )
        )
    )

    firefly_corner_plot(
        full_params_posterior_in_amplitude_flat_prior,
        excluded_keys=["geocent_time", "azimuthal", "inclination"],
        outdir=config["save_path"],
        file_name=config["label"] + "_amplitude_flat_prior.png",
        legend_text="firefly (amplitude flat prior)",
    )
    logger.info(
        "Corner plot of Full-parameter posteriors in amplitude_flat_prior is saved in {}".format(
            os.path.join(
                config["save_path"], config["label"] + "_amplitude_flat_prior.png"
            )
        )
    )

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Ringdown analysis with firefly")
    parser.add_argument(
        "--config_path", type=str, required=True, help="The path of yaml configuration"
    )
    args = parser.parse_args()
    config_path = args.config_path
    config = load_config(config_path, type="firefly")
    main(config)
