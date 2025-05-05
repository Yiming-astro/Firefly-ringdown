import os
import lal
import yaml
import warnings
import numpy as np
from pycbc.types import TimeSeries


def load_pycbc_timeseries(txt_path):
    data = np.loadtxt(txt_path)
    times, strain = data[:, 0], data[:, 1]
    delta_t = times[1] - times[0]
    start_time = times[0]
    strain_series = TimeSeries(strain, delta_t=delta_t, epoch=start_time)
    return strain_series


def load_config(config_path, type):

    # load the config
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if type != config["sampling_type"]:
        raise ValueError(
            f"Configuration error: 'sampling_type' in yaml file does not correspond with Python file executed. Please check."
        )

    detectors = config.get("det_name", [])
    for det in detectors:
        if det not in config.get("strain_path", {}):
            raise ValueError(
                f"Configuration error: no strain_path specified for detector '{det}'."
            )
        if det not in config.get("acf_path", {}):
            raise ValueError(
                f"Configuration error: no acf_path specified for detector '{det}'."
            )

    config["acf"] = {}
    for det in detectors:
        acf = load_pycbc_timeseries(
            txt_path=config["acf_path"][det],
        )
        if acf.sample_times[-1] < config["duration_time"]:
            raise ValueError(
                f"Configuration error: the acf for detector '{det}' is shorter than the specified duration_time {config['duration_time']}s."
            )
        config["acf"][det] = acf

    rd_analysis_start_time = (
        config["coalescence_time"]
        + config["delta_t"] * config["final_mass"] * lal.MTSUN_SI
    )
    rd_analysis_end_time = rd_analysis_start_time + config["duration_time"]

    config["rd_analysis_start_time_geocentric"] = rd_analysis_start_time
    # TODO (yiming) : for multiple detectors, perform the conversion between geocentric time and detector time.
    config["strain"] = {}
    config["rd_analysis_time"] = {}
    for det in detectors:
        strain = load_pycbc_timeseries(
            txt_path=config["strain_path"][det],
        )
        if (strain.sample_times[0] - rd_analysis_start_time) > (
            1 / config["sampling_frequency"]
        ) or (rd_analysis_end_time - strain.sample_times[-1]) > (
            1 / config["sampling_frequency"]
        ):
            raise ValueError(
                f"Configuration error: the strain data for detector '{det}' does not support analysis of the {rd_analysis_start_time}-{rd_analysis_end_time}s segment."
            )
        config["strain"][det] = strain
        config["rd_analysis_time"][det] = {}
        config["rd_analysis_time"][det][
            "rd_analysis_start_time"
        ] = rd_analysis_start_time
        config["rd_analysis_time"][det]["rd_analysis_end_time"] = rd_analysis_end_time

    lmns = config.get("lmns", [])
    for mode in lmns:
        if not (isinstance(mode, str) and len(mode) == 3 and mode.isdigit()):
            raise ValueError(
                f"Configuration error: invalid lmns format '{mode}'. lmns should be written as 'lmN' (e.g., '221'), other formats are not supported currently."
            )
        if mode[-1] == "0":
            raise ValueError(
                f"Configuration error: invalid lmns format: '{mode}'. The overtone number N should not be 0; if you wish to include only the fundamental mode, please set N=1."
            )

    if not (
        config["prior_setting"]["final_mass"]["min"]
        <= config["final_mass"]
        <= config["prior_setting"]["final_mass"]["max"]
    ):
        warnings.warn(
            f"Configuration warning: specified prior {config['prior_setting']['final_mass']['min']}-{config['prior_setting']['final_mass']['max']} does not include the estimated final mass {config['final_mass']}."
        )

    if config["prior_setting"]["amplitude"]["type"] not in ["flat", "quadrature_flat"]:
        raise ValueError(
            f"Configuration error: currently, only 'flat' and 'quadrature_flat' amplitude prior forms are supported."
        )

    # check sampling configuration
    if config["sampling_algorithm"] not in ["nested_sampling", "mcmc"]:
        raise ValueError(
            f"Configuration error: unsupported sampling_algorithm '{config['sampling_algorithm']}' (should be 'nested_sampling' or 'mcmc')."
        )
    if config["sampling_algorithm"] in ["nested_sampling"]:
        if config["nested_sampling"]["sampler"] not in ["dynesty"]:
            raise ValueError(
                f"Configuration error: currently, only 'dynesty' is supported for nested_sampling sampler."
            )
    else:
        if config["mcmc"]["sampler"] not in ["bilby_mcmc"]:
            raise ValueError(
                f"Configuration error: currently, only 'bilby_mcmc' is supported for mcmc sampler."
            )

    print("Configuration passed consistency test.")

    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])

    return config
