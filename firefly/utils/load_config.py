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
    if config["sampling_algorithm"] not in ["nested_sampling", "mcmc", "nessai"]:
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


def load_config_signal_injections(config_path):

    # load the config
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # lmns check
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
    lmn_all = [
        "%s%d" % (lmn[:2], n) for lmn in lmns for n in range(int("%s" % lmn[-1]))
    ]
    config["lmn_all"] = lmn_all

    # detectors check
    detectors = config.get("det_name", [])
    if config["add_noise"]:
        for det in detectors:
            if det not in config.get("psd_path", {}):
                raise ValueError(
                    f"Configuration error: no psd_path specified for detector '{det}'."
                )

    # check injection parameters
    for lmn in lmn_all:
        amp_key = "amp" + lmn
        phi_key = "phi" + lmn
        if (
            amp_key not in config["injection_params"].keys()
            or phi_key not in config["injection_params"].keys()
        ):
            raise ValueError(
                f"Configuration error: injected parameters for `{lmn}` are missing; injected_params should include `{amp_key}` and `{phi_key}`."
            )

    print("Configuration passed consistency test.")

    strain_save_path = os.path.join(config["save_path"], config["label"])
    config["strain_save_path"] = strain_save_path
    if not os.path.exists(strain_save_path):
        os.makedirs(strain_save_path)

    return config
