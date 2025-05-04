import os

os.environ["PYCBC_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import argparse
import numpy as np
import scipy.linalg as sl

from gwpy.timeseries import TimeSeries as gtts
from bilby.gw.prior import PriorDict
from bilby.core.prior import Uniform
from bilby.gw.detector import InterferometerList

from firefly.utils.load_config import load_config
from firefly.utils.ringdown_signal import QNMs_lmn
from firefly.likelihood.firefly_likelihood import (
    TD_WaveformGenerator,
    Auxiliary_likelihood,
)


def main(config):

    print("Firefly parameter estimation for {} starts.".format(config["label"]))

    # setting the strain and covariance matrix
    ifos = InterferometerList(config["det_name"])
    length_cov = int(config["duration_time"] * config["sampling_frequency"])
    acf = config["acf"]
    strain = config["strain"]
    rd_analysis_time = config["rd_analysis_time"]
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
    IMR_paramters = {}
    IMR_paramters["inclination"] = config["inclination"]
    IMR_paramters["phase"] = config["phase"]

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

    # set the injected parameters
    params_ = {
        "azimuthal": 0,
        "geocent_time": config["coalescence_time"],
        "inclination": config["truth_value"]["inclination"],
        "final_mass": config["truth_value"]["final_mass"],
        "final_spin": config["truth_value"]["final_spin"],
    }

    MLE, Cov = likelihood._MLE_cov_calculation(params_)
    Integral = 0.5 * np.log(np.linalg.det(Cov)) + 0.5 * len(MLE) * np.log(2 * np.pi)

    save_path = "./results/evidence-test/fisher_matrix"
    label = config["lmns"][0]
    file_name = f"Fisher_matrix_{label}.jsonl"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        record = {"MLE": MLE.tolist(), "Cov": Cov.tolist(), "Integral": Integral}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Save to {file_path}.")

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fisher information matrix calculation"
    )
    parser.add_argument(
        "--config_path", type=str, default="config/ZeroNoise_223_firefly_config.yaml"
    )
    args = parser.parse_args()
    config_path = args.config_path
    config = load_config(config_path, type="firefly")
    main(config)
