import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

import pycbc.psd as pp
from bilby.gw.detector import InterferometerList
from pycbc.noise.reproduceable import colored_noise

from firefly.utils.ringdown_signal import QNM_injection
from firefly.utils.load_config import load_config_signal_injections


def main(config):

    injected_params = config["injection_params"]
    injected_params["lmns"] = config["lmns"]
    geocent_time = injected_params["geocent_time"]
    sampling_frequency = config["sampling_frequency"]
    slice_duration = config["slice_duration"]
    duration = 1
    delta_t = 1.0 / sampling_frequency
    injected_params["delta_t"] = delta_t
    # TODO (yiming) : currently, we are using sky_average and combined the ET factor here
    sky_average = 1.5 / np.sqrt(5)
    det_names = config["det_name"]
    f_filter = config["f_filter"]

    waveform_QNM = QNM_injection(config, **injected_params)

    h_plus, h_cross = waveform_QNM["plus"], waveform_QNM["cross"]
    QNM_times = np.arange(geocent_time, slice_duration, delta_t)
    signal_TD = (h_plus + h_cross) * sky_average

    # TODO (yiming) sky_average case
    strain_det = {}
    for det in det_names:
        strain_det[det] = {"time": QNM_times, "strain": signal_TD}

    # add noise
    if config["add_noise"]:
        ifos = InterferometerList(det_names)
        psds = {}
        for det in det_names:
            f_asd = config["psd_path"][det]
            freq0, _, _, asd0 = np.loadtxt(f_asd, unpack=True)
            psds[det] = pp.from_numpy_arrays(
                freq0,
                asd0**2,
                int(duration * sampling_frequency),
                delta_f=1.0 / duration,
                low_freq_cutoff=f_filter,
            )
        for i, ifo in enumerate(ifos):
            noise = colored_noise(
                psds[ifo.name],
                geocent_time - duration / 2,
                geocent_time + duration / 2,
                seed=42 + i,
                sample_rate=sampling_frequency,
                low_frequency_cutoff=f_filter,
                filter_duration=duration,
            )
            strain_det[ifo.name]["strain"] = (
                strain_det[ifo.name]["strain"]
                + noise.data[: len(strain_det[ifo.name]["strain"])]
            )

    # plot the strain
    if config["plot_strain"]:

        plt.figure(figsize=(10, 4))
        for det in det_names:
            plt.plot(strain_det[det]["time"], strain_det[det]["strain"], label=det)
        plt.xlabel("Time (s)")
        plt.ylabel("Strain")
        plt.title("Strain of injected Ringdown Signal")
        plt.legend()
        plt.grid(True)
        plt.show()

    # save the strain
    for det in det_names:
        strain_path = os.path.join(
            config["strain_save_path"], config["label"] + "-" + det + ".txt"
        )
        txt_col = np.column_stack([strain_det[det]["time"], strain_det[det]["strain"]])
        np.savetxt(
            strain_path,
            txt_col,
            delimiter="  ",
        )
    # Save the meta data of generated strain
    meta_path = os.path.join(
        config["strain_save_path"], config["label"] + "_meta.jsonl"
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(config, ensure_ascii=False) + "\n")

    save_path = config["strain_save_path"]
    print(f"Strain file is saved in {save_path}.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simulate signals")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/simulate_signals.yaml",
        help="The path to the config_file",
    )
    args = parser.parse_args()
    config = load_config_signal_injections(args.config_path)

    main(config)
