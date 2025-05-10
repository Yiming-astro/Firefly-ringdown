import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from firefly.utils.ringdown_signal import QNM_injection
from firefly.utils.load_config import load_config_signal_injections


def main(config):

    injected_params = config["injection_params"]
    injected_params["lmns"] = config["lmns"]
    geocent_time = injected_params["geocent_time"]
    sampling_frequency = config["sampling_frequency"]
    duration = config["duration"]
    delta_t = 1.0 / sampling_frequency
    injected_params["delta_t"] = delta_t
    # TODO (yiming) : currently, we are using sky_average and combined the ET factor here
    sky_average = 1.5 / np.sqrt(5)

    waveform_QNM = QNM_injection(config, **injected_params)

    h_plus, h_cross = waveform_QNM["plus"], waveform_QNM["cross"]
    QNM_times = np.arange(geocent_time, duration, delta_t)
    signal_TD = (h_plus + h_cross) * sky_average

    # plot the strain
    if config["plot_strain"]:

        plt.figure(figsize=(10, 4))
        plt.plot(QNM_times, signal_TD, label="hcross (Cross Polarization)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Cross Polarization of Ringdown Signal")
        plt.legend()
        plt.grid(True)
        plt.show()

    # save the strain
    strain_path = os.path.join(config["strain_save_path"], config["label"] + ".txt")
    txt_col = np.column_stack([QNM_times, signal_TD])
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
