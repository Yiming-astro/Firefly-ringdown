from __future__ import division
import numpy as np
from bilby.gw.detector import InterferometerList
from bilby.core.likelihood import Likelihood
from bilby.gw.waveform_generator import WaveformGenerator


class TD_WaveformGenerator(WaveformGenerator):

    def time_domain_strain(self, parameters=None):

        if parameters == self._cache["parameters"]:
            waveform_polarizations = self._cache["waveform"]
        else:
            try:
                waveform_polarizations = self.time_domain_source_model(
                    **{**parameters, **self.waveform_arguments}
                )
            except RuntimeError:
                return None

        self._cache["waveform"] = waveform_polarizations
        self._cache["parameters"] = parameters.copy()

        return waveform_polarizations


# Full-parameter estimation likelihood
"""
Refer to paper: https://arxiv.org/abs/2107.05609 for further theoretical details.
"""


class Full_params_likelihood(Likelihood):

    def __init__(
        self,
        interferometers,
        waveform_generator,
        acfs,
        normalisations={"H1": 0.0, "L1": 0.0, "V1": 0.0},
        priors=None,
        sky_average=False,
    ):

        self.waveform_generator = waveform_generator
        self.acfs = acfs
        self.normalisations = normalisations
        super(Full_params_likelihood, self).__init__(dict())
        self.interferometers = InterferometerList(interferometers)
        self.priors = priors
        self.sky_average = sky_average
        self._meta_data = {}

    def noise_log_likelihood(self):
        log_l = 0.0
        if "nll" in self._meta_data.keys():
            return self._meta_data["nll"]

        for ifm in self.interferometers:
            signal_ifo = ifm.strain_data.to_pycbc_timeseries()
            s_s = (self.acfs[ifm.name]) @ (signal_ifo.data)
            log_l -= 0.5 * sum(s_s * s_s) + self.normalisations[ifm.name]

        self._meta_data["nll"] = log_l
        return log_l

    def get_pycbc_detector_response_td(self, ifo, waveform_polarizations, start_t):
        """Get the detector response for a particular waveform

        Parameters
        -------
        waveform_polarizations: dict
            polarizations of the waveform

        Returns
        -------
        array_like: (signal observed in the interferometer)
        """
        signal = {}
        for mode in waveform_polarizations.keys():
            det_response = ifo.antenna_response(
                self.parameters["ra"],
                self.parameters["dec"],
                self.parameters["geocent_time"],
                self.parameters["psi"],
                mode,
            )
            signal[mode] = waveform_polarizations[mode] * det_response

        signal_ifo = sum(signal.values())
        shift_t = ifo.time_delay_from_geocenter(
            self.parameters["ra"],
            self.parameters["dec"],
            self.parameters["geocent_time"],
        )
        dt = (
            shift_t
            + self.parameters["geocent_time"]
            - start_t.__float__()
            + signal_ifo.end_time.__float__()
        )
        signal_ifo.prepend_zeros(
            int((dt + ifo.strain_data.duration) / signal_ifo.delta_t)
        )  ## append zeros for roll
        signal_ifo.roll(int(round(dt / signal_ifo.delta_t, 0)))
        signal_ifo.start_time = start_t

        return signal_ifo

    def log_likelihood(self):
        try:
            waveform_polarizations = self.waveform_generator.time_domain_strain(
                self.parameters
            )
        except RuntimeError:
            return np.nan_to_num(-np.inf)

        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        log_l = 0.0
        for ifm in self.interferometers:
            signal_ifo = ifm.strain_data.to_pycbc_timeseries()
            l0 = len(signal_ifo.data)
            if self.sky_average:
                waveform_det = (
                    waveform_polarizations["plus"] + waveform_polarizations["cross"]
                ) * self.sky_average
                waveform_det.append_zeros(l0)
                waveform_det.start_time = signal_ifo.start_time.__float__()
            else:
                waveform_det = self.get_pycbc_detector_response_td(
                    ifm, waveform_polarizations, signal_ifo.start_time
                )
            s_h = signal_ifo.data - waveform_det.data[:l0]
            w_s_h = (self.acfs[ifm.name]) @ s_h
            log_l -= 0.5 * sum(w_s_h * w_s_h) + self.normalisations[ifm.name]
        return log_l

    def log_likelihood_ratio(self):
        return self.log_likelihood() - self.noise_log_likelihood()
