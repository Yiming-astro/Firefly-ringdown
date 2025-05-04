import numpy as np
import mpmath as mp

from bilby.gw.detector import InterferometerList
from bilby.core.likelihood import Likelihood
from bilby.gw.waveform_generator import WaveformGenerator

from firefly.utils.ringdown_signal import spher_harms


# Waveform Generator
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


# Auxiliary inference likelihood in firefly
class Auxiliary_likelihood(Likelihood):

    def __init__(
        self, interferometers, waveform_generator, acfs, priors=None, sky_average=False
    ):

        self.waveform_generator = waveform_generator
        self.acfs = acfs
        super(Auxiliary_likelihood, self).__init__(dict())
        self.interferometers = InterferometerList(interferometers)
        self.priors = priors
        self.sky_average = sky_average  ## This works for ET/LISA/TQ
        self.harmonics = self.waveform_generator.waveform_arguments["harmonics"]
        self.lmns = self.waveform_generator.waveform_arguments["lmns"]
        self._meta_data = {}
        self.A_all = None
        self.lmn_all = (
            [
                "%s%d" % (lmn[:2], n)
                for lmn in self.lmns
                for n in range(int("%s" % lmn[-1]))
            ]
            if self.harmonics == "spherical" or self.harmonics == "arbitrary"
            else [
                "%s%d" % (lmn[:2], n)
                for lmn in self.lmns
                for n in range(int("%s" % lmn[-1]))
            ]
            + [
                "n%s%d" % (lmn[:2], n)
                for lmn in self.lmns
                for n in range(int("%s" % lmn[-1]))
            ]
        )

    def noise_log_likelihood(self):
        log_l = 0.0
        if "nll" in self._meta_data.keys():
            return self._meta_data["nll"]

        for ifm in self.interferometers:
            signal_ifo = ifm.strain_data.to_pycbc_timeseries()
            s_s = (self.acfs[ifm.name]) @ (signal_ifo)
            log_l -= 0.5 * sum(s_s * s_s)

        self._meta_data["nll"] = log_l
        return log_l

    def get_pycbc_detector_response_td(self, ifo, Omegas, start_t):
        """Get the detector response for a particular waveform

        Parameters
        -------
        Omegas: complex
            Omega = 2*pi*f*+i/tau

        Returns
        -------
        array_like: (signal observed in the interferometer)
        """
        det_response = {}
        for mode in ["plus", "cross"]:
            if self.sky_average:
                det_response[mode] = (
                    self.sky_average
                )  ## 1.5/np.sqrt(5) for ET, which have three detectors, triangular frame and we consider the sky average case.
            else:
                det_response[mode] = ifo.antenna_response(
                    self.parameters["ra"],
                    self.parameters["dec"],
                    self.parameters["geocent_time"],
                    self.parameters["psi"],
                    mode,
                )

        delta_t = 1.0 / ifo.strain_data.sampling_frequency
        t_list = np.arange(0.0, ifo.strain_data.duration, delta_t)

        omegas = {lmn: Omegas[lmn].real for lmn in self.lmn_all}
        rtaus = {lmn: abs(Omegas[lmn].imag) for lmn in self.lmn_all}
        waves = []
        for lmn in self.lmn_all:
            if self.harmonics == "arbitrary":
                A_plus = det_response["plus"]
                A_cross = det_response["cross"]
                ht1 = (
                    1.0e-20
                    * A_plus
                    * np.cos(omegas[lmn] * (t_list))
                    * np.exp(-t_list * rtaus[lmn])
                )
                ht2 = (
                    1.0e-20
                    * A_plus
                    * np.sin(omegas[lmn] * (t_list))
                    * np.exp(-t_list * rtaus[lmn])
                )
                ht3 = (
                    1.0e-20
                    * A_cross
                    * np.sin(omegas[lmn] * (t_list))
                    * np.exp(-t_list * rtaus[lmn])
                )
                ht4 = (
                    1.0e-20
                    * A_cross
                    * np.cos(omegas[lmn] * (t_list))
                    * np.exp(-t_list * rtaus[lmn])
                )
                waves.append(ht1)
                waves.append(ht2)
                waves.append(ht3)
                waves.append(ht4)
            elif self.harmonics == "spherical":
                fspin = (
                    self.parameters["final_spin"]
                    if "final_spin" in self.parameters.keys()
                    else None
                )
                Y_lm, Y_lnm = spher_harms(
                    harmonics=self.harmonics,
                    l=int(lmn[0]),
                    m=int(lmn[1]),
                    n=int(lmn[2]),
                    inclination=self.parameters["inclination"],
                    azimuthal=self.parameters["azimuthal"],
                    spin=fspin,
                )
                lm_p = Y_lm.real + (-1) ** int(lmn[0]) * Y_lnm.real
                lm_c = Y_lm.real - (-1) ** int(lmn[0]) * Y_lnm.real
                A_plus = det_response["plus"] * lm_p
                A_cross = det_response["cross"] * lm_c
                ht1 = (
                    1.0e-20
                    * (
                        A_plus * np.cos(omegas[lmn] * (t_list))
                        + A_cross * np.sin(omegas[lmn] * (t_list))
                    )
                    * np.exp(-t_list * rtaus[lmn])
                )
                ht2 = (
                    -1.0e-20
                    * (
                        A_plus * np.sin(omegas[lmn] * (t_list))
                        - A_cross * np.cos(omegas[lmn] * (t_list))
                    )
                    * np.exp(-t_list * rtaus[lmn])
                )
                waves.append(ht1)
                waves.append(ht2)
            else:
                assert (
                    self.harmonics == "spheroidal"
                ), "the harmonics can only be spherical or spheroidal"
                Y_lm, Y_lnm = spher_harms(
                    harmonics=self.harmonics,
                    l=int(lmn[0]),
                    m=int(lmn[1]),
                    n=int(lmn[2]),
                    inclination=self.parameters["inclination"],
                    azimuthal=self.parameters["azimuthal"],
                    spin=self.parameters["final_spin"],
                )
                A_plus = det_response["plus"] * (Y_lm.__abs__())
                A_cross = det_response["cross"] * (Y_lm.__abs__())
                nA_plus = det_response["plus"] * (Y_lnm.__abs__())
                nA_cross = det_response["cross"] * (Y_lnm.__abs__())
                ht1 = (
                    1.0e-20
                    * (
                        A_plus * np.cos(omegas[lmn] * (t_list))
                        + A_cross * np.sin(omegas[lmn] * (t_list))
                    )
                    * np.exp(-t_list * rtaus[lmn])
                )
                ht2 = (
                    -1.0e-20
                    * (
                        A_plus * np.sin(omegas[lmn] * (t_list))
                        - A_cross * np.cos(omegas[lmn] * (t_list))
                    )
                    * np.exp(-t_list * rtaus[lmn])
                )
                waves.append(ht1)
                waves.append(ht2)
                ht3 = (
                    1.0e-20
                    * (
                        nA_plus * np.cos(omegas["n" + lmn] * (t_list))
                        + nA_cross * np.sin(omegas["n" + lmn] * (t_list))
                    )
                    * np.exp(-t_list * rtaus["n" + lmn])
                )
                ht4 = (
                    -1.0e-20
                    * (
                        nA_plus * np.sin(omegas["n" + lmn] * (t_list))
                        - nA_cross * np.cos(omegas["n" + lmn] * (t_list))
                    )
                    * np.exp(-t_list * rtaus["n" + lmn])
                )
                waves.append(ht3)
                waves.append(ht4)

        return waves

    def F_matrix(self, Omegas):
        l0 = (
            4 * len(self.lmn_all)
            if self.harmonics == "arbitrary"
            else 2 * len(self.lmn_all)
        )
        S_all = np.zeros(l0)
        M_all = np.zeros((l0, l0))
        for ifm in self.interferometers:
            hts = self.get_pycbc_detector_response_td(
                ifm, Omegas, ifm.strain_data.start_time
            )
            Sd = np.zeros(l0)
            Md = np.zeros((l0, l0))
            wd = (self.acfs[ifm.name]) @ (ifm.strain_data.time_domain_strain)
            whs = [(self.acfs[ifm.name]) @ (hts[i]) for i in range(l0)]

            for i in range(l0):
                Sd[i] = sum(whs[i] * wd)
                for j in range(l0):
                    Md[i][j] = sum(whs[i] * whs[j]) if j >= i else Md[j][i]

            S_all = S_all + Sd
            M_all = M_all + Md
        return S_all, M_all

    def log_likelihood_ratio(self):

        mp.mp.dps = 100
        try:
            Omegas = self.waveform_generator.time_domain_strain(self.parameters)[
                "Omegas"
            ]
        except RuntimeError:
            return np.nan_to_num(-np.inf)

        if Omegas is None:
            return np.nan_to_num(-np.inf)

        S_all, M_all = self.F_matrix(Omegas)
        M_all_mp = mp.matrix(M_all)
        M_all_inv_mp = mp.inverse(M_all_mp)
        M_all_inv = np.array(M_all_inv_mp.tolist(), dtype=np.float64)
        self.A_all = S_all @ M_all_inv
        log_l = 0.5 * sum(self.A_all * S_all)
        dim_M = M_all.shape[0]
        log_M = -0.5 * np.log(np.linalg.det(M_all)) + 0.5 * dim_M * np.log(2 * np.pi)
        return log_l + log_M

    def log_likelihood(self):
        return self.log_likelihood_ratio() + self.noise_log_likelihood()
