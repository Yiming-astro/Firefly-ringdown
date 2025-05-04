import os
import numpy as np
import pandas as pd
import mpmath as mp
import multiprocessing

from bilby.core.likelihood import Likelihood
from bilby.gw.detector import InterferometerList

from firefly.utils.ringdown_signal import spher_harms


# TODO (yiming) : future implement of the parameter variations for the 4 parameters.
def Transform_quadrature_to_physical(quadrature_sample):
    # Notice: The default form of quadrature_sample is assumed to appear
    # in a pattern like [B^{220,0}, B^{220,1}, B^{221,0}, B^{221,1}, ...].
    physical_sample = np.zeros_like(quadrature_sample)
    x = quadrature_sample[::2]
    y = quadrature_sample[1::2]
    physical_sample[::2] = np.sqrt(x**2 + y**2)
    physical_sample[1::2] = np.arctan2(y, x)
    physical_sample[1::2][physical_sample[1::2] < 0] += 2 * np.pi

    return physical_sample


class firefly_resampling(Likelihood):

    def __init__(
        self,
        interferometers,
        waveform_generator,
        acfs,
        priors=None,
        sky_average=False,
        n_MC=10,
        n_w=50000,
        n_target=20000,
        n_QNM=5000,
        n_queue=16,
    ):

        self.waveform_generator = waveform_generator
        self.acfs = acfs
        super(firefly_resampling, self).__init__(dict())
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
        self.N_mode = len(self.lmn_all)

        # hyperparameters of FIREFLY
        # Repetitions for evidence calculation
        self.n_MC = n_MC
        # Samples for calculating IS weight in the fisrt-step importance sampling
        self.n_w = n_w
        self.n_target = n_target
        # Samples after the first importance sampling
        self.n_QNM = n_QNM
        # Samples for performing second-step importance sampling
        self.n_queue = n_queue

        # Cache the MLE and Fisher matrix results for auxiliary inference samples
        self._cache_quadrature_Fisher_matrix = False
        self._cache_quadrature_MLE = []
        self._cache_quadrature_cov_matrix = []

    def get_pycbc_detector_response_td(self, ifo, Omegas, start_t):
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

    def _MLE_cov_calculation(self, params):
        try:
            self.parameters = params
            Omegas = self.waveform_generator.time_domain_strain(self.parameters)[
                "Omegas"
            ]
        except RuntimeError:
            return np.nan_to_num(-np.inf)

        if Omegas is None:
            raise ValueError(
                f"Resampling error: something wrong for params {params} in Fisher_matrix_calculation."
            )

        S_all, M_all = self.F_matrix(Omegas)
        M_all_mp = mp.matrix(M_all)
        M_all_inv_mp = mp.inverse(M_all_mp)
        M_all_inv = np.array(M_all_inv_mp.tolist(), dtype=np.float64)
        B_mean = M_all_inv @ S_all
        B_cov = M_all_inv
        return (B_mean, B_cov)

    def _quadrature_Fisher_matrix_calculation(self, sample_dict_list):

        self._cache_quadrature_Fisher_matrix = True
        self._cache_quadrature_MLE = [0 for _ in range(len(sample_dict_list))]
        self._cache_quadrature_cov_matrix = [0 for _ in range(len(sample_dict_list))]

        with multiprocessing.Pool(processes=self.n_queue) as pool:
            results = pool.map(self._MLE_cov_calculation, sample_dict_list)
        for i, res in enumerate(results):
            self._cache_quadrature_MLE[i] = res[0]
            self._cache_quadrature_cov_matrix[i] = res[1]
        return

    def _resampling_amplitude_params(self, auxiliary_posterior):
        """
        Resampling amplitude parameters from the posterior samples.

        Input:
            posterior: (bilby.result.posterior) posterior samples from auxiliary inference

        Output:
            Full_parameters_keys: A list of parameter keys (or labels).
            Full_parameters_sample_array: A numpy array containing resampled parameters.
        """

        # filter the 'log_likelihood', 'log_prior' in auxiliary_posterior
        auxiliary_keys = auxiliary_posterior.keys()
        auxiliary_samples = auxiliary_posterior.values
        excluded_keys = ["log_likelihood", "log_prior"]
        auxiliary_keys_mask = [
            True if key not in excluded_keys else False for key in auxiliary_keys
        ]
        filtered_keys = [key for key in auxiliary_keys if key not in excluded_keys]
        filtered_samples = auxiliary_samples[:, auxiliary_keys_mask]
        amp_keys = [item for s in self.lmn_all for item in (f"amp{s}", f"phi{s}")]
        resampled_keys = np.concatenate([filtered_keys, amp_keys])

        # Calculate and record the mean value and the cov matrix of B params
        sample_dict_list = [
            dict(zip(filtered_keys, sample)) for sample in filtered_samples
        ]
        if not self._cache_quadrature_Fisher_matrix:
            self._quadrature_Fisher_matrix_calculation(sample_dict_list)

        # Construct the Full-parameter samples with A params
        # TODO (yiming) : for 4 params qnms, `2 * self.N_mode` should be corrected.
        physical_amplitude_samples = np.zeros(
            (filtered_samples.shape[0], 2 * self.N_mode)
        )
        for i in range(len(physical_amplitude_samples)):
            quadrature_sample = np.random.multivariate_normal(
                self._cache_quadrature_MLE[i], self._cache_quadrature_cov_matrix[i]
            )
            physical_sample = Transform_quadrature_to_physical(quadrature_sample)
            physical_amplitude_samples[i] = physical_sample
        resampled_samples = np.concatenate(
            [filtered_samples, physical_amplitude_samples], axis=1
        )

        resampled_posterior = {}
        resampled_posterior["param_keys"] = resampled_keys
        resampled_posterior["samples"] = resampled_samples

        return resampled_posterior

    def evidence_calculation(
        self,
        auxiliary_posterior,
        auxiliary_log_evidence,
        amp_prior_max_list,
        save_posterior_in_quadrature_flat_prior=True,
        n_save=5,
    ):

        n_save = self.n_MC if self.n_MC < n_save else n_save
        log_evidence_list = []
        full_params_samples_list = []

        # calculation the evidence in quadrature_flat_prior form
        log_evidence_quadrature_flat_prior = auxiliary_log_evidence - np.sum(
            [np.log(np.pi * Amp_i_Max**2) for Amp_i_Max in amp_prior_max_list]
        )

        for MC_round in range(self.n_MC):
            resampled_posterior = self._resampling_amplitude_params(auxiliary_posterior)
            resampled_param_keys, resampled_samples = (
                resampled_posterior["param_keys"],
                resampled_posterior["samples"],
            )

            amp_index = [
                i for i, key in enumerate(resampled_param_keys) if key.startswith("amp")
            ]
            amp_samples = resampled_samples[:, amp_index]

            log_corr_factor = (
                -np.log(2**self.N_mode)
                + np.log((1 / np.prod(amp_samples, axis=1)).mean())
                + np.log(np.prod(amp_prior_max_list, axis=0))
            )
            Corrected_log_evidence = (
                log_evidence_quadrature_flat_prior + log_corr_factor
            )

            log_evidence_list.append(Corrected_log_evidence)
            if save_posterior_in_quadrature_flat_prior and MC_round < n_save:
                full_params_samples_list.append(resampled_samples)

        log_evidence_array = np.array(log_evidence_list)
        mean_log_evidence = np.mean(log_evidence_array)
        std_mean_log_evidence = np.std(log_evidence_array, ddof=1) / np.sqrt(self.n_MC)

        full_params_posterior_in_quadrature_flat_prior = {}
        full_params_posterior_in_quadrature_flat_prior["param_keys"] = (
            resampled_param_keys
        )
        if save_posterior_in_quadrature_flat_prior:
            full_params_posterior_in_quadrature_flat_prior["samples"] = np.concatenate(
                full_params_samples_list, axis=0
            )

        return (
            mean_log_evidence,
            std_mean_log_evidence,
            full_params_posterior_in_quadrature_flat_prior,
        )

    def _marginal_importance_weights(self, params_index):

        # Notice : currently only suit for transform between amplitude flat prior and quadrature flat prior
        sampled_B_MLE = self._cache_quadrature_MLE[params_index]
        sampled_B_cov = self._cache_quadrature_cov_matrix[params_index]
        sampled_B_list = np.random.multivariate_normal(
            sampled_B_MLE, sampled_B_cov, size=self.n_w
        )

        # proposal term
        q = 1
        # target distribution
        x = sampled_B_list[:, ::2]
        y = sampled_B_list[:, 1::2]
        physical_amp_list = np.sqrt(x**2 + y**2)
        physical_amp_inverse_list = 1 / np.prod(
            physical_amp_list, axis=1, keepdims=True
        )
        p = physical_amp_inverse_list.mean()

        return p / q

    def _IS_resampling_amp_params(self, params_index):

        sampled_B_MLE = self._cache_quadrature_MLE[params_index]
        sampled_B_cov = self._cache_quadrature_cov_matrix[params_index]
        sampled_B_list = np.random.multivariate_normal(
            sampled_B_MLE, sampled_B_cov, size=self.n_QNM
        )

        x = sampled_B_list[:, ::2]
        y = sampled_B_list[:, 1::2]
        weight_list = 1 / np.prod(np.sqrt(x**2 + y**2), axis=1)
        weight_list /= weight_list.sum()
        selected_index = np.random.choice(len(weight_list), size=1, p=weight_list)
        selected_B_sample = sampled_B_list[selected_index].reshape(-1)

        physical_amp_sample = Transform_quadrature_to_physical(selected_B_sample)
        return physical_amp_sample

    def posterior_importance_sampling(self, auxiliary_posterior):
        auxiliary_keys = auxiliary_posterior.keys()
        auxiliary_samples = auxiliary_posterior.values
        excluded_keys = ["log_likelihood", "log_prior"]
        auxiliary_keys_mask = [
            True if key not in excluded_keys else False for key in auxiliary_keys
        ]
        filtered_keys = [key for key in auxiliary_keys if key not in excluded_keys]
        filtered_samples = auxiliary_samples[:, auxiliary_keys_mask]
        self.N_samples = filtered_samples.shape[0]

        if not self._cache_quadrature_Fisher_matrix:
            sample_dict_list = [
                dict(zip(filtered_keys, sample)) for sample in filtered_samples
            ]
            self._quadrature_Fisher_matrix_calculation(sample_dict_list)

        # First-step importance sampling for final mass and spin
        with multiprocessing.Pool(processes=self.n_queue) as pool:
            marginal_weight_list = pool.map(
                self._marginal_importance_weights, np.arange(self.N_samples)
            )

        marginal_weight_list = marginal_weight_list / np.sum(marginal_weight_list)

        first_IS_sampled_indices = np.random.choice(
            self.N_samples, size=self.n_target, replace=True, p=marginal_weight_list
        )
        first_IS_samples = filtered_samples[first_IS_sampled_indices, :]

        # Second-step importance sampling for amplitude params
        with multiprocessing.Pool(processes=self.n_queue) as pool:
            amp_samples = pool.map(
                self._IS_resampling_amp_params, first_IS_sampled_indices
            )

        amp_keys = [item for s in self.lmn_all for item in (f"amp{s}", f"phi{s}")]
        full_keys = np.concatenate([filtered_keys, amp_keys])
        full_samples = np.concatenate((first_IS_samples, amp_samples), axis=1)
        firefly_posterior = {}
        firefly_posterior["param_keys"] = full_keys
        firefly_posterior["samples"] = full_samples

        return firefly_posterior

    @staticmethod
    def save_posterior(firefly_posterior, outdir, file_name="sampling.csv"):
        firefly_posterior_keys = firefly_posterior["param_keys"]
        firefly_posterior_samples = firefly_posterior["samples"]
        df = pd.DataFrame(firefly_posterior_samples, columns=firefly_posterior_keys)
        file_path = os.path.join(outdir, file_name)
        df.to_csv(file_path, index=False)
        return
