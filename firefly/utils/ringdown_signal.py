import lal
import qnm
import pykerr
import numpy as np
from pycbc import waveform


def spher_harms(
    harmonics="spherical",
    l=None,
    m=None,
    n=0,
    inclination=0.0,
    azimuthal=0.0,
    spin=None,
):
    # serving for `fstat_likelihood` to generate the spherical harmonics / spheroidal harmonics
    # TODO (yiming) : implement spheroidal harmonics for estimation pipeline
    if harmonics == "spherical":
        xlm = lal.SpinWeightedSphericalHarmonic(inclination, azimuthal, -2, l, m)
        xlnm = lal.SpinWeightedSphericalHarmonic(inclination, azimuthal, -2, l, -m)
    else:
        assert (
            harmonics == "spheroidal"
        ), "The harmonics must be either spherical or spheroidal."
        if spin is None:
            raise ValueError("must provide a spin for spheroidal harmonics")
        xlm = pykerr.spheroidal(inclination, spin, l, m, n, -2, phi=azimuthal)
        xlnm = pykerr.spheroidal(inclination, spin, l, -m, n, -2, phi=azimuthal)
    return xlm, xlnm


def QNMs_lmn(**kwargs):
    # time_domain_source_model for firefly auxiliary inference
    # Since the auxiliary inference requires polarization waveforms for each `lmn`
    # It is recommanded to use `qnm` to generate it, runtime of likelihood is similar to pycbc
    # TODO (yiming) : implement other cases for estimation pipeline
    waveform_params = dict(
        final_mass=100.0,
        final_spin=0.68,
        lmns=["222"],
        harmonics="spherical",
        azimuthal=0.0,
        model="qnm",
    )
    waveform_params.update(kwargs)
    lmn_all = [
        "%s%d" % (lmn[:2], n)
        for lmn in waveform_params["lmns"]
        for n in range(int("%s" % lmn[-1]))
    ]
    omegas = dict()
    for lmn in lmn_all:
        if (
            waveform_params["harmonics"] == "spherical"
            or waveform_params["harmonics"] == "arbitrary"
        ):
            if waveform_params["model"] == "qnm":
                bbh = qnm.modes_cache(-2, int(lmn[0]), int(lmn[1]), int(lmn[2]))
                omega, _, _ = bbh(a=waveform_params["final_spin"])
                omega0 = omega / (lal.MTSUN_SI * float(waveform_params["final_mass"]))
                f0, tau0 = omega0.real, 1.0 / abs(omega0.imag)
            elif waveform_params["model"] == "ftau":
                f0, tau0 = (
                    2 * np.pi * waveform_params["f_{}".format(str(lmn))],
                    1.0 / waveform_params["tau_{}".format(str(lmn))],
                )
            else:
                assert (
                    waveform_params["model"] == "pykerr"
                ), "The waveform model can only be qnm or pykerr."
                f0 = (
                    2
                    * np.pi
                    * pykerr.qnmfreq(
                        waveform_params["final_mass"],
                        waveform_params["final_spin"],
                        int(lmn[0]),
                        int(lmn[1]),
                        int(lmn[2]),
                    )
                )
                tau0 = pykerr.qnmtau(
                    waveform_params["final_mass"],
                    waveform_params["final_spin"],
                    int(lmn[0]),
                    int(lmn[1]),
                    int(lmn[2]),
                )

            if ("delta_f{}".format(lmn) in waveform_params) and (
                "delta_tau{}".format(lmn) in waveform_params
            ):
                omegas[lmn] = (
                    f0
                    + waveform_params["delta_f{}".format(lmn)] * f0
                    - 1.0j / (tau0 + waveform_params["delta_tau{}".format(lmn)] * tau0)
                )
            elif "delta_f{}".format(lmn) in waveform_params:
                ## omegas.data[i] = f0+waveform_params['delta_f{}'.format(lmn)]-1.j/tau0
                omegas[lmn] = (
                    f0 + waveform_params["delta_f{}".format(lmn)] * f0 - 1.0j / tau0
                )
            elif "delta_tau{}".format(lmn) in waveform_params:
                ## omegas.data[i] = f0-1.j/(tau0+waveform_params['delta_tau{}'.format(lmn)])
                omegas[lmn] = f0 - 1.0j / (
                    tau0 + waveform_params["delta_tau{}".format(lmn)] * tau0
                )
            else:
                ## omegas.data[i] = f0-1.j/tau0
                omegas[lmn] = f0 - 1.0j / tau0
        else:
            assert (
                waveform_params["harmonics"] == "spheroidal"
            ), "The harmonics can only be spherical or spheroidal"
            if waveform_params["model"] == "qnm":
                bbh = qnm.modes_cache(-2, int(lmn[0]), int(lmn[1]), int(lmn[2]))
                omega, _, _ = bbh(a=waveform_params["final_spin"])
                omega0 = omega / (lal.MTSUN_SI * float(waveform_params["final_mass"]))
                f0, tau0 = omega0.real, 1.0 / abs(omega0.imag)
                bbh1 = qnm.modes_cache(-2, int(lmn[0]), -int(lmn[1]), int(lmn[2]))
                omega, _, _ = bbh1(a=waveform_params["final_spin"])
                omega1 = omega / (lal.MTSUN_SI * float(waveform_params["final_mass"]))
                f1, tau1 = omega1.real, 1.0 / abs(omega1.imag)
            else:
                assert (
                    waveform_params["model"] == "pykerr"
                ), "The waveform model can only be qnm or pykerr."
                f0 = (
                    2.0
                    * np.pi
                    * pykerr.qnmfreq(
                        waveform_params["final_mass"],
                        waveform_params["final_spin"],
                        int(lmn[0]),
                        int(lmn[1]),
                        int(lmn[2]),
                    )
                )
                tau0 = pykerr.qnmtau(
                    waveform_params["final_mass"],
                    waveform_params["final_spin"],
                    int(lmn[0]),
                    int(lmn[1]),
                    int(lmn[2]),
                )
                f1 = (
                    2.0
                    * np.pi
                    * pykerr.qnmfreq(
                        waveform_params["final_mass"],
                        waveform_params["final_spin"],
                        int(lmn[0]),
                        -int(lmn[1]),
                        int(lmn[2]),
                    )
                )
                tau1 = pykerr.qnmtau(
                    waveform_params["final_mass"],
                    waveform_params["final_spin"],
                    int(lmn[0]),
                    -int(lmn[1]),
                    int(lmn[2]),
                )
            if ("delta_f{}".format(lmn) in waveform_params) and (
                "delta_tau{}".format(lmn) in waveform_params
            ):
                omegas[lmn] = (
                    f0
                    + waveform_params["delta_f{}".format(lmn)] * f0
                    - 1.0j / (tau0 + waveform_params["delta_tau{}".format(lmn)] * tau0)
                )
                omegas["n" + lmn] = (
                    f1
                    + waveform_params["delta_f{}".format(lmn)] * f1
                    - 1.0j / (tau1 + waveform_params["delta_tau{}".format(lmn)] * tau1)
                )
            elif "delta_f{}".format(lmn) in waveform_params:
                omegas[lmn] = (
                    f0 + waveform_params["delta_f{}".format(lmn)] * f0 - 1.0j / tau0
                )
                omegas["n" + lmn] = (
                    f1 + waveform_params["delta_f{}".format(lmn)] * f1 - 1.0j / tau1
                )
            elif "delta_tau{}".format(lmn) in waveform_params:
                omegas[lmn] = f0 - 1.0j / (
                    tau0 + waveform_params["delta_tau{}".format(lmn)] * tau0
                )
                omegas["n" + lmn] = f1 - 1.0j / (
                    tau1 + waveform_params["delta_tau{}".format(lmn)] * tau1
                )
            else:
                omegas[lmn] = f0 - 1.0j / tau0
                omegas["n" + lmn] = f1 - 1.0j / tau1

    return {"Omegas": omegas}


def Pycbc_ringdown_lmn(**kwargs):
    # time_domain_source_model for fullparams inference
    # There is no need to get polarization waveforms for each `lmn`
    # So we just use pycbc to generate the full waveform
    waveform_params = dict(
        taper=False,
        final_mass=20.0,
        final_spin=None,
        lmns=["222"],
        amp220=1.0,
        phi220=0.0,
        inclination=0.0,
        delta_t=1.0 / 2048,
        model="kerr",
    )
    waveform_params.update(kwargs)
    model = waveform_params["model"]
    lmn_all = [
        "%s%d" % (lmn[:2], n)
        for lmn in waveform_params["lmns"]
        for n in range(int("%s" % lmn[-1]))
    ]
    if len(lmn_all) > 1:
        lmn_all.remove("220")  ## except 220
        if model == "ftau":
            waveform_params["tau_220"] = 1.0 / waveform_params["tau_220"]
        for lmn in lmn_all:
            waveform_params["amp%s" % lmn] = (
                waveform_params["amp%s" % lmn] / waveform_params["amp220"]
            )

    waveform_params["amp220"] = waveform_params["amp220"] * 1.0e-20
    if model == "kerr":
        hplus, hcross = waveform.ringdown.get_td_from_final_mass_spin(**waveform_params)
    elif model == "ftau":
        for lmn in lmn_all:
            waveform_params["tau_%s" % str(lmn)] = (
                1.0 / waveform_params["tau_%s" % str(lmn)]
            )

        hplus, hcross = waveform.ringdown.get_td_from_freqtau(**waveform_params)

    return {"plus": hplus, "cross": hcross}
