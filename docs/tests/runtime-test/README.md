# Runtime Test

We provide a set of configuration files under various setups to help gain a deeper understanding of the runtime of Bayesian inference in ringdown analysis. We compare the runtimes of the Bayesian analysis under different sampler settings and detector sensitivities.

---------------------------------

### Detector Sensitivity

We use the same zero-noise signal as the strain and perform Bayesian inference under both LIGO and ET sensitivities, with all other sampling settings kept identical. We can see that the sampling time is significantly longer under the ET sensitivity compared to LIGO sensitivity.

Make sure that `firefly` is in your current operating path, run the following command in the terminal. 

For LIGO sensitivity,
```sh
python -m firefly.cli.fullparams_ringdown --config_path docs/tests/runtime-test/config/NR0305_223_fullparams_LIGO.yaml
```

For ET sensitivity,
```sh
python -m firefly.cli.fullparams_ringdown --config_path docs/tests/runtime-test/config/NR0305_223_fullparams_ET.yaml
```

---------------------------------

### Sampler Setting

By choosing lower sampling settings, the Bayesian inference time is significantly reduced. We provide the running scripts for Bayesian inference using nested sampling and MCMC under lower sampling settings. 

For nested sampling (`nlive` : 256, `dlogz` : 1),
```sh
python -m firefly.cli.fullparams_ringdown --config_path docs/tests/runtime-test/config/NR0305_222_fullparams_ns_low.yaml
```

For MCMC (`nsamples` : 100, `thin_by_nact` : 1),
```sh
python -m firefly.cli.fullparams_ringdown --config_path docs/tests/runtime-test/config/NR0305_222_fullparams_mcmc_low.yaml
```

---------------------------------

### Two-step importance sampling setting

We provide a script with stricter hyperparameter configurations for the two-step importance sampling, to help compare how those hyperparameters affect the importance sampling.

```sh
python -m firefly.cli.firefly_ringdown --config_path docs/tests/runtime-test/config/NR0305_224_firefly_config_stringent_setting.yaml
```

We can adjust the hyperparameter settings to check whether the posterior distributions remain consistent. The examples in the `Firefly-ringdown/config` already use sufficiently conservative settings, and applying stricter hyperparameters will not produce distinguishable changes in the posteriors.