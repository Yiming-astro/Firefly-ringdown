# Evidence-test

### 1 for all

```sh
chmod +x docs/tests/evidence-test/run_evidence_test.sh
docs/tests/evidence-test/run_evidence_test.sh
```

### 1

```sh
export PYTHONPATH="${PYTHONPATH}:$PWD"
python docs/tests/evidence-test/fisher_matrix_calculations.py --config_path config/ZeroNoise_221_firefly_config.yaml
```

### 2

```sh
python docs/tests/evidence-test/fullparams_evidence_test.py --lmns 223 --version 1
```

### 3
```sh
python docs/tests/evidence-test/summary.py --lmns 221
```
