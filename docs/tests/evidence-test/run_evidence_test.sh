#!/usr/bin/env bash
set -euo pipefail
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate firefly

export PYTHONPATH="${PYTHONPATH:-}:$PWD"

# setting
lmns=${1:-221}
fisher_script="docs/tests/evidence-test/fisher_matrix_calculations.py"
config="config/ZeroNoise_${lmns}_firefly_config.yaml"

# Calculate the Fisher information matrix
python "$fisher_script" --config_path "$config"

# Run the full parameters sampling repeatedly
evidence_script="docs/tests/evidence-test/fullparams_evidence_test.py"
for version in {0..9}; do
  echo "=== Running lmns=$lmns, version=$version ==="
  python "$evidence_script" --lmns "$lmns" --version "$version"
  echo "=== Finished version $version ==="
done

# Plot the result
plot_script="docs/tests/evidence-test/summary.py"
python "$plot_script" --lmns "$lmns"
