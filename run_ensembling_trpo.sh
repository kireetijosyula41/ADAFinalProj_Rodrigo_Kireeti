#!/usr/bin/env bash
set -euo pipefail

# Which env version to use: "new" or "original"
VERSION="new"

# How many ensemble members per algorithm
N_RUNS=7
EXP_ID=0  # Experiment ID to distinguish different ensemble experiments

echo "Running ensembles for version: ${VERSION}"
echo "Number of runs per algorithm: ${N_RUNS}"
echo "Experiment ID: ${EXP_ID}"
echo

# Helper function to train one algorithm N_RUNS times
train_algo_ensemble() {
  local algo="$1"   # ppo / a2c / trpo

  echo "============================"
  echo " Training ${algo} ensemble "
  echo "============================"

  for (( i=0; i<${N_RUNS}; i++ )); do
    echo
    echo "--- ${algo} run ${i} ---"
    python "train_${algo}.py" "${VERSION}" "${i}" "${EXP_ID}"
  done

  echo
  echo "Running ensemble evaluation for ${algo}..."
  python ensemble_eval.py "${algo}" "${VERSION}"
  echo
}

train_algo_ensemble "trpo"

echo "All ensembles completed."