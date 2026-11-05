#!/bin/bash
#SBATCH --job-name=DRCFIT
#SBATCH --output=logs/drc_%A_%a.out
#SBATCH --error=logs/drc_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=small
#SBATCH --time=1-00:00:00
#
# DRC at the final fit, LT+MT+HT node 31 (WHY.txt) -- ONE (ARM, TEMPERATURE) PER ARRAY ELEMENT.
# 19 directions x +-0.3 + stick_M_step minus only = 39 arms + baseline = 40, at 15 temperatures = 600 elements; 0-14 the baseline.
# Every temperature grows at node 31, so all 600 can run at once. Submit from this folder, with a Python that has
# numpy and numba on PATH (KMC_CODE optional, see below):
#   mkdir -p logs && sbatch --array=0-599%5 run_drc.sh
# (%5 caps how many elements run at once; set it to suit your allocation. Partition and node exclusions are
#  site-specific: adjust the #SBATCH lines above for your cluster.)
#
# Element i runs line i+1 of drc_tasklist.txt ("<arm file> <T> <seed_base>"): NREP replicates of
# that arm. Every arm uses the same seed base, so arms pair with the baseline replicate by
# replicate. --shard 0 --nshards 1 IS NOT OPTIONAL: run_kmc.py otherwise takes its shard from
# SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT and would run a fraction of the arm.
#
# THE ENGINE RUNS FROM THE ONE CODEBASE, not from a copy: $KMC_CODE, else this repository's Codebase/
# (../../../../Codebase from this folder).
# The job refuses to start unless its files match CODEBASE.md5. RESUME IS AUTOMATIC.

set -euo pipefail
mkdir -p logs results

HERE=$(pwd)
CODE="${KMC_CODE:-$HERE/../../../../Codebase}"
export KMC_CODE="$CODE"
[ -f "$CODE/kmc/run_kmc.py" ] || { echo "[run_drc] no codebase at $CODE -- set KMC_CODE"; exit 1; }
if ! (cd "$CODE" && md5sum -c --quiet "$HERE/CODEBASE.md5"); then
  echo "[run_drc] the codebase at $CODE is not the one this stage was verified with (CODEBASE.md5)"
  exit 1
fi

NREP="${NREP:-144}"
IDX=${SLURM_ARRAY_TASK_ID:?submit with sbatch --array=0-10 run_drc.sh}
read -r ARM T SEED <<< "$(sed -n "$((IDX + 1))p" drc_tasklist.txt)"
[ -n "${ARM:-}" ] || { echo "no entry for index $IDX in drc_tasklist.txt"; exit 1; }
TAG="${ARM%.json}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "[run_drc] host=$(hostname) arm=$TAG T=$T seed_base=$SEED idx=$IDX cpus=${SLURM_CPUS_PER_TASK:-?} job=${SLURM_JOB_ID:-none} code=$CODE"
python warm_numba.py

python "$CODE/kmc/run_kmc.py" \
  --config    config.toml \
  --lattice   "$CODE/kmc/lattice/Lattice_10nm_h232.xyz" \
  --params    "drc_params/$ARM" \
  --temps     "$T" \
  --seed-base "$SEED" \
  --nrep      "$NREP" \
  --shard     0 \
  --nshards   1 \
  --workers   "${SLURM_CPUS_PER_TASK:-48}" \
  --out       "results/drc_${TAG}_T${T}.csv"

echo "[run_drc] done arm=$TAG T=$T"
