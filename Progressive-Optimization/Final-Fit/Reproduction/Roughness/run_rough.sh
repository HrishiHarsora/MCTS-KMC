#!/bin/bash
#SBATCH --job-name=ROUGH
#SBATCH --output=logs/rough_%A_%a.out
#SBATCH --error=logs/rough_%A_%a.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=1-00:00:00
#
# Roughness vs thickness and temperature at the LT+MT+HT fit (roughness_study.py). One array element = one chunk
# of the 8,160 films (34 T x 240), on one 48-core node; 15 chunks in all. Submit from this folder, with a Python
# that has numpy and numba on PATH:
#   mkdir -p logs && sbatch --array=0-14%5 run_rough.sh
# (%5 caps how many chunks run at once; set it to suit your allocation. Partition and node exclusions are
#  site-specific: adjust the #SBATCH lines above for your cluster.)
# KMC_CODE = the engine's Codebase folder; it defaults to this repository's Codebase/ (../../../../Codebase from this folder).
set -euo pipefail
mkdir -p logs results
export KMC_CODE="${KMC_CODE:-$SLURM_SUBMIT_DIR/../../../../Codebase}"
if ! (cd "$KMC_CODE" && md5sum -c --quiet "$SLURM_SUBMIT_DIR/CODEBASE.md5"); then
  echo "[run_rough] the codebase at $KMC_CODE is not the pinned one (CODEBASE.md5)"; exit 1
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[run_rough] host=$(hostname) chunk=${SLURM_ARRAY_TASK_ID} code=$KMC_CODE $(python -c 'import sys,numpy,numba;print(sys.version.split()[0],numpy.__version__,numba.__version__)')"
python warm_numba.py
python roughness_study.py --params best_params.json --chunk "$SLURM_ARRAY_TASK_ID" --nchunks 15 \
       --workers "${SLURM_CPUS_PER_TASK:-48}" --out results
echo "[run_rough] done"
