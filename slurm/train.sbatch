#!/bin/bash
#SBATCH --job-name=emgfm-train
#SBATCH --partition=staff
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train-%j.out
# usage (from the repository root):   NAME=<run> ARGS="<train.py flags>" sbatch [--gres=gpu:<type>:1] slurm/train.sbatch
# Memory: the refine stage peaks at ~24.4 GB of GPU memory (does not fit a 24 GB RTX 3090; use --exclude
# or a GPU type) and 2-4 GB of host RAM; ~1 CPU core is busy.  Speed: ~3 min/epoch on an RTX PRO 6000
# Blackwell, 10-18 min/epoch on an A40.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"     # submit from the repository root
source slurm/env.sh
echo "host $(hostname) job ${SLURM_JOB_ID} name ${NAME} args: ${ARGS}"
exec "$EMGFM_PYTHON" scripts/train.py --name "${NAME}" ${ARGS}
