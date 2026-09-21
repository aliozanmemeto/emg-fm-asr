#!/bin/bash
#SBATCH --job-name=emgfm
#SBATCH --partition=staff
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# Generic one-GPU wrapper for the non-training scripts, e.g.
#   sbatch --job-name=eval-s1 --gres=gpu:rtx6000:1 slurm/run.sbatch scripts/evaluate.py --ckpt runs/refine_s1/best.pt
#   sbatch --job-name=prep slurm/run.sbatch scripts/prep_mel_targets.py --pca-from data/in16k_pca_basis.npz
# Pin all evaluations you intend to compare to ONE GPU type (bf16 numerics differ between architectures).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"     # submit from the repository root
source slurm/env.sh
echo "host $(hostname) job ${SLURM_JOB_ID}: $*"
exec "$EMGFM_PYTHON" "$@"
