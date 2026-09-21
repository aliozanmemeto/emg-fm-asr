#!/bin/bash
# Align stage: the adapter learns to predict the spoken utterance's spectrogram; Granite stays frozen and no
# transcript is used.  Two runs: plain MSE first, then the whitened loss warm-started from it.  Selection is
# on the dev-vocal spectrogram loss (--select mel); the refine stage starts from the whitened run's epoch-8
# snapshot.  Fits any GPU (~7 GB); ~10 min/epoch on an A40.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON="--text-weight 0 --mel-weight 1 --encoder frozen --adapter-lr 3e-4 --warmup-steps 500 --epochs 60 --early-stop 8 --select mel"
A=$(NAME=align_plain ARGS="$COMMON --mel-whiten 0" sbatch --parsable slurm/train.sbatch)
B=$(NAME=align_whitened ARGS="$COMMON --mel-whiten 16 --init-from runs/align_plain/best.pt --snapshot-epochs 8" \
    sbatch --parsable --dependency=afterok:$A slurm/train.sbatch)
echo "align_plain $A -> align_whitened $B (initialisation for refine: runs/align_whitened/snapshot_ep008.pt)"
