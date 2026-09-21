#!/bin/bash
# Refine stage: adapter + encoder LoRA on Granite's text loss alone, from the aligned adapter.
#   bash recipes/03_refine.sh [seeds, default "0 1 2 3 4"]
# One run per seed: they are the ensemble members.  Needs > 24 GB of GPU memory (peaks at ~24.6 GB);
# ~3 min/epoch on an RTX PRO 6000 Blackwell (6 h for the 120 epochs), 10-18 min on an A40.  The schedule is a full
# 120-epoch cosine WITHOUT early stopping: the last points are made in the annealing tail, and a patience of 10
# stops a run before it gets there (README, "Schedule").  FP32=0 reproduces the runs trained before the
# input-projection precision fix.
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${1:-"0 1 2 3 4"}
FP32=${FP32:-1}
INIT=${INIT:-runs/align_whitened/snapshot_ep008.pt}
for s in $SEEDS; do
  NAME=refine_s$s ARGS="--seed $s --text-weight 1 --mel-weight 0 --encoder lora --init-from $INIT \
    --adapter-lr 1e-4 --encoder-lr 5e-5 --lora-r 16 --lora-alpha 32 --input-linear-fp32 $FP32 \
    --warmup-steps 300 --epochs 120 --early-stop 0 --select wer" \
    sbatch --gres=gpu:rtx6000:1 --time=16:00:00 slurm/train.sbatch
done
