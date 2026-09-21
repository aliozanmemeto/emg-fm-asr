#!/bin/bash
# Exact dev evaluation of every refine run (batch 1, recalibrated BatchNorm, beams 1/4/8), all on one GPU
# type, then the paired comparison of two groups of runs.
#   bash recipes/04_evaluate.sh                      # evaluate runs/refine_s*/best.pt
#   python scripts/compare.py runs/refine_s0/eval_dev.json,runs/refine_s1/eval_dev.json runs/other/eval_dev.json
set -euo pipefail
cd "$(dirname "$0")/.."
for d in runs/refine_s*/; do
  sbatch --job-name=eval-$(basename $d) --gres=gpu:rtx6000:1 slurm/run.sbatch scripts/evaluate.py \
      --ckpt $d/best.pt --beams 1,4,8 --bs 1 --bn recal --out $d/eval_dev.json
done
