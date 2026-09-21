#!/bin/bash
# Ensemble of the refine seeds: n-best generation per member, then pooled scoring and the report.
#   bash recipes/05_ensemble.sh                      # members = runs/refine_s*/best.pt
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cache/ens
DEPS=""; FILES=""; NAMES=""
for d in runs/refine_s*/; do
  n=$(basename $d)
  j=$(sbatch --parsable --job-name=gen-$n --gres=gpu:rtx6000:1 slurm/run.sbatch scripts/ensemble.py --mode gen \
      --ckpt $d/best.pt --beams 1,4,8 --bn recal --out cache/ens/dev_$n.pt)
  DEPS="$DEPS:$j"; FILES="$FILES,cache/ens/dev_$n.pt"; NAMES="$NAMES,$n"
done
sbatch --job-name=ens-score --gres=gpu:rtx6000:1 --dependency=afterok${DEPS} slurm/run.sbatch scripts/ensemble.py \
    --mode score --gen-files ${FILES#,} --names ${NAMES#,} --out cache/ens/dev_score_refine.json
echo "then: python scripts/ensemble.py --mode subsets --out cache/ens/dev_score_refine.json"
