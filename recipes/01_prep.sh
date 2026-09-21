#!/bin/bash
# Data preparation.  EMG_DATA = Gaddy & Klein's corpus root (contains silent_parallel_data/ etc.).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${EMG_DATA:?set EMG_DATA to the corpus root}"
source slurm/env.sh
mkdir -p logs
# 1) filtered 1000-Hz EMG + manifest (CPU only, ~20 min)
"$EMGFM_PYTHON" scripts/prep_emg_cache.py --data "$EMG_DATA" --out cache/emg1k
# 2) spectrogram targets, feature statistics and the whitening basis (one GPU, ~30 min).
#    --pca-from ships the basis the reported runs used; omit it to fit a fresh basis on all training frames.
sbatch --job-name=prep-mel slurm/run.sbatch scripts/prep_mel_targets.py --manifest cache/emg1k/manifest.csv \
    --out cache/in16k --pca-from data/in16k_pca_basis.npz
