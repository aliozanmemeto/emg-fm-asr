#!/bin/bash
# The same recipe on a second public benchmark: the emg2speech General Corpus (Gowda et al., ACL 2026; OSF 65vbx,
# CC-BY 4.0; one speaker, VOCALISED speech, 31 channels at 5 kHz, 9,660 sentences, sequential split 8500/760/400).
# Everything is identical to recipes/01-03 except the corpus-specific input geometry:
#   --in-ch 31            31 electrodes instead of Gaddy's 8
#   --conv-strides 4,5,5  100 EMG samples per 20 ms frame at 5 kHz (Gaddy: 2,2,5 = 20 samples at 1 kHz)
#   --time-shift 55       Gaddy's 0-10 ms random shift, in 5 kHz samples (11 at 1 kHz)
#   --time-mask 2,1500    Gaddy-recipe time masks of up to 300 ms, in 5 kHz samples (2,300 at 1 kHz)
# Steps, from the repository root; each submits SLURM jobs.  Decisions on dev (760 sentences); test (400) once.
set -euo pipefail
cd "$(dirname "$0")/.."
STEP=${1:?usage: bash recipes/emg2speech.sh prep|mel|align|refine|eval [seeds]}
GEOM="--in-ch 31 --conv-strides 4,5,5 --time-shift 55 --time-mask 2,1500"
DATA="--manifest cache/emg2speech/manifest.csv --mel-cache cache/emg2speech/in16k"
source slurm/env.sh
case $STEP in
  prep)   # 40 GB pickle -> per-utterance EMG (notch 50 Hz, z-score x 20) and 16 kHz audio stretched to the EMG duration
    : "${E2S_DATA:?set E2S_DATA to the GeneralCorpusData directory of the OSF release (https://osf.io/65vbx/)}"
    mkdir -p logs
    sbatch --job-name=e2s-prep --mem=64G --cpus-per-task=2 --time=02:00:00 --output=logs/e2s-prep-%j.out \
      --wrap "cd $PWD && source slurm/env.sh && \$EMGFM_PYTHON scripts/prep_emg2speech.py --data $E2S_DATA --out cache/emg2speech" ;;
  mel)    # Granite's own input features for all 9,660 utterances + statistics and a fresh whitening basis (training split)
    sbatch --job-name=e2s-mel --nice=500 slurm/run.sbatch scripts/prep_mel_targets.py --manifest cache/emg2speech/manifest.csv \
      --out cache/emg2speech/in16k ;;
  align)  # adapter alone, spectrogram targets (own audio, frame for frame), plain MSE then whitened warm start
    COMMON="$DATA $GEOM --text-weight 0 --mel-weight 1 --encoder frozen --adapter-lr 3e-4 --warmup-steps 500 --epochs 60 --early-stop 8 --select mel"
    A=$(NAME=e2s_align_plain ARGS="$COMMON --mel-whiten 0" sbatch --parsable slurm/train.sbatch)
    B=$(NAME=e2s_align_whitened ARGS="$COMMON --mel-whiten 16 --init-from runs/e2s_align_plain/best.pt --snapshot-epochs 8" \
        sbatch --parsable --dependency=afterok:$A slurm/train.sbatch)
    echo "e2s_align_plain $A -> e2s_align_whitened $B (initialisation for refine: runs/e2s_align_whitened/snapshot_ep008.pt)" ;;
  refine) # adapter + encoder LoRA on the text loss alone: recipes/03_refine.sh with the geometry flags
    SEEDS=${2:-"0"}
    INIT=${INIT:-runs/e2s_align_whitened/snapshot_ep008.pt}
    for s in $SEEDS; do
      NAME=e2s_refine_s$s ARGS="$DATA $GEOM --seed $s --text-weight 1 --mel-weight 0 --encoder lora --init-from $INIT \
        --adapter-lr 1e-4 --encoder-lr 5e-5 --lora-r 16 --lora-alpha 32 --input-linear-fp32 1 \
        --warmup-steps 300 --epochs 120 --early-stop 0 --select wer" sbatch --gres=gpu:rtx6000:1 --time=16:00:00 slurm/train.sbatch
    done ;;
  eval)   # exact dev evaluation of every refine run: batch 1, recalibrated BatchNorm, beams 1/4/8, one GPU type
    for d in runs/e2s_refine_s*/; do
      sbatch --job-name=eval-$(basename $d) --gres=gpu:rtx6000:1 --nice=500 --mem=16G --cpus-per-task=2 --time=01:00:00 \
        slurm/run.sbatch scripts/evaluate.py --ckpt $d/best.pt --beams 1,4,8 --bs 1 --bn recal --out $d/eval_dev.json
    done ;;
  *) echo "unknown step $STEP"; exit 1 ;;
esac
