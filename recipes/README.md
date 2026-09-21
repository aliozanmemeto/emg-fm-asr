# Recipes

The exact commands behind the reported numbers, in order.  Every script is submitted through the SLURM wrappers in
`slurm/` (edit `slurm/env.sh` for your account); run them from the repository root.

| step | script | what it produces |
|---|---|---|
| 1 | `01_prep.sh` | `cache/emg1k` (filtered EMG + manifest), `cache/in16k` (spectrogram targets, statistics, whitening basis) |
| 2 | `02_align.sh` | `runs/align_plain`, then `runs/align_whitened/snapshot_ep008.pt` -- the adapter initialisation |
| 3 | `03_refine.sh` | `runs/refine_s<seed>/best.pt` for each seed -- the single models (120-epoch cosine, no early stopping) |
| 4 | `04_evaluate.sh` | exact dev evaluations (`eval_dev.json` per run) and paired comparisons |
| 5 | `05_ensemble.sh` | n-best generation, pooled scoring, ensemble report |
| all | `emg2speech.sh prep\|mel\|align\|refine\|eval` | the same recipe on the emg2speech General Corpus (31 channels, 5 kHz) |

Two stages of the align step are deliberate: the whitened loss converges to a better adapter when it starts from the
plain-MSE adapter.  The refine step uses the whitened adapter after 8 epochs; later snapshots start better but
converge to the same or worse WER.  The refine schedule runs to its end: the last points are made in the annealing
tail, and early stopping with a patience of 10 cuts a run off before it.
