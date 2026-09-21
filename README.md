# emg-fm-asr — speech recognition from surface EMG through a speech foundation model

Open-vocabulary speech recognition from surface EMG of the face and neck by routing the EMG **through a pretrained
speech foundation model**: a small adapter turns EMG into the spectrogram features the model expects, the model's
encoder is adapted with LoRA, and its LLM decodes — frozen.  One recipe, two public corpora.

```
EMG ─► EMGAdapter ─► predicted mel features (160-d / 20 ms)
       ─► Granite Speech 4.0-1B encoder (LoRA r16 + input projection) ─► projector, LLM (frozen) ─► text
```

Two training stages, **align then let go**:

1. **align** — the adapter alone learns to reproduce the spectrogram of the *spoken* utterance (a silent utterance is
   paired to its vocalised twin by Gaddy's monotonic DTW); Granite is frozen, no transcript is used.  Plain MSE
   first, then a whitened loss warm-started from it; its epoch-8 snapshot initialises the next stage.
2. **refine** — adapter and encoder LoRA train on Granite's own next-token loss, with **no** spectrogram term
   (keeping one costs 3.5 points of silent WER), over a full 120-epoch cosine **without early stopping**.

## Results (dev sets, exact protocol below; each test set is evaluated once and is still pending)

**Gaddy & Klein's silent-speech corpus** (single speaker, 8 channels at 1 kHz; 199 dev utterances per condition):

| system | silent WER | vocal WER |
|---|---|---|
| single model, final recipe, seeds 0 / 1, beam 4 | **12.8 / 13.1** | **5.8 / 6.3** |
| ensemble of 18 checkpoints, MBR, no external LM | **10.3** | **5.1** |
| align stage alone, Granite frozen, no transcript ever used | 56.5 | 26.6 |

The 18 ensemble members were accumulated during development (the final recipe, earlier 60/80-epoch seeds, and
variants that are not part of this repository); the unbiased mean over random subsets still falls with size
(11.4 at 4 members, 10.2 at 16).  Reference points (MONA-LISA, Benster et al. 2024): best single MONA model 20.6
dev / 22.2 test with a 4-gram LM; ensemble of 10 + GPT-3.5 13.1 dev; 12.2 silent / 3.7 vocal **test** with GPT-3.5
fine-tuned on the validation set.

**emg2speech General Corpus** (Gowda et al. 2026; single speaker, vocalised, 31 channels at 5 kHz; 760 dev sentences),
the same recipe with three input-geometry flags and no tuning:

| system | WER |
|---|---|
| single model, seeds 0 / 1, beam 4 (dev) | **9.2 / 10.0** |
| published, test: emg2text baseline / Myovox single model / Myovox ensemble + fine-tuned 7B reranker | 51.2 / 26.1 / 18.5 |

What the experiments established: the 120-epoch schedule beats 80 epochs by 0.98 silent points [0.09, 1.91] (2 vs 2
seeds) because the gains sit in the annealing tail, which early stopping cuts off; **two runs of the identical recipe
and seed differ by about one point** under the exact protocol, so single-run differences below that are not evidence.
Tested and not adopted (each as one change, at convergence): a spectrogram anchor in the refine stage, silent
oversampling, a representation loss at the encoder output, LoRA on the LLM, an auxiliary CTC loss on the encoder
output (-0.5 [-1.3, +0.3]), rescoring with local 8B language models (-0.4, interval includes 0), trailing-silence
padding, per-condition BatchNorm statistics.

## Layout

```
emgfm/     package: data, adapter, granite (front-end, LoRA, BatchNorm), losses, checkpoint, evaluation
gaddy/     Gaddy's text normaliser, EMG front-end and DTW, copied verbatim (MIT, see LICENSE.gaddy)
scripts/   prep_emg_cache, prep_mel_targets, prep_emg2speech, train, evaluate, ensemble, compare
recipes/   the exact commands behind the numbers (01_prep … 05_ensemble for Gaddy; emg2speech.sh)
slurm/     SLURM wrappers and env.sh (cluster paths)
data/      Gaddy's dev/test split file; the whitening basis the reported Gaddy runs used
tests/     CPU unit tests
```

`cache/` (features) and `runs/` (checkpoints, logs) are created by the recipes and never committed: Gaddy's corpus
is cleared for our chair's infrastructure only, and checkpoints derive from it.  On the chair cluster the models
behind the table are `runs/z_a0_il32_e120ns_s{0,1}/best.pt` and `runs/e2s_refine_s{0,1}/best.pt` under
`/data/chi-gpu8/ge62niq/emg-asr/`.

## Running

```bash
# Python 3.10, requirements.txt; download ibm-granite/granite-4.0-1b-speech once into HF_HOME; edit slurm/env.sh
EMG_DATA=/path/to/gaddy_corpus bash recipes/01_prep.sh   # filtered EMG + manifest, spectrogram targets
bash recipes/02_align.sh                                  # -> runs/align_whitened/snapshot_ep008.pt
bash recipes/03_refine.sh "0 1"                           # -> runs/refine_s*/best.pt  (~6 h each on an RTX PRO 6000)
bash recipes/04_evaluate.sh                               # -> runs/refine_s*/eval_dev.json
python scripts/compare.py runs/refine_s0/eval_dev.json,runs/refine_s1/eval_dev.json runs/<other>/eval_dev.json
bash recipes/05_ensemble.sh                               # n-best generation, pooled scoring, report

# emg2speech (public, CC-BY 4.0: https://osf.io/65vbx/, folder GeneralCorpusData; the prep step needs ~64 GB of RAM)
E2S_DATA=/path/to/GeneralCorpusData bash recipes/emg2speech.sh prep
bash recipes/emg2speech.sh mel && bash recipes/emg2speech.sh align       # wait for each step to finish
bash recipes/emg2speech.sh refine "0 1" && bash recipes/emg2speech.sh eval

python tests/test_units.py
```

Any run can be repeated from its `config.json`: `scripts/train.py --name x --from-config runs/<run>/config.json`.
Refine needs more than 24 GB of GPU memory; align fits 8 GB.

## Evaluation protocol (what every reported number follows)

* **BatchNorm.** Granite's conformer has a BatchNorm in each of its 16 conv modules; training runs the encoder in
  train mode, so their running statistics drift from audio to EMG.  They are saved with every checkpoint and, for
  reported numbers, **re-estimated on the training set** (`--bn recal`, the default).  The pretrained audio
  statistics cost 3–4 points, the saved moving averages up to 1.4.
* **Batch 1** for encoder, decoder and ensemble scoring: bf16 with padding changes outputs slightly with the batch.
* **One GPU type** for every model in a comparison (A40 vs Blackwell numerics differ by 0.1–0.35 points).
* Gaddy's normaliser on hypotheses and references; corpus WER; 95 % bootstrap interval over utterances; paired
  bootstrap for comparisons (`scripts/compare.py`, several seeds per side); beams 1, 4 and 8.
* **Dev for every decision; test once** (`--final-test`), with the same protocol.

## Notes

* `gaddy/` files are byte-identical extracts of github.com/dgaddy/silent_speech; the DTW boundary quirk of
  `align.py` is documented in `gaddy_align.py` and deliberately kept.
* `data/in16k_pca_basis.npz` is the whitening basis the reported Gaddy align runs used; for a new corpus
  `prep_mel_targets.py` fits one on all training frames.
* `--input-linear-fp32 1` keeps fp32 master weights for the encoder's input projection: trained in place in bf16 at
  lr 5e-5, most AdamW updates round away.
* emg2speech: the release is already band-passed (80–1000 Hz); `prep_emg2speech.py` adds a 50 Hz notch, z-scores each
  utterance per channel, and stretches each audio clip by 1.15 % to the EMG duration (the two recorders' clocks
  differ), so spectrogram and EMG frames pair one-to-one.
* All Gaddy dev/test sentences come from *The War of the Worlds*; LibriSpeech (in Granite's and MONA's training
  audio) contains two of the recorded chapters.  Sentences from those chapters are not easier for any model we
  tested, and no evaluation sentence occurs in the training transcripts beyond one two-word exclamation.
