"""Step 1 for the emg2speech General Corpus (Gowda, Comstock & Miller, ACL 2026; OSF 65vbx, CC-BY 4.0): per-utterance
EMG and audio caches plus a manifest in the format of prep_emg_cache.py, so that prep_mel_targets.py, train.py,
evaluate.py and ensemble.py run on this corpus unchanged (train with --in-ch 31 --conv-strides 4,5,5;
recipes/emg2speech.sh has the full commands).

  EMG    DATA.pkl: a list of 9,660 arrays, 31 channels at 5 kHz (float64, ~40 GB; this script needs ~64 GB of RAM)
         -> <out>/emg5k/voiced_parallel/s0/<idx>.npy, float32 (T, 31).  The release is already band-passed
         80-1000 Hz (inspection 2026-09-20: 96.8 % of the power in that band, none below 10 Hz) and zero-mean, but
         carries a 50 Hz mains line 13x above its neighbours, so Gaddy's notch (50 Hz and harmonics 1-7, Q 30,
         zero-phase; --notch 50, 0 = off) is applied as in prep_emg_cache.py.  Each utterance is then z-scored per
         channel (the authors' own normalisation) and multiplied by --gain (default 20), so that the loader's Gaddy
         scaling (x / 20, then 50 tanh(x / 50)) returns unit variance and only compresses beyond 2.5 sigma.
         --bandpass 1 would add the authors' Butterworth 80-1000 Hz; off by default because it is already applied.
  audio  groundTruthAudioFiles.pkl: 9,660 int16 mono arrays at 44.1 kHz (the rate is in the authors' notebook).
         The audio is systematically 1.15 % shorter than the EMG (length ratio 8.719 +- 0.026 against the nominal
         8.82): a clock mismatch between the recorders, so each clip is resampled to EXACTLY the EMG duration at
         16 kHz (a linear stretch of 1.15 %, 0.2 semitones) and written to <out>/audio16k/<idx>.flac, so that the
         align stage pairs Granite frames one-to-one with EMG frames.
  split  the authors' sequential split, used by Myovox as well: [:8500] train, [8500:9260] dev, [9260:] test.
  manifest.csv: the columns of prep_emg_cache.py; dirtag voiced_parallel (every utterance is vocalised, so the
         align stage targets each utterance's own audio and needs no twins), session s0, book willett,
         sentence_index = index in the release, text_clean = Gaddy's normaliser (the identity on these labels).
20 ms frames = 100 samples at 5 kHz, exactly one Granite encoder frame; n_enc follows enc_geometry() as for Gaddy.
"""
import argparse
import csv
import os
import pickle
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, resample

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gaddy.gaddy_emg import apply_to_all, notch_harmonics  # noqa: E402
from gaddy.gaddy_text import TextTransform  # noqa: E402
from scripts.prep_emg_cache import COLS, enc_geometry  # noqa: E402

FS_EMG, FS_AUDIO, FS_OUT = 5000, 44100, 16000
SPLITS = [("train", 0, 8500), ("dev", 8500, 9260), ("test", 9260, 9660)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="GeneralCorpusData/ of the OSF release (DATA.pkl, textLABELS.pkl, groundTruthAudioFiles.pkl)")
    ap.add_argument("--out", default=f"{ROOT}/cache/emg2speech")
    ap.add_argument("--gain", type=float, default=20.0)
    ap.add_argument("--notch", type=float, default=50.0, help="mains frequency for Gaddy's notch of harmonics 1-7 (0 = off)")
    ap.add_argument("--bandpass", type=int, default=0, choices=[0, 1], help="1 = Butterworth 80-1000 Hz, order 3, zero-phase")
    ap.add_argument("--limit", type=int, default=0, help="first N utterances only (smoke test)")
    args = ap.parse_args()
    t0 = time.time()
    texts = pickle.load(open(f"{args.data}/textLABELS.pkl", "rb"))
    audio = pickle.load(open(f"{args.data}/groundTruthAudioFiles.pkl", "rb"))
    emg = pickle.load(open(f"{args.data}/DATA.pkl", "rb"))
    assert len(texts) == len(audio) == len(emg) == 9660, (len(texts), len(audio), len(emg))
    print(f"loaded {len(emg)} utterances in {time.time() - t0:.0f}s", flush=True)
    n = args.limit or len(emg)
    tt = TextTransform()
    dirtag, sess = "voiced_parallel", "s0"
    os.makedirs(f"{args.out}/emg5k/{dirtag}/{sess}", exist_ok=True)
    os.makedirs(f"{args.out}/audio16k", exist_ok=True)
    if args.bandpass:
        b, a = butter(3, [80, 1000], "bandpass", fs=FS_EMG)
    rows, ratios = [], []
    for i in range(n):
        x = np.asarray(emg[i], dtype=np.float64)
        if x.shape[0] == 31:                                  # (31, T) in the release -> (T, 31)
            x = x.T
        assert x.shape[1] == 31, x.shape
        if args.bandpass:
            x = filtfilt(b, a, x, axis=0)
        if args.notch:
            x = apply_to_all(notch_harmonics, x, args.notch, FS_EMG)
        x = (x - x.mean(0)) / (x.std(0) + 1e-9) * args.gain
        emg_path = f"{args.out}/emg5k/{dirtag}/{sess}/{i}.npy"
        np.save(emg_path, x.astype(np.float32))
        src = np.asarray(audio[i], dtype=np.float64) / 32768.0
        ratios.append(len(src) / x.shape[0])
        n_out = int(round(x.shape[0] / FS_EMG * FS_OUT))     # the EMG duration at 16 kHz: frames pair one-to-one
        wav = resample(src, n_out)
        audio_path = f"{args.out}/audio16k/{i}.flac"
        sf.write(audio_path, np.clip(wav, -1, 1).astype(np.float32), FS_OUT)
        split = next(s for s, lo, hi in SPLITS if lo <= i < hi)
        n_ms = int(round(x.shape[0] / FS_EMG * 1000))
        n_enc, n_tok = enc_geometry(n_ms)
        rows.append(dict(uid=f"{dirtag}/{sess}/{i}", dirtag=dirtag, session=sess, idx=i, silent=0, split=split,
                         book="willett", sentence_index=i, text=texts[i], text_clean=tt.clean_text(texts[i]),
                         n_ms=n_ms, n_enc=n_enc, n_tok=n_tok, twin_uid="", audio_path=audio_path, emg_path=emg_path))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    with open(f"{args.out}/manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    for split, _, _ in SPLITS:
        sel = [r for r in rows if r["split"] == split]
        if sel:
            print(f"{split:5s} n={len(sel):5d}  {sum(r['n_ms'] for r in sel) / 3.6e6:.2f} h  "
                  f"mean {np.mean([r['n_ms'] for r in sel]) / 1000:.2f} s")
    r = np.array(ratios)
    print(f"audio/EMG length ratio: mean {r.mean():.4f} sd {r.std():.4f} (nominal {FS_AUDIO / FS_EMG:.2f}); "
          f"audio stretched by {100 * (FS_AUDIO / FS_EMG / r.mean() - 1):+.2f} % to the EMG duration")
    print(f"wrote {args.out}/manifest.csv ({len(rows)} rows) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
