"""Step 1: filtered 1000-Hz EMG cache + manifest from Gaddy & Klein's silent-speech corpus.

Front-end = Gaddy's read_emg.load_utterance() up to (not including) his subsampling step:
neighbour-utterance concatenation -> 60-Hz notch and harmonics (1..7, Q 30, filtfilt) -> 3rd-order
Butterworth 2-Hz high-pass (filtfilt) -> cut back to the utterance.  Gaddy then subsamples to 689 Hz
(8 x his 86-Hz audio frame rate); our frame rate is Granite's 50 Hz (20 ms), for which 1000 Hz / 20 is
exact, so the cache stays at 1000 Hz.  Gaddy's amplitude scaling (x/20, 50*tanh(x/50)) is applied at
load time (emgfm.data.gaddy_scale), not here.

Output: <out>/<dirtag>/<session>/<idx>.npy   float32 (T_ms, 8)
        <out>/manifest.csv                    one row per utterance
Split: dev/test by [book, sentence_index] membership in Gaddy's testset_largedev.json (the MONA-LISA
protocol; its test list has one duplicate key -> 99 unique test utterances per condition).
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gaddy.gaddy_emg import apply_to_all, notch_harmonics, remove_drift  # noqa: E402
from gaddy.gaddy_text import TextTransform  # noqa: E402

DIRS = {"silent_parallel": ("silent_parallel_data", True),        # Gaddy's open-vocabulary directories
        "voiced_parallel": ("voiced_parallel_data", False),
        "nonparallel": ("nonparallel_data", False)}
COLS = ["uid", "dirtag", "session", "idx", "silent", "split", "book", "sentence_index", "text", "text_clean",
        "n_ms", "n_enc", "n_tok", "twin_uid", "audio_path", "emg_path"]


def load_filtered_emg(base_dir, index):
    """read_emg.load_utterance() from 'raw_emg = ...' to the cut-back line, minus the subsampling."""
    index = int(index)
    raw_emg = np.load(os.path.join(base_dir, f"{index}_emg.npy"))
    before, after = os.path.join(base_dir, f"{index-1}_emg.npy"), os.path.join(base_dir, f"{index+1}_emg.npy")
    raw_emg_before = np.load(before) if os.path.exists(before) else np.zeros([0, raw_emg.shape[1]])
    raw_emg_after = np.load(after) if os.path.exists(after) else np.zeros([0, raw_emg.shape[1]])
    x = np.concatenate([raw_emg_before, raw_emg, raw_emg_after], 0)
    x = apply_to_all(notch_harmonics, x, 60, 1000)
    x = apply_to_all(remove_drift, x, 1000)
    x = x[raw_emg_before.shape[0]:x.shape[0] - raw_emg_after.shape[0], :]
    return x.astype(np.float32)


def enc_geometry(n_ms):
    """Granite's feature-extractor arithmetic for audio of the same duration: n16 = 16*n_ms samples ->
    mel_len = n16//160 + 1 -> enc = mel_len//2 -> tokens = ceil(enc/15)*3."""
    n_enc = (n_ms // 10 + 1) // 2
    return n_enc, math.ceil(n_enc / 15) * 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Gaddy corpus root (contains silent_parallel_data/ etc.)")
    ap.add_argument("--testset", default=f"{ROOT}/data/testset_largedev.json")
    ap.add_argument("--out", default=f"{ROOT}/cache/emg1k")
    args = ap.parse_args()
    ts = json.load(open(args.testset))
    dev = {tuple(k) for k in ts["dev"]}
    test = {tuple(k) for k in ts["test"]}
    tt = TextTransform()
    rows, t0 = [], time.time()
    for dirtag, (dname, silent) in DIRS.items():
        for sess in sorted(os.listdir(f"{args.data}/{dname}")):
            sdir = f"{args.data}/{dname}/{sess}"
            infos = sorted(glob.glob(f"{sdir}/*_info.json"), key=lambda p: int(os.path.basename(p).split("_")[0]))
            os.makedirs(f"{args.out}/{dirtag}/{sess}", exist_ok=True)
            n_sess = 0
            for ip in infos:
                info = json.load(open(ip))
                if info["sentence_index"] < 0:          # boundary silence (read_emg.py skips it too)
                    continue
                idx = int(os.path.basename(ip).split("_")[0])
                x = load_filtered_emg(sdir, idx)
                emg_path = f"{args.out}/{dirtag}/{sess}/{idx}.npy"
                np.save(emg_path, x)
                key = (info["book"], info["sentence_index"])
                split = "test" if key in test else "dev" if key in dev else "train"
                n_enc, n_tok = enc_geometry(x.shape[0])
                rows.append(dict(uid=f"{dirtag}/{sess}/{idx}", dirtag=dirtag, session=sess, idx=idx,
                                 silent=int(silent), split=split, book=info["book"],
                                 sentence_index=info["sentence_index"], text=info["text"],
                                 text_clean=tt.clean_text(info["text"]), n_ms=x.shape[0], n_enc=n_enc,
                                 n_tok=n_tok, twin_uid="", audio_path=f"{sdir}/{idx}_audio_clean.flac",
                                 emg_path=emg_path))
                n_sess += 1
            print(f"{dirtag}/{sess}: {n_sess} utterances  ({time.time()-t0:.0f}s)", flush=True)
    # vocal/silent twins of the parallel sentences
    by_key = {(r["dirtag"], r["book"], r["sentence_index"]): r["uid"] for r in rows
              if r["dirtag"] in ("silent_parallel", "voiced_parallel")}
    for r in rows:
        other = {"silent_parallel": "voiced_parallel", "voiced_parallel": "silent_parallel"}.get(r["dirtag"])
        if other:
            r["twin_uid"] = by_key.get((other, r["book"], r["sentence_index"]), "")
    with open(f"{args.out}/manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    print("\n=== summary ===")
    for split in ["train", "dev", "test"]:
        for dirtag in DIRS:
            sel = [r for r in rows if r["split"] == split and r["dirtag"] == dirtag]
            if sel:
                print(f"{split:5s} {dirtag:16s} n={len(sel):5d}  {sum(r['n_ms'] for r in sel)/3.6e6:.2f} h  "
                      f"twins {sum(bool(r['twin_uid']) for r in sel)}")
    print(f"total {len(rows)} utterances; wrote {args.out}/manifest.csv")


if __name__ == "__main__":
    main()
