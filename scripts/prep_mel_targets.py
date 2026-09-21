"""Step 2: the align-stage targets -- Granite's own input features for every utterance with speech audio.

Each utterance is featurised ALONE (the extractor normalises per utterance), 22.05 kHz audio is
resampled to 16 kHz.  Also written, from the TRAINING utterances only:
  in_stats.npz   per-dimension mean / std -- the adapter predicts z-scores and is de-normalised with these;
  in_pca.npz     principal directions (V, 160 x 160) and variances (ev) of the z-scored features, for the
                 whitened align loss.  --pca-from FILE copies an existing basis instead of fitting one
                 (data/in16k_pca_basis.npz is the basis the reported runs used; it was fitted on 600 training
                 utterances and agrees with a fit on all training frames to |cos| >= 0.98 on the top 15
                 directions).
Output: <out>/<dirtag>/<session>/<idx>.npy  float16 (T, 160), plus the two npz files.
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from emgfm import MEL_DIM, SILENT  # noqa: E402
from emgfm.data import load_manifest  # noqa: E402
from emgfm.granite import load_granite  # noqa: E402


def features(processor, path, device):
    x, sr = sf.read(path)
    assert x.ndim == 1, (path, x.shape)
    if sr == 22050:
        x = resample_poly(x, 320, 441)
    else:
        assert sr == 16000, (path, sr)
    fe = processor.audio_processor([torch.from_numpy(np.asarray(x, np.float32))], device=device)
    return fe["input_features"][0].float()                       # (T, 160)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=f"{ROOT}/cache/emg1k/manifest.csv")
    ap.add_argument("--out", default=f"{ROOT}/cache/in16k")
    ap.add_argument("--pca-from", default="", help="copy this basis as in_pca.npz instead of fitting one")
    args = ap.parse_args()
    device = "cuda"
    _, processor = load_granite(device)
    m = load_manifest(args.manifest)
    m = m[m.dirtag != SILENT]                                     # silent utterances have no usable audio
    t0 = time.time()
    for r in m.itertuples():
        p = f"{args.out}/{r.dirtag}/{r.session}/{r.idx}.npy"
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.save(p, features(processor, r.audio_path, device).cpu().numpy().astype(np.float16))
    print(f"{len(m)} utterances featurised or found in {args.out} ({time.time()-t0:.0f}s)", flush=True)

    tr = m[m.split == "train"]
    s1, s2, n = np.zeros(MEL_DIM), np.zeros(MEL_DIM), 0
    for r in tr.itertuples():
        f = np.load(f"{args.out}/{r.dirtag}/{r.session}/{r.idx}.npy").astype(np.float64)
        s1 += f.sum(0); s2 += (f ** 2).sum(0); n += len(f)
    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 1e-8))
    np.savez(f"{args.out}/in_stats.npz", mean=mu.astype(np.float32), std=sd.astype(np.float32), n_frames=n,
             n_utts=len(tr))
    print(f"in_stats.npz from {len(tr)} training utterances ({n} frames)", flush=True)

    if args.pca_from:
        shutil.copy(args.pca_from, f"{args.out}/in_pca.npz")
        print(f"in_pca.npz copied from {args.pca_from}")
        return
    S, z1, n = np.zeros((MEL_DIM, MEL_DIM)), np.zeros(MEL_DIM), 0
    for r in tr.itertuples():
        z = (np.load(f"{args.out}/{r.dirtag}/{r.session}/{r.idx}.npy").astype(np.float64) - mu) / sd
        S += z.T @ z; z1 += z.sum(0); n += len(z)
    C = S / n - np.outer(z1 / n, z1 / n)
    ev, V = np.linalg.eigh(C)
    o = np.argsort(ev)[::-1]
    np.savez(f"{args.out}/in_pca.npz", V=V[:, o].astype(np.float32), ev=ev[o].astype(np.float32), n_frames=n,
             n_utts=len(tr))
    print(f"in_pca.npz fitted on {n} training frames; top eigenvalues {np.round(ev[o][:6], 2)}")


if __name__ == "__main__":
    main()
