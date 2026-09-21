"""Manifest, utterance dataset, length-bucketed batching.

The manifest (cache/emg1k/manifest.csv, written by scripts/prep_emg_cache.py) has one row per
utterance: uid, dirtag, session, idx, split, book, sentence_index, text, text_clean, n_ms, n_enc,
twin_uid, audio_path, emg_path.  Splits follow Gaddy's testset_largedev.json (the MONA-LISA protocol).
"""
import os

import numpy as np
import pandas as pd
import torch

from . import SILENT


def gaddy_scale(x):
    """Gaddy's amplitude scaling, read_emg.EMGDataset.__getitem__: x/20 then 50*tanh(x/50)."""
    x = x / 20
    return 50 * np.tanh(x / 50.)


def load_manifest(path):
    df = pd.read_csv(path, dtype={"session": str})
    return df[df.text_clean.fillna("") != ""]        # 4 utterances have an empty normalised text


def twin_index(df):
    """uid -> (dirtag, session, idx): resolves a silent utterance's vocalised twin (same sentence)."""
    return {str(r.uid): (r.dirtag, str(r.session), int(r.idx)) for r in df.itertuples()}


class EMGSet:
    """EMG plus, when wanted, the spectrogram target for the align stage.

    mel_cache      directory written by scripts/prep_mel_targets.py: Granite's own input features
                   (160-d, float16) for every utterance that has speech audio.  "" = no targets.
    silent_targets a silent utterance has no usable audio; with True its target is the vocalised
                   TWIN's features, which have a different duration (0.92x on average), so the loss
                   pairs frames with Gaddy's monotonic DTW (item["mel_dtw"] = True).
    """

    def __init__(self, df, mel_cache="", twin=None, silent_targets=True):
        self.df = df.reset_index(drop=True)
        self.mel_cache = mel_cache
        self.twin = twin if twin is not None else {}
        self.silent_targets = silent_targets

    def __len__(self):
        return len(self.df)

    def target_path(self, r):
        """(path, needs_dtw) of the spectrogram target of manifest row r, or ("", False)."""
        if not self.mel_cache:
            return "", False
        if r.dirtag == SILENT:
            if not self.silent_targets:
                return "", False
            t = self.twin.get(getattr(r, "twin_uid", ""))
            if t is None:
                return "", False
            p = f"{self.mel_cache}/{t[0]}/{t[1]}/{t[2]}.npy"
            return (p, True) if os.path.exists(p) else ("", False)
        p = f"{self.mel_cache}/{r.dirtag}/{r.session}/{r.idx}.npy"
        return (p, False) if os.path.exists(p) else ("", False)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = gaddy_scale(np.load(r.emg_path)).astype(np.float32)
        item = dict(emg=torch.from_numpy(x), n_ms=int(r.n_ms), n_enc=int(r.n_enc),
                    text=str(r.text_clean), uid=str(r.uid), session=str(r.session),
                    mel=None, mel_dtw=False)
        p, dtw = self.target_path(r)
        if p:
            item["mel"] = torch.from_numpy(np.load(p).astype(np.float32))
            item["mel_dtw"] = dtw
        return item


def make_batches(n_ms, batch_ms, max_bs, rng, jitter=0.1):
    """Length-bucketed batches under a padded-size budget: max_len * batch_size <= batch_ms.
    Lengths are jittered by up to 10 % before sorting so buckets change between epochs."""
    n_ms = np.asarray(n_ms, dtype=np.float64)
    order = np.argsort(n_ms * (1 + jitter * rng.random(len(n_ms))))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = n_ms[i]
        if cur and (max(cur_max, L) * (len(cur) + 1) > batch_ms or len(cur) >= max_bs):
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(int(i))
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def collate(items):
    B, T = len(items), max(it["emg"].shape[0] for it in items)
    emg = torch.zeros(B, T, items[0]["emg"].shape[1])
    for i, it in enumerate(items):
        emg[i, :it["emg"].shape[0]] = it["emg"]
    return dict(emg=emg, n_enc=torch.tensor([it["n_enc"] for it in items]),
                texts=[it["text"] for it in items], uids=[it["uid"] for it in items],
                sessions=[it["session"] for it in items], n_ms=[it["n_ms"] for it in items],
                mel=[it["mel"] for it in items], mel_dtw=[it["mel_dtw"] for it in items])
