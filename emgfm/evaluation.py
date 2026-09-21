"""Word error rate the way Gaddy and MONA-LISA compute it, bootstrap intervals, and the training-time
dev evaluation."""
import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from gaddy.gaddy_text import TextTransform

from .data import collate
from .losses import mel_loss

normalise = TextTransform().clean_text     # Gaddy's normaliser: unidecode, strip punctuation, lower-case


def word_errors(ref, hyp):
    """Minimal substitutions + deletions + insertions between two normalised strings, in words."""
    return Levenshtein.distance(ref.split(), hyp.split())


def wer(refs, hyps):
    """Corpus WER = total word errors / total reference words (identical to jiwer.wer)."""
    return sum(word_errors(r, h) for r, h in zip(refs, hyps)) / sum(len(r.split()) for r in refs)


def bootstrap_wer(errs, lens, n_boot=4000, seed=0):
    """(WER, 2.5 %, 97.5 %) by resampling utterances.  errs may be fractional (e.g. a mean over seeds)."""
    errs, lens = np.asarray(errs, float), np.asarray(lens, float)
    rng = np.random.default_rng(seed)
    n = len(lens)
    bs = [errs[i].sum() / lens[i].sum() for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return errs.sum() / lens.sum(), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_bootstrap(errs_a, errs_b, lens, n_boot=4000, seed=0):
    """(WER_a - WER_b, 2.5 %, 97.5 %), resampling the SAME utterances for both systems."""
    ea, eb, lens = (np.asarray(x, float) for x in (errs_a, errs_b, lens))
    rng = np.random.default_rng(seed)
    n = len(lens)
    d = [(ea[i].sum() - eb[i].sum()) / lens[i].sum() for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return (ea.sum() - eb.sum()) / lens.sum(), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


@torch.no_grad()
def evaluate_dev(dset, adapter, front, inj, device, bs, max_new_tokens, pca=None):
    """Training-time evaluation of one condition: CE, greedy WER, mel R^2 where targets exist, and the
    per-session WERs.  Batched for speed; the reported numbers (scripts/evaluate.py) use batch 1."""
    adapter.eval()
    order = np.argsort([int(n) for n in dset.df.n_ms.values])
    refs, hyps, uids, sessions = [], [], [], []
    ce_sum, tok_sum, mel_sum, mel_n = 0.0, 0, 0.0, 0
    for s in range(0, len(order), bs):
        c = collate([dset[int(i)] for i in order[s:s + bs]])
        emg, n_enc = c["emg"].to(device), c["n_enc"].to(device)
        y, z = inj(adapter, emg, n_enc)
        if any(t is not None for t in c["mel"]):
            m, npair, _ = mel_loss(z, c, inj, device, pca)
            mel_sum += float(m) * npair
            mel_n += npair
        feats = [y[i, :int(n_enc[i])] for i in range(len(c["texts"]))]
        loss, n = front.loss(feats, c["texts"])
        ce_sum += float(loss) * n
        tok_sum += n
        hyps += [normalise(h) for h in front.generate(feats, max_new_tokens)]
        refs += c["texts"]
        uids += c["uids"]
        sessions += c["sessions"]
    res = dict(n=len(refs), ce=ce_sum / max(tok_sum, 1), wer=wer(refs, hyps),
               mel=mel_sum / mel_n if mel_n else float("nan"),
               per_session={}, samples=list(zip(uids, refs, hyps)))
    for sess in sorted(set(sessions)):
        idx = [i for i, x in enumerate(sessions) if x == sess]
        res["per_session"][sess] = (len(idx), wer([refs[i] for i in idx], [hyps[i] for i in idx]))
    return res
