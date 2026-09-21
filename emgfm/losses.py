"""The align-stage loss: predicted z-scored features against the real spectrogram of the spoken
utterance (Gaddy & Klein 2020's digital-voicing target, here in Granite's own input space)."""
import numpy as np
import torch

from gaddy.gaddy_align import align_from_distances


def mel_loss(z, batch, inj, device, pca=None):
    """Mean squared error between the adapter's z-scored prediction and the real features, z-scored with
    the same training statistics.  Vocalised utterances pair frame for frame; a silent utterance's
    target is its vocalised twin's spectrogram, paired by Gaddy's monotonic DTW on the current
    predictions' Euclidean distance matrix (recomputed every step, no gradient through the alignment).

    With pca=(V_k, ev_k) the error is measured in per-component standardised coordinates over the top-k
    principal directions of the training features -- mean over pairs and directions of
    ((pred - tgt) . v_j)^2 / ev_j -- so each direction counts equally.  In both metrics the constant-mean
    predictor scores 1.0 and an exact match 0, so `1 - loss` is a variance-explained (R^2).
    Returns (loss, n_pairs, n_dtw_rows); loss is a zero tensor with a graph if the batch has no targets."""
    sse, n_slots, n_pairs, n_dtw = None, 0, 0, 0
    for i, tgt in enumerate(batch["mel"]):
        if tgt is None:
            continue
        t = (tgt.to(device, torch.float32) - inj.mu) / inj.sd
        p = z[i, :int(batch["n_enc"][i])]
        if batch["mel_dtw"][i]:
            costs = torch.cdist(p.unsqueeze(0), t.unsqueeze(0)).squeeze(0)
            al = torch.as_tensor(align_from_distances(
                np.ascontiguousarray(costs.T.detach().cpu().numpy())), device=device)
            a_, b_ = p[al], t
            n_dtw += 1
        else:
            n = min(p.shape[0], t.shape[0])
            a_, b_ = p[:n], t[:n]
        if pca is not None:
            V, ev = pca
            c = ((a_ - b_) @ V) ** 2 / ev
            term, slots = c.sum(), c.shape[0] * c.shape[1]
        else:
            term, slots = ((a_ - b_) ** 2).sum(), a_.shape[0] * a_.shape[1]
        sse = term if sse is None else sse + term
        n_slots += slots
        n_pairs += a_.shape[0]
    if sse is None:
        return z.sum() * 0.0, 0, 0
    return sse / n_slots, n_pairs, n_dtw
