"""CPU unit tests (no model download): python tests/test_units.py"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from emgfm.adapter import EMGAdapter  # noqa: E402
from emgfm.checkpoint import arg, check_supported  # noqa: E402
from emgfm.data import collate, gaddy_scale, make_batches  # noqa: E402
from emgfm.evaluation import bootstrap_wer, paired_bootstrap, wer, word_errors  # noqa: E402
from emgfm.granite import build_ids, n_tokens  # noqa: E402
from emgfm.losses import mel_loss  # noqa: E402
from gaddy.gaddy_align import align_from_distances  # noqa: E402
from gaddy.gaddy_text import TextTransform  # noqa: E402


def test_text_normaliser():
    tt = TextTransform()
    assert tt.clean_text("Woking's \"district\" -- 12 o'clock!") == "wokings district  12 oclock"
    assert wer(["a b c"], ["a x c d"]) == 2 / 3
    assert word_errors("keep back", "keep him back") == 1


def test_build_ids():
    ids, am, lab = build_ids([1, 2], [3], 9, [2, 2], [[7, 7], [8]], pad_id=0, eos_id=5)
    assert ids.shape == (2, 8) and am.sum(1).tolist() == [8, 7]      # left padding
    assert ids[0].tolist() == [1, 2, 9, 9, 3, 7, 7, 5] and ids[1].tolist() == [0, 1, 2, 9, 9, 3, 8, 5]
    assert lab[0].tolist() == [-100] * 5 + [7, 7, 5]
    assert lab[1].tolist() == [-100] * 6 + [8, 5]                    # answer 8 + EOS
    assert n_tokens(15) == 3 and n_tokens(16) == 6


def test_batches_and_collate():
    n_ms = np.array([1000, 2000, 3000, 4000, 9000])
    b = make_batches(n_ms, batch_ms=8000, max_bs=32, rng=np.random.default_rng(0), jitter=0.0)
    assert sorted(i for bb in b for i in bb) == list(range(5))
    for bb in b:
        assert n_ms[bb].max() * len(bb) <= 8000 or len(bb) == 1     # an over-long utterance goes alone
    items = [dict(emg=torch.ones(200, 8), n_ms=200, n_enc=10, text="a", uid="u1", session="s", mel=None,
                  mel_dtw=False),
             dict(emg=torch.ones(100, 8), n_ms=100, n_enc=5, text="b", uid="u2", session="s", mel=None,
                  mel_dtw=False)]
    c = collate(items)
    assert c["emg"].shape == (2, 200, 8) and c["emg"][1, 100:].abs().sum() == 0
    assert np.isclose(gaddy_scale(np.array([1000.0])), 50 * np.tanh(1.0))


def test_adapter_shapes_and_masking():
    torch.manual_seed(0)
    ad = EMGAdapter((8, 8, 8), (3, 3, 3), (2, 2, 5), tf_layers=1, tf_dim=8, heads=2, ff=16, dropout=0.0,
                    out_dim=6).eval()
    emg = torch.randn(2, 400, 8)
    n_enc = torch.tensor([20, 12])
    y = ad(emg, n_enc)
    assert y.shape == (2, 20, 6)
    assert y[1, 12:].abs().sum() == 0 and y[1, :12].abs().sum() > 0
    y2 = ad(emg[:1], n_enc[:1])                          # batching does not change a valid frame
    assert torch.allclose(y[0], y2[0], atol=1e-5)


def test_mel_loss_reference_points():
    """Constant-mean predictor -> 1.0 in both metrics; exact match -> 0; whitened metric per direction."""
    rng = np.random.default_rng(0)
    n, d = 5000, 6
    tgt_raw = rng.normal(size=(n, d)) * np.array([5, 3, 2, 1, 0.5, 0.2]) + 1.0
    mu, sd = tgt_raw.mean(0), tgt_raw.std(0)

    class Inj:
        pass
    inj = Inj()
    inj.mu, inj.sd = torch.tensor(mu, dtype=torch.float32), torch.tensor(sd, dtype=torch.float32)
    z_t = (tgt_raw - mu) / sd
    ev, V = np.linalg.eigh(np.cov(z_t.T, bias=True))
    o = np.argsort(ev)[::-1]
    pca = (torch.tensor(V[:, o], dtype=torch.float32), torch.tensor(ev[o], dtype=torch.float32))
    batch = dict(mel=[torch.tensor(tgt_raw, dtype=torch.float32)], mel_dtw=[False], n_enc=torch.tensor([n]))
    zeros = torch.zeros(1, n, d)                          # z = 0 is the constant-mean predictor
    for p in (None, pca):
        loss, npair, ndtw = mel_loss(zeros, batch, inj, "cpu", p)
        assert abs(float(loss) - 1.0) < 0.02, float(loss)
        exact, _, _ = mel_loss(torch.tensor(z_t, dtype=torch.float32)[None], batch, inj, "cpu", p)
        assert float(exact) < 1e-8 and npair == n and ndtw == 0
    empty, npair, _ = mel_loss(zeros.requires_grad_(), dict(mel=[None], mel_dtw=[False], n_enc=torch.tensor([n])),
                               inj, "cpu")
    assert float(empty.detach()) == 0 and empty.requires_grad and npair == 0


def test_dtw_pairs_monotonically():
    costs = np.abs(np.arange(6)[:, None] - np.arange(9)[None, :] * 6 / 9).astype(np.float64)
    al = align_from_distances(np.ascontiguousarray(costs.T))
    assert len(al) == 9 and all(al[i] <= al[i + 1] for i in range(8)) and al[-1] == 5


def test_bootstrap_and_paired():
    errs, lens = np.array([1, 0, 2, 0]), np.array([4, 5, 6, 5])
    w, lo, hi = bootstrap_wer(errs, lens, n_boot=200)
    assert abs(w - 3 / 20) < 1e-12 and lo <= w <= hi
    d, lo, hi = paired_bootstrap(errs, errs, lens, n_boot=50)
    assert d == 0 and lo == 0 and hi == 0


def test_args_compat():
    old = dict(lr=1e-4, train_encoder="lora", early_stop_metric="aux_in", in_cache="/x", inject="encoder_in")
    assert arg(old, "adapter_lr") == 1e-4 and arg(old, "encoder") == "lora" and arg(old, "select") == "mel"
    assert arg(old, "mel_cache") == "/x" and arg(old, "input_linear_fp32", 0) == 0
    check_supported(old)
    try:
        check_supported(dict(old, inject="encoder_out"))
        assert False, "encoder_out must be refused"
    except SystemExit:
        pass


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
