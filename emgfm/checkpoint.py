"""Loading a trained checkpoint back into a Granite model, and the BatchNorm protocol at evaluation.

Checkpoint contents (torch.save dict): adapter (state_dict), epoch, step, dev, args, and for the refine
stage encoder_trainable (LoRA + input_linear) and encoder_bn (the BatchNorm running statistics).
`args` may come from this repository (format 2) or from the original lab code (older keys); `arg()`
reads both, so every checkpoint that produced a reported number stays loadable.
"""
import numpy as np
import torch

from . import ENCODER_LORA_TARGETS
from .adapter import EMGAdapter
from .data import EMGSet, collate, load_manifest
from .granite import (GraniteFront, MelInjector, add_encoder_lora, input_linear_fp32, load_granite,
                      recalibrate_bn)

_OLD = {"adapter_lr": "lr", "text_weight": "text_lambda", "mel_weight": "aux_in_lambda",
        "mel_whiten": "aux_in_whiten", "mel_silent_targets": "aux_silent", "mel_cache": "in_cache",
        "encoder": "train_encoder", "select": "early_stop_metric"}
_OLD_VALUES = {"encoder": {"none": "frozen", "lora": "lora"},
               "select": {"ce": "ce", "wer": "wer", "aux_in": "mel"}}


def arg(a, name, default=None):
    """Read a training argument from a new- or old-format args dict."""
    if name in a:
        return a[name]
    old = _OLD.get(name)
    if old is not None and old in a:
        v = a[old]
        return _OLD_VALUES.get(name, {}).get(v, v) if isinstance(v, str) else v
    return default


def check_supported(a):
    """Old lab configs could enable paths this repository no longer has; refuse those checkpoints."""
    bad = []
    if a.get("inject", "encoder_in") != "encoder_in":
        bad.append("inject=encoder_out (the encoder-bypass route)")
    if a.get("out_mode", "linear") != "linear":
        bad.append(f"out_mode={a['out_mode']}")
    if a.get("in_pad", "floor") != "floor":
        bad.append(f"in_pad={a['in_pad']}")
    if a.get("aux_lambda", 0) or a.get("train_projector") or a.get("freeze_adapter"):
        bad.append("representation loss / projector training / frozen adapter")
    if (a.get("silent_oversample") or 1) != 1:
        bad.append("silent oversampling")
    if a.get("llm_lora_r"):
        bad.append("LLM LoRA")
    if a.get("train_encoder") == "full":
        bad.append("full encoder training")
    if bad:
        raise SystemExit("checkpoint uses options this repository does not implement: " + "; ".join(bad))


def build_adapter(a, device, dropout):
    ints = lambda s: tuple(int(x) for x in str(s).split(","))  # noqa: E731
    return EMGAdapter(ints(arg(a, "conv_dims")), ints(arg(a, "conv_kernels")), ints(arg(a, "conv_strides")),
                      arg(a, "tf_layers"), arg(a, "tf_dim"), arg(a, "tf_heads"), arg(a, "tf_ff"), dropout,
                      in_ch=arg(a, "in_ch", 8), time_shift=arg(a, "time_shift", 0),
                      time_mask=ints(arg(a, "time_mask", "0,0")), channel_mask=arg(a, "channel_mask", 0)).to(device)


def load_checkpoint(ckpt_path, device="cuda", mel_cache=None):
    """Granite + the checkpoint's adapter and encoder weights, exactly as trained, in eval mode.
    BatchNorm statistics are NOT set here -- call setup_bn next.
    Returns dict(model, processor, front, inj, adapter, ck, args)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    check_supported(a)
    model, processor = load_granite(device)
    trained_encoder = arg(a, "encoder", "frozen") == "lora"
    if trained_encoder:
        add_encoder_lora(model, arg(a, "lora_r"), arg(a, "lora_alpha"),
                         arg(a, "lora_targets", ENCODER_LORA_TARGETS), 0.0)
        if arg(a, "input_linear_fp32", 0):
            input_linear_fp32(model.encoder)          # before loading, so the fp32 weights load exactly
        res = model.encoder.load_state_dict(ck["encoder_trainable"], strict=False)
        assert not res.unexpected_keys, res.unexpected_keys
    model.eval()
    front = GraniteFront(model, processor.tokenizer, device)
    inj = MelInjector(model, f"{mel_cache or arg(a, 'mel_cache')}/in_stats.npz")
    adapter = build_adapter(a, device, 0.0).eval()
    adapter.load_state_dict(ck["adapter"])
    n_enc = len(ck.get("encoder_trainable", {}))
    print(f"{ckpt_path}: epoch {ck['epoch']}, dev at save {ck.get('dev')}; "
          f"encoder tensors loaded: {n_enc}", flush=True)
    return dict(model=model, processor=processor, front=front, inj=inj, adapter=adapter, ck=ck, args=a,
                trained_encoder=trained_encoder)


def setup_bn(R, mode="auto", manifest=None):
    """Choose the conformer's BatchNorm running statistics:
      recal       re-estimated on the run's own training set (the standard, reproducible protocol);
      saved       the EMA statistics stored in the checkpoint (noisy: last ~10 batches);
      pretrained  Granite's audio statistics -- correct only for a checkpoint whose encoder was never
                  trained (an align-stage adapter through the frozen encoder);
      auto        recal for a trained encoder, pretrained otherwise.
    Returns dict(mode, n_layers[, batches, pool])."""
    a, model = R["args"], R["model"]
    if mode == "auto":
        mode = "recal" if R["trained_encoder"] else "pretrained"
    info = dict(mode=mode, n_layers=16)
    if mode == "saved":
        assert "encoder_bn" in R["ck"], "this checkpoint carries no BatchNorm statistics; use recal"
        res = model.encoder.load_state_dict(R["ck"]["encoder_bn"], strict=False)
        assert not res.unexpected_keys, res.unexpected_keys
    elif mode == "recal":
        assert not arg(a, "limit_train", 0), "recalibration assumes the full training set"
        df = load_manifest(manifest or arg(a, "manifest"))
        tags = arg(a, "train_dirtags", "silent_parallel,voiced_parallel,nonparallel").split(",")
        tr = df[(df.split == "train") & df.dirtag.isin(tags)]
        if arg(a, "max_ms", 0):
            tr = tr[tr.n_ms <= arg(a, "max_ms")]
        ds = EMGSet(tr)
        info["batches"] = recalibrate_bn(model, R["adapter"], R["inj"], ds, np.arange(len(ds)),
                                         arg(a, "batch_ms"), arg(a, "max_bs"))
        info["pool"] = len(ds)
    elif mode != "pretrained":
        raise ValueError(mode)
    print(f"BatchNorm statistics: {info}", flush=True)
    return info


@torch.no_grad()
def encode(R, ds, bs=1, device="cuda"):
    """Encoder-output frames for every utterance of ds, in order: (list of (T_i, 1024), refs, uids)."""
    feats, refs, uids = [], [], []
    for s in range(0, len(ds), bs):
        c = collate([ds[i] for i in range(s, min(s + bs, len(ds)))])
        n_enc = c["n_enc"].to(device)
        y, _ = R["inj"](R["adapter"], c["emg"].to(device), n_enc)
        feats += [y[i, :int(n_enc[i])] for i in range(len(c["texts"]))]
        refs += c["texts"]
        uids += c["uids"]
    return feats, refs, uids


def condition_set(df, split, tag, n=0):
    """The utterances of one benchmark condition, shortest first (the evaluation order)."""
    d = df[(df.split == split) & (df.dirtag == tag)].sort_values("n_ms")
    if n:
        d = d.head(n)
    return EMGSet(d), d
