"""Train the EMG adapter (align stage) or adapter + encoder LoRA (refine stage) into frozen Granite.

The stage is set by two loss weights:
    align    --text-weight 0 --mel-weight 1 --mel-whiten 16 --encoder frozen
             (the adapter learns spectrograms; Granite only reports the untrained-for text CE)
    refine   --text-weight 1 --mel-weight 0 --encoder lora --init-from <align checkpoint>
             (Granite's next-token loss alone; --mel-weight > 0 here is the "anchor" ablation)
See recipes/ for the exact commands behind the reported numbers, and --from-config to rerun any
run from its config.json (this repository's or the original lab code's).

Outputs in <runs-dir>/<name>/: config.json, steps.csv, epochs.csv, dev_sessions.csv,
samples_epochNNN.txt, last.pt, best.pt (by --select on dev: silent if the corpus has it, else vocal),
snapshot_epNNN.pt (--snapshot-epochs).
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from emgfm import CONDITIONS, ENCODER_LORA_TARGETS, MODEL  # noqa: E402
from emgfm.adapter import EMGAdapter  # noqa: E402
from emgfm.checkpoint import _OLD, _OLD_VALUES, check_supported  # noqa: E402
from emgfm.data import EMGSet, collate, load_manifest, make_batches, twin_index  # noqa: E402
from emgfm.evaluation import evaluate_dev  # noqa: E402
from emgfm.granite import (GraniteFront, MelInjector, add_encoder_lora, bn_layers, bn_state,  # noqa: E402
                           input_linear_fp32, load_granite, load_whitening)
from emgfm.losses import mel_loss  # noqa: E402

DEFAULTS = dict(
    seed=0, manifest=f"{ROOT}/cache/emg1k/manifest.csv", mel_cache=f"{ROOT}/cache/in16k",
    train_dirtags="silent_parallel,voiced_parallel,nonparallel", max_ms=0, limit_train=0, dev_subset=0,
    batch_ms=100000, max_bs=32, eval_bs=32, max_new_tokens=200,
    in_ch=8, conv_dims="256,512,768", conv_kernels="9,9,11", conv_strides="2,2,5", tf_layers=6, tf_dim=768,
    tf_heads=8, tf_ff=3072, dropout=0.2, time_mask="2,300", channel_mask=1, time_shift=11,
    adapter_lr=1e-4, text_weight=1.0, mel_weight=0.0, mel_whiten=0, mel_silent_targets=1,
    init_from="", encoder="lora", encoder_lr=5e-5, lora_r=16, lora_alpha=32,
    lora_targets=ENCODER_LORA_TARGETS, input_linear_fp32=0, bn_momentum=0.1,
    wd=0.01, warmup_steps=300, epochs=60, grad_clip=1.0, accum=1,
    eval_every=1, no_eval_init=False, early_stop=10, select="wer", snapshot_epochs="", log_every=100,
    runs_dir=f"{ROOT}/runs")
_IGNORED = {"name", "n_train", "n_dev_silent", "n_dev_vocal", "trainable_params", "prompt", "model", "gpu",
            "format", "aux_lambda", "aux_whiten", "enc_cache", "inject", "out_mode", "in_pad", "freeze_adapter",
            "train_projector", "projector_lr", "oracle_check", "silent_oversample", "llm_lora_r",
            "llm_lora_alpha", "llm_lora_targets", "llm_lr", "code", "from_config", "ctc_weight", "measure_grads"}


def translate_config(cfg):
    """A config.json of this repository or of the original lab code -> defaults for this parser."""
    check_supported(cfg)
    if cfg.get("ctc_weight"):
        raise SystemExit("this config used an auxiliary CTC loss, which this repository does not implement")
    new_of_old = {v: k for k, v in _OLD.items()}
    out = {}
    for k, v in cfg.items():
        if k in _IGNORED:
            continue
        nk = new_of_old.get(k, k)
        if nk in _OLD_VALUES and isinstance(v, str):
            v = _OLD_VALUES[nk].get(v, v)
        if nk not in DEFAULTS:
            raise SystemExit(f"config key {k!r} is not a flag of this trainer")
        out[nk] = v
    return out


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--from-config", default="", help="start from a run's config.json; flags override")
    g = ap.add_argument_group("data")
    g.add_argument("--seed", type=int)
    g.add_argument("--manifest")
    g.add_argument("--mel-cache", help="prep_mel_targets.py output: features, in_stats.npz, in_pca.npz")
    g.add_argument("--train-dirtags")
    g.add_argument("--max-ms", type=int, help="drop longer training utterances (0 = keep all)")
    g.add_argument("--limit-train", type=int, help="random subset of the training set (smoke tests)")
    g.add_argument("--dev-subset", type=int, help="per condition (0 = all)")
    g.add_argument("--batch-ms", type=int, help="padded-size budget per batch: max_len(ms) * batch_size")
    g.add_argument("--max-bs", type=int)
    g.add_argument("--eval-bs", type=int)
    g.add_argument("--max-new-tokens", type=int)
    g = ap.add_argument_group("adapter")
    g.add_argument("--in-ch", type=int, help="EMG channels (Gaddy 8; emg2speech 31)")
    g.add_argument("--conv-dims"); g.add_argument("--conv-kernels")
    g.add_argument("--conv-strides", help="product = EMG samples per 20 ms frame (1 kHz: 2,2,5; 5 kHz: 4,5,5)")
    g.add_argument("--tf-layers", type=int); g.add_argument("--tf-dim", type=int)
    g.add_argument("--tf-heads", type=int); g.add_argument("--tf-ff", type=int)
    g.add_argument("--dropout", type=float)
    g.add_argument("--time-mask", help="N,W: N spans of up to W samples zeroed per utterance")
    g.add_argument("--channel-mask", type=int, help="up to this many EMG channels zeroed per utterance")
    g.add_argument("--time-shift", type=int, help="random left shift of 0..N-1 samples (Gaddy: 11)")
    g = ap.add_argument_group("losses")
    g.add_argument("--text-weight", type=float, help="Granite's next-token loss (0 = align stage)")
    g.add_argument("--mel-weight", type=float, help="spectrogram loss against the spoken twin (0 = off)")
    g.add_argument("--mel-whiten", type=int, help="whiten the spectrogram loss over the top-k PCs (0 = MSE)")
    g.add_argument("--mel-silent-targets", type=int, choices=[0, 1],
                   help="1 = silent utterances get their vocal twin's spectrogram (DTW-paired) as target")
    g = ap.add_argument_group("encoder")
    g.add_argument("--init-from", help="checkpoint whose adapter (and encoder weights) initialise this run")
    g.add_argument("--encoder", choices=["frozen", "lora"])
    g.add_argument("--encoder-lr", type=float)
    g.add_argument("--lora-r", type=int); g.add_argument("--lora-alpha", type=int); g.add_argument("--lora-targets")
    g.add_argument("--input-linear-fp32", type=int, choices=[0, 1],
                   help="1 = fp32 master weights for the encoder's input projection (see granite.py)")
    g.add_argument("--bn-momentum", type=float, help="running-statistics momentum of the encoder's BatchNorms")
    g = ap.add_argument_group("optimisation")
    g.add_argument("--adapter-lr", type=float); g.add_argument("--wd", type=float)
    g.add_argument("--warmup-steps", type=int); g.add_argument("--epochs", type=int)
    g.add_argument("--grad-clip", type=float); g.add_argument("--accum", type=int)
    g = ap.add_argument_group("evaluation and selection")
    g.add_argument("--eval-every", type=int)
    g.add_argument("--no-eval-init", action="store_true", default=None)
    g.add_argument("--early-stop", type=int, help="stop after this many epochs without improving --select (0 = off)")
    g.add_argument("--select", choices=["wer", "ce", "mel"],
                   help="dev-silent WER / dev-silent CE / dev-vocal spectrogram loss (align stage)")
    g.add_argument("--snapshot-epochs", help="comma-separated epochs whose checkpoint is kept as snapshot_epNNN.pt")
    g.add_argument("--log-every", type=int)
    g.add_argument("--runs-dir")
    pre, _ = ap.parse_known_args()
    defaults = dict(DEFAULTS)
    if pre.from_config:
        defaults.update(translate_config(json.load(open(pre.from_config))))
    ap.set_defaults(**defaults)
    args = ap.parse_args()
    if args.no_eval_init is None:
        args.no_eval_init = bool(defaults.get("no_eval_init", False))
    return args


def main():
    args = parse()
    out = f"{args.runs_dir}/{args.name}"
    os.makedirs(out, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    align_only = args.text_weight == 0
    assert not align_only or args.mel_weight > 0, "the align stage needs --mel-weight > 0"
    assert not (align_only and args.encoder != "frozen"), "the align stage trains the adapter alone"

    def log(*a):
        print(*a, flush=True)

    # ---- data
    df = load_manifest(args.manifest)
    tw = twin_index(df)
    tr = df[(df.split == "train") & df.dirtag.isin(args.train_dirtags.split(","))]
    if args.max_ms:
        tr = tr[tr.n_ms <= args.max_ms]
    if args.limit_train:
        tr = tr.sample(n=min(args.limit_train, len(tr)), random_state=args.seed)
    dev = {}
    conds = [(c, t) for c, t in CONDITIONS if ((df.split == "dev") & (df.dirtag == t)).any()]   # the corpus's conditions
    assert conds, "no dev utterances of any known condition in the manifest"
    for cond, tag in conds:
        d = df[(df.split == "dev") & (df.dirtag == tag)].sort_values(["session", "idx"])
        if args.dev_subset:
            d = d.sample(n=min(args.dev_subset, len(d)), random_state=0)
        # dev always carries the spectrogram targets, so the logged R^2 shows how far the predicted
        # spectrograms drift from audio-likeness when the refine stage has no spectrogram term
        dev[cond] = EMGSet(d, args.mel_cache, tw, bool(args.mel_silent_targets))
    train = EMGSet(tr, args.mel_cache if args.mel_weight > 0 else "", tw, bool(args.mel_silent_targets))
    paths = [train.target_path(r) for r in train.df.itertuples()]
    log(f"train {len(train)} utterances ({tr.n_ms.sum()/3.6e6:.2f} h, max {tr.n_ms.max()/1000:.1f} s) from "
        f"{args.train_dirtags}; spectrogram targets for {sum(1 for p, _ in paths if p)} "
        f"({sum(1 for p, d_ in paths if p and d_)} DTW-paired silent); dev "
        + " ".join(f"{c} {len(dev[c])}" for c, _ in conds))

    # ---- model
    t0 = time.time()
    model, processor = load_granite(device)
    tok = processor.tokenizer
    ints = lambda s: tuple(int(x) for x in s.split(","))  # noqa: E731
    adapter = EMGAdapter(ints(args.conv_dims), ints(args.conv_kernels), ints(args.conv_strides), args.tf_layers,
                         args.tf_dim, args.tf_heads, args.tf_ff, args.dropout, in_ch=args.in_ch,
                         time_shift=args.time_shift, time_mask=ints(args.time_mask),
                         channel_mask=args.channel_mask).to(device)
    inj = MelInjector(model, f"{args.mel_cache}/in_stats.npz")
    pca = load_whitening(f"{args.mel_cache}/in_pca.npz", args.mel_whiten, device) if args.mel_whiten else None
    front = GraniteFront(model, tok, device)
    groups = [dict(params=list(adapter.parameters()), lr=args.adapter_lr, name="adapter")]
    if args.encoder == "lora":
        enc_params = add_encoder_lora(model, args.lora_r, args.lora_alpha, args.lora_targets, 0.05)
        if args.input_linear_fp32:
            input_linear_fp32(model.encoder)
        enc_params += list(model.encoder.input_linear.parameters())
        for p in enc_params:
            p.requires_grad_(True)
        groups.append(dict(params=enc_params, lr=args.encoder_lr, name="encoder"))
        for m in bn_layers(model.encoder):
            m.momentum = args.bn_momentum
        log(f"encoder: LoRA r{args.lora_r} alpha {args.lora_alpha} on {args.lora_targets} + input_linear "
            f"({model.encoder.input_linear.weight.dtype}); {len(bn_layers(model.encoder))} BatchNorm layers, "
            f"momentum {args.bn_momentum}")
    trainable = [p for g in groups for p in g["params"]]
    if args.init_from:
        ck0 = torch.load(args.init_from, map_location="cpu", weights_only=False)
        adapter.load_state_dict(ck0["adapter"])
        if "encoder_trainable" in ck0 and args.encoder == "lora":
            res = model.encoder.load_state_dict(ck0["encoder_trainable"], strict=False)
            assert not res.unexpected_keys, res.unexpected_keys
        if "encoder_bn" in ck0:
            model.encoder.load_state_dict(ck0["encoder_bn"], strict=False)
        log(f"initialised from {args.init_from} (epoch {ck0['epoch']}, dev {ck0.get('dev')})")
    log(f"model loaded in {time.time()-t0:.0f} s on {torch.cuda.get_device_name(0)}; " + ", ".join(
        f"{g['name']} {sum(p.numel() for p in g['params'])/1e6:.1f}M params @ lr {g['lr']:.1e}" for g in groups))
    config = dict(vars(args), format=2, model=MODEL, n_train=len(train), gpu=torch.cuda.get_device_name(0),
                  trainable_params=sum(p.numel() for p in trainable))
    json.dump(config, open(f"{out}/config.json", "w"), indent=1)

    # ---- optimiser and schedule (warmup, then cosine to zero over the epoch cap)
    def epoch_batches(r):
        return make_batches(train.df.n_ms.values, args.batch_ms, args.max_bs, r)
    opt = torch.optim.AdamW(groups, lr=args.adapter_lr, betas=(0.9, 0.98), weight_decay=args.wd)
    steps_per_epoch = math.ceil(len(epoch_batches(np.random.default_rng(0))) / args.accum)
    total_steps = steps_per_epoch * args.epochs

    def lr_lambda(s):
        if s < args.warmup_steps:
            return (s + 1) / args.warmup_steps
        p = (s - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lr_lambda] * len(groups))
    log(f"steps/epoch ~{steps_per_epoch}, total {total_steps}, warmup {args.warmup_steps}")

    # ---- logs
    f_steps = open(f"{out}/steps.csv", "w", newline="")
    w_steps = csv.writer(f_steps)
    w_steps.writerow(["step", "epoch", "loss", "ce", "mel", "lr", "grad_norm", "answer_tokens", "batch_size",
                      "batch_max_ms", "sec"])
    f_ep = open(f"{out}/epochs.csv", "w", newline="")
    w_ep = csv.writer(f_ep)
    w_ep.writerow(["epoch", "step", "train_loss", "train_ce", "train_mel", "dev_ce_silent", "dev_wer_silent",
                   "dev_ce_vocal", "dev_wer_vocal", "dev_mel_r2_vocal", "dev_mel_r2_silent_dtw", "train_sec",
                   "eval_sec", "peak_gpu_gb"])
    f_sess = open(f"{out}/dev_sessions.csv", "w", newline="")
    w_sess = csv.writer(f_sess)
    w_sess.writerow(["epoch", "condition", "session", "n", "wer"])
    snapshots = {int(e) for e in args.snapshot_epochs.split(",") if e}
    best = [float("inf"), 0]        # best selection metric, epoch
    step = 0

    def run_eval(epoch, train_loss, train_sec, tr_ce=float("nan"), tr_mel=float("nan")):
        model.encoder.eval()
        t = time.time()
        res = {c: evaluate_dev(dev[c], adapter, front, inj, device, args.eval_bs, args.max_new_tokens, pca)
               for c, _ in conds}
        es = time.time() - t
        peak = torch.cuda.max_memory_allocated() / 2**30
        nan = float("nan")
        g = lambda c, k: res[c][k] if c in res else nan       # noqa: E731  (a corpus may lack a condition)
        w_ep.writerow([epoch, step, f"{train_loss:.4f}", f"{tr_ce:.4f}", f"{tr_mel:.4f}",
                       f"{g('silent', 'ce'):.4f}", f"{g('silent', 'wer'):.4f}", f"{g('vocal', 'ce'):.4f}",
                       f"{g('vocal', 'wer'):.4f}", f"{1 - g('vocal', 'mel'):.4f}",
                       f"{1 - g('silent', 'mel'):.4f}", f"{train_sec:.0f}", f"{es:.0f}", f"{peak:.2f}"])
        f_ep.flush()
        for c, _ in conds:
            for sess, (n, w) in res[c]["per_session"].items():
                w_sess.writerow([epoch, c, sess, n, f"{w:.4f}"])
        f_sess.flush()
        with open(f"{out}/samples_epoch{epoch:03d}.txt", "w") as f:
            for c, _ in conds:
                f.write(f"=== {c} dev (n={res[c]['n']}, WER {100*res[c]['wer']:.1f}%, CE {res[c]['ce']:.3f})\n")
                for (u, r, h) in res[c]["samples"][:25]:
                    f.write(f"{u}\n  REF {r}\n  HYP {h}\n")
        log(f"[epoch {epoch}] train loss {train_loss:.4f} | " + " | ".join(
            f"dev {c} CE {res[c]['ce']:.3f} WER {100*res[c]['wer']:.1f}% spectrogram R2 {1 - res[c]['mel']:+.3f}"
            for c, _ in conds) + f" | train {train_sec:.0f}s eval {es:.0f}s peak {peak:.1f} GB")
        state = dict(adapter=adapter.state_dict(), epoch=epoch, step=step, args=config,
                     dev={c: dict(ce=res[c]["ce"], wer=res[c]["wer"], mel=res[c]["mel"]) for c in res})
        if args.encoder == "lora":
            state["encoder_trainable"] = {n_: p.detach().cpu() for n_, p in model.encoder.named_parameters()
                                          if p.requires_grad}
            state["encoder_bn"] = bn_state(model.encoder)   # buffers: the running statistics the eval used
        torch.save(state, f"{out}/last.pt")
        if epoch in snapshots:
            torch.save(state, f"{out}/snapshot_ep{epoch:03d}.pt")
        first = conds[0][0]                                    # silent when the corpus has it, else vocal
        metric = {"wer": res[first]["wer"], "ce": res[first]["ce"], "mel": g("vocal", "mel")}[args.select]
        assert metric == metric, f"--select {args.select} is undefined for this corpus"
        if metric < best[0]:
            best[0], best[1] = metric, epoch
            torch.save(state, f"{out}/best.pt")
            log(f"   new best dev {args.select} {metric:.4f} -> best.pt")
        return epoch - best[1]

    if not args.no_eval_init:
        run_eval(0, float("nan"), 0)

    # ---- training
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        if args.encoder == "lora":
            model.encoder.train()          # BatchNorm uses batch statistics; LoRA dropout on
        batches = epoch_batches(rng)
        t_ep, losses, ces, mels = time.time(), [], [], []
        opt.zero_grad(set_to_none=True)
        for bi, b in enumerate(batches):
            t_b = time.time()
            c = collate([train[i] for i in b])
            emg, n_enc = c["emg"].to(device), c["n_enc"].to(device)
            if align_only:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = adapter(emg, n_enc).float()
                with torch.no_grad():           # frozen Granite only reports the text CE
                    y = model.encoder(inj.features(z.detach(), n_enc).to(torch.bfloat16))
                    ce, n_ans = front.loss([y[i, :int(n_enc[i])] for i in range(len(b))], c["texts"])
                loss = 0.0
            else:
                y, z = inj(adapter, emg, n_enc)
                ce, n_ans = front.loss([y[i, :int(n_enc[i])] for i in range(len(b))], c["texts"])
                loss = args.text_weight * ce
            melv = float("nan")
            if args.mel_weight > 0:
                mel, _, _ = mel_loss(z, c, inj, device, pca)
                loss = loss + args.mel_weight * mel
                melv = mel.detach().item()
                mels.append(melv)
            assert torch.is_tensor(loss) and loss.requires_grad, "nothing to train on this batch"
            (loss / args.accum).backward()
            lv, cev = loss.detach().item(), ce.detach().item()
            losses.append(lv)
            ces.append(cev)
            if (bi + 1) % args.accum == 0 or bi == len(batches) - 1:
                gn = float(torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip))
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                w_steps.writerow([step, epoch, f"{lv:.4f}", f"{cev:.4f}", f"{melv:.4f}",
                                  f"{sched.get_last_lr()[0]:.2e}", f"{gn:.3f}", n_ans, len(b), max(c["n_ms"]),
                                  f"{time.time()-t_b:.2f}"])
                if step % args.log_every == 0:
                    f_steps.flush()
                    log(f"  step {step} ep {epoch} [{bi+1}/{len(batches)}] loss {lv:.4f} ce {cev:.4f}"
                        + (f" mel {melv:.4f} (ep mean {np.mean(mels):.4f}, R2 {1-np.mean(mels):+.3f})" if mels else "")
                        + f" | ep mean loss {np.mean(losses):.4f} ce {np.mean(ces):.4f} | lr "
                        f"{sched.get_last_lr()[0]:.2e} gn {gn:.2f} bs {len(b)} maxms {max(c['n_ms'])} "
                        f"{time.time()-t_b:.2f}s")
        train_sec = time.time() - t_ep
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            stale = run_eval(epoch, float(np.mean(losses)), train_sec, float(np.mean(ces)),
                             float(np.mean(mels)) if mels else float("nan"))
            if args.early_stop and stale >= args.early_stop:
                log(f"early stop: {stale} epochs since the best dev {args.select} ({best[0]:.4f} at epoch {best[1]})")
                break
        else:
            log(f"[epoch {epoch}] train loss {np.mean(losses):.4f} ce {np.mean(ces):.4f} ({train_sec:.0f}s)")
    for f in (f_steps, f_ep, f_sess):
        f.close()
    log(f"done; best dev {args.select} {best[0]:.4f} at epoch {best[1]} -> best.pt; logs in {out}")


if __name__ == "__main__":
    main()
