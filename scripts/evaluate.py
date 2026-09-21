"""Exact evaluation of one checkpoint: WER on the dev (or, once, the test) split of both conditions.

Protocol behind every reported number:
  * batch size 1 for the encoder AND the decoder (bf16 + padding make batched outputs differ slightly);
  * BatchNorm statistics re-estimated on the training set (--bn recal; the default for a trained encoder);
  * greedy and beam search (beam 4 and 8; beam 16 was worse on dev);
  * Gaddy's normaliser on hypotheses and references, corpus WER, 95 % bootstrap CI over utterances;
  * compare models evaluated on the SAME GPU type (A40 vs Blackwell differ by 0.1-0.35 points);
  * the test split is refused without --final-test: select everything on dev, evaluate test once.
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from emgfm import CONDITIONS  # noqa: E402
from emgfm.checkpoint import arg, condition_set, encode, load_checkpoint, setup_bn  # noqa: E402
from emgfm.data import load_manifest  # noqa: E402
from emgfm.evaluation import bootstrap_wer, normalise, wer, word_errors  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final-test", action="store_true", help="required for --split test (evaluate test ONCE)")
    ap.add_argument("--beams", default="1,4,8")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--bn", default="auto", choices=["auto", "recal", "saved", "pretrained"])
    ap.add_argument("--n", type=int, default=0, help="first N (shortest) utterances per condition (smoke tests)")
    ap.add_argument("--manifest", default="", help="override the manifest recorded in the checkpoint")
    ap.add_argument("--mel-cache", default="", help="override the feature statistics dir recorded in the checkpoint")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--out", default="", help="JSON output (default <run dir>/eval_<split>.json)")
    args = ap.parse_args()
    if args.split == "test" and not args.final_test:
        sys.exit("refusing the TEST split without --final-test (select on dev; evaluate test once)")
    device = "cuda"
    R = load_checkpoint(args.ckpt, device, args.mel_cache or None)
    bn = setup_bn(R, args.bn, args.manifest or None)
    df = load_manifest(args.manifest or arg(R["args"], "manifest"))
    beams = [int(b) for b in args.beams.split(",")]
    out = dict(ckpt=args.ckpt, split=args.split, epoch=R["ck"]["epoch"], bn=bn, bs=args.bs,
               gpu=torch.cuda.get_device_name(0), results={})
    for cond, tag in CONDITIONS:
        ds, d = condition_set(df, args.split, tag, args.n)
        if len(d) == 0:                                   # a corpus without this condition
            continue
        feats, refs, uids = encode(R, ds, args.bs, device)
        sidx = [int(x) for x in d.sentence_index]
        for k in beams:
            hyps = []
            for s in range(0, len(feats), args.bs):
                hyps += [normalise(h) for h in R["front"].generate(feats[s:s + args.bs], args.max_new_tokens, k)]
            errs = [word_errors(r, h) for r, h in zip(refs, hyps)]
            lens = [len(r.split()) for r in refs]
            w, lo, hi = bootstrap_wer(errs, lens)
            assert abs(w - wer(refs, hyps)) < 1e-12
            out["results"][f"{cond}/beam{k}"] = dict(
                wer=w, ci95=[lo, hi], n=len(refs), n_words=int(sum(lens)),
                per_utt=[dict(uid=u, sentence_index=si, ref=r, hyp=h, errors=int(e))
                         for u, si, r, h, e in zip(uids, sidx, refs, hyps, errs)])
            print(f"{cond:6s} beam {k}: WER {100*w:5.2f}%  [{100*lo:.1f}, {100*hi:.1f}]  ({len(refs)} utterances)",
                  flush=True)
    path = args.out or os.path.join(os.path.dirname(args.ckpt), f"eval_{args.split}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
