"""Paired comparison of evaluation outputs (scripts/evaluate.py JSONs) on the same utterances.

    compare.py A.json[,A2.json,...] B.json[,B2.json,...] [--beam 4] [--conds silent,vocal]

Each side may be several files (e.g. seeds); their per-utterance word errors are averaged before the
comparison, so the test is "recipe A vs recipe B" with seed noise averaged out.  Reports each side's
corpus WER with a 95 % bootstrap CI and the paired difference A - B with its CI (resampling the same
utterances on both sides), plus how many utterances got better / worse.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from emgfm.evaluation import bootstrap_wer, paired_bootstrap  # noqa: E402


def errors(files, key):
    """Per-utterance word errors averaged over `files`, in uid order, plus reference lengths."""
    per_file = []
    lens = None
    for f in files:
        d = json.load(open(f))["results"][key]["per_utt"]
        d = sorted(d, key=lambda u: u["uid"])
        e = np.array([u["errors"] if "errors" in u else _errors(u) for u in d], float)
        L = np.array([len(u["ref"].split()) for u in d])
        assert lens is None or np.array_equal(L, lens), f"{f}: different utterances"
        lens = L
        per_file.append(e)
    return np.mean(per_file, 0), lens


def _errors(u):
    from emgfm.evaluation import word_errors
    return word_errors(u["ref"], u["hyp"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="comma-separated evaluation JSONs of side A")
    ap.add_argument("b", help="comma-separated evaluation JSONs of side B")
    ap.add_argument("--beam", default="4")
    ap.add_argument("--conds", default="silent,vocal")
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()
    A, B = args.a.split(","), args.b.split(",")
    for cond in args.conds.split(","):
        for k in args.beam.split(","):
            key = f"{cond}/beam{k}"
            ea, lens = errors(A, key)
            eb, _ = errors(B, key)
            wa, la, ha = bootstrap_wer(ea, lens, args.n_boot)
            wb, lb, hb = bootstrap_wer(eb, lens, args.n_boot)
            d, lo, hi = paired_bootstrap(ea, eb, lens, args.n_boot)
            print(f"{key:13s} A {100*wa:5.2f} [{100*la:.1f}, {100*ha:.1f}]  B {100*wb:5.2f} [{100*lb:.1f}, {100*hb:.1f}]"
                  f"  A-B {100*d:+5.2f} [{100*lo:+.2f}, {100*hi:+.2f}]  utterances better {(ea < eb).sum()} "
                  f"worse {(ea > eb).sum()} (n={len(lens)}, {len(A)} vs {len(B)} files)")


if __name__ == "__main__":
    main()
