"""Ensembling several checkpoints without any tuned parameter.

  --mode gen      one checkpoint: its encoder-output frames for every utterance of the split (batch 1,
                  BatchNorm as evaluate.py) and its n-best lists for each beam size in --beams
                  (num_return_sequences = beam; 1 = greedy).  Saved to --out (.pt).
  --mode score    several gen files: per utterance the POOL = union of every member's n-best lists,
                  deduplicated by exact string, scored under every member by teacher forcing
                  (log p_m(candidate | member m's frames); the projector and LLM are frozen and shared,
                  so only the cached frames differ).  Saved to --out (.json), then the report.
  --mode report   own top-1 of each member per beam size | oracle (best candidate in the pool) |
                  PoE = argmax_c sum_m log p_m(c) | MBR = argmin_c sum_c' P(c') word_edits(c, c')
                  with P the softmax over the pool of the mean log-probability.
  --mode subsets  PoE and MBR for every subset of the members, by subset size.

Generation and scoring run at batch 1 by default: bf16 + left padding change teacher-forced scores by
up to 0.36 nats and flip n-best lists on 8-21 utterances between batch sizes; batch 1 is reproducible.
"""
import argparse
import json
import os
import sys
from itertools import combinations

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from emgfm import CONDITIONS  # noqa: E402
from emgfm.checkpoint import arg, condition_set, encode, load_checkpoint, setup_bn  # noqa: E402
from emgfm.data import load_manifest  # noqa: E402
from emgfm.evaluation import normalise, wer, word_errors  # noqa: E402
from emgfm.granite import GraniteFront, load_granite  # noqa: E402


def gen(args):
    device = "cuda"
    R = load_checkpoint(args.ckpt, device, args.mel_cache or None)
    bn = setup_bn(R, args.bn, args.manifest or None)
    df = load_manifest(args.manifest or arg(R["args"], "manifest"))
    beams = [int(b) for b in args.beams.split(",")]
    res = dict(ckpt=args.ckpt, epoch=R["ck"]["epoch"], args=R["args"], bn=bn, split=args.split, beams=beams,
               gpu=torch.cuda.get_device_name(0), conds={})
    for cond, tag in CONDITIONS:
        if cond not in args.conds.split(","):
            continue
        ds, d = condition_set(df, args.split, tag, args.n)
        if len(d) == 0:                                   # a corpus without this condition
            continue
        feats, refs, uids = encode(R, ds, 1, device)
        nbest = {}
        for k in beams:
            lists = []
            for s in range(0, len(feats), args.gen_bs):
                txt, sc = R["front"].generate(feats[s:s + args.gen_bs], args.max_new_tokens, k, k, scores=True)
                assert len(txt) % k == 0
                for i in range(len(txt) // k):
                    lists.append([(txt[i * k + j], sc[i * k + j]) for j in range(k)])
            assert len(lists) == len(feats)
            nbest[k] = lists
            print(f"{cond:6s} beam {k}: own top-1 WER {100*wer(refs, [normalise(l[0][0]) for l in lists]):5.2f}%",
                  flush=True)
        res["conds"][cond] = dict(uids=uids, refs=refs, sentence_index=[int(x) for x in d.sentence_index],
                                  feats=[f.detach().to("cpu", torch.bfloat16) for f in feats], nbest=nbest)
    torch.save(res, args.out)
    print("saved", args.out)


def score(args):
    device = "cuda"
    files = args.gen_files.split(",")
    G = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    names = args.names.split(",") if args.names else [os.path.basename(f).rsplit(".", 1)[0] for f in files]
    assert len(names) == len(G) and len({g["split"] for g in G}) == 1
    model, processor = load_granite(device)
    front = GraniteFront(model, processor.tokenizer, device)
    out = dict(models=names, files=files, split=G[0]["split"], epochs=[g["epoch"] for g in G],
               gpu=torch.cuda.get_device_name(0), conds={})
    for cond, _ in CONDITIONS:
        if not all(cond in g["conds"] for g in G):
            continue
        uids = G[0]["conds"][cond]["uids"]
        assert all(g["conds"][cond]["uids"] == uids for g in G), "gen files cover different utterances"
        refs = G[0]["conds"][cond]["refs"]
        pools = []
        for u in range(len(uids)):
            cands, index, src = [], {}, []
            for mi, g in enumerate(G):
                for k, lists in g["conds"][cond]["nbest"].items():
                    for rank, (txt, _) in enumerate(lists[u]):
                        if txt not in index:
                            index[txt] = len(cands)
                            cands.append(txt)
                            src.append([])
                        src[index[txt]].append([mi, int(k), rank])
            pools.append((cands, src))
        S = [[None] * len(G) for _ in uids]
        ntok = [None] * len(uids)
        for mi, g in enumerate(G):
            for u in range(len(uids)):
                S[u][mi], ntok[u] = front.log_prob(g["conds"][cond]["feats"][u].to(device), pools[u][0], args.score_bs)
            print(f"   {cond}: scored {len(uids)} pools (mean {np.mean([len(p[0]) for p in pools]):.1f} candidates) "
                  f"under {names[mi]}", flush=True)
        out["conds"][cond] = [dict(uid=uids[u], sentence_index=G[0]["conds"][cond]["sentence_index"][u], ref=refs[u],
                                   cands=pools[u][0], clean=[normalise(c) for c in pools[u][0]], src=pools[u][1],
                                   scores=S[u], ntok=ntok[u]) for u in range(len(uids))]
    json.dump(out, open(args.out, "w"))
    print("saved", args.out, flush=True)
    report(out)


def combine(u, members, rule):
    """Index into u["clean"] chosen by `rule` using only `members`: the pool is restricted to candidates one
    of them produced, and only their scores are used."""
    idx = [ci for ci, s in enumerate(u["src"]) if any(x[0] in members for x in s)]
    sc = np.asarray(u["scores"])[members][:, idx]
    if rule == "poe":
        return idx[int(np.argmax(sc.sum(0)))]
    m = sc.mean(0)
    p = np.exp(m - m.max()); p /= p.sum()
    words = [u["clean"][i].split() for i in idx]
    risk = [sum(p[j] * word_errors(" ".join(words[i]), " ".join(words[j])) for j in range(len(words)))
            for i in range(len(words))]
    return idx[int(np.argmin(risk))]


def own_top1(u, mi, k):
    top = [ci for ci, s in enumerate(u["src"]) if [mi, k, 0] in s]
    assert len(top) == 1
    return u["clean"][top[0]]


def report(out):
    names = out["models"]
    M = list(range(len(names)))
    for cond, utts in out["conds"].items():
        refs = [u["ref"] for u in utts]
        n_words = sum(len(r.split()) for r in refs)
        beams = sorted({s[1] for u in utts for c in u["src"] for s in c})
        print(f"\n=== {cond}: {len(utts)} utterances, {n_words} words, mean pool "
              f"{np.mean([len(u['cands']) for u in utts]):.1f} candidates, {len(names)} members ===")
        for mi, nm in enumerate(names):
            print(f"   own top-1  {nm:14s} " + "  ".join(
                f"beam {k} {100*wer(refs, [own_top1(u, mi, k) for u in utts]):6.2f}%" for k in beams))
        oracle = sum(min(word_errors(u["ref"], c) for c in u["clean"]) for u in utts)
        print(f"   {'oracle (best candidate in the pool)':45s} {100*oracle/n_words:6.2f}%")
        for rule, lab in [("poe", "PoE  argmax sum_m log p_m(c)"), ("mbr", "MBR  mean-log-p posterior")]:
            print(f"   {lab:45s} {100*wer(refs, [u['clean'][combine(u, M, rule)] for u in utts]):6.2f}%")


def subsets(out, own_beam=4):
    names = out["models"]
    M = len(names)
    for cond, utts in out["conds"].items():
        refs = [u["ref"] for u in utts]
        own = [100 * wer(refs, [own_top1(u, mi, own_beam) for u in utts]) for mi in range(M)]
        print(f"\n=== {cond}: single models (own top-1, beam {own_beam}): "
              + ", ".join(f"{n} {w:.2f}" for n, w in zip(names, own)))
        for size in range(2, M + 1):
            for rule in ("poe", "mbr"):
                res = []
                for sub in combinations(range(M), size):
                    w = 100 * wer(refs, [u["clean"][combine(u, list(sub), rule)] for u in utts])
                    res.append((w, sub, min(own[i] for i in sub)))
                ws = np.array([r[0] for r in res])
                gain = np.array([r[0] - r[2] for r in res])
                best = min(res)
                print(f"   size {size} {rule.upper()}: mean {ws.mean():6.2f}%  range [{ws.min():.2f}, {ws.max():.2f}]"
                      f"  | vs best member: mean {gain.mean():+5.2f}, worst {gain.max():+5.2f}"
                      f"  | best subset {'+'.join(names[i] for i in best[1])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["gen", "score", "report", "subsets"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--gen-files", default="", help="comma-separated gen outputs (score)")
    ap.add_argument("--names", default="", help="comma-separated member names (score)")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--conds", default="silent,vocal")
    ap.add_argument("--beams", default="1,4,8")
    ap.add_argument("--gen-bs", type=int, default=1)
    ap.add_argument("--score-bs", type=int, default=1)
    ap.add_argument("--bn", default="auto", choices=["auto", "recal", "saved", "pretrained"])
    ap.add_argument("--manifest", default="")
    ap.add_argument("--mel-cache", default="")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", required=True, help="gen: .pt to write; score: .json to write; report/subsets: .json to read")
    args = ap.parse_args()
    if args.split == "test" and not args.final_test:
        sys.exit("refusing the TEST split without --final-test (select on dev; evaluate test once)")
    if args.mode == "gen":
        gen(args)
    elif args.mode == "score":
        score(args)
    elif args.mode == "report":
        report(json.load(open(args.out)))
    else:
        subsets(json.load(open(args.out)))


if __name__ == "__main__":
    main()
