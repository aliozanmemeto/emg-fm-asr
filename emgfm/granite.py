"""Everything around IBM Granite Speech 4.0-1B: loading, the prompt/projector/LLM front-end, feeding
predicted features through the conformer encoder, encoder LoRA, and the encoder's BatchNorm layers.

Mechanics (verified against transformers 4.57.6 modeling_granite_speech.py):
  encoder(input_features (B, T, 160)) -> (B, T, 1024); projector -> (B, ceil(T/15)*3, 2048);
  get_merged_audio_embeddings scatters the projected tokens into the <|audio|> slots of the prompt;
  the LLM computes the attention-mask-shifted CE (ignore_index -100).
Tokenisation: prompt ids + answer ids tokenised SEPARATELY + <|end_of_text|> (the chat template
appends no end-of-turn token, and joint tokenisation can merge the prompt's final ':' with the answer).
The tokenizer adds no BOS/EOS by default; padding side is left.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from . import BN_BUFFERS, ENC_DIM, MODEL, PROMPT_TEXT, QUERIES, WINDOW


def load_granite(device="cuda"):
    """The pretrained model in bf16, frozen, in eval mode, plus its processor."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL, device_map=device, dtype=torch.bfloat16)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor


# ------------------------------------------------------------------ prompt -> projector -> LLM
def n_tokens(n_enc):
    return math.ceil(n_enc / WINDOW) * QUERIES


def build_ids(pre, post, audio_id, ntoks, answers, pad_id, eos_id):
    """Left-padded input_ids / attention_mask / labels for a batch.
    answers: list of token-id lists, or None for generation.  Pure function."""
    rows, labs = [], []
    for i, nt in enumerate(ntoks):
        p = pre + [audio_id] * nt + post
        if answers is None:
            rows.append(p)
            labs.append(None)
        else:
            a = list(answers[i]) + [eos_id]
            rows.append(p + a)
            labs.append([-100] * len(p) + a)
    L = max(len(r) for r in rows)
    ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    am = torch.zeros(len(rows), L, dtype=torch.long)
    lab = torch.full((len(rows), L), -100, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, L - len(r):] = torch.tensor(r)
        am[i, L - len(r):] = 1
        if labs[i] is not None:
            lab[i, L - len(r):] = torch.tensor(labs[i])
    return ids, am, (lab if answers is not None else None)


class GraniteFront:
    """Projector -> merge into the prompt -> LLM.  Takes encoder-OUTPUT frames, (T_i, 1024) per utterance."""

    def __init__(self, model, tok, device):
        self.model, self.tok, self.device = model, tok, device
        self.prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT_TEXT}],
                                              tokenize=False, add_generation_prompt=True)
        ids = tok(self.prompt, add_special_tokens=False)["input_ids"]
        self.audio_id = model.config.audio_token_id
        assert ids.count(self.audio_id) == 1, ids
        k = ids.index(self.audio_id)
        self.pre, self.post = ids[:k], ids[k + 1:]
        self.eos, self.pad = tok.eos_token_id, tok.pad_token_id

    def build(self, feats, texts=None):
        B, Tm = len(feats), max(f.shape[0] for f in feats)
        Fp = torch.zeros(B, Tm, ENC_DIM, device=self.device, dtype=self.model.dtype)
        for i, f in enumerate(feats):
            Fp[i, :f.shape[0]] = f.to(self.model.dtype)
        proj = self.model.projector(Fp)                            # (B, ceil(Tm/15)*3, 2048)
        ntoks = [n_tokens(f.shape[0]) for f in feats]
        mask = torch.arange(proj.shape[1], device=self.device)[None, :] \
            < torch.tensor(ntoks, device=self.device)[:, None]
        answers = None if texts is None else \
            [self.tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        ids, am, lab = build_ids(self.pre, self.post, self.audio_id, ntoks, answers, self.pad, self.eos)
        ids, am = ids.to(self.device), am.to(self.device)
        emb = self.model.get_merged_audio_embeddings(ids, proj, mask)
        return dict(inputs_embeds=emb, attention_mask=am,
                    labels=None if lab is None else lab.to(self.device),
                    n_answer=0 if lab is None else int((lab != -100).sum()))

    def loss(self, feats, texts):
        """(mean next-token CE over the answer tokens, number of answer tokens)."""
        b = self.build(feats, texts)
        out = self.model(inputs_embeds=b["inputs_embeds"], attention_mask=b["attention_mask"],
                         labels=b["labels"])
        return out.loss, b["n_answer"]

    @torch.no_grad()
    def generate(self, feats, max_new_tokens=200, num_beams=1, num_return_sequences=1, scores=False):
        """Decoded strings (raw, before Gaddy's normaliser).  With scores=True also the sequence scores."""
        b = self.build(feats, None)
        g = self.model.generate(inputs_embeds=b["inputs_embeds"], attention_mask=b["attention_mask"],
                                max_new_tokens=max_new_tokens, do_sample=False, num_beams=num_beams,
                                num_return_sequences=num_return_sequences, pad_token_id=self.pad,
                                eos_token_id=self.eos, output_scores=scores, return_dict_in_generate=True)
        txt = self.tok.batch_decode(g.sequences, skip_special_tokens=True)
        if not scores:
            return txt
        sc = g.sequences_scores.float().cpu().tolist() if getattr(g, "sequences_scores", None) is not None \
            else [float("nan")] * len(txt)
        return txt, sc

    @torch.no_grad()
    def log_prob(self, feat, texts, bs=1):
        """Teacher-forced log p(text | one utterance's frames): sum of answer-token log-probs incl. EOS,
        built exactly as the training loss is.  Returns (list of log-probs, list of token counts).
        NB: bf16 + left padding make the value depend slightly on the batch (up to 0.36 nats measured
        between batch 16 and 1); bs=1 is exactly reproducible."""
        lps, nts = [], []
        for s in range(0, len(texts), bs):
            cs = texts[s:s + bs]
            b = self.build([feat] * len(cs), cs)
            logits = self.model(inputs_embeds=b["inputs_embeds"], attention_mask=b["attention_mask"]).logits
            lab = b["labels"]
            for i in range(len(cs)):
                pos = (lab[i] != -100).nonzero().squeeze(1)
                lp = torch.log_softmax(logits[i, pos - 1].float(), -1).gather(1, lab[i, pos][:, None]).squeeze(1)
                lps.append(float(lp.sum()))
                nts.append(int(len(pos)))
        return lps, nts


# ------------------------------------------------------------------ adapter output -> encoder
class MelInjector:
    """De-normalises the adapter's z-scored prediction with the training feature statistics and runs it
    through the conformer encoder.  Frames past each utterance's n_enc are filled with the utterance's
    own feature floor (the extractor's silence level) before the encoder, because its block attention
    (context 200) lets padding reach the last block of valid frames; 'floor' was the fill that changed
    the encoder output least in the padding check (prep_mel_targets.py of the original lab code).
    Returns (encoder-output frames (B, T, 1024) in bf16, the z-scored prediction in fp32)."""

    def __init__(self, model, stats_path):
        st = np.load(stats_path)
        dev = next(model.parameters()).device
        self.model = model
        self.mu = torch.from_numpy(st["mean"]).to(dev)
        self.sd = torch.from_numpy(st["std"]).to(dev)

    def features(self, z, n_enc):
        feat = self.mu + self.sd * z.float()
        pad = torch.arange(feat.shape[1], device=feat.device)[None, :] >= n_enc[:, None]
        if pad.any():
            fl = feat.masked_fill(pad[..., None], float("inf")).amin(dim=(1, 2), keepdim=True)
            feat = torch.where(pad[..., None], fl.expand_as(feat), feat)
        return feat

    def __call__(self, adapter, emg, n_enc):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = adapter(emg, n_enc)
        return self.model.encoder(self.features(z, n_enc).to(torch.bfloat16)), z.float()


def load_whitening(pca_path, k, device):
    """Top-k principal directions (V: 160 x k) and variances of the z-scored TRAINING features, for the
    whitened align loss.  Plain MSE weights direction j by its variance and so weights PC1 ~300x more
    than PC16, i.e. it optimises loudness and spectral tilt and leaves the directions the encoder needs
    (PCs up to ~8) nearly free; whitening makes each of the top k count equally."""
    zp = np.load(pca_path)
    V = torch.from_numpy(zp["V"][:, :k].copy()).to(device)
    ev = torch.from_numpy(zp["ev"][:k].copy()).to(device).clamp(min=1e-8)
    return V, ev


# ------------------------------------------------------------------ encoder adaptation
def add_encoder_lora(model, r, alpha, targets, dropout):
    """LoRA on the conformer encoder's attention and feed-forward linears.  LoRA weights are kept in
    fp32.  Returns the list of LoRA parameters (requires_grad set by the caller)."""
    from peft import LoraConfig, inject_adapter_in_model
    model.encoder = inject_adapter_in_model(
        LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets.split(",")),
        model.encoder)
    params = []
    for n_, p in model.encoder.named_parameters():
        if "lora_" in n_:
            p.data = p.data.float()
            params.append(p)
    assert params, f"no encoder module matched {targets}"
    return params


def input_linear_fp32(encoder):
    """Keep the conformer's input projection (160 -> 1024) in fp32 and compute it in fp32.
    Trained in place in bf16, AdamW's per-step change (~lr = 5e-5) is below half a bf16 ULP for most
    of its weights (|w| median 0.038 -> ULP 2.4e-4), so the update rounds away: after 26-38 epochs only
    22 % of its elements had changed at all.  Input cast up, output cast back to bf16; idempotent."""
    lin = encoder.input_linear
    if getattr(lin, "_fp32_wrapped", False):
        return lin
    lin.float()
    lin.register_forward_pre_hook(lambda m, args: (args[0].float(),) + tuple(args[1:]))
    lin.register_forward_hook(lambda m, args, out: out.to(torch.bfloat16))
    lin._fp32_wrapped = True
    return lin


# ------------------------------------------------------------------ BatchNorm
def bn_layers(encoder):
    """The conformer has a BatchNorm1d in each of its 16 conv modules.  In train mode they normalise
    with batch statistics and their running statistics drift from the pretrained audio values to
    EMG-derived ones -- those running statistics are BUFFERS and must travel with the checkpoint."""
    return [m for m in encoder.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def bn_state(encoder):
    return {n_: b.detach().cpu() for n_, b in encoder.named_buffers() if n_.rsplit(".", 1)[-1] in BN_BUFFERS}


@torch.no_grad()
def recalibrate_bn(model, adapter, inj, ds, pool, batch_ms, max_bs, seed=0):
    """Re-estimate the encoder's BatchNorm running statistics as the exact average over training-style
    batches of `ds` rows `pool` (momentum=None), everything else in eval mode.  The statistics saved
    during training are an EMA over the last ~10 batches (momentum 0.1) and cost up to 1.4 points of
    WER against this; the standard post-hoc recalibration.  Returns the number of batches."""
    from .data import collate, make_batches
    bns = bn_layers(model.encoder)
    model.encoder.eval()
    adapter.eval()
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    rng = np.random.default_rng(seed)
    batches = [[int(pool[i]) for i in b] for b in make_batches(ds.df.n_ms.values[pool], batch_ms, max_bs, rng)]
    dev = next(model.parameters()).device
    for b in batches:
        c = collate([ds[i] for i in b])
        inj(adapter, c["emg"].to(dev), c["n_enc"].to(dev))
    for m in bns:
        m.eval()
    return len(batches)
