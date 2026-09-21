"""The EMG adapter: 1000-Hz EMG (B, T, 8) -> predicted Granite input features (B, n_enc, 160) at 50 Hz."""
import math
import random

import numpy as np
import torch
import torch.nn as nn

from . import MEL_DIM


def sinusoid(T, d, device, dtype):
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(dtype)


class EMGAdapter(nn.Module):
    """Conv stack with total stride 20 (each conv: odd kernel, padding k//2, so the output length is
    ceil(T/stride) and the chain gives ceil(T/20) >= n_enc), then a pre-norm Transformer encoder with
    absolute sinusoidal positions, then LayerNorm + Linear to the 160 z-scored feature dimensions.
    Frames beyond each utterance's n_enc are zeroed.

    Training-time augmentation on the raw EMG (all off in eval mode):
      time_shift   Gaddy's augmentation (architecture.py Model.forward): a random left shift of
                   0..time_shift-1 samples.  He shifts 0-7 samples at 689 Hz = 0-10 ms; at 1000 Hz the
                   same range is time_shift=11.
      time_mask    (N, W): N random spans of up to W samples zeroed per utterance (SpecAugment-style;
                   ours, not Gaddy's -- overfitting was the binding constraint without it).
      channel_mask up to this many of the 8 channels zeroed per utterance (ours).
    """

    def __init__(self, conv_dims=(256, 512, 768), kernels=(9, 9, 11), strides=(2, 2, 5),
                 tf_layers=6, tf_dim=768, heads=8, ff=3072, dropout=0.2, in_ch=8,
                 time_shift=0, time_mask=(0, 0), channel_mask=0, out_dim=MEL_DIM):
        super().__init__()
        self.time_shift = time_shift
        self.time_mask_n, self.time_mask_w = time_mask
        self.channel_mask = channel_mask
        convs, c = [], in_ch
        for d, k, s in zip(conv_dims, kernels, strides):
            assert k % 2 == 1, "odd kernels keep the frame grid centred"
            convs += [nn.Conv1d(c, d, k, stride=s, padding=k // 2), nn.GELU()]
            c = d
        self.convs = nn.Sequential(*convs)
        self.stride = int(np.prod(strides))
        self.inp = nn.Linear(c, tf_dim)
        self.in_norm = nn.LayerNorm(tf_dim)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(tf_dim, heads, ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, tf_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(tf_dim)
        self.out = nn.Linear(tf_dim, out_dim)

    def forward(self, emg, n_enc):
        if self.training and self.time_shift:
            r = random.randrange(self.time_shift)
            if r > 0:
                emg = torch.cat([emg[:, r:, :], torch.zeros_like(emg[:, :r, :])], 1)
        if self.training and (self.channel_mask or self.time_mask_n):
            B, T, C = emg.shape
            if self.channel_mask:
                k = random.randint(0, self.channel_mask)
                if k:
                    keep = torch.ones(B, C, device=emg.device, dtype=emg.dtype)
                    drop = torch.rand(B, C, device=emg.device).argsort(1)[:, :k]
                    keep.scatter_(1, drop, 0.)
                    emg = emg * keep[:, None, :]
            if self.time_mask_n and self.time_mask_w:
                ar = torch.arange(T, device=emg.device)[None, :]
                keep = torch.ones(B, T, device=emg.device, dtype=emg.dtype)
                for _ in range(self.time_mask_n):
                    w = torch.randint(0, self.time_mask_w + 1, (B, 1), device=emg.device)
                    st = (torch.rand(B, 1, device=emg.device)
                          * (T - w).clamp(min=1).float()).long()
                    keep = keep * (~((ar >= st) & (ar < st + w))).to(emg.dtype)
                emg = emg * keep[:, :, None]
        h = self.convs(emg.transpose(1, 2)).transpose(1, 2)      # (B, ceil(T/20), C)
        Tm = int(n_enc.max())
        assert h.shape[1] >= Tm, (h.shape, Tm)
        h = h[:, :Tm]
        h = self.in_norm(self.inp(h))
        h = h + sinusoid(Tm, h.shape[-1], h.device, h.dtype)
        pad = torch.arange(Tm, device=h.device)[None, :] >= n_enc[:, None]
        h = self.tf(self.drop(h), src_key_padding_mask=pad)
        y = self.out(self.out_norm(h))
        return y.masked_fill(pad[..., None], 0.)
