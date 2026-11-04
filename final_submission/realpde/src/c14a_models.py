"""C14-A -- "SimVP design scaling": bounded, one-factor-at-a-time screen around
`c11a_models.N1SimVP` (the C11-A backbone-family winner: strided conv encoder
-> 8 gated spatio-temporal attention (gSTA) blocks @16x32 hidden256 ->
transposed-conv decoder + skip). SAME contract, SAME ctx-norm wrapper, SAME
Welsch recipe as C11-A -- only the inner network varies.

Imports `c11a_models` (read-only, unedited) for the shape contract
(`IN_CH`/`OUT_CH`/`HH`/`WW`/`T_STEPS`/`C_IN`/`C_OUT`/`EXTRA_CHANNELS`),
`fold_window`/`unfold_output`, `param_count`/`fp16_mb`, and the streaming
wrapper (`GenericCtxNormCore`/`GenericCtxNormWrapper`) unchanged -- every
variant here only replaces the `core(z): (B,64,32,64) -> (B,60,32,64)`
network, exactly the same contract C11-A's five families all shared.

New file, per the family's file-scope rule (`src/c14a_*.py`). Imports
`c11a_models` and stdlib/torch only.

Variants (family brief, `experiments/c14a_simvp_scaling/notes.md`):
    V1 depth:        n_blocks 12, 16 (hidden 256, kernel unchanged)
    V2 hidden width:  384 (8 blocks)
    V3 kernel:        effective ~31 (7x7 + dilated 9x9 d=3) and ~11 (3x3 + dilated 5x5 d=2)
    V4 latent res:    no downsampling (32x64, hidden128, 8 blocks) and
                      stronger downsampling (8x16, hidden384, single skip)
    V5 encoder/dec:   two strided stages WITH two skips (U-shaped SimVP), hidden256
    V6 temporal:      shared per-frame 2D stem/encoder + decoder, gSTA translator
                      mixes the (T x C') channel axis (SimVP-original style)
    V7 dilation cyc:  gSTA large-kernel branch dilation cycles (1, 2, 3) across blocks
    V8 combo:         (added once V1-V7 results are in) the >=1 winning factor(s)
                      combined, + a second seed of the combination
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

import c11a_models as C11A  # noqa: E402

IN_CH, OUT_CH = C11A.IN_CH, C11A.OUT_CH
HH, WW, T_STEPS = C11A.HH, C11A.WW, C11A.T_STEPS
C_IN, C_OUT, EXTRA_CHANNELS = C11A.C_IN, C11A.C_OUT, C11A.EXTRA_CHANNELS

# re-exported unedited (same contract every C11-A model used)
fold_window = C11A.fold_window
unfold_output = C11A.unfold_output
param_count = C11A.param_count
fp16_mb = C11A.fp16_mb
GenericCtxNormCore = C11A.GenericCtxNormCore
GenericCtxNormWrapper = C11A.GenericCtxNormWrapper


# --------------------------------------------------------------------------- #
# generalised gSTA block: parametrised kernel sizes / dilation (n1_simvp's
# GSTABlock is the k1=5, k2=7, d2=3 special case; same architecture otherwise,
# byte-for-byte -- proj_in -> dw(k1) -> dw(k2, dilation d2) -> sigmoid-gated
# multiply -> proj_out residual, then an inverted-bottleneck MLP residual).
# --------------------------------------------------------------------------- #
class GSTABlockCfg(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4, k1: int = 5, k2: int = 7, d2: int = 3):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, dim)
        self.proj_in = nn.Conv2d(dim, dim, 1)
        self.dw1 = nn.Conv2d(dim, dim, k1, padding=k1 // 2, groups=dim)
        pad2 = d2 * (k2 - 1) // 2
        self.dw2 = nn.Conv2d(dim, dim, k2, padding=pad2, dilation=d2, groups=dim)
        self.gate = nn.Conv2d(dim, dim, 1)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()

        self.norm2 = nn.GroupNorm(8, dim)
        self.mlp1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.mlp2 = nn.Conv2d(mlp_ratio * dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.act(self.proj_in(self.norm1(x)))
        attn = self.dw2(self.dw1(u))
        g = torch.sigmoid(self.gate(u))
        x = x + self.proj_out(attn * g)
        h = self.act(self.mlp1(self.norm2(x)))
        x = x + self.mlp2(h)
        return x


def eff_rf(k1: int, k2: int, d2: int) -> int:
    """Sequential-composition effective receptive field of dw1(k1) -> dw2(k2,
    dilation d2): RF = k1 + d2*(k2-1). n1_simvp's own k1=5,k2=7,d2=3 -> 23
    (docstring there rounds to "~21"; this screen uses the exact formula
    consistently across variants so the deltas are apples-to-apples)."""
    return k1 + d2 * (k2 - 1)


# --------------------------------------------------------------------------- #
# V1/V2/V3/V7: one strided down/up stage, exactly n1_simvp's topology, only
# depth / hidden width / kernel(s) / dilation-cycling parametrised.
# --------------------------------------------------------------------------- #
class SimVP1Stage(nn.Module):
    __name__ = "SimVP1Stage"

    def __init__(self, in_ch: int = IN_CH, out_ch: int = OUT_CH, enc_dim: int = 128,
                 hidden: int = 256, n_blocks: int = 8, k1: int = 5, k2: int = 7, d2: int = 3,
                 dilations: Optional[Sequence[int]] = None):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, enc_dim, 3, padding=1), nn.GroupNorm(8, enc_dim), nn.GELU())
        self.down = nn.Sequential(
            nn.Conv2d(enc_dim, hidden, 4, stride=2, padding=1), nn.GroupNorm(8, hidden), nn.GELU())
        if dilations:
            self.blocks = nn.ModuleList(
                [GSTABlockCfg(hidden, k1=k1, k2=k2, d2=dilations[i % len(dilations)])
                 for i in range(n_blocks)])
        else:
            self.blocks = nn.ModuleList(
                [GSTABlockCfg(hidden, k1=k1, k2=k2, d2=d2) for _ in range(n_blocks)])
        self.up = nn.ConvTranspose2d(hidden, enc_dim, 4, stride=2, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * enc_dim, enc_dim, 1), nn.GroupNorm(8, enc_dim), nn.GELU())
        self.head = nn.Conv2d(enc_dim, out_ch, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        skip = self.stem(z)
        h = self.down(skip)
        for b in self.blocks:
            h = b(h)
        h = self.up(h)
        h = self.fuse(torch.cat([h, skip], dim=1))
        return self.head(h)


# --------------------------------------------------------------------------- #
# V4a: no spatial downsampling at all -- gSTA translator runs at full 32x64.
# --------------------------------------------------------------------------- #
class SimVPNoDown(nn.Module):
    __name__ = "SimVPNoDown"

    def __init__(self, in_ch: int = IN_CH, out_ch: int = OUT_CH, enc_dim: int = 128,
                 hidden: int = 128, n_blocks: int = 8, k1: int = 5, k2: int = 7, d2: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, enc_dim, 3, padding=1), nn.GroupNorm(8, enc_dim), nn.GELU())
        self.pre = nn.Conv2d(enc_dim, hidden, 1) if hidden != enc_dim else nn.Identity()
        self.blocks = nn.ModuleList([GSTABlockCfg(hidden, k1=k1, k2=k2, d2=d2) for _ in range(n_blocks)])
        self.post = nn.Conv2d(hidden, enc_dim, 1) if hidden != enc_dim else nn.Identity()
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * enc_dim, enc_dim, 1), nn.GroupNorm(8, enc_dim), nn.GELU())
        self.head = nn.Conv2d(enc_dim, out_ch, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        skip = self.stem(z)
        h = self.pre(skip)
        for b in self.blocks:
            h = b(h)
        h = self.post(h)
        h = self.fuse(torch.cat([h, skip], dim=1))
        return self.head(h)


# --------------------------------------------------------------------------- #
# V4b: stronger downsampling (two stride-2 stages, 32x64 -> 8x16), a SINGLE
# skip only (from the full-res stem) -- isolates "does more downsampling
# help/hurt" from "does adding a second skip level help" (that's V5).
# --------------------------------------------------------------------------- #
class SimVP2StageSingleSkip(nn.Module):
    __name__ = "SimVP2StageSingleSkip"

    def __init__(self, in_ch: int = IN_CH, out_ch: int = OUT_CH, enc_dim: int = 128,
                 mid_dim: int = 192, hidden: int = 384, n_blocks: int = 8,
                 k1: int = 5, k2: int = 7, d2: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, enc_dim, 3, padding=1), nn.GroupNorm(8, enc_dim), nn.GELU())
        self.down1 = nn.Sequential(
            nn.Conv2d(enc_dim, mid_dim, 4, stride=2, padding=1), nn.GroupNorm(8, mid_dim), nn.GELU())
        self.down2 = nn.Sequential(
            nn.Conv2d(mid_dim, hidden, 4, stride=2, padding=1), nn.GroupNorm(8, hidden), nn.GELU())
        self.blocks = nn.ModuleList([GSTABlockCfg(hidden, k1=k1, k2=k2, d2=d2) for _ in range(n_blocks)])
        self.up2 = nn.ConvTranspose2d(hidden, mid_dim, 4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(mid_dim, enc_dim, 4, stride=2, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * enc_dim, enc_dim, 1), nn.GroupNorm(8, enc_dim), nn.GELU())
        self.head = nn.Conv2d(enc_dim, out_ch, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        skip = self.stem(z)             # 32x64, enc_dim
        h = self.down1(skip)             # 16x32, mid_dim
        h = self.down2(h)                # 8x16, hidden
        for b in self.blocks:
            h = b(h)
        h = self.up2(h)                  # 16x32, mid_dim
        h = self.up1(h)                  # 32x64, enc_dim
        h = self.fuse(torch.cat([h, skip], dim=1))
        return self.head(h)


# --------------------------------------------------------------------------- #
# V5: U-shaped SimVP -- two strided stages, a skip at EACH level (proper
# U-Net-style multi-scale skips), gSTA translator still only at the bottleneck.
# --------------------------------------------------------------------------- #
class SimVPUShape(nn.Module):
    __name__ = "SimVPUShape"

    def __init__(self, in_ch: int = IN_CH, out_ch: int = OUT_CH, enc_dim: int = 128,
                 mid_dim: int = 128, hidden: int = 256, n_blocks: int = 8,
                 k1: int = 5, k2: int = 7, d2: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, enc_dim, 3, padding=1), nn.GroupNorm(8, enc_dim), nn.GELU())
        self.down1 = nn.Sequential(
            nn.Conv2d(enc_dim, mid_dim, 4, stride=2, padding=1), nn.GroupNorm(8, mid_dim), nn.GELU())
        self.down2 = nn.Sequential(
            nn.Conv2d(mid_dim, hidden, 4, stride=2, padding=1), nn.GroupNorm(8, hidden), nn.GELU())
        self.blocks = nn.ModuleList([GSTABlockCfg(hidden, k1=k1, k2=k2, d2=d2) for _ in range(n_blocks)])
        self.up2 = nn.ConvTranspose2d(hidden, mid_dim, 4, stride=2, padding=1)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(2 * mid_dim, mid_dim, 1), nn.GroupNorm(8, mid_dim), nn.GELU())
        self.up1 = nn.ConvTranspose2d(mid_dim, enc_dim, 4, stride=2, padding=1)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(2 * enc_dim, enc_dim, 1), nn.GroupNorm(8, enc_dim), nn.GELU())
        self.head = nn.Conv2d(enc_dim, out_ch, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        skip1 = self.stem(z)             # 32x64, enc_dim
        skip2 = self.down1(skip1)        # 16x32, mid_dim
        h = self.down2(skip2)            # 8x16, hidden
        for b in self.blocks:
            h = b(h)
        h = self.up2(h)                  # 16x32, mid_dim
        h = self.fuse2(torch.cat([h, skip2], dim=1))
        h = self.up1(h)                  # 32x64, enc_dim
        h = self.fuse1(torch.cat([h, skip1], dim=1))
        return self.head(h)


# --------------------------------------------------------------------------- #
# V6: SimVP-original-style temporal handling -- a shared per-frame 2D stem +
# strided encoder (weights shared across all T=20 frames, applied as one
# batched (B*T, ...) conv call), gSTA translator mixes the combined (T x C')
# channel axis at the latent resolution, a shared per-frame decoder maps back.
# The 4 broadcast ctx-norm maps are not literally part of any one frame, so
# they are concatenated onto every frame's 3 channels before the shared stem.
# --------------------------------------------------------------------------- #
class SimVPFrameShared(nn.Module):
    __name__ = "SimVPFrameShared"

    def __init__(self, enc_dim: int = 48, frame_dim: int = 16, n_blocks: int = 8,
                 k1: int = 5, k2: int = 7, d2: int = 3):
        super().__init__()
        self.enc_dim = enc_dim
        self.frame_dim = frame_dim
        per_frame_in = C_IN + EXTRA_CHANNELS  # 3 + 4 = 7
        self.stem = nn.Sequential(
            nn.Conv2d(per_frame_in, enc_dim, 3, padding=1), nn.GroupNorm(8, enc_dim), nn.GELU())
        self.down = nn.Sequential(
            nn.Conv2d(enc_dim, frame_dim, 4, stride=2, padding=1), nn.GroupNorm(8, frame_dim), nn.GELU())
        hidden = T_STEPS * frame_dim
        self.blocks = nn.ModuleList([GSTABlockCfg(hidden, k1=k1, k2=k2, d2=d2) for _ in range(n_blocks)])
        self.up = nn.ConvTranspose2d(frame_dim, enc_dim, 4, stride=2, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * enc_dim, enc_dim, 1), nn.GroupNorm(8, enc_dim), nn.GELU())
        self.head = nn.Conv2d(enc_dim, C_OUT, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        frames = z[:, :T_STEPS * C_IN].reshape(b, T_STEPS, C_IN, HH, WW)
        extra = z[:, T_STEPS * C_IN:]
        extra_t = extra.unsqueeze(1).expand(b, T_STEPS, EXTRA_CHANNELS, HH, WW)
        x = torch.cat([frames, extra_t], dim=2).reshape(
            b * T_STEPS, C_IN + EXTRA_CHANNELS, HH, WW)

        skip = self.stem(x)                            # (B*T, enc_dim, 32, 64)
        h = self.down(skip)                              # (B*T, frame_dim, 16, 32)
        _, fd, hh2, ww2 = h.shape
        h = h.reshape(b, T_STEPS * fd, hh2, ww2)          # (B, T*frame_dim, 16, 32)
        for blk in self.blocks:
            h = blk(h)
        h = h.reshape(b * T_STEPS, fd, hh2, ww2)
        h = self.up(h)                                    # (B*T, enc_dim, 32, 64)
        h = self.fuse(torch.cat([h, skip], dim=1))
        out = self.head(h)                                 # (B*T, C_OUT, 32, 64)
        return out.reshape(b, T_STEPS * C_OUT, HH, WW)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
_CLS = {
    "1stage": SimVP1Stage,
    "nodown": SimVPNoDown,
    "2stage_singleskip": SimVP2StageSingleSkip,
    "ushape": SimVPUShape,
    "frameshared": SimVPFrameShared,
}

VARIANTS: dict[str, dict] = {
    # sanity replica of n1_simvp (enc128/hidden256/8blk/k1=5,k2=7,d2=3) --
    # should match C11A.N1SimVP's param count exactly (7,111,868).
    "v0_baseline_replica": dict(cls="1stage", enc_dim=128, hidden=256, n_blocks=8),

    # V1: depth
    "v1_depth12": dict(cls="1stage", enc_dim=128, hidden=256, n_blocks=12),
    "v1_depth16": dict(cls="1stage", enc_dim=128, hidden=256, n_blocks=16),

    # V2: hidden width
    "v2_hidden384": dict(cls="1stage", enc_dim=128, hidden=384, n_blocks=8),

    # V3: kernel (eff RF: baseline k1=5,k2=7,d2=3 -> 23)
    "v3_kernel31": dict(cls="1stage", enc_dim=128, hidden=256, n_blocks=8, k1=7, k2=9, d2=3),  # RF=31
    "v3_kernel11": dict(cls="1stage", enc_dim=128, hidden=256, n_blocks=8, k1=3, k2=5, d2=2),  # RF=11

    # V4: latent resolution
    "v4_nodown": dict(cls="nodown", enc_dim=128, hidden=128, n_blocks=8),
    "v4_strongdown": dict(cls="2stage_singleskip", enc_dim=128, mid_dim=192, hidden=384, n_blocks=8),

    # V5: U-shaped (two skips)
    "v5_ushape": dict(cls="ushape", enc_dim=128, mid_dim=128, hidden=256, n_blocks=8),

    # V6: SimVP-original temporal handling (shared per-frame stem/decoder)
    "v6_frameshared": dict(cls="frameshared", enc_dim=48, frame_dim=16, n_blocks=8),

    # V7: dilation cycling on the large-kernel branch
    "v7_dilcycle": dict(cls="1stage", enc_dim=128, hidden=256, n_blocks=8, k1=5, k2=7,
                         dilations=(1, 2, 3)),

    # V8: combine the two structurally-orthogonal positive factors from V1-V7
    # (hidden width 384 from V2 + kernel RF~11 from V3) -- added after V1-V7
    # results were in, per the family brief.
    "v8_combo": dict(cls="1stage", enc_dim=128, hidden=384, n_blocks=8, k1=3, k2=5, d2=2),
}


def build_core(name: str, **overrides) -> nn.Module:
    if name not in VARIANTS:
        raise ValueError(f"unknown c14a variant {name!r}; choices: {sorted(VARIANTS)}")
    cfg = dict(VARIANTS[name])
    cfg.update(overrides)
    cls_key = cfg.pop("cls")
    return _CLS[cls_key](**cfg)


MODEL_NAMES = tuple(VARIANTS.keys())
