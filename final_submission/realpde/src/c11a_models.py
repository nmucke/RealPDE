"""C11-A -- second screen of backbone families, dropped into the SAME ctx-norm
wrapper / Welsch recipe as the shipped `unet_local` (`src/c4r1_models.py`,
`experiments/c7a_robust_centre/notes.md`). Only the inner network changes.

Contract (identical for every model here, verified by `scripts/c11a_unit_checks.py`):
    core(z)  where z: (B, 64, 32, 64)  ->  (B, 60, 32, 64)

The 64 input channels are the 20-frame window folded to channels (T=20 outer,
C_in=3 inner, i.e. flat index t*3+c -- the same convention
`unet_convnext.UNetConvNeXt.forward` uses for its own stem) plus the 4
broadcast ctx-norm conditioning channels (m_u, m_v, log s_u, log s_v); the 60
output channels unfold the same way to (T=20, C_out=3). `fold_window` /
`unfold_output` below implement exactly that reshape both directions so every
model's internals only ever see plain (B, C, H, W) tensors.

`GenericCtxNormWrapper` is `c4r1_models.CtxNormWrapper`'s streaming contract
(`reset()` / `step(x)`, causal (S1, S2, n) accumulation, same variance floor)
re-implemented around an arbitrary `core` instead of a `CtxNormUNet` -- it
imports `c4r1_models`'s pure functions (`causal_ms`, `standardise_input`,
`unstandardise_output`, `build_extra_maps`) unedited, so the causal-statistics
math is byte-identical to the shipped pipeline; only the "what happens between
standardise and un-standardise" step is swapped for a generic `core.forward`.

New file, per the family's file-scope rule. Imports `c4r1_models` (read-only)
and stdlib/torch only.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import c4r1_models as CTX  # causal_ms, standardise_input, unstandardise_output, build_extra_maps

# --------------------------------------------------------------------------- #
# shape contract
# --------------------------------------------------------------------------- #
T_STEPS, HH, WW = 20, 32, 64
C_IN, C_OUT = 3, 3
EXTRA_CHANNELS = 4
IN_CH = T_STEPS * C_IN + EXTRA_CHANNELS   # 64
OUT_CH = T_STEPS * C_OUT                  # 60


def fold_window(xs: torch.Tensor) -> torch.Tensor:
    """(B,T,H,W,C) input-normalised window -> (B,T*C,H,W), T outer / C inner."""
    b, t, h, w, c = xs.shape
    return xs.permute(0, 1, 4, 2, 3).reshape(b, t * c, h, w)


def unfold_output(z: torch.Tensor) -> torch.Tensor:
    """(B,T*C_out,H,W) -> (B,T,H,W,C_out)."""
    b, tc, h, w = z.shape
    t = tc // C_OUT
    return z.reshape(b, t, C_OUT, h, w).permute(0, 1, 3, 4, 2)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def fp16_mb(model: nn.Module) -> float:
    """Same recipe as `scripts/c1d_bench_latency.py`'s inline estimate:
    real floats -> half (2B), complex -> view_as_real().half() (4B), no I/O."""
    nbytes = 0
    for v in model.state_dict().values():
        if not torch.is_tensor(v):
            continue
        if v.is_complex():
            nbytes += v.numel() * 4
        elif torch.is_floating_point(v):
            nbytes += v.numel() * 2
        else:
            nbytes += v.numel() * v.element_size()
    return nbytes / 1e6


# --------------------------------------------------------------------------- #
# N1 -- SimVP-style: strided conv encoder -> gSTA blocks @ 16x32 -> transposed
#       conv decoder with an encoder skip. GroupNorm only, no BatchNorm.
# --------------------------------------------------------------------------- #
class GSTABlock(nn.Module):
    """Gated spatio-temporal attention block (SimVP/TAU-style large-kernel
    depthwise "attention" + gate), plus an inverted-bottleneck MLP sub-block
    (standard transformer-block shape) for capacity. Kernel 21 receptive field
    via a 5x5 depthwise conv followed by a 7x7 depthwise conv dilated 3
    (dilated RF = 3*(7-1)+1 = 19, combined with the 5x5 -> ~21 total, per the
    family brief)."""

    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, dim)
        self.proj_in = nn.Conv2d(dim, dim, 1)
        self.dw5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.dw7d3 = nn.Conv2d(dim, dim, 7, padding=9, dilation=3, groups=dim)
        self.gate = nn.Conv2d(dim, dim, 1)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()

        self.norm2 = nn.GroupNorm(8, dim)
        self.mlp1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.mlp2 = nn.Conv2d(mlp_ratio * dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.act(self.proj_in(self.norm1(x)))
        attn = self.dw7d3(self.dw5(u))
        g = torch.sigmoid(self.gate(u))
        x = x + self.proj_out(attn * g)
        h = self.act(self.mlp1(self.norm2(x)))
        x = x + self.mlp2(h)
        return x


class N1SimVP(nn.Module):
    __name__ = "N1_SimVP"

    def __init__(self, in_ch: int = IN_CH, out_ch: int = OUT_CH,
                 enc_dim: int = 128, hidden: int = 256, n_blocks: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, enc_dim, 3, padding=1), nn.GroupNorm(8, enc_dim), nn.GELU())
        self.down = nn.Sequential(
            nn.Conv2d(enc_dim, hidden, 4, stride=2, padding=1), nn.GroupNorm(8, hidden), nn.GELU())
        self.blocks = nn.ModuleList([GSTABlock(hidden) for _ in range(n_blocks)])
        self.up = nn.ConvTranspose2d(hidden, enc_dim, 4, stride=2, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * enc_dim, enc_dim, 1), nn.GroupNorm(8, enc_dim), nn.GELU())
        self.head = nn.Conv2d(enc_dim, out_ch, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        skip = self.stem(z)                     # (B, enc_dim, 32, 64)
        h = self.down(skip)                      # (B, hidden, 16, 32)
        for b in self.blocks:
            h = b(h)
        h = self.up(h)                            # (B, enc_dim, 32, 64)
        h = self.fuse(torch.cat([h, skip], dim=1))
        return self.head(h)


# --------------------------------------------------------------------------- #
# N2 -- large-kernel ConvNeXt, full resolution, no U-shape.
# --------------------------------------------------------------------------- #
class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class LKConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, kernel: int = 13, mlp_ratio: int = 4, layer_scale: float = 0.1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.pw2 = nn.Conv2d(mlp_ratio * dim, dim, 1)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.dwconv(x))
        h = self.pw2(F.gelu(self.pw1(h)))
        return x + h * self.gamma[None, :, None, None]


class N2LargeKernelConvNeXt(nn.Module):
    __name__ = "N2_LKConvNeXt"

    def __init__(self, in_ch: int = IN_CH, out_ch: int = OUT_CH,
                 dim: int = 256, depth: int = 9, kernel: int = 13):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, dim, 3, padding=1)
        self.blocks = nn.ModuleList([LKConvNeXtBlock(dim, kernel) for _ in range(depth)])
        self.head_norm = LayerNorm2d(dim)
        self.head = nn.Conv2d(dim, out_ch, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.stem(z)
        for b in self.blocks:
            h = b(h)
        return self.head(self.head_norm(h))


# --------------------------------------------------------------------------- #
# N3 -- axial-attention / ViT hybrid. Plain matmul/softmax attention (no
#       nn.functional.scaled_dot_product_attention / flash paths).
# --------------------------------------------------------------------------- #
class AxialAttention(nn.Module):
    """Multi-head self-attention along one spatial axis at a time, hand-rolled
    with matmul + softmax (no SDPA/flash path, so it runs identically under
    plain eager PyTorch 2.2.2)."""

    def __init__(self, dim: int, n_heads: int = 7):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.hd = dim // n_heads
        self.scale = self.hd ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, L, D) -- N independent sequences of length L, attention over L."""
        n, l, d = x.shape
        qkv = self.qkv(x).reshape(n, l, 3, self.n_heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                  # each (n, heads, l, hd)
        attn = (q @ k.transpose(-2, -1)) * self.scale     # (n, heads, l, l)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v                                    # (n, heads, l, hd)
        out = out.transpose(1, 2).reshape(n, l, d)
        return self.out(out)


class AxialBlock(nn.Module):
    def __init__(self, dim: int, h: int, w: int, n_heads: int = 7, mlp_ratio: int = 4):
        super().__init__()
        self.h, self.w = h, w
        self.norm_row = nn.LayerNorm(dim)
        self.row_attn = AxialAttention(dim, n_heads)
        self.norm_col = nn.LayerNorm(dim)
        self.col_attn = AxialAttention(dim, n_heads)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(),
                                  nn.Linear(mlp_ratio * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H*W, D)."""
        b, n, d = x.shape
        h, w = self.h, self.w
        # row attention: attend over W within each row
        r = self.norm_row(x).reshape(b, h, w, d).reshape(b * h, w, d)
        r = self.row_attn(r).reshape(b, h, w, d).reshape(b, n, d)
        x = x + r
        # column attention: attend over H within each column
        c = self.norm_col(x).reshape(b, h, w, d).permute(0, 2, 1, 3).reshape(b * w, h, d)
        c = self.col_attn(c).reshape(b, w, h, d).permute(0, 2, 1, 3).reshape(b, n, d)
        x = x + c
        x = x + self.mlp(self.norm_mlp(x))
        return x


class N3AxialViT(nn.Module):
    __name__ = "N3_AxialViT"

    def __init__(self, in_ch: int = IN_CH, out_ch: int = OUT_CH,
                 dim: int = 224, depth: int = 8, patch: int = 2, n_heads: int = 7):
        super().__init__()
        self.patch = patch
        self.Hp, self.Wp = HH // patch, WW // patch
        self.embed = nn.Conv2d(in_ch, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.Hp * self.Wp, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [AxialBlock(dim, self.Hp, self.Wp, n_heads) for _ in range(depth)])
        self.norm_f = nn.LayerNorm(dim)
        self.unembed = nn.ConvTranspose2d(dim, out_ch, patch, stride=patch)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        t = self.embed(z)                              # (B, dim, Hp, Wp)
        t = t.flatten(2).transpose(1, 2) + self.pos      # (B, Hp*Wp, dim)
        for blk in self.blocks:
            t = blk(t)
        t = self.norm_f(t)
        t = t.transpose(1, 2).reshape(b, -1, self.Hp, self.Wp)
        return self.unembed(t)


# --------------------------------------------------------------------------- #
# N4 -- dilated residual network, full resolution, GroupNorm.
# --------------------------------------------------------------------------- #
class DilatedResBlock(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=dilation, dilation=dilation)
        self.gn1 = nn.GroupNorm(8, dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=dilation, dilation=dilation)
        self.gn2 = nn.GroupNorm(8, dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return self.act(x + h)


class N4DilatedResNet(nn.Module):
    __name__ = "N4_DilatedResNet"

    def __init__(self, in_ch: int = IN_CH, out_ch: int = OUT_CH,
                 dim: int = 144, n_blocks: int = 14, dilations: Sequence[int] = (1, 2, 4, 8)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.SiLU())
        self.blocks = nn.ModuleList(
            [DilatedResBlock(dim, dilations[i % len(dilations)]) for i in range(n_blocks)])
        self.head = nn.Conv2d(dim, out_ch, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.stem(z)
        for b in self.blocks:
            h = b(h)
        return self.head(h)


# --------------------------------------------------------------------------- #
# N5 -- "classic" residual U-Net: 3x3 (not 7x7 depthwise) convs, SiLU (not
#       GELU), GroupNorm (not LayerNorm), same widths/depths as unet_local.
#       Control for "is it ConvNeXt specifically".
# --------------------------------------------------------------------------- #
class ClassicResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return self.act(x + h)


class ClassicStage(nn.Module):
    def __init__(self, dim: int, n: int):
        super().__init__()
        self.blocks = nn.ModuleList([ClassicResBlock(dim) for _ in range(n)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class N5ClassicUNet(nn.Module):
    __name__ = "N5_ClassicUNet"

    def __init__(self, in_ch: int = IN_CH, out_ch: int = OUT_CH,
                 widths: Sequence[int] = (56, 112, 224, 448), depths: Sequence[int] = (2, 2, 2, 2)):
        super().__init__()
        assert len(widths) == len(depths) >= 2
        self.n_levels = len(widths)
        self.stem = nn.Conv2d(in_ch, widths[0], 3, padding=1)

        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(self.n_levels):
            self.enc.append(ClassicStage(widths[i], depths[i]))
            if i < self.n_levels - 1:
                self.down.append(nn.Sequential(
                    nn.GroupNorm(8, widths[i]), nn.SiLU(),
                    nn.Conv2d(widths[i], widths[i + 1], 2, stride=2)))

        self.up = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(self.n_levels - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(widths[i], widths[i - 1], 2, stride=2))
            self.fuse.append(nn.Conv2d(2 * widths[i - 1], widths[i - 1], 1))
            self.dec.append(ClassicStage(widths[i - 1], depths[i - 1]))

        self.head_norm = nn.GroupNorm(8, widths[0])
        self.head_act = nn.SiLU()
        self.head = nn.Conv2d(widths[0], out_ch, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.stem(z)
        skips = []
        for i in range(self.n_levels):
            h = self.enc[i](h)
            if i < self.n_levels - 1:
                skips.append(h)
                h = self.down[i](h)
        for j, i in enumerate(range(self.n_levels - 1, 0, -1)):
            h = self.up[j](h)
            h = self.fuse[j](torch.cat([h, skips[i - 1]], dim=1))
            h = self.dec[j](h)
        return self.head(self.head_act(self.head_norm(h)))


# --------------------------------------------------------------------------- #
# builder registry
# --------------------------------------------------------------------------- #
MODEL_NAMES = ("n1_simvp", "n2_lkconvnext", "n3_axialvit", "n4_dilated", "n5_classicunet",
               "n1_simvp_2x")


def build_core(name: str) -> nn.Module:
    if name == "n1_simvp":
        return N1SimVP()
    if name == "n1_simvp_2x":
        # N6: the winner at ~2x width (enc_dim/hidden doubled), only run because
        # n1_simvp cleared the proxy gate (experiments/c11a_backbones2/notes.md S4).
        return N1SimVP(enc_dim=256, hidden=512)
    if name == "n2_lkconvnext":
        return N2LargeKernelConvNeXt()
    if name == "n3_axialvit":
        return N3AxialViT()
    if name == "n4_dilated":
        return N4DilatedResNet()
    if name == "n5_classicunet":
        return N5ClassicUNet()
    raise ValueError(f"unknown c11a model {name!r}; choices: {MODEL_NAMES}")


# --------------------------------------------------------------------------- #
# generic ctx-norm streaming wrapper (mirrors c4r1_models.CtxNormWrapper's
# contract exactly; only the "what happens to the standardised window" middle
# step is generic instead of tied to UNetConvNeXt's stem/enc/dec/head).
# --------------------------------------------------------------------------- #
class GenericCtxNormCore(nn.Module):
    """Folds the (already standardised) window + extra maps to (B,64,H,W),
    runs an arbitrary `core(z) -> (B,60,H,W)`, unfolds back to (B,T,H,W,3)."""

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, xs: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        b, t, h, w, c = xs.shape
        assert (t, h, w, c) == (T_STEPS, HH, WW, C_IN), xs.shape
        assert extra.shape == (b, h, w, EXTRA_CHANNELS), extra.shape
        z = torch.cat([fold_window(xs), extra.permute(0, 3, 1, 2)], dim=1)
        assert z.shape[1] == IN_CH, z.shape
        out = self.core(z)
        assert out.shape[1] == OUT_CH, out.shape
        return unfold_output(out)


class GenericCtxNormWrapper(nn.Module):
    """Stateful streaming wrapper, byte-for-byte the same causal-statistics
    contract as `c4r1_models.CtxNormWrapper` (imports its pure functions
    unedited), wrapping a `GenericCtxNormCore` instead of a `CtxNormUNet`."""

    def __init__(self, generic_core: GenericCtxNormCore, ctx_const: dict):
        super().__init__()
        self.core = generic_core
        self.ctx_const = ctx_const
        self.register_buffer("S1", None, persistent=False)
        self.register_buffer("S2", None, persistent=False)
        self.n = 0

    def reset(self) -> None:
        self.S1 = None
        self.S2 = None
        self.n = 0

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        uv = x[..., :2]
        window_sum1 = uv.sum(dim=1)
        window_sum2 = (uv * uv).sum(dim=1)
        if self.S1 is None:
            self.S1 = window_sum1.detach().clone()
            self.S2 = window_sum2.detach().clone()
        else:
            self.S1 = self.S1 + window_sum1.detach()
            self.S2 = self.S2 + window_sum2.detach()
        self.n += x.shape[1]

        m, s = CTX.causal_ms(self.S1, self.S2, self.n)
        xs = CTX.standardise_input(x, m, s)
        extra = CTX.build_extra_maps(m, s, self.ctx_const)
        y_s = self.core(xs, extra)
        return CTX.unstandardise_output(y_s, m, s)
