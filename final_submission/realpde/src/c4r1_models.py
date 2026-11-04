"""C4-R1 -- causal in-context standardisation (RevIN-style) of the UNet backbone.

Hypothesis (family brief): the leaderboard shift is an unseen MEAN-FLOW regime;
the network currently must emit absolute fields it has never seen. This module
wraps `src/unet_convnext.py`'s `UNetConvNeXt` (via `src/c1d_models.py`, imported
not edited) so it only ever sees O(1) per-pixel anomalies:

  - a per-pixel CAUSAL running mean `m` and std `s` of the u,v channels of every
    frame revealed so far in the trajectory (accumulated from the *input window*
    only -- windows tile at stride 20, so the input covers `[0, t+20)` exactly
    once; using `prev_target` too would double-count those frames);
  - the network's input is `(x-m)/s` for u,v (`p` passed through unchanged --
    it is identically 0 on real data, `mean_in[2]=std_in[2]=0` confirmed against
    `starting_kit/example_data/mean_std_real.pt`), plus 4 broadcast conditioning
    channels `(m_u, m_v, log s_u, log s_v)`, each independently rescaled by a
    train-set constant so all four feed the network at roughly unit scale;
  - the network's output `y_hat_s` is un-standardised as `y_hat_s * s + m` for
    u,v (p passed through) before being compared to the ordinary
    target-normalised label `y` used everywhere else in this repo.

All of `m`, `s`, and the two running sums it is built from (`S1`, `S2`, plus a
scalar count `n`) are **causal**: at test time they are cleared in
`reset_ttt_state` and updated once per `ttt_step` from that step's own input
window before that window is standardised, exactly mirroring how
`scripts/c4r1_finetune.py` builds the same quantities from per-file prefix
sums (`causal_ms_from_cumsum`) so training sees the identical statistic the
streaming evaluator will compute, with no leakage from target frames.

Architecture change to make this possible: `UNetConvNeXt.stem` is a
`Conv2d(T*C_in, widths[0], 3, ...)` (`T*C_in` = 20*3 = 60, time folded into
channels, see `src/unet_convnext.py` docstring). `CtxNormUNet` replaces that
stem with a `Conv2d(64, widths[0], ...)` (60 folded window channels + 4
broadcast conditioning channels), copying the pretrained 60 input columns
unchanged and **zero-initialising** the 4 new columns, then re-implements the
rest of `forward()` (encoder/decoder/head) by calling the *same* public
submodules (`base.enc`, `base.down`, `base.up`, `base.fuse`, `base.dec`,
`base.head_norm`, `base.head`) `UNetConvNeXt.forward` already uses -- so
apart from the stem, every parameter tensor and its numerics are byte-for-byte
identical to the un-wrapped model. `src/unet_convnext.py` / `src/c1d_models.py`
are imported only, never edited (family rule).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

import c1d_models as C1D
from unet_convnext import UNetConvNeXt

EXTRA_CHANNELS = 4          # m_u, m_v, log s_u, log s_v
VAR_EPS = 1e-3               # variance floor, normalised-x units (design default)
# "global std" floor: x is normalised by the official train_real std_in, so by
# construction its global std over in-distribution (unmasked) pixels is ~1.0;
# using the exact constant 1.0 here (rather than re-measuring it on each local
# fold) is a documented simplification -- see experiments/c4r1_ctxnorm/notes.md
# S1. floor_frac=0.15 -> variance floor (0.15*1.0)**2 = 0.0225, i.e. s >= 0.15,
# which dominates VAR_EPS (0.001) and keeps masked/zero-variance pixels from
# blowing up when divided into.
GLOBAL_STD_UV = (1.0, 1.0)
FLOOR_FRAC = 0.15


def causal_ms(s1: torch.Tensor, s2: torch.Tensor, n, eps: float = VAR_EPS,
              floor_frac: float = FLOOR_FRAC,
              global_std=GLOBAL_STD_UV) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel causal mean/std from running sums.

    s1, s2: (..., 32, 64, 2) running sum / sum-of-squares of the u,v channels
        of every input-window frame revealed so far (INCLUDING the current
        window -- see module docstring).
    n: scalar or broadcastable frame count (>=1 whenever this is called; the
        wrapper never calls it before the first window has been accumulated).
    Returns (m, s), both (..., 32, 64, 2).
    """
    m = s1 / n
    var = s2 / n - m * m
    gs = torch.as_tensor(global_std, dtype=var.dtype, device=var.device)
    floor = torch.clamp_min(gs * floor_frac, 0.0) ** 2
    floor = torch.clamp_min(floor, eps)
    var = torch.clamp_min(var, floor)
    return m, torch.sqrt(var)


class CtxNormUNet(nn.Module):
    """`UNetConvNeXt` with an extended stem taking 4 extra broadcast channels.

    `base` keeps in_channels=3 / out_channels=3 (the ordinary window contract)
    everywhere except the stem, which is swapped for a
    `Conv2d(T*3 + EXTRA_CHANNELS, widths[0], ...)`. Call `forward(xs, extra)`
    with the *already context-standardised* window `xs` (B,T,32,64,3) and the
    broadcast maps `extra` (B,32,64,EXTRA_CHANNELS); the return value is in the
    same standardised space the base model would have produced (un-standardise
    with the wrapper's `m`,`s` before comparing to a physical/target-normalised
    label).
    """

    def __init__(self, base: Optional[UNetConvNeXt] = None):
        super().__init__()
        if base is None:
            base = C1D.build("unet_local")
        assert isinstance(base, UNetConvNeXt)
        assert base.cond_mlp is None, "C4-R1 wrapper assumes cond=False (matches unet_local)"
        self.base = base
        old_stem = base.stem
        c_in_old = old_stem.in_channels
        c_out = old_stem.out_channels
        new_stem = nn.Conv2d(c_in_old + EXTRA_CHANNELS, c_out,
                              kernel_size=old_stem.kernel_size, padding=old_stem.padding)
        with torch.no_grad():
            new_stem.weight.zero_()
            new_stem.weight[:, :c_in_old].copy_(old_stem.weight)
            new_stem.bias.copy_(old_stem.bias)
        base.stem = new_stem      # in-place swap; the rest of `base` is untouched

    def forward(self, xs: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        base = self.base
        b, t, h, w, c = xs.shape
        assert (t, h, w, c) == (base.T, base.Hs, base.Ws, base.in_channels), xs.shape
        assert extra.shape == (b, h, w, EXTRA_CHANNELS), extra.shape

        z = xs.permute(0, 1, 4, 2, 3).reshape(b, t * c, h, w)
        z = torch.cat([z, extra.permute(0, 3, 1, 2)], dim=1)
        z = base.stem(z)

        skips = []
        for i in range(base.n_levels):
            z = base.enc[i](z, None)
            if i < base.n_levels - 1:
                skips.append(z)
                z = base.down[i](z)
        for j, i in enumerate(range(base.n_levels - 1, 0, -1)):
            z = base.up[j](z)
            z = base.fuse[j](torch.cat([z, skips[i - 1]], dim=1))
            z = base.dec[j](z, None)

        z = base.head(base.head_norm(z))
        return z.reshape(b, t, base.out_dim, h, w).permute(0, 1, 3, 4, 2)


def build_extra_maps(m: torch.Tensor, s: torch.Tensor, ctx_const: dict) -> torch.Tensor:
    """(m_u, m_v, log s_u, log s_v), each independently rescaled by a
    train-set (mean, std) pair stored in `ctx_const` (keys `m_mean`, `m_std`,
    `logs_mean`, `logs_std`, each a length-2 (u,v) tensor/sequence)."""
    m_mean = torch.as_tensor(ctx_const["m_mean"], dtype=m.dtype, device=m.device)
    m_std = torch.as_tensor(ctx_const["m_std"], dtype=m.dtype, device=m.device)
    logs_mean = torch.as_tensor(ctx_const["logs_mean"], dtype=m.dtype, device=m.device)
    logs_std = torch.as_tensor(ctx_const["logs_std"], dtype=m.dtype, device=m.device)
    logs = torch.log(s)
    mc = (m - m_mean) / m_std
    lc = (logs - logs_mean) / logs_std
    return torch.cat([mc, lc], dim=-1)


def standardise_input(x: torch.Tensor, m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """x: (B,T,32,64,3) input-normalised window. m,s: (B,32,64,2) causal stats.
    Returns xs (B,T,32,64,3): u,v standardised, p passed through unchanged."""
    uv = (x[..., :2] - m.unsqueeze(1)) / s.unsqueeze(1)
    return torch.cat([uv, x[..., 2:3]], dim=-1)


def unstandardise_output(y_s: torch.Tensor, m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Inverse of `standardise_input`'s u,v transform, applied to the network's
    raw output (B,T,32,64,3) -> combined prediction in the ordinary
    target-normalised space."""
    uv = y_s[..., :2] * s.unsqueeze(1) + m.unsqueeze(1)
    return torch.cat([uv, y_s[..., 2:3]], dim=-1)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class CtxNormWrapper(nn.Module):
    """Stateful streaming wrapper: `CtxNormUNet` + the running (S1, S2, n)
    causal accumulator, one call per `ttt_step`-shaped window.

    `reset()` clears the state (trajectory boundary); `step(x)` accumulates
    `x`'s own u,v frames into the running sums *first*, computes `(m, s)` from
    the updated sums (so the current window's own frames are included, exactly
    matching `scripts/c4r1_finetune.py`'s `[0, t+20)` training target), then
    returns the un-standardised prediction in the ordinary target-normalised
    space (drop-in replacement for a plain backbone's `forward(x)`).
    """

    def __init__(self, ctx_unet: CtxNormUNet, ctx_const: dict):
        super().__init__()
        self.ctx_unet = ctx_unet
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
        """x: (B,20,32,64,3) input-normalised window (organizer convention)."""
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

        m, s = causal_ms(self.S1, self.S2, self.n)
        xs = standardise_input(x, m, s)
        extra = build_extra_maps(m, s, self.ctx_const)
        y_s = self.ctx_unet(xs, extra)
        return unstandardise_output(y_s, m, s)
