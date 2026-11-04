"""Shipping copy of `src/c4r1_models.py`'s causal in-context standardisation
(RevIN-style) wrapper, with the reviewer's latency fix applied.

Family rule (C4-R1 / C4-R1-review): copy, do not edit `src/c4r1_models.py`
(`experiments/c4r1_ctxnorm/notes.md`, `experiments/c4r1_review/`). This module
is that copy, used only by the C4-P shipping candidate
(`submissions/2026-09-20_unet-ctxnorm/`).

Reviewer finding (`experiments/c4r1_review/latency_fix_check.py`,
`latency_fix.log`): the original `CtxNormWrapper.forward` and its helpers
(`causal_ms`, `build_extra_maps`) reconstruct FIVE small constant tensors
from Python floats/lists via `torch.as_tensor(...)` on EVERY `ttt_step` call
-- `GLOBAL_STD_UV` inside `causal_ms`, plus `m_mean`/`m_std`/`logs_mean`/
`logs_std` inside `build_extra_maps`. Each `torch.as_tensor` call is a fresh
host -> device tensor construction; measured cost: 26.09 ms mean / 25.43 ms
median per step (original) vs 6.21 ms mean / 5.98 ms median (pre-cached),
identical GPU otherwise. Fix (verified bit-identical, `latency_fix_check.py`:
"bit-identical outputs: True max abs diff: 0.0"): build those five tensors
ONCE, as registered buffers, when the wrapper is constructed (so `.to(device)`
moves them together with the model's own parameters), and reuse them on every
step instead of rebuilding.

Everything else (the causal accumulator semantics, the standardisation
transform, `CtxNormUNet`'s stem-extension trick) is copied UNCHANGED from
`src/c4r1_models.py` -- see that module's docstring for the design rationale.
This file re-states the numerics inline (rather than importing
`c4r1_models`) so the shipped module is self-contained and the fix is a
single, auditable diff against the original functions.
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
# fold) is a documented simplification, carried over unchanged from
# src/c4r1_models.py -- see experiments/c4r1_ctxnorm/notes.md S1 and this
# candidate's manifest ("GLOBAL_STD_UV floor is approximate", non-blocking
# per the C4-R1 reviewer). floor_frac=0.15 -> variance floor
# (0.15*1.0)**2 = 0.0225, i.e. s >= 0.15, which dominates VAR_EPS (0.001) and
# keeps masked/zero-variance pixels from blowing up when divided into.
GLOBAL_STD_UV = (1.0, 1.0)
FLOOR_FRAC = 0.15


def causal_ms(s1: torch.Tensor, s2: torch.Tensor, n, gs_cached: torch.Tensor,
              eps: float = VAR_EPS, floor_frac: float = FLOOR_FRAC,
              floor: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel causal mean/std from running sums -- identical numerics to
    `c4r1_models.causal_ms`, except `global_std` is passed in as an
    already-constructed device tensor (`gs_cached`, built once by the caller)
    instead of being rebuilt with `torch.as_tensor` on every call.

    c5t (speed): `floor = clamp_min(gs_cached * floor_frac, 0) ** 2` clamped
    to `eps` depends only on `gs_cached` / `floor_frac` / `eps` -- all fixed
    for the lifetime of a `CtxNormWrapper` instance (`gs_cached` is itself a
    registered buffer built once, never reassigned; `causal_ms` is always
    called with the default `eps`/`floor_frac`, i.e. every call recomputes
    the SAME tensor). The caller may pass that tensor in pre-built via the
    new `floor` kwarg to skip recomputing it every step; omitting it (the
    default) reproduces the original per-call computation bit-for-bit, so
    this is purely additive -- no existing call site's behaviour changes
    unless it opts in.

    s1, s2: (..., 32, 64, 2) running sum / sum-of-squares of the u,v channels
        of every input-window frame revealed so far (INCLUDING the current
        window -- see module docstring).
    n: scalar or broadcastable frame count (>=1 whenever this is called).
    Returns (m, s), both (..., 32, 64, 2).
    """
    m = s1 / n
    var = s2 / n - m * m
    if floor is None:
        floor = torch.clamp_min(gs_cached * floor_frac, 0.0) ** 2
        floor = torch.clamp_min(floor, eps)
    var = torch.clamp_min(var, floor)
    return m, torch.sqrt(var)


class CtxNormUNet(nn.Module):
    """`UNetConvNeXt` with an extended stem taking 4 extra broadcast channels.

    Byte-for-byte the same class as `c4r1_models.CtxNormUNet` (copied, not
    imported, per the family's "copy the wrapper" rule) -- no latency issue
    lives here, this class is untouched.
    """

    def __init__(self, base: Optional[UNetConvNeXt] = None):
        super().__init__()
        if base is None:
            base = C1D.build("unet_local")
        assert isinstance(base, UNetConvNeXt)
        assert base.cond_mlp is None, "ctx-norm wrapper assumes cond=False (matches unet_local)"
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


def build_extra_maps(m: torch.Tensor, s: torch.Tensor,
                      m_mean: torch.Tensor, m_std: torch.Tensor,
                      logs_mean: torch.Tensor, logs_std: torch.Tensor) -> torch.Tensor:
    """(m_u, m_v, log s_u, log s_v), each independently rescaled by a
    train-set (mean, std) pair. Same numerics as `c4r1_models.build_extra_maps`,
    except the four rescaling tensors are passed in pre-built (registered
    buffers on the wrapper, constructed once) instead of being rebuilt with
    `torch.as_tensor` from `ctx_const` on every call -- the fix."""
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
    causal accumulator -- FIXED version.

    Identical streaming semantics to `c4r1_models.CtxNormWrapper` (`reset()`
    clears state at a trajectory boundary; `step(x)`/`forward(x)` accumulates
    `x`'s own u,v frames into the running sums first, computes `(m, s)` from
    the updated sums, standardises, runs the network, un-standardises). The
    ONLY change: the five small constant tensors the original rebuilt via
    `torch.as_tensor` on every call (`GLOBAL_STD_UV` and the four `ctx_const`
    entries) are built ONCE here, at construction, as registered buffers
    (`persistent=False`, like the causal accumulator itself) -- so `.to(device)`
    moves them alongside the model's parameters and every subsequent
    `step()`/`forward()` call reuses the same device tensors. Verified
    bit-identical to the original wrapper's output (`torch.equal`, see
    `experiments/c4p_packaging/bit_identity_check.py`), ~4x faster on a
    contended GPU (`experiments/c4r1_review/latency_fix_check.py`:
    26.09 ms -> 6.21 ms mean per step, orig vs cached).
    """

    def __init__(self, ctx_unet: CtxNormUNet, ctx_const: dict):
        super().__init__()
        self.ctx_unet = ctx_unet
        self.ctx_const = ctx_const
        try:
            ref = next(ctx_unet.parameters())
            dev, dt = ref.device, ref.dtype
        except StopIteration:  # pragma: no cover - CtxNormUNet always has params
            dev, dt = torch.device("cpu"), torch.float32
        # the fix: build the five small constants ONCE, as buffers, instead of
        # reconstructing them with torch.as_tensor on every ttt_step.
        self.register_buffer("_gs", torch.as_tensor(GLOBAL_STD_UV, dtype=dt, device=dev),
                             persistent=False)
        self.register_buffer("_m_mean", torch.as_tensor(ctx_const["m_mean"], dtype=dt, device=dev),
                             persistent=False)
        self.register_buffer("_m_std", torch.as_tensor(ctx_const["m_std"], dtype=dt, device=dev),
                             persistent=False)
        self.register_buffer("_logs_mean", torch.as_tensor(ctx_const["logs_mean"], dtype=dt, device=dev),
                             persistent=False)
        self.register_buffer("_logs_std", torch.as_tensor(ctx_const["logs_std"], dtype=dt, device=dev),
                             persistent=False)
        self.register_buffer("S1", None, persistent=False)
        self.register_buffer("S2", None, persistent=False)
        self.n = 0
        # c5t (speed): causal_ms's variance floor depends only on _gs /
        # FLOOR_FRAC / VAR_EPS -- all fixed above and never varied by any
        # call site in this module -- so it is the same tensor on every
        # step. Precompute it once here (bit-identical to what causal_ms
        # would compute inline every call, see its own docstring) and reuse
        # it via the new `floor=` kwarg instead of recomputing 4 small ops
        # (mul, pow, 2x clamp_min) per step.
        self.register_buffer(
            "_ms_floor",
            torch.clamp_min(torch.clamp_min(self._gs * FLOOR_FRAC, 0.0) ** 2, VAR_EPS),
            persistent=False)

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

        m, s = causal_ms(self.S1, self.S2, self.n, self._gs, floor=self._ms_floor)
        xs = standardise_input(x, m, s)
        extra = build_extra_maps(m, s, self._m_mean, self._m_std, self._logs_mean, self._logs_std)
        y_s = self.ctx_unet(xs, extra)
        return unstandardise_output(y_s, m, s)
