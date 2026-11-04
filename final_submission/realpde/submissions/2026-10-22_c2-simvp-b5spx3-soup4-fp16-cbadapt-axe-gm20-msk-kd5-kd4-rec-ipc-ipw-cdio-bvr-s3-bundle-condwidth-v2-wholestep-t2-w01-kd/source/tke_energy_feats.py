"""Extra per-window features for the energy forecaster (TKE tracking track).

`notes/research_2026-09-15/tke_analysis.md` and `scripts/tke_tracking.py` measure the one
number that matters for TKE: the across-window tracking correlation of the predicted and
true per-pixel energy. `src/ke_net_flow.py` already feeds the net the input frames and the
FNO fluctuation as scale-free fields; this module adds the two channel groups the analysis
left open, both cheap and both about *when in the shedding cycle* the next window falls:

* `lead_energy_features` -- the FNO forecast's fluctuation energy resolved in lead time.
  The window's `var_t(F')` is one number per pixel for the whole 20-frame horizon, so a
  forecast whose energy rises or falls across the window looks the same as a flat one.
  Splitting the leads into `n_blocks` blocks and feeding `log(var_block / var_window)`
  exposes that shape (2 channels per block).

* `phase_features` -- where in the shedding cycle the input window ends. Because the
  target window's energy is essentially white from window to window (lag-1 autocorrelation
  ~0 at Re >= 20k), any predictable part of it has to come from the *state*, and the cycle
  phase is the most compact state descriptor there is. Two phases are offered, both as
  (cos, sin, frequency, amplitude) broadcast to constant maps:

  - `probe_phase`: the analytic-signal (Hilbert) angle of the wake-averaged transverse
    velocity `s(t) = sum_x w(x) v'(x, t)`, with `w` a fixed downstream indicator. This is
    the one that is comparable *across* windows -- its phase reference is the flow, not
    the window.
  - `pod_phase`: the angle of the leading POD pair's temporal coefficients. This is the
    textbook shedding phase, but measured on these 20-frame windows the leading mode is
    a half-cycle ramp (the window's own low-frequency content) and its sign convention
    comes from `eigh`, so the angle barely moves between windows; it is kept for the
    ablation, not recommended as the feature.

Both builders return `(B, C, H, W)` so they concatenate straight onto
`ke_net_flow.features`. Pure torch, no SciPy: the Hilbert transform is an rFFT.
"""

from __future__ import annotations

from typing import Tuple

import torch

EPS = 1e-8
LOG_CLAMP = 6.0


def n_lead_features(n_blocks: int) -> int:
    return 2 * n_blocks


def lead_energy_features(ffl: torch.Tensor, ef: torch.Tensor,
                         n_blocks: int = 4) -> torch.Tensor:
    """(B, T, H, W, 2) forecast fluctuation -> (B, 2*n_blocks, H, W) log energy shape.

    `ef` is the window's own `var_t(ffl)`, so the ratio is scale-free and the channel is
    exactly zero for a forecast whose energy is flat in lead time.
    """
    b, t = ffl.shape[0], ffl.shape[1]
    edges = torch.linspace(0, t, n_blocks + 1).round().long().tolist()
    out = []
    for i in range(n_blocks):
        blk = ffl[:, edges[i]:edges[i + 1]]
        v = ((blk - blk.mean(1, keepdim=True)) ** 2).mean(1)          # (B, H, W, 2)
        r = torch.log((v + EPS) / (ef + EPS)).clamp(-LOG_CLAMP, LOG_CLAMP)
        out.append(r.permute(0, 3, 1, 2))
    return torch.cat(out, 1)


def _hilbert_angle(a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Analytic-signal phase of a real (B, T) sequence: (angle at the last sample,
    median increment per sample). The Hilbert transform is one rFFT-length FFT."""
    t = a.shape[1]
    spec = torch.fft.fft(a.to(torch.float32), dim=1)
    h = torch.zeros(t, device=a.device, dtype=spec.dtype)
    h[0] = 1.0
    if t % 2 == 0:
        h[t // 2] = 1.0
        h[1:t // 2] = 2.0
    else:
        h[1:(t + 1) // 2] = 2.0
    ang = torch.angle(torch.fft.ifft(spec * h, dim=1))
    d = torch.diff(ang, dim=1)
    d = torch.remainder(d + torch.pi, 2 * torch.pi) - torch.pi
    return ang[:, -1].to(a.dtype), d.median(dim=1).values.to(a.dtype)


def probe_phase(xw: torch.Tensor, mask: torch.Tensor, col0: int = 24):
    """Shedding phase from the wake-averaged transverse velocity: (angle, freq, amp).

    `s(t) = sum_x w(x) v'(x, t)` over unoccluded pixels downstream of `col0`; the von
    Karman street makes it oscillate at the shedding frequency with a phase referenced to
    the flow rather than to the window, so consecutive windows' angles are comparable.
    `amp` is the signal's rms relative to the window's rms transverse fluctuation.
    """
    v = xw[..., 1]
    v = v - v.mean(1, keepdim=True)
    w = (~mask).to(xw.dtype).clone()
    w[..., :col0] = 0.0
    s = (v * w.unsqueeze(1)).sum(dim=(2, 3))                            # (B, T)
    ang, freq = _hilbert_angle(s)
    den = (v.pow(2) * w.unsqueeze(1)).mean(dim=(1, 2, 3)).clamp_min(EPS).sqrt()
    amp = (s.std(dim=1) / (den * w.sum(dim=(1, 2)).clamp_min(1.0))).clamp(0, 4)
    return ang, freq, amp


def pod_phase(xw: torch.Tensor, mask: torch.Tensor,
              n_modes: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Leading POD pair of the window's fluctuation -> (angle, frequency, amplitude).

    `xw` is (B, T, H, W, 2) and `mask` (B, H, W) with True = occluded (those pixels are
    dropped, they are exactly zero in the data). The temporal coefficients come from the
    eigenvectors of the T x T snapshot correlation matrix -- 20 x 20, so this is free --
    and the analytic signal of the leading coefficient gives the phase at the last frame
    and its mean increment per frame.

    Returns three (B,) tensors: the phase (radians, in [-pi, pi]), the frequency
    (radians/frame, the median increment) and the pair's amplitude relative to the
    window's total fluctuation energy.
    """
    b, t = xw.shape[0], xw.shape[1]
    keep = (~mask).to(xw.dtype).unsqueeze(-1)                          # (B, H, W, 1)
    f = (xw - xw.mean(1, keepdim=True)) * keep.unsqueeze(1)
    f = f.reshape(b, t, -1)                                            # (B, T, N)
    norm = f.pow(2).sum(dim=(1, 2)).clamp_min(EPS)
    c = torch.bmm(f, f.transpose(1, 2)) / t                            # (B, T, T)
    evals, evecs = torch.linalg.eigh(c.double())
    evals, evecs = evals.flip(-1).to(xw.dtype), evecs.flip(-1).to(xw.dtype)
    a1 = evecs[..., 0] * evals[:, :1].clamp_min(0).sqrt()              # (B, T)
    a2 = evecs[..., 1] * evals[:, 1:2].clamp_min(0).sqrt()
    amp = (evals[:, :n_modes].sum(1) * t / norm).clamp(0, 1)
    ang = torch.atan2(a2[:, -1], a1[:, -1])
    _, freq = _hilbert_angle(a1)
    return ang.to(xw.dtype), freq.to(xw.dtype), amp


N_PHASE_FEATURES = 4


def phase_features(xw: torch.Tensor, mask: torch.Tensor, kind: str = "probe") -> torch.Tensor:
    """(B, T, H, W, 2) input window -> (B, 4, H, W) constant phase maps."""
    ang, freq, amp = (probe_phase if kind == "probe" else pod_phase)(xw, mask)
    cols = torch.stack([ang.cos(), ang.sin(), freq * 3.0, amp], 1)      # (B, 4)
    return cols[..., None, None].expand(-1, -1, xw.shape[2], xw.shape[3]).contiguous()


def extra_features(xw: torch.Tensor, ffl: torch.Tensor, ef: torch.Tensor,
                   mask: torch.Tensor, leads: bool = True, phase: bool = True,
                   n_blocks: int = 4, kind: str = "probe") -> torch.Tensor:
    """Concatenation of whichever extra groups are switched on, (B, C, H, W)."""
    out = []
    if leads:
        out.append(lead_energy_features(ffl, ef, n_blocks))
    if phase:
        out.append(phase_features(xw, mask, kind))
    return torch.cat(out, 1) if out else xw.new_zeros(xw.shape[0], 0, *xw.shape[2:4])


def n_extra(leads: bool, phase: bool, n_blocks: int = 4) -> int:
    return (n_lead_features(n_blocks) if leads else 0) + (N_PHASE_FEATURES if phase else 0)
