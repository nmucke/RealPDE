"""C7-A -- robust losses for training an interval-CENTRE network.

All losses operate in the ordinary TARGET-NORMALISED space (same space
`scripts/c4r1_finetune.py`'s MSE loss uses: `MSE(y_hat_s * s + m, y)`), on the
full 3-channel (u, v, p) tensor with no masking, exactly mirroring the MSE
recipe. `p` is identically zero on real data and normalises to zero, so its
threshold choice is immaterial; it reuses the u threshold.

Read-only w.r.t. every existing module (only imports `torch`).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

# In target-normalised units, every channel's global std is 1.0 by
# construction (the normalisation divides by exactly this std), so a
# "delta = 0.1 x global std" threshold is simply 0.1 in this space -- verified
# empirically against starting_kit/example_data/mean_std_real.pt (std_tgt is
# what defines the space) in experiments/c7a_robust_centre/notes.md S0.
HUBER_DELTA_NORM = 0.1


def l1_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """R1: mean absolute error (-> conditional median)."""
    return (pred - y).abs().mean()


def huber_loss(pred: torch.Tensor, y: torch.Tensor,
               delta: float = HUBER_DELTA_NORM) -> torch.Tensor:
    """R2: elementwise Huber, quadratic inside +-delta, linear outside."""
    r = pred - y
    a = r.abs()
    quad = 0.5 * r * r
    lin = delta * (a - 0.5 * delta)
    return torch.where(a <= delta, quad, lin).mean()


def welsch_loss(pred: torch.Tensor, y: torch.Tensor,
                h: Sequence[float]) -> torch.Tensor:
    """R3: smooth "inside" surrogate (Welsch / correntropy),
    `1 - exp(-(pred-y)^2 / (2 h^2))`, per-channel `h` (3-vector: u, v, p),
    broadcast over the trailing channel dim. Bounded in [0, 1) per element,
    a smooth relaxation of `1{|r| > h}` that still has a gradient everywhere
    (unlike the hard indicator), directly modelling the SPS inside-indicator.
    """
    hv = torch.as_tensor(h, dtype=pred.dtype, device=pred.device).view(
        *([1] * (pred.dim() - 1)), -1)
    r = pred - y
    return (1.0 - torch.exp(-(r * r) / (2.0 * hv * hv))).mean()


def make_loss(name: str, huber_delta: float = HUBER_DELTA_NORM,
              welsch_h: Optional[Sequence[float]] = None):
    """Returns a `(pred, y) -> scalar` callable for `name` in {mse, l1, huber, welsch}."""
    if name == "mse":
        return lambda pred, y: torch.mean((pred - y) ** 2)
    if name == "l1":
        return l1_loss
    if name == "huber":
        return lambda pred, y: huber_loss(pred, y, delta=huber_delta)
    if name == "welsch":
        assert welsch_h is not None, "welsch loss needs --welsch-h"
        return lambda pred, y: welsch_loss(pred, y, h=welsch_h)
    raise ValueError(f"unknown loss {name!r}")
