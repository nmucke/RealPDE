"""I1 -- "element-equal" (standardised-space) losses for the ctx-norm UNet.

Hypothesis (family brief): the shipped loss (`src/c7a_losses.py`'s Welsch,
computed in the ORIGINAL target-normalised space) spends its capacity on the
energetic wake, which dominates rel_l2, while 28% of the final score (SPS)
rewards every element EQUALLY relative to its own local scale. This module
adds a Welsch surrogate computed in the STANDARDISED space the ctx-norm
backbone already produces internally (`y_hat_s`, before `unstandardise_output`)
-- i.e. every pixel's residual is measured in units of its own causal std `s`,
so a fixed threshold `h_std` treats calm and wake pixels identically.

Read-only w.r.t. every existing module: imports only `torch`, `c4r1_models`
(`CTX.standardise_input`, to standardise the TARGET `y` with the exact same
causal `(m, s)` the wrapper/training loop already computed for the input --
no new statistic, no separate causality surface) and `c7a_losses`
(`welsch_loss`, the shipped original-space term, reused unmodified for S2/S3).

Design notes
------------
- `y` (target-normalised) and `x` (input-normalised) share the same per-
  channel physical (mean, std) pair (documented in
  `experiments/c7a_robust_centre/notes.md` S0 and `c7a_finetune.py`'s
  `STD_TGT_UV`/`mean_std_real.pt`: input and target normalisation use the
  IDENTICAL constants), so the causal `(m, s)` computed from the INPUT
  window (`c4r1_models.causal_ms`, via `gather_ms`) standardises the TARGET
  `y` correctly too: `CTX.standardise_input(y, m, s)` -- literally the same
  function call the training loop already uses for `x`, applied to `y`
  instead. This is what "the target's standardisation uses the SAME causal
  (m, s) as the input" means operationally, and it is why no new causal
  statistic is introduced by this family.
- `s` is exactly the wrapper's own floored `s` (`c4r1_models.causal_ms`'s
  `FLOOR_FRAC=0.15` floor, already applied by the caller before `y_hat_s`
  is produced) -- "clamp s from below exactly as the wrapper does" is
  satisfied by construction (no second clamp is applied here).
- Masking: none, on all 3 channels (u, v, p), exactly mirroring
  `c7a_losses.welsch_loss` ("mask as the shipped loss does" -- the shipped
  loss does not mask; `p` is identically 0 on real data and reuses the u
  threshold in both spaces).
- `h_std = 0.30` (S1 family spec, a priori, standardised units) applies
  UNIFORMLY to u, v, p -- in standardised space every pixel's local scale
  has already been divided out, so (unlike the original-space `h_phys /
  std_tgt_c` conversion in `c7a_finetune.welsch_h_norm`) no per-channel
  physical-std conversion is needed or appropriate; that uniformity is the
  entire point of the "element-equal" family.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

import c4r1_models as CTX
from c7a_losses import welsch_loss  # shipped original-space term, reused unmodified

H_STD_DEFAULT = 0.30  # S1 a priori standardised-space Welsch scale, uniform u/v/p


def standardised_target(y: torch.Tensor, m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """y (B,T,32,64,3) target-normalised -> y_std, same transform/floor as the
    input window's own standardisation (`CTX.standardise_input`), reused
    verbatim (not reimplemented) since input and target share one physical
    normalisation constant per channel."""
    return CTX.standardise_input(y, m, s)


def std_welsch_loss(y_hat_s: torch.Tensor, y: torch.Tensor, m: torch.Tensor,
                     s: torch.Tensor, h_std: float = H_STD_DEFAULT) -> torch.Tensor:
    """S1: Welsch loss entirely in standardised space.
    y_hat_s: raw network output (standardised space, pre-`unstandardise_output`).
    y: target-normalised label (ordinary space); standardised here with the
    SAME (m, s) already used to standardise the input window.
    """
    y_std = standardised_target(y, m, s)
    hv = (h_std, h_std, h_std)
    return welsch_loss(y_hat_s, y_std, h=hv)


class MixedWelschLoss:
    """S2/S3: `w_orig * norm_orig * L_orig + w_std * norm_std * L_std`, where
    `L_orig` is the shipped original-space Welsch term (`c7a_losses.welsch_loss`,
    same `h_orig` the shipped recipe uses) and `L_std` is `std_welsch_loss`'s
    (S1) term. `norm_orig`/`norm_std` are fixed reciprocals of each term's OWN
    value on the very first training batch (iteration 0), captured lazily on
    the first `__call__` so both terms start at equal size (1.0) before the
    weights are applied -- stated explicitly per the family brief ("each
    normalised by its own running magnitude at iteration 0").
    """

    def __init__(self, w_orig: float, w_std: float, h_orig: Sequence[float],
                 h_std: float = H_STD_DEFAULT):
        assert abs(w_orig + w_std - 1.0) < 1e-9, (w_orig, w_std)
        self.w_orig = w_orig
        self.w_std = w_std
        self.h_orig = h_orig
        self.h_std = h_std
        self.norm_orig: Optional[float] = None
        self.norm_std: Optional[float] = None

    def __call__(self, pred: torch.Tensor, y: torch.Tensor, y_hat_s: torch.Tensor,
                 m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        l_orig = welsch_loss(pred, y, h=self.h_orig)
        l_std = std_welsch_loss(y_hat_s, y, m, s, h_std=self.h_std)
        if self.norm_orig is None:
            # iteration 0: fix both normalisers from this batch's own raw values.
            self.norm_orig = 1.0 / max(float(l_orig.detach()), 1e-12)
            self.norm_std = 1.0 / max(float(l_std.detach()), 1e-12)
        return self.w_orig * self.norm_orig * l_orig + self.w_std * self.norm_std * l_std


def weighted_orig_welsch_loss(pred: torch.Tensor, y: torch.Tensor,
                               h_orig: Sequence[float],
                               weight_map: torch.Tensor) -> torch.Tensor:
    """S4 (optional): original-space Welsch with STATIC per-pixel weights
    `weight_map` (32,64,3), w ~ 1/(s_global_pixel + eps) -- a milder,
    non-causal analogue of S1 (fixed at train time from pooled TRAIN-set
    statistics, not updated online). `weight_map` is normalised by the
    caller (`i1_finetune.py`) to have mean 1 over scored pixels so the
    overall loss scale matches the unweighted shipped Welsch term.
    pred/y: (B,T,32,64,3).
    """
    hv = torch.as_tensor(h_orig, dtype=pred.dtype, device=pred.device).view(
        *([1] * (pred.dim() - 1)), -1)
    r = pred - y
    elem = 1.0 - torch.exp(-(r * r) / (2.0 * hv * hv))
    w = weight_map.to(pred.dtype).to(pred.device)          # (32,64,3)
    w = w.view(1, 1, *w.shape)                              # (1,1,32,64,3) broadcasts over B,T
    return (elem * w).mean()
