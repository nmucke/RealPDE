#!/usr/bin/env python3
"""M1 shared machinery -- learned conditional (per-element) interval widths.

New code only (family M1). Read-only w.r.t. everything else in the repo:
reuses the already-validated primitives in `src/c6s_common.py` (exact SPS
re-scoring, optimal-k table, sigma<->half conversion, causal input
reconstruction from `lco_run_dump.py` dumps) rather than re-deriving them,
and adds what C6-S did not need: a local (space-time neighbourhood) error
smoothing oracle, causal window features shaped for a small conv net, and
the net itself.

Everything here operates on the `.npz` dumps written by
`scripts/lco_run_dump.py` (pred/target/lower/upper/sim_ids/time_ids/is_first,
physical units), loaded via `c6s_common.load_dump`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
import c6s_common as C  # noqa: E402

SIGMA_GLOBAL = C.SIGMA_GLOBAL


# --------------------------------------------------------------------------- #
# Local (space-time neighbourhood) error-scale oracle -- Stage 0(c).
# --------------------------------------------------------------------------- #
def local_error_scale(abs_resid: np.ndarray, n_lead: int, n_px: int,
                       scored: np.ndarray) -> np.ndarray:
    """Box-filtered |residual| over a (n_lead, n_px, n_px) neighbourhood.

    `abs_resid`, `scored`: (N, T, H, W) float32 / bool, per channel already.
    A *weighted* box filter (sum |r| * scored / sum scored) so unmeasured
    (target==0) pixels near the FOV/airfoil boundary don't bias the local
    scale toward zero. This is a HINDSIGHT oracle: it uses the true
    residual, smoothed, as an upper bound on what any smooth causal
    predictor of the error scale could achieve -- not a deployable rule.
    """
    from scipy.ndimage import uniform_filter

    size = (1, n_lead, n_px, n_px)  # per-window filtering only (axis 0 = window, untouched)
    num = uniform_filter((abs_resid * scored).astype(np.float64), size=size, mode="nearest")
    den = uniform_filter(scored.astype(np.float64), size=size, mode="nearest")
    return (num / np.maximum(den, 1e-6)).astype(np.float32)


def local_error_scale_loo(abs_resid: np.ndarray, n_lead: int, n_px: int,
                          scored: np.ndarray) -> np.ndarray:
    """Leave-CENTRE-out version of `local_error_scale`: the neighbourhood
    average excludes the element's own |residual|.

    `local_error_scale` includes the target element itself (weight 1/N_eff in
    the box average), which lets a "smooth predictor" partly see its own
    answer -- not something a real conditional-scale model has access to for
    the very element it is scoring. This is the honest ceiling for what a
    smooth function of NEIGHBOURING errors (not this element's own) could
    achieve; still an oracle (neighbours' TRUE residuals), just without the
    self-leak.
    """
    from scipy.ndimage import uniform_filter

    N = n_lead * n_px * n_px
    size = (1, n_lead, n_px, n_px)
    num_field = (abs_resid * scored).astype(np.float64)
    den_field = scored.astype(np.float64)
    num_sum = uniform_filter(num_field, size=size, mode="nearest") * N
    den_sum = uniform_filter(den_field, size=size, mode="nearest") * N
    num_loo = num_sum - num_field
    den_loo = den_sum - den_field
    return (num_loo / np.maximum(den_loo, 1e-6)).astype(np.float32)


def oracle_level_and_score(sigma_hat: np.ndarray, r: np.ndarray, target: np.ndarray,
                           A: np.ndarray, ctr: np.ndarray, scored: np.ndarray,
                           lam_scan=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4)
                           ) -> Dict[str, float]:
    """Calibrate a single per-channel level c on `sigma_hat` (oracle: uses the
    full scored set, matching C6-S's `eval_sigma` oracle-level convention),
    convert to the Gaussian score-optimal half-width, and report the best sps
    over a lambda re-scan on top of that level (also oracle)."""
    Cc = sigma_hat.shape[-1]
    lev = np.ones(Cc, dtype=np.float64)
    for c in range(Cc):
        m = scored[..., c]
        lev[c] = np.sqrt(np.mean((r[..., c][m] / np.maximum(sigma_hat[..., c][m], 1e-8)) ** 2))
    sig = sigma_hat * lev[None, None, None, None, :]
    best = None
    best_lam = None
    for lam in lam_scan:
        hw = C.half_from_sigma(sig * lam).astype(np.float32)
        res = C.sps_from_bounds(A, target, ctr - hw, ctr + hw)
        if best is None or res["sps"] > best["sps"]:
            best, best_lam = res, lam
    best["level_u"], best["level_v"] = float(lev[0]), float(lev[1]) if Cc > 1 else float("nan")
    best["best_lambda"] = best_lam
    return best


# --------------------------------------------------------------------------- #
# Causal features for the learned net (Stage 1) -- one field per feature,
# broadcastable to (N, T, H, W); everything here is available BEFORE the
# target of the CURRENT window is revealed.
# --------------------------------------------------------------------------- #
def _grad_mag(f: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(f)
    gy = np.zeros_like(f)
    gx[..., 1:-1] = 0.5 * (f[..., 2:] - f[..., :-2])
    gx[..., 0] = f[..., 1] - f[..., 0]
    gx[..., -1] = f[..., -1] - f[..., -2]
    gy[..., 1:-1, :] = 0.5 * (f[..., 2:, :] - f[..., :-2, :])
    gy[..., 0, :] = f[..., 1, :] - f[..., 0, :]
    gy[..., -1, :] = f[..., -1, :] - f[..., -2, :]
    return np.sqrt(gx * gx + gy * gy)


def causal_feature_maps(x: np.ndarray, pred: np.ndarray, sig_pipe: np.ndarray) -> np.ndarray:
    """(N, T, H, W, F) causal feature maps, one channel at a time.

    x, pred, sig_pipe: (N, T, H, W) physical, single channel (sig_pipe already
    lead-pooled, broadcast over T). Feature list mirrors C6-S's F3 (validated
    there to carry the causal predictability ceiling for this residual):
      0 log sigma_pipe            5 log |pred - last frame| (per lead)
      1 log input temporal std    6 lead index / 19 (linear)
      2 log |last - window mean|  7 log local spatial std (3x3, of input std)
      3 log |grad| of last frame  8 log |grad| of mean prediction
      4 log pred temporal std
    """
    EPS = 1e-5
    N, T, H, W = x.shape
    last = x[:, -1]
    xm = x.mean(1)
    sx = x.std(1)
    F = np.empty((N, T, H, W, 9), dtype=np.float32)
    F[..., 0] = np.log(sig_pipe + EPS)
    F[..., 1] = np.log(sx + EPS)[:, None].repeat(T, axis=1)
    F[..., 2] = np.log(np.abs(last - xm) + EPS)[:, None].repeat(T, axis=1)
    F[..., 3] = np.log(_grad_mag(last) + EPS)[:, None].repeat(T, axis=1)
    F[..., 4] = np.log(pred.std(1) + EPS)[:, None].repeat(T, axis=1)
    dp = np.empty_like(pred)
    dp[:, 0] = pred[:, 0] - last
    dp[:, 1:] = pred[:, 1:] - pred[:, :-1]
    F[..., 5] = np.log(np.abs(dp) + EPS)
    lead = np.arange(T, dtype=np.float32) / (T - 1.0)
    F[..., 6] = lead[None, :, None, None]
    p = np.pad(sx, ((0, 0), (1, 1), (1, 1)), mode="edge")
    s2 = sum(p[:, i:i + H, j:j + W] for i in range(3) for j in range(3)) / 9.0
    var_local = np.maximum(sum((p[:, i:i + H, j:j + W] - s2) ** 2 for i in range(3) for j in range(3)) / 9.0, 0.0)
    F[..., 7] = np.log(np.sqrt(var_local) + EPS)[:, None].repeat(T, axis=1)
    F[..., 8] = np.log(_grad_mag(pred.mean(1)) + EPS)[:, None].repeat(T, axis=1)
    return F


FEATURE_NAMES = ["log_sigma_pipe", "log_input_std", "log_last_minus_mean",
                 "log_grad_last", "log_pred_std", "log_dpred_lead",
                 "lead_frac", "log_local_spatial_std", "log_grad_meanpred",
                 "log_prevwin_err", "log_prevwin_err_ema", "log_prevwin_ratio_to_sigma"]


# --------------------------------------------------------------------------- #
# Part B: previous-window / short-EMA error-map features (causal handle on
# the LOO local-oracle coherence found in Stage 0(c)) -- see
# `scripts/m1_causal_baseline.py` for the no-network version of the same
# construction; these are the net-facing feature-map equivalents.
# --------------------------------------------------------------------------- #
def emap_lead_reduced(ar_c: np.ndarray, scored_c: np.ndarray, n_px: int,
                      lead_slice: slice = slice(None)) -> np.ndarray:
    """(N,H,W) weighted-box-smoothed (n_px x n_px), lead-reduced |residual|."""
    from scipy.ndimage import uniform_filter

    num = (ar_c[:, lead_slice] * scored_c[:, lead_slice]).sum(axis=1)
    den = scored_c[:, lead_slice].sum(axis=1)
    nf = uniform_filter(num.astype(np.float64), size=(1, n_px, n_px), mode="nearest") * (n_px * n_px)
    df = uniform_filter(den.astype(np.float64), size=(1, n_px, n_px), mode="nearest") * (n_px * n_px)
    return (nf / np.maximum(df, 1e-6)).astype(np.float32)


def ema_map_causal(emap_win: np.ndarray, is_first: np.ndarray, rho: float = 0.6) -> np.ndarray:
    """Per-trajectory EMA of `emap_win` (N,H,W), reset at trajectory starts
    (~2-3 window memory at rho=0.6). Includes the current window; shift by
    one (via `shift1_causal`) before using as a feature of the NEXT window."""
    out = np.empty_like(emap_win)
    prev = None
    for i in range(emap_win.shape[0]):
        if is_first[i] or prev is None:
            out[i] = emap_win[i]
        else:
            out[i] = rho * prev + (1.0 - rho) * emap_win[i]
        prev = out[i]
    return out


def shift1_causal(x: np.ndarray, is_first: np.ndarray) -> np.ndarray:
    """x shifted by one window within each trajectory (k=0 filled with its
    own value -- irrelevant wherever the caller also has a first-window
    reset, e.g. `m1_causal_baseline.py`'s multiplier=1 rule; here it's just
    fed to the net as a feature, so the net can learn to ignore it via the
    lead_frac / window-index signal already in the base 9 features)."""
    out = x.copy()
    out[1:] = x[:-1]
    out[is_first] = x[is_first]
    return out


def extra_causal_features(ar_c: np.ndarray, scored_c: np.ndarray, sigma_pipe_c: np.ndarray,
                          is_first: np.ndarray, n_px: int = 5) -> np.ndarray:
    """3 extra causal feature maps (N,T,H,W), broadcast over T like the
    window-level features in `causal_feature_maps`:
      0 log prev-window error (all-lead, box-smoothed)
      1 log short-EMA (~2-3 window memory) of the same map
      2 log ratio of (0) to the pipeline's own running scale for this pixel
    """
    EPS = 1e-5
    N, T, H, W = ar_c.shape
    e_all = emap_lead_reduced(ar_c, scored_c, n_px)
    e_prev = shift1_causal(e_all, is_first)
    e_ema = shift1_causal(ema_map_causal(e_all, is_first, rho=0.6), is_first)
    ratio = e_prev / np.maximum(sigma_pipe_c[:, 0], EPS)
    out = np.empty((N, T, H, W, 3), dtype=np.float32)
    out[..., 0] = np.log(e_prev + EPS)[:, None].repeat(T, axis=1)
    out[..., 1] = np.log(e_ema + EPS)[:, None].repeat(T, axis=1)
    out[..., 2] = np.log(ratio + EPS)[:, None].repeat(T, axis=1)
    return out


def standardize(F: np.ndarray, mu=None, sd=None):
    if mu is None:
        mu = F.reshape(-1, F.shape[-1]).mean(0)
        sd = F.reshape(-1, F.shape[-1]).std(0) + 1e-6
    return (F - mu) / sd, mu, sd
