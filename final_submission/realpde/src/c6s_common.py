#!/usr/bin/env python3
"""C6-S shared machinery: exact SPS re-scoring of arbitrary interval widths,
recovery of the shipped pipeline's per-element sigma from its dumped bounds,
causal feature construction, and the Gaussian score-optimal half-width map.

Read-only w.r.t. everything else in the repo. Everything here operates on the
`.npz` dumps written by `scripts/lco_run_dump.py`
(pred/target/lower/upper/sim_ids/time_ids/is_first, all in PHYSICAL units).

Notation
--------
  SPS/100 = mean_{scored elements} [ Abar(window) * exp(-nil) * 1{inside} ]
  nil     = (upper - lower) / SIGMA_GLOBAL
  scored  = (target != 0)
  Abar(w) = 0.5(1-pm_dm) + 0.3(1-pm_tke) + 0.2(1-pm_mvpe),  pm = e/(0.5+e)

With `pred` (and hence Abar) frozen, SPS is a pure functional of the bounds,
which is what every experiment in this family varies.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "starting_kit") not in sys.path:
    sys.path.insert(0, str(REPO / "starting_kit"))

import scoring as official  # noqa: E402

SIGMA_GLOBAL = official.SIGMA_GLOBAL


# --------------------------------------------------------------------------- #
# Gaussian score-optimal half-width (the pipeline's own rule, re-derived).
# --------------------------------------------------------------------------- #
def optimal_k(a: np.ndarray) -> np.ndarray:
    """k*(a) = argmax_k erf(k/sqrt2) * exp(-a k),  a = 2 sigma_phys / SIGMA_GLOBAL.

    Element score for a Gaussian residual of std sigma and symmetric half-width
    k*sigma is P(inside)*exp(-2 k sigma / SIGMA_GLOBAL) = erf(k/sqrt2) e^{-a k}.
    d/dk = 0 gives  a erf(k/sqrt2) = sqrt(2/pi) exp(-k^2/2).  Bisection.
    """
    a = np.asarray(a, dtype=np.float64)
    lo = np.zeros_like(a)
    hi = np.full_like(a, 12.0)
    for _ in range(60):
        k = 0.5 * (lo + hi)
        g = a * erf(k / math.sqrt(2.0)) - math.sqrt(2.0 / math.pi) * np.exp(-0.5 * k * k)
        hi = np.where(g > 0, k, hi)
        lo = np.where(g > 0, lo, k)
    return 0.5 * (lo + hi)


def erf(x):
    from scipy.special import erf as _e  # scipy is available in the dev venv
    return _e(x)


_SIG_GRID = None
_HALF_GRID = None


def _grids(n: int = 4096):
    """Tabulate sigma_phys -> optimal half-width (physical), for inversion."""
    global _SIG_GRID, _HALF_GRID
    if _SIG_GRID is None:
        s = np.exp(np.linspace(math.log(1e-6), math.log(10.0), n))
        k = optimal_k(2.0 * s / SIGMA_GLOBAL)
        _SIG_GRID, _HALF_GRID = s, k * s
    return _SIG_GRID, _HALF_GRID


def half_from_sigma(sigma: np.ndarray) -> np.ndarray:
    """Score-optimal physical half-width for a physical residual std.

    Interpolated on the same log-log grid as the inverse (the direct bisection
    is exact but costs 60 erf evaluations per element, which is unaffordable at
    66 M elements x tens of re-scorings).  Grid error < 1e-5 relative.
    """
    sg, hg = _grids()
    s = np.clip(np.asarray(sigma, dtype=np.float32), sg[0], sg[-1])
    return np.exp(np.interp(np.log(s), np.log(sg), np.log(hg))).astype(np.float32)


def sigma_from_half(half: np.ndarray) -> np.ndarray:
    """Invert `half_from_sigma` (monotone increasing) by log-log interpolation."""
    sg, hg = _grids()
    h = np.clip(np.asarray(half, dtype=np.float64), hg[0], hg[-1])
    return np.exp(np.interp(np.log(h), np.log(hg), np.log(sg)))


# --------------------------------------------------------------------------- #
# Dumps.
# --------------------------------------------------------------------------- #
def load_dump(path, channels: int = 2, mmap: bool = True) -> Dict[str, np.ndarray]:
    """Load an lco_run_dump .npz.  Compressed npz cannot be mmapped, so this
    materialises the arrays; `channels=2` keeps only the measured u,v."""
    z = np.load(path, allow_pickle=False)
    c = channels
    out = {
        "pred": np.ascontiguousarray(z["pred"][..., :c]),
        "target": np.ascontiguousarray(z["target"][..., :c]),
        "lower": np.ascontiguousarray(z["lower"][..., :c]),
        "upper": np.ascontiguousarray(z["upper"][..., :c]),
        "sim_ids": np.array([str(v) for v in z["sim_ids"]]),
        "time_ids": np.asarray(z["time_ids"]).astype(np.int64),
        "is_first": np.asarray(z["is_first"]).astype(bool),
    }
    # full 3-channel pred/target are needed by the official metric helpers
    out["pred3"] = np.ascontiguousarray(z["pred"])
    out["target3"] = np.ascontiguousarray(z["target"])
    return out


def abar(pred3: np.ndarray, target3: np.ndarray, c: int = 2) -> np.ndarray:
    """Per-window accuracy factor Abar, exactly as `scoring.aggregate_sps` forms it."""
    dm = official.rel_l2_per_sample(pred3, target3, c)
    tke = official.tke_rel_l2_per_sample(pred3, target3, c)
    mv = official.mvpe_rel_l2_per_sample(pred3, target3)
    f = lambda e: np.where(np.isfinite(e), 1.0 - e / (0.5 + e), 0.0)
    return 0.5 * f(dm) + 0.3 * f(tke) + 0.2 * f(mv)


def sps_from_bounds(A: np.ndarray, target: np.ndarray,
                    lower: np.ndarray, upper: np.ndarray,
                    chunk: int = 64) -> Dict[str, float]:
    """Exact SPS (0-100) for given per-element bounds, with diagnostics.

    `A` is the per-window Abar from `abar`; `target/lower/upper` are (N,T,H,W,2)
    physical.  Reproduces `scoring.aggregate_sps` to float64 round-off.
    """
    n = target.shape[0]
    tot = 0.0
    n_sc = 0
    cov = 0
    nil_sum = 0.0
    nil_in_sum = 0.0
    exp_in = 0.0
    for i in range(0, n, chunk):
        t = target[i:i + chunk]
        lo = lower[i:i + chunk].astype(np.float32)
        up = upper[i:i + chunk].astype(np.float32)
        a = A[i:i + chunk].reshape((-1,) + (1,) * (t.ndim - 1))
        sc = t != 0.0
        inside = (t >= lo) & (t <= up)
        nil = (up - lo).astype(np.float64) / SIGMA_GLOBAL
        e = a * np.exp(-nil)
        e = np.where(inside, e, 0.0)
        tot += float(np.sum(e, where=sc, dtype=np.float64))
        n_sc += int(np.count_nonzero(sc))
        cov += int(np.count_nonzero(inside & sc))
        nil_sum += float(np.sum(nil, where=sc, dtype=np.float64))
        m = inside & sc
        nil_in_sum += float(np.sum(nil, where=m, dtype=np.float64))
        exp_in += float(np.sum(np.exp(-nil), where=m, dtype=np.float64))
    return {
        "sps": 100.0 * tot / n_sc,
        "coverage": cov / n_sc,
        "mean_nil": nil_sum / n_sc,
        "mean_nil_inside": nil_in_sum / max(cov, 1),
        "I": (tot / n_sc) / float(np.mean(A)) if np.mean(A) > 0 else float("nan"),
        "Abar": float(np.mean(A)),
    }


# --------------------------------------------------------------------------- #
# Trajectory bookkeeping.
# --------------------------------------------------------------------------- #
def traj_index(is_first: np.ndarray) -> np.ndarray:
    """Per-step trajectory id (0-based)."""
    return np.cumsum(is_first.astype(np.int64)) - 1


def window_index(is_first: np.ndarray) -> np.ndarray:
    """Per-step index k of the window within its trajectory."""
    k = np.zeros(len(is_first), dtype=np.int64)
    c = -1
    for i, f in enumerate(is_first):
        c = 0 if f else c + 1
        k[i] = c
    return k


def inputs_from_dump(d: Dict[str, np.ndarray], data_dir: Optional[str] = None) -> np.ndarray:
    """The 20-frame INPUT window of every step, physical, (N,20,32,64,2).

    The protocol's stride equals the horizon (INTERVAL == IN_STEP == 20), so
    input(k) == target(k-1) inside a trajectory.  Only k == 0 needs the file,
    and when `data_dir` is None it is filled with target(0) reversed in time --
    marked by `first_mask` so callers can drop those windows.
    """
    tgt = d["target"]
    k = window_index(d["is_first"])
    x = np.empty_like(tgt)
    x[1:] = tgt[:-1]
    first = k == 0
    if data_dir is not None:
        import h5py
        for i in np.nonzero(first)[0]:
            fn = os.path.join(data_dir, str(d["sim_ids"][i]))
            with h5py.File(fn, "r") as f:
                u = f["u"][0:20, ::2, ::2]
                v = f["v"][0:20, ::2, ::2]
            x[i] = np.stack([u, v], axis=-1).astype(np.float32)
    else:
        x[first] = tgt[first]
    return x
