#!/usr/bin/env python3
"""W2 -- conditional interval widths v2: shared machinery.

New file (family W2). Read-only reuse of `src/c6s_common.py` (exact CPU sps,
sigma<->half tables, dump loading) and `src/m1_common.py` (M1's causal
features, box oracles). Adds:
  * a compact feature CACHE format (one .npz per out-of-sample stream),
  * an exact, chunked TORCH implementation of the sps element term (same
    formula as `scoring.aggregate_sps`, float32, GPU if available) so that
    (alpha, lambda) grids over ~1e8 elements cost seconds, not minutes,
  * the Stage-0 oracle family (box / Gaussian / shared-channel /
    coverage-aware / asymmetric / centre-shift, all LEAVE-CENTRE-OUT),
  * the multiplier convention every oracle and every net is applied with:
        half = lambda * half0 * clip(exp(alpha * logm), clip_lo, clip_hi)
    (M1's shipped convention, `ship_condwidth_fit.py`).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
import c6s_common as C  # noqa: E402
import m1_common as M  # noqa: E402

SIGMA_GLOBAL = C.SIGMA_GLOBAL
EPS = 1e-5
CLIP = (0.25, 4.0)

WIN_FEATS = ["log_sigma_pipe", "log_input_std", "log_last_minus_mean", "log_grad_last",
             "log_pred_std", "log_local_spatial_std", "log_grad_meanpred",
             "log_prevwin_err", "log_prevwin_err_ema", "log_prevwin_ratio_to_sigma",
             # ---- W2 / H4 additions ----
             "log_prev_ema_short", "log_prev_ema_long", "log_prev_cummean", "log_prev_cumstd",
             "log1p_k", "log_ctx_run_std", "log_pred_std_early", "log_pred_std_late",
             "pos_x", "pos_y", "log_dist_unscored",
             "log_gamma", "log_e_f", "log_kx", "log_etrue_over_epred", "log_mu_std", "ctr_g",
             "log_g2b"]
LEAD_FEATS = ["log_dpred_lead", "lead_frac", "log_fluc_lead"]
M1_WIN = WIN_FEATS[:10]          # M1's 12 features = these 10 window-level + 2 lead-level
M1_LEAD = LEAD_FEATS[:2]


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def load_cache(path, feats: bool = True) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in ("r", "scored", "half0", "A", "is_first", "kidx")}
    out["sim_ids"] = np.array([str(v) for v in z["sim_ids"]])
    out["win_names"] = [str(v) for v in z["win_names"]]
    out["lead_names"] = [str(v) for v in z["lead_names"]]
    if feats:
        out["featw"] = z["featw"]
        out["featl"] = z["featl"]
    return out


# --------------------------------------------------------------------------- #
# exact sps (torch)
# --------------------------------------------------------------------------- #
def _dev():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class SpsEval:
    """Holds r / scored / A / half0 on the compute device once; evaluates the
    exact sps for arbitrary (half_lo, half_up) multipliers in chunks.

    r: (N,T,H,W,2) float32 = target - ctr (ctr = mid of the dumped bounds);
    half0: (N,H,W,2) float32 (lead-pooled pipeline half-width) or (N,T,H,W,2).
    scored = (target != 0).  inside := (-half_lo <= r) & (r <= half_up), which
    reproduces `scoring.aggregate_sps` (target >= lower & target <= upper) up
    to float32 rounding of the bound reconstruction (checked per cache at
    extraction time, `[check]` line of scripts/w2_extract.py).
    """

    def __init__(self, r, scored, A, half0, device: Optional[str] = None, chunk: int = 64,
                 resident: bool = True):
        import torch
        self.torch = torch
        self.device = device or _dev()
        self.chunk = chunk
        self.N = r.shape[0]
        dev = self.device if resident else "cpu"
        self.r = torch.as_tensor(r, dtype=torch.float32).to(dev)
        self.sc = torch.as_tensor(scored, dtype=torch.bool).to(dev)
        self.A = torch.as_tensor(A, dtype=torch.float32).to(dev)
        h = torch.as_tensor(half0, dtype=torch.float32)
        self.half0 = (h if h.dim() == 5 else h[:, None]).to(dev)
        self.n_sc = int(self.sc.sum())

    def _b(self, x, i, j):
        return x[i:j].to(self.device, non_blocking=True)

    def to_dev(self, a):
        """Move a numpy/tensor field to the device once (for grids)."""
        return self.torch.as_tensor(a, dtype=self.torch.float32).to(self.device)

    def score(self, logm_up=None, logm_lo=None, alpha: float = 1.0, lam: float = 1.0,
              clip: Tuple[float, float] = CLIP, shift=None, alpha_lo=None,
              half_abs=None) -> Dict[str, float]:
        """logm_*: (N,T,H,W,2) or (N,H,W,2) log-multiplier fields (numpy or tensor),
        None = 0.  shift: optional additive centre shift (same shape as r).
        half_abs: optional ABSOLUTE symmetric half-width field (N,T,H,W,2) --
        used instead of half0 * multiplier (then only lam applies)."""
        torch = self.torch
        tot = 0.0
        cov = 0
        nil_sum = 0.0
        a_lo = alpha if alpha_lo is None else alpha_lo
        for i in range(0, self.N, self.chunk):
            j = min(i + self.chunk, self.N)
            r = self._b(self.r, i, j)
            sc = self._b(self.sc, i, j)
            h0 = self._b(self.half0, i, j)
            if shift is not None:
                r = r - self._b(torch.as_tensor(shift, dtype=torch.float32), i, j)

            def mult(lm, al):
                if lm is None:
                    return h0 * lam
                lm = self._b(torch.as_tensor(lm, dtype=torch.float32), i, j)
                if lm.dim() == 4:
                    lm = lm[:, None]
                return h0 * lam * torch.exp(al * lm).clamp(clip[0], clip[1])
            if half_abs is not None:
                hu = self._b(torch.as_tensor(half_abs, dtype=torch.float32), i, j) * lam
                hl = hu
            else:
                hu = mult(logm_up, alpha)
                hl = hu if (logm_lo is None and alpha_lo is None) else mult(
                    logm_lo if logm_lo is not None else logm_up, a_lo)
            inside = (r >= -hl) & (r <= hu) & sc
            nil = (hu + hl) / SIGMA_GLOBAL
            e = torch.exp(-nil) * self._b(self.A, i, j).view(-1, 1, 1, 1, 1)
            tot += float(torch.where(inside, e, torch.zeros_like(e)).sum(dtype=torch.float64))
            cov += int(inside.sum())
            nil_sum += float(torch.where(sc, nil, torch.zeros_like(nil)).sum(dtype=torch.float64))
        return {"sps": 100.0 * tot / self.n_sc, "coverage": cov / self.n_sc, "mean_nil": nil_sum / self.n_sc}

    def grid(self, logm_up, logm_lo=None, alphas=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
             lambdas=None, clip=CLIP, shift=None, half_abs=None):
        """Best (alpha, lambda) by exact sps; returns (best_dict, table)."""
        if lambdas is None:
            lambdas = [round(x, 3) for x in np.linspace(0.7, 1.3, 13)]
        lu = self.to_dev(logm_up) if logm_up is not None else None
        ll = self.to_dev(logm_lo) if logm_lo is not None else None
        ha = self.to_dev(half_abs) if half_abs is not None else None
        sh = self.to_dev(shift) if shift is not None else None
        best = None
        table = []
        for al in (alphas if ha is None else (1.0,)):
            for lam in lambdas:
                res = self.score(lu, ll, alpha=al, lam=lam, clip=clip, shift=sh, half_abs=ha)
                table.append((al, lam, res["sps"]))
                if best is None or res["sps"] > best["sps"]:
                    best = dict(res, alpha=al, lam=lam)
        del lu, ll, ha, sh
        return best, table


# --------------------------------------------------------------------------- #
# LOO neighbourhood operators (per channel, arrays (N,T,H,W))
# --------------------------------------------------------------------------- #
def box_sum(f: np.ndarray, n_lead: int, n_px: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    size = (1, n_lead, n_px, n_px)
    return uniform_filter(f.astype(np.float64), size=size, mode="nearest") * (n_lead * n_px * n_px)


def loo_box_mean(f: np.ndarray, scored: np.ndarray, n_lead: int, n_px: int) -> np.ndarray:
    """Weighted LOO box mean of f over scored neighbours (f itself excluded)."""
    fs = (f * scored).astype(np.float64)
    ss = scored.astype(np.float64)
    num = box_sum(fs, n_lead, n_px) - fs
    den = box_sum(ss, n_lead, n_px) - ss
    return (np.maximum(num, 0.0) / np.maximum(den, 1e-6)).astype(np.float32)


def loo_gauss_mean(f: np.ndarray, scored: np.ndarray, s_lead: float, s_px: float,
                   truncate: float = 2.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    sig = (0.0, s_lead, s_px, s_px)
    fs = (f * scored).astype(np.float64)
    ss = scored.astype(np.float64)
    num = gaussian_filter(fs, sigma=sig, mode="nearest", truncate=truncate)
    den = gaussian_filter(ss, sigma=sig, mode="nearest", truncate=truncate)
    # centre weight of the (separable, normalised, truncated) kernel
    delta = np.zeros((1, 21, 21, 21))
    delta[0, 10, 10, 10] = 1.0
    w0 = float(gaussian_filter(delta, sigma=sig, mode="constant", truncate=truncate)[0, 10, 10, 10])
    num = num - w0 * fs
    den = den - w0 * ss
    return (np.maximum(num, 0.0) / np.maximum(den, 1e-9)).astype(np.float32)


def coverage_aware_oracle(abs_r: np.ndarray, scored: np.ndarray, half0: np.ndarray,
                          n_lead: int, n_px: int, mults=None) -> np.ndarray:
    """Per-element LOO width that maximises  cov_box(w) * exp(-2 w / SIGMA_GLOBAL)
    over candidate widths w = half0 * m (m on a log grid), where cov_box(w) is the
    fraction of the element's scored NEIGHBOURS with |r| <= w.  Returns
    log(w*/half0) (a log-multiplier at alpha = 1).  half0: (N,1,H,W) or (N,T,H,W).
    """
    if mults is None:
        mults = np.exp(np.linspace(math.log(0.2), math.log(5.0), 33))
    ss = scored.astype(np.float64)
    den = box_sum(ss, n_lead, n_px) - ss
    den = np.maximum(den, 1e-6)
    best_obj = np.full(abs_r.shape, -1.0, dtype=np.float32)
    best_m = np.zeros(abs_r.shape, dtype=np.float32)
    for m in mults:
        w = half0 * np.float32(m)
        ind = ((abs_r <= w) & scored).astype(np.float64)
        cov = (box_sum(ind, n_lead, n_px) - ind) / den
        obj = (cov * np.exp(-2.0 * w / SIGMA_GLOBAL)).astype(np.float32)
        better = obj > best_obj
        best_obj = np.where(better, obj, best_obj)
        best_m = np.where(better, np.float32(m), best_m)
    return np.log(np.maximum(best_m, 1e-3)).astype(np.float32)


def logm_from_scale(oracle_scale: np.ndarray, sp: np.ndarray) -> np.ndarray:
    """M1's supervised target: log(oracle local error scale / pipeline sigma)."""
    t = np.log(oracle_scale / np.maximum(sp, EPS) + EPS).astype(np.float32)
    return np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def sigma_pipe(half0: np.ndarray) -> np.ndarray:
    return C.sigma_from_half(half0).astype(np.float32)


# --------------------------------------------------------------------------- #
# trajectory helpers
# --------------------------------------------------------------------------- #
def traj_split(traj: np.ndarray, val_frac: float, seed: int = 0):
    uniq = sorted(set(traj))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_set = set(uniq[:n_val])
    is_val = np.array([t in val_set for t in traj])
    return ~is_val, is_val


def heldout_files(fold: str) -> set:
    import json
    info = {"fold0": ("experiments/lco_split/split.json", "fold0_aoa20_plus_re_extremes"),
            "fold1": ("experiments/lco_split/split.json", "fold1_aoa15_plus_re_extremes"),
            "foldR": ("experiments/lco_split/split_rehold.json", None)}
    path, key = info[fold]
    d = json.loads((REPO / path).read_text())
    key = key or d["primary_fold"]
    return set(d["folds"][key]["heldout_files"])
