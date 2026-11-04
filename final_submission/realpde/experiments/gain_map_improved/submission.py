"""gain_map_improved -- RealPDE Track 2 LTTTA submission.

Scaffolded by scripts/create_experiment.py (forked from experiments/gain_map). This
file is the experiment: CONFIG below is the whole configuration, and it is what
ships in the zip.

Runs in two places unchanged:
  * in place, `python src/local_eval.py --submission experiments/gain_map_improved \
        --data starting_kit/example_data` (kit modules resolved from src/,
    checkpoint resolved from data/competition/baseline_checkpoints/);
  * from the packaged zip, where scripts/create_submission.py has placed the
    imported kit modules and `model.pth` next to this file.

Contract (docs/track2.md): every tensor is (1, 20, 32, 64, 3) in normalized
space, `ttt_step` and `reset_ttt_state` are timed, and `info["lower"]` /
`info["upper"]` are all-or-nothing for the entire run.

Idea: gain_map's three online per-pixel maps (fluctuation gain, bias, spread),
plus two things the stream itself can tell us -- the wake's propagation speed
and the flow parameters -- used where each provably helps:

  * propagation speed -> the *shape* of the fluctuation. The vortex street
    convects downstream at Uc px/frame, read off every input window as the
    shift that best maps frame t onto frame t+tau (grid scan, running average
    over the trajectory; 0.80 px/frame = 0.74 U_inf at Re 11k / AoA 0, 1.36 =
    0.60 U_inf at Re 24k / AoA 15). Taylor's hypothesis makes the last input
    frame, shifted by Uc (k+1), a forecast of lead k with the right *phase*:
    its correlation with the truth is 0.94 at lead 0 against 0.75 for the FNO,
    and it decays with lead time -- slowly at low Re (0.4 at lead 19), within
    two frames at high Re. It exists only where the source column lies
    downstream of the airfoil / laser shadow. The fluctuation shape is
    a * (F(x) - mean_t F) + b * advected with (a, b) per lead and channel the
    ridge least-squares fit to the revealed targets (per-element sufficient
    statistics, pooled separately over pixels with and without an advected
    value), then rescaled per pixel and window to the energy below.
  * energy -> gain_map's rule, untouched: per (pixel, channel) the output's
    temporal variance this window is var_t(F') times the running ratio of the
    input's energy to the forecast's, to the power 0.75. Measured on the two test
    trajectories, the FNO's window-to-window energy modulation tracks the true
    TKE better than anything the propagation speed offers (the advected
    pattern's energy adds nothing at high Re), and the per-window TKE varies by
    ~50 %, so the energy is not the blend's to set: the least-squares blend is
    rel_l2-optimal but zeroes the fluctuation where nothing predicts the phase,
    which is exactly where TKE still needs the energy. Splitting shape and
    energy keeps the TKE map identical to gain_map's and improves the phase
    where it can.
  * parameters -> the starting point of the running statistics. (Re, AoA) are
    read off the first input window (src/parameter_estimation.py) and select,
    by a Gaussian kernel in (AoA, log Re), the training files whose converged
    maps (mean innovation, residual variance, input and forecast energies;
    scripts/fit_gain_prior.py) seed the bias, spread and gain with the weight
    of a few windows, so the first windows of every trajectory -- where
    gain_map's statistics start from nothing -- are no longer the worst.
  * bias / spread: gain_map's running mean and pooled variance of the
    innovation; SPS interval at the closed-form score-optimal width.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch

CONFIG = {
    "base": "fno",            # cno | fno | transolver
    "ckpt": "sim_real_fno.pth",            # ships as model.pth; dev-resolved from data/competition/
    "adapt": False,                # kept for the notes.md summary: no gradient TTT here
    # --- propagation-speed fluctuation source ---
    "advect": True,                # blend in the Taylor-advected last input frame
    "uc_max": 4.0,                 # px/frame: top of the convection-speed scan grid
    "uc_step": 0.1,                # scan resolution (parabolic refinement on top)
    "uc_taus": (2, 3),             # frame lags the scan correlates over
    "uc_fixed": None,              # px/frame: bypass the scan with a constant (diagnostics)
    "uc_warm": 4,                  # scan every window until this many are seen ...
    "uc_every": 4,                 # ... then every uc_every-th window (the speed is a stable running average)
    "blend": "region",             # "region": (a, b) per lead & channel, split valid/invalid | "pixel": per element on top
    "blend_ridge": 1.0,            # ridge toward (a, b) = (1, 0) [gain_map], relative to the local energies; 0 = pure LS
    "blend_pixel_n0": 8.0,         # "pixel" only: pseudo-windows of ridge toward the region solution (fades as 1/n)
    "blend_smooth": True,          # "pixel" only: 3x3 box-smooth the sufficient statistics before solving
    # --- gain / bias / spread ---
    "gain": True,                  # gain_map's rule: per-window energy = var_t(F') * (E_input / E_F)^power, running per pixel
    "gain_power": 0.75,            # gamma = ratio^(power/2); < 1 shrinks the gain toward 1
    "gain_max": 5.0,               # clip gamma to [1/gain_max, gain_max]
    "shape_cap": 2.0,              # blend shape scaled up to this factor to reach the window energy; F' fills the rest
    # --- (Re, AoA)-conditioned prior for the running statistics ---
    "prior": True,                 # start mu / var / energies from prior.npz maps if the file exists
    "prior_windows": 6.0,          # pseudo-windows the prior counts for before the stream takes over
    "prior_sigma_aoa": 2.5,        # deg: kernel width over the training files' AoA
    "prior_sigma_re": 0.15,        # log-Re kernel width (0.15 ~ 16 % in Re, one grid step is ~10 %)
    "bias": True,                  # add the running mean innovation
    "bounds": True,                # SPS intervals from the running innovation variance
    "pool_t": True,                # innovation variance pooled over the 20 frames: True | False | "sep" (pooled x lead profile)
    "alpha_min": 0.05,             # floor on the running-average gain max(1/(n+prior_n), alpha_min)
    "prior_n": 1.0,                # pseudo-count of the initial statistics
    "k_scale": 1.0,                # multiplier on the optimal interval half-width
    "seed": 0,
}

# Official train_real per-channel statistics (mean_std_real.pt). The evaluator
# normalises the input with the *input* stats and the target with the *target*
# stats; SPS measures interval width against SIGMA_GLOBAL in physical units.
_MEAN_IN = (0.15496086, -0.00051399, 0.0)
_STD_IN = (0.09680565, 0.01596068, 1.0)
_STD_TGT = (0.09681041, 0.01596364, 1.0)
_SIGMA_GLOBAL = 0.0563870259

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))

# Packaged: load_baseline.py / rpde_baselines/ sit next to this file. Dev: they
# live in the repo's src/. _HERE is inserted last so the packaged copy wins.
for _p in (os.path.join(_REPO, "src"), _HERE):
    if os.path.isfile(os.path.join(_p, "load_baseline.py")) and _p not in sys.path:
        sys.path.insert(0, _p)

from load_baseline import build_model, load_checkpoint_state  # noqa: E402
from parameter_estimation import estimate_aoa, estimate_re  # noqa: E402


def _resolve_ckpt(submission_dir: str) -> str:
    """model.pth from the archive root, else the dev checkpoint tree."""
    packaged = os.path.join(submission_dir, "model.pth")
    if os.path.isfile(packaged):
        return packaged
    for sub in ("sim_real_ft", "sim_pretrain"):
        dev = os.path.join(_REPO, "data", "competition", "baseline_checkpoints",
                           sub, CONFIG["ckpt"])
        if os.path.isfile(dev):
            return dev
    raise FileNotFoundError(
        "no model.pth in %s and no %s under data/competition/baseline_checkpoints/"
        % (submission_dir, CONFIG["ckpt"]))


def _load_prior(device: str):
    """prior.npz next to this file (scripts/fit_gain_prior.py), as device tensors, or None.

    Per training file: mean innovation (physical, fp16), pooled residual variance and
    the input / forecast energies, plus its (Re, AoA). Converted here to the
    normalised spaces the online statistics live in (target stats for the
    innovation and the forecast energy, input stats for the input energy).
    """
    path = os.path.join(_HERE, "prior.npz")
    if not os.path.isfile(path):
        return None
    import numpy as np
    z = np.load(path)
    std_t = torch.tensor(_STD_TGT[:2]); std_i = torch.tensor(_STD_IN[:2])
    pad = lambda t: torch.cat([t, torch.zeros_like(t[..., :1])], dim=-1)   # p channel
    return {
        "re": torch.tensor(z["re"], dtype=torch.float32, device=device),
        "aoa": torch.tensor(z["aoa"], dtype=torch.float32, device=device),
        "mu": pad(torch.tensor(z["mu"].astype(np.float32)) / std_t).to(device),            # (F, T, H, W, 3)
        "var": pad(torch.tensor(z["var"]) / std_t ** 2).unsqueeze(1).to(device),         # (F, 1, H, W, 3)
        "e_true": pad(torch.tensor(z["e_true"]) / std_i ** 2).unsqueeze(1).to(device),
        "e_pred": pad(torch.tensor(z["e_pred"]) / std_t ** 2).unsqueeze(1).to(device),
    }


# --------------------------------------------------------------------------- #
# SPS-optimal interval width for a Gaussian error.
# --------------------------------------------------------------------------- #
def _optimal_k_table(n: int = 512, a_min: float = 1e-3, a_max: float = 1e2):
    """k*(a) maximising exp(-a k) * erf(k / sqrt 2), a = 2 sigma_phys / SIGMA_GLOBAL.

    For an element with Gaussian error of std sigma and a symmetric interval of
    half-width k sigma, the expected SPS element score is
    P(inside) * exp(-width / SIGMA_GLOBAL) = erf(k/sqrt2) * exp(-a k). The
    maximiser solves a erf(k/sqrt2) = sqrt(2/pi) exp(-k^2/2); solved once by
    bisection on a log grid of `a` (untimed, at construction).
    """
    log_a = torch.linspace(math.log(a_min), math.log(a_max), n, dtype=torch.float64)
    ks = []
    for a in log_a.exp().tolist():
        lo, hi = 0.0, 10.0
        for _ in range(60):
            k = 0.5 * (lo + hi)
            g = a * math.erf(k / math.sqrt(2.0)) - math.sqrt(2.0 / math.pi) * math.exp(-0.5 * k * k)
            if g > 0:
                hi = k
            else:
                lo = k
        ks.append(0.5 * (lo + hi))
    return log_a.float(), torch.tensor(ks, dtype=torch.float32)


def _ke(x: torch.Tensor) -> torch.Tensor:
    """Per-pixel, per-channel temporal variance of a (1, T, H, W, C) window: (1, 1, H, W, C)."""
    return x.var(dim=1, keepdim=True, unbiased=False)


# --------------------------------------------------------------------------- #
# Streamwise shift with linear interpolation (Taylor's hypothesis).
# --------------------------------------------------------------------------- #
class _Shifter:
    """Shift fields along W by a fractional number of columns, source-validity aware.

    A pixel at column c shifted by d takes its value from column c - d (linear
    interpolation). The value is valid when the whole path c - d .. c lies in
    the domain and downstream of the last masked pixel (airfoil body / laser
    shadow) upstream of c on that row, so nothing is advected through the
    airfoil or from outside the field of view.
    """

    def __init__(self, w: int, device: str):
        self.w = w
        self.cols = torch.arange(w, device=device, dtype=torch.float32)

    def last_obstacle(self, mask: torch.Tensor) -> torch.Tensor:
        """(H, W) column index of the last masked pixel at or upstream of each column, -1 if none."""
        idx = torch.where(mask, self.cols.expand_as(mask), torch.full_like(self.cols, -1.0).expand_as(mask))
        return torch.cummax(idx, dim=-1).values

    def __call__(self, f: torch.Tensor, d: float, obstacle: torch.Tensor):
        """f (..., H, W, C) shifted by d columns; returns (shifted, valid (H, W))."""
        src = self.cols - d
        i0 = torch.floor(src)
        wgt = (src - i0).view(-1, 1)
        i0 = i0.long()
        valid = (i0.view(1, -1) >= obstacle.long() + 1) & (i0 >= 0).view(1, -1)
        i0c = i0.clamp(0, self.w - 1)
        i1c = (i0 + 1).clamp(0, self.w - 1)
        out = (1.0 - wgt) * f[..., i0c, :] + wgt * f[..., i1c, :]
        return out, valid


class GainMapImprovedModel:
    """F(x) blended with the advected input, then online gain, bias and spread."""

    def __init__(self, base: torch.nn.Module, device: str, cfg: Dict[str, Any]):
        self.base = base.to(device).eval()
        self.device = device
        self.cfg = cfg
        self.log_a, self.k_tab = (t.to(device) for t in _optimal_k_table())
        self.std_tgt = torch.tensor(_STD_TGT, device=device)
        self.mean_in = torch.tensor(_MEAN_IN, device=device)
        self.std_in = torch.tensor(_STD_IN, device=device)
        self.shift = _Shifter(64, device)
        self.uc_grid = torch.arange(0.0, float(cfg["uc_max"]) + 1e-6, float(cfg["uc_step"]), device=device)
        self.prior = _load_prior(device) if cfg["prior"] else None
        self.reset()

    def reset(self) -> None:
        self.n = 0                  # innovations seen
        self.n_e = 0                # energy pairs seen
        self.n_uc = 0               # convection-speed estimates seen
        self.n_win = 0              # windows seen in this trajectory
        self.n_b = 0                # blend pairs seen
        self.uc = None              # convection speed, px/frame (0-d tensor on the device)
        self.params: Optional[Tuple[float, float]] = None   # (Re, AoA) read off the first window
        self.prev_out = self.prev_fl = self.prev_adv = self.prev_mf = self.prev_mask = None
        self.mu = self.var = self.prof = self.e_true = self.e_pred = None
        self.S = None               # blend sufficient statistics: dict of (1, T, H, W, C) maps

    def _a(self, n: int) -> float:
        return max(1.0 / (n + float(self.cfg["prior_n"])), float(self.cfg["alpha_min"]))

    # ------------------------------------------------------------------ masks
    def _mask(self, x: torch.Tensor) -> torch.Tensor:
        """(H, W) pixels that are exactly zero (physical u and v) in every frame."""
        phys = x[0, ..., :2] * self.std_in[:2] + self.mean_in[:2]
        return (phys.abs() < 1e-6).all(-1).all(0)

    # ------------------------------------------------------- convection speed
    def _scan_uc(self, fl: torch.Tensor, obstacle: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Shift (px/frame) that best maps frame t onto frame t+tau inside the window.

        Correlation over valid, unmasked pixels of the shifted earlier frames with
        the later frames, summed over the lags in `uc_taus`, maximised on the grid
        and refined by a parabola through the three best points. All candidates are
        gathered in one batched indexing op and the result stays on the device: no
        host sync per candidate (the per-candidate loop this replaces cost ~30 ms per
        step on a GPU, four times the FNO forward).
        """
        f = fl[..., :2]
        grid = self.uc_grid
        u, w = grid.numel(), self.shift.w
        keep = (~mask).view(1, 1, *mask.shape, 1, 1).to(f.dtype)          # (1, 1, H, W, 1, 1)
        obs = obstacle.long().unsqueeze(0)                                  # (1, H, W)
        num = den_a = den_b = torch.zeros(u, device=f.device, dtype=f.dtype)
        for tau in self.cfg["uc_taus"]:
            src = self.shift.cols.view(1, -1) - grid.view(-1, 1) * tau      # (U, W)
            i0 = torch.floor(src)
            wgt = (src - i0).view(1, 1, 1, u, w, 1)
            i0 = i0.long()
            valid = (i0.unsqueeze(1) >= obs + 1) & (i0 >= 0).unsqueeze(1)   # (U, H, W)
            i0c, i1c = i0.clamp(0, w - 1), (i0 + 1).clamp(0, w - 1)
            a = f[:, :-tau]                                                 # (1, T', H, W, 2)
            shifted = (1.0 - wgt) * a[:, :, :, i0c, :] + wgt * a[:, :, :, i1c, :]   # (1, T', H, U, W, 2)
            b = f[:, tau:].unsqueeze(3)                                     # (1, T', H, 1, W, 2)
            k = valid.permute(1, 0, 2).reshape(1, 1, mask.shape[0], u, w, 1).to(f.dtype) * keep.view(1, 1, mask.shape[0], 1, w, 1)
            num = num + (shifted * b * k).sum(dim=(0, 1, 2, 4, 5))
            den_a = den_a + (shifted * shifted * k).sum(dim=(0, 1, 2, 4, 5))
            den_b = den_b + (b * b * k).sum(dim=(0, 1, 2, 4, 5))
        s = num / (den_a * den_b + 1e-30).sqrt()
        k = s.argmax()
        km, kp = (k - 1).clamp(0, u - 1), (k + 1).clamp(0, u - 1)
        den = s[km] - 2.0 * s[k] + s[kp]
        interior = (k > 0) & (k < u - 1) & (den < 0)
        off = torch.where(interior, 0.5 * (s[km] - s[kp]) / torch.where(den < 0, den, -torch.ones_like(den)),
                          torch.zeros_like(den))
        return grid[k] + off * float(self.cfg["uc_step"])

    def _update_uc(self, uc: float) -> None:
        if self.uc is None:
            self.uc = uc
            return
        self.n_uc += 1
        self.uc = self.uc + self._a(self.n_uc) * (uc - self.uc)

    def _advect(self, fl_last: torch.Tensor, obstacle: torch.Tensor, mask: torch.Tensor, t: int):
        """Last input frame's fluctuation shifted by uc*(k+1) for k < t, in one gather:
        (1, T, H, W, C), and the validity (1, T, H, W, 1)."""
        w = self.shift.w
        d = self.uc * torch.arange(1, t + 1, device=fl_last.device, dtype=fl_last.dtype)   # (T,)
        src = self.shift.cols.view(1, -1) - d.view(-1, 1)                                   # (T, W)
        i0 = torch.floor(src)
        wgt = (src - i0).view(1, 1, t, w, 1)
        i0 = i0.long()
        valid = (i0.unsqueeze(1) >= obstacle.long().unsqueeze(0) + 1) & (i0 >= 0).unsqueeze(1) & (~mask).unsqueeze(0)  # (T, H, W)
        i0c, i1c = i0.clamp(0, w - 1), (i0 + 1).clamp(0, w - 1)
        adv = (1.0 - wgt) * fl_last[:, :, i0c, :] + wgt * fl_last[:, :, i1c, :]              # (1, H, T, W, C)
        adv = adv.permute(0, 2, 1, 3, 4) * valid.view(1, t, *mask.shape, 1).to(fl_last.dtype)
        adv[..., 2] = 0.0
        return adv, valid.view(1, t, *mask.shape, 1)

    # ------------------------------------------------------------------ blend
    def _update_blend(self, fl: torch.Tensor, adv: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> None:
        keep = (~mask).view(1, 1, *mask.shape, 1).to(fl.dtype)
        cur = {"ff": fl * fl, "aa": adv * adv, "fa": fl * adv, "fy": fl * y, "ay": adv * y}
        cur = {k: v * keep for k, v in cur.items()}
        if self.S is None:
            self.S = cur
            return
        self.n_b += 1
        a = self._a(self.n_b)
        self.S = {k: self.S[k] + a * (cur[k] - self.S[k]) for k in cur}

    @staticmethod
    def _smooth(m: torch.Tensor) -> torch.Tensor:
        """3x3 box average over (H, W) of a (1, T, H, W, C) map, edge-aware."""
        t, h, w, c = m.shape[1:]
        flat = m.permute(0, 1, 4, 2, 3).reshape(t * c, 1, h, w)
        out = torch.nn.functional.avg_pool2d(flat, 3, stride=1, padding=1, count_include_pad=False)
        return out.reshape(1, t, c, h, w).permute(0, 1, 3, 4, 2)

    @staticmethod
    def _solve(ff, aa, fa, fy, ay, lam: float, with_b: bool, a0=1.0, b0=0.0):
        """Ridge least squares y ~ a f + b adv toward (a0, b0), weight lam x own energies.

        Cauchy-Schwarz gives det >= ff aa (1 - (1+lam)^-2) > 0 whenever lam > 0; the
        clamp guards float32 and lam = 0.
        """
        p11 = ff * (1.0 + lam)
        r1 = fy + lam * ff * a0
        if not with_b:
            return r1 / (p11 + 1e-30), torch.zeros_like(r1)
        p22 = aa * (1.0 + lam)
        r2 = ay + lam * aa * b0
        det = (p11 * p22 - fa * fa).clamp_min(1e-3 * p11 * p22 + 1e-30)
        return (p22 * r1 - fa * r2) / det, (p11 * r2 - fa * r1) / det

    def _coefficients(self, valid: torch.Tensor, mask: torch.Tensor):
        """(a, b) maps of shape (1, T, H, W, C) from the running statistics.

        Ridge least squares toward gain_map's (a, b) = (1, 0), weight `blend_ridge`
        relative to the pooled energies, so that where the revealed targets say the
        blend explains nothing the FNO fluctuation is kept rather than zeroed:
            [ff (1+l), fa; fa, aa (1+l)] [a; b] = [fy + l ff; ay].
        """
        T = valid.shape[1]
        shape = (1, T, *mask.shape, 3)
        if self.S is None:
            return torch.ones(shape, device=self.device), torch.zeros(shape, device=self.device)
        lam = float(self.cfg["blend_ridge"])
        keep = (~mask).view(1, 1, *mask.shape, 1)
        v = (valid & keep).to(self.S["ff"].dtype)
        i = (~valid & keep).to(self.S["ff"].dtype)
        pooled = {}
        for region, w, with_b in (("v", v, True), ("i", i, False)):
            sums = [(self.S[k] * w).sum(dim=(2, 3), keepdim=True) for k in ("ff", "aa", "fa", "fy", "ay")]
            pooled[region] = self._solve(*sums, lam=lam, with_b=with_b)
        a = torch.where(valid, pooled["v"][0], pooled["i"][0])
        b = torch.where(valid, pooled["v"][1], torch.zeros_like(pooled["v"][1]))
        if self.cfg["blend"] == "pixel":
            # per element, ridge toward the region solution with n0 pseudo-windows
            S = self.S
            if self.cfg["blend_smooth"]:
                S = {k: self._smooth(v_) for k, v_ in S.items()}
            lam_p = float(self.cfg["blend_pixel_n0"]) / float(self.n_b + 1)
            ap, bp = self._solve(*(S[k] for k in ("ff", "aa", "fa", "fy", "ay")), lam=lam_p, with_b=True, a0=a, b0=b)
            a = ap
            b = torch.where(valid, bp, torch.zeros_like(bp))
        a[..., 2] = 1.0
        b[..., 2] = 0.0
        return a, b

    # ------------------------------------------------------------------- gain
    def _update_energy(self, e_true: torch.Tensor, e_pred: torch.Tensor) -> None:
        if self.e_true is None:
            self.e_true, self.e_pred = e_true.clone(), e_pred.clone()
            return
        self.n_e += 1
        a = self._a(self.n_e)
        self.e_true = self.e_true + a * (e_true - self.e_true)
        self.e_pred = self.e_pred + a * (e_pred - self.e_pred)

    def _gamma(self) -> torch.Tensor:
        g_max = float(self.cfg["gain_max"])
        ratio = (self.e_true + 1e-8) / (self.e_pred + 1e-8)
        return ratio.pow(0.5 * float(self.cfg["gain_power"])).clamp(1.0 / g_max, g_max)

    def _shape(self, zc, fl_f, e_z, e_f, c_zf, e_tgt):
        """Fluctuation with per-(pixel, channel) temporal energy e_tgt from the blend's shape zc.

        zc is scaled by up to `shape_cap`; if that does not reach e_tgt (the blend has
        (nearly) zeroed the fluctuation because nothing predicts the phase there) the
        remainder comes from the FNO's own fluctuation fl_f, whose energy is the right
        one: s solves s^2 e_f + 2 s cov(gamma zc, fl_f) = e_tgt - gamma^2 e_z. Where the
        blend is (1, 0) this reduces exactly to gain_map's gamma * fl_f.
        """
        gamma = ((e_tgt + 1e-10) / (e_z + 1e-10)).sqrt().clamp(0.0, float(self.cfg["shape_cap"]))
        d_e = (e_tgt - gamma * gamma * e_z).clamp_min(0.0)
        c = gamma * c_zf
        s = ((c * c + e_f * d_e).clamp_min(0.0).sqrt() - c) / (e_f + 1e-10)
        return gamma * zc + s.clamp(0.0, float(self.cfg["gain_max"])) * fl_f

    # ------------------------------------------------------------ bias/spread
    def _update_error(self, err: torch.Tensor) -> None:
        pool = self.cfg["pool_t"]
        if self.mu is None:
            self.mu = torch.zeros_like(err)
            self.var = _ke(err).clone() if pool else _ke(err).expand_as(err).clone()
            self.prof = torch.ones_like(err.mean(dim=(2, 3), keepdim=True))
        self.n += 1
        a = self._a(self.n)
        d = err - self.mu
        self.mu = self.mu + a * d
        d2 = d * d
        self.var = (1.0 - a) * (self.var + a * (d2.mean(dim=1, keepdim=True) if pool else d2))
        if pool == "sep":  # lead-time profile pooled over the field, (1, T, 1, 1, C)
            self.prof = self.prof + a * (d2.mean(dim=(2, 3), keepdim=True) - self.prof)

    def _var(self) -> torch.Tensor:
        if self.cfg["pool_t"] == "sep":
            return self.var * self.prof / self.prof.mean(dim=1, keepdim=True)
        return self.var

    # ------------------------------------------------------------------ prior
    def _apply_prior(self, x: torch.Tensor) -> None:
        """Start the running statistics from the kernel-weighted prior maps of the
        training files nearest in (AoA, log Re) to what the first window says."""
        state = x[0].cpu().numpy()
        re, aoa = float(estimate_re(state)), float(estimate_aoa(state))
        self.params = (re, aoa)
        p = self.prior
        d_aoa = (p["aoa"] - aoa) / float(self.cfg["prior_sigma_aoa"])
        d_re = (p["re"].clamp_min(1.0).log() - math.log(max(re, 1.0))) / float(self.cfg["prior_sigma_re"])
        w = torch.exp(-0.5 * (d_aoa * d_aoa + d_re * d_re))
        w = w / w.sum().clamp_min(1e-12)
        mix = lambda m: torch.einsum("f,f...->...", w, m).unsqueeze(0)
        self.mu, self.var = mix(p["mu"]), mix(p["var"])
        self.prof = torch.ones_like(self.mu.mean(dim=(2, 3), keepdim=True))
        self.e_true, self.e_pred = mix(p["e_true"]), mix(p["e_pred"])
        n0 = float(self.cfg["prior_windows"])
        self.n = self.n_e = n0

    # ------------------------------------------------------------------- step
    def step(self, x: torch.Tensor, prev_target: Optional[torch.Tensor]):
        loss = None
        if prev_target is not None and self.prev_out is not None:
            err = prev_target - self.prev_out
            loss = float((err ** 2).mean())
            self._update_error(err)
            if self.cfg["advect"]:
                self._update_blend(self.prev_fl, self.prev_adv, prev_target - self.prev_mf, self.prev_mask)

        if self.prior is not None and self.params is None:
            self._apply_prior(x)

        T = x.shape[1]
        mask = self._mask(x)
        raw = self.base(x)
        mf = raw.mean(dim=1, keepdim=True)
        fl_f = raw - mf

        if self.cfg["advect"]:
            obstacle = self.shift.last_obstacle(mask)
            mx = x.mean(dim=1, keepdim=True)
            fl_x = x - mx
            if self.cfg["uc_fixed"] is not None:
                self.uc = float(self.cfg["uc_fixed"])
            elif (self.uc is None or self.n_win < int(self.cfg["uc_warm"])
                  or self.n_win % int(self.cfg["uc_every"]) == 0):
                self._update_uc(self._scan_uc(fl_x, obstacle, mask))
            adv, valid = self._advect(fl_x[:, -1], obstacle, mask, T)
            a, b = self._coefficients(valid, mask)
            z = a * fl_f + b * adv
        else:
            adv = None
            z = fl_f

        out = raw
        if self.cfg["gain"]:
            # gain_map's energy rule, untouched: the output's per-(pixel, channel) temporal
            # energy this window is var_t(F') * gamma^2 with gamma from the running ratio of
            # the input window's energy to the forecast's (the current window included).
            e_f = _ke(fl_f)
            self._update_energy(_ke(x), e_f)
            gamma = self._gamma()
            if self.cfg["advect"] and self.S is not None:
                # the blend supplies the *shape*: rescale it to that energy, per window
                mz = z.mean(dim=1, keepdim=True)
                zc = z - mz
                out = mf + mz + self._shape(zc, fl_f, _ke(zc), e_f, (zc * fl_f).mean(dim=1, keepdim=True), gamma * gamma * e_f)
            else:
                out = mf + gamma * fl_f
        elif self.cfg["advect"]:
            out = mf + z
        if self.cfg["bias"] and self.mu is not None:
            out = out + self.mu

        self.prev_out = out.detach()
        self.prev_fl, self.prev_adv, self.prev_mf, self.prev_mask = fl_f.detach(), adv, mf.detach(), mask
        self.n_win += 1
        return out.detach(), loss

    def bounds(self, mean: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        var = self._var() if self.var is not None else _ke(x)   # first window: input variance
        std = var.expand_as(mean).sqrt()
        a = 2.0 * std * self.std_tgt / _SIGMA_GLOBAL
        pos = (a.clamp_min(1e-12).log() - self.log_a[0]) / (self.log_a[1] - self.log_a[0])
        k = self.k_tab[pos.round().long().clamp(0, len(self.k_tab) - 1)]
        half = float(self.cfg["k_scale"]) * k * std
        return mean - half, mean + half


class ExperimentTTT:
    def __init__(self, base: torch.nn.Module, device: str, cfg: Dict[str, Any]):
        self.model = GainMapImprovedModel(base, device, cfg)
        self.device = device
        self.cfg = cfg

    def reset_ttt_state(self) -> None:
        self.model.reset()

    @torch.no_grad()
    def ttt_step(self, input_norm, prev_target_norm=None):
        x = torch.as_tensor(input_norm).to(self.device)
        y = torch.as_tensor(prev_target_norm).to(self.device) if prev_target_norm is not None else None
        out, loss = self.model.step(x, y)
        info: Dict[str, Any] = {"adapt_loss": loss}
        if self.cfg["bounds"]:
            info["lower"], info["upper"] = self.model.bounds(out, x)
        return out, info


def get_ttt_model(submission_dir: str, device: str):
    """Entry point. Called once before the stream; not timed."""
    torch.manual_seed(int(CONFIG["seed"]))
    base = build_model(CONFIG["base"], device=device)
    state, _meta = load_checkpoint_state(_resolve_ckpt(submission_dir))
    base.load_state_dict(state)
    return ExperimentTTT(base, device, CONFIG)
