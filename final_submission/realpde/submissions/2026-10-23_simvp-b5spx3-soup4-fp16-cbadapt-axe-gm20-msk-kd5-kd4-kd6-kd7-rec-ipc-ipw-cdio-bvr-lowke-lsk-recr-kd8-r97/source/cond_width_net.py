"""P3 -- vendored M1 conditional-width net + causal feature computation.

Pure torch, NO repo imports (this file ships inside the submission archive
and must work standalone from the extraction). Architecture mirrors
`src/m1_net.py` (`M1WidthNet`, 22209 params) and the 12 causal features
mirror `src/m1_common.py` (`causal_feature_maps` + `extra_causal_features`),
EXCEPT for the coordinator-directed leak fix (experiments/m1_review/notes.md,
also applied in `scripts/ship_condwidth_fit.py`'s offline training): the 3
previous-window features (`log_prevwin_err`, its EMA, the ratio-to-sigma) use
a NEUTRAL "no previous-window information yet" fallback at a trajectory's
first window -- the pipeline's own running scale, ratio 1 -- computed here
identically online and offline so the two paths agree (the acceptance test
this package must pass, submissions/2026-09-28_.../manifest.md S3(c)).

State (`CondWidthState.prev_err`/`ema_err`, per measured channel u,v) is
updated once per `step()` call, EAGERLY (same untimed-relative-to-graphs
philosophy `submission.py`'s own `_update_R` already uses for the D2/H1
bundle): as soon as `prev_target` is revealed, using `self.ctr_out` as it
stood BEFORE this call overwrites it for the new window -- i.e. exactly the
centre that was actually shipped for the window `prev_target` is the truth
of. `CondWidthState.apply` (called from `bounds()`) only READS this state;
`update_channel` (called from `step()`, before `self.ctr_out` is
reassigned) is the only writer. Both run EAGERLY, not inside any CUDA graph
-- see the package manifest's launches/step acceptance number and the
"graph fusion not attempted" risk note for why.

Box filters (`_box`, used both for the previous-window error-map smoothing
and the 3x3 local-spatial-std feature) use edge-replicate padding + a
size-preserving average pool, which is NOT bit-identical to the offline
training pipeline's `scipy.ndimage.uniform_filter(mode="nearest")` (same
edge convention, different implementation/rounding) -- an intentional,
documented simplification: exact bit-parity is not required, only score-level
parity (<= 0.02 sps, the S3(c) acceptance bound), and the difference is
confined to a boundary ring of a few pixels out of 32x64.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_FEAT = 12
EPS = 1e-5
LOG1P_EPS = math.log(1.0 + EPS)


class M1WidthNet(nn.Module):
    """Bit-for-bit the same architecture as `src/m1_net.py:M1WidthNet`."""

    def __init__(self, hidden: int = 32, n_feat: int = N_FEAT):
        super().__init__()
        self.c1 = nn.Conv2d(n_feat, hidden, 3, padding=1)
        self.g1 = nn.GroupNorm(8, hidden)
        self.c2 = nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2)
        self.g2 = nn.GroupNorm(8, hidden)
        self.c3 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.g3 = nn.GroupNorm(8, hidden)
        self.out = nn.Conv2d(hidden, 1, 1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.g1(self.c1(x)))
        h = self.act(self.g2(self.c2(h)))
        h = self.act(self.g3(self.c3(h)))
        return self.out(h)


def _grad_mag(f: torch.Tensor) -> torch.Tensor:
    """(..., H, W) -> (..., H, W); same forward/backward-at-edges, central-
    interior finite difference as `m1_common._grad_mag`."""
    gx = torch.zeros_like(f)
    gy = torch.zeros_like(f)
    gx[..., 1:-1] = 0.5 * (f[..., 2:] - f[..., :-2])
    gx[..., 0] = f[..., 1] - f[..., 0]
    gx[..., -1] = f[..., -1] - f[..., -2]
    gy[..., 1:-1, :] = 0.5 * (f[..., 2:, :] - f[..., :-2, :])
    gy[..., 0, :] = f[..., 1, :] - f[..., 0, :]
    gy[..., -1, :] = f[..., -1, :] - f[..., -2, :]
    return torch.sqrt(gx * gx + gy * gy)


# MSK: "scored" mask on the de-standardised revealed target. False (default) = `pt != 0.0` (bit-exact with the
# parent); True = `|pt| > 1e-6`, the scorer's `t != 0.0` on the PHYSICAL target (a physical 0 comes back as
# +-~4e-9 through the harness normalise / our de-normalise) = the mask the W2 net was trained with. Set once by
# submission.py from CONFIG["exact_mask"] before any capture (python-time branch; +1 kernel when True).
EXACT_MASK = False
SCORED_TOL = 1e-6


def _scored(pt, dtype):
    if EXACT_MASK:
        return (pt.abs() > SCORED_TOL).to(dtype)
    return (pt != 0.0).to(dtype)


def _box(t: torch.Tensor, k: int) -> torch.Tensor:
    """(1, H, W) -> (1, H, W), size-preserving box average, edge-replicate
    padding (see module docstring: an intentional near-match, not a bit-exact
    port, of `scipy.ndimage.uniform_filter(mode="nearest")`)."""
    p = k // 2
    x = F.pad(t.unsqueeze(1), (p, p, p, p), mode="replicate")
    return F.avg_pool2d(x, kernel_size=k, stride=1).squeeze(1)


def causal_base_features(x_phys: torch.Tensor, pred_phys: torch.Tensor,
                         sig_pipe: torch.Tensor) -> torch.Tensor:
    """x_phys, pred_phys, sig_pipe: (1, T, H, W) single channel, physical.
    Returns (1, T, H, W, 9) -- mirrors `m1_common.causal_feature_maps`."""
    _, T, H, W = x_phys.shape
    last = x_phys[:, -1]
    xm = x_phys.mean(dim=1)
    sx = x_phys.std(dim=1, unbiased=True)
    f0 = torch.log(sig_pipe + EPS)
    f1 = torch.log(sx + EPS).unsqueeze(1).expand(-1, T, -1, -1)
    f2 = torch.log((last - xm).abs() + EPS).unsqueeze(1).expand(-1, T, -1, -1)
    f3 = torch.log(_grad_mag(last) + EPS).unsqueeze(1).expand(-1, T, -1, -1)
    pred_std = pred_phys.std(dim=1, unbiased=True)
    f4 = torch.log(pred_std + EPS).unsqueeze(1).expand(-1, T, -1, -1)
    dp = torch.empty_like(pred_phys)
    dp[:, 0] = pred_phys[:, 0] - last
    dp[:, 1:] = pred_phys[:, 1:] - pred_phys[:, :-1]
    f5 = torch.log(dp.abs() + EPS)
    lead = torch.arange(T, device=x_phys.device, dtype=x_phys.dtype) / (T - 1.0)
    f6 = lead.view(1, T, 1, 1).expand(1, T, H, W)
    mean3 = _box(sx, 3)
    meansq3 = _box(sx * sx, 3)
    var_local = (meansq3 - mean3 * mean3).clamp_min(0.0)
    f7 = torch.log(torch.sqrt(var_local) + EPS).unsqueeze(1).expand(-1, T, -1, -1)
    meanpred = pred_phys.mean(dim=1)
    f8 = torch.log(_grad_mag(meanpred) + EPS).unsqueeze(1).expand(-1, T, -1, -1)
    return torch.stack([f0, f1, f2, f3, f4, f5, f6, f7, f8], dim=-1)


class CondWidthState:
    """Per-`GainMapImprovedModel`-instance state + the loaded net/normalisers.

    `reset()` must be called from the model's own `reset()` (trajectory
    boundary) -- clears `prev_err`/`ema_err` so the first window of every
    trajectory takes the neutral fallback (see module docstring).
    """

    def __init__(self, ckpt_path: str, device):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.net = M1WidthNet(hidden=int(ck["hidden"]), n_feat=int(ck["n_feat"])).to(device)
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        mu = ck["mu"]
        sd = ck["sd"]
        self.mu = torch.as_tensor(mu, device=device, dtype=torch.float32).view(1, 1, 1, 1, -1)
        self.sd = torch.as_tensor(sd, device=device, dtype=torch.float32).view(1, 1, 1, 1, -1)
        self.alpha = float(ck["alpha"])
        self.lam = float(ck["lambda"])
        self.clip_lo = float(ck.get("clip_lo", 0.25))
        self.clip_hi = float(ck.get("clip_hi", 4.0))
        self.n_px = int(ck.get("n_px_extra", 5))
        self.rho = 0.6
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.prev_err = [None, None]
        self.ema_err = [None, None]

    @torch.no_grad()
    def update_channel(self, c: int, prev_target_phys_c: torch.Tensor,
                       ctr_phys_c: torch.Tensor) -> None:
        """prev_target_phys_c / ctr_phys_c: (1, T, H, W) physical. Called
        from `step()`, once per measured channel, BEFORE `self.ctr_out` is
        reassigned for the new window (see module docstring)."""
        resid = (prev_target_phys_c - ctr_phys_c).abs()
        scored = _scored(prev_target_phys_c, resid.dtype)
        num = (resid * scored).sum(dim=1)
        den = scored.sum(dim=1)
        numf = _box(num, self.n_px)
        denf = _box(den, self.n_px)
        e_all = (numf / denf.clamp_min(1e-6)).clamp_min(0.0)
        prev_ema = self.ema_err[c]
        self.ema_err[c] = e_all if prev_ema is None else (self.rho * prev_ema + (1.0 - self.rho) * e_all)
        self.prev_err[c] = e_all

    @torch.no_grad()
    def apply(self, half: torch.Tensor, x_norm: torch.Tensor, pred_out_norm: torch.Tensor,
              sig_pipe_phys: torch.Tensor, std_in: torch.Tensor, mean_in: torch.Tensor,
              std_tgt: torch.Tensor, mean_tgt: torch.Tensor) -> torch.Tensor:
        """half: (1, T, H, W, 3) normalised half-width (only channels 0,1
        rescaled; channel 2 (p, unmeasured) is left untouched). x_norm:
        (1, T, H, W, 3) input-normalised. pred_out_norm: (1, T, H, W, 3)
        target-normalised (the pipeline's OWN returned prediction, `out`).
        sig_pipe_phys: (1, T, H, W, 2), `std[..., :2] * std_tgt[:2]` (the
        bounds() half-width's own physical std, BEFORE the k-table/k_scale
        multiply). Returns a NEW tensor (half is never mutated in place)."""
        T, H, W = half.shape[1], half.shape[2], half.shape[3]
        out_half = half.clone()
        mults = []
        for c in (0, 1):
            x_phys = x_norm[..., c] * std_in[c] + mean_in[c]
            pred_phys = pred_out_norm[..., c] * std_tgt[c] + mean_tgt[c]
            sp_c = sig_pipe_phys[..., c]
            base = causal_base_features(x_phys, pred_phys, sp_c)  # (1,T,H,W,9)
            if self.prev_err[c] is None:
                neutral = torch.log(sp_c + EPS)
                prevwin, ema = neutral, neutral
                ratio_log = torch.full_like(sp_c, LOG1P_EPS)
            else:
                e_prev = self.prev_err[c].unsqueeze(1).expand(-1, T, -1, -1)
                e_ema = self.ema_err[c].unsqueeze(1).expand(-1, T, -1, -1)
                ratio = (self.prev_err[c] / sp_c[:, 0].clamp_min(EPS)).clamp_min(0.0)
                prevwin = torch.log(e_prev + EPS)
                ema = torch.log(e_ema + EPS)
                ratio_log = torch.log(ratio + EPS).unsqueeze(1).expand(-1, T, -1, -1)
            extra = torch.stack([prevwin, ema, ratio_log], dim=-1)  # (1,T,H,W,3)
            feats = torch.cat([base, extra], dim=-1)  # (1,T,H,W,12)
            feats = (feats - self.mu) / self.sd
            inp = feats.squeeze(0).permute(0, 3, 1, 2).contiguous()  # (T,12,H,W)
            m = self.net(inp).squeeze(1)  # (T,H,W)
            mult = torch.exp(self.alpha * m).clamp(self.clip_lo, self.clip_hi) * self.lam
            mults.append(mult.unsqueeze(0))  # (1,T,H,W)
        mult2 = torch.stack(mults, dim=-1)  # (1,T,H,W,2)
        out_half[..., :2] = half[..., :2] * mult2
        return out_half


# =========================================================================== #
# T2 batched cond-width forward (experiments/t2_gpu_compute/notes.md).
# Appended to the packaged module; mirrors src/t2_fast_forward.py verbatim.
# Both measured channels are computed in ONE pass on the batch axis instead of
# two, and `_grad_mag` is replaced by the bit-exact pad+narrow form.
# =========================================================================== #
_T2_GRAD_W = {}


def _t2_grad_weights(h, w, device, dtype):
    key = (h, w, str(device), dtype)
    wt = _T2_GRAD_W.get(key)
    if wt is None:
        wx = torch.full((1, 1, 1, w), 0.5, device=device, dtype=dtype)
        wx[..., 0] = 1.0
        wx[..., -1] = 1.0
        wy = torch.full((1, 1, h, 1), 0.5, device=device, dtype=dtype)
        wy[..., 0, :] = 1.0
        wy[..., -1, :] = 1.0
        wt = (wx, wy)
        _T2_GRAD_W[key] = wt
    return wt


def _t2_grad_mag(f):
    shp = f.shape
    h, w = shp[-2], shp[-1]
    x = f.reshape(-1, 1, h, w)
    wx, wy = _t2_grad_weights(h, w, f.device, f.dtype)
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = (p[:, :, 1:-1, 2:] - p[:, :, 1:-1, :-2]) * wx
    gy = (p[:, :, 2:, 1:-1] - p[:, :, :-2, 1:-1]) * wy
    out = gx * gx
    out = torch.addcmul(out, gy, gy)
    return out.sqrt_().reshape(shp)


def _t2_base_features(x_phys, pred_phys, sig_pipe):
    b, T, H, W = x_phys.shape
    last = x_phys[:, -1]
    xm = x_phys.mean(dim=1)
    sx = x_phys.std(dim=1, unbiased=True)
    f0 = torch.log(sig_pipe + EPS)
    f1 = torch.log(sx + EPS).unsqueeze(1).expand(-1, T, -1, -1)
    f2 = torch.log((last - xm).abs() + EPS).unsqueeze(1).expand(-1, T, -1, -1)
    f3 = torch.log(_t2_grad_mag(last) + EPS).unsqueeze(1).expand(-1, T, -1, -1)
    pred_std = pred_phys.std(dim=1, unbiased=True)
    f4 = torch.log(pred_std + EPS).unsqueeze(1).expand(-1, T, -1, -1)
    dp = torch.empty_like(pred_phys)
    dp[:, 0] = pred_phys[:, 0] - last
    dp[:, 1:] = pred_phys[:, 1:] - pred_phys[:, :-1]
    f5 = torch.log(dp.abs() + EPS)
    lead = torch.arange(T, device=x_phys.device, dtype=x_phys.dtype) / (T - 1.0)
    f6 = lead.view(1, T, 1, 1).expand(b, T, H, W)
    mean3 = _box(sx, 3)
    meansq3 = _box(sx * sx, 3)
    var_local = (meansq3 - mean3 * mean3).clamp_min(0.0)
    f7 = torch.log(torch.sqrt(var_local) + EPS).unsqueeze(1).expand(-1, T, -1, -1)
    meanpred = pred_phys.mean(dim=1)
    f8 = torch.log(_t2_grad_mag(meanpred) + EPS).unsqueeze(1).expand(-1, T, -1, -1)
    return torch.stack([f0, f1, f2, f3, f4, f5, f6, f7, f8], dim=-1)


@torch.no_grad()
def _t2_update_both(self, prev_target_phys, ctr_phys):
    resid = (prev_target_phys - ctr_phys).abs()
    scored = _scored(prev_target_phys, resid.dtype)
    num = (resid * scored).sum(dim=1)
    den = scored.sum(dim=1)
    numf = _box(num, self.n_px)
    denf = _box(den, self.n_px)
    e_all = (numf / denf.clamp_min(1e-6)).clamp_min(0.0)
    prev_ema = getattr(self, "_ema2", None)
    ema = e_all if prev_ema is None else (self.rho * prev_ema + (1.0 - self.rho) * e_all)
    self._ema2, self._prev2 = ema, e_all
    self.ema_err = [ema[c:c + 1] for c in (0, 1)]
    self.prev_err = [e_all[c:c + 1] for c in (0, 1)]


@torch.no_grad()
def _t2_update_channel(self, c, pt, ctr):
    if c == 0:
        self._pend = (pt, ctr)
        return
    pt0, ctr0 = self._pend
    self._pend = None
    _t2_update_both(self, torch.cat([pt0, pt], 0), torch.cat([ctr0, ctr], 0))


@torch.no_grad()
def _t2_apply(self, half, x_norm, pred_out_norm, sig_pipe_phys,
              std_in, mean_in, std_tgt, mean_tgt):
    T, H, W = half.shape[1], half.shape[2], half.shape[3]
    x_phys = (x_norm[..., :2] * std_in[:2] + mean_in[:2]).squeeze(0).permute(3, 0, 1, 2)
    pred_phys = (pred_out_norm[..., :2] * std_tgt[:2] + mean_tgt[:2]).squeeze(0).permute(3, 0, 1, 2)
    sp = sig_pipe_phys.squeeze(0).permute(3, 0, 1, 2)
    base = _t2_base_features(x_phys, pred_phys, sp)
    prev2 = getattr(self, "_prev2", None)
    if prev2 is None:
        neutral = torch.log(sp + EPS)
        prevwin = ema = neutral
        ratio_log = torch.full_like(sp, LOG1P_EPS)
    else:
        e_prev = prev2.unsqueeze(1).expand(-1, T, -1, -1)
        e_ema = self._ema2.unsqueeze(1).expand(-1, T, -1, -1)
        ratio = (prev2 / sp[:, 0].clamp_min(EPS)).clamp_min(0.0)
        prevwin = torch.log(e_prev + EPS)
        ema = torch.log(e_ema + EPS)
        ratio_log = torch.log(ratio + EPS).unsqueeze(1).expand(-1, T, -1, -1)
    extra = torch.stack([prevwin, ema, ratio_log], dim=-1)
    feats = torch.cat([base, extra], dim=-1)
    feats = (feats - self.mu) / self.sd
    inp = feats.permute(0, 1, 4, 2, 3).reshape(2 * T, -1, H, W).contiguous()
    m = self.net(inp).view(2, T, H, W)
    mult = torch.exp(self.alpha * m).clamp(self.clip_lo, self.clip_hi) * self.lam
    mult2 = mult.permute(1, 2, 3, 0).unsqueeze(0)
    out_half = half.clone()
    out_half[..., :2] = half[..., :2] * mult2
    return out_half


_t2_reset_orig = CondWidthState.reset


def _t2_reset(self):
    _t2_reset_orig(self)
    self._prev2 = self._ema2 = self._pend = None


_grad_mag = _t2_grad_mag
causal_base_features = _t2_base_features
CondWidthState.apply = _t2_apply
CondWidthState.update_channel = _t2_update_channel
CondWidthState.update_both = _t2_update_both
CondWidthState.reset = _t2_reset


# =========================================================================== #
# KD (kernel diet) = T2 tier-2 track A2 (experiments/t2_gpu_compute/notes.md
# S9-S11, `t2_cwfuse_patch.py`): the batched feature builder with every group
# of same-shaped operations stacked onto a new leading axis and computed in ONE
# launch. Used ONLY by the graphed body (t2_cw_graph_body.py); the eager
# `_t2_base_features` above is untouched. Bit-exact: elementwise pieces by
# construction; the stacked reductions (mean/std over the lead axis,
# avg_pool2d inside _box) verified torch.equal over whole streams.
# =========================================================================== #
def _kd_logs(ts):
    """`[log(t + EPS) for t in ts]` (all same shape) in three kernels."""
    return list(torch.log(torch.stack(ts, 0) + EPS).unbind(0))


def _kd_base_features(x_phys, pred_phys, sig_pipe):
    return torch.stack(_kd_base_feature_list(x_phys, pred_phys, sig_pipe), dim=-1)


def _kd_base_feature_list(x_phys, pred_phys, sig_pipe):
    """The 9 base features as a list of (b,T,H,W) tensors (expanded views where
    constant in T), in `causal_base_features`' channel order."""
    b, T, H, W = x_phys.shape
    xp = torch.stack([x_phys, pred_phys], 0)                 # (2,b,T,H,W)
    means = xp.mean(dim=2)                                   # (2,b,H,W)
    stds = xp.std(dim=2, unbiased=True)                      # (2,b,H,W)
    xm, meanpred = means[0], means[1]
    sx, pred_std = stds[0], stds[1]
    last = x_phys[:, -1]

    gm = _t2_grad_mag(torch.stack([last, meanpred], 0))      # (2,b,H,W)
    bx = _box(torch.stack([sx, sx * sx], 0).reshape(2 * b, H, W), 3).view(2, b, H, W)
    var_local = (bx[1] - bx[0] * bx[0]).clamp_min(0.0)

    l1, l2, l3, l4, l7, l8 = _kd_logs([sx, (last - xm).abs(), gm[0],
                                       pred_std, torch.sqrt(var_local), gm[1]])
    dp = torch.empty_like(pred_phys)
    dp[:, 0] = pred_phys[:, 0] - last
    dp[:, 1:] = pred_phys[:, 1:] - pred_phys[:, :-1]
    f0, f5 = _kd_logs([sig_pipe, dp.abs()])

    e = lambda t: t.unsqueeze(1).expand(-1, T, -1, -1)  # noqa: E731
    lead = torch.arange(T, device=x_phys.device, dtype=x_phys.dtype) / (T - 1.0)
    f6 = lead.view(1, T, 1, 1).expand(b, T, H, W)
    return [f0, e(l1), e(l2), e(l3), e(l4), f5, f6, e(l7), e(l8)]
