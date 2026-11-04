"""KeNet -- a learned per-window TKE-map predictor for the Track 2 energy rule.

WHAT IT PREDICTS.  `gain_map`'s energy rule fixes the output's temporal variance per
(pixel, channel) for the coming window to

    E_rule = ef * (runx / runf) ** power                       (power = 0.75)

with `ef = var_t(F(x) - mean_t F(x))` the FNO's fluctuation energy in this window and
`runx`, `runf` the trajectory's running means of the input's and the forecast's energy.
In logs that is one *hand-set* linear combination of two features; everything else the
stream knows about this window (how the FNO's energy trends inside the window, how the
local level compares to the field, what the mask says) is thrown away.

`KeNet` learns the rest as a **residual correction in log space**:

    log E_out = log E_rule  +  s * clamp(net(features), -clip, clip)

The last convolution is zero-initialised, so an untrained net -- and `s = 0` -- reproduce
the shipped rule bit-for-bit.  The net can only spend capacity on what the rule gets
wrong, and `s` (`ke_net_scale` in the submission) is a shrink knob at test time.

WHY IT TRAINS ON THE METRIC.  The loss is the `tke_score` integrand itself: the mean over
windows of the relative L2 between the predicted and the true kinetic-energy map
`ke = 0.5 (su^2 e_u + sv^2 e_v)` in physical units.  Earlier log-space regressions on the
same features did not beat the rule, because a log error weights the near-zero freestream
pixels the metric ignores and ignores the wake pixels it lives on.

FEATURES (20 channels, all scale-free log-ratios so nothing depends on the flow's units):
per channel `log(ef/runf)`, `log(ex/runx)`, the within-window trends `log(ef2/ef1)` and
`log(ex2/ex1)` (second vs first half), `log(runx/runf)`, the local level
`log(runx / field-mean runx)`, and two lagged terms `log(ex/ef_prev)` -- the just-revealed
energy error of the previous forecast -- and `log(ex/ex_prev)`; that is 8 x 2 = 16, plus
the occlusion mask, the two mean-velocity components and the window-index fraction.

HOW IT PLUGS IN.  The submission already rescales its blended fluctuation to a requested
per-(pixel, channel) energy in `_shape`, so the only thing that changes downstream is the
number handed to it.  `predict_log_energy` returns the log-space correction; the caller
exponentiates it into the energy gain.  The mask region keeps the rule's value (the truth
is exactly zero there and is not scored).

Pure torch, deterministic, no dependencies beyond the eval image's torch 2.2.2.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

EPS = 1e-8
FEATS = 20          # 8 per-channel log-ratios x 2 channels + mask + mean u + mean v + window fraction

# Official train_real target-channel stds; the ke map is scored in physical units.
SU, SV = 0.09681041, 0.01596364


# --------------------------------------------------------------------------- #
# Features.
# --------------------------------------------------------------------------- #
def features(ex, ef, ex1, ex2, ef1, ef2, runx, runf, mx, mask, wfrac,
             ex_prev=None, ef_prev=None) -> torch.Tensor:
    """Build the (B, FEATS, H, W) feature stack.

    All energy/mean arguments are (B, H, W, 2) -- u, v only; `mask` is (B, H, W) bool and
    `wfrac` is (B,).  `ex_prev` / `ef_prev` default to `ex`, which is what the first window
    of a trajectory has to use.
    """
    if ex_prev is None:
        ex_prev = ex
    if ef_prev is None:
        ef_prev = ex
    lg = lambda a, b: torch.log((a + EPS) / (b + EPS))
    lvl = lambda a: torch.log((a + EPS) / a.mean(dim=(1, 2), keepdim=True).clamp_min(EPS))
    f = [lg(ef, runf), lg(ex, runx), lg(ef2, ef1), lg(ex2, ex1), lg(runx, runf), lvl(runx),
         lg(ex, ef_prev), lg(ex, ex_prev)]
    f = torch.cat([a.permute(0, 3, 1, 2) for a in f], dim=1)            # 8 * 2 = 16
    extra = torch.stack([mask.to(f.dtype),
                         mx[..., 0], mx[..., 1] * 6.0,   # v is ~6x smaller than u: match scales
                         wfrac.view(-1, 1, 1).expand_as(mask)], dim=1)  # 4
    return torch.cat([f, extra], dim=1)


def window_features(x, fl_f, e_x, e_f, run_x, run_f, mask, wfrac, ex_prev=None, ef_prev=None):
    """Features straight from the tensors `ttt_step` already holds.

    `x` and `fl_f` are (1, T, H, W, 3) -- the input window and the FNO's fluctuation --
    and `e_x`, `e_f`, `run_x`, `run_f` are the (1, 1, H, W, 3) energy maps the submission
    keeps.  The two half-window variances are the only extra arithmetic (~2.7 ms on CPU).
    Returns (features, ex, ef) so the caller can cache `ex`/`ef` as the next step's lag.
    """
    ex, ef = e_x[:, 0, ..., :2], e_f[:, 0, ..., :2]
    ex1 = x[:, :10, ..., :2].var(dim=1, unbiased=False)
    ex2 = x[:, 10:, ..., :2].var(dim=1, unbiased=False)
    ef1 = fl_f[:, :10, ..., :2].var(dim=1, unbiased=False)
    ef2 = fl_f[:, 10:, ..., :2].var(dim=1, unbiased=False)
    feat = features(ex, ef, ex1, ex2, ef1, ef2, run_x[:, 0, ..., :2], run_f[:, 0, ..., :2],
                    x[:, :, ..., :2].mean(dim=1), mask, wfrac, ex_prev, ef_prev)
    return feat, ex, ef


# --------------------------------------------------------------------------- #
# The net.
# --------------------------------------------------------------------------- #
class KeNet(nn.Module):
    """`depth` convolutions of `width` channels, last layer zero-initialised.

    3x3 kernels with replicate padding: the correction is a local function of the energy
    summaries, and the wake is a connected region a few pixels wide.  Widening beyond
    ~5k parameters or to 5x5 kernels made held-out error worse -- the limit is how
    predictable the energy is, not capacity.
    """

    def __init__(self, width: int = 16, kernel: int = 3, depth: int = 3):
        super().__init__()
        self.cfg = dict(width=width, kernel=kernel, depth=depth)
        chans = [FEATS] + [width] * (depth - 1) + [2]
        layers: List[nn.Module] = []
        for i in range(depth):
            layers.append(nn.Conv2d(chans[i], chans[i + 1], kernel, padding=kernel // 2,
                                    padding_mode="replicate"))
            if i < depth - 1:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)      # start exactly at the rule

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """(B, FEATS, H, W) -> (B, H, W, 2) log-space correction."""
        return self.net(feat).permute(0, 2, 3, 1)


class KeNetEnsemble(nn.Module):
    """Mean of several members' log-corrections; a single member behaves like `KeNet`."""

    def __init__(self, members: List[KeNet]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        out = self.members[0](feat)
        for m in self.members[1:]:
            out = out + m(feat)
        return out / len(self.members)


# --------------------------------------------------------------------------- #
# The energy target.
# --------------------------------------------------------------------------- #
def rule_log_energy(ef, runx, runf, power: float = 0.75) -> torch.Tensor:
    """log of gain_map's rule energy `ef * (runx/runf)**power`."""
    return torch.log(ef + EPS) + power * (torch.log(runx + EPS) - torch.log(runf + EPS))


def predict_log_energy(net, feat, mask=None, scale: float = 1.0, clip: float = 1.5) -> torch.Tensor:
    """The clamped, scaled, mask-zeroed log correction: (B, H, W, 2).

    Clamping at +-`clip` (a factor e^1.5 ~ 4.5 in energy) bounds the damage a feature far
    outside the training distribution can do.  Inside the occlusion mask the rule's value
    is kept: the truth is exactly zero there and the pixel is not scored.
    """
    corr = net(feat).clamp(-clip, clip) * scale
    if mask is not None:
        corr = corr * (~mask).unsqueeze(-1).to(corr.dtype)
    return corr


# --------------------------------------------------------------------------- #
# The loss: the tke_score integrand.
# --------------------------------------------------------------------------- #
def ke_phys(e: torch.Tensor) -> torch.Tensor:
    """(..., H, W, 2) normalized-space energies -> the physical ke map (..., H, W)."""
    return 0.5 * (SU ** 2 * e[..., 0] + SV ** 2 * e[..., 1])


def ke_rel_l2(e_pred: torch.Tensor, e_true: torch.Tensor) -> torch.Tensor:
    """Per-window relative L2 of the ke map -- exactly what `tke_score` averages."""
    p, t = ke_phys(e_pred).flatten(1), ke_phys(e_true).flatten(1)
    return (p - t).norm(dim=1) / t.norm(dim=1).clamp_min(1e-12)


def metric_loss(net, feat, log_rule, e_true, mask=None, scale: float = 1.0,
                clip: float = 1.5) -> torch.Tensor:
    """Mean over the batch of `ke_rel_l2(exp(log_rule + correction), e_true)`."""
    e_pred = (log_rule + predict_log_energy(net, feat, mask, scale, clip)).exp()
    return ke_rel_l2(e_pred, e_true).mean()


# --------------------------------------------------------------------------- #
# Loading.
# --------------------------------------------------------------------------- #
def load_ke_net(path: str, device: str = "cpu") -> nn.Module:
    """Load `scripts/fit_ke_net.py`'s checkpoint as an eval-mode module.

    The file holds `{"cfg": {...}, "states": [state_dict, ...]}` -- one entry per seed.
    A single state is returned as a `KeNet`, several as a `KeNetEnsemble` averaging their
    log-corrections (worth ~+0.1 TKE for one extra millisecond).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"KeNet weights not found: {path}")
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg: Dict = ck.get("cfg", {})
    states = ck.get("states") or [ck["state"]]
    members = []
    for sd in states:
        net = KeNet(**cfg)
        net.load_state_dict(sd)
        members.append(net)
    net: nn.Module = members[0] if len(members) == 1 else KeNetEnsemble(members)
    return net.to(device).eval()


def save_ke_net(path: str, cfg: Dict, states: List[Dict]) -> None:
    """Counterpart of `load_ke_net`; `states` is one state_dict per seed."""
    torch.save({"cfg": dict(cfg), "states": [{k: v.detach().cpu() for k, v in s.items()}
                                             for s in states]}, path)


def n_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


__all__ = ["EPS", "FEATS", "SU", "SV", "KeNet", "KeNetEnsemble", "features", "window_features",
           "rule_log_energy", "predict_log_energy", "ke_phys", "ke_rel_l2", "metric_loss",
           "load_ke_net", "save_ke_net", "n_params"]
