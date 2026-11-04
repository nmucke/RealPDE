"""Flow-field TKE-map predictor: a learned correction to `gain_map`'s energy rule.

`gain_map_improved` sets, per (pixel, channel), the temporal energy of the output
fluctuation this window to the hand-set rule

    e_rule = var_t(F') * (E_input / E_F)^0.75        (running energies, per pixel)

and `_shape` rescales the blended fluctuation to it.  `tke_score` is the relative
L2 of that energy map against the truth's, so the rule *is* the TKE prediction.

Idea 2 (`experiments/gain_map_improved/tke_ideas/idea_2.md`) learned a log-space
residual to the rule from 20 per-pixel **energy summary** features and bought
+1.3 / +1.6 TKE; its report also measured the ceiling -- feeding the true
per-window energy map scores tke 94 / 97 against 84 / 83 -- and identified the
missing information as the window-to-window modulation of the coherent shedding
band, which no summary of the energy can see.

This module keeps idea 2's two honest choices (a residual to the rule with a
zero-initialised last layer, so `scale = 0` is bit-identical to the current
submission; the metric itself as the loss) and changes the input: the net sees
the **flow fields**.  Per window it gets the last `k_in` input frames as
fluctuations about the window's temporal mean, the FNO forecast's fluctuation at
`leads` subsampled lead times, and the usual energy context.  Everything is
divided by the square root of a *running* field-mean energy, so nothing carries
the trajectory's Reynolds number as a scale and the net only sees where the wake
is and what phase / amplitude state it is in.

Used by `scripts/fit_ke_net_flow.py` (training) and by
`experiments/ke_net_flow/submission.py` (inference), which must build features
the same way -- hence one builder here rather than two copies.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8

# Official train_real target statistics: the ke map is scored in physical units,
# ke = 0.5 (su^2 var_t(u) + sv^2 var_t(v)), and u and v differ by 6x.
SU, SV = 0.09681041, 0.01596364

DEFAULT_LEADS: Tuple[int, ...] = (0, 4, 9, 14, 19)
FIELD_CLAMP = 8.0          # scale-free fields are O(1); clip freak PIV pixels
LOG_CLAMP = 6.0            # log-ratio features, likewise


# --------------------------------------------------------------------------- #
# Features.
# --------------------------------------------------------------------------- #
def n_features(k_in: int, n_leads: int, summary: bool) -> int:
    """Input channel count of `features`: fields + 4 log-ratios + 4 context (+ idea 2's 4).

    `k_in = 0, n_leads = 0, summary = True` is exactly idea 2's 20-channel energy
    summary, so the two feature sets are comparable under one builder.
    """
    return 2 * k_in + 2 * n_leads + 8 + 4 + (8 if summary else 0)


def features(
    xw: torch.Tensor,
    ffl: torch.Tensor,
    ex: torch.Tensor,
    ef: torch.Tensor,
    runx: torch.Tensor,
    runf: torch.Tensor,
    mask: torch.Tensor,
    wfrac: torch.Tensor,
    k_in: int = 6,
    leads: Sequence[int] = DEFAULT_LEADS,
    summary: bool = False,
    halves: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ex_prev: Optional[torch.Tensor] = None,
    ef_prev: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Stack the per-window inputs into (B, C, H, W).

    Args (all normalised space, u/v only):
        xw     (B, T, H, W, 2)  input window
        ffl    (B, T, H, W, 2)  FNO forecast fluctuation (window mean removed)
        ex, ef (B, H, W, 2)     their per-pixel temporal variances
        runx, runf (B, H, W, 2) the stream's running means of ex, ef (this window included)
        mask   (B, H, W) bool   pixels that are exactly zero in the physical input
        wfrac  (B,)             min(window index, 20) / 20 -- how warm the statistics are

    The fields are made scale-free by dividing by `sqrt(mean_field(run*))`, a
    per-channel scalar from the *running* energy: a per-window normalisation
    would hide exactly the window-to-window modulation we are trying to predict,
    and the raw amplitude would leak Re in as a scale.  `summary` appends idea
    2's remaining four log-ratios (the within-window trends and the two lagged
    features) so the two feature sets can be compared.
    """
    b = xw.shape[0]
    lg = lambda a, c: torch.log((a + EPS) / (c + EPS)).clamp(-LOG_CLAMP, LOG_CLAMP)
    lvl = lambda a: lg(a, a.mean(dim=(1, 2), keepdim=True))
    flat = lambda t: t.permute(0, 3, 1, 2)                      # (B,H,W,2) -> (B,2,H,W)

    sx = runx.mean(dim=(1, 2), keepdim=True).clamp_min(EPS).sqrt()   # (B,1,1,2)
    sf = runf.mean(dim=(1, 2), keepdim=True).clamp_min(EPS).sqrt()

    stack = lambda t: t.permute(0, 1, 4, 2, 3).reshape(b, -1, *xw.shape[2:4])
    fields = []
    if k_in:
        fields.append(stack((xw[:, -k_in:] - xw.mean(dim=1, keepdim=True)) / sx.unsqueeze(1)))
    if len(leads):
        fields.append(stack(ffl[:, list(leads)] / sf.unsqueeze(1)))
    fields = [torch.cat(fields, 1).clamp(-FIELD_CLAMP, FIELD_CLAMP)] if fields else []

    energy = torch.cat([flat(lg(ef, runf)), flat(lg(ex, runx)),
                        flat(lg(runx, runf)), flat(lvl(runx))], 1)
    mx = xw.mean(dim=1)
    context = torch.stack([mask.to(xw.dtype),
                           mx[..., 0], mx[..., 1] * 6.0,      # v is 6x smaller than u
                           wfrac.view(-1, 1, 1).expand_as(mask)], 1)
    out = fields + [energy, context]
    if summary:
        ex1, ex2, ef1, ef2 = halves
        if ex_prev is None:
            ex_prev = ef_prev = ex
        out.append(torch.cat([flat(lg(ex2, ex1)), flat(lg(ef2, ef1)),
                              flat(lg(ex, ef_prev)), flat(lg(ex, ex_prev))], 1))
    return torch.cat(out, 1)


# --------------------------------------------------------------------------- #
# Net.
# --------------------------------------------------------------------------- #
class KeFlowNet(nn.Module):
    """Small 2-D conv net over the stacked window: (B, C, H, W) -> (B, H, W, 2).

    A 1x1 stem compresses the ~40 input channels before any 3x3 touches them
    (the stem is 90 % of the naive first-layer cost), then two dilated 3x3 blocks
    at full resolution and one downsampled branch give ~20 px of receptive field
    -- enough to place the wake and see a shedding wavelength -- for ~35k
    parameters and ~40 MMAC, which is 5-10 ms on 4 CPU threads against the FNO's
    ~350 ms.  The output layer is zero-initialised: an untrained net is exactly
    `gain_map`'s rule.
    """

    def __init__(self, in_ch: int, width: int = 24, wide: int = 32):
        super().__init__()
        c3 = lambda a, c, d=1: nn.Conv2d(a, c, 3, padding=d, dilation=d,
                                         padding_mode="replicate")
        self.stem = nn.Conv2d(in_ch, width, 1)
        self.b1 = c3(width, width)
        self.b2 = c3(width, width, 2)
        self.d1 = c3(width, wide)
        self.d2 = c3(wide, wide)
        self.fuse = nn.Conv2d(width + wide, width, 1)
        self.b3 = c3(width, width)
        self.head = nn.Conv2d(width, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.stem(feat))
        h = F.gelu(self.b1(h))
        h = F.gelu(self.b2(h))
        d = F.avg_pool2d(h, 2)
        d = F.gelu(self.d1(d))
        d = F.gelu(self.d2(d))
        d = F.interpolate(d, size=h.shape[-2:], mode="bilinear", align_corners=False)
        h = F.gelu(self.fuse(torch.cat([h, d], 1)))
        h = F.gelu(self.b3(h))
        return self.head(h).permute(0, 2, 3, 1)          # (B, H, W, 2) log-correction


# --------------------------------------------------------------------------- #
# The rule, the correction, the metric.
# --------------------------------------------------------------------------- #
def rule_log_energy(ef: torch.Tensor, runx: torch.Tensor, runf: torch.Tensor,
                    power: float = 0.75) -> torch.Tensor:
    """log of `gain_map`'s target energy var_t(F') (E_input / E_F)^power."""
    return (torch.log(ef + EPS)
            + power * (torch.log(runx + EPS) - torch.log(runf + EPS)))


def apply_correction(corr: torch.Tensor, log_rule: torch.Tensor, mask: torch.Tensor,
                     scale: float = 1.0, clip: float = 1.5) -> torch.Tensor:
    """exp(log_rule + scale * clip(corr)), with the rule kept inside the mask.

    Masked pixels are exactly zero in the reference, so the metric never sees
    them; leaving the rule there keeps the correction from drifting on pixels
    that carry no gradient.
    """
    corr = corr.clamp(-clip, clip) * scale * (~mask).unsqueeze(-1).to(corr.dtype)
    return (log_rule + corr).exp()


def ke_phys(e: torch.Tensor) -> torch.Tensor:
    """(..., H, W, 2) per-channel energy -> the physical ke map the metric scores."""
    return 0.5 * (SU ** 2 * e[..., 0] + SV ** 2 * e[..., 1])


def ke_rel_l2(e_pred: torch.Tensor, e_true: torch.Tensor) -> torch.Tensor:
    """Per-window relative L2 of the ke map -- the `tke_score` integrand."""
    p, t = ke_phys(e_pred).flatten(1), ke_phys(e_true).flatten(1)
    return (p - t).norm(dim=1) / t.norm(dim=1).clamp_min(1e-12)


# --------------------------------------------------------------------------- #
# Checkpoints.
# --------------------------------------------------------------------------- #
def save_ke_flow_net(path: str, net: KeFlowNet, cfg: Dict) -> None:
    torch.save({"cfg": dict(cfg),
                "state": {k: v.detach().cpu() for k, v in net.state_dict().items()}},
               path)


def load_ke_flow_net(path: str, device: str = "cpu") -> Tuple[KeFlowNet, Dict]:
    """Rebuild the net saved by `save_ke_flow_net`; returns (net.eval(), cfg)."""
    ck = torch.load(path, map_location=device)
    cfg = dict(ck["cfg"])
    net = KeFlowNet(cfg["in_ch"], cfg["width"], cfg["wide"]).to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, cfg
