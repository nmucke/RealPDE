"""Grouped-conv packed K-seed `ke_net_flow` ensemble (KE Stage 1, brief
`runs/briefs/KE_seed_ensemble.md`).

Stage 0 (`experiments/ke_seed_ens/notes.md`) showed a 4-seed log-space average of
independently trained `KeFlowNet`s beats the mean single-seed val e by 1.25 % (>= the
1 % KILL bar). Running 4 separate `nn.Module.forward` calls costs ~4x the single net's
kernels; this module packs the K nets into ONE grouped-convolution net so the ensemble
costs a fraction of 4x (grouped conv is one kernel launch per layer regardless of K,
same as the single net's launch count) while being numerically equivalent (<= 1e-6) to
averaging the K separate nets' raw (pre-clamp) log-corrections.

Construction, per layer of `ke_net_flow.KeFlowNet`:
  * every layer becomes a `groups=K` conv with in/out channels scaled by K;
  * the input feature map is tiled K times along channels (`feat.repeat(1, K, 1, 1)`) so
    each group sees the SAME features (an ensemble, not K different inputs);
  * each group's weight/bias slice is loaded verbatim from one trained seed net, so group
    i's activations are byte-for-byte what that seed's own conv would produce on the same
    input (grouped conv with groups=K, in_channels/g == that seed's own in-channels, is
    exactly K independent convolutions, batched into one cuDNN call);
  * `stem`/`b1`/`b2`/`d1`/`d2`/`b3`/`head` all keep a fixed per-seed channel block
    ordering (block i = width*i .. width*(i+1) for width-sized layers, wide*i..wide*(i+1)
    for wide-sized ones) so each grouped conv's input slice IS the corresponding previous
    layer's own group's output -- no cross-seed mixing at any point;
  * `fuse` needs `cat([h, d], 1)` per seed contiguous (not all-h-then-all-d globally): the
    packed `h`/`d` tensors are reshaped (B,K,C,H,W) -> concat on the channel axis -> reshape
    back, an exact reordering (no compute), before the grouped conv;
  * `avg_pool2d` / bilinear `interpolate` / `gelu` are channel-independent (spatial-only or
    elementwise) so applying them to the whole packed tensor is exactly K independent
    applications -- no change needed;
  * the final K*2-channel raw correction is reshaped (B,K,2,H,W) and averaged over K,
    reproducing Stage 0's ensemble exactly (mean of the K nets' raw `net(feat)` outputs,
    before `apply_correction`'s clamp/scale).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ke_net_flow as K  # noqa: E402


def _block_diag_weight(weights: Sequence[torch.Tensor]) -> torch.Tensor:
    """Stack K per-group conv weights (each (out_g, in_g, kh, kw)) along dim 0 ->
    (K*out_g, in_g, kh, kw), the layout `nn.Conv2d(..., groups=K)` expects."""
    return torch.cat(list(weights), dim=0)


def _block_diag_bias(biases: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat(list(biases), dim=0)


class KeFlowNetPacked(nn.Module):
    """K independent `KeFlowNet`s, packed into one grouped-conv graph.

    `forward(feat)` takes the SAME (B, in_ch, H, W) feature map `KeFlowNet.forward` takes
    (in_ch = cfg["in_ch"], NOT multiplied by K) and returns the K-averaged raw log-correction,
    shape (B, H, W, 2) -- a drop-in replacement for a single `KeFlowNet.forward`'s output
    wherever the caller was going to average K separate nets' outputs.
    """

    def __init__(self, in_ch: int, width: int, wide: int, k: int):
        super().__init__()
        self.in_ch, self.width, self.wide, self.k = in_ch, width, wide, k
        c3g = lambda a, c, d=1: nn.Conv2d(a * k, c * k, 3, padding=d, dilation=d,
                                          groups=k, padding_mode="replicate")
        self.stem = nn.Conv2d(in_ch * k, width * k, 1, groups=k)
        self.b1 = c3g(width, width)
        self.b2 = c3g(width, width, 2)
        self.d1 = c3g(width, wide)
        self.d2 = c3g(wide, wide)
        self.fuse = nn.Conv2d((width + wide) * k, width * k, 1, groups=k)
        self.b3 = c3g(width, width)
        self.head = nn.Conv2d(width * k, 2 * k, 1, groups=k)

    @torch.no_grad()
    def load_seeds(self, nets: Sequence[K.KeFlowNet]) -> None:
        assert len(nets) == self.k, f"expected {self.k} nets, got {len(nets)}"
        for name in ("stem", "b1", "b2", "d1", "d2", "fuse", "b3", "head"):
            dst = getattr(self, name)
            dst.weight.copy_(_block_diag_weight([getattr(n, name).weight for n in nets]))
            dst.bias.copy_(_block_diag_bias([getattr(n, name).bias for n in nets]))

    def _regroup_cat(self, h: torch.Tensor, d: torch.Tensor, ch_h: int, ch_d: int,
                     hw) -> torch.Tensor:
        """`cat([h_i, d_i])` per seed i, seed-contiguous -- exact reorder, no compute."""
        b = h.shape[0]
        hh = h.view(b, self.k, ch_h, *hw)
        dd = d.view(b, self.k, ch_d, *hw)
        return torch.cat([hh, dd], dim=2).reshape(b, self.k * (ch_h + ch_d), *hw)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        k = self.k
        x = feat.repeat(1, k, 1, 1)               # same features, K groups
        h = F.gelu(self.stem(x))
        h = F.gelu(self.b1(h))
        h = F.gelu(self.b2(h))
        hw = h.shape[-2:]
        d = F.avg_pool2d(h, 2)
        d = F.gelu(self.d1(d))
        d = F.gelu(self.d2(d))
        d = F.interpolate(d, size=hw, mode="bilinear", align_corners=False)
        hd = self._regroup_cat(h, d, self.width, self.wide, hw)
        h = F.gelu(self.fuse(hd))
        h = F.gelu(self.b3(h))
        out = self.head(h)                         # (B, 2K, H, W)
        b = out.shape[0]
        out = out.view(b, k, 2, *hw).mean(dim=1)    # average over the K seeds
        return out.permute(0, 2, 3, 1)              # (B, H, W, 2)


def build_packed(nets: Sequence[K.KeFlowNet], cfg: Dict) -> KeFlowNetPacked:
    k = len(nets)
    packed = KeFlowNetPacked(cfg["in_ch"], cfg["width"], cfg["wide"], k)
    packed.load_seeds(nets)
    packed.eval()
    for p in packed.parameters():
        p.requires_grad_(False)
    return packed


def save_packed(path: str, packed: KeFlowNetPacked, cfg: Dict) -> None:
    torch.save({"cfg": dict(cfg), "k": packed.k,
               "state": {kk: v.detach().cpu() for kk, v in packed.state_dict().items()}},
              path)


def load_packed(path: str, device: str = "cpu") -> KeFlowNetPacked:
    ck = torch.load(path, map_location=device)
    cfg = dict(ck["cfg"])
    packed = KeFlowNetPacked(cfg["in_ch"], cfg["width"], cfg["wide"], ck["k"]).to(device)
    packed.load_state_dict(ck["state"])
    packed.eval()
    for p in packed.parameters():
        p.requires_grad_(False)
    return packed
