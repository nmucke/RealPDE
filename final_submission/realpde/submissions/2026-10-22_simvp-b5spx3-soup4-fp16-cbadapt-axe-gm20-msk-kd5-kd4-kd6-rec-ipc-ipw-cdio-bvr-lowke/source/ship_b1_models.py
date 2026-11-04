"""Vendored B1 backbone `b5_h192_blk6_rf11` (P8, 2026-09-23).

A self-contained, pure-torch copy of `src/c14a_models.py`'s `GSTABlockCfg` and
`SimVP1Stage` (unedited formulas; only the module-level shape constants are
taken from the already-vendored `ship_simvp_models.py`, which copies them from
`src/c11a_models.py` exactly as `c14a_models.py` itself does). `b5` is the
registry entry `src/b1_models.py` VARIANTS["b5_h192_blk6_rf11"] =
`SimVP1Stage(enc_dim=128, hidden=192, n_blocks=6, k1=3, k2=5, d2=2)`:
n1_simvp's topology (strided-conv encoder -> gSTA blocks @16x32 -> transposed
conv decoder + skip) with a narrower hidden width, fewer blocks and a smaller
receptive field (RF = k1 + d2*(k2-1) = 11). 3,390,460 params.

Checkpoints written by `scripts/b1_finetune.py` wrap this core in
`c11a_models.GenericCtxNormCore` (attribute `core`), i.e. the exact state-dict
convention of `ship_simvp_models.GenericCore` -- `build_b5_core()` returns that
wrapper, so a `b1_finetune.py` state dict loads with zero key remapping.

Parameter names differ from `ship_simvp_models.GSTABlock` (`dw1`/`dw2` here vs
`dw5`/`dw7d3` there), which is why this is a separate class rather than a
reuse of `N1SimVP`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import ship_simvp_models as _SIMVP  # noqa: E402  (shape constants + GenericCore)

IN_CH, OUT_CH = _SIMVP.IN_CH, _SIMVP.OUT_CH

ARCH_B5 = "b5_h192_blk6_rf11"
B5_CFG = dict(enc_dim=128, hidden=192, n_blocks=6, k1=3, k2=5, d2=2)


class GSTABlockCfg(nn.Module):
    """Verbatim copy of `src/c14a_models.py` GSTABlockCfg."""

    def __init__(self, dim: int, mlp_ratio: int = 4, k1: int = 5, k2: int = 7, d2: int = 3):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, dim)
        self.proj_in = nn.Conv2d(dim, dim, 1)
        self.dw1 = nn.Conv2d(dim, dim, k1, padding=k1 // 2, groups=dim)
        pad2 = d2 * (k2 - 1) // 2
        self.dw2 = nn.Conv2d(dim, dim, k2, padding=pad2, dilation=d2, groups=dim)
        self.gate = nn.Conv2d(dim, dim, 1)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()

        self.norm2 = nn.GroupNorm(8, dim)
        self.mlp1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.mlp2 = nn.Conv2d(mlp_ratio * dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.act(self.proj_in(self.norm1(x)))
        attn = self.dw2(self.dw1(u))
        g = torch.sigmoid(self.gate(u))
        x = x + self.proj_out(attn * g)
        h = self.act(self.mlp1(self.norm2(x)))
        x = x + self.mlp2(h)
        return x


class SimVP1Stage(nn.Module):
    """Verbatim copy of `src/c14a_models.py` SimVP1Stage (the `dilations`
    option is not used by any shipped variant and is omitted)."""

    __name__ = "SimVP1Stage"

    def __init__(self, in_ch: int = IN_CH, out_ch: int = OUT_CH, enc_dim: int = 128,
                 hidden: int = 256, n_blocks: int = 8, k1: int = 5, k2: int = 7, d2: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, enc_dim, 3, padding=1), nn.GroupNorm(8, enc_dim), nn.GELU())
        self.down = nn.Sequential(
            nn.Conv2d(enc_dim, hidden, 4, stride=2, padding=1), nn.GroupNorm(8, hidden), nn.GELU())
        self.blocks = nn.ModuleList(
            [GSTABlockCfg(hidden, k1=k1, k2=k2, d2=d2) for _ in range(n_blocks)])
        self.up = nn.ConvTranspose2d(hidden, enc_dim, 4, stride=2, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * enc_dim, enc_dim, 1), nn.GroupNorm(8, enc_dim), nn.GELU())
        self.head = nn.Conv2d(enc_dim, out_ch, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        skip = self.stem(z)
        h = self.down(skip)
        for b in self.blocks:
            h = b(h)
        h = self.up(h)
        h = self.fuse(torch.cat([h, skip], dim=1))
        return self.head(h)


def build_b5_core() -> nn.Module:
    """`GenericCore(SimVP1Stage(**B5_CFG))` -- the `ctx_unet` the packaged
    `CtxNormWrapper` wraps (same `.forward(xs, extra)` contract as the n1
    path's `GenericCore(N1SimVP())`)."""
    return _SIMVP.GenericCore(SimVP1Stage(**B5_CFG))
