"""Model constructors for family C1-D (capacity / architecture ablation).

New file for the C1-D task -- existing files (`src/load_baseline.py`,
`scripts/finetune_fno.py`, `src/unet_convnext.py`, `src/rpde_baselines/...`) are
imported, never edited. Builds:

  - fno_baseline : the shipped FNO3d exactly (modes (4,12,16), width 64, 4 layers,
    `src/load_baseline.py:MODEL_KWARGS["fno"]`) -- used for the shortened-sim-
    pretrain apples-to-apples rerun (see `experiments/c1d_capacity/notes.md`).
  - fno_wide     : same modes/layers, width 96 (candidate 1).
  - fno_deep     : modes (6,16,20), 6 layers, width 64 (candidate 2).
  - unet_local   : ConvNeXt U-Net (`src/unet_convnext.py`), FiLM conditioning off
    (`cond=False`) so it is directly comparable to the unconditioned FNO
    backbones -- LayerNorm-only, no BatchNorm (candidate 3).

All four expose `forward(x, params=None) -> (B, 20, 32, 64, 3)` so the same
training/eval loops (which sometimes pass `params`, sometimes don't) work for
every variant without branching on architecture.
"""
from __future__ import annotations

import torch.nn as nn

from rpde_baselines.model.fno import FNO3d
from unet_convnext import UNetConvNeXt

IN_SHAPE = (20, 32, 64, 3)
OUT_SHAPE = (20, 32, 64, 3)


class PlainFNO(nn.Module):
    """FNO3d, no FiLM conditioning; `params` accepted and ignored so it matches
    the (x, params) call signature the other c1d models / Trainer use."""

    def __init__(self, modes1: int, modes2: int, modes3: int, n_layers: int, width: int):
        super().__init__()
        self.net = FNO3d(modes1=modes1, modes2=modes2, modes3=modes3,
                          n_layers=n_layers, width=width,
                          shape_in=IN_SHAPE, shape_out=OUT_SHAPE)

    def forward(self, x, params=None):
        return self.net(x)


FNO_SPECS = {
    "fno_baseline": dict(modes1=4, modes2=12, modes3=16, n_layers=4, width=64),
    "fno_wide":     dict(modes1=4, modes2=12, modes3=16, n_layers=4, width=96),
    "fno_deep":     dict(modes1=6, modes2=16, modes3=20, n_layers=6, width=64),
}

# UNet widths chosen in scripts/c1d_bench_latency.py to land near the FNO
# baseline's per-step cost ("similar step cost" requirement); depths left at
# the module default (2 ConvNeXt blocks per stage).
UNET_WIDTHS = (56, 112, 224, 448)
UNET_DEPTHS = (2, 2, 2, 2)

MODEL_NAMES = ("fno_baseline", "fno_wide", "fno_deep", "unet_local")


def build(name: str) -> nn.Module:
    if name in FNO_SPECS:
        return PlainFNO(**FNO_SPECS[name])
    if name == "unet_local":
        return UNetConvNeXt(n_steps=IN_SHAPE[0], height=IN_SHAPE[1], width=IN_SHAPE[2],
                             in_channels=IN_SHAPE[3], out_channels=OUT_SHAPE[3],
                             widths=UNET_WIDTHS, depths=UNET_DEPTHS, cond=False)
    raise ValueError(f"unknown c1d model {name!r}; choices: {MODEL_NAMES}")


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
