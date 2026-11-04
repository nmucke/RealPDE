"""Vendored N1SimVP backbone (SimVP-style: strided-conv encoder -> 8 gated
spatio-temporal attention blocks @ 16x32, hidden 256 -> transposed-conv
decoder + encoder skip, GroupNorm only) -- copied VERBATIM from
`src/c11a_models.py`'s `GSTABlock`/`N1SimVP` classes
(`experiments/c11a_backbones2/notes.md`), 7,111,868 params, no formula
re-derived.

`GenericCore` below is a `ship_ctxnorm_models.CtxNormUNet`-COMPATIBLE wrapper
(same `.forward(xs, extra) -> (B,T,H,W,3)` signature, same nn.Module
contract) so it slots into `ship_ctxnorm_models.CtxNormWrapper` completely
unchanged -- that class only ever calls `self.ctx_unet(xs, extra)` on
whatever module it is given at construction, never touching UNet-specific
internals (`base.stem`/`enc`/`down`/... -- those live inside `CtxNormUNet`
itself, not `CtxNormWrapper`). `cuda_graph_ctxnorm.py`'s own `_forward_body`
is equally backbone-agnostic in the SAME way: it calls `w.ctx_unet(xs,
extra)` as a black box (see that file's own `_forward_body`, line `y_s =
w.ctx_unet(xs, extra)`) -- so capturing this backbone's forward inside the
existing ctx-norm CUDA graph needed NO change to `cuda_graph_ctxnorm.py`.

State-dict key convention: attribute name `core` (`self.core = core`),
matching `scripts/c11a_finetune.py`'s own `C11A.GenericCtxNormCore(core)`
wrapper -- so a checkpoint produced by `c11a_finetune.py` (or this task's
`scripts/ship_simvp_*.py`, which call the SAME training script unedited)
loads into `GenericCore` below with zero key remapping.

New file (packaging-task rule: `submissions/2026-09-25_.../` files are new).
"""
from __future__ import annotations

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# shape contract -- identical constants to src/c11a_models.py
# --------------------------------------------------------------------------- #
T_STEPS, HH, WW = 20, 32, 64
C_IN, C_OUT = 3, 3
EXTRA_CHANNELS = 4
IN_CH = T_STEPS * C_IN + EXTRA_CHANNELS   # 64
OUT_CH = T_STEPS * C_OUT                  # 60


def fold_window(xs: torch.Tensor) -> torch.Tensor:
    """(B,T,H,W,C) input-normalised window -> (B,T*C,H,W), T outer / C inner."""
    b, t, h, w, c = xs.shape
    return xs.permute(0, 1, 4, 2, 3).reshape(b, t * c, h, w)


def unfold_output(z: torch.Tensor) -> torch.Tensor:
    """(B,T*C_out,H,W) -> (B,T,H,W,C_out)."""
    b, tc, h, w = z.shape
    t = tc // C_OUT
    return z.reshape(b, t, C_OUT, h, w).permute(0, 1, 3, 4, 2)


# --------------------------------------------------------------------------- #
# N1 -- SimVP-style backbone (verbatim copy of src/c11a_models.py's N1SimVP)
# --------------------------------------------------------------------------- #
class GSTABlock(nn.Module):
    """Gated spatio-temporal attention block (SimVP/TAU-style large-kernel
    depthwise "attention" + gate), plus an inverted-bottleneck MLP sub-block.
    Kernel 21 receptive field via a 5x5 depthwise conv followed by a 7x7
    depthwise conv dilated 3."""

    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, dim)
        self.proj_in = nn.Conv2d(dim, dim, 1)
        self.dw5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.dw7d3 = nn.Conv2d(dim, dim, 7, padding=9, dilation=3, groups=dim)
        self.gate = nn.Conv2d(dim, dim, 1)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()

        self.norm2 = nn.GroupNorm(8, dim)
        self.mlp1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.mlp2 = nn.Conv2d(mlp_ratio * dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.act(self.proj_in(self.norm1(x)))
        attn = self.dw7d3(self.dw5(u))
        g = torch.sigmoid(self.gate(u))
        x = x + self.proj_out(attn * g)
        h = self.act(self.mlp1(self.norm2(x)))
        x = x + self.mlp2(h)
        return x


class N1SimVP(nn.Module):
    __name__ = "N1_SimVP"

    def __init__(self, in_ch: int = IN_CH, out_ch: int = OUT_CH,
                 enc_dim: int = 128, hidden: int = 256, n_blocks: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, enc_dim, 3, padding=1), nn.GroupNorm(8, enc_dim), nn.GELU())
        self.down = nn.Sequential(
            nn.Conv2d(enc_dim, hidden, 4, stride=2, padding=1), nn.GroupNorm(8, hidden), nn.GELU())
        self.blocks = nn.ModuleList([GSTABlock(hidden) for _ in range(n_blocks)])
        self.up = nn.ConvTranspose2d(hidden, enc_dim, 4, stride=2, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * enc_dim, enc_dim, 1), nn.GroupNorm(8, enc_dim), nn.GELU())
        self.head = nn.Conv2d(enc_dim, out_ch, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        skip = self.stem(z)                     # (B, enc_dim, 32, 64)
        h = self.down(skip)                      # (B, hidden, 16, 32)
        for b in self.blocks:
            h = b(h)
        h = self.up(h)                            # (B, enc_dim, 32, 64)
        h = self.fuse(torch.cat([h, skip], dim=1))
        return self.head(h)


# --------------------------------------------------------------------------- #
# CtxNormUNet-compatible wrapper
# --------------------------------------------------------------------------- #
class GenericCore(nn.Module):
    """Fold/concat/unfold around an arbitrary `core(z) -> (B,60,H,W)`, exposing
    the exact `.forward(xs, extra) -> (B,T,H,W,3)` call signature
    `ship_ctxnorm_models.CtxNormWrapper.forward` uses for its own `ctx_unet`
    attribute, and that `cuda_graph_ctxnorm.py`'s captured `_forward_body`
    also uses (`y_s = w.ctx_unet(xs, extra)`). No other file needs to know
    this wraps a SimVP-style network instead of a UNetConvNeXt."""

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, xs: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        b, t, h, w, c = xs.shape
        assert (t, h, w, c) == (T_STEPS, HH, WW, C_IN), xs.shape
        assert extra.shape == (b, h, w, EXTRA_CHANNELS), extra.shape
        z = torch.cat([fold_window(xs), extra.permute(0, 3, 1, 2)], dim=1)
        assert z.shape[1] == IN_CH, z.shape
        out = self.core(z)
        assert out.shape[1] == OUT_CH, out.shape
        return unfold_output(out)
