"""Convolutional U-Net with ConvNeXt blocks, FiLM-conditioned on (Re, AoA).

A third backbone alongside `transolver_film.TransolverFiLM` and `fno_film.FNOFiLM`,
sharing their interface so the training scripts and the evaluation path treat all
three the same way.

## Shape handling

Evaluation is a fixed 20-frame window on a 32x64 grid, so the time axis is folded
into channels and the network is a **2D** U-Net over `(H, W)` with `T * C` channels
in and out. That is the standard treatment for fixed-length windows (the U-Net
baselines in PDEBench / PDEArena do the same) and it is much cheaper than a 3D
U-Net: every convolution then mixes the whole window in time at once, and the
depthwise 7x7 kernel gives a large spatial receptive field for the cost of a small
one. The cost is that the model is tied to `n_steps`; that is fine here, since the
protocol fixes it at 20.

## Stages

`depths` gives the number of ConvNeXt blocks per stage -- "multiple convnext layers
in each encoder and decoder layer" -- and `widths` the channel count. With the
default four stages the grid runs 32x64 -> 16x32 -> 8x16 -> 4x8, which is as deep as
a 32-row field usefully goes.

A ConvNeXt block (Liu et al. 2022) is: depthwise 7x7 conv, LayerNorm, pointwise
expansion by `mlp_ratio`, GELU, pointwise projection back, LayerScale, residual.
The heavy lifting is done by the depthwise kernel, so widening is cheap.

`layer_scale` defaults to **0.1**, not the paper's 1e-6. That value is calibrated
for 300-epoch ImageNet training of a much deeper stack: it makes every block a
near-identity at initialisation, which here would leave the network effectively
linear -- and the FiLM modulation, which lives inside the residual branch,
scaled to nothing -- for most of a handful of epochs. Pass `layer_scale=1e-6`
to reproduce the paper's schedule, or `1.0` to disable LayerScale entirely.

## Conditioning

`(Re, AoA)` go through `param_features` -- shared with the other two backbones -- and
every block modulates its LayerNorm output with a per-channel scale and shift. The
modulation layers are zero-initialised, so at initialisation the network is a plain
ConvNeXt U-Net and the conditioning is learned on top. `cond=False` removes the
pathway entirely.

Interface (identical to the other backbones):

    model = UNetConvNeXt()
    pred = model(x, params)          # x (B, 20, 32, 64, 3) normalised
    pred = model(x)                  # params estimated off x
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from transolver_film import estimate_params, param_features


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of an (B, C, H, W) tensor."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class ConvNeXtBlock(nn.Module):
    """Depthwise 7x7 -> LN (FiLM) -> pointwise 4x -> GELU -> pointwise -> LayerScale.

    `cond_dim=0` drops the conditioning and leaves a stock ConvNeXt block.
    """

    def __init__(self, dim: int, cond_dim: int = 0, mlp_ratio: int = 4,
                 layer_scale: float = 0.1, drop: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.pw2 = nn.Conv2d(mlp_ratio * dim, dim, 1)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim)) if layer_scale > 0 else None
        self.modulation = None
        if cond_dim:
            # Zero-initialised, so the block starts as a plain ConvNeXt block.
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))
            nn.init.zeros_(self.modulation[1].weight)
            nn.init.zeros_(self.modulation[1].bias)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm(self.dwconv(x))
        if self.modulation is not None and cond is not None:
            scale, shift = self.modulation(cond).chunk(2, dim=-1)
            h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.pw2(self.drop(F.gelu(self.pw1(h))))
        if self.gamma is not None:
            h = h * self.gamma[None, :, None, None]
        return x + h


class Stage(nn.Module):
    """`n` ConvNeXt blocks at one resolution, threading the conditioning through."""

    def __init__(self, dim: int, n: int, cond_dim: int, mlp_ratio: int, drop: float,
                 layer_scale: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock(dim, cond_dim, mlp_ratio, layer_scale, drop) for _ in range(n)])

    def forward(self, x, cond=None):
        for b in self.blocks:
            x = b(x, cond)
        return x


class UNetConvNeXt(nn.Module):
    """ConvNeXt U-Net over a folded (T*C, H, W) window, conditioned on (Re, AoA)."""

    def __init__(self, n_steps: int = 20, height: int = 32, width: int = 64,
                 in_channels: int = 3, out_channels: int = 3,
                 widths: Sequence[int] = (64, 128, 256, 512),
                 depths: Sequence[int] = (2, 2, 2, 2),
                 mlp_ratio: int = 4, drop: float = 0.0, layer_scale: float = 0.1,
                 cond: bool = True, cond_hidden: int = 128,
                 discrete_aoa: bool = False, si_state: bool = False):
        super().__init__()
        self.__name__ = "UNet_ConvNeXt"
        assert len(widths) == len(depths) >= 2, "need at least two stages"
        self.T, self.Hs, self.Ws = n_steps, height, width
        self.in_channels, self.out_dim = in_channels, out_channels
        self.widths, self.depths = tuple(widths), tuple(depths)
        self.n_levels = len(widths)
        self.discrete_aoa = discrete_aoa

        stride = 2 ** (self.n_levels - 1)
        if height % stride or width % stride:
            raise ValueError(f"{height}x{width} is not divisible by {stride}; "
                             f"use fewer stages")

        c_in, c_out = n_steps * in_channels, n_steps * out_channels
        cond_dim = cond_hidden if cond else 0
        self.cond_mlp = None
        if cond:
            self.cond_mlp = nn.Sequential(nn.Linear(2, cond_hidden), nn.SiLU(),
                                          nn.Linear(cond_hidden, cond_hidden), nn.SiLU())

        self.stem = nn.Conv2d(c_in, widths[0], 3, padding=1)

        # --- pseudo-state pathway (denoising stochastic interpolant) ---------------
        # A second stem, dedicated to the SDE pseudo-state X_tau, summed with the
        # window's own stem -- Xaurora's extra patch embedding (Sec. 5.2). Zeroed
        # below, so at initialisation the model ignores X_tau and is exactly the
        # deterministic prior. `si_state=False` adds no keys.
        self.state_stem = nn.Conv2d(c_out, widths[0], 3, padding=1) if si_state else None

        # --- encoder: a stage at each resolution, strided conv between them --------
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(self.n_levels):
            self.enc.append(Stage(widths[i], depths[i], cond_dim, mlp_ratio, drop, layer_scale))
            if i < self.n_levels - 1:
                self.down.append(nn.Sequential(LayerNorm2d(widths[i]),
                                               nn.Conv2d(widths[i], widths[i + 1], 2, stride=2)))

        # --- decoder: upsample, concatenate the skip, fuse, then a stage -----------
        self.up = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(self.n_levels - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(widths[i], widths[i - 1], 2, stride=2))
            self.fuse.append(nn.Conv2d(2 * widths[i - 1], widths[i - 1], 1))
            self.dec.append(Stage(widths[i - 1], depths[i - 1], cond_dim, mlp_ratio, drop, layer_scale))

        self.head_norm = LayerNorm2d(widths[0])
        self.head = nn.Conv2d(widths[0], c_out, 1)
        self.apply(self._init_weights)
        # `apply` walks the zero-initialised modulation layers too; restore them.
        # Only the *residual* branches are zeroed -- the head is the sole output
        # path, so zeroing it would emit an all-zero field and, worse, block the
        # gradient to every layer upstream of it.
        for m in self.modules():
            if isinstance(m, ConvNeXtBlock) and m.modulation is not None:
                nn.init.zeros_(m.modulation[1].weight)
                nn.init.zeros_(m.modulation[1].bias)
        if self.state_stem is not None:
            nn.init.zeros_(self.state_stem.weight)
            nn.init.zeros_(self.state_stem.bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, params: Optional[torch.Tensor] = None,
                x_tau: Optional[torch.Tensor] = None,
                cond_extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, T, H, W, C_in) normalised window.
        params: (B, 2) physical [re, aoa], or None to read them off `x`.
        x_tau: (B, T, H, W, out_dim) pseudo-state, folded and passed through
               `state_stem`, then summed with the window's stem (needs
               `si_state=True`).
        cond_extra: (B, cond_hidden) added to the conditioning vector -- how the
               pseudo-time enters in `denoising_si.py`.
        Returns (B, T, H, W, out_dim)."""
        b, t, h, w, c = x.shape
        assert (t, h, w, c) == (self.T, self.Hs, self.Ws, self.in_channels), x.shape

        cond = None
        if self.cond_mlp is not None:
            if params is None:
                params = estimate_params(x, discrete_aoa=self.discrete_aoa)
            assert params.shape == (b, 2), params.shape
            cond = self.cond_mlp(param_features(params.to(x.device, x.dtype)))
        if cond_extra is not None:
            if cond is None:
                raise ValueError("cond_extra needs a model built with cond=True: with "
                                 "cond=False the blocks have no modulation layers")
            cond = cond + cond_extra.to(x.device, x.dtype)

        # (B, T, H, W, C) -> (B, T*C, H, W): time folded into channels
        z = x.permute(0, 1, 4, 2, 3).reshape(b, t * c, h, w)
        z = self.stem(z)
        if x_tau is not None:
            if self.state_stem is None:
                raise ValueError("x_tau needs a model built with si_state=True")
            assert x_tau.shape == (b, t, h, w, self.out_dim), x_tau.shape
            z = z + self.state_stem(
                x_tau.permute(0, 1, 4, 2, 3).reshape(b, t * self.out_dim, h, w))

        skips = []
        for i in range(self.n_levels):
            z = self.enc[i](z, cond)
            if i < self.n_levels - 1:
                skips.append(z)
                z = self.down[i](z)

        for j, i in enumerate(range(self.n_levels - 1, 0, -1)):
            z = self.up[j](z)
            z = self.fuse[j](torch.cat([z, skips[i - 1]], dim=1))
            z = self.dec[j](z, cond)

        z = self.head(self.head_norm(z))                    # (B, T*out_dim, H, W)
        return z.reshape(b, t, self.out_dim, h, w).permute(0, 1, 3, 4, 2)

    def train_loss(self, input, target, params=None):
        """Mean squared error, mirroring the other backbones' API."""
        return F.mse_loss(self.forward(input, params), target)
