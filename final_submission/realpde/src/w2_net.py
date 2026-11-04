#!/usr/bin/env python3
"""W2 width nets.  All take per-(window, channel) WINDOW-level feature maps
(B, Fw, H, W) and LEAD-level maps (B, T, Fl, H, W) and return a log-multiplier
(B, T, n_out, H, W); n_out = 1 (symmetric) or 2 (asym: up, lo).

  M1Net       : M1's 3-conv net applied per lead on cat(featw, featl[t]) (the
                shipped architecture when dil=(1,2,1), hidden=32).
  LeadHeadNet : shared conv trunk on the WINDOW-level maps (run ONCE per
                window/channel), then a per-lead 1x1 MLP head that also sees
                that lead's lead-level maps (H3: per-lead structure without
                20x the trunk compute).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Trunk(nn.Module):
    def __init__(self, n_in, hidden=32, dil=(1, 2, 1), groups=8):
        super().__init__()
        layers = []
        c = n_in
        for d in dil:
            layers += [nn.Conv2d(c, hidden, 3, padding=d, dilation=d), nn.GroupNorm(groups, hidden), nn.SiLU()]
            c = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class M1Net(nn.Module):
    def __init__(self, n_w, n_l, hidden=32, dil=(1, 2, 1), n_out=1):
        super().__init__()
        self.trunk = Trunk(n_w + n_l, hidden, dil)
        self.out = nn.Conv2d(hidden, n_out, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.n_out = n_out

    def forward(self, fw, fl):
        B, T = fl.shape[:2]
        x = torch.cat([fw[:, None].expand(-1, T, -1, -1, -1), fl], dim=2)  # (B,T,F,H,W)
        x = x.reshape(B * T, x.shape[2], x.shape[3], x.shape[4])
        y = self.out(self.trunk(x))
        return y.reshape(B, T, self.n_out, y.shape[-2], y.shape[-1])


class LeadHeadNet(nn.Module):
    def __init__(self, n_w, n_l, T=20, hidden=32, dil=(1, 2, 1), head_hidden=16, n_out=1):
        super().__init__()
        self.trunk = Trunk(n_w, hidden, dil)
        self.T, self.n_out = T, n_out
        self.W1 = nn.Parameter(torch.randn(T, hidden + n_l, head_hidden) * (1.0 / (hidden + n_l) ** 0.5))
        self.b1 = nn.Parameter(torch.zeros(T, head_hidden))
        self.W2 = nn.Parameter(torch.zeros(T, head_hidden, n_out))
        self.b2 = nn.Parameter(torch.zeros(T, n_out))
        self.act = nn.SiLU()

    def forward(self, fw, fl):
        h = self.trunk(fw)                                    # (B,C,H,W)
        B, C, H, W = h.shape
        x = torch.cat([h[:, None].expand(-1, self.T, -1, -1, -1), fl], dim=2)   # (B,T,C+Fl,H,W)
        z = torch.einsum("btchw,tck->btkhw", x, self.W1) + self.b1[None, :, :, None, None]
        z = self.act(z)
        y = torch.einsum("btkhw,tko->btohw", z, self.W2) + self.b2[None, :, :, None, None]
        return y


def build(arch: str, n_w: int, n_l: int, hidden=32, dil=(1, 2, 1), n_out=1, T=20):
    if arch == "m1":
        return M1Net(n_w, n_l, hidden, dil, n_out)
    if arch == "leadhead":
        return LeadHeadNet(n_w, n_l, T, hidden, dil, 16, n_out)
    raise ValueError(arch)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
