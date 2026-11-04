"""Vendored PACKED K = 2 ensemble of the B1 backbone `b5_h192_blk6_rf11` (EN-S, 2026-09-23).

Self-contained copy (torch + the already-vendored `ship_b1_models` / `ship_simvp_models`) of
`src/en_packed.py`'s `PackedSimVP1Stage` / `PackedGenericCore`, restricted to the `bmmtf32`
packing EN measured as the only viable one (+0.50 ms vs a single b5 inside the stack's graph):

  * stem conv: one dense conv with K*enc output channels (members share the input);
  * down / up / depthwise convs: grouped convs (groups = K, K*hidden);
  * every 1x1 conv (proj_in / gate / proj_out / mlp1 / mlp2 / fuse / head): one batched GEMM over
    the K members (`torch.baddbmm` at B = 1, `torch.matmul` otherwise) with cuBLAS TF32 enabled
    for the duration of the forward only (the previous flag value is restored);
  * GroupNorm(8, C) per member == GroupNorm(8K, K*C) on the packed tensor (exact identity);
  * output = mean over the K members' heads, in standardised space (the same space a single
    `GenericCore` returns), so `ship_ctxnorm_models.CtxNormWrapper` and every CUDA graph treat it
    as the same black box (`w.ctx_unet(xs, extra) -> (B, T, H, W, 3)`).

The formulas are EN's, unchanged (bit-exact vs `src/en_packed.PackedSimVP1Stage(mode="bmmtf32")`,
checked by `scripts/ens_equiv_vendored.py`). The packed parameters are built at load time from the
members' own `b1_finetune.py` state dicts (fp32), so the shipped `model.pth` is simply

    {"arch": "b5x2_h192_blk6_rf11", "member_states": [state0, state1], "ctx_const": {...}}

(`scripts/ens_pack_b5x2.py`; both members were trained on the same files, so they share one
ctx_const, asserted by the packer).
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import ship_b1_models as _B1  # noqa: E402  (member class SimVP1Stage + build_b5_core)
import lsk_head as _LSK  # noqa: E402  (LSK: LS head-input hook; inert when the LS head is off)

ARCH_B5X2 = "b5x2_h192_blk6_rf11"
ARCH_B5X3 = "b5x3_h192_blk6_rf11"   # K3 (2026-09-24): same packing, K = 3 members


def _cat(ts: Sequence[torch.Tensor], dim: int = 0) -> torch.Tensor:
    return torch.cat([t.detach().float().clone() for t in ts], dim=dim).contiguous()


class PackedSimVP1Stage(nn.Module):
    """K members of `ship_b1_models.SimVP1Stage` (same config) as one packed network
    (bmmtf32 packing). `cores`: the raw `SimVP1Stage` modules (`GenericCore.core`)."""

    def __init__(self, cores: List[nn.Module]):
        super().__init__()
        assert len(cores) >= 1
        self.K = K = len(cores)
        c0 = cores[0]
        self.enc = c0.stem[0].out_channels
        self.hid = c0.down[0].out_channels
        self.n_blocks = len(c0.blocks)
        self.out_ch = c0.head.out_channels
        b0 = c0.blocks[0]
        self.pad1 = b0.dw1.padding[0]
        self.pad2 = b0.dw2.padding[0]
        self.dil2 = b0.dw2.dilation[0]
        for c in cores:
            for gn in (c.stem[1], c.down[1], c.fuse[1]):
                assert gn.num_groups == 8 and abs(gn.eps - 1e-5) < 1e-12
            assert len(c.blocks) == self.n_blocks
            bl0 = c.blocks[0]
            assert bl0.dw1.padding[0] == self.pad1 and bl0.dw2.padding[0] == self.pad2 \
                and bl0.dw2.dilation[0] == self.dil2, "members must share k1/k2/d2"
        rb = lambda n, t: self.register_parameter(n, nn.Parameter(t, requires_grad=False))  # noqa: E731
        rb("stem_w", _cat([c.stem[0].weight for c in cores]))
        rb("stem_b", _cat([c.stem[0].bias for c in cores]))
        rb("stem_gw", _cat([c.stem[1].weight for c in cores]))
        rb("stem_gb", _cat([c.stem[1].bias for c in cores]))
        rb("down_w", _cat([c.down[0].weight for c in cores]))
        rb("down_b", _cat([c.down[0].bias for c in cores]))
        rb("down_gw", _cat([c.down[1].weight for c in cores]))
        rb("down_gb", _cat([c.down[1].bias for c in cores]))
        for i in range(self.n_blocks):
            bl = [c.blocks[i] for c in cores]
            for gname in ("norm1", "norm2"):
                rb(f"b{i}_{gname}_w", _cat([getattr(b, gname).weight for b in bl]))
                rb(f"b{i}_{gname}_b", _cat([getattr(b, gname).bias for b in bl]))
            for cname in ("dw1", "dw2"):
                rb(f"b{i}_{cname}_w", _cat([getattr(b, cname).weight for b in bl]))
                rb(f"b{i}_{cname}_b", _cat([getattr(b, cname).bias for b in bl]))
        rb("up_w", _cat([c.up.weight for c in cores]))
        rb("up_b", _cat([c.up.bias for c in cores]))
        rb("fuse_gw", _cat([c.fuse[1].weight for c in cores]))
        rb("fuse_gb", _cat([c.fuse[1].bias for c in cores]))
        # 1x1 convs as (K, C_out, C_in) GEMM weights + (K, C_out, 1) biases -- exactly EN's
        # `w.reshape(K, kco // K, ci)` of the concatenated conv weight.
        pw = {"fuse": [c.fuse[0] for c in cores], "head": [c.head for c in cores]}
        for i in range(self.n_blocks):
            for n in ("proj_in", "gate", "proj_out", "mlp1", "mlp2"):
                pw[f"b{i}_{n}"] = [getattr(c.blocks[i], n) for c in cores]
        for n, mods in pw.items():
            w = _cat([m.weight for m in mods])
            b = _cat([m.bias for m in mods])
            kco, ci = w.shape[0], w.shape[1]
            rb(f"{n}_W3", w.reshape(K, kco // K, ci).contiguous())
            rb(f"{n}_B3", b.reshape(K, kco // K, 1).contiguous())

    def _gn(self, x, w, b):
        return F.group_norm(x, 8 * self.K, w, b, 1e-5)

    def _pw(self, x, name):
        B, KC, H, W = x.shape
        K = self.K
        W3 = getattr(self, f"{name}_W3")
        B3 = getattr(self, f"{name}_B3")
        if B == 1:
            y = torch.baddbmm(B3, W3, x.reshape(K, KC // K, H * W))
        else:
            y = torch.matmul(W3, x.reshape(B, K, KC // K, H * W)) + B3
        return y.reshape(B, -1, H, W)

    def _block(self, i, x):
        K, hid = self.K, self.hid
        g = lambda n: getattr(self, f"b{i}_{n}")  # noqa: E731
        u = F.gelu(self._pw(self._gn(x, g("norm1_w"), g("norm1_b")), f"b{i}_proj_in"))
        a = F.conv2d(u, g("dw1_w"), g("dw1_b"), padding=self.pad1, groups=K * hid)
        a = F.conv2d(a, g("dw2_w"), g("dw2_b"), padding=self.pad2, dilation=self.dil2, groups=K * hid)
        gt = torch.sigmoid(self._pw(u, f"b{i}_gate"))
        x = x + self._pw(a * gt, f"b{i}_proj_out")
        h = F.gelu(self._pw(self._gn(x, g("norm2_w"), g("norm2_b")), f"b{i}_mlp1"))
        x = x + self._pw(h, f"b{i}_mlp2")
        return x

    def forward_members(self, z: torch.Tensor) -> torch.Tensor:
        K, enc = self.K, self.enc
        B, _, H, W = z.shape
        skip = F.gelu(self._gn(F.conv2d(z, self.stem_w, self.stem_b, padding=1),
                               self.stem_gw, self.stem_gb))
        h = F.gelu(self._gn(F.conv2d(skip, self.down_w, self.down_b, stride=2, padding=1, groups=K),
                            self.down_gw, self.down_gb))
        for i in range(self.n_blocks):
            h = self._block(i, h)
        h = F.conv_transpose2d(h, self.up_w, self.up_b, stride=2, padding=1, groups=K)
        hc = torch.cat([h.view(B, K, enc, H, W), skip.view(B, K, enc, H, W)], dim=2)
        hc = hc.view(B, K * 2 * enc, H, W)
        h = F.gelu(self._gn(self._pw(hc, "fuse"), self.fuse_gw, self.fuse_gb))
        if _LSK.HOOK is not None:
            _LSK.HOOK(h)   # LSK: head input -> phi
        out = self._pw(h, "head")
        return out.view(B, K, self.out_ch, H, W)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            return self.forward_members(z).mean(dim=1)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old


class PackedSimVP1StageFP16(PackedSimVP1Stage):
    """FP16X: the same packed network with an fp16 core (fp16 tensor-core GEMMs with fp32 accumulation,
    `allow_fp16_reduced_precision_reduction` off for the forward).

    The fp32 parameters of the parent stay registered and FIRST (so `CtxNormWrapper` /
    `GraphedCtxNormWrapper`, which take their dtype from `next(parameters())`, keep every ctx-norm
    buffer, accumulator and the de-standardisation in fp32); half copies are non-persistent buffers
    made once at construction. torch's GroupNorm accumulates its statistics in fp32 for half input.

    scope "all"    : z.half() -> stem .. head in half -> one cast of the member heads to fp32 -> mean.
    scope "blocks" : stem / down (fp32, as shipped) -> cast -> the 6 blocks in half -> cast -> up / fuse / head (TF32).
    scope "gemm"   : only the 1x1 GEMMs in half (cast in / out around every GEMM; everything else as shipped).
    """

    def __init__(self, cores: List[nn.Module], scope: str = "all"):
        super().__init__(cores)
        assert scope in ("all", "blocks", "gemm"), scope
        self.scope = scope
        for n, p in list(self.named_parameters()):
            self.register_buffer(n + "_h", p.detach().half().clone(), persistent=False)

    def _hp(self, n):
        return getattr(self, n + "_h")

    def _pw_h(self, x, name):
        B, KC, H, W = x.shape
        K = self.K
        W3 = self._hp(f"{name}_W3")
        B3 = self._hp(f"{name}_B3")
        if B == 1:
            y = torch.baddbmm(B3, W3, x.reshape(K, KC // K, H * W))
        else:
            y = torch.matmul(W3, x.reshape(B, K, KC // K, H * W)) + B3
        return y.reshape(B, -1, H, W)

    def _pw(self, x, name):                      # scope "gemm": cast around every 1x1 GEMM
        if self.scope != "gemm":
            return super()._pw(x, name)
        return self._pw_h(x.half(), name).float()

    def _block_h(self, i, x):
        K, hid = self.K, self.hid
        g = lambda n: self._hp(f"b{i}_{n}")  # noqa: E731
        u = F.gelu(self._pw_h(F.group_norm(x, 8 * K, g("norm1_w"), g("norm1_b"), 1e-5), f"b{i}_proj_in"))
        a = F.conv2d(u, g("dw1_w"), g("dw1_b"), padding=self.pad1, groups=K * hid)
        a = F.conv2d(a, g("dw2_w"), g("dw2_b"), padding=self.pad2, dilation=self.dil2, groups=K * hid)
        gt = torch.sigmoid(self._pw_h(u, f"b{i}_gate"))
        x = x + self._pw_h(a * gt, f"b{i}_proj_out")
        h = F.gelu(self._pw_h(F.group_norm(x, 8 * K, g("norm2_w"), g("norm2_b"), 1e-5), f"b{i}_mlp1"))
        x = x + self._pw_h(h, f"b{i}_mlp2")
        return x

    def _forward_all_h(self, z):
        K, enc = self.K, self.enc
        B, _, H, W = z.shape
        hp = self._hp
        z = z.half()                                                                   # cast 1
        skip = F.gelu(F.group_norm(F.conv2d(z, hp("stem_w"), hp("stem_b"), padding=1),
                                   8 * K, hp("stem_gw"), hp("stem_gb"), 1e-5))
        h = F.gelu(F.group_norm(F.conv2d(skip, hp("down_w"), hp("down_b"), stride=2, padding=1, groups=K),
                                8 * K, hp("down_gw"), hp("down_gb"), 1e-5))
        for i in range(self.n_blocks):
            h = self._block_h(i, h)
        h = F.conv_transpose2d(h, hp("up_w"), hp("up_b"), stride=2, padding=1, groups=K)
        hc = torch.cat([h.view(B, K, enc, H, W), skip.view(B, K, enc, H, W)], dim=2)
        hc = hc.view(B, K * 2 * enc, H, W)
        h = F.gelu(F.group_norm(self._pw_h(hc, "fuse"), 8 * K, hp("fuse_gw"), hp("fuse_gb"), 1e-5))
        if _LSK.HOOK is not None:
            _LSK.HOOK(h)   # LSK: head input -> phi
        out = self._pw_h(h, "head").float()                                          # cast 2
        return out.view(B, K, self.out_ch, H, W)

    def _forward_blocks_h(self, z):
        K, enc = self.K, self.enc
        B, _, H, W = z.shape
        skip = F.gelu(self._gn(F.conv2d(z, self.stem_w, self.stem_b, padding=1),
                               self.stem_gw, self.stem_gb))
        h = F.gelu(self._gn(F.conv2d(skip, self.down_w, self.down_b, stride=2, padding=1, groups=K),
                            self.down_gw, self.down_gb))
        h = h.half()                                                                   # cast 1
        for i in range(self.n_blocks):
            h = self._block_h(i, h)
        h = h.float()                                                                  # cast 2
        h = F.conv_transpose2d(h, self.up_w, self.up_b, stride=2, padding=1, groups=K)
        hc = torch.cat([h.view(B, K, enc, H, W), skip.view(B, K, enc, H, W)], dim=2)
        hc = hc.view(B, K * 2 * enc, H, W)
        h = F.gelu(self._gn(super()._pw(hc, "fuse"), self.fuse_gw, self.fuse_gb))
        if _LSK.HOOK is not None:
            _LSK.HOOK(h)   # LSK: head input -> phi
        out = super()._pw(h, "head")
        return out.view(B, K, self.out_ch, H, W)

    def forward_members(self, z: torch.Tensor) -> torch.Tensor:
        if z.device.type != "cuda":              # CPU-only box: the shipped fp32 packing (fp32 params kept)
            return super().forward_members(z)
        if self.scope == "all":
            return self._forward_all_h(z)
        if self.scope == "blocks":
            return self._forward_blocks_h(z)
        return super().forward_members(z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        m = torch.backends.cuda.matmul
        old = (m.allow_tf32, m.allow_fp16_reduced_precision_reduction)
        m.allow_tf32 = True
        m.allow_fp16_reduced_precision_reduction = False
        try:
            return self.forward_members(z).mean(dim=1)
        finally:
            m.allow_tf32, m.allow_fp16_reduced_precision_reduction = old


def _fold_window(xs: torch.Tensor) -> torch.Tensor:
    b, t, h, w, c = xs.shape
    return xs.permute(0, 1, 4, 2, 3).reshape(b, t * c, h, w)


def _unfold_output(z: torch.Tensor, c_out: int = 3) -> torch.Tensor:
    b, tc, h, w = z.shape
    t = tc // c_out
    return z.reshape(b, t, c_out, h, w).permute(0, 1, 3, 4, 2)


class PackedGenericCore(nn.Module):
    """`GenericCore`-compatible wrapper: `.forward(xs, extra) -> (B, T, H, W, 3)`."""

    def __init__(self, packed: PackedSimVP1Stage):
        super().__init__()
        self.packed = packed

    def forward(self, xs: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        z = torch.cat([_fold_window(xs), extra.permute(0, 3, 1, 2)], dim=1)
        return _unfold_output(self.packed(z))


def build_b5x2_core(member_states: Sequence[dict], fp16_core=False) -> PackedGenericCore:
    """Load each member's `b1_finetune.py` state dict into a fresh `ship_b1_models.build_b5_core()`
    (strict), then pack them. Returns the `ctx_unet` for `CtxNormWrapper`.
    FP16X: `fp16_core` (False = the shipped TF32 packing, bit-exact) -> `PackedSimVP1StageFP16` with that scope
    (True == "all")."""
    assert len(member_states) in (2, 3), len(member_states)   # K3: b5x2 (K = 2) or b5x3 (K = 3)
    cores = []
    for sd in member_states:
        gc = _B1.build_b5_core()
        gc.load_state_dict(sd)
        gc.eval()
        cores.append(gc.core)
    if fp16_core:
        packed = PackedGenericCore(PackedSimVP1StageFP16(cores, "all" if fp16_core is True else str(fp16_core)))
    else:
        packed = PackedGenericCore(PackedSimVP1Stage(cores))
    packed.eval()
    return packed


def build_packed_fp16_core(member_states: Sequence[dict], scope=True) -> PackedGenericCore:
    """FP16X: any K >= 1 (K = 1: a single `b5_h192_blk6_rf11` state run through the packed fp16 core)."""
    assert len(member_states) >= 1
    cores = []
    for sd in member_states:
        gc = _B1.build_b5_core()
        gc.load_state_dict(sd)
        gc.eval()
        cores.append(gc.core)
    packed = PackedGenericCore(PackedSimVP1StageFP16(cores, "all" if scope is True else str(scope)))
    packed.eval()
    return packed
