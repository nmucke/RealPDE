"""KD7 (experiments/kd7, 2026-09-26): bit-exact jiterator fusion of the post-join serial tail of the KD6 body
(CONFIG "kd7"; requires "kd6_concurrency").

KD6 moved everything that does not read the backbone onto side streams; what is left on the main stream after the
backbone is a serial chain of ~210 latency-bound kernels. The longest pure-elementwise runs of it are fused here into
single jiterator (NVRTC) kernels, each written with IEEE round-to-nearest intrinsics (__fmul_rn / __fadd_rn / __fsub_rn
/ __fdiv_rn / __fsqrt_rn -- NVRTC contracts a*b+c into an FMA by default, which is NOT what the native one-op-per-kernel
chain computes), the native clamp functor (NaN passes through, else fminf(fmaxf(v, lo), hi)) and expf (bit-identical to
torch.exp over all 2^32 float inputs, experiments/kd7/scripts/kd7_jit_probe.py). Every scalar constant is embedded as
the exact float32 bit pattern the native kernel receives (Python double -> float, round to nearest).

Fused (kd6_conc.forward_body, main stream; the native lines are kept verbatim in the else-branch):
  A  corr = clamp(kf_net, -1.5, 1.5) [* kf_scale] * KP; e_new = exp(log_rule + corr);
     e_t2 = wv * e_new + omv * R_run[..., :2]                                  (7 kernels -> 1)
     then the native index_select by eidx, written into the [..., :2] slice of a per-step 3-channel buffer E3
     (channel 2 is never used -- the kernel below selects g2 there).
  B  g_flow = (e_t2 + 1e-10) / (e_f + 1e-10) | cat g2[..., 2] | clamp(1/g^2, g^2) | sqrt -> gamma; e_tgt2 =
     gamma^2 e_f; _shape's per-pixel part: gamma_s = clamp(sqrt((e_tgt2 + 1e-10) / (e_z + 1e-10)), 0, cap),
     d_e, c, s = clamp((sqrt(clamp_min(c^2 + e_f d_e, 0)) - c) / (e_f + 1e-10), 0, gain_max)
                                                                                (28 kernels -> 1, two outputs)
  C  fluc = gamma_s * zc + s * fl_f                                              (3 big kernels -> 1)
  D  disc = clamp_min(cFM^2 - e_ach vM + e_ach^2, 0); c = clamp_min((-cFM + sqrt(disc)) / clamp_min(e_ach, 1e-12), 0)
                                                                                (12 kernels -> 1)
  E  constK + c * fluc                                                          (2 big kernels -> 1; the native
     add.out(., new_mu, out=_eb_outK) follows). The third fork's emitted row fluc_k = (c * fluc)[k] is computed there
     as c[k] * fluc[k] (elementwise -> the same values).
Build: every kernel is compiled at construction (NVRTC) and self-tested against the verbatim native op chain on
random inputs at the real shapes, including NaN / +-inf / +-0 / denormals / huge / tiny values: any bitwise
difference (NaN payloads included), any exception or a missing NVRTC -> KD7 off (kd7_error), KD6 body unchanged.
The KD6 graph self-test (vs the parent graph) then covers the fused body end to end."""
from __future__ import annotations

import os
import struct

import torch


def _fbits(v: float) -> str:
    """float32 literal with exactly the bits of float(v) rounded to float32 (as Scalar.to<float>() does)."""
    i = struct.unpack("<i", struct.pack("<f", float(v)))[0]
    return f"__int_as_float({i})"


# NaN-propagating clamp == the native clamp / clamp_min scalar functors (ATen TensorCompare.cu)
_PRE = ("auto cl = [](float v, float lo, float hi) -> float { return (v != v) ? v : fminf(fmaxf(v, lo), hi); };\n"
        "  auto cmin = [](float v, float lo) -> float { return (v != v) ? v : fmaxf(v, lo); };\n")


class Fused:
    """The compiled KD7 kernels of one whole-step wrapper (constants read from its model)."""

    def __init__(self, w):
        from torch.cuda.jiterator import _create_jit_fn, _create_multi_output_jit_fn
        m = w.model
        mega = m._mega
        st = w._kd3s
        s = w._io.shadow
        self.kf_scale = float(st.kf_scale)
        g_max = mega._gain_max
        self.g_lo, self.g_hi = 1.0 / g_max ** 2, g_max ** 2
        self.cap = float(s.cfg["shape_cap"])
        self.gmax_s = float(s.cfg["gain_max"])
        ks = "" if self.kf_scale == 1.0 else f"  corr = __fmul_rn(corr, {_fbits(self.kf_scale)});\n"
        E10 = _fbits(1e-10)
        self.code = {}
        self.code["A"] = (
            "template <typename T> T kd7_A(T net, T kp, T lr, T wv, T omv, T rr) {\n  " + _PRE +
            f"  float corr = cl(net, {_fbits(-1.5)}, {_fbits(1.5)});\n" + ks +
            "  corr = __fmul_rn(corr, kp);\n"
            "  float e = expf(__fadd_rn(lr, corr));\n"
            "  return __fadd_rn(__fmul_rn(wv, e), __fmul_rn(omv, rr));\n}")
        self.code["B"] = (
            "template <typename T> void kd7_B(T e, T ef, T g2, T ch, T ez, T czf, T& gs_out, T& s_out) {\n  " + _PRE +
            f"  float g = (ch != 0.0f) ? g2 : __fdiv_rn(__fadd_rn(e, {E10}), __fadd_rn(ef, {E10}));\n"
            f"  float gam = __fsqrt_rn(cl(g, {_fbits(self.g_lo)}, {_fbits(self.g_hi)}));\n"
            "  float et = __fmul_rn(__fmul_rn(gam, gam), ef);\n"
            f"  float gs = cl(__fsqrt_rn(__fdiv_rn(__fadd_rn(et, {E10}), __fadd_rn(ez, {E10}))), 0.0f, {_fbits(self.cap)});\n"
            "  float de = cmin(__fsub_rn(et, __fmul_rn(__fmul_rn(gs, gs), ez)), 0.0f);\n"
            "  float c = __fmul_rn(gs, czf);\n"
            "  float r = __fsqrt_rn(cmin(__fadd_rn(__fmul_rn(c, c), __fmul_rn(ef, de)), 0.0f));\n"
            f"  float sv = __fdiv_rn(__fsub_rn(r, c), __fadd_rn(ef, {E10}));\n"
            f"  gs_out = gs;\n  s_out = cl(sv, 0.0f, {_fbits(self.gmax_s)});\n}}")
        self.code["C"] = ("template <typename T> T kd7_C(T gs, T zc, T sc, T flf) {\n"
                          "  return __fadd_rn(__fmul_rn(gs, zc), __fmul_rn(sc, flf));\n}")
        self.code["D"] = (
            "template <typename T> T kd7_D(T cfm, T ea, T vm) {\n  " + _PRE +
            "  float disc = cmin(__fadd_rn(__fsub_rn(__fmul_rn(cfm, cfm), __fmul_rn(ea, vm)), __fmul_rn(ea, ea)), 0.0f);\n"
            f"  return cmin(__fdiv_rn(__fadd_rn(-cfm, __fsqrt_rn(disc)), cmin(ea, {_fbits(1e-12)})), 0.0f);\n}}")
        self.code["E"] = ("template <typename T> T kd7_E(T k, T c, T f) {\n"
                          "  return __fadd_rn(k, __fmul_rn(c, f));\n}")
        # ---- tier 2 (the W2 cond-width features / width multiplier and LOWKE; KD5 near path only)
        cw = m.cw
        import cond_width_net as _CW
        self.eps_w = float(_CW.EPS)
        self.alpha, self.lam = float(cw.alpha), float(cw.lam)
        self.clip_lo, self.clip_hi = float(cw.clip_lo), float(cw.clip_hi)
        lk = getattr(w, "_lowke", None)
        self.lk = None
        if lk and float(w._lk_delta) not in (0.0, 1.0, 0.5, -0.5, -1.0, 2.0, 3.0, -2.0):   # native pow's general path
            self.lk = (float(w._lk_theta), float(w._lk_delta), float(w._lk_gmin))
        # P: pred_phys = prev_pred[..., :2] * std_tgt[:2] + mean_tgt[:2]
        self.code["P"] = ("template <typename T> T kd7_P(T a, T s, T mu) {\n"
                          "  return __fadd_rn(__fmul_rn(a, s), mu);\n}")
        # G: _t2_grad_mag after the replicate pad: sqrt(((r - l) wx)^2 + ((d - u) wy)^2) (addcmul value 1)
        self.code["G"] = ("template <typename T> T kd7_G(T r, T l, T wx, T d, T u, T wy) {\n"
                          "  float gx = __fmul_rn(__fsub_rn(r, l), wx);\n"
                          "  float gy = __fmul_rn(__fsub_rn(d, u), wy);\n"
                          "  return __fsqrt_rn(__fadd_rn(__fmul_rn(gx, gx), __fmul_rn(gy, gy)));\n}")
        # F: (log(|a - b| + EPS) - mu5) / sd5   (dp, f5, the INH16 channel-10 normalisation)
        self.code["F"] = ("template <typename T> T kd7_F(T a, T b, T mu, T sd) {\n"
                          f"  float f5 = logf(__fadd_rn(fabsf(__fsub_rn(a, b)), {_fbits(self.eps_w)}));\n"
                          "  return __fdiv_rn(__fsub_rn(f5, mu), sd);\n}")
        # N: (CT - mu10) / sd10
        self.code["N"] = ("template <typename T> T kd7_N(T a, T mu, T sd) {\n"
                          "  return __fdiv_rn(__fsub_rn(a, mu), sd);\n}")
        # M: half1[..., :2] * (clamp(exp(alpha * mm), lo, hi) * lam)
        self.code["M"] = ("template <typename T> T kd7_M(T h, T mm) {\n  " + _PRE +
                          f"  float e = expf(__fmul_rn(mm, {_fbits(self.alpha)}));\n"
                          f"  float mult = __fmul_rn(cl(e, {_fbits(self.clip_lo)}, {_fbits(self.clip_hi)}), {_fbits(self.lam)});\n"
                          "  return __fmul_rn(h, mult);\n}")
        if self.lk is not None:
            th, de, gm = self.lk
            # L: LOWKE gain  G = negchm + clamp(pow(K / (clamp_min(mean K, 1e-20) * theta), delta), gmin, 1) * chm
            self.code["L"] = ("template <typename T> T kd7_L(T k, T mk, T neg, T chm) {\n  " + _PRE +
                              f"  float kb = __fmul_rn(cmin(mk, {_fbits(1e-20)}), {_fbits(th)});\n"
                              f"  float g = cl(powf(__fdiv_rn(k, kb), {_fbits(de)}), {_fbits(gm)}, 1.0f);\n"
                              "  return __fadd_rn(neg, __fmul_rn(g, chm));\n}")
        self._t2 = {k: _create_jit_fn(self.code[k]) for k in ("P", "G", "F", "N", "M", "L") if k in self.code}
        self.t2 = False           # set by selftest() when every tier-2 kernel passed
        self._A = _create_jit_fn(self.code["A"])
        self._B = _create_multi_output_jit_fn(self.code["B"], num_outputs=2)
        self._C = _create_jit_fn(self.code["C"])
        self._D = _create_jit_fn(self.code["D"])
        self._E = _create_jit_fn(self.code["E"])
        # shape of the per-step 3-channel landing buffer of the eidx gather (E3(): a body temporary, channel 2 unused)
        Kc = w._br_ad.Kc
        H, W_ = w._dev_in.shape[2], w._dev_in.shape[3]
        dev = w._dev_in.device
        self.e3_shape = (Kc, 1, H, W_, 3)
        self.CH = torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=torch.float32).view(1, 1, 1, 1, 3)

    def E3(self):
        """uninitialised (Kc, 1, H, W, 3) buffer: [..., :2] receives the eidx gather; channel 2 is never selected by B
        (a per-step temporary of the capturing body, like every other intermediate -- not state)."""
        return torch.empty(self.e3_shape, device=self.CH.device, dtype=torch.float32)

    @staticmethod
    def _chk(*ts):
        for t in ts:
            if t.dtype != torch.float32 or t.device.type != "cuda":
                raise RuntimeError("kd7: fused kernel operand not a float32 CUDA tensor")

    # ---- the fused calls (argument order == the kernels')
    def A(self, net, kp, lr, wv, omv, rr):
        self._chk(net, kp, lr, wv, omv, rr)
        return self._A(net, kp, lr, wv, omv, rr)

    def B(self, e3, ef, g2, ez, czf):
        self._chk(e3, ef, g2, ez, czf)
        return self._B(e3, ef, g2, self.CH, ez, czf)

    def C(self, gs, zc, sc, flf):
        self._chk(gs, zc, sc, flf)
        return self._C(gs, zc, sc, flf)

    def D(self, cfm, ea, vm):
        self._chk(cfm, ea, vm)
        return self._D(cfm, ea, vm)

    def E(self, k, c, f):
        self._chk(k, c, f)
        return self._E(k, c, f)

    def P(self, a, s, mu):
        self._chk(a, s, mu)
        return self._t2["P"](a, s, mu)

    def G(self, r, l_, wx, d, u, wy):
        self._chk(r, l_, wx, d, u, wy)
        return self._t2["G"](r, l_, wx, d, u, wy)

    def F(self, a, b, mu, sd):
        self._chk(a, b, mu, sd)
        return self._t2["F"](a, b, mu, sd)

    def N(self, a, mu, sd):
        self._chk(a, mu, sd)
        return self._t2["N"](a, mu, sd)

    def M(self, h, mm):
        self._chk(h, mm)
        return self._t2["M"](h, mm)

    def L(self, k, mk, neg, chm):
        self._chk(k, mk, neg, chm)
        return self._t2["L"](k, mk, neg, chm)

    def grad_mag(self, f2):
        """== cond_width_net._t2_grad_mag(f2[None])[0] for f2 (b, H, W): replicate pad + one fused kernel."""
        import torch.nn.functional as F_
        import cond_width_net as _CW
        h, w_ = f2.shape[-2], f2.shape[-1]
        x = f2.reshape(-1, 1, h, w_)
        wx, wy = _CW._t2_grad_weights(h, w_, f2.device, f2.dtype)
        p = F_.pad(x, (1, 1, 1, 1), mode="replicate")
        return self.G(p[:, :, 1:-1, 2:], p[:, :, 1:-1, :-2], wx, p[:, :, 2:, 1:-1], p[:, :, :-2, 1:-1], wy).reshape(f2.shape)

    def lowke_apply(self, w):
        """== GraphedWholeStepWrapper._lowke_apply with clamp_min_ / mul_ / div / pow_ / clamp_ / addcmul in kernel L."""
        P = w._dev_out_views[0]
        w._pred_raw.copy_(P)
        var, mean = torch.var_mean(P, dim=1, keepdim=True, unbiased=False)
        K = torch.matmul(var, w._lk_kw)
        G = self.L(K, torch.mean(K), w._lk_negchm, w._lk_chm)
        P.addcmul_(G, torch.sub(P, mean))

    # ---- the verbatim native chains (the parent's lines), for the self-test
    @staticmethod
    def ref_P(a, s, mu):
        return a * s + mu

    @staticmethod
    def ref_G(f2):
        import cond_width_net as _CW
        return _CW._t2_grad_mag(torch.stack([f2], 0))[0]

    def ref_F(self, pred_phys, last, mu5, sd5):
        dp = torch.empty_like(pred_phys)
        dp[:, 0] = pred_phys[:, 0] - last
        dp[:, 1:] = pred_phys[:, 1:] - pred_phys[:, :-1]
        f5 = torch.log(dp.abs() + self.eps_w)
        return torch.div(f5 - mu5, sd5)

    @staticmethod
    def ref_N(CT, mu10, sd10):
        return torch.div(CT - mu10, sd10)

    def ref_M(self, half1, mm):
        mult = torch.exp(self.alpha * mm).clamp(self.clip_lo, self.clip_hi) * self.lam
        mult2 = mult.permute(1, 2, 3, 0).unsqueeze(0)
        return half1[..., :2] * mult2

    def ref_L(self, K, negchm, chm):
        th, de, gm = self.lk
        kb = torch.mean(K).clamp_min_(1e-20).mul_(th)
        g = torch.div(K, kb).pow_(de).clamp_(gm, 1.0)
        return torch.addcmul(negchm, g, chm)

    def ref_A(self, net, kp, lr, wv, omv, rr):
        corr = net.clamp(-1.5, 1.5)
        if self.kf_scale != 1.0:
            corr = corr * self.kf_scale
        corr = corr * kp
        e_new = (lr + corr).exp()
        return wv * e_new + omv * rr

    def ref_B(self, e3, ef, g2, ez, czf):
        Kc = e3.shape[0]
        e_t2 = e3[..., :2]
        g_flow = (e_t2 + 1e-10) / (ef[..., :2] + 1e-10)
        g2c = torch.cat([g_flow, g2[..., 2:].expand(Kc, -1, -1, -1, -1)], dim=-1)
        gamma = g2c.clamp(self.g_lo, self.g_hi).sqrt()
        e_tgt = gamma * gamma * ef
        e_z, e_f, c_zf = ez, ef, czf
        gamma = ((e_tgt + 1e-10) / (e_z + 1e-10)).sqrt().clamp(0.0, self.cap)
        d_e = (e_tgt - gamma * gamma * e_z).clamp_min(0.0)
        c = gamma * c_zf
        s = ((c * c + e_f * d_e).clamp_min(0.0).sqrt() - c) / (e_f + 1e-10)
        return gamma, s.clamp(0.0, self.gmax_s)

    @staticmethod
    def ref_C(gs, zc, sc, flf):
        return gs * zc + sc * flf

    @staticmethod
    def ref_D(cFM, e_tgt_ach, vM):
        disc = (cFM * cFM - e_tgt_ach * vM + e_tgt_ach * e_tgt_ach).clamp_min(0.0)
        return ((-cFM + disc.sqrt()) / e_tgt_ach.clamp_min(1e-12)).clamp_min(0.0)

    @staticmethod
    def ref_E(constK, c, fluc):
        return constK + c * fluc


def _nasty(shape, gen, dev, scale=1.0, positive=False):
    """random float32 tensor: log-uniform magnitudes 1e-30 .. 1e30 around `scale`, random signs, plus exact specials
    (NaN, +-inf, +-0, denormals, FLT_MAX, 1e-10-ish) at random positions."""
    n = 1
    for d in shape:
        n *= d
    mag = torch.exp(torch.empty(n, dtype=torch.float64).uniform_(-69.0, 69.0, generator=gen)) * scale
    core = torch.randn(n, dtype=torch.float64, generator=gen) * scale
    pick = torch.rand(n, generator=gen)
    v = torch.where(pick < 0.5, core, mag)
    if not positive:
        v = v * torch.where(torch.rand(n, generator=gen) < 0.5, -1.0, 1.0)
    v = v.to(torch.float32)
    sp = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1e-40, -1e-40, 3.4e38, -3.4e38, 1e-10,
                       -1e-10, 1e-12, 1.0, -1.0, 2.0, 0.0025, 400.0], dtype=torch.float32)
    k = max(1, n // 50)
    pos = torch.randint(0, n, (k,), generator=gen)
    v[pos] = sp[torch.randint(0, len(sp), (k,), generator=gen)]
    return v.view(shape).to(dev)


def _bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.int32),
                                                                   b.contiguous().view(torch.int32))


def selftest(J, w, rounds: int = 4, seed: int = 20260926):
    """Every fused kernel vs its verbatim native chain, at the real shapes (and the real operand layouts: sliced /
    broadcast views), `rounds` random draws each; bitwise equality required. Raises on any difference."""
    dev = w._dev_in.device
    Kc = w._br_ad.Kc
    K5 = w._io._eb_ad.wv.shape[0]
    T, H, W_ = w._dev_in.shape[1], w._dev_in.shape[2], w._dev_in.shape[3]
    gen = torch.Generator().manual_seed(seed)
    n_el = 0
    for r in range(rounds):
        pos = r % 2 == 1
        N = lambda *sh, **kw: _nasty(sh, gen, dev, **kw)  # noqa: E731
        # A: net (1,1,H,W,2), kp (1,1,H,W,1) in {0,1} (+ specials once), lr, wv / omv (K5,1,1,1,1), R_run[..., :2]
        net = N(1, H, W_, 2).unsqueeze(1)
        kp = (torch.rand(1, 1, H, W_, 1, generator=gen) < 0.7).float().to(dev) if r < 2 else N(1, 1, H, W_, 1)
        lr = N(1, H, W_, 2, scale=3.0).unsqueeze(1)
        wv, omv = N(K5, 1, 1, 1, 1), N(K5, 1, 1, 1, 1)
        rr = N(1, 1, H, W_, 3, positive=pos)[..., :2]
        a1, a0 = J.A(net, kp, lr, wv, omv, rr), J.ref_A(net, kp, lr, wv, omv, rr)
        if not _bits_equal(a1, a0):
            raise RuntimeError(f"kd7: self-test A mismatch (round {r})")
        # B: e3 (Kc,1,H,W,3) (channel 2 junk), ef = var[0:1] view, g2 (1,1,H,W,3), ez, czf
        e3 = N(Kc, 1, H, W_, 3, positive=pos)
        ef = N(2, 1, H, W_, 3, positive=pos)[0:1]
        g2 = N(1, 1, H, W_, 3, positive=pos)
        ez, czf = N(Kc, 1, H, W_, 3, positive=pos), N(Kc, 1, H, W_, 3)
        (b1, b2), (b01, b02) = J.B(e3, ef, g2, ez, czf), J.ref_B(e3, ef, g2, ez, czf)
        if not (_bits_equal(b1, b01) and _bits_equal(b2, b02)):
            raise RuntimeError(f"kd7: self-test B mismatch (round {r})")
        # C: gs, sc (Kc,1,H,W,3), zc (Kc,T,H,W,3), fl_f = slot 0 of a (2,T,H,W,3) pair
        gs, sc = N(Kc, 1, H, W_, 3), N(Kc, 1, H, W_, 3)
        zc = N(Kc, T, H, W_, 3)
        flf = N(2, T, H, W_, 3)[0:1]
        if not _bits_equal(J.C(gs, zc, sc, flf), J.ref_C(gs, zc, sc, flf)):
            raise RuntimeError(f"kd7: self-test C mismatch (round {r})")
        # D: cFM, e_ach (Kc,1,H,W,3), vM = var[1:2] view
        cfm, ea = N(Kc, 1, H, W_, 3), N(Kc, 1, H, W_, 3, positive=pos)
        vm = N(2, 1, H, W_, 3, positive=pos)[1:2]
        if not _bits_equal(J.D(cfm, ea, vm), J.ref_D(cfm, ea, vm)):
            raise RuntimeError(f"kd7: self-test D mismatch (round {r})")
        # E: constK (Kc,1,H,W,3), c (Kc,1,H,W,3), fluc (Kc,T,H,W,3)
        k_, c_, f_ = N(Kc, 1, H, W_, 3), N(Kc, 1, H, W_, 3, positive=pos), N(Kc, T, H, W_, 3)
        if not _bits_equal(J.E(k_, c_, f_), J.ref_E(k_, c_, f_)):
            raise RuntimeError(f"kd7: self-test E mismatch (round {r})")
        n_el += a1.numel() + b1.numel() + zc.numel() + cfm.numel() + f_.numel()
    torch.cuda.synchronize()
    out = dict(rounds=rounds, elements=n_el)
    try:
        out["t2_elements"] = _selftest_t2(J, w, rounds, gen)
        J.t2 = True
    except Exception as e:  # noqa: BLE001 - tier 2 off, tier 1 kept
        J.t2 = False
        out["t2_error"] = repr(e)
    return out


def _selftest_t2(J, w, rounds, gen):
    """Tier-2 kernels vs the verbatim native lines at the real shapes / layouts (bitwise)."""
    dev = w._dev_in.device
    T, H, W_ = w._dev_in.shape[1], w._dev_in.shape[2], w._dev_in.shape[3]
    bW = 2
    n_el = 0
    for r in range(rounds):
        pos = r % 2 == 1
        N = lambda *sh, **kw: _nasty(sh, gen, dev, **kw)  # noqa: E731
        # P: prev_out[..., :2] (a (1,T,H,W,3) slice), std_tgt[:2], mean_tgt[:2]
        a = N(1, T, H, W_, 3)[..., :2]
        sv, mv = N(3)[:2], N(3)[:2]
        if not _bits_equal(J.P(a, sv, mv), J.ref_P(a, sv, mv)):
            raise RuntimeError(f"kd7: self-test P mismatch (round {r})")
        # G: grad-magnitude of a (bW, H, W) field (meanpred: T-mean of a contiguous (bW, T, H, W) buffer)
        f2 = N(bW, T, H, W_, positive=pos).mean(dim=1)
        if not _bits_equal(J.grad_mag(f2), J.ref_G(f2)):
            raise RuntimeError(f"kd7: self-test G mismatch (round {r})")
        # F: pred_phys = permuted (bW, T, H, W) view of a (1, T, H, W, 2) tensor, last (bW, H, W) = a row view
        pp = N(1, T, H, W_, 2).squeeze(0).permute(3, 0, 1, 2)
        last = N(bW, T, H, W_)[:, -1]
        mu5, sd5 = N(1).view(()), N(1, positive=pos).view(())
        prev = torch.cat([last.unsqueeze(1), pp[:, :-1]], dim=1)
        if not _bits_equal(J.F(pp, prev, mu5, sd5), J.ref_F(pp, last, mu5, sd5)):
            raise RuntimeError(f"kd7: self-test F mismatch (round {r})")
        # N: CT (10, bW, H, W), mu10 / sd10 (10, 1, 1, 1)
        CT = N(10, bW, H, W_)
        mu10, sd10 = N(10, 1, 1, 1), N(10, 1, 1, 1, positive=pos)
        if not _bits_equal(J.N(CT, mu10, sd10), J.ref_N(CT, mu10, sd10)):
            raise RuntimeError(f"kd7: self-test N mismatch (round {r})")
        # M: half1 (1,1,H,W,3), mm (2,T,H,W) (a view of the net's float output)
        half1 = N(1, 1, H, W_, 3, positive=pos)
        mm = N(2 * T, 1, H, W_, scale=2.0).view(2, T, H, W_)
        if not _bits_equal(J.M(half1[..., :2], mm.permute(1, 2, 3, 0).unsqueeze(0)), J.ref_M(half1, mm)):
            raise RuntimeError(f"kd7: self-test M mismatch (round {r})")
        # L: K (1,1,H,W,1) (a matmul output), chm / negchm (3,)
        if J.lk is not None:
            K = N(1, 1, H, W_, 1, positive=True if r < 2 else pos)
            chm = torch.tensor([1.0, 1.0, 0.0], device=dev) if r < 2 else N(3)
            neg = -chm
            if not _bits_equal(J.L(K, torch.mean(K), neg, chm), J.ref_L(K, neg, chm)):
                raise RuntimeError(f"kd7: self-test L mismatch (round {r})")
        n_el += a.numel() + f2.numel() + pp.numel() + CT.numel() + mm.numel()
    torch.cuda.synchronize()
    return n_el


def build(w):
    """Compile + self-test the KD7 kernels for wrapper `w`. Returns (Fused, stats); raises on any failure."""
    if not torch.cuda.is_available():
        raise RuntimeError("kd7: CUDA required")
    if w._br_ad is None or w._kd3s is None:
        raise RuntimeError("kd7: AXE br + KD3 path required")
    # no on-disk jiterator kernel cache (~/.cache/torch/kernels): torch reads USE_PYTORCH_KERNEL_CACHE once, at the
    # process's first jiterator compile (the self-test below compiles every kernel), so it is set only around it
    old = os.environ.get("USE_PYTORCH_KERNEL_CACHE")
    os.environ["USE_PYTORCH_KERNEL_CACHE"] = "0"
    rng = torch.cuda.get_rng_state()
    try:
        J = Fused(w)
        stats = selftest(J, w)
    finally:
        torch.cuda.set_rng_state(rng)
        if old is None:
            os.environ.pop("USE_PYTORCH_KERNEL_CACHE", None)
        else:
            os.environ["USE_PYTORCH_KERNEL_CACHE"] = old
    return J, stats
