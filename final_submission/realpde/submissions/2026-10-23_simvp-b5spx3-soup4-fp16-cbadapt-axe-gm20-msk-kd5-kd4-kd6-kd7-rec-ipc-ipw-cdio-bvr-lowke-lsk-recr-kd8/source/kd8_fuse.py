"""KD8 (experiments/kd8, 2026-09-26): bit-exact jiterator kernels for the KD6 body's main-stream chains that KD7 left
native (CONFIG "kd8"; requires "kd6_concurrency" + "kd7" with its tier 2). Same rules as kd7_fuse.py: IEEE round-to-
nearest intrinsics only (no FMA contraction), ATen's NaN-propagating clamp / maximum functors, scalar constants as the
exact float32 bits the native kernel receives, logf / __fsqrt_rn (bit-identical to torch.log / sqrt over all 2^32
floats, experiments/kd7). A kernel's output is a fresh tensor (jiterator has no out=): where the native chain ends in a
persistent buffer, the body copies the result into it (a native copy_, exact), off the main stream where possible.

  V   ctx-norm std: sqrt(maximum(S2 * recip - m * m, floor))                          (mul, mul, sub, clamp_min, sqrt)
  EP  e_pred + a * (e_f - e_pred)                                                     (sub, mul, add)
  SF  sqrt(clamp_min(mean, EPS))                                                      (clamp_min, sqrt)
  LG  clamp(log((n + EPS) / (d + EPS)), -LC, LC) for the three backbone rows of the ke-flow log-energy features
      (the native stack / add / stack / add / div / log / clamp over 8 rows; the x-only rows run natively on a side
      stream), three outputs
  E2  constK + c * fluc + new_mu                                                      (kd7 E + the native add)
  MH  (no new kernel) kd7 M reading the fp16 W2-net output directly (drops the .float() copy; the load cast is exact)
  BD  the interval pack: [mean - half; mean + half] + cbw * d  (+ f2 * (upper0 - lower0) on u, v), and the width
      (upper0 - lower0)                                                               (sub, add, sub, addcmul_, addcmul_)
Build: every kernel compiled + self-tested bitwise against the verbatim native chain at the real shapes / layouts on
random draws incl. NaN / +-inf / +-0 / denormals / huge / tiny (kd7_fuse._nasty); any difference or exception -> KD8
off (kd8_error), the KD7 body unchanged."""
from __future__ import annotations

import torch

import kd7_fuse as _K7

_fbits, _PRE = _K7._fbits, _K7._PRE
# ATen maximum (clamp_min with a Tensor bound): NaN in either operand propagates (the first one checked first)
_MAX = "  auto mx = [](float a, float b) -> float { return (a != a) ? a : ((b != b) ? b : fmaxf(a, b)); };\n"


class Fused8:
    KERNELS = ("V", "EP", "SF", "LG", "E2", "BD", "MH")

    def __init__(self, w):
        from torch.cuda.jiterator import _create_jit_fn, _create_multi_output_jit_fn
        import ke_net_flow as KF
        self.eps_f = float(KF.EPS)
        self.lc = float(KF.LOG_CLAMP)
        E, LC = _fbits(self.eps_f), _fbits(self.lc)
        self.code = {}
        self.code["V"] = ("template <typename T> T kd8_V(T s2, T rc, T m, T fl) {\n" + _MAX +
                          "  float v = __fsub_rn(__fmul_rn(s2, rc), __fmul_rn(m, m));\n"
                          "  return __fsqrt_rn(mx(v, fl));\n}")
        self.code["EP"] = ("template <typename T> T kd8_EP(T ep, T a, T ef) {\n"
                           "  return __fadd_rn(ep, __fmul_rn(a, __fsub_rn(ef, ep)));\n}")
        self.code["SF"] = ("template <typename T> T kd8_SF(T m) {\n  " + _PRE +
                           f"  return __fsqrt_rn(cmin(m, {E}));\n}}")
        lgf = (f"  auto lg = [&](float n, float d) -> float {{ return cl(logf(__fdiv_rn(__fadd_rn(n, {E}), "
               f"__fadd_rn(d, {E}))), -{LC}, {LC}); }};\n")
        self.code["LG"] = ("template <typename T> void kd8_LG(T ef, T runx, T ef2, T runf, T ef1, T& o0, T& o2, T& o5) {\n  "
                           + _PRE + lgf +
                           "  o0 = lg(ef, runf);\n  o2 = lg(runx, runf);\n  o5 = lg(ef2, ef1);\n}")
        self.code["E2"] = ("template <typename T> T kd8_E2(T k, T c, T f, T mu) {\n"
                           "  return __fadd_rn(__fadd_rn(k, __fmul_rn(c, f)), mu);\n}")
        # BD: iteration (2, T, H, W, C); rs = row (0 lower, 1 upper), ch = channel flag (0 on u, v; 1 on p)
        self.code["BD"] = ("template <typename T> void kd8_BD(T mean, T oh, T d, T cbw, T f2, T rs, T ch, T& pk, T& wd) {\n"
                           "  float lo = __fsub_rn(mean, oh);\n"
                           "  float up = __fadd_rn(mean, oh);\n"
                           "  float wv = __fsub_rn(up, lo);\n"
                           "  float v = __fadd_rn((rs != 0.0f) ? up : lo, __fmul_rn(cbw, d));\n"
                           "  pk = (ch != 0.0f) ? v : __fadd_rn(v, __fmul_rn(f2, wv));\n"
                           "  wd = wv;\n}")
        self._k = {}
        for k in ("V", "EP", "SF", "E2"):
            self._k[k] = _create_jit_fn(self.code[k])
        self._k["LG"] = _create_multi_output_jit_fn(self.code["LG"], num_outputs=3)
        self._k["BD"] = _create_multi_output_jit_fn(self.code["BD"], num_outputs=2)
        dev = w._dev_in.device
        self.RS = torch.tensor([0.0, 1.0], device=dev, dtype=torch.float32).view(2, 1, 1, 1, 1)
        self.CH = torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=torch.float32).view(1, 1, 1, 1, 3)

    _chk = staticmethod(_K7.Fused._chk)

    def V(self, s2, rc, m, fl):
        self._chk(s2, rc, m, fl)
        return self._k["V"](s2, rc, m, fl)

    def EP(self, ep, a, ef):
        self._chk(ep, a, ef)
        return self._k["EP"](ep, a, ef)

    def SF(self, m):
        self._chk(m)
        return self._k["SF"](m)

    def LG(self, ef, runx, ef2, runf, ef1):
        self._chk(ef, runx, ef2, runf, ef1)
        return self._k["LG"](ef, runx, ef2, runf, ef1)

    def E2(self, k, c, f, mu):
        self._chk(k, c, f, mu)
        return self._k["E2"](k, c, f, mu)

    def BD(self, mean, oh, d, cbw, f2):
        self._chk(mean, oh, d, cbw, f2)
        return self._k["BD"](mean, oh, d, cbw, f2, self.RS, self.CH)

    @staticmethod
    def MH(J7, h, mm16):
        """kd7 M with the fp16 net output as operand (jiterator promotes it to float on load)."""
        if h.dtype != torch.float32 or mm16.dtype != torch.float16 or h.device.type != "cuda" or mm16.device.type != "cuda":
            raise RuntimeError("kd8: MH operands")
        return J7._t2["M"](h, mm16)

    # ---- the verbatim native chains (the parent's lines), for the self-test
    @staticmethod
    def ref_V(s2, rc, m, fl):
        var = s2 * rc - m * m
        var = torch.clamp_min(var, fl)
        return torch.sqrt(var)

    @staticmethod
    def ref_EP(ep, a, ef):
        e = ep.clone()
        d_ep = ef - e
        return torch.add(e, a * d_ep, out=e)

    def ref_SF(self, m):
        return m.clamp_min(self.eps_f).sqrt()

    def ref_LG(self, nums, dens):
        lg = torch.log((torch.stack(nums, 0) + self.eps_f) / (torch.stack(dens, 0) + self.eps_f))
        return torch.clamp(lg, -self.lc, self.lc)

    @staticmethod
    def ref_E2(k, c, f, mu):
        return torch.add(k + c * f, mu)

    @staticmethod
    def ref_BD(mean, oh, d, cbw, f2):
        T_, H, W_, C = mean.shape[1:]
        pack = torch.empty(2, T_, H, W_, C, device=mean.device, dtype=mean.dtype)
        wd = torch.empty(1, T_, H, W_, 2, device=mean.device, dtype=mean.dtype)
        torch.sub(mean, oh, out=pack[0:1])
        torch.add(mean, oh, out=pack[1:2])
        torch.sub(pack[1:2][..., :2], pack[0:1][..., :2], out=wd)
        pack.addcmul_(cbw, d)
        pack[..., :2].addcmul_(f2, wd)
        return pack, wd


def selftest(J, w, rounds: int = 4, seed: int = 20260927):
    """Every KD8 kernel vs its verbatim native chain at the real shapes / operand layouts, `rounds` random draws with
    specials; bitwise equality (NaN payloads included). Raises on any difference."""
    dev = w._dev_in.device
    T, H, W_ = w._dev_in.shape[1], w._dev_in.shape[2], w._dev_in.shape[3]
    Kc = w._br_ad.Kc
    gen = torch.Generator().manual_seed(seed)
    beq = _K7._bits_equal
    n_el = 0
    for r in range(rounds):
        pos = r % 2 == 1
        N = lambda *sh, **kw: _K7._nasty(sh, gen, dev, **kw)  # noqa: E731
        # V: S2 (1,H,W,2), recip 0-dim, m = MW[0] (1,H,W,2) (a row of a (2,1,H,W,2) buffer), floor (2,)
        s2, rc = N(1, H, W_, 2, positive=pos), N(1).view(())
        m = N(2, 1, H, W_, 2)[0]
        fl = N(2, positive=True) if r < 2 else N(2)
        if not beq(J.V(s2, rc, m, fl), J.ref_V(s2, rc, m, fl)):
            raise RuntimeError(f"kd8: self-test V mismatch (round {r})")
        # EP: e_pred (1,1,H,W,3), a 0-dim (a row of a (4,) vector), e_f = var[0:1] of a (2,1,H,W,3)
        ep, a = N(1, 1, H, W_, 3, positive=pos), N(4)[3]
        ef = N(2, 1, H, W_, 3, positive=pos)[0:1]
        if not beq(J.EP(ep, a, ef), J.ref_EP(ep, a, ef)):
            raise RuntimeError(f"kd8: self-test EP mismatch (round {r})")
        # SF: (1,1,1,2) mean
        mm = N(1, 1, 1, 2, positive=pos)
        if not beq(J.SF(mm), J.ref_SF(mm)):
            raise RuntimeError(f"kd8: self-test SF mismatch (round {r})")
        # LG: (1,H,W,2) views ([:, 0, ..., :2] of (1,1,H,W,3) tensors and rows of (1,2,H,W,2))
        v3 = lambda: N(1, 1, H, W_, 3, positive=pos)[:, 0, ..., :2]  # noqa: E731
        fh = N(1, 2, H, W_, 2, positive=pos)
        ef_, runx, runf = v3(), v3(), v3()
        ef1, ef2 = fh[:, 0], fh[:, 1]
        o0, o2, o5 = J.LG(ef_, runx, ef2, runf, ef1)
        ref = J.ref_LG([ef_, runx, ef2], [runf, runf, ef1])
        if not (beq(o0, ref[0]) and beq(o2, ref[1]) and beq(o5, ref[2])):
            raise RuntimeError(f"kd8: self-test LG mismatch (round {r})")
        # E2: constK / c (Kc,1,H,W,3), fluc (Kc,T,H,W,3), new_mu (1,T,H,W,3)
        k_, c_ = N(Kc, 1, H, W_, 3), N(Kc, 1, H, W_, 3, positive=pos)
        f_, mu = N(Kc, T, H, W_, 3), N(1, T, H, W_, 3)
        if not beq(J.E2(k_, c_, f_, mu), J.ref_E2(k_, c_, f_, mu)):
            raise RuntimeError(f"kd8: self-test E2 mismatch (round {r})")
        # BD: mean (1,T,H,W,3), out_half (1,T,H,W,3) (a cat), d (1,T,H,W,3), cbw (1,1,1,1,3), f2 (2,T,1,1,1)
        mean, oh, d = N(1, T, H, W_, 3), N(1, T, H, W_, 3, positive=pos), N(1, T, H, W_, 3)
        cbw = N(1, 1, 1, 1, 3) if r >= 2 else torch.tensor([0.5, 0.25, 0.0], device=dev).view(1, 1, 1, 1, 3)
        f2 = N(2, T, 1, 1, 1)
        pk, wd = J.BD(mean, oh, d, cbw, f2)
        rp, rw = J.ref_BD(mean, oh, d, cbw, f2)
        if not (beq(pk, rp) and beq(wd[0:1][..., :2].contiguous(), rw)):
            raise RuntimeError(f"kd8: self-test BD mismatch (round {r})")
        # MH: kd7 M on (half1[..., :2], the fp16 (2T,1,H,W) net output viewed (1,T,H,W,2)) vs M on its .float()
        J7 = w._kd7j
        half1 = N(1, 1, H, W_, 3, positive=pos)
        mm16 = N(2 * T, 1, H, W_, scale=2.0).half()
        a16 = J.MH(J7, half1[..., :2], mm16.view(2, T, H, W_).permute(1, 2, 3, 0).unsqueeze(0))
        a32 = J7.M(half1[..., :2], mm16.float().view(2, T, H, W_).permute(1, 2, 3, 0).unsqueeze(0))
        if not beq(a16, a32) or not beq(a32, J7.ref_M(half1, mm16.float().view(2, T, H, W_))):
            raise RuntimeError(f"kd8: self-test MH mismatch (round {r})")
        n_el += s2.numel() + ep.numel() + 3 * o0.numel() + f_.numel() + pk.numel()
    torch.cuda.synchronize()
    return dict(rounds=rounds, elements=n_el)


def build(w):
    """Compile + self-test the KD8 kernels for wrapper `w`. Returns (Fused8, stats); raises on any failure."""
    import os
    if not torch.cuda.is_available():
        raise RuntimeError("kd8: CUDA required")
    J7 = getattr(w, "_kd7j", None)
    if J7 is None or not J7.t2:
        raise RuntimeError("kd8: KD7 with tier 2 required")
    old = os.environ.get("USE_PYTORCH_KERNEL_CACHE")
    os.environ["USE_PYTORCH_KERNEL_CACHE"] = "0"
    rng = torch.cuda.get_rng_state()
    try:
        J = Fused8(w)
        stats = selftest(J, w)
    finally:
        torch.cuda.set_rng_state(rng)
        if old is None:
            os.environ.pop("USE_PYTORCH_KERNEL_CACHE", None)
        else:
            os.environ["USE_PYTORCH_KERNEL_CACHE"] = old
    return J, stats
