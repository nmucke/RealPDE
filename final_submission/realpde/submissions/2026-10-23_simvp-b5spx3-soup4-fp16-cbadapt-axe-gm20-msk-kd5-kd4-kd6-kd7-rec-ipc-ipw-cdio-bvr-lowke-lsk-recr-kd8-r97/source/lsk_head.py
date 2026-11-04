"""LSK (experiments/lsk, 2026-09-26): frozen separable least-squares head correction (CX `sep_svd_I_r8_rho0.9_lam10`)
on the K-member PACKED b5 core, shipping form. CONFIG "lsk_head": None (off: nothing below is ever called, the parent
bit for bit) | dict(r=8, rho=0.9, lam=10.0).

Packed head:  out = mean_k (W_k h_k + b_k) = W_cat H + b,  W_cat = [W_1 .. W_K] / K,  H = the members' concatenated head
inputs (post-fuse GELU, (1, K*C, H, W), member-major).  V = top-r right singular vectors of W_cat's 40 u,v rows;
phi = V^T H (r, P) plus a constant feature.  s = the ctx-norm causal std of the current window (the one that
un-standardises it).  Per channel c (u, v), per revealed window (prev_target y; snapshots of the previous window):
    e = y - rawb_prev ;  Phi_c <- rho Phi_c + sum_p s^2 phi phi^T ;  B_c <- rho B_c + e (s phi)^T
    lam_c = lam_rel tr(Pk_c) / (r + 1) at the first update ;  M_c = B_c (Phi_c + lam_c I)^-1       (fp64)
apply (current window):  raw_uv += (M_c[:, :r] phi + M_c[:, r]) * s_c    on all 20 leads, before every downstream stage.
Window 0 of a trajectory: M == 0 (reset) -> raw + 0 * s == raw exactly.

Split for KD6 (the steady whole-step graph runs this on three streams):
  update(y)     -- reads y + the previous window's snapshots, writes Phi/B/lam/M/cnt + Mh (the apply dtype copy of M).
                   Backbone-independent -> a side stream, concurrently with the ctx-norm + backbone forward.
  [backbone]    -- HOOK(h) writes phi = Vt h (one GEMM, inside the packed forward, right before the head GEMM);
                   every ctx-norm path writes s into S_BUF.
  apply_(raw)   -- 2 kernels on the critical path: corr = baddbmm(Mb, M8, phi); raw_uv.addcmul_(corr, s).
  snapshot(raw) -- prev_phi / prev_s / rawb_prev (= raw_uv - corr * s) for the next update (3 kernels). Off the
                   critical path (KD6: side stream of the third fork; nothing downstream reads them).
All state lives in pre-allocated device buffers updated in place: no host syncs, no Python-side state change per
step, no branches on device values -> the same code runs eagerly, inside W01's captures, in the parent whole-step graph
and in KD6's. reset_state() = two zero_ kernels (fp64 moments + the apply copy of M); the snapshots need no reset (every
update reads the snapshot its own trajectory's previous window wrote). Non-finite revealed pairs are skipped (no moment
update; M kept) so one bad window cannot poison the trajectory.
LSR (LSX arm 1): rec_reset=True -> rec_reset_(keep) multiplies the whole LS state (moments, lam, count) and the apply
copy of M by REC's keep (0 on a regime-change reset step, exactly 1.0 otherwise -> bit-exact when REC does not fire).
"""
from __future__ import annotations

import torch

T, H, W = 20, 32, 64
P = H * W

HOOK = None          # LSKHead._hook: called by the patched packed forwards with the head input h
S_BUF = None         # (1, H, W, 2) fp32: every ctx-norm path writes the current window's causal std here


def packed_head_basis(pk, r: int) -> torch.Tensor:
    """(r, K*C) float64: top-r right singular vectors (rows) of the u,v rows of W_cat = [W_1 .. W_K] / K."""
    W3 = pk.head_W3.detach().double().cpu()                     # (K, 60, C) fp32 master weights
    K, co, C = W3.shape
    Wc = W3.permute(1, 0, 2).reshape(co, K * C) / K             # member-major columns == h's channel order
    rows = [t * 3 + 0 for t in range(T)] + [t * 3 + 1 for t in range(T)]
    _, _, vh = torch.linalg.svd(Wc[rows], full_matrices=False)
    return vh[:r].contiguous()


def find_packed(core):
    pk = getattr(core, "packed", None)
    if pk is None or not hasattr(pk, "head_W3"):
        raise RuntimeError("LSK: packed backbone expected")
    return pk


def write_s(s: torch.Tensor) -> None:
    """ctx-norm paths that compute s as a fresh tensor (eager / GraphedCtxNormWrapper._body): one copy."""
    if S_BUF is not None:
        S_BUF.copy_(s)


class LSKHead:
    def __init__(self, core, r: int = 8, rho: float = 0.9, lam: float = 10.0, device=None, rec_reset: bool = False):
        global HOOK, S_BUF
        pk = find_packed(core)
        dev = torch.device(pk.head_W3.device if device is None else device)
        f64 = torch.float64
        # apply dtype = the head input's dtype: fp16 on the CUDA fp16 core (scope "all"), else fp32
        half = dev.type == "cuda" and getattr(pk, "scope", None) == "all"
        self.adt = adt = torch.float16 if half else torch.float32
        self.r, self.rho, self.lam_rel = int(r), float(rho), float(lam)
        self.rec_reset = bool(rec_reset)
        R = self.R = self.r + 1
        V = packed_head_basis(pk, self.r)
        self.Vt = V.to(device=dev, dtype=adt)                                   # (r, K*C) apply dtype
        # fp64 moments (zeroed at reset): PB = [Phi; B] stacked per channel (2, R+T, R), lam, M, cnt
        sizes = [2 * (R + T) * R, 2, 2 * T * R, 1]
        self._state = torch.zeros(sum(sizes), device=dev, dtype=f64)
        v, o = [], 0
        for n in sizes:
            v.append(self._state[o:o + n]); o += n
        self.PB = v[0].view(2, R + T, R)
        self.Phi = self.PB[:, :R]
        self.B = self.PB[:, R:]
        self.lam = v[1]
        self.M = v[2].view(2, T, R)
        self.cnt = v[3]
        self.Mh = torch.zeros(2, T, R, device=dev, dtype=adt)                  # apply copy of M (zeroed at reset)
        # snapshots of the previous window (fp64; never reset -- always written before they are read)
        self.prev_phi = torch.ones(R, P, device=dev, dtype=f64)                 # row r stays 1 (constant feature)
        self.prev_s = torch.zeros(2, P, device=dev, dtype=f64)
        self.prev_raw = torch.zeros(2, T, P, device=dev, dtype=f64)
        # per-step buffers
        self.phi = torch.zeros(self.r, P, device=dev, dtype=adt)               # HOOK output
        self.corr = torch.zeros(2, T, P, device=dev, dtype=adt)
        self.s_buf = torch.zeros(1, H, W, 2, device=dev, dtype=torch.float32)
        self.Z = torch.zeros(2, R + T, P, device=dev, dtype=f64)               # [s*phi; e] of the revealed window
        self.eye = torch.eye(R, device=dev, dtype=f64)
        self._rho_t = torch.tensor(self.rho, device=dev, dtype=f64)
        self._one_t = torch.tensor(1.0, device=dev, dtype=f64)
        self.base = None
        HOOK = self._hook
        S_BUF = self.s_buf

    def set_base(self, base) -> None:
        self.base = base

    def _hook(self, h: torch.Tensor) -> None:
        hv = h[0].reshape(h.shape[1], P)
        if hv.dtype != self.adt:
            hv = hv.to(self.adt)
        torch.matmul(self.Vt, hv, out=self.phi)

    def reset_state(self) -> None:
        self._state.zero_()
        self.Mh.zero_()

    def rec_reset_(self, keep: torch.Tensor) -> None:
        """REC coupling (LSX arm 1): keep = 0 on a regime-change reset step (LS restarts: M -> 0 for this window,
        moments / lam / count cleared, lam refit from the next revealed window), exactly 1.0 otherwise."""
        self._state.mul_(keep)
        self.Mh.mul_(keep)

    @torch.no_grad()
    def update(self, y: torch.Tensor) -> None:
        """y: (1,T,H,W,3) revealed target of the previous window (target-normalised). ~40 small kernels, no GEMMs:
        fp64 GEMMs with a 2048-long reduction and a 9 x 9 / 20 x 9 output run ~0.7 ms each on the 3090 (1/64-rate
        fp64, one CTA -- measured), so the moments are one fp64 broadcast-multiply + sum over Z = [s phi; e]."""
        R = self.R
        Z = self.Z
        torch.mul(self.prev_phi.unsqueeze(0), self.prev_s.unsqueeze(1), out=Z[:, :R])          # s * phi
        torch.sub(y[0, ..., :2].permute(3, 0, 1, 2), self.prev_raw.view(2, T, H, W),
                  out=Z[:, R:].view(2, T, H, W))                                              # e = y - rawb (fp64)
        PBk = (Z.unsqueeze(2) * Z[:, :R].unsqueeze(1)).sum(-1)                                # (2,R+T,R) = [Pk; Bk]
        ok = torch.isfinite(PBk.sum())                  # finite guard: a non-finite revealed pair is skipped
        PBk = torch.where(ok, PBk, 0.0)
        decay = torch.where(ok, self._rho_t, self._one_t)
        self.PB.mul_(decay).add_(PBk)
        lam_new = torch.diagonal(PBk[:, :R], dim1=-2, dim2=-1).sum(-1) * (self.lam_rel / R)
        self.lam.copy_(torch.where((self.cnt == 0) & ok, lam_new, self.lam))
        A = self.Phi + self.lam.view(2, 1, 1) * self.eye
        X, _info = torch.linalg.solve_ex(A, self.B.transpose(1, 2))                           # (2,R,T)
        self.M.copy_(torch.where(ok & torch.isfinite(X).all(), X.transpose(1, 2), self.M))
        self.Mh.copy_(self.M)
        self.cnt.add_(ok.to(self.cnt.dtype))

    @torch.no_grad()
    def apply_(self, raw: torch.Tensor) -> torch.Tensor:
        """raw: (1,T,H,W,3) BASE backbone output of the current window -> u, v corrected in place (2 kernels)."""
        r = self.r
        torch.baddbmm(self.Mh[:, :, r:], self.Mh[:, :, :r], self.phi.expand(2, r, P), out=self.corr)
        raw[0, ..., :2].addcmul_(self.corr.permute(1, 2, 0).view(T, H, W, 2), self.s_buf[0])
        return raw

    @torch.no_grad()
    def snapshot(self, raw: torch.Tensor) -> None:
        """After apply_: the current window's phi, s and BASE raw (= corrected - corr * s) for the next update."""
        self.prev_phi[:self.r].copy_(self.phi)
        self.prev_s.view(2, H, W).copy_(self.s_buf[0].permute(2, 0, 1))
        torch.addcmul(raw[0, ..., :2], self.corr.permute(1, 2, 0).view(T, H, W, 2), self.s_buf[0], value=-1.0,
                      out=self.prev_raw.view(2, T, H, W).permute(1, 2, 3, 0))                # base raw, fp64

    def step_serial(self, raw: torch.Tensor) -> torch.Tensor:
        """apply + snapshot (the serial paths: eager step(), W01, the parent whole-step body)."""
        self.apply_(raw)
        self.snapshot(raw)
        return raw
