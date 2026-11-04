"""REC (experiments/rec_recovery): causal regime-change detector + partial state reset for the KD3 whole-step graph.

Problem (experiments/st_stress, experiments/rec_recovery/notes.md): after an abrupt freestream-velocity change INSIDE a
trajectory the pipeline never recovers from a drop, because (diagnosed by state transplant) the ctx-norm accumulator
(cumulative per-pixel mean / std of every input frame since window 0), R_run (plain 1/n running mean of revealed ke),
the bias / spread EMAs (mu, var), the interval-centre moments (M0..M2, 6 second moments) and the energy EMAs
(e_true, e_pred) stay calibrated on the old regime.

Detector (causal: reads only this step's input window and the accumulator of the earlier input windows):
    U_win = mean physical u of this window's 20 input frames (all pixels)                        + eps
    U_run = pixel mean of the ctx-norm running mean m (all input frames since the last reset)       + eps
    fire  = |log(U_win / U_run)| > thr, evaluated as max(q, 1/q) > e^thr, q = U_win / U_run (NaN never fires).
eps = 1e-4 m/s keeps q finite for an all-zero (dropout) window. On all 81 usable train_real files the statistic never
exceeds 0.085 (scripts/rec_stat_scan.py); thr = log 1.5 = 0.405.
Reset flag g = fire OR fired-on-the-previous-step (sticky: a switch that straddles a window is re-reset on the first
clean window). On g (all device-side, no host sync, captured in the whole-step graph):
  * ctx-norm accumulator restarts from this window (S1, S2, n <- this window's sums: `acc = win + acc * keep`);
  * counters: R_run restarts (n_R = 1 -> R = ke of the revealed window), the energy EMAs are replaced by this window's
    (a_e = 1), mu / var / ctr moments / blend statistics restart with a pseudo-count n0 (prior_windows), the ke_flow
    window-fraction feature restarts at 0;
  * the EMA updates of mu / var / ctr moments / blend statistics are SKIPPED on this step (a = 0): their innovation
    compares the new regime with a prediction made from the old one;
  * velocity-scale rescale r = q: mean-type state (mu, M0..M2) x r (two muls), energy-type state (var, the six second
    moments) x r^2 -- folded into the skipped EMA update itself: var <- (1 - a) (var + a P) with a = 0 and (1 - a) := r^2.
When g is false every touched value is multiplied by exactly 1.0 / where-selected unchanged: outputs are bit-identical to
the parent (checked on the normal streams). Windows 0 / 1 (eager / W01 graphs) are not covered.
Per graphed step: 11 small kernels (1 two-row mean, addcmul, div, gt, 2 any, where, 2 muls, r^2, where), no memcpy.
"""
from __future__ import annotations

import math

import torch


class RECState:
    def __init__(self, cfg, model, T, H, W, device):
        self.thr = float(cfg.get("thr", math.log(1.5)))
        self.n0 = float(cfg.get("n0", model.cfg["prior_windows"]))
        eps = float(cfg.get("u_eps", 1e-4))
        f32 = dict(device=device, dtype=torch.float32)
        sd_u = float(model.std_in[0])
        self.E = math.exp(self.thr)
        # MW = [m of the ctx-norm accumulator (written by ctx_body, read by the NEXT step), this window's sum over t]
        self.MW = torch.zeros(2, 1, H, W, 2, **f32)
        self.V = torch.zeros(2, **f32)                                   # [mean m_u, mean window-sum u]
        self.MU2 = torch.full((2,), float(model.mean_in[0]) + eps, **f32)
        self.SIG2 = torch.tensor([sd_u, sd_u / float(T)], **f32)
        self.U = torch.ones(2, **f32)                                    # [U_run, U_win] physical (+ eps)
        # QB6 = [Q00, Q10, 0, Q01, Q11, -]: Q[i, j] = U[i] / U[j] written through a (1, 3)-strided view, so that
        # QB6[1:3] = [q, 0] (q = U_win / U_run) -> ONE where gives [r, keep] = [q, 0] on a reset step, [1, 1] otherwise.
        self.QB6 = torch.zeros(6, **f32)
        self.Qv = self.QB6.as_strided((2, 2), (1, 3))
        self.B5 = torch.zeros(5, device=device, dtype=torch.bool)       # [Q > e^thr (4) | fired last step]
        self.g = torch.zeros((), device=device, dtype=torch.bool)
        self.ONE2 = torch.ones(2, **f32)
        self.KR = torch.ones(2, **f32)                                   # [r, keep]
        self.r, self.keep = self.KR[0], self.KR[1]
        self.T_t = torch.full((), float(T), **f32)
        n0 = self.n0
        f64 = dict(device=device, dtype=torch.float64)
        # CA = [n, n_c, n_b, n_e, kf_n_pre, n_R (post-increment counters) | a_err, a_ctr, a_bld, a_e | 1 - a (4)];
        # the whole-step counter vector IS CA[0:6]. On a reset step ONE where sets counters -> [n0, n0, n0, 0, 0, 1],
        # a -> [0, 0, 0, 1] (skip / skip / skip / replace), (1 - a)[0:2] -> r^2 (the var / second-moment rescale).
        self.CA = torch.zeros(14, **f64)
        self.CA_RST = torch.tensor([n0, n0, n0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0], **f64)
        self.ONE4 = torch.ones(4, **f64)
        self.cnt = self.CA[0:6]
        self.a4 = self.CA[6:10]
        self.om4 = self.CA[10:14]

    def seed(self, base):
        """First graphed step of a trajectory (capturable): no carried flag; m of the accumulator as it stands."""
        self.B5[4].fill_(False)
        torch.mul(base._static_S1, base._static_n.reciprocal(), out=self.MW[0])

    def detect(self):
        """Sets g, r, keep. Called inside ctx_body after this window's sum is in MW[1] and before MW[0] (the previous
        step's m) is overwritten."""
        torch.mean(self.MW[..., 0], dim=(1, 2, 3), out=self.V)
        torch.addcmul(self.MU2, self.V, self.SIG2, out=self.U)
        torch.div(self.U.view(2, 1), self.U.view(1, 2), out=self.Qv)
        torch.gt(self.Qv, self.E, out=self.B5[0:4].view(2, 2))           # NaN -> False
        torch.any(self.B5, dim=0, out=self.g)                            # fire now OR fired last step
        torch.any(self.B5[0:4], dim=0, out=self.B5[4])                   # carried to the next step
        torch.where(self.g, self.QB6[1:3], self.ONE2, out=self.KR)       # [r, keep]

    def rescale_mean(self, tensors):
        for t in tensors:
            t.mul_(self.r)

    def counters_a4(self, prior_n_t, alpha_min_t):
        """== the parent's a4 = max(1 / (cnt[0:4] + prior_n), alpha_min) and 1 - a4, written into CA, then ONE reset
        where over [counters | a | 1 - a]. Returns (a4, om4) views."""
        CA = self.CA
        torch.maximum(torch.reciprocal(CA[0:4] + prior_n_t), alpha_min_t, out=self.a4)
        torch.sub(self.ONE4, self.a4, out=self.om4)
        r = self.KR[0:1].expand(2)
        torch.mul(r, r, out=self.CA_RST[10:12])
        torch.where(self.g, self.CA_RST, CA, out=CA)
        return self.a4, self.om4
