"""T2 batched cond-width (update + k-table bounds + apply) as ONE capture body
shared by `cond_width_graph.py`'s fallback graph and `cuda_graph_wholestep.py`.
Transcribed from experiments/t2_gpu_compute/scripts/t2_condwidth_patch.py
(`_t2_update_both` / `_t2_apply`, the code the three-fold gate ran); the
k-table half-width lines are verbatim from GraphedCondWidthWrapper._apply_body.
Window >= 2 only (the EMA blend branch is always live).

Args are static buffers: prev_target/ctr_old/mean/x/pred_out (1,T,H,W,3),
var (1,1,H,W,3), prev2/ema2 (2,H,W) persistent (updated in place),
out_lower/out_upper (1,T,H,W,3) written in place."""
import torch


def t2_cw_body(m, cw, k_scale, prev_target, ctr_old, mean, var, x, pred_out,
               prev2, ema2, out_lower, out_upper):
    t2_cw_update(m, cw, prev_target, ctr_old, prev2, ema2)
    t2_cw_apply(m, cw, k_scale, mean, var, x, pred_out, prev2, ema2, out_lower, out_upper)


def t2_cw_update(m, cw, prev_target, ctr_old, prev2, ema2):
    """KD: the update half (previous-window error map + EMA, in place).
    Depends only on (prev_target, ctr_old, prev2, ema2), so the whole-step
    graph runs it at the top of the step (values-preserving reorder)."""
    import cond_width_net as _CW
    pt = (prev_target[..., :2] * m.std_tgt[:2] + m.mean_tgt[:2]).squeeze(0).permute(3, 0, 1, 2)
    ctr = (ctr_old[..., :2] * m.std_tgt[:2] + m.mean_tgt[:2]).squeeze(0).permute(3, 0, 1, 2)
    resid = (pt - ctr).abs()
    scored = _CW._scored(pt, resid.dtype)   # MSK: exact_mask switch
    num = (resid * scored).sum(dim=1)
    den = scored.sum(dim=1)
    numf = _CW._box(num, cw.n_px)
    denf = _CW._box(den, cw.n_px)
    e_all = (numf / denf.clamp_min(1e-6)).clamp_min(0.0)
    ema = cw.rho * ema2 + (1.0 - cw.rho) * e_all
    ema2.copy_(ema)
    prev2.copy_(e_all)
    return scored   # AW: (2,T,H,W) 0/1 mask, reused by cb_adapt (callers may ignore it)


def t2_cw_apply(m, cw, k_scale, mean, var, x, pred_out, prev2, ema2, out_lower, out_upper):
    """KD: the apply half (k-table bounds + width net). Bit-exact restructuring
    vs the original body:
      * the k-table half-width chain runs on var's own (1,1,H,W,3) shape and
        is broadcast afterwards (it was computed on `var.expand_as(mean)`:
        20x the elementwise work for identical values);
      * the 12 net features are stacked ONCE directly in the net's
        channel-first layout (2,T,12,H,W) and normalised there with per-channel
        mu/sd views -- same values as stack(-1) + cat + normalise + permute +
        contiguous, minus four full-size passes;
      * the half-width is assembled by one mul + one cat, the bounds written
        with `out=` straight into out_lower / out_upper."""
    import cond_width_net as _CW
    from submission import _SIGMA_GLOBAL
    std1 = var.sqrt()                                                    # (1,1,H,W,3)
    a = 2.0 * std1 * m.std_tgt / _SIGMA_GLOBAL
    pos = (a.clamp_min(1e-12).log() - m.log_a[0]) / (m.log_a[1] - m.log_a[0])
    k = m.k_tab[pos.round().long().clamp(0, len(m.k_tab) - 1)]
    half1 = k_scale * k * std1                                           # (1,1,H,W,3)
    T, H, W = mean.shape[1], mean.shape[2], mean.shape[3]
    sig_pipe_phys = std1[..., :2] * m.std_tgt[:2]                        # (1,1,H,W,2)
    x_phys = (x[..., :2] * m.std_in[:2] + m.mean_in[:2]).squeeze(0).permute(3, 0, 1, 2)
    pred_phys = (pred_out[..., :2] * m.std_tgt[:2] + m.mean_tgt[:2]).squeeze(0).permute(3, 0, 1, 2)
    sp = sig_pipe_phys.squeeze(0).permute(3, 0, 1, 2)                    # (2,1,H,W)
    # KD (= T2 tier-2 A2): fused feature builder; the three previous-window
    # logs taken at (2,H,W) in one launch and expanded afterwards
    # (log(x).expand() == log(x.expand()) value for value).
    base = _CW._kd_base_feature_list(x_phys, pred_phys, sp.expand(-1, T, -1, -1))
    ratio = (prev2 / sp[:, 0].clamp_min(_CW.EPS)).clamp_min(0.0)
    lp, le, lr = _CW._kd_logs([prev2, ema2, ratio])
    ex = lambda t: t.unsqueeze(1).expand(-1, T, -1, -1)  # noqa: E731
    feats = torch.stack(base + [ex(lp), ex(le), ex(lr)], dim=2)          # (2,T,12,H,W)
    feats = (feats - cw.mu.view(1, 1, -1, 1, 1)) / cw.sd.view(1, 1, -1, 1, 1)
    inp = feats.view(2 * T, -1, H, W)
    # KD-B2 (T2 tier-2 track B2, NOT bit-exact): the width net in fp16 compute
    # on a half-precision COPY of the net made once (the eager fp32 net used by
    # windows 0/1 is untouched); one cast in, one cast out.
    net_h = getattr(cw, "_kd_net_half", None)
    if net_h is None:
        import copy
        net_h = copy.deepcopy(cw.net).half()
        cw._kd_net_half = net_h
    mm = net_h(inp.half()).float().view(2, T, H, W)
    mult = torch.exp(cw.alpha * mm).clamp(cw.clip_lo, cw.clip_hi) * cw.lam
    mult2 = mult.permute(1, 2, 3, 0).unsqueeze(0)                        # (1,T,H,W,2)
    out_half = torch.cat([half1[..., :2] * mult2, half1[..., 2:].expand(1, T, H, W, 1)], dim=-1)
    torch.sub(mean, out_half, out=out_lower)
    torch.add(mean, out_half, out=out_upper)
