#!/usr/bin/env python3
"""W2 -- turn one out-of-sample stream dump (scripts/lco_run_dump.py .npz,
+ optional W2 aux-map .npz from the patched sub-pipeline) into a compact
feature cache (src/w2_common.py format):

  r (N,T,H,W,2) f32     target - ctr        scored (N,T,H,W,2) bool
  half0 (N,H,W,2) f32   pipeline half-width (verified lead-constant)
  A (N,) f32            per-window accuracy factor (frozen: pred untouched)
  featw (N,2,H,W,Fw) f16  window-level causal features (WIN_FEATS)
  featl (N,2,T,H,W,Fl) f16 lead-level causal features (LEAD_FEATS)
  is_first, kidx, sim_ids

Every feature is a function of quantities available BEFORE the window's
truth is revealed (input window = previous window's target, this window's
prediction / pipeline state, previous windows' residuals).  First-window
"previous residual" features use the de-leaked neutral convention
(m1_common_v2 / ship_condwidth_fit): previous error := pipeline sigma.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import c6s_common as C  # noqa: E402
import m1_common as M  # noqa: E402
import w2_common as W  # noqa: E402

EPS = W.EPS


def prev_error_features(ar_c, scored_c, sp, is_first, n_px=5):
    """(N,H,W,7): log prev err, log ema(0.6), log ratio, log ema(0.3), log ema(0.9),
    log cummean, log cumstd -- all shifted by one window, first window := sigma_pipe."""
    N = ar_c.shape[0]
    e_all = np.maximum(M.emap_lead_reduced(ar_c, scored_c, n_px), 0.0)  # (N,H,W) this window's own
    out = np.empty((N,) + e_all.shape[1:] + (7,), dtype=np.float32)
    ema = {0.6: None, 0.3: None, 0.9: None}
    s1 = s2 = None
    n = 0
    for i in range(N):
        if is_first[i]:
            prev = sp[i]
            for k in ema:
                ema[k] = None
            s1 = s2 = None
            n = 0
        else:
            prev = e_all[i - 1]
        vals = [prev]
        for rho in (0.6, 0.3, 0.9):
            vals.append(sp[i] if ema[rho] is None else ema[rho])
        if n == 0:
            cm, cs = sp[i], sp[i]
        else:
            cm = s1 / n
            cs = np.sqrt(np.maximum(s2 / n - cm * cm, 0.0))
        out[i, ..., 0] = np.log(vals[0] + EPS)
        out[i, ..., 1] = np.log(vals[1] + EPS)
        out[i, ..., 2] = np.log(np.maximum(vals[0] / np.maximum(sp[i], EPS), 0.0) + EPS)
        out[i, ..., 3] = np.log(vals[2] + EPS)
        out[i, ..., 4] = np.log(vals[3] + EPS)
        out[i, ..., 5] = np.log(cm + EPS)
        out[i, ..., 6] = np.log(cs + EPS)
        # update the memories with THIS window's own (now revealed for the next window) error
        for rho in ema:
            ema[rho] = e_all[i] if ema[rho] is None else rho * ema[rho] + (1 - rho) * e_all[i]
        if s1 is None:
            s1, s2 = e_all[i].astype(np.float64).copy(), (e_all[i].astype(np.float64) ** 2)
        else:
            s1 = s1 + e_all[i]
            s2 = s2 + e_all[i].astype(np.float64) ** 2
        n += 1
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def ctx_running_std(x_c, is_first):
    """(N,H,W) cumulative per-pixel temporal std of every frame revealed so far
    (input windows 0..k of the trajectory)."""
    N, T = x_c.shape[:2]
    out = np.empty((N,) + x_c.shape[2:], dtype=np.float32)
    for i in range(N):
        if is_first[i]:
            s1 = np.zeros(x_c.shape[2:], np.float64)
            s2 = np.zeros(x_c.shape[2:], np.float64)
            n = 0
        s1 += x_c[i].sum(0)
        s2 += (x_c[i].astype(np.float64) ** 2).sum(0)
        n += T
        m = s1 / n
        out[i] = np.sqrt(np.maximum(s2 / n - m * m, 0.0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--data-dir", required=True, help="dir of the streamed .h5 files (window-0 inputs)")
    ap.add_argument("--aux", default=None, help="W2 aux-map npz written by the patched sub-pipeline")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()

    d = C.load_dump(REPO / args.dump if not Path(args.dump).is_absolute() else args.dump)
    A = C.abar(d["pred3"], d["target3"]).astype(np.float32)
    del d["pred3"], d["target3"]
    ref = C.sps_from_bounds(A, d["target"], d["lower"], d["upper"])
    print(f"[ref] sps={ref['sps']:.4f} cov={ref['coverage']:.4f} nil={ref['mean_nil']:.4f} "
          f"N={d['target'].shape[0]}", flush=True)
    ctr = (0.5 * (d["lower"].astype(np.float64) + d["upper"].astype(np.float64))).astype(np.float32)
    half0 = (0.5 * (d["upper"].astype(np.float64) - d["lower"].astype(np.float64))).astype(np.float32)
    dev = float(np.abs(half0 - half0[:, :1]).max() / max(float(half0.max()), 1e-12))
    assert dev < 1e-5, f"half0 varies over lead (rel {dev:.2e}); cache assumes lead-pooled bounds"
    half0w = np.ascontiguousarray(half0[:, 0])   # (N,H,W,2)
    scored = d["target"] != 0.0
    r = (d["target"] - ctr).astype(np.float32)
    x = C.inputs_from_dump(d, str(REPO / args.data_dir))
    pred = d["pred"]
    is_first = d["is_first"]
    kidx = C.window_index(is_first)
    N, T, H, W_, Cc = r.shape
    print(f"[load] {N} windows, {len(set(d['sim_ids']))} files, {time.time()-t0:.0f}s", flush=True)

    aux = None
    if args.aux:
        z = np.load(REPO / args.aux if not Path(args.aux).is_absolute() else args.aux)
        aux = {k: z[k] for k in z.files}
        n_aux = aux["gamma"].shape[0]
        assert n_aux == N, f"aux has {n_aux} steps, dump {N}"

    Fw, Fl = len(W.WIN_FEATS), len(W.LEAD_FEATS)
    featw = np.empty((N, Cc, H, W_, Fw), dtype=np.float16)
    featl = np.empty((N, Cc, T, H, W_, Fl), dtype=np.float16)
    from scipy.ndimage import distance_transform_edt
    yy, xx = np.meshgrid(np.arange(H) / (H - 1.0), np.arange(W_) / (W_ - 1.0), indexing="ij")
    for c in range(Cc):
        sp = W.sigma_pipe(half0w[..., c])                       # (N,H,W)
        sp_b = np.broadcast_to(sp[:, None], (N, T, H, W_))
        base = M.causal_feature_maps(x[..., c], pred[..., c], sp_b)   # (N,T,H,W,9)
        fw = np.empty((N, H, W_, Fw), dtype=np.float32)
        fw[..., 0:5] = base[:, 0][..., 0:5]
        fw[..., 5] = base[:, 0, :, :, 7]
        fw[..., 6] = base[:, 0, :, :, 8]
        pe = prev_error_features(np.abs(r[..., c]), scored[..., c], sp, is_first, n_px=5)
        fw[..., 7:10] = pe[..., 0:3]
        fw[..., 10:14] = pe[..., 3:7]
        fw[..., 14] = np.log1p(kidx.astype(np.float32))[:, None, None]
        fw[..., 15] = np.log(ctx_running_std(x[..., c], is_first) + EPS)
        fw[..., 16] = np.log(pred[:, :10, :, :, c].std(1) + EPS)
        fw[..., 17] = np.log(pred[:, 10:, :, :, c].std(1) + EPS)
        fw[..., 18] = xx[None]
        fw[..., 19] = yy[None]
        dist = np.stack([distance_transform_edt(scored[i, 0, :, :, c]) for i in range(N)])
        fw[..., 20] = np.log1p(dist)
        if aux is not None:
            g = aux["gamma"][..., c].astype(np.float32)
            fw[..., 21] = np.log(np.maximum(g, 0) + EPS)
            fw[..., 22] = np.log(np.maximum(aux["e_f"][..., c].astype(np.float32), 0) + EPS)
            fw[..., 23] = np.log(np.maximum(aux["kx"][..., c].astype(np.float32), 0) + EPS)
            et = np.maximum(aux["e_true"][..., c].astype(np.float32), 0) + EPS
            ep = np.maximum(aux["e_pred"][..., c].astype(np.float32), 0) + EPS
            fw[..., 24] = np.log(et / ep)
            mus = np.maximum(aux["mu_std"][..., c].astype(np.float32), 0)
            mu_ok = aux["mu_std_present"]
            fw[..., 25] = np.where(mu_ok[:, None, None], np.log(mus + EPS), np.log(EPS))
            gg = aux["g"][..., c].astype(np.float32)
            fw[..., 26] = np.where(aux["g_present"][:, None, None], gg, 1.0)
            fw[..., 27] = np.log(np.maximum(aux["g2b"][..., c].astype(np.float32), 0) + EPS)
        else:
            fw[..., 21:28] = 0.0
        featw[:, c] = np.nan_to_num(fw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
        fl = np.empty((N, T, H, W_, Fl), dtype=np.float32)
        fl[..., 0] = base[..., 5]
        fl[..., 1] = base[..., 6]
        pm = pred[..., c] - pred[..., c].mean(1, keepdims=True)
        fl[..., 2] = np.log(np.abs(pm) + EPS)
        featl[:, c] = np.nan_to_num(fl, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
        del base, fw, fl
        print(f"[feat] channel {c} done, {time.time()-t0:.0f}s", flush=True)

    # exact-reproduction check of the cache scoring path
    ev = W.SpsEval(r, scored, A, half0w)
    chk = ev.score()
    print(f"[check] cache sps={chk['sps']:.4f} vs dump-path {ref['sps']:.4f} "
          f"(diff {chk['sps']-ref['sps']:+.5f}); cov {chk['coverage']:.4f} vs {ref['coverage']:.4f}", flush=True)
    out = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, r=r, scored=scored, half0=half0w, A=A, is_first=is_first, kidx=kidx,
             sim_ids=np.array(d["sim_ids"]), featw=featw, featl=featl,
             win_names=np.array(W.WIN_FEATS), lead_names=np.array(W.LEAD_FEATS),
             has_aux=np.array(aux is not None), ref_sps=np.array(ref["sps"]),
             ref_cov=np.array(ref["coverage"]))
    print(f"wrote {out} ({out.stat().st_size/1e6:.0f} MB) in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
