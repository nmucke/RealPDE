#!/usr/bin/env python3
"""W2 -- fit / evaluate a conditional-width net on out-of-sample feature caches.

  --train  caches pooled for fitting (trajectory-level 20% val split inside)
  --test   ONE cache, touched exactly once at the end
  --exclude-fold F  drop training windows whose file is in F's held-out set
                    (M1's rule when training on other folds' held-out streams)
  --feats  m1 | all | noaux | comma-list of names (src/w2_common WIN_FEATS + LEAD_FEATS)
  --target box5x5 | box7x7 | gauss1.5 | cov5x5 | asym5x5 ...  (LOO oracle regression target)
  --arch   m1 | leadhead      --dil 1,2,1     --hidden 32
  --finetune-sps E   H5: E epochs of scale-free exact-sps fine-tune after the regression

Selection (early stop on val Huber, alpha/lambda by exact val sps, fine-tune
epoch by val sps) is strictly on the training-side validation split.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import w2_common as W  # noqa: E402
import w2_net as NET  # noqa: E402

CACHE_DIR = REPO / "experiments/w2_condwidth_v2/cache"


# --------------------------------------------------------------------------- #
def make_target(kind: str, cache_path: Path, d) -> np.ndarray:
    """(N,T,H,W,2[,2]) float16 log-multiplier target; disk-cached per (kind, cache)."""
    tp = CACHE_DIR / f"tgt_{kind}_{cache_path.stem}.npy"
    if tp.exists():
        return np.load(tp, mmap_mode=None)
    r, sc, h0 = d["r"], d["scored"], d["half0"]
    Cc = r.shape[-1]
    ar = np.abs(r)
    sp5 = np.stack([W.sigma_pipe(h0[..., c]) for c in range(Cc)], -1)[:, None]
    if kind.startswith("box"):
        nl, npx = map(int, kind[3:].split("x"))
        o = np.stack([W.loo_box_mean(ar[..., c], sc[..., c], nl, npx) for c in range(Cc)], -1)
        t = W.logm_from_scale(o, sp5)
    elif kind.startswith("gauss"):
        s = float(kind[5:])
        o = np.stack([W.loo_gauss_mean(ar[..., c], sc[..., c], s, s) for c in range(Cc)], -1)
        t = W.logm_from_scale(o, sp5)
    elif kind.startswith("cov"):
        nl, npx = map(int, kind[3:].split("x"))
        t = np.stack([W.coverage_aware_oracle(ar[..., c], sc[..., c], h0[:, None, :, :, c], nl, npx)
                      for c in range(Cc)], -1)
    elif kind.startswith("asym"):
        nl, npx = map(int, kind[4:].split("x"))
        rp, rn = np.maximum(r, 0), np.maximum(-r, 0)
        ou = np.stack([W.loo_box_mean(rp[..., c], sc[..., c], nl, npx) for c in range(Cc)], -1) * 2
        ol = np.stack([W.loo_box_mean(rn[..., c], sc[..., c], nl, npx) for c in range(Cc)], -1) * 2
        t = np.stack([W.logm_from_scale(ou, sp5), W.logm_from_scale(ol, sp5)], -1)  # (N,T,H,W,2,2)
    else:
        raise ValueError(kind)
    t = t.astype(np.float16)
    tp.parent.mkdir(parents=True, exist_ok=True)   # fresh clone: CACHE_DIR does not exist yet (2026-09-22)
    np.save(tp, t)
    return t


def select_feats(spec: str, win_names, lead_names):
    if spec == "m1":
        w, l = W.M1_WIN, W.M1_LEAD
    elif spec == "all":
        w, l = list(W.WIN_FEATS), list(W.LEAD_FEATS)
    elif spec == "noaux":
        w, l = [n for n in W.WIN_FEATS if n not in W.WIN_FEATS[21:]], list(W.LEAD_FEATS)
    else:
        names = spec.split(",")
        w = [n for n in W.WIN_FEATS if n in names]
        l = [n for n in W.LEAD_FEATS if n in names]
        missing = set(names) - set(w) - set(l)
        if missing:
            raise ValueError(f"unknown features {missing}")
    return [win_names.index(n) for n in w], [lead_names.index(n) for n in l], w, l


def load_pool(paths, target_kind, exclude: set):
    caches = []
    for pi, p in enumerate(paths):
        p = REPO / p if not Path(p).is_absolute() else Path(p)
        d = W.load_cache(p)
        keep = ~np.isin(d["sim_ids"], np.array(sorted(exclude))) if exclude else np.ones(len(d["sim_ids"]), bool)
        tgt = make_target(target_kind, p, d)
        n_drop = len(set(d["sim_ids"][~keep]))
        for k in ("r", "scored", "half0", "A", "is_first", "kidx", "sim_ids", "featw", "featl"):
            d[k] = d[k][keep]
        d["tgt"] = tgt[keep]
        d["traj"] = np.array([f"{pi}:{s}" for s in d["sim_ids"]])
        d["has_aux"] = bool(np.load(p)["has_aux"])
        print(f"[load] {p.name}: {int(keep.sum())} windows kept, {n_drop} files dropped, aux={d['has_aux']}", flush=True)
        caches.append(d)
    out = {}
    for k in ("r", "scored", "half0", "A", "is_first", "kidx", "traj", "featw", "featl", "tgt"):
        out[k] = np.concatenate([c[k] for c in caches], 0)
    out["win_names"], out["lead_names"] = caches[0]["win_names"], caches[0]["lead_names"]
    out["has_aux"] = all(c["has_aux"] for c in caches)
    return out


class Data:
    """CPU-resident (pinned) tensors indexed by (window, channel) pairs."""

    def __init__(self, d, mask, wi, li, mu_w, sd_w, mu_l, sd_l, device):
        import torch
        self.torch = torch
        self.device = device
        fw = d["featw"][mask][..., wi]                     # (N,2,H,W,Fw)
        fl = d["featl"][mask][..., li]                     # (N,2,T,H,W,Fl)
        self.fw = torch.as_tensor(np.ascontiguousarray(fw.transpose(0, 1, 4, 2, 3)))       # (N,2,Fw,H,W) f16
        self.fl = torch.as_tensor(np.ascontiguousarray(fl.transpose(0, 1, 2, 5, 3, 4)))    # (N,2,T,Fl,H,W)
        tgt = d["tgt"][mask]                               # (N,T,H,W,2[,2])
        if tgt.ndim == 5:
            self.tgt = torch.as_tensor(np.ascontiguousarray(tgt.transpose(0, 4, 1, 2, 3)))         # (N,2,T,H,W)
        else:
            self.tgt = torch.as_tensor(np.ascontiguousarray(tgt.transpose(0, 4, 1, 5, 2, 3)))      # (N,2,T,2,H,W)
        self.sc = torch.as_tensor(np.ascontiguousarray(d["scored"][mask].transpose(0, 4, 1, 2, 3)))  # (N,2,T,H,W)
        self.r = torch.as_tensor(np.ascontiguousarray(np.abs(d["r"][mask]).transpose(0, 4, 1, 2, 3)))  # |r| (N,2,T,H,W)
        self.rs = torch.as_tensor(np.ascontiguousarray(d["r"][mask].transpose(0, 4, 1, 2, 3)))       # signed
        self.h0 = torch.as_tensor(np.ascontiguousarray(d["half0"][mask].transpose(0, 3, 1, 2)))      # (N,2,H,W)
        self.A = torch.as_tensor(d["A"][mask])
        self.N = self.fw.shape[0]
        self.mu_w = torch.as_tensor(mu_w, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        self.sd_w = torch.as_tensor(sd_w, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        self.mu_l = torch.as_tensor(mu_l, dtype=torch.float32, device=device).view(1, 1, -1, 1, 1)
        self.sd_l = torch.as_tensor(sd_l, dtype=torch.float32, device=device).view(1, 1, -1, 1, 1)
        self.pairs = np.array([(n, c) for n in range(self.N) for c in range(2)])

    def batch(self, idx):
        n, c = self.pairs[idx, 0], self.pairs[idx, 1]
        dev = self.device
        fw = (self.fw[n, c].to(dev, non_blocking=True).float() - self.mu_w) / self.sd_w
        fl = (self.fl[n, c].to(dev, non_blocking=True).float() - self.mu_l) / self.sd_l
        return (fw, fl, self.tgt[n, c].to(dev).float(), self.sc[n, c].to(dev),
                self.r[n, c].to(dev), self.h0[n, c].to(dev), self.A[n].to(dev), self.rs[n, c].to(dev))


def forward_all(net, data: Data, bs=16):
    """Net output for every (n,c): numpy (N,T,H,W,2[,2]) float32 (5D layout of the caches)."""
    import torch
    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(data.pairs), bs):
            idx = np.arange(i, min(i + bs, len(data.pairs)))
            fw, fl, *_ = data.batch(idx)
            outs.append(net(fw, fl).cpu())
    y = torch.cat(outs, 0).numpy()                          # (N*2, T, n_out, H, W)
    N = data.N
    y = y.reshape(N, 2, y.shape[1], y.shape[2], y.shape[3], y.shape[4])   # (N,2,T,o,H,W)
    y = y.transpose(0, 2, 4, 5, 1, 3)                        # (N,T,H,W,2,o)
    return y[..., 0] if y.shape[-1] == 1 else y


def huber_eval(net, data: Data, bs=16):
    import torch
    import torch.nn.functional as F
    net.eval()
    tot, n = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(data.pairs), bs):
            idx = np.arange(i, min(i + bs, len(data.pairs)))
            fw, fl, tgt, sc, *_ = data.batch(idx)
            y = net(fw, fl)
            y = y[:, :, 0] if tgt.dim() == 4 else y
            l = F.smooth_l1_loss(y, tgt, reduction="none")
            m = sc if tgt.dim() == 4 else sc[:, :, None].expand_as(l)
            tot += float(l[m].sum())
            n += float(m.sum())
    return tot / max(n, 1.0)


def grid_eval(ev: W.SpsEval, y, asym: bool, alphas=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)):
    if asym:
        return ev.grid(y[..., 0], y[..., 1], alphas=alphas)
    return ev.grid(y, alphas=alphas)


def cost_report(net, n_w, n_l, arch, device):
    import torch
    net.eval()
    fw = torch.zeros(2, n_w, 32, 64, device=device)
    fl = torch.zeros(2, 20, n_l, 32, 64, device=device)
    macs = [0]

    def conv_hook(m, i, o):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        macs[0] += o.numel() * k
    hs = [m.register_forward_hook(conv_hook) for m in net.modules() if isinstance(m, torch.nn.Conv2d)]
    with torch.no_grad():
        net(fw, fl)
    for h in hs:
        h.remove()
    if arch == "leadhead":
        # einsum heads: (B,T,C+Fl,H,W)x(T,C+Fl,K) + (B,T,K,H,W)x(T,K,o)
        T, CF, K = net.W1.shape
        macs[0] += 2 * T * 32 * 64 * (CF * K + K * net.W2.shape[-1])
    n_kern = None
    if device == "cuda":
        from torch.profiler import profile, ProfilerActivity
        with torch.no_grad():
            net(fw, fl)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                net(fw, fl)
                torch.cuda.synchronize()
        n_kern = sum(1 for e in prof.events() if e.device_type.name == "CUDA")
        # timing, eager, one window (2 channels x 20 leads)
        ts = []
        for _ in range(30):
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.no_grad():
                net(fw, fl)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        t_ms = 1000 * float(np.median(ts[5:]))
    else:
        t_ms = None
    return {"params": NET.n_params(net), "macs_per_window": int(macs[0]), "cuda_kernels_per_window": n_kern,
            "eager_ms_per_window": t_ms}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--exclude-fold", default=None)
    ap.add_argument("--feats", default="m1")
    ap.add_argument("--target", default="box5x5")
    ap.add_argument("--arch", default="m1", choices=["m1", "leadhead"])
    ap.add_argument("--dil", default="1,2,1")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bw", type=int, default=12, help="(window,channel) pairs per batch (x20 leads)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-frac", type=float, default=0.2)
    ap.add_argument("--finetune-sps", type=int, default=0)
    ap.add_argument("--ft-lr", type=float, default=1e-4)
    ap.add_argument("--ft-tau", type=float, default=0.15, help="soft-indicator width, x half0 (scale-free)")
    ap.add_argument("--ft-last-only", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    t0 = time.time()
    asym = args.target.startswith("asym")
    n_out = 2 if asym else 1

    exclude = W.heldout_files(args.exclude_fold) if args.exclude_fold else set()
    pool = load_pool(args.train, args.target, exclude)
    wi, li, wn, ln = select_feats(args.feats, pool["win_names"], pool["lead_names"])
    if any(n in W.WIN_FEATS[21:] for n in wn) and not pool["has_aux"]:
        raise SystemExit("aux features requested but a training cache has no aux maps")
    tr_mask, val_mask = W.traj_split(pool["traj"], args.val_frac, args.seed)
    print(f"[split] {int(tr_mask.sum())} train / {int(val_mask.sum())} val windows "
          f"({len(set(pool['traj'][tr_mask]))}/{len(set(pool['traj'][val_mask]))} trajectories); "
          f"feats: {len(wn)} window + {len(ln)} lead; target {args.target}; arch {args.arch}", flush=True)

    # standardisation from the TRAIN split only
    fw_tr = pool["featw"][tr_mask][..., wi].astype(np.float32)
    mu_w, sd_w = fw_tr.reshape(-1, len(wi)).mean(0), fw_tr.reshape(-1, len(wi)).std(0) + 1e-6
    del fw_tr
    fl_tr = pool["featl"][tr_mask][:, :, ::5][..., li].astype(np.float32)
    mu_l, sd_l = fl_tr.reshape(-1, len(li)).mean(0), fl_tr.reshape(-1, len(li)).std(0) + 1e-6
    del fl_tr

    Dtr = Data(pool, tr_mask, wi, li, mu_w, sd_w, mu_l, sd_l, device)
    Dval = Data(pool, val_mask, wi, li, mu_w, sd_w, mu_l, sd_l, device)
    ev_val = W.SpsEval(pool["r"][val_mask], pool["scored"][val_mask], pool["A"][val_mask], pool["half0"][val_mask])
    val_ref = ev_val.score()
    print(f"[val ref] sps={val_ref['sps']:.4f} cov={val_ref['coverage']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    dil = tuple(int(x) for x in args.dil.split(","))
    net = NET.build(args.arch, len(wi), len(li), args.hidden, dil, n_out).to(device)
    print(f"[net] {args.arch} params={NET.n_params(net)} dil={dil} hidden={args.hidden}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    best_loss, best_state, best_ep, no_imp = float("inf"), None, -1, 0
    hist = []
    for ep in range(args.epochs):
        net.train()
        perm = np.random.permutation(len(Dtr.pairs))
        tot, nb = 0.0, 0
        for i in range(0, len(perm), args.bw):
            fw, fl, tgt, sc, *_ = Dtr.batch(perm[i:i + args.bw])
            y = net(fw, fl)
            y = y[:, :, 0] if not asym else y
            l = F.smooth_l1_loss(y, tgt, reduction="none")
            m = sc if not asym else sc[:, :, None].expand_as(l)
            loss = (l * m).sum() / m.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        vl = huber_eval(net, Dval)
        tag = ""
        if vl < best_loss:
            best_loss, best_ep, no_imp = vl, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            tag = " *"
        else:
            no_imp += 1
        hist.append((ep, tot / max(nb, 1), vl))
        print(f"[ep {ep:3d}] train_huber={tot/max(nb,1):.5f} val_huber={vl:.5f}{tag}  ({time.time()-t0:.0f}s)", flush=True)
        if no_imp >= args.patience:
            break
    net.load_state_dict(best_state)
    y_val = forward_all(net, Dval)
    best_val, _ = grid_eval(ev_val, y_val, asym)
    alpha, lam = best_val["alpha"], best_val["lam"]
    print(f"[val grid] sps={best_val['sps']:.4f} (ref {val_ref['sps']:.4f}, delta {best_val['sps']-val_ref['sps']:+.4f}) "
          f"alpha={alpha} lam={lam} cov={best_val['coverage']:.4f}", flush=True)
    result = {"tag": args.tag, "args": vars(args), "feats_win": wn, "feats_lead": ln,
              "n_train_windows": int(tr_mask.sum()), "n_val_windows": int(val_mask.sum()),
              "best_epoch": best_ep, "val_best_huber": best_loss, "val_ref": val_ref,
              "val_best": best_val, "val_delta": best_val["sps"] - val_ref["sps"], "alpha": alpha, "lam": lam,
              "net_params": NET.n_params(net)}

    # ---- H5: scale-free exact-sps fine-tune (selection by val sps) ----
    if args.finetune_sps > 0:
        ft_params = list(net.out.parameters()) if (args.ft_last_only and hasattr(net, "out")) else list(net.parameters())
        opt2 = torch.optim.Adam(ft_params, lr=args.ft_lr)
        sig = W.SIGMA_GLOBAL
        best_ft = {"sps": best_val["sps"], "epoch": -1}
        ft_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        ft_hist = []
        for ep in range(args.finetune_sps):
            net.train()
            perm = np.random.permutation(len(Dtr.pairs))
            tot, nb = 0.0, 0
            for i in range(0, len(perm), args.bw):
                fw, fl, tgt, sc, r_abs, h0, A, rs = Dtr.batch(perm[i:i + args.bw])
                y = net(fw, fl)
                h0b = h0[:, None]                                            # (B,1,H,W)
                tau = args.ft_tau * h0b
                if asym:
                    hu = lam * h0b * torch.exp(alpha * y[:, :, 0]).clamp(*W.CLIP)
                    hl = lam * h0b * torch.exp(alpha * y[:, :, 1]).clamp(*W.CLIP)
                    inside = torch.sigmoid((hu - rs) / tau) * torch.sigmoid((rs + hl) / tau)
                else:
                    hu = lam * h0b * torch.exp(alpha * y[:, :, 0]).clamp(*W.CLIP)
                    hl = hu
                    inside = torch.sigmoid((hu - r_abs) / tau)
                util = A.view(-1, 1, 1, 1) * torch.exp(-(hu + hl) / sig) * inside
                loss = -(util * sc).sum() / sc.sum().clamp_min(1.0)
                opt2.zero_grad(set_to_none=True)
                loss.backward()
                opt2.step()
                tot += float(loss)
                nb += 1
            y_val = forward_all(net, Dval)
            res = ev_val.score(y_val[..., 0] if asym else y_val, y_val[..., 1] if asym else None, alpha=alpha, lam=lam)
            tag = ""
            if res["sps"] > best_ft["sps"]:
                best_ft = {"sps": res["sps"], "epoch": ep, "coverage": res["coverage"]}
                ft_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                tag = " *"
            ft_hist.append((ep, tot / max(nb, 1), res["sps"]))
            print(f"[ft {ep:3d}] surrogate={tot/max(nb,1):.5f} val_sps={res['sps']:.4f}{tag}", flush=True)
        net.load_state_dict(ft_state)
        y_val = forward_all(net, Dval)
        best_val2, _ = grid_eval(ev_val, y_val, asym)
        result["finetune"] = {"val_sps_supervised": best_val["sps"], "val_sps_ft_fixed_al": best_ft,
                              "val_best_regrid": best_val2, "hist": ft_hist,
                              "val_gain_vs_supervised": best_val2["sps"] - best_val["sps"]}
        print(f"[ft] val sps supervised {best_val['sps']:.4f} -> fine-tuned (re-grid) {best_val2['sps']:.4f} "
              f"({best_val2['sps']-best_val['sps']:+.4f}); alpha={best_val2['alpha']} lam={best_val2['lam']}", flush=True)
        alpha, lam = best_val2["alpha"], best_val2["lam"]
        result["alpha"], result["lam"] = alpha, lam
    del Dtr

    # ---- cost ----
    result["cost"] = cost_report(net, len(wi), len(li), args.arch, device)
    print(f"[cost] {result['cost']}", flush=True)
    if args.ckpt:
        cp = REPO / args.ckpt
        cp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "arch": args.arch,
                    "dil": dil, "hidden": args.hidden, "feats_win": wn, "feats_lead": ln, "mu_w": mu_w, "sd_w": sd_w,
                    "mu_l": mu_l, "sd_l": sd_l, "alpha": alpha, "lam": lam, "target": args.target,
                    "clip": W.CLIP}, cp)

    # ---- TEST (once) ----
    tp = REPO / args.test if not Path(args.test).is_absolute() else Path(args.test)
    dt = W.load_cache(tp)
    dt["tgt"] = make_target(args.target, tp, dt)
    Dte = Data(dt, np.ones(dt["r"].shape[0], bool), wi, li, mu_w, sd_w, mu_l, sd_l, device)
    ev_te = W.SpsEval(dt["r"], dt["scored"], dt["A"], dt["half0"])
    te_ref = ev_te.score()
    y_te = forward_all(net, Dte)
    te_new = ev_te.score(y_te[..., 0] if asym else y_te, y_te[..., 1] if asym else None, alpha=alpha, lam=lam)
    print(f"\n[TEST {tp.stem}] ref sps={te_ref['sps']:.4f} cov={te_ref['coverage']:.4f} | new sps={te_new['sps']:.4f} "
          f"cov={te_new['coverage']:.4f} delta={te_new['sps']-te_ref['sps']:+.4f}", flush=True)
    # R^2 vs the test's own box5x5 oracle field, per-k deltas
    tgt5 = make_target("box5x5", tp, dt).astype(np.float32)
    sc = dt["scored"]
    yy = (y_te[..., 0] if asym else y_te)[sc]
    tt = tgt5[sc]
    r2 = 1.0 - float(np.sum((tt - yy) ** 2)) / max(float(np.sum((tt - tt.mean()) ** 2)), 1e-12)
    per_k = {}
    k = dt["kidx"]
    for kk in range(int(k.max()) + 1):
        m = k == kk
        if m.sum() == 0:
            continue
        e = W.SpsEval(dt["r"][m], dt["scored"][m], dt["A"][m], dt["half0"][m])
        a = e.score()["sps"]
        b = e.score((y_te[..., 0] if asym else y_te)[m], (y_te[..., 1] if asym else None) if asym else None,
                    alpha=alpha, lam=lam)["sps"]
        per_k[kk] = (int(m.sum()), b - a)
    result.update({"test": tp.stem, "test_ref": te_ref, "test_new": te_new, "test_delta": te_new["sps"] - te_ref["sps"],
                   "test_r2_vs_box5x5": r2, "test_per_k_delta": per_k, "wall_s": time.time() - t0})
    print(f"[test] R^2 vs box5x5 oracle = {r2:.4f}; per-k deltas: " +
          " ".join(f"k{kk}:{v[1]:+.2f}" for kk, v in per_k.items() if kk < 12), flush=True)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
