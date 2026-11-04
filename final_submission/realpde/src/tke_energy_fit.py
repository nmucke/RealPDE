"""Fit an energy forecaster with the extra state features, on an HONEST backbone.

The last open route of `notes/research_2026-09-15/tke_analysis.md` item 4: give the
`ke_net_flow`-style energy net two more channel groups -- the FNO forecast's fluctuation
energy resolved in lead time and the input window's shedding phase
(`src/tke_energy_feats.py`) -- and see whether the across-window *tracking* correlation
moves at all. Selection is on `data/real/val` by tracking correlation, not only by the
ke relative L2, because tracking is the number the TKE headroom lives in.

Two differences from `scripts/fit_ke_net_flow.py`, both deliberate:

* the cache is rebuilt with whatever backbone `--ckpt` names, into `--cache`, so the
  features can be those of the held-out (honest) FNO rather than the organizers' one,
  which memorised every local file;
* the fit reports, per epoch, the val tracking correlation per Re regime next to the
  val error, so a net that only sharpens the static map is visibly distinguishable from
  one that tracks.

`data/real/test` is never opened. Usage:

    .venv/bin/python src/tke_energy_fit.py --tag honest-base \
        --ckpt data/checkpoints/honest/fno_heldout_v0.pth --epochs 120
    .venv/bin/python src/tke_energy_fit.py --tag honest-extra --leads --phase ...
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import h5py
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ke_net_flow as K  # noqa: E402
import tke_energy_feats as X  # noqa: E402
from load_baseline import build_model, load_checkpoint_state  # noqa: E402

ALPHA_MIN, PRIOR_N = 0.05, 1.0     # GainMapImprovedModel._a with prior=False
KEYS = ("xw", "ffl", "ex", "ef", "ey", "ex1", "ex2", "ef1", "ef2", "runx", "runf")


def split_files():
    """(path, split) for every usable train / val file, train first, sorted."""
    out = []
    for i, sub in enumerate(("train", "val")):
        for p in sorted(glob.glob(os.path.join(_REPO, "data", "real", sub, "*.h5"))):
            if not p.endswith("7575_0.h5"):        # corrupt, per CLAUDE.md
                out.append((p, i))
    return out


def resolve_ckpt(name: str) -> str:
    if os.path.isfile(name):
        return name
    p = os.path.join(_REPO, name)
    if os.path.isfile(p):
        return p
    for sub in ("sim_real_ft", "sim_pretrain"):
        q = os.path.join(_REPO, "data", "competition", "baseline_checkpoints", sub, name)
        if os.path.isfile(q):
            return q
    raise FileNotFoundError(name)


# --------------------------------------------------------------------------- #
# Stage 1: cache the per-window fields of a chosen backbone.
# --------------------------------------------------------------------------- #
def build_cache(cache: str, ckpt: str, device: str, fno_batch: int = 4) -> None:
    os.makedirs(cache, exist_ok=True)
    files = split_files()
    todo = [(p, s) for p, s in files
            if not os.path.isfile(os.path.join(cache, os.path.basename(p)[:-3] + ".npz"))]
    if not todo:
        print(f"cache complete ({len(files)} files) in {cache}", flush=True)
        return
    mi, mt, si, st = torch.load(
        os.path.join(_REPO, "starting_kit", "example_data", "mean_std_real.pt"),
        weights_only=False)
    si = torch.where(si == 0, torch.ones_like(si), si).float()
    st = torch.where(st == 0, torch.ones_like(st), st).float()
    mi, mt = mi.float(), mt.float()

    base = build_model("fno", device=device).eval()
    base.load_state_dict(load_checkpoint_state(resolve_ckpt(ckpt))[0])

    var = lambda a, d=1: a.var(dim=d, unbiased=False)                 # noqa: E731
    for n, (fp, split) in enumerate(todo):
        with h5py.File(fp, "r") as h:
            u, v = h["u"][:, ::2, ::2], h["v"][:, ::2, ::2]
            re, aoa = float(h["re"][()]), float(h["aoa"][()])
        d = np.stack([u, v, np.zeros_like(u)], -1).astype(np.float32)
        t0s = list(range(0, len(d) - 40 + 1, 20))
        Xw = torch.from_numpy(np.stack([d[t:t + 20] for t in t0s]))
        Y = torch.from_numpy(np.stack([d[t + 20:t + 40] for t in t0s]))
        xn = (Xw - mi) / si
        yn = ((Y - mt) / st)[..., :2]
        with torch.no_grad():
            F = torch.cat([base(xn[i:i + fno_batch].to(device)).cpu()
                           for i in range(0, len(xn), fno_batch)])
        ffl = (F - F.mean(1, keepdim=True))[..., :2]
        xw = xn[..., :2]
        rec = dict(
            xw=xw.numpy().astype(np.float16), ffl=ffl.numpy().astype(np.float16),
            ex=var(xw).numpy(), ef=var(ffl).numpy(), ey=var(yn).numpy(),
            ex1=var(xw[:, :10]).numpy(), ex2=var(xw[:, 10:]).numpy(),
            ef1=var(ffl[:, :10]).numpy(), ef2=var(ffl[:, 10:]).numpy(),
            mask=(Xw[0, ..., :2].abs() < 1e-6).all(-1).all(0).numpy(),
            re=np.float32(re), aoa=np.float32(aoa), split=np.int32(split))
        for src, dst in (("ex", "runx"), ("ef", "runf")):
            e = rec[src]
            r = np.empty_like(e)
            r[0] = e[0]
            for i in range(1, len(e)):
                r[i] = r[i - 1] + max(1.0 / (i + PRIOR_N), ALPHA_MIN) * (e[i] - r[i - 1])
            rec[dst] = r
        np.savez(os.path.join(cache, os.path.basename(fp)[:-3] + ".npz"), **rec)
        print(f"[{n + 1:2d}/{len(todo)}] {os.path.basename(fp):14s} Re {re:6.0f} "
              f"AoA {aoa:4.0f}  {len(t0s)} windows", flush=True)


# --------------------------------------------------------------------------- #
# Stage 2: dataset + features.
# --------------------------------------------------------------------------- #
def load_cache(cache: str):
    parts = {k: [] for k in KEYS + ("mask", "wfrac", "split", "ex_prev", "ef_prev",
                                    "file", "re")}
    for j, (fp, _) in enumerate(split_files()):
        z = np.load(os.path.join(cache, os.path.basename(fp)[:-3] + ".npz"))
        n = len(z["ex"])
        for k in KEYS:
            parts[k].append(torch.from_numpy(z[k]))
        parts["mask"].append(torch.from_numpy(z["mask"])[None].expand(n, -1, -1))
        parts["wfrac"].append(torch.minimum(torch.arange(n, dtype=torch.float32),
                                            torch.tensor(20.0)) / 20.0)
        parts["split"].append(torch.full((n,), int(z["split"]), dtype=torch.long))
        parts["file"].append(torch.full((n,), j, dtype=torch.long))
        parts["re"].append(torch.full((n,), float(z["re"])))
        prev = (torch.arange(n) - 1).clamp_min(0)
        parts["ex_prev"].append(torch.from_numpy(z["ex"])[prev])
        parts["ef_prev"].append(torch.where(
            (torch.arange(n) == 0).view(-1, 1, 1, 1),
            torch.from_numpy(z["ex"]), torch.from_numpy(z["ef"])[prev]))
    d = {k: torch.cat(v) for k, v in parts.items()}
    d["mask"] = d["mask"].contiguous()
    return d


def batch_features(d, idx, cfg, dev, flip=None):
    g = lambda k, f=torch.float32: d[k][idx].to(dev, f, non_blocking=True)  # noqa: E731
    fields = [g("xw"), g("ffl")]
    maps = [g(k) for k in ("ex", "ef", "runx", "runf", "ex1", "ex2", "ef1", "ef2",
                           "ex_prev", "ef_prev", "ey")]
    mask = d["mask"][idx].to(dev)
    if flip is not None:
        f = flip.to(dev)
        sgn = torch.tensor([1.0, -1.0], device=dev)
        fields = [torch.where(f.view(-1, 1, 1, 1, 1), t.flip(2) * sgn, t) for t in fields]
        maps = [torch.where(f.view(-1, 1, 1, 1), t.flip(1), t) for t in maps]
        mask = torch.where(f.view(-1, 1, 1), mask.flip(1), mask)
    xw, ffl = fields
    ex, ef, runx, runf, ex1, ex2, ef1, ef2, ex_prev, ef_prev, ey = maps
    feat = K.features(
        xw, ffl, ex, ef, runx, runf, mask, g("wfrac"),
        k_in=cfg["k_in"], leads=cfg["leads"], summary=cfg["summary"],
        halves=(ex1, ex2, ef1, ef2), ex_prev=ex_prev, ef_prev=ef_prev)
    if cfg["lead_e"] or cfg["phase"]:
        feat = torch.cat([feat, X.extra_features(xw, ffl, ef, mask, cfg["lead_e"],
                                                 cfg["phase"], cfg["n_blocks"],
                                                 cfg["phase_kind"])], 1)
    return feat, K.rule_log_energy(ef, runx, runf), mask, ey


# --------------------------------------------------------------------------- #
# The gate metric, on the cached val windows.
# --------------------------------------------------------------------------- #
def tracking(e_pred, e_true, files, top=0.2):
    """Mean over trajectories of the top-20 % energy-weighted tracking correlation."""
    kp, kt = K.ke_phys(e_pred), K.ke_phys(e_true)
    out = {}
    for f in files.unique().tolist():
        i = (files == f).nonzero().flatten()
        a, b = kp[i], kt[i]
        w = b.mean(0)
        am, bm = a - a.mean(0), b - b.mean(0)
        r = (am * bm).sum(0) / ((am ** 2).sum(0) * (bm ** 2).sum(0) + 1e-30).sqrt()
        sel = w >= torch.quantile(w.flatten(), 1.0 - top)
        out[f] = float((r * w)[sel].sum() / w[sel].sum().clamp_min(1e-30))
    return out


@torch.no_grad()
def evaluate(net, d, idx, cfg, dev, scale=1.0, bs=128):
    errs, e_nets, e_rules, eys = [], [], [], []
    for i in range(0, len(idx), bs):
        feat, log_rule, mask, ey = batch_features(d, idx[i:i + bs], cfg, dev)
        e_net = K.apply_correction(net(feat), log_rule, mask, scale)
        errs.append(torch.stack([K.ke_rel_l2(e_net, ey),
                                 K.ke_rel_l2(log_rule.exp(), ey)], 1))
        e_nets.append(e_net.cpu())
        e_rules.append(log_rule.exp().cpu())
        eys.append(ey.cpu())
    err = torch.cat(errs).mean(0).tolist()
    files = d["file"][idx]
    res = d["re"][idx]
    trk_net = tracking(torch.cat(e_nets), torch.cat(eys), files)
    trk_rule = tracking(torch.cat(e_rules), torch.cat(eys), files)
    hi = [f for f in trk_net if float(res[(files == f).nonzero()[0]]) >= 20000]
    lo = [f for f in trk_net if float(res[(files == f).nonzero()[0]]) < 10000]
    mean = lambda t, ks: float(np.mean([t[k] for k in ks])) if ks else float("nan")  # noqa: E731
    return {"e_net": err[0], "e_rule": err[1],
            "trk_net_hi": mean(trk_net, hi), "trk_rule_hi": mean(trk_rule, hi),
            "trk_net_lo": mean(trk_net, lo), "trk_rule_lo": mean(trk_rule, lo),
            "trk_net_all": mean(trk_net, list(trk_net)),
            "trk_rule_all": mean(trk_rule, list(trk_rule))}


def score(e: float) -> float:
    return 100.0 / (1.0 + 0.5 * e)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="extra")
    ap.add_argument("--out", default="")
    ap.add_argument("--ckpt", default="data/checkpoints/honest/fno_heldout_v0.pth")
    ap.add_argument("--cache", default=os.path.join(_REPO, "data", "cache", "tke_flow_honest"))
    ap.add_argument("--k-in", type=int, default=6)
    ap.add_argument("--leads", default="0,4,9,14,19")
    ap.add_argument("--summary", action="store_true", default=True)
    ap.add_argument("--no-summary", dest="summary", action="store_false")
    ap.add_argument("--lead-e", action="store_true", help="add the lead-resolved energies")
    ap.add_argument("--phase", action="store_true", help="add the shedding-phase features")
    ap.add_argument("--phase-kind", default="probe", choices=["probe", "pod"])
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--wide", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--flip", action="store_true", default=True)
    ap.add_argument("--no-flip", dest="flip", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache-only", action="store_true")
    args = ap.parse_args()

    build_cache(args.cache, args.ckpt, args.device)
    if args.cache_only:
        return

    leads = tuple(int(v) for v in args.leads.split(",") if v.strip())
    cfg = {"k_in": args.k_in, "leads": leads, "summary": bool(args.summary),
           "lead_e": bool(args.lead_e), "phase": bool(args.phase),
           "n_blocks": args.n_blocks, "phase_kind": args.phase_kind,
           "width": args.width, "wide": args.wide}
    cfg["in_ch"] = (K.n_features(args.k_in, len(leads), bool(args.summary))
                    + X.n_extra(cfg["lead_e"], cfg["phase"], args.n_blocks))

    torch.manual_seed(args.seed)
    dev = args.device
    d = load_cache(args.cache)
    tr = (d["split"] == 0).nonzero().flatten()
    va = (d["split"] == 1).nonzero().flatten()
    print(f"[{args.tag}] {len(tr)} train / {len(va)} val windows, {cfg['in_ch']} channels "
          f"(lead_e {cfg['lead_e']}, phase {cfg['phase']}/{cfg['phase_kind']})", flush=True)

    net = K.KeFlowNet(cfg["in_ch"], args.width, args.wide).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    base = evaluate(net, d, va, cfg, dev)      # zero-init head == the rule
    print(f"[{args.tag}] {n_par:,} params; rule val e {base['e_rule']:.4f} "
          f"(tke {score(base['e_rule']):.2f})  trk lo {base['trk_rule_lo']:.3f} "
          f"hi {base['trk_rule_hi']:.3f}", flush=True)

    best = (float("inf"), None, None)
    g = torch.Generator().manual_seed(args.seed)
    for ep in range(1, args.epochs + 1):
        net.train()
        perm = tr[torch.randperm(len(tr), generator=g)]
        tot, nb, t0 = 0.0, 0, time.time()
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            flip = (torch.rand(len(idx), generator=g) < 0.5) if args.flip else None
            feat, log_rule, mask, ey = batch_features(d, idx, cfg, dev, flip)
            e_p = K.apply_correction(net(feat), log_rule, mask)
            loss = K.ke_rel_l2(e_p, ey).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()
        net.eval()
        v = evaluate(net, d, va, cfg, dev)
        star = ""
        if v["e_net"] < best[0]:
            best = (v["e_net"], {k: v[k] for k in v}, {k: t.detach().cpu().clone()
                                                       for k, t in net.state_dict().items()})
            star = " *"
        if ep % 5 == 0 or star or ep == 1:
            print(f"[{args.tag}] ep {ep:3d} train {tot / nb:.4f} val {v['e_net']:.4f} "
                  f"(tke {score(v['e_net']):.2f}) trk lo {v['trk_net_lo']:.3f} "
                  f"hi {v['trk_net_hi']:.3f} all {v['trk_net_all']:.3f} "
                  f"({time.time() - t0:.0f}s){star}", flush=True)

    b = best[1]
    print(f"[{args.tag}] BEST val e {best[0]:.4f} (tke {score(best[0]):.2f}) "
          f"trk lo {b['trk_net_lo']:.3f} hi {b['trk_net_hi']:.3f} all {b['trk_net_all']:.3f} "
          f"| rule e {b['e_rule']:.4f} trk lo {b['trk_rule_lo']:.3f} "
          f"hi {b['trk_rule_hi']:.3f} all {b['trk_rule_all']:.3f}", flush=True)
    if args.out:
        net.load_state_dict(best[2])
        K.save_ke_flow_net(args.out, net, cfg)
        print(f"[{args.tag}] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
