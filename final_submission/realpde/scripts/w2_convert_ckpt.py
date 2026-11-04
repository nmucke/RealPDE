#!/usr/bin/env python3
"""W2 -> packaged `cond_width_net.py` checkpoint converter.

Takes a `scripts/w2_fit.py` checkpoint (arch m1, feats m1: 10 window-level +
2 lead-level features in W2 order, `src/w2_net.M1Net` keys) and writes the
format `submissions/2026-09-29_simvp-s3-bundle-condwidth-graphed/source/
cond_width_net.py::CondWidthState` loads (M1WidthNet keys c1/g1/c2/g2/c3/g3/
out, 12 features in m1_common.FEATURE_NAMES order, mu/sd, alpha, lambda,
clip).  The first conv's input channels and mu/sd are permuted accordingly;
everything else is a key rename.  Only valid for the m1 architecture with
M1's 12 features (the online code computes exactly those).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import m1_common as M  # noqa: E402

W2_TO_M1 = {"log_sigma_pipe": "log_sigma_pipe", "log_input_std": "log_input_std",
            "log_last_minus_mean": "log_last_minus_mean", "log_grad_last": "log_grad_last",
            "log_pred_std": "log_pred_std", "log_local_spatial_std": "log_local_spatial_std",
            "log_grad_meanpred": "log_grad_meanpred", "log_prevwin_err": "log_prevwin_err",
            "log_prevwin_err_ema": "log_prevwin_err_ema",
            "log_prevwin_ratio_to_sigma": "log_prevwin_ratio_to_sigma",
            "log_dpred_lead": "log_dpred_lead", "lead_frac": "lead_frac"}
KEYMAP = {"trunk.net.0": "c1", "trunk.net.1": "g1", "trunk.net.3": "c2", "trunk.net.4": "g2",
          "trunk.net.6": "c3", "trunk.net.7": "g3", "out": "out"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ck = torch.load(REPO / args.inp, map_location="cpu", weights_only=False)
    assert ck["arch"] == "m1" and tuple(ck["dil"]) == (1, 2, 1), ck["arch"]
    names = list(ck["feats_win"]) + list(ck["feats_lead"])
    assert sorted(names) == sorted(M.FEATURE_NAMES), names
    mu = np.concatenate([ck["mu_w"], ck["mu_l"]]).astype(np.float32)
    sd = np.concatenate([ck["sd_w"], ck["sd_l"]]).astype(np.float32)
    perm = [names.index(n) for n in M.FEATURE_NAMES]     # m1 position i <- w2 index perm[i]
    sd_new, mu_new = sd[perm], mu[perm]
    state = {}
    for k, v in ck["state_dict"].items():
        pre, suf = k.rsplit(".", 1)
        state[f"{KEYMAP[pre]}.{suf}"] = v.clone()
    w = state["c1.weight"]
    state["c1.weight"] = w[:, perm].contiguous()
    out = {"state_dict": state, "mu": mu_new, "sd": sd_new, "hidden": int(ck["hidden"]), "n_feat": 12,
           "alpha": float(ck["alpha"]), "lambda": float(ck["lam"]), "feature_names": M.FEATURE_NAMES,
           "n_px_extra": 5, "clip_lo": float(ck["clip"][0]), "clip_hi": float(ck["clip"][1]),
           "w2_target": ck["target"], "w2_source": str(args.inp)}
    torch.save(out, REPO / args.out)
    # self-check: same output on random input through both module definitions
    sys.path.insert(0, str(REPO / "submissions/2026-09-29_simvp-s3-bundle-condwidth-graphed/source"))
    import cond_width_net as CW
    import w2_net as NET
    a = NET.build("m1", len(ck["feats_win"]), len(ck["feats_lead"]), ck["hidden"], tuple(ck["dil"]), 1)
    a.load_state_dict(ck["state_dict"]); a.eval()
    b = CW.M1WidthNet(hidden=ck["hidden"], n_feat=12); b.load_state_dict(state); b.eval()
    raw = torch.randn(1, 12, 32, 64)          # standardised M1-order features
    fw = raw[:, [M.FEATURE_NAMES.index(n) for n in ck["feats_win"]]]
    fl = raw[:, [M.FEATURE_NAMES.index(n) for n in ck["feats_lead"]]][:, None]
    with torch.no_grad():
        ya = a(fw, fl)[0, 0, 0]
        yb = b(raw)[0, 0]
    print(f"self-check max|diff| = {float((ya - yb).abs().max()):.2e}; alpha={out['alpha']} lambda={out['lambda']}")
    print("wrote", REPO / args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
