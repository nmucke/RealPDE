"""c2 KE3-dense: write KE's K-seed energy-net ensemble as ONE standard `ke_net_flow.KeFlowNet` (width K*w, wide K*W).

The K seed nets share the input features, so their ensemble mean is exactly a single wider KeFlowNet:
  stem  (1x1): the K stems stacked along the output channels (no input tiling needed);
  b1 b2 d1 d2 b3 (3x3): block-diagonal (seed i's block at [i*out_g:(i+1)*out_g, i*in_g:(i+1)*in_g]);
  fuse (1x1 on cat([h, d])): h channels are [h_0 .. h_{K-1}], d channels [d_0 .. d_{K-1}] -> seed i's weight columns are split
       into its h part (placed at i*w) and its d part (placed at K*w + i*W);
  head (1x1 -> 2): [W_0 .. W_{K-1}] / K, bias = mean of the K biases  (= the mean of the K raw corrections, as KeFlowNetPacked).
Every other cfg entry is copied from seed 0 (all seeds share it; asserted). The file loads with the unchanged
`ke_net_flow.load_ke_flow_net` / submission.py single-net path, so the shipped sources (KD3 gate included) need no change.
usage: c2_ke_dense.py --out ke_net_flow.pt seed0.pt seed1.pt ... [--check]
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))   # training-side ke_net_flow / kes_ke_flow_packed (same KeFlowNet class + file format as shipped)
import ke_net_flow as K  # noqa: E402
import kes_ke_flow_packed as P  # noqa: E402


def densify(nets, cfg):
    k = len(nets)
    w, W, cin = int(cfg["width"]), int(cfg["wide"]), int(cfg["in_ch"])
    out = K.KeFlowNet(cin, w * k, W * k)
    sd = {n: torch.zeros_like(t) for n, t in out.state_dict().items()}
    for i, n in enumerate(nets):
        s = n.state_dict()
        sd["stem.weight"][i * w:(i + 1) * w] = s["stem.weight"]
        sd["stem.bias"][i * w:(i + 1) * w] = s["stem.bias"]
        for name, og, ig in (("b1", w, w), ("b2", w, w), ("d1", W, w), ("d2", W, W), ("b3", w, w)):
            sd[f"{name}.weight"][i * og:(i + 1) * og, i * ig:(i + 1) * ig] = s[f"{name}.weight"]
            sd[f"{name}.bias"][i * og:(i + 1) * og] = s[f"{name}.bias"]
        fw = s["fuse.weight"]                                   # (w, w + W, 1, 1): [h_i | d_i]
        sd["fuse.weight"][i * w:(i + 1) * w, i * w:(i + 1) * w] = fw[:, :w]
        sd["fuse.weight"][i * w:(i + 1) * w, k * w + i * W:k * w + (i + 1) * W] = fw[:, w:]
        sd["fuse.bias"][i * w:(i + 1) * w] = s["fuse.bias"]
        sd["head.weight"][:, i * w:(i + 1) * w] = s["head.weight"] / k
        sd["head.bias"] += s["head.bias"] / k
    out.load_state_dict(sd)
    c = dict(cfg)
    c["width"], c["wide"] = w * k, W * k
    c["c2_dense_ensemble_k"] = k
    return out.eval(), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("seeds", nargs="+")
    a = ap.parse_args()
    loaded = [K.load_ke_flow_net(p, "cpu") for p in a.seeds]
    nets, cfgs = [n for n, _ in loaded], [c for _, c in loaded]
    for c in cfgs[1:]:
        assert {k: v for k, v in c.items()} == cfgs[0], "seed cfgs differ"
    dense, cfg = densify(nets, cfgs[0])
    K.save_ke_flow_net(a.out, dense, cfg)
    print(f"dense K={len(nets)} width {cfg['width']} wide {cfg['wide']} in_ch {cfg['in_ch']} -> {a.out}")
    if a.check:
        torch.manual_seed(0)
        feat = torch.randn(2, int(cfg["in_ch"]), 32, 64, dtype=torch.float64)
        with torch.no_grad():
            ref = torch.stack([n.double()(feat) for n in nets]).mean(0)
            pk = P.build_packed([n.float() for n in nets], cfgs[0]).double()(feat)
            dn, _ = K.load_ke_flow_net(a.out, "cpu")
            dd = dn.double()(feat)
        print(f"fp64 max|dense - mean of seeds| {(dd - ref).abs().max():.3e}  max|packed - mean| {(pk - ref).abs().max():.3e}"
              f"  (|ref| max {ref.abs().max():.3f})")
        rel = float((dd - ref).abs().max() / ref.abs().max().clamp_min(1e-12))
        if not rel < 1e-6:   # FR review C1: a construction mismatch must fail the chain (set -e), not just print
            raise SystemExit(f"FATAL: dense ke != mean of the {len(nets)} seed nets (rel {rel:.3e} >= 1e-6)")


if __name__ == "__main__":
    main()
