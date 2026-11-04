#!/usr/bin/env python3
"""K3: K-generic copy of scripts/ens_pack_b5x2.py (unchanged logic). K = 2 -> arch b5x2_h192_blk6_rf11 (byte-for-byte the
same payload as ens_pack_b5x2.py); K = 3 -> arch b5x3_h192_blk6_rf11 (loaded by experiments/k3_ensemble/src_k3).
    .venv/bin/python scripts/k3_pack_b5xk.py <member0.pth> ... <member{K-1}.pth> <out model.pth>
"""
from __future__ import annotations
import os
import sys
import torch

ARCHS = {2: "b5x2_h192_blk6_rf11", 3: "b5x3_h192_blk6_rf11"}


def main() -> int:
    cks, out = sys.argv[1:-1], sys.argv[-1]
    K = len(cks); arch = ARCHS[K]
    ps = [torch.load(c, map_location="cpu", weights_only=False) for c in cks]
    for p, c in zip(ps, cks):
        assert p.get("model") == "b5_h192_blk6_rf11", (c, p.get("model"))
        assert "model_state_dict" in p and "ctx_const" in p, c
    states = [{k: v.detach().float().clone().contiguous() for k, v in p["model_state_dict"].items()} for p in ps]
    for sd in states[1:]:
        assert set(sd) == set(states[0])
    cc0 = ps[0]["ctx_const"]
    for p in ps[1:]:
        cc1 = p["ctx_const"]; assert set(cc0) == set(cc1)
        for k, v in cc0.items():
            a = torch.as_tensor(v).float(); b = torch.as_tensor(cc1[k]).float()
            assert a.shape == b.shape and torch.allclose(a, b, atol=1e-5, rtol=1e-4), (k, v, cc1[k])
    payload = {"arch": arch, "member_states": states, "ctx_const": cc0, "members": [os.path.basename(c) for c in cks]}
    torch.save(payload, out)
    n = sum(t.numel() for sd in states for t in sd.values())
    print(f"wrote {out}: arch={arch} members={payload['members']} params(total)={n:,} size={os.path.getsize(out) / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
