"""Synthetic stand-ins for the four trained weight files of the final package.

For smoke tests only: the files have the shapes and formats the package's loaders expect
(read off `get_ttt_model`, `_load_prior`, `ke_net_flow.load_ke_flow_net` and
`cond_width_net.CondWidthState`), but every value is random or neutral. Scores computed with
them mean nothing.

    python experiments/tests/dummy_weights.py <out_package_dir>

copies the final source dir to <out_package_dir> and writes model.pth, prior.npz,
ke_net_flow.pt and cond_width_weights.pt next to its submission.py. Needs torch + numpy only
(imports the package's own model modules from the copied source).
"""
from __future__ import annotations

import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ablate  # noqa: E402

WEIGHT_FILES = ("model.pth", "prior.npz", "ke_net_flow.pt", "cond_width_weights.pt")
T, H, W = 20, 32, 64


def write_dummy_weights(pkg_dir: str, seed: int = 0, n_prior_files: int = 6) -> None:
    """Write the four weight files into `pkg_dir` (which must hold the package source)."""
    import numpy as np
    import torch

    sys.path.insert(0, pkg_dir)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        import ship_b1_models as b1          # noqa: PLC0415
        import ship_b5x2_models as b5x2      # noqa: PLC0415
        import ke_net_flow                   # noqa: PLC0415
        import cond_width_net as cw          # noqa: PLC0415
    finally:
        sys.path.remove(pkg_dir)
        sys.dont_write_bytecode = old
        for name in ablate._package_module_names(pkg_dir):     # leave no package modules imported
            sys.modules.pop(name, None)
    g = torch.Generator().manual_seed(seed)

    # model.pth: the packed K = 3 b5 ensemble (scripts/k3_pack_b5xk.py's payload), random init per member.
    states = []
    for k in range(3):
        torch.manual_seed(seed + k)
        states.append({n: v.detach().float().clone() for n, v in b1.build_b5_core().state_dict().items()})
    ctx_const = {"m_mean": [0.0, 0.0], "m_std": [1.0, 1.0], "logs_mean": [0.0, 0.0], "logs_std": [1.0, 1.0]}
    torch.save({"arch": b5x2.ARCH_B5X3, "member_states": states, "ctx_const": ctx_const,
                "members": ["dummy0", "dummy1", "dummy2"]}, os.path.join(pkg_dir, "model.pth"))

    # prior.npz: per training file (Re, AoA) + physical-unit maps (scripts/b1_fit_prior.py's layout).
    rng = np.random.default_rng(seed)
    F = n_prior_files
    np.savez_compressed(
        os.path.join(pkg_dir, "prior.npz"),
        re=rng.uniform(1e4, 3e4, F), aoa=rng.choice([0.0, 5.0, 10.0, 15.0, 20.0], F),
        name=np.array([f"dummy_{i}.h5" for i in range(F)]),
        mu=(0.001 * rng.standard_normal((F, T, H, W, 2))).astype(np.float16),
        var=(1e-4 * (1.0 + rng.random((F, H, W, 2)))).astype(np.float32),
        e_true=(1e-4 * (1.0 + rng.random((F, H, W, 2)))).astype(np.float32),
        e_pred=(1e-4 * (1.0 + rng.random((F, H, W, 2)))).astype(np.float32),
        uc=rng.uniform(0.5, 1.5, F))

    # ke_net_flow.pt: a KeFlowNet whose in_ch matches the feature stack of this cfg (summary on, the
    # graphed paths require it); random weights including the head, so the correction is not zero.
    cfg = {"k_in": 6, "leads": list(ke_net_flow.DEFAULT_LEADS), "summary": True}
    z = lambda *s: torch.rand(*s, generator=g) + 0.1  # noqa: E731
    xw, ffl = torch.randn(1, T, H, W, 2, generator=g), torch.randn(1, T, H, W, 2, generator=g)
    halves = (z(1, H, W, 2), z(1, H, W, 2), z(1, H, W, 2), z(1, H, W, 2))
    feat = ke_net_flow.features(xw, ffl, z(1, H, W, 2), z(1, H, W, 2), z(1, H, W, 2), z(1, H, W, 2),
                                torch.zeros(1, H, W, dtype=torch.bool), torch.zeros(1),
                                k_in=cfg["k_in"], leads=cfg["leads"], summary=True, halves=halves,
                                ex_prev=z(1, H, W, 2), ef_prev=z(1, H, W, 2))
    cfg.update(in_ch=int(feat.shape[1]), width=24, wide=32)
    torch.manual_seed(seed + 10)
    net = ke_net_flow.KeFlowNet(cfg["in_ch"], cfg["width"], cfg["wide"])
    with torch.no_grad():
        net.head.weight.normal_(0.0, 0.01)
    ke_net_flow.save_ke_flow_net(os.path.join(pkg_dir, "ke_net_flow.pt"), net, cfg)

    # cond_width_weights.pt: CondWidthState's checkpoint keys.
    torch.manual_seed(seed + 20)
    wnet = cw.M1WidthNet(hidden=32, n_feat=cw.N_FEAT)
    torch.save({"state_dict": wnet.state_dict(), "hidden": 32, "n_feat": cw.N_FEAT,
                "mu": [0.0] * cw.N_FEAT, "sd": [1.0] * cw.N_FEAT, "alpha": 0.5, "lambda": 1.0,
                "clip_lo": 0.25, "clip_hi": 4.0, "n_px_extra": 5},
               os.path.join(pkg_dir, "cond_width_weights.pt"))


def make_dummy_package(out_dir: str, source_dir: str = ablate.FINAL_SOURCE, seed: int = 0) -> str:
    """Copy the package source to `out_dir` and add synthetic weights. Returns `out_dir`."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree(source_dir, out_dir, ignore=shutil.ignore_patterns("__pycache__"))
    write_dummy_weights(out_dir, seed=seed)
    return out_dir


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(make_dummy_package(os.path.abspath(sys.argv[1])))
