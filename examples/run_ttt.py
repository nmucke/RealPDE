#!/usr/bin/env python3
"""Stream trajectories through the final package's test-time pipeline, window by window.

    python examples/run_ttt.py                                   # shipped weights, kit example data
    python examples/run_ttt.py --data DIR --package PKG --device cuda
    CUDA_VISIBLE_DEVICES= python examples/run_ttt.py --device cpu

This is the Track 2 interface as the evaluation harness drives it: `get_ttt_model(pkg, device)`
once, then per trajectory `reset_ttt_state()` and, per window (stride 20: input frames
[t, t+20), target frames [t+20, t+40)), `ttt_step(input_norm, prev_target_norm)`, where
prev_target_norm is the previous window's target (None on the first window). The stream and the
normalisation are the starting kit's own (`build_stream` and `Normalizer` from
final_submission/realpde/starting_kit/local_eval.py, imported unchanged). Per window it prints the
relative L2 error over u, v, the fraction of u, v target values inside [lower, upper] and the mean
interval width (in units of the competition's sigma_global), and the time of the call.

DIR holds `test_real/*.h5` (kit layout), `train_real/*.h5` (training layout; in-sample for the
shipped weights) or `.h5` files directly; `mean_std_real.pt` is taken from DIR if present, else
from the kit. Without --package, the package is assembled from weights/final_retrain_a100/ by
experiments/make_package.py (hash-checked, cached in build/).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from time import perf_counter

sys.dont_write_bytecode = True                      # keep the package dir and the kit free of __pycache__
REPO = Path(__file__).resolve().parent.parent
KIT = REPO / "final_submission" / "realpde" / "starting_kit"
SIGMA_GLOBAL = 0.0563870259                         # scoring.py's frozen interval-width scale


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                         # the package's own modules do `from submission import ...`
    spec.loader.exec_module(mod)
    return mod


def kit_layout(data: Path, tmp: str) -> Path:
    """A dir with test_real/*.h5 and mean_std_real.pt, as build_stream / Normalizer expect (symlinks)."""
    for sub in ("test_real", "train_real"):
        if (data / sub).is_dir():
            src = data / sub
            break
    else:
        src = data
    stats = data / "mean_std_real.pt"
    if src == data / "test_real" and stats.is_file():
        return data
    out = Path(tmp)
    (out / "test_real").mkdir()
    for f in sorted(src.glob("*.h5")):
        if f.name != "7575_0.h5":                    # duplicate of 6300_0.h5 in the release; skipped by name
            (out / "test_real" / f.name).symlink_to(f.resolve())
    (out / "mean_std_real.pt").symlink_to((stats if stats.is_file() else KIT / "example_data" / "mean_std_real.pt").resolve())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", help="package dir (default: assembled from the shipped weights)")
    ap.add_argument("--data", default=str(KIT / "example_data"), help="trajectory dir (default: kit example data)")
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    args = ap.parse_args()

    import numpy as np
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.package:
        pkg = Path(args.package).resolve()
    else:
        sys.path.insert(0, str(REPO / "experiments"))
        import make_package
        pkg = Path(make_package.make_package(quiet=True))
    kit = load_module("rpde_kit_local_eval", KIT / "local_eval.py")

    sys.path.insert(0, str(pkg))
    submission = load_module("submission", pkg / "submission.py")
    t0 = perf_counter()
    model = submission.get_ttt_model(str(pkg), device)
    print("package %s\nget_ttt_model on %s: %.1f s (untimed by the harness)" % (pkg, device, perf_counter() - t0))

    with tempfile.TemporaryDirectory() as tmp:
        data = kit_layout(Path(args.data).resolve(), tmp)
        stream = kit.build_stream(data)
        norm = kit.Normalizer(data / "mean_std_real.pt")

    times, prev_target = [], None
    print("%-14s %5s %8s %9s %10s %8s" % ("file", "t", "rel_l2", "coverage", "width/sg", "ms"))
    for step in stream:
        inp, tgt = step["input"].unsqueeze(0), step["target"].unsqueeze(0)      # (1, 20, 32, 64, 3)
        inp_norm, tgt_norm = norm.preprocess(inp, tgt)
        t_start = perf_counter()
        if step["is_first"]:                       # new trajectory; the reset is timed with this step
            model.reset_ttt_state()
            prev_target = None
        pred_norm, info = model.ttt_step(inp_norm, prev_target)
        times.append(perf_counter() - t_start)
        prev_target = tgt_norm                     # revealed after the step, used by the next window

        pred = norm.postprocess_pred(torch.as_tensor(pred_norm)).numpy()[0, ..., :2]
        t = tgt.numpy()[0, ..., :2]
        rel = np.linalg.norm(pred - t) / max(np.linalg.norm(t), 1e-8)
        cov = width = float("nan")
        if "lower" in info:
            lo = norm.postprocess_pred(torch.as_tensor(info["lower"])).numpy()[0, ..., :2]
            up = norm.postprocess_pred(torch.as_tensor(info["upper"])).numpy()[0, ..., :2]
            if not (np.isfinite(lo).all() and np.isfinite(up).all() and (lo <= up).all()):
                raise SystemExit("invalid bounds at %s t=%d" % (step["sim_id"], step["time_id"]))
            cov = float(((lo <= t) & (t <= up)).mean())
            width = float((up - lo).mean() / SIGMA_GLOBAL)
        if not np.isfinite(pred).all():
            raise SystemExit("non-finite prediction at %s t=%d" % (step["sim_id"], step["time_id"]))
        print("%-14s %5d %8.4f %9.3f %10.3f %8.1f"
              % (step["sim_id"], step["time_id"], rel, cov, width, 1e3 * times[-1]))
    print("%d windows, mean %.1f ms/step (incl. resets)" % (len(times), 1e3 * sum(times) / len(times)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
