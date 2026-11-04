#!/usr/bin/env python3
"""Stream any experiment dir's submission.py over any data dir and dump raw
arrays for metric analysis -- an LCO-parameterised generalisation of
`experiments/phase0_metrics/run_dump.py` (which hardcodes `data/real/<split>`
and a fixed backbone map). Read-only w.r.t. the repo: imports
`src/local_eval_with_analytics.run_stream` (the same loop `local_eval.py`
uses) and writes one .npz per run to --out.

Usage:
    .venv/bin/python scripts/lco_run_dump.py \\
        --exp-dir experiments/lco_baseline/fold0 \\
        --data-dir data/real_lco/fold0/heldout \\
        --ckpt data/checkpoints/lco/fold0_runB.pth \\
        --out experiments/lco_baseline/runs/fold0_heldout.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def make_shim(data_dir: Path, tmp: Path) -> Path:
    """Stage a `test_real/` + `mean_std_real.pt` layout `local_eval.build_stream`
    expects, from an arbitrary directory of (possibly symlinked) .h5 files."""
    d = tmp / "test_real"
    d.mkdir(parents=True)
    for f in sorted(Path(data_dir).glob("*.h5")):
        (d / f.name).symlink_to(f.resolve())
    (tmp / "mean_std_real.pt").symlink_to(
        (REPO / "starting_kit" / "example_data" / "mean_std_real.pt").resolve())
    return tmp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-dir", required=True, help="dir with submission.py (+ weights)")
    ap.add_argument("--data-dir", required=True, help="dir of .h5 files (symlinks fine)")
    ap.add_argument("--ckpt", default=None,
                     help="sets RPDE_BACKBONE_CKPT for the submission's _resolve_ckpt; "
                          "omit to use whatever the submission's own CONFIG/model.pth resolves")
    ap.add_argument("--out", required=True)
    # NOTE: src/local_eval_with_analytics.py's Normalizer (mean_std_real.pt tensors) is never
    # moved to `device` -- only the model is. On any experiment whose ttt_step returns its
    # prediction on `device` (every one checked here: the bare frozen-forward raw baseline and
    # experiments/integrated's GainMapImprovedModel both return cuda tensors when built with
    # device="cuda"), `Normalizer.postprocess_pred` then multiplies a cuda pred by cpu stats and
    # raises "Expected all tensors to be on the same device". Reproduced even on the *existing*
    # experiments/phase0_metrics/run_dump.py with --device cuda, so this is a pre-existing gap in
    # the shared harness, not something new here -- CPU is the only device that works with it
    # today. Timing is not measured in this task anyway, so default to cpu.
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mem-frac", type=float, default=0.15)
    args = ap.parse_args()

    if args.ckpt:
        os.environ["RPDE_BACKBONE_CKPT"] = args.ckpt
    else:
        os.environ.pop("RPDE_BACKBONE_CKPT", None)

    import torch
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)

    import local_eval_with_analytics as A

    with tempfile.TemporaryDirectory() as td:
        data = make_shim(Path(args.data_dir), Path(td))
        run = A.run_stream(Path(args.exp_dir).resolve(), data, args.device)
    met = A.compute_metrics(run)
    print(f"[lco_run_dump] {args.exp_dir} on {args.data_dir}: " + " ".join(
        f"{k}={v:.3f}" for k, v in met["subscores"].items()))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    A.save_run(run, out)
    print(f"[lco_run_dump] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
