#!/usr/bin/env python3
"""Evaluate presets of the final package on local data with the official streaming loop + scoring.

    python experiments/run_preset.py --data DATA                      # the final model, shipped weights
    python experiments/run_preset.py --data DATA l00_backbone lb0926_fa948762 --off lsk
    python experiments/run_preset.py --list
    python experiments/run_preset.py --diff lb0924_k3f_kd3            # resolved CONFIG vs final; no torch needed
    python experiments/run_preset.py --check                          # validate every preset; no torch needed
    python experiments/run_preset.py --package PKG --data DATA lb0927_final l07_cond_width
    python experiments/run_preset.py --package PKG --data DATA lb0927_final --off lsk --off rec
    python experiments/run_preset.py --package PKG --data DATA --as-packaged     # any package, CONFIG untouched

PKG is a package dir: the final source dir plus model.pth, prior.npz, ke_net_flow.pt and
cond_width_weights.pt (e.g. `<output>/package` written by final_submission/solution.py train(),
or the dir experiments/train_preset.sh assembles). It is never modified. Without --package the
package is assembled by experiments/make_package.py from --weights (default: the shipped
weights in weights/final_retrain_a100/, checked against their SHA256SUMS) into build/ and reused
while unchanged. Without preset ids, lb0927_final (the final package) is evaluated.

DATA is one of: a dir with `test_real/*.h5` (the starting-kit layout); the training layout with
`train_real/*.h5` (scores are then in-sample for the shipped weights); or a plain dir of `.h5`
files. Normalisation stats: --stats, else DATA/mean_std_real.pt, else the kit's copy. Default
DATA: the kit's two synthetic example trajectories (two windows each: a plumbing check only).

The loop and the five subscores are the starting kit's own `local_eval.py` / `scoring.py`
(final_submission/realpde/starting_kit/), called unchanged; only `import_submission` is pointed at
the model this script built. Several presets run one per subprocess (each needs a fresh import of
the package and its CUDA graphs).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import types

sys.dont_write_bytecode = True   # nothing may appear inside final_submission/ (kit) or the package dir
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ablate  # noqa: E402
import composite  # noqa: E402

KIT = os.path.join(ablate.FINAL_REALPDE, "starting_kit")
SUBSCORES = ("rel_l2_score", "tke_score", "mvpe_score", "time_score", "sps_score")


# --------------------------------------------------------------------------- #
# no-torch modes
# --------------------------------------------------------------------------- #
def cmd_list() -> None:
    rows = []
    for pid in ablate.preset_ids():
        p = ablate.load_preset(pid)
        a = p.get("approximates") or {}
        lb = a.get("leaderboard", {})
        rows.append((pid, p.get("group", ""), a.get("date", ""), "%.3f" % lb["final"] if lb else "",
                     p.get("training", {}).get("weights", ""), p.get("fidelity", {}).get("with_final_weights", ""),
                     p.get("title", "")))
    head = ("id", "group", "upload", "LB", "weights", "fidelity (final weights)", "title")
    w = [max(len(r[i]) for r in rows + [head]) for i in range(7)]
    fmt = "  ".join("%%-%ds" % n for n in w)
    print(fmt % head)
    for r in rows:
        print(fmt % r)
    print("\ncomponents (--off NAME):")
    for n, c in ablate.COMPONENTS.items():
        print("  %-14s %-8s %-6s %s" % (n, c["kind"], c["where"], c["doc"]))
    print("\n  KD2-KD8: kernel-diet levels (fewer / fused / concurrent CUDA kernels; KD5 near-exact, the rest bit-exact);"
          "\n  IPC / IPW / BVR: ways of returning outputs when ttt_step runs in a worker process (one shared buffer,"
          "\n  an untimed transport warm-up, pickling by value); CDIO: direct path for CUDA input tensors.")
    print("\nnot switchable at inference:")
    for n, d in ablate.NOT_SWITCHABLE.items():
        print("  %-18s %s" % (n, d))


def cmd_diff(ids, extra_off, extra) -> int:
    final = ablate.read_config()
    rc = 0
    for pid in ids:
        p = ablate.load_preset(pid)
        print("== %s: %s" % (pid, p.get("title", "")))
        try:
            ov, cfg, warn = ablate.resolve_preset(p, extra_off, extra)
        except ValueError as e:
            print("  INVALID: %s" % e)
            rc = 1
            continue
        diff = ablate.config_diff(cfg, final)
        if not diff:
            print("  (identical to the final CONFIG)")
        else:
            print("  %-16s %s" % ("key:", "final value -> preset value"))
        for k, a, b in diff:
            print("  %-16s %s -> %s" % (k, json.dumps(a), json.dumps(b)))
        off = ablate.components_off(cfg, final)
        print("  components off: %s" % (", ".join(off) or "none"))
        eager = ablate.needs_eager(cfg, final)
        if eager:
            print("  runs the eager path (graphs specialised to the shipped values of: %s)" % ", ".join(eager))
        for w in warn:
            print("  note: %s" % w)
        fid = p.get("fidelity", {})
        if fid:
            print("  fidelity: final weights: %s; weights '%s': %s"
                  % (fid.get("with_final_weights"), p["training"]["weights"], fid.get("with_preset_weights")))
    return rc


def cmd_check() -> int:
    final = ablate.read_config()
    bad = 0
    for pid in ablate.preset_ids():
        try:
            ablate.resolve_preset(ablate.load_preset(pid))
            print("ok       %s" % pid)
        except ValueError as e:
            bad += 1
            print("INVALID  %s: %s" % (pid, e))
    print("%d presets, %d invalid; final CONFIG has %d keys" % (len(ablate.preset_ids()), bad, len(final)))
    return 1 if bad else 0


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def _load_local_eval():
    spec = importlib.util.spec_from_file_location("rpde_kit_local_eval", os.path.join(KIT, "local_eval.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_data(data: str, stats: str = None):
    """(dir with the .h5 files, stats path, note)."""
    data = os.path.abspath(data)
    if not os.path.isdir(data):
        raise SystemExit("--data %s is not a directory" % data)
    note = None
    if os.path.isdir(os.path.join(data, "test_real")):
        src = os.path.join(data, "test_real")
    elif os.path.isdir(os.path.join(data, "train_real")):
        src = os.path.join(data, "train_real")
        note = ("DATA/train_real: every shipped weight set was trained on these files, so the scores are in-sample "
                "(see experiments/README.md, Caveats)")
    else:
        src = data
    if not any(f.endswith(".h5") for f in os.listdir(src)):
        raise SystemExit("no .h5 files in %s" % src)
    if stats:
        if not os.path.isfile(stats):
            raise SystemExit("--stats %s does not exist" % stats)
        st = os.path.abspath(stats)
    elif os.path.isfile(os.path.join(data, "mean_std_real.pt")):
        st = os.path.join(data, "mean_std_real.pt")
    else:
        st = os.path.join(KIT, "example_data", "mean_std_real.pt")
    return src, st, note


def _shim_dir(src: str, st: str, limit: int, tmp: str) -> str:
    """The kit's layout (test_real/*.h5 + mean_std_real.pt) as symlinks; 7575_0.h5 is skipped."""
    out = os.path.join(tmp, "data")
    os.makedirs(os.path.join(out, "test_real"))
    files = sorted(f for f in os.listdir(src) if f.endswith(".h5") and f != "7575_0.h5")
    for f in files[:limit or None]:
        os.symlink(os.path.join(src, f), os.path.join(out, "test_real", f))
    os.symlink(st, os.path.join(out, "mean_std_real.pt"))
    return out


def evaluate(model, pkg_dir: str, data_dir: str) -> dict:
    """Run the kit's local_eval.main() on `model`; returns the parsed subscores and the log."""
    le = _load_local_eval()
    shim = types.SimpleNamespace(get_ttt_model=lambda submission_dir, device: model)
    le.import_submission = lambda submission_dir: shim
    argv = sys.argv
    sys.argv = ["local_eval.py", "--submission", pkg_dir, "--data", data_dir]
    buf = io.StringIO()
    status = "ok"
    try:
        with contextlib.redirect_stdout(buf):
            le.main()
    except SystemExit as e:
        status = "failed: %s" % e
    finally:
        sys.argv = argv
    log = buf.getvalue()
    scores = {}
    for name in SUBSCORES:
        m = re.search(r"\[local_eval\]\s+%s\s+(-?[0-9.]+)" % name, log)
        if m:
            scores[name] = float(m.group(1))
    t = re.search(r"mean per-step time: ([0-9.]+) ms", log)
    n = re.search(r"\[local_eval\] (\d+) steps over (\d+) trajectories", log)
    if status == "ok" and len(scores) != 5:
        status = "failed: no subscores"
    return {"status": status, "scores": scores, "ms_per_step": float(t.group(1)) if t else None,
            "steps": int(n.group(1)) if n else None, "trajectories": int(n.group(2)) if n else None, "log": log}


def run_one(args, pid) -> dict:
    if args.as_packaged:
        model, report = ablate.build_packaged(args.package, args.device)
    else:
        model, report = ablate.build_model(args.package, ablate.load_preset(pid), args.device,
                                           extra_off=args.off, extra=parse_set(args.set),
                                           allow_modified_source=args.allow_modified_source)
    src, st, _ = resolve_data(args.data, args.stats)
    with tempfile.TemporaryDirectory(prefix="run_preset_") as tmp:
        res = evaluate(model, os.path.abspath(args.package), _shim_dir(src, st, args.limit, tmp))
    if report.get("problems") and not args.allow_inactive and res["status"] == "ok":
        res["status"] = "failed: expected component inactive (--allow-inactive to keep the scores)"
    if len(res["scores"]) == 5:
        try:
            res["estimate"] = composite.estimate(res["scores"], composite.fit())
        except Exception as e:  # noqa: BLE001 - numpy or leaderboard.csv missing: no estimate
            res["estimate_error"] = repr(e)
    res.update(id=pid, report=report, off=args.off, set=args.set)
    return res


def parse_set(items):
    out = {}
    for it in items or []:
        k, sep, v = it.partition("=")
        if not sep or not k:
            raise SystemExit("--set needs KEY=VALUE (VALUE as JSON), got %r" % it)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _print_table(results) -> None:
    hdr = ("id", "rel_l2", "tke", "mvpe", "time", "sps", "fit", "plan_rel", "ms/step", "path / status")
    print("  ".join("%-18s" % hdr[0] if i == 0 else "%8s" % h for i, h in enumerate(hdr[:-1])) + "  " + hdr[-1])
    extrap = False
    for r in results:
        s = r.get("scores", {})
        e = r.get("estimate") or {}
        cells = ["%-18s" % r["id"]] + ["%8.3f" % s[k] if k in s else "%8s" % "-" for k in SUBSCORES]
        flag = bool(e.get("extrapolated") or e.get("high_leverage"))
        cells.append(("%7.3f%s" % (e["fit"], "*" if flag else " ")) if e else "%8s" % "-")
        extrap |= flag
        cells.append("%8.3f" % e["plan_rel"] if e else "%8s" % "-")
        cells.append("%8.2f" % r["ms_per_step"] if r.get("ms_per_step") else "%8s" % "-")
        rep = r.get("report", {})
        path = ("whole-step graph" if rep.get("whole_step_graph") else
                "eager (forced)" if rep.get("eager_forced_by") else "no whole-step graph")
        if rep.get("compute_device") == "cpu":
            path += ", CPU"
        cells.append(path if r.get("status") == "ok" else r.get("status", "?"))
        print("  ".join(cells))
    print("fit = least-squares estimate of final_score from experiments/leaderboard.csv (python experiments/composite.py"
          " for its residuals)%s; plan_rel = the team's planning weights, about 0.8 below the leaderboard scale, "
          "for differences only. Local time_score is this machine's."
          % ("; * = outside the fit rows' range or above their leverage, estimate not reliable" if extrap else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Output columns: fit = least-squares estimate of the leaderboard final score "
                                        "(experiments/composite.py); plan_rel = the team's planning composite, "
                                        "about 0.8 below the leaderboard scale, meaningful only as differences "
                                        "between presets.")
    ap.add_argument("presets", nargs="*", help="preset ids (experiments/presets/<id>.json) or .json paths")
    ap.add_argument("--list", action="store_true", help="list presets and components")
    ap.add_argument("--diff", action="store_true", help="print the resolved CONFIG diff vs the final package")
    ap.add_argument("--check", action="store_true", help="validate every preset")
    ap.add_argument("--package", help="package dir (final source + the four weight files); default: assembled "
                    "from --weights by make_package.py")
    ap.add_argument("--weights", help="weight dir to assemble the package from when --package is not given "
                    "(default: weights/final_retrain_a100)")
    ap.add_argument("--as-packaged", action="store_true", help="evaluate --package with its own CONFIG")
    ap.add_argument("--data", default=os.path.join(KIT, "example_data"), help="evaluation data dir")
    ap.add_argument("--stats", help="mean_std_real.pt (default: DATA's, else the kit's)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N .h5 files")
    ap.add_argument("--device", default=None, help="device passed to get_ttt_model; default cuda if available, else "
                    "cpu (the package computes on CUDA whenever it is visible: for a CPU run on a GPU machine use "
                    "CUDA_VISIBLE_DEVICES= together with --device cpu)")
    ap.add_argument("--allow-modified-source", action="store_true",
                    help="evaluate presets on a package whose code differs from the final source (not recommended)")
    ap.add_argument("--off", action="append", default=[], help="also switch off this component (repeatable)")
    ap.add_argument("--set", action="append", default=[], help="also set CONFIG KEY=JSON (repeatable)")
    ap.add_argument("--allow-inactive", action="store_true",
                    help="keep a run whose CONFIG asks for a component that did not come up on the GPU")
    ap.add_argument("--json", help="write all results (incl. logs and runtime reports) to this file")
    ap.add_argument("--_child", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.list:
        cmd_list()
        return 0
    if args.check:
        return cmd_check()
    try:
        extra = parse_set(args.set)
        if args.diff:
            if not args.presets:
                ap.error("--diff needs preset ids")
            return cmd_diff(args.presets, args.off, extra)
        if args.package and args.weights:
            ap.error("give --package or --weights, not both")
        if not args.package:
            import make_package
            args.package = make_package.make_package(None, args.weights or make_package.DEFAULT_WEIGHTS, quiet=True)
            print("package: %s" % args.package)
        if args.as_packaged:
            if args.presets or args.off or args.set:
                ap.error("--as-packaged evaluates the package's own CONFIG; it takes no preset ids, --off or --set")
            ids = ["as-packaged"]
        else:
            ids = args.presets or ["lb0927_final"]
            for pid in ids:                             # fail before any model is built
                ablate.resolve_preset(ablate.load_preset(pid), args.off, extra)
            if os.path.isfile(os.path.join(args.package, "submission.py")) and not args.allow_modified_source:
                bad = ablate.source_mismatches(args.package)
                if bad:
                    raise ValueError("package %s differs from the final source in %s; use --as-packaged (its own "
                                     "CONFIG) or --allow-modified-source" % (args.package, ", ".join(bad[:8])))
        src, st, note = resolve_data(args.data, args.stats)
    except (ValueError, RuntimeError) as e:
        raise SystemExit("error: %s" % e)
    if note and not args._child:
        print("WARNING: %s" % note)
    if args.device is None:                             # decided here, so --list / --check / --diff need no torch
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args._child:
        res = run_one(args, ids[0])
        with open(args._child, "w") as f:
            json.dump(res, f, default=str)
        return 0
    results = []
    if len(ids) == 1:
        results.append(run_one(args, ids[0]))
    else:
        common = ["--package", args.package, "--data", args.data, "--device", args.device]
        common += ["--stats", args.stats] if args.stats else []
        common += ["--limit", str(args.limit)] if args.limit else []
        common += ["--allow-inactive"] if args.allow_inactive else []
        common += ["--allow-modified-source"] if args.allow_modified_source else []
        for o in args.off:
            common += ["--off", o]
        for s in args.set:
            common += ["--set", s]
        for pid in ids:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as t:
                out = t.name
            try:
                p = subprocess.run([sys.executable, "-B", os.path.abspath(__file__), pid, "--_child", out] + common)
                if p.returncode == 0:
                    with open(out) as f:
                        results.append(json.load(f))
                else:
                    results.append({"id": pid, "status": "failed (rc %d)" % p.returncode})
            finally:
                os.remove(out)
    for r in results:
        rep = r.get("report") or {}
        for w in rep.get("warnings", []):
            print("[%s] note: %s" % (r["id"], w))
        if rep.get("lsk_error"):
            print("[%s] LSK switched itself off: %s" % (r["id"], rep["lsk_error"]))
        for k, v in (rep.get("graph_errors") or {}).items():
            print("[%s] %s: %s" % (r["id"], k, v))
        for prob in rep.get("problems") or []:
            print("[%s] PROBLEM: %s" % (r["id"], prob))
        if r.get("steps") and r.get("trajectories") and r["steps"] <= 2 * r["trajectories"]:
            print("[%s] note: at most 2 windows per trajectory; windows >= 2 (whole-step graph, kw / eb / br / rec / "
                  "lowke, KD2-KD8) never ran" % r["id"])
        if r.get("status") != "ok" and r.get("log"):
            print(r["log"][-2000:])
    _print_table(results)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1, default=str)
    return 0 if all(r.get("status") == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
