#!/usr/bin/env python3
"""Assemble a ready-to-run package from the final source dir and a set of trained weights.

    python experiments/make_package.py                              # -> build/package_final_retrain_a100/
    python experiments/make_package.py OUT --weights DIR            # other weights / other place

A package dir = the final source dir (final_submission/realpde/submissions/<final>/source) plus
model.pth, prior.npz, ke_net_flow.pt and cond_width_weights.pt. Before copying, the weights are
checked against DIR/SHA256SUMS; after copying, the package's top-level files are checked against
DIR/package_info.json when it exists (the file list train() recorded). The result is what
final_submission/solution.py train() writes to <output>/package, usable by run_preset.py, by
examples/run_ttt.py and as a Codabench submission.

OUT is reused when it already holds this package unchanged, rebuilt when it is a package this
script made earlier, and refused otherwise. Standard library only (no torch needed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ablate  # noqa: E402  (stdlib only at import time)

DEFAULT_WEIGHTS = os.path.join(ablate.REPO, "weights", "final_retrain_a100")
MARKER = ".make_package.json"


def default_out(weights_dir: str = DEFAULT_WEIGHTS) -> str:
    return os.path.join(ablate.REPO, "build", "package_" + os.path.basename(os.path.abspath(weights_dir)))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_weights(weights_dir: str) -> dict:
    """{name: sha256} of the four weight files, checked against weights_dir/SHA256SUMS (ValueError on mismatch)."""
    sums = os.path.join(weights_dir, "SHA256SUMS")
    if not os.path.isfile(sums):
        raise ValueError("%s has no SHA256SUMS" % weights_dir)
    want = {}
    with open(sums) as f:
        for line in f:
            if line.strip():
                h, name = line.split()
                want[name.lstrip("*")] = h
    out = {}
    for name in ablate.WEIGHT_FILES:
        p = os.path.join(weights_dir, name)
        if name not in want:
            raise ValueError("SHA256SUMS in %s has no entry for %s" % (weights_dir, name))
        if not os.path.isfile(p):
            raise ValueError("missing %s" % p)
        got = _sha256(p)
        if got != want[name]:
            raise ValueError("%s: sha256 %s does not match SHA256SUMS (%s)" % (p, got, want[name]))
        out[name] = got
    return out


def _package_ok(out: str, hashes: dict) -> bool:
    """OUT holds the final source (code files) and exactly these weights."""
    try:
        ablate.check_package_dir(out)
    except (FileNotFoundError, RuntimeError):
        return False
    if ablate.source_mismatches(out):
        return False
    return all(_sha256(os.path.join(out, n)) == h for n, h in hashes.items())


def _check_against_info(out: str, weights_dir: str) -> None:
    info = os.path.join(weights_dir, "package_info.json")
    if not os.path.isfile(info):
        return
    with open(info) as f:
        entries = json.load(f).get("entries", {})
    bad = [n for n, h in entries.items() if not os.path.isfile(os.path.join(out, n)) or _sha256(os.path.join(out, n)) != h]
    if bad:
        raise ValueError("assembled package differs from %s in %s" % (info, ", ".join(sorted(bad))))


def make_package(out: str = None, weights_dir: str = DEFAULT_WEIGHTS, quiet: bool = False) -> str:
    """Assemble (or reuse) the package; returns its absolute path."""
    weights_dir = os.path.abspath(weights_dir)
    out = os.path.abspath(out or default_out(weights_dir))
    ablate.check_package_location(out)
    for d in ("final_submission", "weights", "experiments"):
        if ablate._under(out, os.path.join(ablate.REPO, d)) or out == os.path.join(ablate.REPO, d):
            raise ValueError("refusing to write a package inside %s/ (default: build/)" % d)
    hashes = verify_weights(weights_dir)
    if os.path.exists(out):
        if _package_ok(out, hashes):
            if not quiet:
                print("package up to date: %s" % out)
            return out
        if not os.path.isfile(os.path.join(out, MARKER)):
            if os.path.isdir(out) and not os.listdir(out):
                os.rmdir(out)
            else:
                raise ValueError("%s exists and was not made by make_package.py; refusing to overwrite it" % out)
        else:
            shutil.rmtree(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ablate.assemble_package({n: os.path.join(weights_dir, n) for n in ablate.WEIGHT_FILES}, out)
    _check_against_info(out, weights_dir)
    with open(os.path.join(out, MARKER), "w") as f:
        json.dump({"weights_dir": weights_dir, "weights_sha256": hashes,
                   "source_dir": ablate.FINAL_SOURCE}, f, indent=1)
    if not quiet:
        print("package assembled: %s" % out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", help="package dir (default: build/package_<weights dir name>/ in the repo)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="dir with the four weight files and SHA256SUMS "
                    "(default: weights/final_retrain_a100)")
    args = ap.parse_args()
    try:
        path = make_package(args.out, args.weights, quiet=True)
    except (ValueError, RuntimeError, FileExistsError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
