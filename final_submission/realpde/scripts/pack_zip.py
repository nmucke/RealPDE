#!/usr/bin/env python3
"""Flat-zip a package dir into submission.zip, submission.py at root,
__pycache__ excluded -- same convention c8t_launch/c10t_launch used for
their own packaging step.

Usage: python pack_zip.py --pkg <dir> --out <submission.zip>
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pkg = Path(args.pkg).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in pkg.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, p.relative_to(pkg))
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB zipped, {len(files)} files)")


if __name__ == "__main__":
    main()
