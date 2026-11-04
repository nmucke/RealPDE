"""Team nmucke -- RealPDE Track 2 (LTTTA) training package.

Entry points required by the Decision Phase (FINAL_SUBMISSION.md):

  train(data_dir, output_dir, device)   -- retrains the method from scratch on the public training data
  get_ttt_model(output_dir, device)     -- returns the Codabench TTT model (reset_ttt_state / ttt_step)

The method is trained by one shell chain, ``realpde/scripts/retrain_all.sh``: it builds the
packaged submission (code + model.pth + prior.npz + ke_net_flow.pt + cond_width_weights.pt).
With its default flags it builds exactly the lineage of Codabench submission 948041 (archive
sha256 f11d2c1d...). ``train`` runs that chain in a writable copy of ``realpde/`` inside
``output_dir`` and unpacks the resulting ``submission.zip`` into ``output_dir/package``.
``get_ttt_model`` imports that package's ``submission.py`` exactly as Codabench did and calls its
``get_ttt_model(package_dir, device)``.
"""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_SRC = os.path.join(_HERE, "realpde")
_PKG_SUBDIR = "package"
_WORK_SUBDIR = "_work"


def _log(msg, fh=None):
    line = "[solution %s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _same_stats(a, b):
    import torch
    ta = torch.load(a, map_location="cpu", weights_only=False)
    tb = torch.load(b, map_location="cpu", weights_only=False)
    if len(ta) != len(tb):
        return False
    return all(torch.equal(torch.as_tensor(x), torch.as_tensor(y)) for x, y in zip(ta, tb))


def _prepare_work_repo(data_dir, work, fh):
    """Writable copy of the training code with data/competition pointing at data_dir."""
    repo = os.path.join(work, "realpde")
    if os.path.isdir(repo):
        shutil.rmtree(repo)
    shutil.copytree(_REPO_SRC, repo, symlinks=True)

    # data/competition/{train_sim,train_real} -> the read-only mounted release
    comp = os.path.join(repo, "data", "competition")
    os.makedirs(comp, exist_ok=True)
    for name in ("train_sim", "train_real"):
        src = os.path.join(data_dir, name)
        if not os.path.isdir(src):
            raise RuntimeError("missing %s" % src)
        dst = os.path.join(comp, name)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    n_sim = len([f for f in os.listdir(os.path.join(data_dir, "train_sim")) if f.endswith(".h5")])
    real = sorted(f for f in os.listdir(os.path.join(data_dir, "train_real")) if f.endswith(".h5"))
    _log("data: %d train_sim files, %d train_real files (7575_0.h5 present: %s)"
         % (n_sim, len(real), "7575_0.h5" in real), fh)

    # Normalisation statistics: the chain reads starting_kit/example_data/mean_std_real.pt;
    # use the official file mounted with the data if it is present.
    official = os.path.join(data_dir, "mean_std_real.pt")
    ours = os.path.join(repo, "starting_kit", "example_data", "mean_std_real.pt")
    if os.path.isfile(official):
        if _same_stats(official, ours):
            _log("mean_std_real.pt: identical to the official file in %s" % data_dir, fh)
        else:
            shutil.copyfile(official, ours)
            _log("mean_std_real.pt: replaced by the official file from %s" % data_dir, fh)

    # The chain calls $REPO/.venv/bin/python: a small wrapper that execs this interpreter (the image's
    # python) by its own path, so its prefix / site-packages are exactly the ones running train().
    venv_bin = os.path.join(repo, ".venv", "bin")
    os.makedirs(venv_bin, exist_ok=True)
    for exe in ("python", "python3"):
        p = os.path.join(venv_bin, exe)
        if os.path.lexists(p):
            os.remove(p)
        with open(p, "w") as f:
            f.write('#!/bin/sh\nexec "%s" "$@"\n' % sys.executable)
        os.chmod(p, 0o755)
    return repo


def train(data_dir, output_dir, device="cuda"):
    if device != "cuda":
        raise RuntimeError("training needs a CUDA device (got %r)" % (device,))
    os.makedirs(output_dir, exist_ok=True)
    work = os.path.join(output_dir, _WORK_SUBDIR)
    os.makedirs(work, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")
    t0 = time.time()
    with open(log_path, "a") as fh:
        _log("train(data_dir=%s, output_dir=%s, device=%s)" % (data_dir, output_dir, device), fh)
        repo = _prepare_work_repo(data_dir, work, fh)
        tmp = os.path.join(work, "tmp")
        os.makedirs(tmp, exist_ok=True)
        env = dict(os.environ)
        env.update({"TMPDIR": tmp, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                    "PATH": os.path.join(repo, ".venv", "bin") + os.pathsep + env.get("PATH", "")})
        # Default flags = the final lineage (BACKBONE=b5spx3 + FREEZE defaults + LSK/LSKR/KD8/LSKR97).
        env.update({"LSK": "1", "LSKR": "1", "KD8": "1", "LSKR97": "1"})
        cmd = ["bash", os.path.join(repo, "scripts", "retrain_all.sh")]
        _log("running: %s  (cwd %s)" % (" ".join(cmd), repo), fh)
        proc = subprocess.run(cmd, cwd=repo, env=env, stdout=fh, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            _log("retrain_all.sh FAILED rc=%d after %.1f min" % (proc.returncode, (time.time() - t0) / 60), fh)
            raise RuntimeError("retrain_all.sh failed (rc=%d); see %s" % (proc.returncode, log_path))

        subs = os.path.join(repo, "submissions")
        zips = [os.path.join(subs, d, "submission.zip") for d in os.listdir(subs) if d.startswith("retrained_")]
        zips = [z for z in zips if os.path.isfile(z) and "-lsk-recr-kd8-r97" in os.path.dirname(z)]
        if len(zips) != 1:
            raise RuntimeError("expected exactly one final package, found %r" % (zips,))
        zpath = zips[0]
        pkg = os.path.join(output_dir, _PKG_SUBDIR)
        if os.path.isdir(pkg):
            shutil.rmtree(pkg)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(pkg)
        if not os.path.isfile(os.path.join(pkg, "submission.py")):
            raise RuntimeError("package has no submission.py")
        shutil.copyfile(zpath, os.path.join(output_dir, "submission.zip"))
        info = {
            "package_zip_sha256": _sha256(zpath),
            "package_dir": os.path.basename(os.path.dirname(zpath)),
            "entries": {n: _sha256(os.path.join(pkg, n)) for n in sorted(os.listdir(pkg))
                        if os.path.isfile(os.path.join(pkg, n))},
            "train_minutes": round((time.time() - t0) / 60, 1),
        }
        with open(os.path.join(output_dir, "package_info.json"), "w") as f:
            json.dump(info, f, indent=1)
        _log("done in %.1f min; package in %s (zip sha256 %s)"
             % (info["train_minutes"], pkg, info["package_zip_sha256"]), fh)


def get_ttt_model(output_dir, device="cuda"):
    pkg = os.path.join(output_dir, _PKG_SUBDIR)
    sub = os.path.join(pkg, "submission.py")
    if not os.path.isfile(sub):
        raise FileNotFoundError("%s missing -- run train() first" % sub)
    # The package's modules import `submission` by name (as on Codabench), so load it under that name.
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    mod = sys.modules.get("submission")
    if mod is None or os.path.abspath(getattr(mod, "__file__", "")) != os.path.abspath(sub):
        spec = importlib.util.spec_from_file_location("submission", sub)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["submission"] = mod
        spec.loader.exec_module(mod)
    return mod.get_ttt_model(pkg, device)
