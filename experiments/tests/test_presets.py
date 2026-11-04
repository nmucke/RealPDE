"""Checks of the component-selection tooling. CPU only, no data, no trained weights.

    python -m pytest experiments/tests/test_presets.py                       # fast suite
    ABLATE_SLOW_TESTS=1 python -m pytest experiments/tests/test_presets.py   # + the smoke tests
    python experiments/tests/test_presets.py                                 # same, without pytest

Fast suite (torch only for the package-loading tests): presets are well formed, their overrides
are valid CONFIG keys that pass the dependency rules, each lineage preset resolves to exactly the
CONFIG of the historic package it names (where that source is in the repository), the CLI argument checks, the composite fit, package
loading / purging, the graph stand-ins and runtime_report on objects without the newer attributes.

Slow suite (ABLATE_SLOW_TESTS=1): every preset builds through the package's own get_ttt_model on
CPU with synthetic weights (tests/dummy_weights.py) and returns finite, correctly shaped outputs
for a few synthetic windows; two presets run through the kit's local_eval via run_preset.py.
"""
from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import types
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""      # CPU only, also on a GPU machine (set before torch is imported)
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import ablate  # noqa: E402

try:
    import pytest
except ImportError:  # plain-python runner below
    pytest = None

FINAL = ablate.read_config()
PRESETS = {pid: ablate.load_preset(pid) for pid in ablate.preset_ids()}
LINEAGE = sorted(p for p, v in PRESETS.items() if v.get("group") == "lineage")
FIDELITY = {"exact", "same code path, different weights", "approximate"}
SLOW = os.environ.get("ABLATE_SLOW_TESTS") == "1"


class _Skip(Exception):
    pass


def _skip(why):
    if pytest is not None:
        pytest.skip(why)
    raise _Skip(why)


def _need_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        _skip("torch not installed")


def _need_slow():
    if not SLOW:
        _skip("set ABLATE_SLOW_TESTS=1")
    _need_torch()


# --------------------------------------------------------------------------- static
def test_preset_fields():
    assert len(PRESETS) >= 20
    for pid, p in PRESETS.items():
        assert p["id"] == pid
        for k in ("group", "title", "description", "off", "inference", "training", "fidelity"):
            assert k in p, (pid, k)
        assert set(p["training"]) >= {"weights", "env", "outputs", "arch"}, pid
        assert set(p.get("expect_active", [])) <= set(ablate.EXPECTABLE), pid
        if p["group"] == "lineage":
            a = p["approximates"]
            assert set(a["leaderboard"]) == {"rel_l2", "tke", "mvpe", "time", "sps", "final"}, pid
            assert p["fidelity"]["with_final_weights"] in FIDELITY, pid
            assert p["fidelity"]["with_preset_weights"] in FIDELITY, pid


def test_components_reference_real_keys():
    for name, c in ablate.COMPONENTS.items():
        for k, v in c["off"].items():
            assert k in FINAL, (name, k)
        assert any(FINAL[k] != v for k, v in c["off"].items()), name   # the final package has it on
        assert c["kind"] in ("pred", "interval", "time", "time~") and c["where"] in ("eager", "graph", "cuda")


def test_every_preset_resolves():
    for pid, p in PRESETS.items():
        ov, cfg, _ = ablate.resolve_preset(p)
        assert set(ov) <= set(FINAL), pid
        assert set(cfg) == set(FINAL), pid


def test_final_preset_is_the_final_config():
    _, cfg, _ = ablate.resolve_preset(PRESETS["lb0927_final"])
    assert ablate.config_diff(cfg) == []


def test_lineage_matches_historic_configs():
    """Keys in the historic CONFIG must match; keys the historic package did not have must be off.

    Only the lineage presets whose historic source is in the repository (config_source not null)."""
    checked = [pid for pid in LINEAGE if PRESETS[pid]["approximates"]["config_source"]]
    assert len(checked) >= 3, checked
    for pid in checked:
        src = os.path.join(ablate.REPO, PRESETS[pid]["approximates"]["config_source"])
        hist = ablate.read_config(src)
        _, cfg, _ = ablate.resolve_preset(PRESETS[pid])
        bad = []
        for k in sorted(set(hist) | set(cfg)):
            if k == "unet_ckpt":            # dev-only checkpoint path; packaged model.pth wins
                continue
            if k in hist and hist[k] != cfg.get(k):
                bad.append((k, hist[k], cfg.get(k)))
            elif k not in hist and cfg[k] not in (None, False):
                bad.append((k, "<absent>", cfg[k]))
        assert not bad, (pid, bad)


def test_lineage_is_monotone():
    """Each older upload's `off` list contains the next newer one's (raw `inference` keys aside)."""
    order = ["lb0927_final", "lb0926_fa948762", "lb0925_fze", "lb0924_k3f_kd3", "lb0923_b5sp", "lb0922_n1_w2"]
    for newer, older in zip(order, order[1:]):
        assert set(PRESETS[newer]["off"]) <= set(PRESETS[older]["off"]), (newer, older)


def _raises(ov, text):
    try:
        ablate.check_overrides(ov)
    except ValueError as e:
        assert text in str(e), (text, str(e))
        return
    raise AssertionError("no error for %r" % (ov,))


def test_dependency_rules():
    _raises({"cb_blend_w": None}, "cb_adapt (AW) needs cb_blend_w")
    _raises({"cb_adapt": None}, "kw_adapt needs cb_adapt")
    _raises({"eb_adapt": None}, "br_adapt needs eb_adapt")
    _raises({"e_blend_w": None}, "eb_adapt needs e_blend_w")
    _raises({"rec_detect": None}, "lsk_rec_reset needs lsk_head and rec_detect")
    _raises({"kd3_step": False}, "rec_detect (REC) needs kd3_step")
    _raises({"fp16_core": False}, "rec_detect (REC) needs kd3_step")
    _raises({"cb_blend_w": 0.75}, "a list of 20 values")
    _raises({"ctr": False}, "cb_blend_w (CB) needs ctr")
    _raises({"bias": False}, "runs only in the whole-step graph")
    _raises({"cond_width": False}, "runs only in the whole-step graph")
    _raises({"ke_flow_scale": 0.0}, "runs only in the whole-step graph")
    _raises({"no_such_key": 1}, "unknown CONFIG keys")
    _raises({"cond_width_ckpt_path": "x"}, "runtime-only")
    for off in (["kd6"], ["kd7"], ["kd8"], ["kd2"], ["kd4"], ["kd5"], ["ipc", "ipw", "bvr", "cdio"], ["lsk"], ["rec"]):
        ablate.check_overrides(ablate.component_overrides(off))
    try:
        ablate.component_overrides(["no_such_component"])
        raise AssertionError("unknown component accepted")
    except ValueError:
        pass


def test_eager_and_kd6_rules():
    _, cfg, _ = ablate.resolve_preset(PRESETS["l00_backbone"])
    assert "bias" in ablate.needs_eager(cfg)
    _, cfg, w = ablate.resolve_preset(PRESETS["l07_cond_width"])
    assert ablate.needs_eager(cfg) == [] and ablate.whole_step_scope_ok(cfg)[0]
    assert any(x.startswith("kd6_concurrency inert") for x in w)
    for pid in ("l13_axe", "l12_msk"):
        assert any(x.startswith("kd6_concurrency inert") for x in ablate.resolve_preset(PRESETS[pid])[2]), pid
    for pid in ("l14_rec", "lb0927_final"):
        assert not any("kd6" in x for x in ablate.resolve_preset(PRESETS[pid])[2]), pid
    assert ablate.needs_eager(dict(FINAL, ke_flow_scale=0.0)) == ["ke_flow"]


def _cli(args, block_torch=True, env=None):
    code = ("import sys, builtins; real = builtins.__import__\n"
            "def imp(n, *a, **k):\n"
            "    if n == 'torch' or n.startswith('torch.'): raise ImportError('torch blocked')\n"
            "    return real(n, *a, **k)\n"
            + ("builtins.__import__ = imp\n" if block_torch else "") +
            "sys.argv = ['run_preset.py'] + sys.argv[1:]\n"
            "import runpy; runpy.run_path(%r, run_name='__main__')\n" % os.path.join(EXP, "run_preset.py"))
    return subprocess.run([sys.executable, "-B", "-c", code] + args, capture_output=True, text=True, env=env)


def test_cli_without_torch():
    """--list / --check / --diff and the argument errors work in an interpreter without torch."""
    for args in (["--list"], ["--check"], ["--diff", "lb0924_k3f_kd3", "l00_backbone"]):
        r = _cli(args)
        assert r.returncode == 0, (args, r.stdout[-1000:], r.stderr[-2000:])
    r = _cli(["--diff", "lb0927_final", "--off", "bias"])
    assert r.returncode == 1 and "INVALID" in r.stdout
    pkg = tempfile.mkdtemp(prefix="ablate_cli_")
    try:
        for args, text in ((["--diff", "lb0927_final", "--set", "gain_max"], "KEY=VALUE"),
                           (["--diff", "lb0927_final", "--off", "nope"], "unknown component"),
                           (["--package", pkg, "--as-packaged", "lb0927_final"], "takes no preset ids"),
                           (["--package", pkg, "lb0927_final", "--stats", "/no/such.pt"], "does not exist"),
                           (["--package", pkg, "no_such_preset"], "no preset")):
            r = _cli(args)
            assert r.returncode != 0 and text in (r.stdout + r.stderr), (args, r.stdout, r.stderr[-500:])
    finally:
        shutil.rmtree(pkg)


def test_data_layouts():
    import run_preset
    root = tempfile.mkdtemp(prefix="ablate_data_")
    try:
        for sub in ("test_real", "train_real", ""):
            d = os.path.join(root, sub or "flat")
            h5 = os.path.join(d, sub) if sub else d
            os.makedirs(h5)
            open(os.path.join(h5, "1_0.h5"), "w").close()
            src, st, note = run_preset.resolve_data(d)
            assert src == h5 and os.path.isfile(st), sub
            assert (note is not None) == (sub == "train_real"), sub
    finally:
        shutil.rmtree(root)


def test_expect_active_validated():
    d = tempfile.mkdtemp(prefix="ablate_preset_")
    try:
        path = os.path.join(d, "x.json")
        with open(path, "w") as f:
            f.write('{"id": "x", "expect_active": ["rec", "kd9"]}')
        try:
            ablate.load_preset(path)
            raise AssertionError("unknown expect_active name accepted")
        except ValueError as e:
            assert "kd9" in str(e)
    finally:
        shutil.rmtree(d)


def test_source_hash_check():
    d = tempfile.mkdtemp(prefix="ablate_src_")
    try:
        pkg = os.path.join(d, "a", "b", "pkg")
        shutil.copytree(ablate.FINAL_SOURCE, pkg, ignore=shutil.ignore_patterns("__pycache__"))
        assert ablate.source_mismatches(pkg) == []
        p = os.path.join(pkg, "submission.py")
        with open(p) as f:
            txt = f.read()
        with open(p, "w") as f:
            f.write(txt.replace('"rho": 0.97', '"rho": 0.9', 1))       # e.g. the -recr-kd8 source without r97
        assert ablate.source_mismatches(pkg) == ["submission.py"]
        for w in ablate.WEIGHT_FILES:
            open(os.path.join(pkg, w), "w").close()
        try:
            ablate.build_model(pkg, None, device="cpu")
            raise AssertionError("modified source accepted")
        except ValueError as e:
            assert "differs from the final source" in str(e)
    finally:
        shutil.rmtree(d)


WEIGHTS = os.path.join(ablate.REPO, "weights", "final_retrain_a100")


def test_make_package_verifies_hashes():
    if not os.path.isdir(WEIGHTS):
        _skip("no shipped weights")
    import make_package
    d = tempfile.mkdtemp(prefix="ablate_mkpkg_")
    try:
        w = os.path.join(d, "w")
        shutil.copytree(WEIGHTS, w)
        out = os.path.join(d, "a", "b", "pkg")
        assert make_package.make_package(out, w, quiet=True) == out
        assert ablate.source_mismatches(out) == []
        assert make_package.make_package(out, w, quiet=True) == out              # reused
        with open(os.path.join(w, "cond_width_weights.pt"), "ab") as f:
            f.write(b"x")                                                          # corrupt one weight file
        try:
            make_package.make_package(os.path.join(d, "a", "b", "pkg2"), w, quiet=True)
            raise AssertionError("corrupt weights accepted")
        except ValueError as e:
            assert "does not match SHA256SUMS" in str(e)
        other = os.path.join(d, "a", "b", "other")
        os.makedirs(other)
        open(os.path.join(other, "keep.txt"), "w").close()
        shutil.copyfile(os.path.join(WEIGHTS, "cond_width_weights.pt"), os.path.join(w, "cond_width_weights.pt"))
        try:
            make_package.make_package(other, w, quiet=True)
            raise AssertionError("non-package dir overwritten")
        except ValueError as e:
            assert "refusing to overwrite" in str(e)
        assert os.path.isfile(os.path.join(other, "keep.txt"))
    finally:
        shutil.rmtree(d)


def test_composite_fit():
    try:
        import numpy  # noqa: F401
    except ImportError:
        _skip("numpy not installed")
    import composite
    f = composite.fit()
    assert f["n"] >= 10
    assert max(abs(x) for x in f["in_sample"]) < 0.01
    assert max(abs(x) for x in f["loo"]) < 0.05
    e = composite.estimate({"rel_l2": 94.939383, "tke": 81.016714, "mvpe": 94.49995, "time": 92.46405,
                            "sps": 45.210544}, f)
    assert abs(e["fit"] - 82.452) < 0.01 and not e["extrapolated"] and not e["high_leverage"]
    assert f["skipped"] == 0 and max(f["leverage"]) == f["max_leverage"]
    far = composite.estimate({"rel_l2": 96.5, "tke": 85.0, "mvpe": 97.5, "time": 60.0, "sps": 50.0}, f)
    assert far["extrapolated"] and far["high_leverage"]


# --------------------------------------------------------------------------- loading (torch, CPU, fast)
_PKG = {}


def _dummy_package():
    if "dir" not in _PKG:
        import dummy_weights
        root = tempfile.mkdtemp(prefix="ablate_test_")
        atexit.register(shutil.rmtree, root, True)
        _PKG["root"] = root
        _PKG["dir"] = dummy_weights.make_dummy_package(os.path.join(root, "a", "b", "pkg"))
    return _PKG["dir"]


def test_load_package_purges_previous():
    _need_torch()
    a = _dummy_package()
    b = os.path.join(_PKG["root"], "c", "d", "pkg2")
    if not os.path.isdir(b):
        shutil.copytree(a, b)
    ablate.load_package(a)
    ablate.load_package(b)
    assert a not in sys.path and b in sys.path
    for name, m in list(sys.modules.items()):
        assert not (getattr(m, "__file__", None) or "").startswith(a + os.sep), name
    assert sys.modules["submission"].__file__ == os.path.join(b, "submission.py")


def test_graph_standins():
    _need_torch()
    mod = ablate.load_package(_dummy_package())
    log = ablate._instrument(mod, disable=True)
    for name in ablate.SPECIALISED_GRAPH_CLASSES:
        obj = getattr(mod, name)(None)
        assert obj.ok is False and obj.ok0 is False, name
    assert [e["graph"] for e in log] == list(ablate.SPECIALISED_GRAPH_CLASSES)
    assert getattr(mod, "GraphedCtxNormWrapper") is not None


def test_runtime_report_old_wrappers():
    """runtime_report must not need attributes that only newer whole-step wrappers have."""
    whole = types.SimpleNamespace(ok=True, _rec=None)           # e.g. a 2026-10-03 wrapper: no _kd2 .. _kd8
    m = types.SimpleNamespace(prior=1, ke_flow=1, cw=1)
    exp = types.SimpleNamespace(device="cuda", model=m, _whole=whole, _warm=None)
    r = ablate.runtime_report(exp)
    assert r["whole_step_graph"] and r["graph_components"]["kd2"] is None
    assert ablate.expectation_problems(dict(FINAL), dict(r, graph_log=[]), ["kd3"])


# --------------------------------------------------------------------------- smoke (slow)
def _tree_hash(d):
    h = hashlib.sha256()
    for root, dirs, files in sorted(os.walk(d)):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(root, f)
            h.update(os.path.relpath(p, d).encode())
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def _smoke(pid, steps=3, trajectories=2):
    import torch
    model, report = ablate.build_model(_dummy_package(), PRESETS[pid], device="cpu")
    g = torch.Generator().manual_seed(1)
    has_bounds = None
    for _ in range(trajectories):
        model.reset_ttt_state()
        prev = None
        for _ in range(steps):
            x = 0.3 * torch.randn(1, 20, 32, 64, 3, generator=g)
            pred, info = model.ttt_step(x, prev)
            assert tuple(pred.shape) == (1, 20, 32, 64, 3) and bool(torch.isfinite(pred).all()), pid
            assert "adapt_loss" in info
            hb = "lower" in info
            assert ("upper" in info) == hb
            has_bounds = hb if has_bounds is None else has_bounds
            assert hb == has_bounds, pid                     # all-or-nothing over the run
            if hb:
                lo, up = info["lower"], info["upper"]
                assert lo.shape == pred.shape and up.shape == pred.shape
                assert bool(torch.isfinite(lo).all() and torch.isfinite(up).all() and (lo <= up).all()), pid
            prev = 0.3 * torch.randn(1, 20, 32, 64, 3, generator=g)
    cfg = ablate.resolve_preset(PRESETS[pid])[1]
    assert has_bounds == bool(cfg["bounds"]), pid
    assert report["lsk_error"] is None and report["problems"] == [], (pid, report)
    assert report["eager_components"]["lsk"] == bool(cfg["lsk_head"]), pid
    assert report["eager_components"]["ke_flow"] == bool(cfg["ke_flow"]), pid
    assert report["eager_components"]["cond_width"] == bool(cfg["cond_width"]), pid
    assert report["eager_components"]["aw"] == bool(cfg["cb_adapt"]), pid


def test_smoke_all_presets_cpu():
    _need_slow()
    pkg = _dummy_package()
    before = _tree_hash(pkg)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)       # one live model at a time: no warning expected here
        for pid in sorted(PRESETS):
            _smoke(pid)
    assert _tree_hash(pkg) == before, "the package dir was modified"


def test_second_live_model_warns():
    _need_slow()
    pkg = _dummy_package()
    m1, _ = ablate.build_model(pkg, None, device="cpu")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m2, _ = ablate.build_model(pkg, None, device="cpu")
    assert any("still referenced" in str(x.message) for x in w)
    del m1, m2


def test_run_preset_local_eval_cpu():
    _need_slow()
    pkg = _dummy_package()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-B", os.path.join(EXP, "run_preset.py"), "--package", pkg, "--device", "cpu",
                        "lb0924_k3f_kd3", "l00_backbone"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    rows = [ln for ln in r.stdout.splitlines() if ln.startswith(("lb0924_k3f_kd3", "l00_backbone"))]
    assert len(rows) == 2 and all("-" not in ln.split()[1:9] for ln in rows), r.stdout
    assert "never ran" in r.stdout          # the kit example data has 2 windows per trajectory


def test_shipped_weights_cpu():
    """The final model with the shipped weights, on CPU, through run_preset.py and examples/run_ttt.py."""
    _need_slow()
    if not os.path.isdir(WEIGHTS):
        _skip("no shipped weights")
    import make_package
    d = tempfile.mkdtemp(prefix="ablate_shipped_")
    try:
        pkg = make_package.make_package(os.path.join(d, "a", "b", "pkg"), WEIGHTS, quiet=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        js = os.path.join(d, "r.json")
        r = subprocess.run([sys.executable, "-B", os.path.join(EXP, "run_preset.py"), "--package", pkg,
                            "--device", "cpu", "--json", js, "lb0927_final"], capture_output=True, text=True, env=env)
        assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
        import json
        res = json.load(open(js))[0]
        assert res["status"] == "ok" and len(res["scores"]) == 5 and res["report"]["problems"] == [], res["report"]
        assert all(0 < v <= 100 for v in res["scores"].values()), res["scores"]
        r = subprocess.run([sys.executable, "-B", os.path.join(ablate.REPO, "examples", "run_ttt.py"),
                            "--package", pkg, "--device", "cpu"], capture_output=True, text=True, env=env)
        assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
        rows = [ln.split() for ln in r.stdout.splitlines() if ln.endswith(tuple("0123456789")) and ".h5" in ln]
        assert len(rows) == 4, r.stdout                     # 2 example trajectories x 2 windows
        for row in rows:
            rel, cov, width = map(float, row[2:5])
            assert 0 <= rel < 10 and 0 <= cov <= 1 and width > 0, row
    finally:
        shutil.rmtree(d)


if __name__ == "__main__":
    pytest = None                           # plain runner: skips are reported by _Skip
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for n, f in tests:
        try:
            f()
            print("PASS", n)
        except _Skip as e:
            skipped += 1
            print("SKIP", n, e)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL", n, repr(e)[:2000])
    print("%d passed, %d skipped, %d failed" % (len(tests) - failed - skipped, skipped, failed))
    raise SystemExit(1 if failed else 0)
