"""Component selection for the final Track 2 package.

Loads the final package's `submission.py` (unmodified, from a package dir = the final source
dir plus its four weight files), applies CONFIG overrides from a preset, checks that the
combination is one the code actually runs, and returns the TTT model from the package's own
`get_ttt_model`. Nothing in the package dir is written. Import it with `experiments/` on
sys.path:

    import sys; sys.path.insert(0, "experiments")
    import ablate
    preset = ablate.load_preset("lb0924_k3f_kd3")
    model, report = ablate.build_model("/output/package", preset, device="cuda")

How the package reads CONFIG (why overrides are applied the way they are):
  * `CONFIG` is a module-level dict in `submission.py`. `get_ttt_model` reads it at call time,
    passes the same dict object on as `model.cfg`, and the graph wrappers read `model.cfg` while
    they capture (inside `get_ttt_model`). So overrides must be written into `submission.CONFIG`
    after import and before `get_ttt_model`; changing CONFIG after the model is built has no
    effect on the captured CUDA graphs.
  * `get_ttt_model` also writes one key into CONFIG (`cond_width_ckpt_path`). A module object is
    therefore used for one model only; `load_package` imports a fresh copy every time.
  * The package's own modules do `from submission import ...` (cuda_graph_*, kd3/kd4/kd6,
    t2_cw_graph_body). The file must be imported under the module name `submission`, as Codabench
    and `final_submission/solution.py` do. The starting kit's `local_eval.py` imports it as
    `participant_submission`, which makes those modules import a second, unmodified copy, so
    this module never loads it that way.
  * Many components run only inside the whole-step CUDA graph (windows >= 2 on a CUDA device).
    The graph is built only when its scope holds (cond_width, bounds, ctr, advect, gain,
    e_blend_w, prior and the energy net all on). Outside that scope, and on a CPU-only machine,
    those components are silently inactive. `check_overrides` rejects presets that ask for such a
    component where it cannot run; `runtime_report` reads back what was actually built, and
    `expectation_problems` lists components the CONFIG asks for that did not come up.
  * The per-stage graphs hard-code the shipped values of some keys (GRAPH_SPECIALISED, e.g.
    bias, gain, ctr_basis2) and are captured without re-reading them. For a preset that changes
    one of those keys, those graph classes are replaced by disabled stand-ins in the freshly
    loaded module before `get_ttt_model` runs, so nothing is captured under that CONFIG and every
    step runs the eager code (`force_eager` then checks that nothing was attached).
  * On CPU the packed fp16 core runs in fp32 (ship_b5x2_models falls back), so fp16_core has no
    numerical effect there, and no CUDA graph exists: CPU runs check plumbing, not scores.
"""
from __future__ import annotations

import ast
import copy
import gc
import glob
import hashlib
import importlib.util
import json
import os
import sys
import warnings
import weakref
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FINAL_REALPDE = os.path.join(REPO, "final_submission", "realpde")
FINAL_SOURCE = os.path.join(
    FINAL_REALPDE, "submissions",
    "2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk-recr-kd8-r97",
    "source")
PRESET_DIR = os.path.join(HERE, "presets")
WEIGHT_FILES = ("model.pth", "prior.npz", "ke_net_flow.pt", "cond_width_weights.pt")

# Keys `get_ttt_model` adds to CONFIG itself; never part of a preset.
_RUNTIME_KEYS = ("cond_width_ckpt_path",)


# --------------------------------------------------------------------------- #
# CONFIG of a package, read without importing it (no torch needed).
# --------------------------------------------------------------------------- #
def read_config(source_dir: str = FINAL_SOURCE) -> Dict[str, Any]:
    """The module-level CONFIG dict of `<source_dir>/submission.py`, evaluated from its AST.

    The dict uses only literals, `[x] * n` and `list(range(...))`, so it is evaluated with no
    builtins except `list` and `range`."""
    path = os.path.join(source_dir, "submission.py")
    with open(path) as f:
        tree = ast.parse(f.read(), path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CONFIG" for t in node.targets):
            code = compile(ast.Expression(node.value), path, "eval")
            return eval(code, {"__builtins__": {}, "list": list, "range": range})  # noqa: S307
    raise ValueError("no module-level CONFIG in %s" % path)


# --------------------------------------------------------------------------- #
# Components: name -> the CONFIG values that switch it off.
# --------------------------------------------------------------------------- #
# kind:  "pred"      changes the prediction (rel_l2 / tke / mvpe, and sps through it)
#        "interval"  changes only the bounds (sps)
#        "time"      bit-exact with its parent; only the time subscore moves
#        "time~"     near-exact (fp16 rounding-order differences only)
# where: "eager"     runs in every path (also on CPU)
#        "graph"     runs only in the whole-step CUDA graph (windows >= 2, CUDA)
#        "cuda"      a CUDA-graph / transport detail (inactive on CPU)
COMPONENTS: Dict[str, Dict[str, Any]] = {
    # pipeline core (2026-09-04 .. 09-16)
    "gain":        dict(off={"gain": False}, kind="pred", where="eager",
                        doc="energy rule: per-pixel fluctuation energy rescaled by the running input/forecast energy ratio"),
    "bias":        dict(off={"bias": False}, kind="pred", where="eager",
                        doc="running mean innovation added to the prediction"),
    "bounds":      dict(off={"bounds": False}, kind="interval", where="eager",
                        doc="SPS intervals from the running innovation variance (off: no lower/upper, scorer default band)"),
    "advect":      dict(off={"advect": False}, kind="pred", where="eager",
                        doc="last input frame shifted by the measured convection speed, blended into the fluctuation shape"),
    "prior":       dict(off={"prior": False}, kind="pred", where="eager",
                        doc="(Re, AoA)-conditioned prior for the running statistics (prior.npz)"),
    "ke_flow":     dict(off={"ke_flow": None}, kind="pred", where="eager",
                        doc="learned energy net correcting the energy rule (ke_net_flow.pt)"),
    "ctr":         dict(off={"ctr": False}, kind="interval", where="eager",
                        doc="intervals centred on a second, un-inflated centre with its own bias and variance"),
    "ctr_basis2":  dict(off={"ctr_basis2": False}, kind="interval", where="eager",
                        doc="two-vector basis for that centre (blend shape + advected fluctuation)"),
    # 2026-09-20
    "e_blend":     dict(off={"e_blend_w": None}, kind="pred", where="eager",
                        doc="energy target blended with the running mean of revealed target energy (weight 0.85)"),
    "mu_vp":       dict(off={"mu_vp": False}, kind="pred", where="eager",
                        doc="variance-preserving bias term"),
    # interval widths and centre (2026-09-20 .. 09-25)
    "cond_width":  dict(off={"cond_width": False}, kind="interval", where="eager",
                        doc="learned conditional interval widths (cond_width_weights.pt)"),
    "cb":          dict(off={"cb_blend_w": None}, kind="interval", where="eager",
                        doc="CB: interval centre moved 0.75 of the way toward the raw backbone output"),
    "cb_w0":       dict(off={"cb_blend_w0": None}, kind="interval", where="eager",
                        doc="CB weight 1.0 on window 0, where no target has been revealed yet"),
    "aw":          dict(off={"cb_adapt": None}, kind="interval", where="eager",
                        doc="AW: CB weight chosen per trajectory from a grid, by the score on revealed windows"),
    "kw":          dict(off={"kw_adapt": None}, kind="interval", where="graph",
                        doc="per-trajectory interval-width multiplier per lead group (part of AX)"),
    "eb":          dict(off={"eb_adapt": None}, kind="pred", where="graph",
                        doc="per-trajectory choice of the energy-blend weight (part of AX; grid widened with AXE)"),
    "br":          dict(off={"br_adapt": None}, kind="pred", where="graph",
                        doc="per-trajectory choice of the advection-blend ridge, jointly with eb (part of AXE)"),
    "gm20":        dict(off={"gain_max": 5.0}, kind="pred", where="eager",
                        doc="energy gain clip raised from 5 to 20"),
    "msk":         dict(off={"exact_mask": False}, kind="interval", where="eager",
                        doc="the scorer's exact mask |t| > 1e-6 in the width-net error map and the per-trajectory scorers"),
    "scan_warm_only": dict(off={"uc_warm": 4, "uc_every": 4}, kind="pred", where="eager",
                           doc="convection-speed scan on windows 0-1 only (off: every 4th window, as before 09-23)"),
    # backbone numerics / online backbone correction
    "fp16_core":   dict(off={"fp16_core": False}, kind="pred", where="eager",
                        doc="packed backbone in fp16 (off: TF32 packing; also disables KD3 and everything that needs it)"),
    "lsk":         dict(off={"lsk_head": None, "lsk_rec_reset": False}, kind="pred", where="eager",
                        doc="LSK: online least-squares correction of the backbone output in an 8-dim subspace of the head weights"),
    "lsk_rec_reset": dict(off={"lsk_rec_reset": False}, kind="pred", where="graph",
                          doc="a regime-change reset (REC) also resets the LSK moments"),
    "rec":         dict(off={"rec_detect": None, "lsk_rec_reset": False}, kind="pred", where="graph",
                        doc="REC: regime-change detector with partial state reset (does not fire on normal streams)"),
    "lowke":       dict(off={"tka_lowke": None}, kind="pred", where="graph",
                        doc="LOWKE: damp the emitted fluctuation where its own energy is low"),
    # execution only (2026-09-21 .. 09-27)
    "kd2":         dict(off={"kd2_adapt": False}, kind="time", where="cuda",
                        doc="KD2: the per-trajectory selectors fused into fewer kernels"),
    "kd3":         dict(off={"kd3_step": False}, kind="time", where="cuda",
                        doc="KD3: fewer kernels in the whole-step graph (REC, KD5, KD6, KD7 and KD8 need it)"),
    "kd5":         dict(off={"kd5_near": False}, kind="time~", where="cuda",
                        doc="KD5: convolutions rewritten as GEMMs (fp16 rounding-order differences only)"),
    "kd4":         dict(off={"kd4_w01": False}, kind="time", where="cuda", doc="KD4: fewer kernels in the window-0/1 graphs"),
    "kd6":         dict(off={"kd6_concurrency": False, "kd7": False, "kd8": False}, kind="time", where="cuda",
                        doc="KD6: backbone-independent work on side CUDA streams inside the whole-step graph"),
    "kd7":         dict(off={"kd7": False, "kd8": False}, kind="time", where="cuda",
                        doc="KD7: long elementwise chains fused into compiled kernels"),
    "kd8":         dict(off={"kd8": False}, kind="time", where="cuda", doc="KD8: more work moved to side streams"),
    "ipc":         dict(off={"ipc_one_storage": False}, kind="time", where="cuda",
                        doc="IPC: outputs returned as views of one buffer when inputs arrive through shared memory"),
    "ipw":         dict(off={"ipc_prewarm": False}, kind="time", where="cuda",
                        doc="IPW: untimed warm-up of multiprocessing transport in a worker process"),
    "bvr":         dict(off={"ipc_byvalue": False}, kind="time", where="cuda",
                        doc="BVR: outputs pickled by value when running in a worker process"),
    "cdio":        dict(off={"cuda_direct_io": False}, kind="time", where="cuda",
                        doc="CDIO: no host staging when the caller passes CUDA tensors"),
}

# Settings the code cannot switch off at inference (code differences between older uploads and
# the final source, or properties of the weights).
NOT_SWITCHABLE = {
    "weights": "backbone (n1 SimVP / b5 / sim pretraining / soups / 3-member ensemble), prior.npz, energy net and "
               "width net come from the weight files; an older upload needs its own weights (preset `training`)",
    "w01_kd_graphs": "window-0/1 CUDA graphs and the kernel diet (09-23 on) have no CONFIG flag; both are bit-exact",
    "fp16_width_net": "the whole-step graph runs the width net in fp16 (09-23 on, not bit-exact; earlier uploads used "
                      "fp32); no CONFIG flag",
    "wholestep_graph": "the whole-step graph (09-22 on) has no CONFIG flag; it is built whenever its scope holds and "
                       "CUDA is available; its outputs match the per-stage graphs (bounds within 1e-6)",
}


def component_overrides(names: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for n in names:
        if n not in COMPONENTS:
            raise ValueError("unknown component %r (known: %s)" % (n, ", ".join(sorted(COMPONENTS))))
        out.update(copy.deepcopy(COMPONENTS[n]["off"]))
    return out


# --------------------------------------------------------------------------- #
# Dependency rules.
# --------------------------------------------------------------------------- #
WHOLE_STEP_SCOPE = ("cond_width", "bounds", "ctr", "advect", "gain", "prior")
_GRAPH_ONLY_PRED = (("kw_adapt", "kw"), ("eb_adapt", "eb"), ("br_adapt", "br"),
                    ("rec_detect", "rec"), ("tka_lowke", "lowke"))

# CONFIG keys whose shipped value the per-stage CUDA graphs (cuda_graph_tail / _pre / _megatail /
# _bounds, cond_width_graph, and the whole-step and W01 graphs composed from them) hard-code
# ("bias = True for this shipped config", ...). These graphs are captured whenever their state
# preconditions hold, without re-checking these keys, so with any of them changed the graphed
# steps would not follow CONFIG. Compiled from those modules' code and comments.
GRAPH_SPECIALISED = ("advect", "gain", "bias", "bounds", "prior", "ke_flow", "ke_net", "ke_net_online",
                     "er_online", "ctr", "ctr_bias", "ctr_basis2", "ctr_lead", "pool_t", "mu_alpha_min",
                     "mean_mode", "band_split", "uc_fixed", "uc_guard", "blend")
# the graph classes replaced by disabled stand-ins when a preset changes one of those keys
SPECIALISED_GRAPH_CLASSES = ("GraphedTailWrapper", "GraphedPreWrapper", "GraphedMegaWrapper", "GraphedBoundsWrapper",
                             "GraphedCondWidthWrapper", "GraphedWholeStepWrapper", "GraphedWarmupWrapper")
# graph classes that are plain functions of their inputs; always kept
GENERIC_GRAPH_CLASSES = ("GraphedCtxNormWrapper", "GraphedKeFlowWrapper", "GraphedMaskWrapper", "GraphedScanUcWrapper")


def _on(cfg, k) -> bool:
    return cfg.get(k) not in (None, False)


def _ke_flow_on(cfg) -> bool:
    """`_load_ke_flow` returns None for ke_flow None / "" or ke_flow_scale 0."""
    return bool(cfg.get("ke_flow")) and float(cfg.get("ke_flow_scale", 1.0)) != 0.0


def needs_eager(cfg: Dict[str, Any], base: Optional[Dict[str, Any]] = None) -> List[str]:
    """Keys in GRAPH_SPECIALISED whose effect differs from the final CONFIG (empty = graphs are valid)."""
    base = read_config() if base is None else base
    out = [k for k in GRAPH_SPECIALISED if k != "ke_flow" and cfg.get(k) != base.get(k)]
    if _ke_flow_on(cfg) != _ke_flow_on(base):
        out.append("ke_flow")
    return out


def whole_step_scope_ok(cfg: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Static part of GraphedWholeStepWrapper._check_preconditions (CONFIG only)."""
    why = [k + " off" for k in WHOLE_STEP_SCOPE if not cfg.get(k)]
    if cfg.get("e_blend_w") is None:
        why.append("e_blend_w off")
    if not _ke_flow_on(cfg):
        why.append("ke_flow off")
    for k in ("uc_fixed", "uc_guard", "mean_mode", "er_online"):
        if _on(cfg, k):
            why.append(k + " must be off")
    if cfg.get("ke_net") and cfg.get("ke_net_online"):
        why.append("ke_net_online must be off")
    return (not why), why


def graph_scope(cfg: Dict[str, Any], base: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
    """Whether the whole-step graph can be built for `cfg` (static scope and no forced eager path)."""
    ok, why = whole_step_scope_ok(cfg)
    eager = needs_eager(cfg, base)
    if eager:
        ok, why = False, why + ["graphs specialised to the shipped %s" % "/".join(eager)]
    return ok, why


def _kd3_static(cfg) -> List[str]:
    """Static part of kd3_step.KD3State's checks (the rest depends on the weights)."""
    why = []
    if cfg.get("fp16_core") not in (True, "all"):
        why.append("fp16_core True / 'all'")
    if not cfg.get("mu_vp") or not cfg.get("ctr_basis2", True):
        why.append("mu_vp and ctr_basis2")
    if cfg.get("blend") != "region":
        why.append("blend 'region'")
    return why


def _kd2_static(cfg) -> List[str]:
    return [] if (_on(cfg, "cb_adapt") and _on(cfg, "kw_adapt") and _on(cfg, "eb_adapt")) else \
        ["cb_adapt, kw_adapt and eb_adapt"]


def _kd6_static(cfg) -> List[str]:
    """Static part of kd6_conc.covered(): KD2 + KD3 path, REC, eb + br, mu_vp, CB."""
    why = []
    if not cfg.get("kd2_adapt") or _kd2_static(cfg):
        why.append("KD2 (kd2_adapt with cb_adapt, kw_adapt, eb_adapt)")
    if not cfg.get("kd3_step") or _kd3_static(cfg):
        why.append("KD3")
    if not _on(cfg, "rec_detect"):
        why.append("rec_detect")
    if not (_on(cfg, "eb_adapt") and _on(cfg, "br_adapt")):
        why.append("eb_adapt and br_adapt")
    if not cfg.get("mu_vp"):
        why.append("mu_vp")
    if not _on(cfg, "cb_blend_w"):
        why.append("cb_blend_w")
    return why


def dependency_errors(cfg: Dict[str, Any], base: Dict[str, Any]) -> List[str]:
    """Combinations that crash, or that leave a prediction- or interval-changing component silently off."""
    err: List[str] = []
    on = lambda k: _on(cfg, k)  # noqa: E731
    # get_ttt_model would crash
    if on("eb_adapt") and cfg.get("e_blend_w") is None:
        err.append("eb_adapt needs e_blend_w (get_ttt_model calls float(e_blend_w))")
    if on("cb_blend_w"):
        w = cfg["cb_blend_w"]
        if not isinstance(w, (list, tuple)) or len(w) != 20:
            err.append("cb_blend_w must be a list of 20 values (or 20 pairs); a scalar is not accepted")
    # silently ignored by get_ttt_model
    if on("cb_adapt") and not on("cb_blend_w"):
        err.append("cb_adapt (AW) needs cb_blend_w (CB)")
    if on("cb_blend_w0") and not on("cb_blend_w"):
        err.append("cb_blend_w0 needs cb_blend_w (CB)")
    if on("kw_adapt") and not on("cb_adapt"):
        err.append("kw_adapt needs cb_adapt (AW)")
    if on("br_adapt") and not on("eb_adapt"):
        err.append("br_adapt needs eb_adapt")
    if on("lsk_rec_reset") and not (on("lsk_head") and on("rec_detect")):
        err.append("lsk_rec_reset needs lsk_head and rec_detect")
    # interval pieces need bounds (and the centre)
    if not cfg.get("bounds"):
        for k in ("ctr", "cond_width", "cb_blend_w"):
            if on(k):
                err.append("%s needs bounds" % k)
    if on("cb_blend_w") and not cfg.get("ctr"):
        err.append("cb_blend_w (CB) needs ctr (it shifts the decoupled centre)")
    # components that exist only inside the whole-step CUDA graph
    scope_ok, why = graph_scope(cfg, base)
    if not scope_ok:
        for key, name in _GRAPH_ONLY_PRED:
            if on(key):
                err.append("%s (%s) runs only in the whole-step graph, which this combination does not build (%s)"
                           % (key, name, ", ".join(why)))
    kd3_why = _kd3_static(cfg)
    if on("rec_detect") and (not cfg.get("kd3_step") or kd3_why):
        err.append("rec_detect (REC) needs kd3_step, which needs %s" % (", ".join(kd3_why) or "kd3_step True"))
    return err


def dependency_warnings(cfg: Dict[str, Any], base: Dict[str, Any]) -> List[str]:
    """Flags that will be inert in this combination (execution-only, or no effect)."""
    warn: List[str] = []
    on = lambda k: _on(cfg, k)  # noqa: E731
    scope_ok, why = graph_scope(cfg, base)
    if not scope_ok:
        live = [k for k in ("kd2_adapt", "kd3_step", "kd5_near", "kd6_concurrency", "kd7", "kd8",
                            "cuda_direct_io") if on(k)]
        if live:
            warn.append("whole-step graph not built (%s): %s inert" % (", ".join(why), ", ".join(live)))
        return warn + _other_warnings(cfg)
    kd3_why = _kd3_static(cfg)
    if on("kd2_adapt") and _kd2_static(cfg):
        warn.append("kd2_adapt inert (needs %s)" % ", ".join(_kd2_static(cfg)))
    if on("kd3_step") and kd3_why:
        warn.append("kd3_step inert (needs %s)" % ", ".join(kd3_why))
    if on("kd5_near") and not (on("kd3_step") and not kd3_why):
        warn.append("kd5_near inert (needs kd3_step)")
    kd6_why = _kd6_static(cfg)
    if on("kd6_concurrency") and kd6_why:
        warn.append("kd6_concurrency inert, and with it kd7 / kd8 (KD6 needs %s)" % ", ".join(kd6_why))
    elif on("kd7") and not on("kd6_concurrency"):
        warn.append("kd7 inert (needs kd6_concurrency)")
    elif on("kd7") and not on("br_adapt"):
        warn.append("kd7 inert (needs br_adapt)")
    if on("kd8") and not on("kd7"):
        warn.append("kd8 inert (needs kd7)")
    return warn + _other_warnings(cfg)


def _other_warnings(cfg) -> List[str]:
    warn = []
    on = lambda k: _on(cfg, k)  # noqa: E731
    if on("ctr_basis2") and not cfg.get("ctr"):
        warn.append("ctr_basis2 inert without ctr")
    if on("mu_vp") and not cfg.get("bias"):
        warn.append("mu_vp inert without bias")
    if not cfg.get("gain"):
        for k in ("ke_flow", "e_blend_w"):
            if on(k):
                warn.append("%s inert without gain" % k)
    if on("exact_mask") and not (on("cond_width") or on("cb_adapt")):
        warn.append("exact_mask inert without cond_width / cb_adapt / kw_adapt")
    if on("lsk_head") and not on("fp16_core"):
        warn.append("lsk_head needs a packed backbone: works with the 3-member weights, switches itself off "
                    "with a single TF32 backbone (see lsk_error in the runtime report)")
    return warn


def check_overrides(overrides: Dict[str, Any], base: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], List[str]]:
    """Validate `overrides` against the final CONFIG. Returns (resolved CONFIG, warnings); raises
    ValueError on unknown or runtime-only keys and on any `dependency_errors`."""
    base = read_config() if base is None else base
    bad = [k for k in overrides if k in _RUNTIME_KEYS]
    if bad:
        raise ValueError("runtime-only keys cannot be overridden: %s" % bad)
    unknown = sorted(set(overrides) - set(base))
    if unknown:
        raise ValueError("unknown CONFIG keys: %s" % ", ".join(unknown))
    cfg = copy.deepcopy(base)
    cfg.update(copy.deepcopy(overrides))
    err = dependency_errors(cfg, base)
    if err:
        graph_only = [n for k, n in _GRAPH_ONLY_PRED if _on(cfg, k)]
        hint = ("\n  (to run without the whole-step graph, also switch off: %s)" % " ".join(graph_only)
                if graph_only and not graph_scope(cfg, base)[0] else "")
        raise ValueError("invalid combination:\n  - " + "\n  - ".join(err) + hint)
    return cfg, dependency_warnings(cfg, base)


def config_diff(cfg: Dict[str, Any], base: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Any, Any]]:
    base = read_config() if base is None else base
    return [(k, base.get(k), cfg.get(k)) for k in sorted(set(base) | set(cfg))
            if k not in _RUNTIME_KEYS and base.get(k) != cfg.get(k)]


def components_off(cfg: Dict[str, Any], base: Optional[Dict[str, Any]] = None) -> List[str]:
    """Components whose CONFIG keys all sit at their off value in `cfg` but not in `base`."""
    base = read_config() if base is None else base
    out = []
    for name, c in COMPONENTS.items():
        if all(cfg.get(k) == v for k, v in c["off"].items()) and any(base.get(k) != v for k, v in c["off"].items()):
            out.append(name)
    return out


# --------------------------------------------------------------------------- #
# Presets.
# --------------------------------------------------------------------------- #
def preset_ids() -> List[str]:
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(PRESET_DIR, "*.json")))


def load_preset(ref: str) -> Dict[str, Any]:
    """A preset by id (experiments/presets/<id>.json) or by path."""
    path = ref if ref.endswith(".json") and os.path.isfile(ref) else os.path.join(PRESET_DIR, ref + ".json")
    if not os.path.isfile(path):
        raise ValueError("no preset %r (see run_preset.py --list)" % ref)
    with open(path) as f:
        p = json.load(f)
    if p.get("id") and not ref.endswith(".json") and p["id"] != ref:
        raise ValueError("preset file %s has id %r" % (path, p["id"]))
    bad = sorted(set(p.get("expect_active", [])) - set(EXPECTABLE))
    if bad:
        raise ValueError("preset %s: unknown expect_active names %s (known: %s)" % (path, bad, ", ".join(EXPECTABLE)))
    return p


# names a preset may list in `expect_active`: runtime_report's graph_components keys, plus kd4 (W01 graphs)
EXPECTABLE = ("kw", "eb", "br", "rec", "lowke", "kd2", "kd3", "kd6", "kd7", "kd8", "kd4")


def preset_overrides(preset: Dict[str, Any]) -> Dict[str, Any]:
    """CONFIG overrides of a preset: its `off` components first, then its raw `inference` keys."""
    out = component_overrides(preset.get("off", []))
    out.update(copy.deepcopy(preset.get("inference", {})))
    return out


def resolve_preset(preset: Dict[str, Any], extra_off: Optional[List[str]] = None,
                   extra: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """(overrides, resolved CONFIG, warnings) for a preset plus optional extra components off / keys."""
    ov = preset_overrides(preset)
    ov.update(component_overrides(extra_off or []))
    ov.update(copy.deepcopy(extra or {}))
    cfg, warn = check_overrides(ov)
    return ov, cfg, warn


# --------------------------------------------------------------------------- #
# Loading the package and building the model.
# --------------------------------------------------------------------------- #
_LOADED: List[str] = []          # package dirs this process has imported


def _package_module_names(pkg_dir: str) -> List[str]:
    names = []
    for entry in os.listdir(pkg_dir):
        p = os.path.join(pkg_dir, entry)
        if entry.endswith(".py"):
            names.append(entry[:-3])
        elif os.path.isdir(p) and os.path.isfile(os.path.join(p, "__init__.py")):
            names.append(entry)
    return names


def check_package_dir(pkg_dir: str, need_weights: bool = True) -> None:
    pkg_dir = os.path.abspath(pkg_dir)
    if not os.path.isfile(os.path.join(pkg_dir, "submission.py")):
        raise FileNotFoundError("no submission.py in %s" % pkg_dir)
    if need_weights:
        missing = [w for w in WEIGHT_FILES if not os.path.isfile(os.path.join(pkg_dir, w))]
        if missing:
            raise FileNotFoundError("package %s lacks %s (a package = the final source dir + the four weight files; "
                                    "see experiments/README.md)" % (pkg_dir, ", ".join(missing)))
    check_package_location(pkg_dir)


def check_package_location(pkg_dir: str) -> None:
    """Refuse a package location whose <pkg>/../../../src would shadow the package (works before it exists)."""
    pkg_dir = os.path.abspath(pkg_dir)
    # submission.py puts <pkg>/../../../src on sys.path AHEAD of the package dir when that dir has a
    # load_baseline.py and the package dir is already on sys.path; the package would then import repo code.
    trap = os.path.join(os.path.abspath(os.path.join(pkg_dir, os.pardir, os.pardir, os.pardir)), "src")
    if os.path.isfile(os.path.join(trap, "load_baseline.py")):
        raise RuntimeError("package dir %s sits three levels below a tree with src/load_baseline.py (%s); "
                           "submission.py would import that src/ instead of its own modules. Use a package dir "
                           "outside the repo, or nested deeper (the chain uses <out>/build/w/x/pkg)." % (pkg_dir, trap))


def source_mismatches(pkg_dir: str, source_dir: str = FINAL_SOURCE) -> List[str]:
    """Code files (.py, recursively) that differ between `pkg_dir` and the final source, or exist in only one."""
    def files(d):
        out = {}
        for root, dirs, fs in os.walk(d):
            dirs[:] = [x for x in dirs if x != "__pycache__"]
            for f in fs:
                if f.endswith(".py"):
                    p = os.path.join(root, f)
                    with open(p, "rb") as fh:
                        out[os.path.relpath(p, d)] = hashlib.sha256(fh.read()).hexdigest()
        return out
    a, b = files(os.path.abspath(pkg_dir)), files(source_dir)
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def _under(path: Optional[str], d: str) -> bool:
    return bool(path) and os.path.abspath(path).startswith(d + os.sep)


def _purge(dirs: List[str], names: List[str]) -> None:
    """Drop from sys.modules every module whose file lies under one of `dirs`, plus `names`."""
    for key in list(sys.modules):
        f = getattr(sys.modules[key], "__file__", None)
        if any(_under(f, d) for d in dirs) or key in names or key.split(".")[0] in names:
            del sys.modules[key]


def check_module_origins(pkg_dir: str) -> None:
    """Every imported module named like a package file must come from the package dir."""
    pkg_dir = os.path.abspath(pkg_dir)
    for name in _package_module_names(pkg_dir) + ["submission"]:
        m = sys.modules.get(name)
        f = getattr(m, "__file__", None) if m is not None else None
        if f and not _under(f, pkg_dir):
            raise RuntimeError("module %s resolved to %s, outside the package %s" % (name, f, pkg_dir))


def load_package(pkg_dir: str):
    """Import `<pkg_dir>/submission.py` as module `submission`, fresh: modules of any package loaded
    before (by file location) and any module with a package file's name are dropped first, and the
    previous package dirs are removed from sys.path."""
    pkg_dir = os.path.abspath(pkg_dir)
    check_package_dir(pkg_dir, need_weights=False)
    _purge(_LOADED + [pkg_dir], _package_module_names(pkg_dir) + ["submission"])
    for d in _LOADED + [pkg_dir]:
        while d in sys.path:
            sys.path.remove(d)
    del _LOADED[:]
    _LOADED.append(pkg_dir)
    sys.path.insert(0, pkg_dir)
    spec = importlib.util.spec_from_file_location("submission", os.path.join(pkg_dir, "submission.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submission"] = mod
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True        # no __pycache__ inside the package dir (sibling imports included)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    check_module_origins(pkg_dir)
    return mod


class _DisabledGraph:
    """Stand-in for a graph wrapper class: constructing it captures nothing and reports not-ok."""
    ok = ok0 = ok1 = False
    capture_error = "disabled by ablate (CONFIG differs from the shipped values this graph hard-codes)"

    def __init__(self, *args, **kwargs):
        pass


def _instrument(mod, disable: bool) -> List[Dict[str, Any]]:
    """Wrap the module's graph classes so every construction is logged (name, ok, error); with
    `disable`, the specialised ones are replaced by `_DisabledGraph`. Only the module namespace changes."""
    log: List[Dict[str, Any]] = []
    for name in SPECIALISED_GRAPH_CLASSES + GENERIC_GRAPH_CLASSES:
        cls = getattr(mod, name, None)
        if cls is None:
            continue
        real = _DisabledGraph if (disable and name in SPECIALISED_GRAPH_CLASSES) else cls

        def factory(*a, _real=real, _name=name, **k):
            try:
                obj = _real(*a, **k)
            except Exception as e:  # noqa: BLE001 - logged, then re-raised to get_ttt_model's own try/except
                log.append({"graph": _name, "ok": False, "error": repr(e)[:300]})
                raise
            ok = getattr(obj, "ok", None)
            if _name == "GraphedWarmupWrapper":
                ok = getattr(obj, "ok0", ok)
            log.append({"graph": _name, "ok": bool(ok), "error": getattr(obj, "capture_error", None)})
            return obj
        setattr(mod, name, factory)
    return log


def _get_ttt_model(mod, pkg_dir, device):
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True        # get_ttt_model imports more package modules lazily
    try:
        model = mod.get_ttt_model(pkg_dir, device)
    finally:
        sys.dont_write_bytecode = old
    check_module_origins(pkg_dir)         # the lazy imports (kd2 .. kd8, rec_detect, ...) included
    return model


_LIVE: List[Any] = []            # weak reference to the last model built in this process


def _check_one_live_model() -> None:
    """The package keeps module-level state (CUDA graphs, LSK hooks, the width-net mask flag), so a
    second model in the same process can disturb a first one that is still in use."""
    if _LIVE:
        gc.collect()                  # the BVR wrapper makes the model part of a reference cycle
    if _LIVE and _LIVE[0]() is not None:
        warnings.warn("a model built earlier in this process is still referenced; build one model per process "
                      "(run_preset.py runs each preset in its own process)", RuntimeWarning, stacklevel=3)


def build_model(pkg_dir: str, preset: Optional[Dict[str, Any]] = None, device: str = "cuda",
                extra_off: Optional[List[str]] = None, extra: Optional[Dict[str, Any]] = None,
                allow_modified_source: bool = False):
    """(model, report): the package's own get_ttt_model with the preset's resolved CONFIG.

    `preset` None = the final CONFIG unchanged. The package's code must equal the final source file
    for file (sha256 of every .py; `allow_modified_source` to override, at your own risk), since
    the presets and the dependency rules describe that code. One live model per process: building a
    second one while the first is still referenced warns (module-level state is shared). `report` =
    runtime_report(model) plus the resolved diff, the warnings, the graph construction log and
    `problems` (components the CONFIG asks for that did not come up; see expectation_problems)."""
    pkg_dir = os.path.abspath(pkg_dir)
    check_package_dir(pkg_dir)
    final = read_config()
    diff_files = source_mismatches(pkg_dir)
    if diff_files and not allow_modified_source:
        raise ValueError("package %s differs from the final source in %s; evaluate it with run_preset.py "
                         "--as-packaged (its own CONFIG), or pass allow_modified_source" % (pkg_dir, ", ".join(diff_files[:8])))
    base = read_config(pkg_dir)
    if set(base) != set(final):
        raise ValueError("package %s has a different CONFIG schema than the final source; evaluate it with "
                         "run_preset.py --as-packaged" % pkg_dir)
    ov, cfg, warn = resolve_preset(preset or {}, extra_off, extra)
    _check_one_live_model()
    mod = load_package(pkg_dir)
    mod.CONFIG.clear()
    mod.CONFIG.update(copy.deepcopy(cfg))          # the full resolved CONFIG, not only the overrides
    eager = needs_eager(cfg, final)
    graph_log = _instrument(mod, disable=bool(eager))
    model = _get_ttt_model(mod, pkg_dir, device)
    if eager:
        force_eager(model)
    report = runtime_report(model)
    report.update(eager_forced_by=eager, warnings=warn, diff=config_diff(cfg, base), graph_log=graph_log)
    report["problems"] = expectation_problems(cfg, report, (preset or {}).get("expect_active", []), final)
    report["modified_source_files"] = diff_files
    _LIVE[:] = [weakref.ref(model)]
    return model, report


def force_eager(model) -> None:
    """Belt and braces after the stand-ins: make sure no graph specialised to the shipped pipeline
    values is attached, so every step runs the eager code of `GainMapImprovedModel.step` /
    `ExperimentTTT.ttt_step`. The ctx-norm backbone, mask, convection-scan and energy-net graphs
    stay (plain functions of their inputs). Only the built objects are touched, never files."""
    exp = model
    exp._whole = None
    exp._warm = None
    m = exp.model
    m._tail = m._pre = m._mega = m._boundsg = m._cwg = None
    exp.reset_ttt_state()


def build_packaged(pkg_dir: str, device: str = "cuda"):
    """(model, report) for any package dir, CONFIG untouched (e.g. a historic package built by the chain)."""
    pkg_dir = os.path.abspath(pkg_dir)
    check_package_dir(pkg_dir)
    _check_one_live_model()
    mod = load_package(pkg_dir)
    graph_log = _instrument(mod, disable=False)
    model = _get_ttt_model(mod, pkg_dir, device)
    report = runtime_report(model)
    report.update(eager_forced_by=[], warnings=[], diff=[], graph_log=graph_log, problems=[])
    _LIVE[:] = [weakref.ref(model)]
    return model, report


def runtime_report(model) -> Dict[str, Any]:
    """What the built model will actually run (read from the objects get_ttt_model built). Uses
    getattr throughout: older packages' wrappers lack some of these attributes."""
    exp = model                                      # BVR replaces exp.ttt_step by a closure; same object
    m = exp.model
    whole = getattr(exp, "_whole", None)
    warm = getattr(exp, "_warm", None)
    g = lambda o, k: getattr(o, k, None)  # noqa: E731
    r: Dict[str, Any] = {
        "compute_device": g(exp, "device"),
        "eager_components": {
            "prior": g(m, "prior") is not None,
            "ke_flow": g(m, "ke_flow") is not None,
            "cond_width": g(m, "cw") is not None,
            "cb": g(m, "cb_w") is not None,
            "cb_w0": g(m, "cb_w0") is not None,
            "aw": g(m, "cb_adapt") is not None,
            "lsk": g(m, "_lsk") is not None,
        },
        "lsk_error": g(m, "_lsk_error"),
        "rng_probe_ok": g(m, "_p7_rng_probe_ok"),
        "rng_probe_error": g(m, "_p7_rng_probe_err"),
        "whole_step_graph": bool(whole is not None and g(whole, "ok")),
        "w01_graphs": bool(warm is not None and g(warm, "ok0")),
        "stage_graphs": {k: g(m, k) is not None for k in ("_pre", "_tail", "_mega", "_boundsg", "_cwg",
                                                          "_maskg", "_scang")},
    }
    if whole is not None and g(whole, "ok"):
        r["graph_components"] = {
            "kw": g(whole, "_kw_ad") is not None,
            "eb": g(whole, "_eb_ad") is not None,
            "br": g(whole, "_br_ad") is not None,
            "rec": g(whole, "_rec") is not None,
            "lowke": bool(g(whole, "_lowke")),
            "kd2": g(whole, "_kd2"), "kd3": g(whole, "_kd3"), "kd6": g(whole, "_kd6"),
            "kd7": g(whole, "_kd7"), "kd8": g(whole, "_kd8"),
        }
        r["graph_errors"] = {k: g(whole, k) for k in
                             ("kd2_error", "kd3_error", "kd6_error", "kd7_error", "kd8_error", "rec_error")
                             if g(whole, k)}
    else:
        r["graph_components"] = {}
        r["graph_errors"] = {}
    if warm is not None:
        r["kd4"] = bool(g(warm, "_kd4"))
        if g(warm, "kd4_error"):
            r["graph_errors"]["kd4_error"] = g(warm, "kd4_error")
    return r


def expectation_problems(cfg: Dict[str, Any], report: Dict[str, Any], expect_active: List[str] = (),
                         base: Optional[Dict[str, Any]] = None) -> List[str]:
    """Components the CONFIG (or a preset's `expect_active`) asks for that are not active in `report`.

    Checked on any device: LSK. On a CUDA device, when the whole-step scope holds: the whole-step
    graph itself and its components kw / eb / br / rec / lowke; KD levels only via `expect_active`."""
    p: List[str] = []
    if _on(cfg, "lsk_head") and _on(cfg, "fp16_core") and not report["eager_components"]["lsk"]:
        p.append("lsk_head on but LSK not built: %s" % report.get("lsk_error"))
    if report.get("compute_device") != "cuda":
        return p
    if report.get("rng_probe_ok") is False:
        p.append("CUDA RNG probe failed after graph capture: %s" % report.get("rng_probe_error"))
    if not graph_scope(cfg, base)[0]:
        return p
    if not report["whole_step_graph"]:
        errs = [e for e in report.get("graph_log", []) if not e["ok"]]
        p.append("whole-step graph not built (%s)" % (errs or "no construction logged"))
        return p
    gc = report["graph_components"]
    for key, name in _GRAPH_ONLY_PRED:
        if _on(cfg, key) and not gc.get(name):
            p.append("%s on in CONFIG but inactive in the whole-step graph" % key)
    for name in expect_active:
        active = report.get("kd4") if name == "kd4" else gc.get(name)
        if not active:
            p.append("%s expected active but is not (%s)" % (name, report.get("graph_errors")))
    return p


def assemble_package(weights: Dict[str, str], out_dir: str, source_dir: str = FINAL_SOURCE) -> str:
    """Copy the final source and the four weight files {name: path} into `out_dir` (a new dir)."""
    import shutil
    if os.path.exists(out_dir):
        raise FileExistsError(out_dir)
    shutil.copytree(source_dir, out_dir, ignore=shutil.ignore_patterns("__pycache__", "README.md"))
    for name in WEIGHT_FILES:
        shutil.copyfile(weights[name], os.path.join(out_dir, name))
    check_package_dir(out_dir)
    return out_dir
