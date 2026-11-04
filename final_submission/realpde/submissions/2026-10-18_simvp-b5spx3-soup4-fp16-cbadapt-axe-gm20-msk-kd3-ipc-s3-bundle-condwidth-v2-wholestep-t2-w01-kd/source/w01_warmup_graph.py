"""W01 (2026-09-22): windows 0 and 1 of every trajectory as ONE CUDA graph replay each.

Why: from window 2 on the step is one whole-step graph replay (cuda_graph_wholestep.py),
but windows 0 and 1 ran v2's eager code: ~410 / ~270 kernel launches, 23 / 61 memcpys,
3 / 5 graph replays, 6 stream syncs and a `.cpu().numpy()` / `float()` host round trip
each -- 8x / 5x the host CPU of a steady step (experiments/w01_warmup_graph/notes.md,
Stage 0). On the eval box eager steps are CPU-bound, so this is time for free.

## How bit-exactness is kept (nothing is re-derived)

The graphs are captured FROM THE UNCHANGED EAGER CODE of those windows: inside
`torch.cuda.graph(...)` we call `ExperimentTTT.ttt_step` itself on static device
buffers, exactly as the harness would at window 0 / 1. Every op is therefore the op the
control executes, with three mechanical substitutions, each argued bit-exact:

  1. Sub-graphs the eager code replays (ctx-norm backbone, mask, scan_uc, ke_flow, tail)
     cannot be replayed inside a capture; for the duration of the capture each wrapper's
     `.graph` is swapped for a shim whose `replay()` runs that wrapper's own
     `_forward_body()` -- the very function its graph was captured from, on the same
     static buffers (the same composition the whole-step graph uses, T1-V 34/34).
     Everything around the replay (copies in, `_sync_state_in`, clones out, counters)
     is the wrapper's unchanged `forward`.
  2. Window 0's `_apply_prior` needs (Re, AoA) from numpy on the host. The host part
     (`estimate_re` / `estimate_aoa`) runs on the CPU input the harness hands us (the
     same float32 values the control reads back with `x[0].cpu().numpy()`), and the
     device part `_apply_prior_maps(aoa, log_re)` is captured with two float32 0-dim
     device scalars in place of the python floats (same float32 subtraction; the scalars
     travel in the packed H2D buffer).
  3. Window 1's `adapt_loss = float((err**2).mean())` (a host sync) is captured as the
     0-dim tensor (`_w01_loss` override) and copied into the packed output; its float32
     value comes back with the one D2H copy.
Host I/O per replay: ONE pinned H2D (x [, prev_target], scalars), ONE replay, ONE pinned
D2H (pred, lower, upper [, loss]), ONE stream sync.

## State

A capture runs the Python code once: every attribute the step assigns (running
statistics, prev_* tensors, counters, flags, the cond-width seed, ...) is left pointing
at tensors in the graph's private pool, whose CONTENTS each replay rewrites. After the
capture the wrapper diffs the `__dict__` of every object the step touches against its
post-reset state and keeps the changed entries (references into the graph pool, python
ints / bools, shallow copies of lists / dicts); after each replay those entries are
re-applied (a few dozen setattr) -- the Python state is then exactly what the eager step
would have produced, with tensors holding the same values. Window 0's python ints are
the same for every trajectory (a fresh reset precedes it); only `params` = (Re, AoA)
differs and is set from the host values. G1 is captured AFTER G0's state is applied, so
its reads of window-0 state are reads of G0's output buffers; G1 is therefore only
replayed right after G0 in the same trajectory (phase machine below). Each graph has its
own memory pool and the wrapper holds references to every tensor the restored state
points at, so no pool block is ever reused while live.

G1 also performs, in-graph, the whole-step wrapper's one-time `_seed()` (copies of the
same attributes into its buffers, counters filled with window 1's deterministic values),
so window 2 skips it; if window 2 does not take the whole-step path the pre-seed is
withdrawn (`whole.live = False`) and the control's own seeding happens later as usual.

## Phase machine (per trajectory; `reset()` sets phase 0)
  phase 0: window 0 expected. G0 used iff prev_target is None and the input is a CPU
           float32 tensor of the static shape; else -> phase -1 (eager for the rest).
  phase 1: G0 ran; G1 used iff prev_target is given (same input checks); else -1.
  phase 2: G1 ran (whole-step pre-seeded); reaching `try_step` in phase 2 means window 2
           did NOT take the whole-step path -> withdraw the pre-seed, phase -1.
 -1: off until the next reset. CPU-only boxes, CUDA inputs and any capture / self-test
     failure run v2's code unchanged.

## Self-test
At construction, after capture, a dummy 3-window trajectory (random inputs) is run both
ways -- graph replays vs the eager path -- and every output (pred, lower, upper,
adapt_loss) plus the window-2 output (which consumes the whole state) must be
bit-identical; otherwise the wrapper disables itself (ok0 = ok1 = False).
"""
from __future__ import annotations

import math

import torch

STATIC_SHAPE = (1, 20, 32, 64, 3)
N_ELEM = 1 * 20 * 32 * 64 * 3



def _ipc_one_storage(mode, x) -> bool:
    """IPC (experiments/ipc_outproc): return pred / lower / upper as views of ONE clone? (values identical either way)
    mode False -> no; True -> yes; "auto" -> only if `x` (this step's input_norm) is a CPU torch tensor received through
    torch's shared-memory reduction with the "file_descriptor" strategy (multiprocessing Connection / Queue harness)."""
    if mode is True:
        return True
    if mode != "auto":
        return False
    try:
        import torch.multiprocessing as _tmp
        return (isinstance(x, torch.Tensor) and x.device.type == "cpu" and bool(x.is_shared())
                and _tmp.get_sharing_strategy() == "file_descriptor")
    except Exception:  # noqa: BLE001 - any doubt -> the parent's three clones
        return False

class _BodyAsReplay:
    """Stand-in for a wrapper's CUDAGraph during OUR capture: replay() runs the body."""

    def __init__(self, w):
        self.w = w

    def replay(self):
        self.w._forward_body()


def _is_container(v):
    return isinstance(v, (list, dict))


def _shallow(v):
    return list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v)


def _same(a, b):
    if a is b:
        return True
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(x is y for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(a[k] is b[k] for k in a)
    if isinstance(a, (bool, int, float, str, type(None))) and isinstance(b, (bool, int, float, str, type(None))):
        return type(a) is type(b) and a == b
    return False


class GraphedWarmupWrapper:
    def __init__(self, exp, warmup_iters: int = 2):
        self.exp = exp
        self.model = exp.model
        self.ok0 = self.ok1 = False
        self.phase = -1
        self.capture_error = None
        self.selftest = None
        self.n_g0 = self.n_g1 = 0
        self._preseed = False
        # P6 fix 1: a runtime replay failure (after a passing self-test) disables W01 for
        # the rest of the run -- persists across `reset()` calls, unlike `phase`.
        self._disabled = False
        try:
            self._check_preconditions()
            self._alloc()
            self._capture_all(warmup_iters)
            self.ok0 = True
            try:
                self._capture_g1()
                self.ok1 = True
            except Exception as e:  # noqa: BLE001 - G1 failure -> G0 only
                self.ok1 = False
                self.g1 = None
                self.capture_error = "g1: " + repr(e)
            self._self_test()
        except Exception as e:  # noqa: BLE001 - any failure -> v2 path
            self.ok0 = self.ok1 = False
            self.g0 = self.g1 = None
            self.capture_error = (self.capture_error or "") + " " + repr(e)
        finally:
            try:
                exp.reset_ttt_state()
            except Exception:  # noqa: BLE001
                pass
        # P6 (orchestrator note, KD-R review §5.2): a failure INSIDE a `torch.cuda.graph`
        # capture block can leave the process' CUDA stream/RNG in a poisoned state even
        # though the exception itself is caught above -- every LATER `ttt_step` (via ANY
        # path, W01 or the unmodified eager one) could then raise, unrecoverably. This is
        # a DIAGNOSTIC-ONLY probe, run once at construction end regardless of whether
        # capture succeeded: a single draw on its OWN throwaway `torch.Generator` (never
        # `torch.cuda.default_generators`, so it cannot advance the Philox offset any
        # captured graph or the eager path relies on -- outputs are provably unaffected)
        # that would itself raise or come back non-finite if the stream/context is
        # broken. The result is recorded in `self.diag` and never consulted by any
        # control-flow decision (it does not gate `ok0`/`ok1`/`phase`).
        self.diag = {"post_construct_probe_ok": None, "post_construct_probe_err": None}
        try:
            dev = self.model.std_tgt.device
            if dev.type == "cuda":
                pg = torch.Generator(device=dev).manual_seed(0xC0FFEE)
                probe = torch.randn(1, device=dev, generator=pg)
                torch.cuda.synchronize()
                self.diag["post_construct_probe_ok"] = bool(torch.isfinite(probe).all())
            else:
                self.diag["post_construct_probe_ok"] = True  # nothing to probe off-GPU
        except Exception as e:  # noqa: BLE001 - diagnostic only, never affects behaviour
            self.diag["post_construct_probe_ok"] = False
            self.diag["post_construct_probe_err"] = repr(e)
        self.phase = 0 if self.ok0 else -1

    # ------------------------------------------------------------ setup
    def _check_preconditions(self):
        m = self.model
        cfg = m.cfg
        if not torch.cuda.is_available() or m.std_tgt.device.type != "cuda":
            raise RuntimeError("no CUDA")
        base = m.base
        if not (type(base).__name__ == "GraphedCtxNormWrapper" and getattr(base, "ok", False)):
            raise RuntimeError("ctxnorm graph not available")
        for name in ("_maskg", "_scang", "_tail"):
            w = getattr(m, name, None)
            if w is None or not getattr(w, "ok", False):
                raise RuntimeError(f"{name} graph not available")
        if m.ke_flow is None or m.ke_flow._graph is None or not m.ke_flow._graph.ok:
            raise RuntimeError("keflow graph not available")
        if m.prior is None or not cfg["prior"] or not cfg["bounds"]:
            raise RuntimeError("scope: prior/bounds expected")
        if cfg["er_online"] or m.ke_steps or cfg["mean_mode"]:
            raise RuntimeError("scope: er_online/ke_steps/mean_mode must be off")
        self._subgraph_owners = [base, m._maskg, m._scang, m._tail, m.ke_flow._graph]
        for name in ("_pre", "_mega", "_boundsg", "_cwg"):
            w = getattr(m, name, None)
            if w is not None and getattr(w, "graph", None) is not None:
                self._subgraph_owners.append(w)
        whole = getattr(self.exp, "_whole", None)
        self._whole = whole if (whole is not None and whole.ok) else None

    def _alloc(self):
        dev, dt = self.model.std_tgt.device, self.model.std_tgt.dtype
        # G0 in: [x (N), aoa, log_re]; G1 in: [x (N), y (N)]
        self._dev_in0 = torch.zeros(N_ELEM + 2, device=dev, dtype=dt)
        self._host_in0 = torch.zeros(N_ELEM + 2, dtype=dt, pin_memory=True)
        self._dev_in1 = torch.zeros(2 * N_ELEM, device=dev, dtype=dt)
        self._host_in1 = torch.zeros(2 * N_ELEM, dtype=dt, pin_memory=True)
        # out: [pred, lower, upper, loss]
        self._dev_out = torch.zeros(3 * N_ELEM + 1, device=dev, dtype=dt)
        self._host_out = torch.zeros(3 * N_ELEM + 1, dtype=dt, pin_memory=True)
        self._x0 = self._dev_in0[:N_ELEM].view(STATIC_SHAPE)
        self._aoa = self._dev_in0[N_ELEM]
        self._logre = self._dev_in0[N_ELEM + 1]
        self._x1 = self._dev_in1[:N_ELEM].view(STATIC_SHAPE)
        self._y1 = self._dev_in1[N_ELEM:].view(STATIC_SHAPE)
        self._out_views = [self._dev_out[i * N_ELEM:(i + 1) * N_ELEM].view(STATIC_SHAPE) for i in range(3)]
        self._out_loss = self._dev_out[3 * N_ELEM]
        self._host_views = [self._host_out[i * N_ELEM:(i + 1) * N_ELEM].view(STATIC_SHAPE) for i in range(3)]

    def _objects(self):
        m = self.model
        objs = [m, m.ke_flow, m.cw, self.exp] + list(self._subgraph_owners)
        if self._whole is not None:
            objs.append(self._whole)
        out, seen = [], set()
        for o in objs:
            if o is not None and id(o) not in seen:
                seen.add(id(o))
                out.append(o)
        return out

    def _snap(self):
        return [(o, {k: _shallow(v) for k, v in vars(o).items()}) for o in self._objects()]

    @staticmethod
    def _diff(before, after):
        d = []
        for (o, b), (o2, a) in zip(before, after):
            assert o is o2
            ch = {k: v for k, v in a.items() if k not in b or not _same(b[k], v)}
            gone = [k for k in b if k not in a]
            if gone:
                raise RuntimeError(f"attributes deleted by the step: {gone}")
            if ch:
                d.append((o, ch))
        return d

    @staticmethod
    def _apply(diff):
        for o, ch in diff:
            d = o.__dict__
            for k, v in ch.items():
                d[k] = _shallow(v)

    # ------------------------------------------------------------ capture helpers
    def _swap_subgraphs(self):
        saved = [(w, w.graph) for w in self._subgraph_owners]
        for w in self._subgraph_owners:
            w.graph = _BodyAsReplay(w)
        return saved

    @staticmethod
    def _restore_subgraphs(saved):
        for w, g in saved:
            w.graph = g

    def _w0_body(self):
        """The control's window-0 ttt_step on static buffers (see module docstring)."""
        m, exp = self.model, self.exp
        aoa_t, logre_t = self._aoa, self._logre

        def prior_dev(x, _m=m):
            _m.params = (0.0, 0.0)          # placeholder; the real (Re, AoA) is set per replay
            _m._apply_prior_maps(aoa_t, logre_t)
        m._apply_prior = prior_dev
        try:
            out, info = exp.ttt_step(self._x0, None)
        finally:
            del m._apply_prior
        self._out_views[0].copy_(out)
        self._out_views[1].copy_(info["lower"])
        self._out_views[2].copy_(info["upper"])

    def _w1_body(self, seed_whole: bool):
        m, exp = self.model, self.exp
        box = {}

        def keep(t, _box=box):
            _box["loss"] = t
            return t
        m._w01_loss = keep
        try:
            out, info = exp.ttt_step(self._x1, self._y1)
        finally:
            del m._w01_loss
        self._out_views[0].copy_(out)
        self._out_views[1].copy_(info["lower"])
        self._out_views[2].copy_(info["upper"])
        self._out_loss.copy_(box["loss"])
        if seed_whole:
            self._whole._seed()

    def _set_dummy(self, g):
        self._x0.copy_(torch.randn(STATIC_SHAPE, generator=g).to(self._x0.device))
        self._aoa.fill_(7.5)
        self._logre.fill_(math.log(12000.0))
        self._x1.copy_(torch.randn(STATIC_SHAPE, generator=g).to(self._x1.device))
        self._y1.copy_(torch.randn(STATIC_SHAPE, generator=g).to(self._y1.device))

    # ------------------------------------------------------------ capture
    def _capture_all(self, warmup_iters):
        exp = self.exp
        g = torch.Generator().manual_seed(1234)
        self._set_dummy(g)
        torch.cuda.synchronize()
        whole = self._whole
        # G1 may pre-seed the whole-step graph only if its gate would open at window 2
        # from the post-window-1 state (checked on the eager warm-up below).
        self._seed_whole = False
        saved = self._swap_subgraphs()
        try:
            # warm-up: the exact capture bodies, eagerly, on a side stream
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(warmup_iters):
                    exp.reset_ttt_state()
                    self._w0_body()
                    self._w1_body(seed_whole=False)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            if whole is not None:
                live = whole.live
                whole.live = False
                self._seed_whole = bool(whole.should_use())
                whole.live = live
            # capture G0 from a fresh reset
            exp.reset_ttt_state()
            self._post_reset = self._snap()
            self.g0 = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(self.g0):
                self._w0_body()
            self._diff0 = self._diff(self._post_reset, self._snap())
        finally:
            self._restore_subgraphs(saved)

    def _capture_g1(self):
        exp = self.exp
        saved = self._swap_subgraphs()
        try:
            # state = post-G0 (python refs into G0's pool, set during G0's capture)
            self.g0.replay()
            torch.cuda.synchronize()
            before = self._snap()
            self.g1 = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(self.g1):
                self._w1_body(seed_whole=self._seed_whole)
            self._diff1 = self._diff(before, self._snap())
        finally:
            self._restore_subgraphs(saved)

    # ------------------------------------------------------------ self-test
    def _eager_ref(self, x0, x1, y1, x2, y2):
        exp = self.exp
        assert exp._warm is None
        exp.reset_ttt_state()
        outs = []
        for x, y in ((x0, None), (x1, y1), (x2, y2)):
            o, info = exp.ttt_step(x, y)
            outs.append((o.clone(), info["lower"].clone(), info["upper"].clone(), info["adapt_loss"]))
        return outs

    def _self_test(self):
        g = torch.Generator().manual_seed(4321)
        xs = [torch.randn(STATIC_SHAPE, generator=g) for _ in range(5)]
        # masked block like the real airfoil (exact zeros in physical u, v)
        mean_in = self.model.mean_in.cpu(); std_in = self.model.std_in.cpu()
        for x in xs:
            x[:, :, 10:18, 5:15, :2] = (-mean_in[:2] / std_in[:2])
        x0, x1, y1, x2, y2 = xs
        ref = self._eager_ref(x0, x1, y1, x2, y2)
        self.exp.reset_ttt_state()
        self.reset()
        got = []
        self.exp._warm = self
        try:
            for x, y in ((x0, None), (x1, y1), (x2, y2)):
                o, info = self.exp.ttt_step(x, y)
                got.append((o.clone(), info["lower"].clone(), info["upper"].clone(), info["adapt_loss"]))
        finally:
            self.exp._warm = None
        used = (self.n_g0, self.n_g1)
        ok = all(torch.equal(a[i], b[i]) for a, b in zip(ref, got) for i in range(3)) and all(
            (a[3] is None and b[3] is None) or (a[3] == b[3]) for a, b in zip(ref, got))
        self.selftest = {"bit_exact": bool(ok), "g0_used": used[0], "g1_used": used[1]}
        self.n_g0 = self.n_g1 = 0
        if not ok or used[0] != 1 or (self.ok1 and used[1] != 1):
            raise RuntimeError(f"W01 self-test failed: {self.selftest}")

    # ------------------------------------------------------------ runtime
    def reset(self) -> None:
        if self._preseed and self._whole is not None:
            self._preseed = False
        # P6 fix 1: once a runtime replay failure has disabled W01, stay off for every
        # later trajectory of the run too (a fresh reset must NOT re-arm phase 0).
        self.phase = -1 if self._disabled else (0 if self.ok0 else -1)

    def _cpu_f32(self, t):
        t = torch.as_tensor(t)
        if t.device.type != "cpu" or t.dtype != torch.float32 or tuple(t.shape) != STATIC_SHAPE:
            return None
        return t

    def _guarded(self, fn, *args):
        """P6 fix 1: run a G0/G1 replay call guarded against a runtime failure (e.g. a
        capture replay that raises after construction's self-test already passed). On
        any exception: restore every tracked object's attributes to their PRE-CALL
        values (`self._snap()`/`self._apply()` -- the same mechanism `_diff0`/`_diff1`
        already use, so `self._whole.live` etc. are covered too), i.e. exactly the state
        the eager control path expects to find; disable W01 for the rest of the run
        (`reset()` then keeps `phase == -1` forever); and return None so the caller
        (`try_step`) falls back to the unchanged eager code for THIS call, the same as
        the existing "preconditions failed" branches. `_snap()` is host-only (no device
        sync) and only ever runs here, i.e. at most twice per trajectory (windows 0/1) --
        the steady-state steps (k>=2, the whole-step graph path) never call this."""
        pre = self._snap()
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001 - runtime replay failure -> eager fallback
            self._apply(pre)
            self._preseed = False
            self._disabled = True
            self.capture_error = (self.capture_error or "") + " runtime: " + repr(e)
            self.phase = -1
            return None

    def try_step(self, input_norm, prev_target_norm):
        # IPC: decide the return layout from the caller's own input (see _ipc_one_storage / _w01_out)
        self._ipc_one = _ipc_one_storage(self.model.cfg.get("ipc_one_storage", False), input_norm)
        ph = self.phase
        if ph == 0:
            x = self._cpu_f32(input_norm) if prev_target_norm is None else None
            if x is None:
                self.phase = -1
                return None
            return self._guarded(self._run0, x)
        if ph == 1:
            x = self._cpu_f32(input_norm)
            y = self._cpu_f32(prev_target_norm) if prev_target_norm is not None else None
            if x is None or y is None or not self.ok1:
                self.phase = -1
                return None
            return self._guarded(self._run1, x, y)
        if ph == 2:
            # window 2 did not take the whole-step path: withdraw the pre-seed so the
            # control's own seeding (from the model attributes) happens when it does.
            if self._preseed and self._whole is not None:
                self._whole.live = False
            self._preseed = False
        self.phase = -1
        return None

    def _run0(self, x):
        from parameter_estimation import estimate_aoa, estimate_re
        xs = x.reshape(STATIC_SHAPE)
        state = xs[0].numpy()
        re, aoa = float(estimate_re(state)), float(estimate_aoa(state))
        hin = self._host_in0
        hin[:N_ELEM].copy_(xs.reshape(-1))
        hin[N_ELEM] = aoa
        hin[N_ELEM + 1] = math.log(max(re, 1.0))
        self._dev_in0.copy_(hin, non_blocking=True)
        self.g0.replay()
        self._host_out.copy_(self._dev_out, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        self._apply(self._diff0)
        self.model.params = (re, aoa)
        self.phase = 1
        self.n_g0 += 1
        hv = self._w01_out()
        return hv[0], {"adapt_loss": None, "lower": hv[1], "upper": hv[2]}

    def _run1(self, x, y):
        hin = self._host_in1
        hin[:N_ELEM].copy_(x.reshape(-1))
        hin[N_ELEM:].copy_(y.reshape(-1))
        self._dev_in1.copy_(hin, non_blocking=True)
        self.g1.replay()
        self._host_out.copy_(self._dev_out, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        self._apply(self._diff1)
        if self._seed_whole:
            self._whole.live = True
            self._preseed = True
        self.phase = 2
        self.n_g1 += 1
        hv = self._w01_out()
        return hv[0], {"adapt_loss": float(self._host_out[3 * N_ELEM]),
                       "lower": hv[1], "upper": hv[2]}

    def _w01_out(self):
        """IPC: views of ONE clone when try_step decided so (CONFIG "ipc_one_storage"); else three clones (parent)."""
        hv = self._host_views
        if getattr(self, "_ipc_one", False):
            c = self._host_out[:3 * N_ELEM].clone()
            return tuple(c[i * N_ELEM:(i + 1) * N_ELEM].view(STATIC_SHAPE) for i in range(3))
        return hv[0].clone(), hv[1].clone(), hv[2].clone()
