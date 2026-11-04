"""Occlusion masks for the real PIV field, derived from an evaluation window.

`train_real` carries an exact-zero region -- airfoil body, laser shadow, and the
edges of the PIV field of view -- covering 5-13 % of the grid. It is *not* fixed:
it pivots and shrinks with AoA (11.6 % of the grid at 0 deg down to 6.1 % at
20 deg; IoU between the 0 deg and 20 deg consensus masks is only 0.36), and even
at one AoA it varies with Re and run (IoU 0.84-0.89). A single hard-coded mask is
not usable: over all 81 usable `train_real` files, the pixels masked in *every*
run cover 3.1 % of the grid and the pixels masked in *any* run cover 16.2 %,
an IoU of 0.19.

So the mask has to be read off the window at hand, which `docs/track2.md` notes is
possible because the evaluator's inputs are ground-truth frames.

Two different masks live here, because two callers want different things:

* `scored_elements` mirrors `scoring.py`'s `scored = t != 0.0` -- a per-element,
  per-channel, per-frame test. Use it to weight a training loss so it optimizes what
  SPS actually measures. Note it is not a geometry mask: it also drops the ~1.5 % of
  elements where `v` alone happens to land on exact zero.
* `occlusion_mask` returns the *persistent geometry* -- one (32, 64) footprint for
  the window. Use it to mask model internals (e.g. Transolver's slice attention),
  where a stable region matters more than per-frame exactness.

Why `occlusion_mask` votes rather than intersecting: PIV dropout flickers, so only
1-6 % of pixels are zero in *every* frame while 5-13 % are zero in a typical one.
Measured on 1620 random 20-frame windows, recovering each run's own footprint:

    rule                       IoU mean   p05     worst
    zero in all 20 frames        0.799   0.646    0.480   (recall 0.80, prec 1.00)
    zero in >= 50 % of window    0.955   0.914    0.838   (recall 0.98, prec 0.98)

and at the 32x64 eval resolution, through the float32 normalize/denormalize
roundtrip with the official stats, the majority rule holds up: IoU 0.957 mean,
0.919 p05, 0.812 worst.

`occlusion_mask_robust` adds a spatial prior: within one window the geometry is
fixed and the occluded region is a solid blob, so free-stream specks and pinholes
inside the body are artefacts by construction, and dropping them is right on physical
grounds. Be careful how you judge it -- see its docstring. It is *not* flatter in the
threshold, and against a raw full-run footprint it scores slightly worse, because that
reference carries the very specks and pinholes it removes.

`parameter_estimation._mask` keeps using the all-frames rule on purpose -- reading
AoA off the footprint's shape wants precision, not recall.
"""

from __future__ import annotations

import numpy as np

# official train_real per-channel stats (mean_std_real.pt), for u and v.
# Kept local so this module is self-contained in a flat submission archive.
MEAN = np.array([0.15496086, -0.00051399])
STD = np.array([0.09680565, 0.01596068])

HEIGHT, WIDTH = 32, 64
# A physical exact zero does not come back bitwise zero through the evaluator's
# float32 normalize and our denormalize, so the zero test needs a tolerance: with
# `!= 0.0` on the denormalized value, 10 % of elements disagree with the scorer.
# 1e-6 sits ~4 orders below the PIV signal and leaves 99.975 % agreement, the
# residue being elements whose true magnitude is itself <= 1e-6.
ZERO_TOL = 1e-6


def _as_numpy(state):
    """Accept a torch tensor or an array; return float64 numpy (T, 32, 64, C)."""
    if hasattr(state, "detach"):  # torch.Tensor, without importing torch
        state = state.detach().cpu().numpy()
    arr = np.asarray(state, dtype=np.float64)
    return arr.reshape(-1, HEIGHT, WIDTH, arr.shape[-1])


def denorm_uv(state):
    """(T, 32, 64) physical u and v from a normalized window of any leading shape."""
    s = _as_numpy(state)
    return s[..., 0] * STD[0] + MEAN[0], s[..., 1] * STD[1] + MEAN[1]


def scored_elements(state, tol=ZERO_TOL):
    """Per-element mask mirroring `scoring.py`'s `scored = t != 0.0`, shape (T, 32, 64, 2).

    True where the element counts toward SPS -- per-channel and per-frame, as the
    scorer computes it (see `docs/track2.md`). Feed it a *target* window to weight a
    loss the way SPS weights its element sum.

    Agreement with the scorer, measured on 75 real windows (6.1 M elements): 99.975 %.
    Every disagreement is an element whose true magnitude is <= `tol`, which the
    scorer counts and we drop; see `ZERO_TOL` for why the tolerance is unavoidable.
    """
    u, v = denorm_uv(state)
    return np.stack([np.abs(u) > tol, np.abs(v) > tol], axis=-1)


def zero_frequency(state):
    """Fraction of the window's frames in which each pixel reads exact zero, (32, 64).

    The raw evidence both mask rules are built on. Strongly bimodal: ~88 % of pixels
    are never zero and ~8 % are zero almost always, with only 2-3 % genuinely
    undecided -- which is why a threshold anywhere in the middle does roughly the
    right thing, and why the interesting failures are all on the boundary.
    """
    u, v = denorm_uv(state)
    return ((np.abs(u) < ZERO_TOL) & (np.abs(v) < ZERO_TOL)).mean(axis=0)


def occlusion_mask(state, threshold=0.5):
    """Persistent occluded footprint of a window: (32, 64) bool, True = occluded.

    A pixel is occluded when both velocity components read exact zero in at least
    `threshold` of the window's frames. The default majority rule recovers the run's
    own footprint at IoU 0.96 (see the module docstring); `threshold=1.0` gives the
    intersection rule, which is near-perfectly precise but only ~80 % complete.

    Purely pointwise -- no pixel knows what its neighbours decided. `occlusion_mask_
    robust` adds the spatial prior and is much less sensitive to `threshold`.
    """
    return zero_frequency(state) >= threshold


# --- spatial cleanup ---------------------------------------------------------
# The occluded region is one or two solid blobs with a smooth boundary, so the
# failures of the pointwise rule have a shape: isolated specks in the free stream
# (a `v` quantization zero that happens to land twice in 20 frames), pinholes inside
# the body (a spurious valid vector), and a ragged edge. Each is cheap to undo with
# a classical operator, and none needs a trained anything.

_SMOOTH = np.array([[0.0, 1.0, 0.0],
                    [1.0, 4.0, 1.0],
                    [0.0, 1.0, 0.0]])  # center-weighted: keeps 1-px-wide shadow streaks


def _convolve3(a, weights=_SMOOTH):
    """Normalized 3x3 convolution with edge padding. Pure numpy, 9 adds."""
    pad = np.pad(np.asarray(a, dtype=np.float64), 1, mode="edge")
    out = np.zeros(np.shape(a), dtype=np.float64)
    h, w = out.shape
    for dy in range(3):
        for dx in range(3):
            if weights[dy, dx]:
                out += weights[dy, dx] * pad[dy:dy + h, dx:dx + w]
    return out / weights.sum()


def _dilate(m):
    """One step of 4-connected binary dilation."""
    out = m.copy()
    out[1:, :] |= m[:-1, :]
    out[:-1, :] |= m[1:, :]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    return out


def _propagate(seed, allowed):
    """Flood `seed` through `allowed` to a fixed point (vectorized BFS)."""
    cur = seed & allowed
    while True:
        nxt = _dilate(cur) & allowed
        if np.array_equal(nxt, cur):
            return cur
        cur = nxt


def _components(m):
    """Yield each 4-connected component of `m` as its own boolean array."""
    todo = m.copy()
    while todo.any():
        seed = np.zeros_like(m)
        seed.flat[int(np.argmax(todo))] = True
        comp = _propagate(seed, m)
        yield comp
        todo &= ~comp


def drop_small(m, min_area=6):
    """Delete components smaller than `min_area` pixels -- the free-stream specks.

    Area, not smoothing, is what removes specks here: a blur wide enough to kill a
    2-px speck also erases the 1-px-wide shadow streak at AoA 20. On its own this
    barely moves the score (specks are few); `fill_holes` is the stage that does work.
    """
    out = np.zeros_like(m)
    for comp in _components(m):
        if comp.sum() >= min_area:
            out |= comp
    return out


def fill_holes(m):
    """Close pinholes -- any non-occluded region that cannot reach the border."""
    free = ~m
    seed = np.zeros_like(m)
    seed[0, :] = seed[-1, :] = True
    seed[:, 0] = seed[:, -1] = True
    return m | (free & ~_propagate(seed & free, free))


def occlusion_mask_robust(state, threshold=0.4, min_area=6, smooth=False):
    """`occlusion_mask` plus the spatial prior: (32, 64) bool, True = occluded.

    Threshold the zero-frequency map, drop undersized components, fill pinholes.

    **How to judge this.** Scored against a run's raw full-run footprint it comes out
    slightly *worse* than the pointwise rule (0.933 vs 0.957 at each one's best
    threshold, 42 windows), and it is no flatter in `threshold`. That comparison is
    close to meaningless, though: the reference is itself a majority vote and carries
    the same specks and pinholes, so cleaning is scored as error. Cleaning the
    reference the same way flips the verdict (0.971 vs 0.935). The reference moves by
    IoU 0.95 depending on which version you pick -- as large as the effect being
    measured -- so this benchmark cannot settle it either way.

    Use it because the physical prior says specks and pinholes are artefacts, not
    because a number says so. For masking model internals that matters: a pinhole is a
    lone "valid" pixel in the middle of the body, which is exactly the kind of outlier
    an attention slice will latch onto.

    `smooth=True` runs a 3x3 center-weighted convolution over the frequency map first.
    Off by default: it neither helps at the optimum nor widens the usable threshold
    range -- it narrows it (IoU 0.868 vs 0.935 at threshold 0.2).
    """
    score = zero_frequency(state)
    if smooth:
        score = _convolve3(score)
    return fill_holes(drop_small(score >= threshold, min_area=min_area))


def keep_mask_from_window(state, threshold=0.5, n_steps=None, flatten=True):
    """Valid-point mask for a model's spatial axis: 1.0 = keep, 0.0 = ignore.

    The complement of `occlusion_mask`, broadcast over time and (by default)
    flattened the way `Transolver_Structured_Mesh_3D.Model.forward` flattens its
    input -- `(T, 32, 64) -> (N,)`, t-major -- so it lines up element for element
    with the `N` axis of `Physics_Attention_Structured_Mesh_3D`.

    `n_steps` defaults to the window's own frame count; pass it explicitly when the
    mask has to match a different-length axis (e.g. masking a 20-frame prediction
    with a footprint read off a 20-frame input, where the counts happen to agree).
    Returns float64 (N,) if `flatten`, else (T, 32, 64).
    """
    occluded = occlusion_mask(state, threshold=threshold)
    t = _as_numpy(state).shape[0] if n_steps is None else int(n_steps)
    keep = np.broadcast_to(~occluded, (t, HEIGHT, WIDTH)).astype(np.float64)
    return keep.reshape(-1) if flatten else keep


def torch_keep_mask(state, threshold=0.5, n_steps=None, device=None, dtype=None):
    """`keep_mask_from_window` as a torch tensor of shape (1, N), ready for the
    `mask` argument of the masked Physics-Attention.

    Batch is always 1 at evaluation time; expand if you batch during training.
    """
    import torch

    mask = keep_mask_from_window(state, threshold=threshold, n_steps=n_steps)
    if dtype is None:
        dtype = getattr(state, "dtype", torch.float32)
        if not isinstance(dtype, torch.dtype):
            dtype = torch.float32
    if device is None:
        device = getattr(state, "device", None)
    return torch.as_tensor(mask, dtype=dtype, device=device).unsqueeze(0)
