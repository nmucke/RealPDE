"""Model-free Re / AoA estimation from a single evaluation window.

Every function takes the window handed to `ttt_step` -- shape (..., T, 32, 64, 3),
channels [u, v, p], in the official normalized space -- and returns a scalar.

How it works:

* the PIV is in m/s and the tunnel viscosity is fixed, so Re is linear in the
  freestream speed, read off the top/bottom margins of the downstream quarter;
* the exact-zero pixels (airfoil body + laser shadow) pivot with the airfoil, so
  the zero mask alone pins down AoA. `estimate_aoa` reads the rotation off three
  continuous shape descriptors, which interpolates to angles outside the five in
  `train_real`; `estimate_aoa_discrete` matches per-angle occupancy templates and
  is sharper, but can only ever answer with a trained angle.

Every constant below is fitted on all 81 usable `train_real` files at the stride the
evaluator streams -- 3422 windows, `7575_0.h5` excluded as corrupt -- with nothing held
out. The leave-one-out figures in the docstrings come from separate runs and say what
accuracy to expect; they are not the fit that ships. `train_sim` is unusable here: it
has no occlusion mask to read an angle off (0.2 % zeros against 11 %) and, being
non-dimensional, no Re signal in its velocity scale at all.
"""

import base64
import zlib

import numpy as np

# official train_real per-channel stats (mean_std_real.pt), for u and v
MEAN = np.array([0.15496086, -0.00051399])
STD = np.array([0.09680565, 0.01596068])

# Re = RE_FIT[0] * U + RE_FIT[1]; leave-one-file-out error: median 0.86 %, p90 2.3 %, max 7.0 %
RE_FIT = (71272.0, -287.83)
RE_MEDIAN = 13977.0  # train_real median, the fallback when a window measures nothing

AOAS = np.array([0.0, 5.0, 10.0, 15.0, 20.0])

# AoA ~= W . (elongation, upper-surface slope, leading-edge tip row, 1), then the piecewise
# linear recalibration ANCHORS -> AOAS. Leave-one-file-out MAE 1.14 deg (p90 2.4);
# leave-one-*angle*-out, i.e. an angle absent from training: 2.2 / 0.8 / 1.9 deg for 5 / 10 / 15.
AOA_W = np.array([-0.25149, 35.57629, -1.10605, 22.46608])
AOA_ANCHORS = np.array([1.0979, 5.5649, 9.658, 13.6006, 20.636])

# P(pixel is masked | AoA), uint8, shape (5, 32, 64), fitted on all 3422 train_real windows
_TEMPLATES = (
    "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"
)
# clamped off 0 and 1 so both logs below stay finite whatever the fit produced
_P = np.frombuffer(zlib.decompress(base64.b64decode(_TEMPLATES)), np.uint8).reshape(5, -1)
_P = np.clip(_P, 1, 254) / 255.0
_LOG_P, _LOG_Q = np.log(_P), np.log(1.0 - _P)


def _uv(state):
    """(T, 32, 64) physical u and v from a normalized window of any leading shape."""
    s = np.asarray(state, dtype=np.float64).reshape(-1, 32, 64, 3)
    return s[..., 0] * STD[0] + MEAN[0], s[..., 1] * STD[1] + MEAN[1]


def _mask(u, v):
    """Pixels that are zero in every frame: airfoil body, laser shadow, outside the FOV."""
    return ((np.abs(u) < 1e-6) & (np.abs(v) < 1e-6)).all(0)


def estimate_re(state):
    """Reynolds number, from the freestream speed in the undisturbed downstream margins.

    Leave-one-file-out on train_real (3422 windows): median 0.86 %, p90 2.3 %, max 7.0 %.
    """
    u, v = _uv(state)
    margin = np.ix_(np.r_[0:4, 28:32], np.arange(48, 64))
    mask = _mask(u, v)
    free = u.mean(0)[margin][~mask[margin]]
    if not free.size:  # margins fully masked: fall back to whatever was measured
        free = u.mean(0)[~mask]
    if not free.size:  # nothing measured at all
        return RE_MEDIAN
    return RE_FIT[0] * np.median(free) + RE_FIT[1]


def estimate_aoa_discrete(state):
    """Angle of attack in degrees, by matching the zero mask to a per-angle template.

    Leave-one-file-out on train_real (3422 windows): 99.7 % of single windows exactly
    right (the misses are 5 deg read as 0 deg), 100 % of trajectories when the caller
    votes over windows. Only ever returns a trained angle {0, 5, 10, 15, 20}, so prefer
    `estimate_aoa` unless the angle is known to be one of those five.
    """
    m = _mask(*_uv(state)).reshape(1, -1)
    return AOAS[(m * _LOG_P + (1 - m) * _LOG_Q).sum(1).argmax()]


def estimate_aoa(state):
    """Angle of attack in degrees, continuous -- not snapped to the five trained angles.

    Three descriptors of the zero mask, all monotone in AoA: how elongated it is, the
    slope of the airfoil's upper surface, and how high the leading-edge tip sits.

    Leave-one-file-out on train_real: MAE 1.14 deg, p90 2.4 deg. Holding out a whole
    angle -- the honest test for an AoA the training set never saw -- gives MAE 2.2 /
    0.8 / 1.9 deg for 5 / 10 / 15 deg. Extrapolating past the ends is weaker (~7 deg
    beyond 20), so trust it inside roughly 0-20 deg.
    """
    mask = _mask(*_uv(state))
    rows, cols = np.nonzero(mask)
    if cols.size < 8:  # nothing occluded, so nothing to read the angle off
        return 0.0
    elongation = np.sqrt(np.divide(*np.linalg.eigvalsh(np.cov(cols, rows))[::-1]))

    # upper surface: topmost masked pixel of each column across the airfoil
    tops = [(c, np.nonzero(mask[:24, c])[0]) for c in range(1, 21)]
    tops = [(c, k[0]) for c, k in tops if len(k)]
    slope = -np.polyfit(*np.transpose(tops).astype(float), 1)[0] if len(tops) > 3 else 0.0
    tip = float(min(r for _, r in tops)) if tops else 24.0

    s = AOA_W @ [elongation, slope, tip, 1.0]
    a, lo, hi = AOA_ANCHORS, AOA_ANCHORS[:2], AOA_ANCHORS[-2:]
    if s < a[0]:
        return 5.0 * (s - a[0]) / (lo[1] - lo[0])
    if s > a[-1]:
        return 20.0 + 5.0 * (s - a[-1]) / (hi[1] - hi[0])
    return float(np.interp(s, a, AOAS))
