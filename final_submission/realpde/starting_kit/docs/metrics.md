# LTTTA Metrics

Leaderboard ranking uses `final_score`, a single 0-100 score combined from the
five 0-100 subscores below (each clipped to `[0, 100]`), where a higher
`final_score` places you higher. The combination is not published. This page
summarizes the Evaluation page; that page is authoritative.

- `rel_l2_score`: data fidelity from relative L2 error.
- `tke_score`: turbulent kinetic energy consistency.
- `mvpe_score`: mean velocity profile error at probe locations.
- `time_score`: per-step runtime efficiency.
- `sps_score`: safe prediction score from interval quality.

## Error Scores

Rel-L2, TKE, and MVPE are computed per evaluation window on the measured channels
`u, v` (pressure `p` is zero-filled and not scored), averaged over all windows,
and each maps to a 0-100 score via

```text
score = 100 / (1 + 0.5 * error)
```

- **Rel-L2** (data misfit): `||pred - target|| / ||target||` over the `u, v` field.
- **TKE**: relative L2 of the turbulent kinetic energy
  `0.5 * (var_t(u) + var_t(v))` (variance over the 20 time frames).
- **MVPE**: relative L2 of the time-averaged `u, v` at a fixed grid of wake probe
  points behind the airfoil.

## Time

Track 2 times **every `ttt_step` call and every `reset_ttt_state` call**:
adaptation on the previous pair, controller decisions, optional LLM round-trips,
the forward prediction, and whatever you do at a trajectory boundary all count.
The reset is charged to the step that follows it, so the mean stays a per-step
time and moving work between the two changes nothing. Model construction,
checkpoint loading, and the evaluator's own pre/post-processing are not timed. With `t_neural` the mean per-step wall time over the
whole stream and `t_numerical = 0.72896` s:

```text
r = t_neural / t_numerical
time_score = 100 * 1 / (1 + sqrt(r))
```

A missing, zero, negative, or non-finite per-step time scores 0.

## SPS

The Safe Prediction Score rewards predictions that are accurate *and* paired with
tight, well-calibrated intervals. It combines three physical error branches under
a coverage gate; the Evaluation page gives the exact formula. Summary:

- Each branch error `e` (DM = Rel-L2, TKE, MVPE) is squashed to `[0, 1)` by
  `pm = e / (0.5 + e)`.
- Per element, the reward is `(1 - pm) * exp(-(upper - lower) / sigma_global)`,
  counted only where the target lies inside `[lower, upper]`, then averaged.
- Targets outside the PIV field of view or inside the airfoil body are not
  scored: they leave that average, numerator and denominator both.
- Branches combine with weights `DM 0.5 / TKE 0.3 / MVPE 0.2`, then map to
  `sps_score = 100 * weighted`.

Interval width is normalized by the frozen constant `sigma_global = 0.0563870`
(mean of the `u, v` channel standard deviations on the official `train_real`
split, constant for the whole season). Bounds are optional: set both
`info["lower"]` and `info["upper"]` from `ttt_step` (same normalized space and
shape as `pred_norm`; see the Submission page) to supply them, or omit both to
let the scorer use a default band of `±5%` of `|prediction|`. Every element
contributes between 0 and 1 and the branch weights sum to 1, so `weighted` lands
in `[0, 1]` and `sps_score` in `[0, 100]`.

## Composite

The five subscores above are combined into a single 0-100 `final_score`, the only
column on the public leaderboard. The combination is not published.

## Execution Time Limit

Separate from the per-step Time metric above, each submission must finish within
a wall-clock **10-minute** execution limit (Warm-up and Development phases,
container execution only; data download excluded). Submissions exceeding it are
marked Failed with no score.
