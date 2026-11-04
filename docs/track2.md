# Track 2 (LTTTA): technical reference

This page summarises the Codabench Track 2 pages (Overview, Data, Evaluation, Submission, Rules, FAQ) as of
2026-08-20, read through `https://www.codabench.org/api/competitions/17385/`, and the behaviour of starting kit v6
checked against its code. The starting kit's `docs/interface.md` and `docs/metrics.md` refine it; for metrics the
Evaluation page is authoritative.

## Task

Long-horizon autoregressive prediction on real PIV trajectories (NACA4418 airfoil) with continuous adaptation to
streaming observations. The evaluator walks each hidden trajectory window by window:

- Windows have 20 input frames and 20 target frames, with stride 20: window k's target frames are window k+1's
  input frames.
- At step t the model receives the current input window (ground-truth frames, normalised) and the ground truth of
  step t−1's target, and returns the prediction for step t.
- Trajectories are processed in strict order with batch size 1. Trajectory boundaries are signalled by
  `reset_ttt_state()`.
- All adaptation (gradient steps, calibration, memory, expert selection, LLM-guided decisions) happens inside
  `ttt_step` and is timed.

Behaviour of the official loop, as reproduced by starting kit v6's `local_eval.py`: windows start at
`time_id = 0, 20, 40, …` with input = frames `[t, t+20)` and target = `[t+20, t+40)`. `prev_target_norm`
therefore covers the same physical frames as the current `input_norm`, but it is normalised with the target
statistics while the input uses the input statistics (the `mean_std_real.pt` tuple is
`(mean_in, mean_tgt, std_in, std_tgt)`, per channel). The reference pattern in `submission_template.py` adapts on
(cached previous input → revealed previous target), predicts from the current input, and caches the current input.
The evaluator caches the `(input, target)` pair only after the timed `ttt_step` returns, so the current target is
never reachable. Short trajectory tails are padded by repeating the last frame.

## Data seen at evaluation

- Shape per step: `(1, 20, 32, 64, 3)` = (batch, T, H, W, C), channels `u, v, p`.
- Raw fields are 64×128; evaluation uses a 2× subsampling to 32×64.
- Real data have measured `u, v` only; `p` is zero-filled and not scored (returning zeros for `p` is allowed; the
  scorer treats channels that are not all-zero as measured).
- Everything is in normalised space (official `train_real` statistics); the evaluator handles normalisation and
  de-normalisation.
- Conditions: AoA ∈ {0, 5, 10, 15, 20}°, Re 2,968–27,975. The leaderboard uses hidden validation data; the final
  evaluation uses a private test set with unseen parameter regimes.

## Interface contract

The submission is a zip with `submission.py` at the root, plus weights and vendored pure-Python dependencies, at most
256 MB after extraction. The kit has an fp16 packing tool, but the pages discourage relying on it.

```python
def get_ttt_model(submission_dir: str, device: str):
    """Return an object implementing the TTT interface."""

class TTTModel:  # informal
    def reset_ttt_state(self) -> None:
        # start of every trajectory: restore checkpoint weights, clear adaptation state.
        # TIMED (charged to the following step).

    def ttt_step(self, input_norm, prev_target_norm=None):
        # input_norm       : (1, 20, 32, 64, 3) current input window
        # prev_target_norm : (1, 20, 32, 64, 3) ground truth of the PREVIOUS step,
        #                    or None on the first step of a trajectory
        # returns (pred_norm, info)
        #   pred_norm : (1, 20, 32, 64, 3)
        #   info      : {"adapt_loss": float | None,
        #                # optional, both-or-neither, consistently for the whole run:
        #                "lower": ..., "upper": ...}   # same shape/space as pred_norm
```

Interval rules: `lower` and `upper` must both be present or both absent, have the same shape as `pred_norm`, and be in
normalised space. Once a run starts returning bounds it must return them on every remaining step, and it must not start
partway through. Violations fail the run at ingestion.

### Zero scores and failed runs

All subscores are 0 if the predictions contain NaN/Inf, if the prediction shape is wrong, or if bounds are non-finite
or reversed. The run fails outright (no score) on a `pred_norm` shape the evaluator cannot proceed with, a bounds shape
mismatch, a lone `lower` or `upper`, or bounds that start or stop mid-run.

## Evaluation environment

- Docker image `w3nhao/realpde-track2@sha256:f07d1d68e19cb7ed5f2f405b6775f4fafb985887ac78356a9599f35365ccf972`:
  PyTorch 2.2.2, CUDA 12.1, Python 3.10. Nothing is installed at evaluation time; pure-Python dependencies must be
  vendored in the archive.
- Available packages: torch / torchvision / torchaudio 2.2.2, numpy 1.26, scipy, pandas, matplotlib, h5py, einops,
  scikit-learn, tensorboard, safetensors, openai, pillow, sympy, networkx, pyyaml, requests, tqdm. The manifest is at
  `/opt/image_manifest.txt` in the image.
- Offline, except the organizers' LLM gateway (`OPENAI_BASE_URL` / `OPENAI_API_KEY` injected; model
  `REALPDE_GATEWAY_MODEL`, `gemini-3.5-flash` at the time). Chat completions only; LLM latency is timed.
- Evaluation hardware: 8× NVIDIA A800 (organizer side).
- Execution limit: 10 minutes of wall-clock time per submission (container execution; download excluded). Model
  loading counts toward this limit but not toward the Time metric.

## Scoring

The leaderboard shows a single `final_score` (0–100); a team's detailed results show the five subscores. The
combination into `final_score` is not published. The starting kit's `scoring.py` computes the five subscores exactly
as the leaderboard does.

Error-to-score map for the three error metrics (per-window errors averaged over the stream):

```
score = 100 / (1 + 0.5 * error)        # error 0 → 100; error 2 → 50
```

1. `rel_l2_score`: per window, `‖pred − target‖₂ / max(‖target‖₂, 1e−8)` over the flattened `u, v`.
2. `tke_score`: per spatial location `ke = 0.5·(var_t(u) + var_t(v))` over the 20 frames; relative L2 between
   `ke(pred)` and `ke(target)` per window.
3. `mvpe_score`: time-averaged `u, v` at a fixed probe grid in the wake (4 streamwise stations × up to 9 spanwise
   rows); relative L2 between predicted and reference profiles.
4. `time_score`: every `ttt_step` and `reset_ttt_state` is timed (reset is charged to the next step). With
   `t_neural` the mean per-step wall time, `t_numerical = 0.72896 s` and `r = t_neural / t_numerical`:

   ```
   time_score = 100 / (1 + sqrt(r))
   ```

   r = 1 gives 50, r = 0.1 gives 76, r = 0.01 gives 90.9. Model construction and checkpoint loading are not timed.
5. `sps_score`, the Safe Prediction Score (accuracy combined with calibrated, tight intervals):

   ```
   sigma_global = 0.0563870          # frozen; mean u,v std on official train_real
   inside = (lower <= target <= upper)          # per element
   nil    = (upper - lower) / sigma_global      # normalised interval width, per element
   pm     = e / (0.5 + e)                       # per window, e = branch error, in [0,1)
   elem   = (1 - pm) * exp(-nil)  if inside else 0
   branch = mean(elem) over scored elements     # branches: DM (Rel-L2), TKE, MVPE
   sps_score = 100 * (0.5*branch_dm + 0.3*branch_tke + 0.2*branch_mvpe)
   ```

   - Bounds are de-normalised with the same transform as `pred_norm`.
   - Without returned bounds the scorer uses a default band `pred ± 0.05·|pred|` (after de-normalisation), which
     gives zero-width bands in zero-velocity regions.
   - Targets outside the PIV field of view or inside the airfoil body are not scored in SPS.
   - Elements outside the interval or with inaccurate windows contribute 0 (never a negative amount).

   Details checked in the kit's `scoring.py`:
   - All error metrics and SPS are computed in de-normalised (physical) space; the local evaluation de-normalises
     predictions and bounds before scoring.
   - The SPS scored mask is `target != 0.0` in physical space; out-of-view and inside-airfoil pixels are exactly zero
     in the reference. Since the inputs are ground-truth frames, the mask can be inferred at test time from the
     de-normalised input window.
   - The coverage and interval terms are masked; the per-window accuracy factors `(1 − pm)` use norms over the whole
     measured field, unmasked.
   - `σ_global = 0.0563870259 = (0.09681041 + 0.01596364)/2`, i.e. the train_real u std is about 0.0968 and the v std
     about 0.0160. The u and v scales differ by a factor of 6.
   - MVPE probe geometry (raw-grid constants d = 16, centre = (x 10, y 32), at the 32×64 evaluation resolution):
     probe columns `x ∈ {13, 21, 29, 37}`, probe rows `y ∈ {8, 10, …, 24}` (9 rows spaced 2 around y = 16). Only
     `u, v` at those points, time-averaged over the window.
   - A non-finite per-window metric zeroes that window's SPS contribution but keeps it in the average; any exception
     while scoring (bad bounds, for example) zeroes all subscores.

## Phases and submission limits

| Phase | Dates (UTC) | Limits |
|-------|-------------|--------|
| Warm-up | Jul 5 – Jul 19 | 3 per day, execution limit 600 s |
| Development (archived) | Jul 20 – Aug 4 | frozen; scoring superseded |
| Development (current) | Aug 5 – Sep 27 | 1 per day, 100 in total, execution limit 600 s |
| Final Decision | Sep 28 – Oct 25 | organizer-controlled; the top 10 are retrained and the private test decides the top 3 |

The leaderboard uses Force_Best: a team's best submission counts.

## Starting kit v6

A copy is in `final_submission/realpde/starting_kit/`. The
baseline checkpoints are not included; they come with the data release.

```
starting_kit/
├── README.md               quickstart
├── submission_template.py  copy to submission.py; reference prev-target TTT pattern
├── ttt_model.py            optional TTT base class (shape/timing contract)
├── local_eval.py           streaming loop + the five subscores on local data
├── scoring.py              the five leaderboard subscores, as computed on the leaderboard
├── load_baseline.py        builds CNO / FNO / Transolver and loads the official checkpoints
├── pack_ckpt_fp16.py       fp16 packing (FNO weights are complex64; a naive .half() breaks them)
├── docs/interface.md, docs/metrics.md
├── agentic_demo/           submittable bounded-controller baseline (+ optional gateway LLM)
├── rpde_baselines/         vendored CNO / FNO / Transolver code (offline imports)
└── example_data/           two small synthetic trajectories + official normalisation statistics
```

`python local_eval.py --submission .` runs the streaming loop locally; `--data` pointed at real data gives numbers
comparable to the leaderboard subscores (except time, which depends on the machine).
