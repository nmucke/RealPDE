# Track 2 (LTTTA) — Technical Reference

*Working spec, compiled 2026-08-20 from the Codabench Track 2 pages (Overview / Data /
Evaluation / Submission / Rules / FAQ). Raw dump archived from
`https://www.codabench.org/api/competitions/17385/`. The starting kit's `docs/interface.md`
and `docs/metrics.md` refine this; the Evaluation page is authoritative for metrics.*

## Task

Long-horizon autoregressive prediction on **real PIV trajectories** (NACA4418 airfoil) with
**continuous adaptation to streaming observations**. The evaluator walks each hidden trajectory
window by window:

- Windows: **20 input frames → 20 target frames, stride 20** — window *k*'s target frames are
  window *k+1*'s input frames.
- At step *t* the model receives the current input window (ground-truth frames, normalized) and
  the ground truth of step *t−1*'s target, and must return the prediction for step *t*.
- Strict trajectory order, batch size 1. Trajectory boundaries signaled via `reset_ttt_state()`.
- All adaptation (gradient steps, calibration, memory, expert selection, LLM-guided decisions)
  happens inside `ttt_step` and **is timed**.

**Verified against starting kit v6** (`local_eval.py` mirrors the official ingestion loop):
windows start at `time_id = 0, 20, 40, …` with input = frames `[t, t+20)`, target =
`[t+20, t+40)` — so `prev_target_norm` covers the *same physical frames* as the current
`input_norm`, but normalized with the **target** stats while the input uses the **input**
stats (the `mean_std_real.pt` tuple is `(mean_in, mean_tgt, std_in, std_tgt)`, per-channel).
The intended TTT pattern (reference implementation in `submission_template.py`): adapt on
(cached previous input → revealed previous target), predict the current input, cache the
current input. The evaluator caches the `(input, target)` pair only *after* the timed
`ttt_step` returns, so the current target is never reachable. Short trajectory tails are
padded by repeating the last frame.

## Data seen at evaluation

- Shape per step: `(1, 20, 32, 64, 3)` = (batch, T, H, W, C), channels `[u, v, p]`.
- Raw fields are 64×128; evaluation is 2× downsampled to **32×64**.
- Real data: measured `[u, v]` only; `p` is zero-filled and **not scored** (returning zeros
  for `p` is explicitly fine; the scorer counts measured channels as those not all-zero).
- Everything is in normalized space (official `train_real` statistics); the evaluator handles
  (de)normalization.
- Conditions: AoA ∈ {0, 5, 10, 15, 20}°, Re 2,968–27,975. Hidden validation for the leaderboard;
  private test with **unseen parameter regimes** for the final.

## Interface contract

Submit a zip with `submission.py` at the root (+ weights, + vendored pure-Python deps).
**≤ 256 MB after extraction** (fp16 packing tool exists but is discouraged — design to fit).

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

Interval rules: `lower`/`upper` must both be present or both absent, with the same shape as
`pred_norm`, in normalized space; once a run starts returning bounds it must do so on **every**
remaining step (and must not start partway through). Violations fail the run at ingestion.

### Zero-score / failure conditions

Score 0 on **all** subscores: NaN/Inf in predictions; wrong prediction shape; non-finite or
reversed bounds. Run **fails outright** (no score): wrong `pred_norm` shape the evaluator can't
proceed with, bounds shape mismatch, a lone `lower`/`upper`, or starting/stopping bounds
mid-run.

## Evaluation environment

- Docker image `w3nhao/realpde-track2@sha256:f07d1d6...` — **PyTorch 2.2.2, CUDA 12.1,
  Python 3.10**. Nothing installed at eval time; vendor pure-Python deps in the archive.
- Available: torch/torchvision/torchaudio 2.2.2, numpy 1.26, scipy, pandas, matplotlib, h5py,
  einops, scikit-learn, tensorboard, safetensors, openai, pillow, sympy, networkx, pyyaml,
  requests, tqdm. Manifest at `/opt/image_manifest.txt` in the image.
- Offline except the organizer LLM gateway (`OPENAI_BASE_URL`/`OPENAI_API_KEY` injected; model
  `REALPDE_GATEWAY_MODEL`, currently `gemini-3.5-flash`). Chat completions only. LLM latency is
  timed.
- Evaluation hardware: 8× NVIDIA A800 (organizer side).
- **10-minute wall-clock execution limit** per submission (container execution; download
  excluded; model loading *does* count toward the wall limit though not toward the Time metric).

## Scoring

Leaderboard shows a single `final_score` (0–100); our detailed results show the five subscores.
**The combination into `final_score` is unpublished.** The starting kit's `scoring.py` computes
the five subscores exactly as the leaderboard does.

Error → score map for the three error metrics (per-window errors averaged over the stream):

```
score = 100 / (1 + 0.5 * error)        # error 0 → 100; error 2 → 50
```

1. **`rel_l2_score`** — per window, `‖pred − target‖₂ / max(‖target‖₂, 1e−8)` over flattened
   `u, v`.
2. **`tke_score`** — per spatial location `ke = 0.5·(var_t(u) + var_t(v))` over the 20 frames;
   relative L2 between `ke(pred)` and `ke(target)` per window.
3. **`mvpe_score`** — time-averaged `u, v` at a fixed probe grid in the wake (4 streamwise
   stations × up to 9 spanwise rows); relative L2 between predicted and reference profiles.
4. **`time_score`** — every `ttt_step` and `reset_ttt_state` is timed (reset charged to the next
   step). With `t_neural` = mean per-step wall time, `t_numerical = 0.72896 s`,
   `r = t_neural / t_numerical`:

   ```
   time_score = 100 / (1 + sqrt(r))
   ```

   r = 1 → 50; r = 0.1 → 76; r = 0.01 → 90.9. Model construction / checkpoint loading not timed.
5. **`sps_score`** — Safe Prediction Score (accuracy + calibrated, tight intervals):

   ```
   sigma_global = 0.0563870          # frozen; mean u,v std on official train_real
   inside = (lower <= target <= upper)          # per element
   nil    = (upper - lower) / sigma_global      # normalized interval width, per element
   pm     = e / (0.5 + e)                       # per window, e = branch error, in [0,1)
   elem   = (1 - pm) * exp(-nil)  if inside else 0
   branch = mean(elem) over scored elements     # branches: DM (Rel-L2), TKE, MVPE
   sps_score = 100 * (0.5*branch_dm + 0.3*branch_tke + 0.2*branch_mvpe)
   ```

   - Bounds are denormalized with the same transform as `pred_norm`.
   - No bounds returned ⇒ default band `pred ± 0.05·|pred|` (after denormalization) — weak:
     zero-velocity regions get zero-width bands. **Returning calibrated intervals is a major
     lever.**
   - Targets outside the PIV field of view or inside the airfoil body are not scored in SPS.
   - Uncovered or inaccurate elements contribute 0 (never negative).

   Implementation details verified in `scoring.py`:
   - All error metrics and SPS are computed in **denormalized (physical) space**; the local
     eval denormalizes predictions/bounds before scoring.
   - The SPS "scored" mask is simply `target != 0.0` (physical space) — out-of-FOV and
     inside-airfoil pixels are exactly zero in the reference. Since inputs are ground-truth
     frames, **the mask is inferable at test time from the denormalized input window**.
   - Coverage/interval terms are masked; the per-window accuracy factors `(1 − pm)` are norms
     over the whole measured field, unmasked.
   - `σ_global = 0.0563870259 = (0.09681041 + 0.01596364)/2` — i.e. train_real u-std ≈ 0.0968,
     v-std ≈ 0.0160. Note u and v scales differ by 6×; per-channel interval widths should too.
   - MVPE probe geometry (raw-grid constants d=16, center=(x 10, y 32), at 32×64 eval res):
     probe columns `x ∈ {13, 21, 29, 37}`, probe rows `y ∈ {8, 10, …, 24}` (9 rows spaced 2
     around y=16). Only `u, v` at those points, time-averaged over the window.
   - A non-finite per-window metric zeroes that window's SPS contribution but keeps it in the
     average; any exception while scoring (bad bounds etc.) zeroes **all** subscores.

## Phases & submission limits

| Phase | Dates (UTC) | Limits |
|-------|-------------|--------|
| Warm-up | Jul 5 – Jul 19 | 3/day, exec limit 600 s |
| Development (archived) | Jul 20 – Aug 4 | frozen; scoring superseded |
| **Development (current)** | **Aug 5 – Sep 27** | **1/day**, 100 total, exec limit 600 s |
| Final Decision | Sep 28 – Oct 25 | organizer-controlled; top 10 re-trained, private test decides top 3 |

Leaderboard uses **Force_Best** (best submission counts).

## Starting kit v6 (committed under `starting_kit/`; baseline checkpoints NOT included — they
come with the data release)

```
starting_kit/
├── README.md               quickstart
├── submission_template.py  copy to submission.py; correct prev-target TTT reference
├── ttt_model.py            optional TTT base class (shape/timing contract)
├── local_eval.py           streaming loop + real subscores on local data
├── scoring.py              the five leaderboard subscores, exactly as the leaderboard
├── load_baseline.py        build CNO / FNO / Transolver + load official checkpoint
├── pack_ckpt_fp16.py       last-resort fp16 packing (FNO weights are complex64 — naive .half() breaks)
├── docs/interface.md, docs/metrics.md
├── agentic_demo/           submittable bounded-controller baseline (+ optional gateway LLM)
├── rpde_baselines/         vendored CNO / FNO / Transolver code (offline imports)
└── example_data/           two tiny synthetic trajectories + official normalization stats
```

`python local_eval.py --submission .` runs the streaming loop locally; point `--data` at real
data for comparable numbers.
