

# PAGE: Overview

# NeurIPS 2026 RealPDE Competition - Track 2: LTTTA

Part of the [NeurIPS 2026 RealPDE Competition](https://realpdecompetition.github.io/). See the competition website for both tracks, timeline, and announcements. Track 1 (Sim2Real) is already live on Codabench: [codabench.org/competitions/17363](https://www.codabench.org/competitions/17363/).

**Registration required.** Registration via the team form is required for both tracks (deadline: **August 20, 2026**):

- [Track 1 registration form](https://forms.gle/LYeTgUTfr4ygntNu6)
- [Track 2 registration form](https://forms.gle/qmMYaK5u9r86rPGKA)

Joining on Codabench alone is not registration: after the registration deadline, Codabench join requests without a submitted form will be denied.

Track 2 - Long-Term Test-Time Adaptation (LTTTA) studies long-horizon autoregressive prediction with **continuous adaptation to streaming real-world observations**. Your model walks through real PIV trajectories window by window: at each step it predicts the next 20 frames, and after each prediction the ground truth of the *previous* window is revealed for adaptation. Agent-augmented methods (bounded controllers deciding when and how to adapt, optionally through an organizer-provided LLM gateway) are explicitly welcome.

## Announcements

**This section is updated throughout the competition.** Check back regularly; substantive changes are also announced on the competition forum.

**Development Phase restarted (5 August).** 

---

A new Track 2 Development Phase opens on **5th August 2026, 00:00 UTC** with an empty leaderboard. Scoring has been updated, so please submit again. The earlier phase and its leaderboard stay visible and frozen. The restart moves no other date: registration still closes on **20th August 2026**, and the Development Phase still closes at **24:00 UTC on 27th September 2026**.

Multiple teams flagged problems on Track 1 during the phase and shared what they found. We are grateful to them for debugging the RealPDE Competition for us. A scientific benchmark stays honest because its community keeps checking it. Please keep telling us when something looks wrong, and we will keep fixing it.

Your submission still runs unmodified. The `ttt_step` and `reset_ttt_state` signatures, the returned dictionary, the evaluation data and the 10-minute execution limit are all as they were. What changed is how the run is scored:

- **[Evaluation]** `sps_score` now maps its aggregate to 0-100 linearly, replacing the logistic it used before, so every SPS score on the old board moves. Targets outside the PIV field of view or inside the airfoil body are not scored in SPS. `reset_ttt_state` is now timed, charged to the step that follows the trajectory boundary.
- **How the total is formed.** The leaderboard shows a single `final_score`, and your own submission's detailed results show the five subscores it is combined from. **The combination is not published.** The starting kit computes the five subscores exactly as the leaderboard does, so you can still check those on your own data.
- **[Rules]** Training data must come from the competition's own release: no self-generated data, and no pretrained models trained on anything else. Checkpoints trained from the release are permitted, including the official baselines. Standard augmentation of the release is allowed; see the FAQ.
- **Starting kit v6** replaces the copy you have. Download it from this phase; older copies of `scoring.py` and `local_eval.py` are out of date.

---

- 2026-08-12: **Data issue in `train_real/7575_0.h5`.** That file reuses the data from `6300_0`, so it does not hold measurements of case 7575. Please exclude it from training. Only this file is affected: the other angles of attack at Re 7575 are fine, and so is `6300_0` itself. Case 7575 at 0 degrees cannot be measured again at present, so the file stays in the release as published and is simply not to be used. Thanks to the team who inspected the data and told us.

- 2026-08-09: **Data augmentation clarified.** Standard augmentation of the released data is allowed in both tracks. Every training sample must trace back to the released training data. The full boundary is in a new FAQ entry, linked from the Rules page.

- 2026-08-05: **Scoring updated, Development Phase restarted.** Starting kit v6. See the notice above.

- 2026-07-22: Starting kit v4 ships a worked example, updated interface docs, and local-eval support for bounds.

- 2026-07-22: **SPS interval bounds now supported in Track 2.** `ttt_step` can now return optional per-element `lower` / `upper` bounds via the `info` dict, in the same normalized space and shape as `pred_norm` (both keys together, on every step). The SPS formula, its weights, and the default ±5%·|pred| band for point-only submissions are unchanged, and existing submissions are unaffected.

- 2026-07-05: **LLM gateway online.** Agents may call an organizer-provided OpenAI-compatible LLM gateway. Usage and a code example are on the Submission page; restrictions are in the Rules.

- 2026-07-05: Competition launched. Warm-up Phase is open.

## Task

- Data: real-world PIV flow around a NACA4418 airfoil.
- Streaming protocol: strict trajectory order, batch size 1, windows of 20 input frames to 20 output frames.
- At step t your model receives the current input window (normalized) plus the ground truth of step t-1, and must return the prediction for step t.
- Adaptation (gradient steps, calibration, memory, expert selection, LLM-guided decisions) happens inside your `ttt_step`; all of it counts toward the Time metric.
- Ranking: `final_score` is a single 0-100 score combined from five 0-100 subscores (Rel-L2, TKE, MVPE, Time, SPS). The combination is not published.

## Phases

- Warm-up Phase: July 5 - July 19, 2026 (UTC). Format validation and smoke test.
- Main Development Phase: July 20 - September 27, 2026 (UTC). Hidden validation leaderboard.
- Final Decision Phase: September 28 - October 25, 2026 (UTC). The top 10 teams from the development leaderboard are shortlisted; organizers re-train their methods and evaluate on a private test set to decide the top 3.
- Final results announced: November 10, 2026.
- Code submission deadline: November 25, 2026.
- NeurIPS presentation: December 6, 2026.

All phase boundaries are in UTC (see FAQ).

## Prizes

Track 2 shares the RealPDE prize pool: 6,000 / 3,000 / 1,500 USD for the top three teams of each track (funded by Uniforce AI Ltd.). The top 3 teams in each track are invited to deliver oral presentations at the competition workshop. The top 5 teams in each track are invited to co-author a joint results paper submitted to a relevant academic journal or the NeurIPS Datasets and Benchmarks Track, and each of these teams may nominate one advisor of their own (for example, a faculty supervisor) as a co-author. Certificates are awarded to the top 5 teams in each track and may optionally name one advisor.

## Resources

Start from the downloadable **starting kit** available on the **Files tab** of this competition. *The Files tab is only visible after you sign in and your participation request is approved.*

It contains everything needed to build, locally score, and submit a Track 2 entry:

```text
starting_kit/
├── README.md               quickstart and file guide
├── submission_template.py  copy to submission.py; correct prev-target TTT reference
├── ttt_model.py            optional TTT base class (interface + shape/timing contract)
├── local_eval.py           CPU smoke test on example_data; calls scoring.py for real subscores
├── scoring.py              computes the five leaderboard subscores locally
├── load_baseline.py        build CNO / FNO / Transolver and load an official checkpoint
├── pack_ckpt_fp16.py       complex-safe fp16 packing to fit the 256 MB size cap
├── docs/
│   ├── interface.md        full submission interface contract
│   └── metrics.md          metric summary (the Evaluation page is authoritative)
├── agentic_demo/           submittable agentic baseline: bounded controller + optional gateway LLM
│   ├── submission.py
│   ├── policy.yaml
│   └── README.md
├── rpde_baselines/         vendored CNO / FNO / Transolver model code (imports offline)
└── example_data/           two tiny synthetic trajectories + official normalization stats
    ├── mean_std_real.pt
    ├── make_example.py
    └── test_real/*.h5
```

Key entry points:

- **Build**: copy `submission_template.py` to `submission.py` and wrap your model, or start from `agentic_demo/` for a controller-based method. Use `load_baseline.py` to adapt a pretrained CNO / FNO / Transolver.
- **Score locally**: `python local_eval.py --submission .` runs your model through the streaming loop and reports the five real subscores via the bundled `scoring.py` (illustrative on the tiny example data; point `--data` at real data for comparable numbers).
- **Fit the size cap**: `pack_ckpt_fp16.py` packs an fp32 checkpoint to fp16 under the 256 MB limit.

**Pretrained baseline checkpoints** (CNO / FNO / Transolver, sim-pretrained and real-finetuned) ship with the training release on [Hugging Face](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data), mirrored on [Google Drive](https://drive.google.com/drive/folders/1Cg23DoTuSvWXR3Mm1uRfmMNAbkyaIhrQ). All were trained with the [RealPDEBench](https://github.com/AI4Science-WestlakeU/RealPDEBench) code. Track 1 and Track 2 share the same data release.

## Contact

realpde-competition@googlegroups.com, or the competition forum.

# PAGE: Data

# Data

Track 2 uses paired simulated and real-world flow trajectories around a NACA4418 airfoil, consumed as a **stream** for long-term test-time adaptation.

## Physical Setting

- Geometry: 3D NACA4418 airfoil cross-sectional flow.
- Real-world measurement: time-resolved Particle Image Velocimetry (PIV) in a circulating tunnel.
- Simulation: matched 3D CFD under the same geometry and operating conditions.
- Angles of attack: 0, 5, 10, 15, and 20 degrees.
- Reynolds numbers: 2,968 to 27,975.
- Channels: simulation has `[u, v, p]`; real data is scored on measured `[u, v]`, with `p` set to zero.

## Resolution

The raw flow fields are recorded at `64 x 128`. Evaluation is performed at a `2x` spatially down-sampled resolution of `32 x 64`. Your model sees one normalized window of shape `(1, 20, 32, 64, 3)` per step (`T = 20` frames, `H = 32`, `W = 64`, `C = 3`).

## Streaming Layout

Evaluation trajectories are split into consecutive windows: 20 input frames to 20 target frames, stride 20 (window k's target frames are window k+1's input frames). Windows are revealed strictly in trajectory order with batch size 1. Trajectory boundaries are announced to your model via `reset_ttt_state()`.

Evaluation runs in the normalized space of the official train_real statistics; normalization and denormalization are handled by the evaluator, and your model only sees normalized tensors.

## Training Data

Training data and examples are distributed outside the bundle because the full dataset and baseline checkpoints are large. The public training release is hosted on Hugging Face at [AI4Science-WestlakeU/RealPDE-Competition-Data](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data), and mirrored on [Google Drive](https://drive.google.com/drive/folders/1Cg23DoTuSvWXR3Mm1uRfmMNAbkyaIhrQ). 

Prefer Hugging Face: the Drive copy has a per-file download quota that a shared link runs into when many people fetch it at once. Both hold the same files:

- `train_sim`: simulated training trajectories for pretraining.
- `train_real`: real-world training trajectories for finetuning.
- `example_data`: small samples for checking tensor shapes and local code (also included in the starting kit).

**Do not use `train_real/7575_0.h5`.** It duplicates the contents of `6300_0` instead of holding measurements of case 7575. The file is still present in the release, so please exclude it on your side. The other angles of attack at Re 7575 are unaffected. Announced on 12 August 2026.

## Important

The evaluation stream (validation and final test trajectories) is hidden. Attempting to read evaluation data files from inside the container beyond what the interface provides is a rule violation and leads to disqualification (see Rules).

# PAGE: Evaluation

# Evaluation

Submissions are ranked by `final_score`, a 0-100 score combined from the five subscores below (each 0-100, clipped). The combination is not published.

## Scores You See

The public leaderboard shows a single column: `final_score`. Your own submission's detailed results additionally show the five subscores:

- `rel_l2_score`: data fidelity from relative L2 error.
- `tke_score`: turbulent kinetic energy score.
- `mvpe_score`: mean velocity profile error score.
- `time_score`: per-step runtime efficiency score.
- `sps_score`: safe prediction score.

## Error Metrics

Rel-L2, TKE, and MVPE are computed per evaluation window on the measured channels `u, v` only (the pressure channel `p` is zero-filled and not scored), averaged over all windows, and each maps to a 0-100 score via `score = 100 / (1 + 0.5 * error)` (an error of 0 gives 100).

- **Relative L2 (data misfit, DM)**: per window, `||pred - target||_2 / max(||target||_2, 1e-8)` over the flattened `u, v` field.
- **TKE**: turbulent kinetic energy error. Per spatial location, `ke = 0.5 * (var_t(u) + var_t(v))`, where `var_t` is the variance over the 20 time frames; the relative L2 between `ke(pred)` and `ke(target)` is taken per window.
- **MVPE**: mean velocity profile error in the wake. `u, v` are time-averaged at a fixed grid of probe points behind the airfoil (4 streamwise stations x up to 9 spanwise rows), and the relative L2 between the predicted and reference probe profiles is taken per window. Lower MVPE means better long-horizon physical consistency.

## Time

Track 2 times **every `ttt_step` call and every `reset_ttt_state` call**. The reset is charged to the step that follows it. The evaluator's own pre/post-processing, model construction and checkpoint loading are not timed. With `t_neural` the mean per-step wall time over the whole stream, `r = t_neural / t_numerical`, `t_numerical = 0.72896` s:

```text
time_score = 100 * 1 / (1 + sqrt(r))
```

## SPS

The Safe Prediction Score (`sps_score`) rewards predictions that are accurate *and* paired with tight, well-calibrated uncertainty intervals.

**Intervals.** You may return per-element bounds by adding `lower` / `upper` keys to the `info` dict from `ttt_step`, in the same normalized space as `pred_norm`:

```python
return pred_norm, {"adapt_loss": ..., "lower": lower_norm, "upper": upper_norm}
```

`lower_norm` / `upper_norm` must have the same shape as `pred_norm`. The evaluator denormalizes them with the same transform it applies to `pred_norm`, so bounds live in the prediction's normalized space, not physical units. Bounds are optional, but if you return one you must return both, since a lone `lower` or `upper` key is invalid. Once a run starts returning bounds it must keep returning them on every remaining step, and a run that never returns them must not start partway through either; the ingestion program enforces this for the whole run, not just within a single call. If you omit both, the scorer uses a default band `lower = pred - 0.05*|pred|`, `upper = pred + 0.05*|pred|` (width `0.1*|pred|`), computed after denormalization.

**Ingredients.** Let `sigma_global = 0.0563870` (frozen for the season; the mean of the `u, v` channel standard deviations on the official `train_real` split). Per element define the coverage gate and the normalized interval width:

```text
inside = (lower <= target <= upper)      # per element, boolean
nil    = (upper - lower) / sigma_global  # per element
```

**Branches.** SPS sums three physical error branches: DM (Rel-L2), TKE, and MVPE. For a branch with per-window error `e`, squash it to `[0, 1)` and reward each element of that window:

```text
pm     = e / (0.5 + e)                    # per window, in [0, 1)
elem   = (1 - pm) * exp(-nil)  if inside  # else 0
branch = mean(elem)  over scored elements
```

`(1 - pm)` is the accuracy term (near 1 when the window error is small) and `exp(-nil)` is the sharpness term (near 1 for tight intervals); elements whose target falls outside `[lower, upper]` contribute 0.

Targets outside the PIV field of view or inside the airfoil body are not scored in SPS.

**Aggregate.** Combine the branches and read the result as a 0-100 score:

```text
weighted  = 0.5*branch_dm + 0.3*branch_tke + 0.2*branch_mvpe
sps_score = 100 * weighted
```

Because `pm` is bounded in `[0, 1)` and `exp(-nil)` is at most 1, every element contributes between 0 and 1, and the three branch weights sum to 1, so `weighted` lands in `[0, 1]` and `sps_score` in `[0, 100]`. A confidently wrong or uncovered prediction contributes 0, never a negative value, so narrow intervals around bad predictions cannot buy score.

## Zero-Score Conditions

A submission scores 0 on **every** subscore (not just SPS) if any of these hold, so guard against them:

- the prediction contains non-finite values (`NaN` / `Inf`);
- the prediction shape does not match the reference `(N, 20, 32, 64, 3)`;
- supplied `lower` / `upper` bounds are non-finite, or reversed (`lower > upper`).

The reason appears in your submission's detailed results.

A run **fails outright** and receives no score at all, rather than a zero, if the evaluator cannot proceed: a `pred_norm` of the wrong shape, bounds whose shape does not match it, returning only one of `lower` / `upper`, or starting or stopping bounds partway through a run.

## Execution Time Limit

Each submission runs under a wall-clock execution limit of 10 minutes in the Warm-up and Development phases, measured on container execution only (data download is excluded). Submissions exceeding the limit are marked Failed and receive no score. This limit is independent of the per-step Time metric above.

## Composite

The five 0-100 subscores are combined into a single 0-100 `final_score`. The combination is not published, so `scoring.py` and `local_eval.py` stop at the five subscores.

# PAGE: Submission

# Submission

Submit a zip file containing a `submission.py` at the root of the archive, exposing a TTT model in the interface below. Model weights go in the same archive.

## Expected Archive Layout

```text
submission.zip
  submission.py
  model.pth              # your base-model checkpoint
  any_supporting_files/  # optional, e.g. vendored pure-Python packages
```

## Submission Size Limit

Your submission archive must stay under **256 MB** after extraction, checkpoint included. Oversized submissions fail immediately.

Aim to meet this budget **by design** (model width, depth, and architecture) rather than by compressing an oversized checkpoint after the fact. Half-precision (fp16) packing is a lossy workaround, not the intended path, so we **discourage** relying on it; a complex-safe fp16 packing tool is included in the starting kit only as a last resort for a checkpoint marginally over budget.

## Interface Contract

`submission.py` must expose:

```python
def get_ttt_model(submission_dir: str, device: str):
    """Return an object implementing the TTT interface below."""
```

The returned object must implement:

```python
def reset_ttt_state(self) -> None
    # Called at the start of every trajectory: restore checkpoint weights,
    # clear adaptation state.

def ttt_step(self, input_norm, prev_target_norm=None)
    # Called once per step, batch size 1, normalized space.
    #   input_norm       : (1, 20, 32, 64, 3) current input window.
    #   prev_target_norm : (1, 20, 32, 64, 3) ground truth of the PREVIOUS
    #                      step, or None on the first step of a trajectory.
    # Returns (pred_norm, info): prediction of shape (1, 20, 32, 64, 3) and a
    # dict. The evaluator reads info["adapt_loss"] (a python float, or None
    # when no adaptation happened) and, optionally, info["lower"]/info["upper"]
    # (see Optional Interval Outputs below); any other keys you add are ignored.
```

Typical `ttt_step` structure: (1) if `prev_target_norm` is not None, adapt using your cached previous input and the revealed previous target; (2) predict on the current input; (3) cache the current input for the next step. A complete reference implementation is included in the starting kit.

## Optional Interval Outputs

Add `lower` / `upper` keys to the `info` dict to return per-element uncertainty bounds, in the same normalized space as `pred_norm`:

```python
return pred_norm, {
    "adapt_loss": adapt_loss,
    "lower": lower_norm,
    "upper": upper_norm,
}
```

`lower_norm` / `upper_norm` must have the same shape as `pred_norm`. The evaluator denormalizes them with the same transform it applies to `pred_norm`. Bounds are optional, but if you return one you must return both, since a lone `lower` or `upper` key fails the run at ingestion. This applies for the whole run, not just one call: once you start returning bounds you must return them on every remaining step, and if you never return them you must not start partway through. If you omit both, the scorer falls back to a default interval (see the Evaluation page's SPS section).

## Evaluation Environment

Submissions run offline in the official Track 2 image `w3nhao/realpde-track2@sha256:f07d1d68e19cb7ed5f2f405b6775f4fafb985887ac78356a9599f35365ccf972` (PyTorch 2.2.2, CUDA 12.1, Python 3.10). **Nothing is installed at evaluation time.** The only permitted network access is the organizer LLM gateway (see LLM Access below); no other outbound network is allowed.

Available libraries: `torch` / `torchvision` / `torchaudio` 2.2.2, `numpy` 1.26, `scipy`, `pandas`, `matplotlib`, `h5py`, `einops`, `scikit-learn`, `tensorboard`, `safetensors`, `openai` (for the organizer LLM gateway), `pillow`, `sympy`, `networkx`, `pyyaml`, `requests`, `tqdm`. Exact resolved versions are recorded in `/opt/image_manifest.txt` inside the image.

Vendor any other pure-Python dependency inside your archive next to `submission.py`.

Your model runs in an isolated subprocess. It reads your submission normally, writes temp files, caches and checkpoints as usual, and uses torch/CUDA/numpy as usual. Only the hidden evaluation data (ground truth) is unreadable from disk; any attempt to open it is denied by the kernel.

## LLM Access (optional)

Agent-style methods may call an LLM through the organizer-provided gateway. It is OpenAI-compatible and reachable inside the container via environment variables; no participant key is needed.

- `OPENAI_BASE_URL`, `OPENAI_API_KEY`: gateway endpoint and credential (injected).
- `REALPDE_GATEWAY_MODEL`: available model name (currently `gemini-3.5-flash`).

```python
import os
from openai import OpenAI

client = OpenAI()  # reads OPENAI_BASE_URL and OPENAI_API_KEY
resp = client.chat.completions.create(
    model=os.environ["REALPDE_GATEWAY_MODEL"],
    messages=[{"role": "user", "content": "..."}],
)
```

Only chat completions are exposed; admin and key-management routes are blocked. The gateway is the only permitted network endpoint; participant keys, external LLM services, and other direct network access are prohibited (see Rules). LLM latency counts toward the Time metric, so budget your calls.

## Practical Notes

- The evaluator handles normalization; work entirely in normalized space.
- Everything inside `ttt_step` is timed, and `reset_ttt_state` as well, charged to the step that follows the boundary. Model construction and checkpoint loading are not timed.
- Keep per-step cost bounded: the whole run must finish within the execution time limit (see Rules).
- Use relative paths inside your submission archive.

# PAGE: Rules

# Rules

## Eligibility

The challenge is open to individuals and academic or industrial teams worldwide. Each team may have at most three members. Multiple accounts for the same team or individual are prohibited.

Anyone who has had access to the hidden evaluation data, to the scoring configuration, or to the private test set may not compete or share in a prize. This covers the organizers and anyone they have given that access to, whatever their affiliation. It is about access, not about where you work: a colleague of an organizer who has had none of it is eligible like anyone else.

## Submission Limits

- Warm-up Phase: each team is limited to 3 submissions per day.
- Main Development Phase: each team is limited to 1 submission per day.
- Each team is limited to 100 submissions per phase.
- Submission archives must stay under 256 MB after extraction; oversized submissions are rejected without evaluation.
- Each submission must finish evaluation within the execution time limit (10 minutes in the Warm-up and Development phases, container execution only; data download time is not counted). Submissions exceeding the limit are marked Failed and receive no score.

## Evaluation Data Integrity

The TTT protocol necessarily reveals ground truth step by step through the `ttt_step` interface. The following are **strictly prohibited** and lead to disqualification:

- Reading evaluation data files from the container filesystem instead of (or in addition to) the values passed through the interface.
- Using the current step's ground truth to influence the current step's prediction by any means.
- Encoding evaluation data into prediction outputs, logs, or any other channel that leaves the container.

Suspicious score patterns are flagged and manually reviewed. Adaptation logs are recorded on the organizer side and may be audited.

## External Resources and LLM Access

To ensure fairness, participants are not allowed to generate additional data for training, nor are they allowed to rely on pretrained models trained on anything other than the competition's released training data. Checkpoints trained from that release are permitted, including the official baselines the organizers provide. This will be guaranteed during the final retraining evaluation stage. See the FAQ for how this applies to standard data augmentation.

For development, participants may freely use open-source software. During evaluation:

- Optional LLM calls must go **exclusively** through the organizer-provided gateway (OpenAI-compatible; usage on the Submission page). Fixed model versions, token limits, timeout rules, and request logging are enforced.
- Participant-controlled API keys, external or private LLM services, and any direct network access other than the gateway are **prohibited**.
- All LLM latency counts toward the Time metric.

## Reproducibility

Winning teams must provide complete code and instructions sufficient for organizers to reproduce their solution, and must agree to open-source their complete solution before the NeurIPS results presentation event to be eligible for prizes.

## Final Verification

Final rankings are determined in the organizer-controlled Decision Phase, not by the development leaderboard. At the end of the Development Phase the **top 10 teams** on the Track 2 leaderboard are shortlisted. Organizers re-train each shortlisted team's method from scratch on the competition's GPU cluster to ensure reproducibility, then evaluate on a private test set with unseen parameter regimes; the resulting private-test scores decide the **top 3**. Organizers verify code integrity, check rule compliance, and run robustness tests. Suspicious score patterns are flagged and manually reviewed. A shortlisted solution that cannot be reproduced through re-training, or that shows evidence of improper use of evaluation data, is disqualified and replaced by the next-ranked team.

The exact final submission package format will be announced on the FAQ page about 5 days before the end of the Development Phase.

## Ranking

Track 2 ranking is determined by `final_score` on the official LTTTA leaderboard. Organizers reserve the right to audit top submissions for rule compliance and reproducibility.

## Rights Reserved

Final placements are decided by the Decision Phase procedure described under Final Verification, not by the development leaderboard and not at anyone's discretion. Reviewing a shortlisted team's code before confirming a placement is part of that procedure.

Where the procedure does not settle a question, the organizers decide, and will publish the reasoning alongside the result.

Participants agree under the Terms page not to acquire, redistribute or retain the evaluation data outside the competition. The organizers reserve the right to pursue remedies for a breach of that undertaking.

The Terms page also sets out what the organizers may do with a submission, and what may be changed during the competition.

# PAGE: FAQ

# FAQ

## FAQ

**Q: What is the submission size limit?**
A: The extracted archive, checkpoint included, must stay under 256 MB.

**Q: Can I install extra packages (requirements.txt)?**
A: No. Evaluation runs fully offline and nothing is installed. The Track 2 image already ships the vendored-framework stack (scipy, h5py, einops, and more; see the Submission page), and you can vendor pure-Python packages inside your archive if needed.

**Q: What exactly is timed?**
A: Everything inside your `ttt_step`, and `reset_ttt_state` as well, charged to the step that follows it. Model construction and checkpoint loading are not timed. The Time score uses the mean per-step wall time. Note the platform-level execution time limit (wall clock for the whole run) does include loading, so keep it reasonable.

**Q: What is the execution time limit?**
A: 10 minutes of container execution per submission in the Warm-up and Development phases (data download does not count). Exceeding it marks the submission Failed with no score. The value may be recalibrated on the official baseline; any change is announced here.

**Q: My submission failed with "Transient platform storage error while downloading the 'submission' bundle". What happened?**
A: This is a platform-side (Codabench) storage issue, not a problem with your submission. When the platform's object storage stalls while the evaluation worker fetches your archive, the submission is marked Failed. Failed submissions do NOT count against your quota: refresh the page and you will see the attempt returned. Please resubmit, ideally about an hour later so the platform has time to recover.

**Q: How do I use an LLM in my agent?**
A: Through the organizer-provided OpenAI-compatible gateway. See the **Submission** page ("LLM Access") for the environment variables, model name, and a code example; see the **Rules** page for what is prohibited. In short: the gateway is the only allowed endpoint, no participant keys, and LLM latency counts toward Time.

**Q: Why can't I download my prediction output?**
A: The TTT protocol reveals ground truth to your code step by step, so the prediction file is withheld to protect the hidden evaluation stream. You still see your scores and subscores.

**Q: What timezone are the phase deadlines in?**
A: All phase boundaries are in UTC. Each phase starts at 00:00:00 UTC on its start date and ends at 23:59:59 UTC on its end date. Deadlines are NOT Anywhere-on-Earth.

**Q: Does "not allowed to generate additional data for training" ban standard data augmentation?**
A: No. Standard augmentation of the released data is fine in both tracks: anything that takes a released sample and produces a variant of it on the spot during training. What we check is provenance. Every training sample must trace back to the released training data through a transform you can point at. Extra simulations, external datasets, and pretrained models from other sources stay out. The reason is fairness at the final retraining stage. Shortlisted methods are re-trained from scratch and organizers hold only the released training data, so we accept training code only: no new simulation data, no code that runs new simulations from scratch, no model weights, and no pretrained models from other sources. If your training cannot be reproduced from the release alone, it cannot survive that stage. Augmentation code ships with your training code and runs as part of training.

**Q: How are final rankings determined?**
A: The top 10 teams on the development leaderboard are shortlisted; organizers re-train their methods on the competition's GPU cluster and evaluate on a private test set with unseen parameter regimes, and those private-test scores decide the top 3. See Rules → Final Verification.

**Q: How and when do finalists submit their training code?**
A: The exact final submission package format (training code layout, environment specification, and an accompanying spec document) will be announced on this page about 5 days before the end of the Development Phase. Keep your training pipeline reproducible from the start so the package is easy to assemble.

**Q: Where do I ask questions?**
A: The competition forum, or realpde-competition@googlegroups.com.

# PHASES

{
 "name": "Warm-up Phase",
 "start": "2026-07-05T00:00:00Z",
 "end": "2026-07-19T23:59:00Z",
 "description": "Format validation and public smoke-test phase for Track 2 LTTTA.",
 "execution_time_limit": 600,
 "max_submissions_per_day": 3,
 "max_submissions_per_person": 100,
 "hide_output": false
}
{
 "name": "Main Development Phase (Archived)",
 "start": "2026-07-20T00:00:00Z",
 "end": "2026-08-04T22:59:00Z",
 "description": "Primary Track 2 LTTTA leaderboard on hidden validation data. (Archived)",
 "execution_time_limit": 600,
 "max_submissions_per_day": 1,
 "max_submissions_per_person": 100,
 "hide_output": false
}
{
 "name": "Main Development Phase",
 "start": "2026-08-05T00:00:00Z",
 "end": "2026-09-27T23:59:00Z",
 "description": "Primary Track 2 LTTTA leaderboard on hidden validation data.",
 "execution_time_limit": 600,
 "max_submissions_per_day": 1,
 "max_submissions_per_person": 100,
 "hide_output": false
}
{
 "name": "Final Decision Phase",
 "start": "2026-09-28T00:00:00Z",
 "end": "2026-10-25T23:59:00Z",
 "description": "Organizer-controlled final verification phase on private test data.",
 "execution_time_limit": 1,
 "max_submissions_per_day": 1,
 "max_submissions_per_person": 1,
 "hide_output": true
}


# LEADERBOARDS

[
 {
  "id": 19282,
  "primary_index": 0,
  "title": "LTTTA Results",
  "key": "lttta",
  "columns": [
   {
    "id": 84483,
    "computation": null,
    "computation_indexes": null,
    "title": "Final",
    "key": "final_score",
    "sorting": "desc",
    "index": 0,
    "hidden": false,
    "precision": 5
   },
   {
    "id": 87762,
    "computation": null,
    "computation_indexes": null,
    "title": "Rel-L2",
    "key": "rel_l2_score",
    "sorting": "desc",
    "index": 1,
    "hidden": true,
    "precision": 5
   },
   {
    "id": 87763,
    "computation": null,
    "computation_indexes": null,
    "title": "TKE",
    "key": "tke_score",
    "sorting": "desc",
    "index": 2,
    "hidden": true,
    "precision": 5
   },
   {
    "id": 87764,
    "computation": null,
    "computation_indexes": null,
    "title": "MVPE",
    "key": "mvpe_score",
    "sorting": "desc",
    "index": 3,
    "hidden": true,
    "precision": 5
   },
   {
    "id": 87765,
    "computation": null,
    "computation_indexes": null,
    "title": "Time",
    "key": "time_score",
    "sorting": "desc",
    "index": 4,
    "hidden": true,
    "precision": 5
   },
   {
    "id": 87766,
    "computation": null,
    "computation_indexes": null,
    "title": "SPS",
    "key": "sps_score",
    "sorting": "desc",
    "index": 5,
    "hidden": true,
    "precision": 5
   }
  ],
  "hidden": false,
  "submission_rule": "Force_Best"
 }
]