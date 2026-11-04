# Data

## Competition release (what we train on)

Hosted on Hugging Face: [AI4Science-WestlakeU/RealPDE-Competition-Data](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data)
(mirrored on [Google Drive](https://drive.google.com/drive/folders/1Cg23DoTuSvWXR3Mm1uRfmMNAbkyaIhrQ) — prefer HF, the Drive link hits per-file quotas).
Tracks 1 and 2 share this release. License CC BY-NC 4.0.

| File | Size | Contents |
|------|------|----------|
| `train_sim.tar.gz` | ~8.8 GB | simulated CFD trajectories (pretraining); channels `[u, v, p]` |
| `train_real.tar.gz` | ~7.4 GB | real PIV trajectories (finetuning); measured `[u, v]`, `p = 0` |
| `example_data/` | small | e.g. `3750_0.h5` for layout checks + normalization stats |
| `baseline_checkpoints/` | — | CNO, FNO (+fp16), Transolver; sim-pretrained and real-finetuned |

- File naming: `<Re>_<AoA>.h5` (e.g. `3750_0.h5` = Re 3750, AoA 0°).
- h5 layout (example file): `u`, `v` at top level with shape `(T≈600–868, 64, 128)`, plus
  `aoa`, `re` metadata and `x`, `y` grids. Data under `measured_data/` in the tarballs —
  inspect after download.
- Physical setting: NACA4418 airfoil cross-section, water tunnel, time-resolved PIV;
  matched 3D CFD. AoA ∈ {0, 5, 10, 15, 20}°, Re 2,968–27,975, ~100 paired trajectories.
- **Exclude `train_real/7575_0.h5`** — it duplicates `6300_0` (announced 2026-08-12).
  The file remains in the release; filter it in our loading code, permanently.

Download (once approved and set up):

```bash
pip install -U "huggingface_hub[cli]"
hf download AI4Science-WestlakeU/RealPDE-Competition-Data --repo-type dataset \
  --local-dir data/competition
tar -C data/competition -xzf data/competition/train_sim.tar.gz
tar -C data/competition -xzf data/competition/train_real.tar.gz
```

## `train_sim` and `train_real` are in different units

Matched conditions, **not** matched values. Comparing `train_sim/X.h5` to `train_real/X.h5`
directly is meaningless: the velocities are in different unit systems, and the fields are
independent realizations of a turbulent flow, never time-synchronized.

Measured over the whole release (95th-percentile `u` per file):

| split | scale | fit |
|-------|-------|-----|
| `train_sim` | `u95 = 1.26 ± 0.04`, **no Re dependence** | non-dimensional, freestream ≈ 1 |
| `train_real` | `u95 = 1.54e-5 · Re`, R² = 0.991 | physical m/s |

So the CFD is non-dimensionalized by freestream velocity (the solver fixes `U = 1` and sets
viscosity to hit the target Re), while the PIV is in m/s and therefore scales linearly with Re.
The `x`, `y` grids are byte-identical between the two splits and in metres either way.

To put sim on the real scale, multiply by the freestream velocity:

```python
U_inf = 1.263e-5 * re          # m/s; = Re·ν/c with ν = 1e-6 m²/s, c ≈ 79 mm
u_sim_physical = u_sim * U_inf
```

That reconciles the mean fields to within a few percent on `u`. The remaining gap is the real
sim2real gap the competition is about: sim `v` fluctuations are still ~2× the PIV ones, and
`train_real` carries an 11 % masked region (exact zeros — airfoil body plus laser shadow) that
the sim does not have (0.2 %).

Practical consequences:

- The official `mean_std_real.pt` stats are **real** statistics. Feeding raw `train_sim` through
  them puts inputs many σ off-distribution; rescale sim by `U_inf` first.
- The baselines were pretrained on sim then finetuned on real, so sim-space pretraining works —
  but any sim/real mixing we do has to apply the `U_inf` factor.

Evaluation happens at 32×64 (2× downsample of the raw 64×128) on normalized data using the
official `train_real` statistics — mirror that pipeline exactly in local eval (starting kit
`scoring.py` / `local_eval.py` are the reference).

## Hidden data

- Development leaderboard: hidden validation trajectories, streamed via the interface only.
- Final: private test set with **unseen parameter regimes** (likely unseen Re/AoA combinations —
  design for interpolation/extrapolation across conditions).
- Reading evaluation files from the container filesystem = disqualification.

## RealPDEBench (background, NOT competition training data)

The parent benchmark ([site](https://realpdebench.github.io), [paper](https://arxiv.org/abs/2601.01829),
[GitHub](https://github.com/AI4Science-WestlakeU/RealPDEBench), [HF dataset](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench))
has 5 scenarios (cylinder, controlled cylinder, FSI, foil = NACA0025, combustion), ~735
trajectories, ~783 GB. Useful for reading the baseline code (`pip install -e .`; configs under
`configs/<scenario>/`), understanding metrics, and the paper's findings (sim pretraining + real
finetuning beats either alone; long rollouts expose error accumulation).

**Provenance warning**: the competition rules restrict training to the *competition* release.
Do not train on other RealPDEBench scenarios — the final re-training stage would fail us.
Reading their code for architecture/training recipes is fine (open-source software is allowed).
