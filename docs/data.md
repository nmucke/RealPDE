# Data

This page summarises the competition's data release as described on the Codabench Track 2 pages on 2026-08-20
(source: `https://www.codabench.org/api/competitions/17385/`), plus measurements of the release made during
development. The data are not redistributed in this repository.

## The release

Hosted on Hugging Face: [AI4Science-WestlakeU/RealPDE-Competition-Data](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data)
(a [Google Drive mirror](https://drive.google.com/drive/folders/1Cg23DoTuSvWXR3Mm1uRfmMNAbkyaIhrQ) exists but hits
per-file quotas). Tracks 1 and 2 share this release. License: CC BY-NC 4.0.

| File | Size | Contents |
|------|------|----------|
| `train_sim.tar.gz` | about 8.8 GB | 100 simulated CFD trajectories; channels `u, v, p` |
| `train_real.tar.gz` | about 7.4 GB | 82 real PIV trajectories; measured `u, v`, `p = 0` |
| `example_data/` | small | an example file (e.g. `3750_0.h5`) and the official normalisation statistics `mean_std_real.pt` |
| `baseline_checkpoints/` | | CNO, FNO (+ fp16), Transolver; sim-pretrained and real-fine-tuned (not used by the final solution) |

- File names are `<Re>_<AoA>.h5` (e.g. `3750_0.h5` = Re 3750, AoA 0°).
- Each file holds `u` and `v` at the top level with shape `(T, 64, 128)`, T between about 600 and 868 frames, plus
  scalar `re` and `aoa` and the `x`, `y` grids. The training code reads `f["u"]`, `f["v"]`, `f["re"]`, `f["aoa"]`
  directly.
- Physical setting: NACA4418 airfoil cross-section in a water tunnel, time-resolved PIV, matched 3D CFD. AoA ∈
  {0, 5, 10, 15, 20}°, Re 2,968–27,975.
- `train_real/7575_0.h5` duplicates `train_real/6300_0.h5` (organizer announcement of 2026-08-12) and is not used
  for training. It can stay in the directory: the training chain skips it by name, which leaves 81 usable real
  files.

## Download and layout for retraining

`final_submission/solution.py` provides `train(data_dir, output_dir, device)`. It expects:

```
<data_dir>/
├── train_sim/          # 100 .h5 files
├── train_real/         # 82 .h5 files as released (7575_0.h5 is skipped), or the 81 usable ones
└── mean_std_real.pt    # optional: the official normalisation statistics from example_data/
```

If `mean_std_real.pt` is absent, the chain uses the identical copy in
`final_submission/realpde/starting_kit/example_data/mean_std_real.pt`; if it is present and differs, it replaces that
copy (`final_submission/solution.py`, `_prepare_work_repo`).

From download to that layout, using `data/competition/` in this repository as `<data_dir>`:

```bash
pip install -U "huggingface_hub[cli]"
DATA=$PWD/data/competition
hf download AI4Science-WestlakeU/RealPDE-Competition-Data --repo-type dataset --local-dir "$DATA"
tar -C "$DATA" -xzf "$DATA/train_sim.tar.gz"     # -> $DATA/train_sim/*.h5
tar -C "$DATA" -xzf "$DATA/train_real.tar.gz"    # -> $DATA/train_real/*.h5
cp "$DATA/example_data/mean_std_real.pt" "$DATA/"   # optional
ls "$DATA"/train_sim/*.h5 | wc -l    # 100
ls "$DATA"/train_real/*.h5 | wc -l   # 82 (81 usable)
```

These are the commands used during development, which put the files at `data/competition/train_sim/*.h5` and
`data/competition/train_real/*.h5`; the training chain checks exactly those paths. The internal directory structure of
the tarballs is not recorded in this repository: if an archive unpacks to a different directory name or an extra
level, move the `.h5` files so that they sit directly in `train_sim/` and `train_real/`. The `.tar.gz` files and
`baseline_checkpoints/` can be deleted after extraction.

Then, from `final_submission/`:

```bash
python -c "import solution; solution.train('$DATA', '/path/to/output', 'cuda')"
```

`final_submission/README.md` describes the same run inside the official image (`/data` mounted as `<data_dir>`), and
`data/README.md` the alternative of running the chain directly in a checkout.

## `train_sim` and `train_real` use different units

The two splits have matched conditions, not matched values. Comparing `train_sim/X.h5` with `train_real/X.h5`
directly is meaningless: the velocities are in different unit systems, and the fields are independent realisations of
a turbulent flow that are never time-synchronised.

Measured over the whole release (95th-percentile `u` per file):

| split | scale | fit |
|-------|-------|-----|
| `train_sim` | `u95 = 1.26 ± 0.04`, no Re dependence | non-dimensional, free stream ≈ 1 |
| `train_real` | `u95 = 1.54e-5 · Re`, R² = 0.991 | physical m/s |

The CFD is non-dimensionalised by the free-stream velocity (the solver fixes U = 1 and sets the viscosity to reach
the target Re), while the PIV is in m/s and scales linearly with Re. The `x`, `y` grids are byte-identical between
the two splits and in metres in both.

To put simulation data on the real scale, multiply by the free-stream velocity:

```python
U_inf = 1.263e-5 * re          # m/s; = Re·ν/c with ν = 1e-6 m²/s, c ≈ 79 mm
u_sim_physical = u_sim * U_inf
```

This brings the mean fields within a few percent on `u`. The remaining difference is the sim-to-real gap: simulated
`v` fluctuations are still about 2× the PIV ones, and `train_real` has an 11 % masked region (exact zeros: airfoil
body plus laser shadow) that the simulation does not have (0.2 %).

Consequences:

- The official `mean_std_real.pt` holds real-data statistics. Raw `train_sim` passed through them is many standard
  deviations off-distribution; simulation data have to be rescaled by `U_inf` first.
- The official baselines were pretrained on simulation and fine-tuned on real data; any mixing of the two needs the
  `U_inf` factor.

Evaluation runs at 32×64 (a 2× subsampling of the raw 64×128 grid) on data normalised with the official `train_real`
statistics. The starting kit's `scoring.py` and `local_eval.py` reproduce this pipeline.

## Hidden data

- Development leaderboard: hidden validation trajectories, streamed through the interface only.
- Final evaluation: a private test set with unseen parameter regimes.
- Reading evaluation files from the container filesystem leads to disqualification.

## RealPDEBench (background, not competition training data)

The parent benchmark ([site](https://realpdebench.github.io), [paper](https://arxiv.org/abs/2601.01829),
[GitHub](https://github.com/AI4Science-WestlakeU/RealPDEBench), [HF dataset](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench))
has 5 scenarios (cylinder, controlled cylinder, FSI, foil = NACA0025, combustion), about 735 trajectories and about
783 GB. Its code documents the baselines (`pip install -e .`; configs under `configs/<scenario>/`) and its paper
reports that simulation pretraining followed by real fine-tuning beats either alone, and that long rollouts expose
error accumulation.

The competition rules restrict training to the competition release, so the other RealPDEBench scenarios are not
allowed as training data; reading their open-source code is allowed.
