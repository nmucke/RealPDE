# RealPDE 2026, Track 2: team nmucke

Code for team nmucke's (Nikolaj T. Mücke and Syver Døving Agdestein) entry to the [NeurIPS 2026 RealPDE Competition](https://realpdecompetition.github.io/),
Track 2 (Long-Term Test-Time Adaptation), which won the track. The task is streaming prediction of real
particle-image-velocimetry (PIV) measurements of the flow around a NACA 4418 airfoil: at every step the model gets 20
input frames and the ground truth of the previous step, and returns the next 20 frames of `(u, v)` on a 32×64 grid,
with optional per-element prediction intervals. Scoring combines five subscores: field error (rel-L2), turbulent
kinetic energy (TKE), mean velocity profile error (MVPE), run time, and an interval score (SPS).

The final solution is Codabench submission 948041 (2026-09-27). It ranked 1st on the development-phase leaderboard
with a final score of 82.452 (rel_l2 94.939, tke 81.017, mvpe 94.500, time 92.464, sps 45.211). The organizers then
retrained it from scratch with our training package and evaluated it on the private test set, and it won Track 2.

## Quick start

The repository ships trained weights of the final model ([`weights/final_retrain_a100/`](weights/final_retrain_a100/README.md)),
so the test-time pipeline runs without training. On the first run the package (final source code plus weights) is
assembled into `build/` after a checksum check, and it is reused afterwards.

The environment is managed with [uv](https://docs.astral.sh/uv/). [`pyproject.toml`](pyproject.toml) pins Python
3.10 and the versions of the training environment (`torch==2.2.2`, `numpy==1.26.4`, `h5py`, `scipy`, `einops`,
`matplotlib`), and `uv.lock` fixes the rest. `uv sync` creates `.venv/` (uv downloads Python 3.10 if it is missing), and
`uv run` runs a command inside it. On Linux the torch wheel is the CUDA 12.1 build, which also runs on CPU.

```bash
uv sync
```

On CPU:

```bash
CUDA_VISIBLE_DEVICES= uv run python -B examples/run_ttt.py --device cpu         # window-by-window loop: error, interval coverage, ms
```

```bash
CUDA_VISIBLE_DEVICES= uv run python -B experiments/run_preset.py --device cpu   # the five competition subscores
```

On a GPU, inside the published final image, from the repository root:

```bash
docker run --rm --network none --gpus all -v "$PWD":/repo -w /repo \
  docker.io/nmucke/realpde-t2-nmucke@sha256:20140bd5f8c86875cf9e78bd576559d4017d047f5c57a3a6c1515c69473e5dbf \
  python -B experiments/run_preset.py --data /repo/data/competition
```

Without `--data`, both scripts use the starting kit's two synthetic example trajectories. These have two windows
each, so they only check that everything runs; the steady-state CUDA graph and the per-trajectory selectors start from
the third window. With `--data` pointing at the released data (downloaded as in [`data/README.md`](data/README.md)),
the scores are in-sample, since the weights were trained on `train_real`. The evaluation data of the competition is
not public. [`experiments/README.md`](experiments/README.md) lists the presets for earlier uploads and for switching
single components off (for example `--off lsk`).

## The method in short

- The backbone is an ensemble of three SimVP-style convolutional networks (gated spatiotemporal attention, 3.39 M
  parameters each). Each member is pretrained on the 100 simulated trajectories, fine-tuned on the 81 usable real
  ones with a robust Welsch loss, and is the uniform weight average of four fine-tunes. The three members run packed
  as one network with an fp16 core.
- Test-time adaptation is causal and gradient-free, inside `ttt_step`. It consists of running context
  normalisation, an online least-squares correction of the backbone output (LSK), energy calibration with a learned
  energy net, low-energy damping, a regime-change reset, and prediction intervals from a learned width net whose
  centre and width are tuned per trajectory on the previous window's interval score.
- Each step runs as pre-recorded CUDA graphs on several streams with fused kernels, at 4.84 ms per step on the
  evaluation harness.

[`final_submission/METHOD.md`](final_submission/METHOD.md) is the method description sent to the organizers.
[`docs/writeup/final_long.pdf`](docs/writeup/final_long.pdf) is the 7-page write-up and
[`docs/components.md`](docs/components.md) describes every component with its measured effect.

## Repository layout

| Path | Contents |
|---|---|
| [`final_submission/`](final_submission/) | The training package exactly as sent to the organizers: `Dockerfile`, `solution.py`, the training and inference code (`realpde/`), `README.md`, `METHOD.md` and the training logs of the run behind submission 948041. Self-contained; see below. |
| [`experiments/`](experiments/) | Presets that switch the final solution's components on and off, to approximate each leaderboard upload of the final lineage and to measure what each component contributes. Also `leaderboard.csv`, the score history of every upload with a recorded result. |
| [`docs/`](docs/) | Competition and data reference, component glossary, per-submission overview, development history, and a 1-page and a 7-page write-up ([`docs/writeup/`](docs/writeup/)). |
| [`weights/`](weights/final_retrain_a100/README.md) | The four trained weight files of the final model (about 59 MB): a retrain by the published final image, code-identical to submission 948041. Add them to the final source dir to get a ready package. |
| [`data/`](data/) | Instructions for the competition data (not included). |
| [`pyproject.toml`](pyproject.toml), `uv.lock` | The Python environment for running the scripts in `examples/` and `experiments/` with uv (see Quick start). |

The development history (experiment notes, earlier packaged submissions, the upload ledger) is not part of this
repository. Where the docs cite it as the source of a number, they say so; the numbers themselves are reproduced here.

## Running the final submission

`final_submission/` can be downloaded and used on its own. It follows the organizers' Decision-Phase interface:
`solution.train(data_dir, output_dir, device)` retrains everything from the public training data, and
`solution.get_ttt_model(output_dir, device)` returns the model with the competition's `reset_ttt_state` /
`ttt_step` interface. The files in it are a snapshot of the development repository at release time, so a few of them
(`realpde/README.md`, `realpde/CLAUDE.md`) are leftover development notes. Two files were changed after submission:
`realpde/README.md` (its docs list and layout were corrected; three dead links removed) and `METHOD.md` (a pointer to a
paper that is not in this repository was removed). The training-log provenance cited in
`final_submission/README.md` is at `logs/run_948041/PROVENANCE.md`.

You need Docker with the NVIDIA container toolkit, one CUDA GPU (training peaks at 7.1 GiB; the chain caps each
process at a fraction of the card and was tested on 24 GB and 40 GB cards), and the competition training data in this
layout ([`data/README.md`](data/README.md) has the download commands):

```
DATA/
├── train_sim/          # 100 files
├── train_real/         # 82 files; 7575_0.h5 duplicates 6300_0.h5 and is skipped by name
└── mean_std_real.pt    # optional: official normalisation statistics (an identical copy ships with the code)
```

Build the image. The Dockerfile only adds our code to the official `w3nhao/realpde-track2` image (Python 3.10,
PyTorch 2.2.2 + CUDA 12.1, numpy 1.26). Build from a clean checkout, because everything in `final_submission/` is
copied into the image:

```bash
docker build --platform linux/amd64 -t realpde-t2-nmucke final_submission/
```

The image we submitted is also published as
`docker.io/nmucke/realpde-t2-nmucke@sha256:20140bd5f8c86875cf9e78bd576559d4017d047f5c57a3a6c1515c69473e5dbf`.

Train with the organizers' command. `DATA` and `OUT` must be absolute paths and `OUT` an existing, empty directory.
This took 2 h 17 min on one A100 40 GB with 16 CPU cores and writes about 9.4 GB to `OUT`, of which `OUT/_work/`
(8.7 GB of intermediate checkpoints) can be deleted afterwards:

```bash
docker run --rm --network none --gpus '"device=0"' --shm-size 32g -v "$DATA":/data:ro -v "$OUT":/output \
  realpde-t2-nmucke python -c "import solution; solution.train('/data', '/output', 'cuda')"
```

`OUT/package/` is then a Codabench submission (code plus four weight files, about 60 MB). Its code is identical file
for file to submission 948041's. The weights are a fresh retrain, which is bit-exact only on the same GPU model;
independent retrains on other GPUs scored within about ±0.03 of the submission per subscore.

To skip training, assemble the same package from the weights in the repository. They come from a `train()` run of
the published image on an A100 and scored within 0.007 of submission 948041's own archive on every local subscore
([`weights/final_retrain_a100/README.md`](weights/final_retrain_a100/README.md)):

```bash
cp -r final_submission/realpde/submissions/2026-10-23_*-r97/source "$OUT/package"
cp weights/final_retrain_a100/{model.pth,prior.npz,ke_net_flow.pt,cond_width_weights.pt} "$OUT/package/"
```

To score the package with the starting kit's streaming loop and scorer, run this from the repository root, inside
the same image:

```bash
docker run --rm --network none --gpus '"device=0"' -v "$PWD":/repo:ro -v "$OUT":/output \
  -v "$EVAL":/eval:ro realpde-t2-nmucke \
  python -B /repo/experiments/run_preset.py --package /output/package --data /eval lb0927_final
```

`EVAL` is a directory of `.h5` trajectories in the release format, or a starting-kit style directory with
`test_real/*.h5` and `mean_std_real.pt`. The private test set is not public, and the shipped weights are trained on
all of `train_real`, so scores on those files are in-sample. `run_preset.py` prints the five subscores; it wraps the
kit's `local_eval.py`, which also works on its own
(`python -B final_submission/realpde/starting_kit/local_eval.py --submission OUT/package --data DIR`, with `DIR`
holding `test_real/*.h5` and `mean_std_real.pt`).

Without Docker, the same chain runs from a copy of `final_submission/realpde/` (copy it first, because the chain
writes data links, checkpoints and the package into the tree it runs in). It needs a Python 3.10 environment at
`.venv/` inside the copy with the pins of `requirements-train.txt`, and the data linked under `data/competition/`:

```bash
cp -r final_submission/realpde ~/realpde-work && cd ~/realpde-work
```

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv -r requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

The chain calls `.venv/bin/python` inside the copy, so it needs this environment of its own; the root `uv sync`
environment is not used here.

```bash
mkdir -p data/competition && ln -s "$DATA"/train_sim "$DATA"/train_real data/competition/
```

```bash
DRY_RUN=1 bash scripts/retrain_all.sh   # print the resolved configuration, run nothing
```

```bash
KEEP_DUMPS=0 bash scripts/retrain_all.sh 2>&1 | tee retrain.log
```

[`final_submission/README.md`](final_submission/README.md) has the hardware, timing, seeds and determinism details,
and [`final_submission/realpde/docs/REPRODUCE.md`](final_submission/realpde/docs/REPRODUCE.md) the full
reproduction notes. To check that the code in `final_submission/realpde/` is unmodified, run this in that directory
(it skips `README.md`, the one file edited after submission; `release_MANIFEST.txt` keeps its submitted hash):

```bash
grep -v '^#' release_MANIFEST.txt | awk 'NF>=3 && $3!="README.md"{print $1"  "$3}' | sha256sum -c --quiet && echo CLOSURE_OK
```

## Which components helped

The final lineage was built from daily uploads between 2026-09-20 and 2026-09-27 (the first upload was on
2026-09-03), and almost every addition is a switch in the `CONFIG` dict of the final `submission.py`. [`experiments/`](experiments/README.md) turns
those switches into named components and presets. With one trained package you can run the final solution with any
subset of its test-time components, or with the configuration of an earlier upload. The package can come from
`train()` or be assembled from the shipped weights (`weights/final_retrain_a100/README.md`):

```bash
uv run python experiments/run_preset.py --list                             # presets and components
uv run python experiments/run_preset.py --diff lb0924_k3f_kd3              # how the 09-24 CONFIG differs from the final one
uv run python experiments/run_preset.py --package OUT/package --data EVAL l00_backbone l07_cond_width lb0927_final
uv run python experiments/run_preset.py --package OUT/package --data EVAL lb0927_final --off lsk --off lowke
```

`--list`, `--check` and `--diff` need only the Python standard library. Evaluation needs the evaluation stack (Python
3.10, PyTorch 2.2.2, numpy 1.26, h5py, scipy, einops): the `uv sync` environment above, or the image built above. `EVAL` is as described in
the previous section; pointing it at `DATA/train_real` works but gives in-sample scores for the shipped weights.

Presets that need different weights (an earlier backbone, a single network instead of the ensemble) list the
training flags to rebuild them; `experiments/train_preset.sh` runs that retrain.

Leaderboard history of the final lineage (development phase, hidden validation set):

| Upload | Added | Final score | Rank at upload |
|---|---|---|---|
| 09-20 | SimVP backbone, mixed-space Welsch loss, energy-target blend, learned interval widths, nine CUDA graphs | 81.701 | 7 |
| 09-22 | Width net v2 fitted on cross-fitted residuals, whole-step CUDA graph | 81.828 | 7 |
| 09-23 | Smaller b5 backbone with sim pretraining, first-window graphs, kernel cuts | 81.999 | 3 |
| 09-24 | K=3 ensemble of 4-way soups with fp16 core, interval-centre blend, per-trajectory width and energy-blend selection | 82.308 | 1 |
| 09-25 | Retrain on a second GPU model; extended selectors, regime-change reset, more kernel cuts | 82.311 | 2 |
| 09-26 | 4-seed energy net, low-energy damping, multi-stream graph, fused kernels, by-value output transport | 82.419 | 2 |
| 09-27 | Online least-squares correction (LSK), more side-stream work | 82.452 | 1 |

[`docs/submissions.md`](docs/submissions.md) goes through every upload: what was added, the subscores, the change
against the previous upload and the best score so far. The
same numbers are in [`experiments/leaderboard.csv`](experiments/leaderboard.csv), and
[`docs/history.md`](docs/history.md) describes how the solution got there, including the approaches that were
dropped. Measured per-component effects on held-out folds are in [`docs/components.md`](docs/components.md) and
§8 of [`final_submission/METHOD.md`](final_submission/METHOD.md).

Package source directories (in `final_submission/realpde/submissions/`) carry date
prefixes such as `2026-10-23_…`. These are sequence labels that ran ahead of the calendar during development, not
upload dates; the upload dates are the ones in `experiments/leaderboard.csv`.

## Data and rules

Training uses only the released `train_sim` and `train_real` files. No external data and no pretrained weights are
used, including the official baseline checkpoints. No evaluation data is read anywhere. The repository contains no
competition data. The only trained weights are those of the final model, in
[`weights/final_retrain_a100/`](weights/final_retrain_a100/README.md): a retrain by the published final image,
distributed under CC BY-NC 4.0 like the data they were trained on.

## License

Our code is under the MIT license ([`LICENSE`](LICENSE)). The official starting kit and the third-party model code
vendored with it keep their own licenses, some of them non-commercial; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The competition data (RealPDEBench) is CC BY-NC 4.0.

## How this was built

Much of the engineering, the reviews of each change and the bookkeeping were done with Claude-based AI agents working
under our direction.

## Citation

```bibtex
@misc{muecke2026realpde,
  title        = {Team nmucke: Track 2 solution to the NeurIPS 2026 RealPDE Competition},
  author       = {M{\"u}cke, Nikolaj T. and Agdestein, Syver D{\o}ving},
  year         = {2026},
  howpublished = {\url{https://github.com/nmucke/RealPDE}}
}
```
