# nmucke: Track 2 training package

## Codabench
- Username: nmucke
- Submission id this package reproduces: **948041** (uploaded 2026-09-27 11:25 UTC; archive sha256
  `f11d2c1d0e00b6270ad71316117eb44e0de303b747bf9302ddcbfcdc7804504e`)
- final_score of that submission: **82.452** (rel_l2 94.939, tke 81.017, mvpe 94.500, time 92.464, sps 45.211)

## How to reproduce
- Build command: the default, `docker build -t <name> .` (the Dockerfile is the unmodified template: the official
  `w3nhao/realpde-track2` image plus `COPY . /workspace`).
- Train command: the default,
  `python -c "import solution; solution.train('/data', '/output', 'cuda')"`.
- What `train()` does:
  1. It copies `realpde/` (our training code) into `/output/_work/`.
  2. It points that copy's `data/competition/{train_sim,train_real}` at `/data`, and uses `/data/mean_std_real.pt`.
  3. It runs our one-command training chain, `scripts/retrain_all.sh`. Its default flags build exactly the method
     of submission 948041.
  4. It unpacks the resulting package into `/output/package/`.
  - The full chain log goes to `/output/train.log`, and `/output/package_info.json` records the sha256 of every
    package file.
  - `/output/_work/` holds the intermediate checkpoints and can be deleted after training.
- Inference: `solution.get_ttt_model('/output', 'cuda')` imports `/output/package/submission.py` under the module
  name `submission`, exactly as Codabench did, and returns its `get_ttt_model('/output/package', 'cuda')`.

## Reproducing the development-set result (submission 948041)
- **What `train()` reproduces:**
  - The code is identical: `/output/package`'s code equals submission 948041's, file for file.
  - The weights are a fresh retrain of the same recipe and seeds. GPU training is not bit-exact across GPU models,
    so the weights match the submitted ones statistically, not byte for byte. On one GPU model the backbone
    (`model.pth`, `prior.npz`) is reproduced byte for byte; only small refits (energy net, interval-width net) vary.
- **Expected deviation on the development set: about ±0.03 in final score.** This covers run-to-run timing noise on
  the evaluation harness (the time subscore alone varies by roughly ±0.03 final between identical uploads) and
  retrain spread.
- **Evidence:**
  1. **The leaderboard itself.** On 2026-09-25 we uploaded an archive whose weights came from an independent
     retrain on a different GPU model (RTX 4090) of that day's lineage. It scored 82.311, against 82.31 predicted
     from the original RTX 3090 build. Each accuracy subscore was within ±0.02 of the 3090 lineage.
  2. **Retrains inside the official image on an A100 (see Notes).** Every local subscore was within 0.02–0.03 of
     submission 948041's. Leave-conditions-out validation composites were within 0.007 of the reference (fold0
     81.875 vs 81.868, fold1 82.358 vs 82.353, foldR 82.365 vs 82.365).
- **Original archive available:** the original trained archive of submission 948041 (sha256 `f11d2c1d…`) is not
  used by `train()`. We can provide it on request to compare the retrained model against the leaderboard run
  directly.

## Hardware your submitted model was trained on
- GPU model and count: 1 × NVIDIA GeForce RTX 3090 (24 GB), driver 560.35
- CPU: AMD Ryzen 9 3950X (16 cores / 32 threads)
- Memory: 128 GB
- Training time on this hardware: **2 h 04 min** for the full chain from a fresh clone with nothing reused.
  The run behind submission 948041 took 19:22–21:26 UTC on 2026-09-24, plus 24 min for the energy-net refit step.
  - The same chain run from our release inside the official image on 1 × A100-SXM4 40 GB took **2 h 27 min**.
  - `train()` of this exact image, built from this folder, on 1 × A100-SXM4 40 GB with 16 cores took **2 h 17 min**
    (see Notes).
- Peak GPU memory: **7.1 GiB** (7,319 MiB; `nvidia-smi` sampled every 5 s), measured during `train()` of this image on
  1 × A100-SXM4 40 GB (an A800-class GPU; see Notes). The peak is reached in the first fold fine-tunes, when three
  training processes run at once. The chain caps each training process at a fraction of the card
  (`--mem-frac 0.15–0.30`), and runs at most three processes at once.

## Environment
- Base image: `w3nhao/realpde-track2@sha256:f07d1d68e19cb7ed5f2f405b6775f4fafb985887ac78356a9599f35365ccf972`
  (Python 3.10, PyTorch 2.2.2 + CUDA 12.1, numpy 1.26)
- Added packages and versions: **none**. Every import used by training and inference is in the base image. The
  fused inference kernels use the NVRTC library that ships with the image's PyTorch; if it is unavailable they fall
  back automatically to the unfused path, with identical outputs.

## Data, weights and code
- Training data used: only the released `train_sim` (100 files) and `train_real` (81 files; `7575_0.h5` is excluded
  and is not mounted). No other data and no pretrained weights are used.
- Augmentation: the backbone uses none. The energy-net fits use an on-the-fly flip in y of released samples.
- Official baseline checkpoints used: **none**. All checkpoints are produced by `train()`.
- Third-party code and its license:
  - `realpde/starting_kit/`, and the `rpde_baselines/` modules vendored in the package, are the competition's
    official starting kit, redistributed unchanged for the local evaluation step. Some files carry NVIDIA copyright
    headers.
  - Everything else under `realpde/` is our own code.
  - PyTorch, numpy, h5py and scipy come from the base image.

## Randomness
- Seed(s): fixed throughout, and every stage's seed is set in the chain.
  - Sim pretraining: seeds 0, 1 and 2.
  - Real fine-tunes: seeds 0–3 for each pretrain.
  - Energy nets: seeds 0–3.
  - All other fits: seed 0.
- Determinism: GPU training is not guaranteed bit-exact across runs or GPU models. On one GPU model the backbone
  `model.pth` and `prior.npz` have been reproduced byte for byte. Across machines, retrains score within about
  ±0.03 per local subscore.
- How runs or checkpoints are selected: all of it happens inside `train()`, on the training data only.
  - Backbones: no selection. Every sim pretrain and fine-tune keeps its last iterate. The ensemble is a fixed K=3
    over pretrain seeds 0/1/2, each member a uniform weight average (soup) of four fine-tunes.
  - Interval-width net (W2): three seeds (0, 1, 2) are fitted on the same cross-fitted training residuals. The one
    with the best score on its own internal validation split, carved from the training residuals, is kept.
  - Energy nets: each keeps the epoch with the best internal validation loss. The internal split is carved from the
    81 training files.
  - All pipeline constants are fixed in the code: grids, thresholds, the LSK configuration, the interval rules.

## Track 2 only
- Does the method call the LLM gateway: **no**. The method is fully offline.

## Contact
- Name and email, reachable during the Decision Phase: Nikolaj T. Mücke, nmucke@gmail.com

## Notes
- Training log of the run behind submission 948041: `logs/`; `logs/PROVENANCE.md` maps each artefact to the log
  that produced it.
- Package rehearsal: `train()` was run end to end with this exact package on the machine above, against a
  `/data`-shaped directory (see `logs/package_rehearsal/`). The package it produced was checked with the
  `get_ttt_model` interface.
- In-image rehearsal (2026-10-03): this folder was built with `docker build --platform linux/amd64` (image
  `d77f084adeb3`, 8.05 GB) and run on 1 × NVIDIA A100-SXM4 40 GB with 16 CPU cores, as an Apptainer container of
  that image. The conditions followed the organizers' command:
  - no network;
  - `/data` read-only, holding only `train_sim/` (100), `train_real/` (81, no `7575_0.h5`) and `mean_std_real.pt`;
  - a fresh empty `/output`;
  - one GPU;
  - `python -c "import solution; solution.train('/data', '/output', 'cuda')"`, run from `/workspace`.
  - Result: exit 0 in 2 h 16 min 47 s. Peak GPU memory was 7.1 GiB. Peak host RAM was 13.7 GB of process memory
    (17.4 GB with page cache). `/output` holds 9.4 GB, of which 8.7 GB is `/output/_work/` and can be deleted.
  - All 79 code files of `/output/package` (67 `.py`) are identical to those of submission 948041. The backbone
    `model.pth` and `prior.npz` are byte-identical across our independent A100 retrains.
  - Evaluation through `get_ttt_model('/output', 'cuda')` in the same image, without network, gives subscores within
    0.02 of submission 948041's on the same local data (168 steps). All speed and test-time-correction components
    are active. CPU and CUDA inputs give bit-identical outputs.
- Detailed method: `METHOD.md`. Longer reproduction notes: `realpde/docs/REPRODUCE.md` and `realpde/docs/METHOD.md`.
