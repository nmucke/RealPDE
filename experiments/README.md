# experiments/: component selection for the final package

The final Track 2 package (`final_submission/`, Codabench submission 948041, final score 82.452)
was built up over a month of daily uploads. Almost every piece added on the way is a switch in
the module-level `CONFIG` dict of the final `submission.py`, documented there as "None / False =
off = the parent bit for bit". This directory turns those switches into named components and
presets, so that one package can be run with any subset of its components, and each leaderboard
upload of the final lineage can be approximated or rebuilt.

| file | what it does |
|---|---|
| `ablate.py` | library: reads the final CONFIG, maps component names to CONFIG values, checks dependencies, loads a package and builds the model through the package's own `get_ttt_model` |
| `presets/*.json` | 26 presets: 7 leaderboard uploads, an 18-rung component ladder, 1 execution ablation |
| `make_package.py` | assembles a ready package (final source + four weight files, hash-checked) from `weights/final_retrain_a100/` or another weight dir |
| `run_preset.py` | CLI: `--list`, `--diff`, `--check` (no torch needed); evaluation with the starting kit's own `local_eval.py` and `scoring.py` |
| `composite.py` | least-squares estimate of the leaderboard final score from the five subscores, with its residuals |
| `train_preset.sh` | retrains the weights a preset needs with `retrain_all.sh` and assembles a package |
| `tests/test_presets.py` | CPU tests, no data or trained weights needed; `tests/dummy_weights.py` writes synthetic weights |
| `leaderboard.csv` | every recorded leaderboard upload with its subscores (maintained separately) |

Nothing here edits `final_submission/`. Packages are read, never written: bytecode writing is
switched off while they are imported.

Naming: the date prefix of a package source dir (`2026-09-29_…` to `2026-10-23_…`) is a sequence
label given when the package was built, not its upload date; upload dates are in
`leaderboard.csv` and in each preset's `approximates.date`. The source dirs in this repository are
the 13 the training chain still packages, under `final_submission/realpde/submissions/<name>/source/`;
those of earlier uploads (`lb0920` to `lb0924`) are not published, and their presets give only the
package name (`approximates.package`).

## Quick start

Host requirements: `--list`, `--check` and `--diff` need only the Python standard library.
Evaluation and training need the eval-image stack (Python 3.10, torch 2.2.2, numpy 1.26, h5py,
scipy, einops) or the image itself.

With the shipped weights (`weights/final_retrain_a100/`) nothing has to be trained or assembled by
hand:

```bash
uv run python -B experiments/run_preset.py --data DATA                                  # the final model
uv run python -B experiments/run_preset.py --data DATA l00_backbone lb0926_fa948762 --off lsk
uv run python -B examples/run_ttt.py --data DATA                                        # the bare TTT loop, per window
CUDA_VISIBLE_DEVICES= uv run python -B examples/run_ttt.py --device cpu                 # CPU, kit example data
```

Without `--package`, `run_preset.py` and `examples/run_ttt.py` call `make_package.py`, which checks
the weights against `weights/final_retrain_a100/SHA256SUMS`, copies the final source dir and the four
weight files to `build/package_final_retrain_a100/` (gitignored), checks the result against the
`package_info.json` that `train()` wrote, and reuses the dir while it is unchanged. Without preset
ids, `run_preset.py` evaluates `lb0927_final`, the final package. `--weights DIR` assembles from
another weight dir (with its own `SHA256SUMS`). `examples/run_ttt.py` is a short stand-alone script
(no dependency on `ablate.py`) that drives `get_ttt_model`, `reset_ttt_state` and `ttt_step` as the
harness does and prints rel-L2, interval coverage and width per window.

A package dir (PKG below) is the final source dir plus the four weight files `model.pth`,
`prior.npz`, `ke_net_flow.pt` and `cond_width_weights.pt`. `make_package.py` builds one from a
weight dir, `final_submission/solution.py train()` writes one to `<output>/package`, and
`train_preset.sh` assembles one for older weight sets.

```bash
uv run python experiments/make_package.py [OUT] [--weights DIR]       # prints the package path
uv run python experiments/run_preset.py --list                        # presets and components
uv run python experiments/run_preset.py --diff lb0924_k3f_kd3         # each changed key: final value -> preset value
uv run python experiments/run_preset.py --check                       # validate every preset

uv run python experiments/run_preset.py --package PKG --data DATA lb0927_final lb0926_fa948762 l07_cond_width
uv run python experiments/run_preset.py --package PKG --data DATA lb0927_final --off lsk --off rec
uv run python experiments/run_preset.py --package PKG --data DATA lb0927_final --set gain_max=10
uv run python experiments/run_preset.py --package PKG --data DATA --as-packaged      # CONFIG untouched
```

DATA may be a starting-kit dir (`test_real/*.h5`), the training layout (`train_real/*.h5`; a
warning says the scores are in-sample, see Caveats) or a plain dir of `.h5` files. Normalisation
statistics come from `--stats`, else `DATA/mean_std_real.pt`, else the kit's copy. Without
`--data` the kit's example data is used: 2 synthetic trajectories of 2 windows each, so only
windows 0 and 1 run and the whole-step graph, the per-trajectory selectors (kw, eb, br), REC,
LOWKE and KD2-KD8 are never exercised, also on a GPU. `run_preset.py` prints a note when that
happens.

The package's code must equal the final source file for file (sha256 of every `.py`); a package
built from another source dir, for example the 09-27 candidate without the 0.97 forgetting factor,
is refused for presets and can be evaluated with `--as-packaged` (its own CONFIG), or with
`--allow-modified-source` at your own risk. `--device` defaults to `cuda` when a GPU is visible,
else `cpu`.

The output has the five subscores, the two composite numbers described under Caveats, ms/step and
the execution path. A run fails, unless `--allow-inactive` is given, when a component the CONFIG
asks for did not come up on a CUDA device (for example a whole-step graph that did not capture, or
REC missing from it). `--json FILE` keeps everything, including the `local_eval` log, the runtime
report and the log of every CUDA-graph construction.

From Python, with `experiments/` on `sys.path`:

```python
import sys; sys.path.insert(0, "experiments")
import ablate
model, report = ablate.build_model(PKG, ablate.load_preset("lb0925_fze"), device="cuda")
# or, as the final package minus components:
# model, report = ablate.build_model(PKG, None, extra_off=["lsk", "kd8"])
```

Build one model per process. The package keeps module-level state (CUDA graphs, LSK hooks, a
width-net mask flag), so a second model built while the first is still referenced can disturb it;
`build_model` warns in that case. `run_preset.py` runs each preset in its own process.

## Presets

### Leaderboard uploads of the final lineage

Each preset sets the CONFIG of that upload on the final source. The last two columns give the
fidelity when run on the final package's weights, and after retraining with the preset's
`training.env` flags (column "weights").

| preset | upload | LB final | LB rel_l2 / tke / mvpe / time / sps | added relative to the row below | weights | final weights | own weights |
|---|---|---|---|---|---|---|---|
| `lb0927_final` | 09-27 f11d2c1d (948041) | 82.452 | 94.939 / 81.017 / 94.500 / 92.464 / 45.211 | LS head correction (LSK, forgetting 0.97) and its reset on regime change, KD8 | final | exact | exact |
| `lb0926_fa948762` | 09-26 fa948762 | 82.419 | 94.909 / 81.009 / 94.395 / 92.417 / 45.196 | 4-seed energy net, low-energy damping (LOWKE), IPW, CDIO, BVR, KD6, KD7 | final | exact | exact |
| `lb0925_fze` | 09-25 53f7865c (retrain of FZ-E) | 82.311 | 94.863 / 81.032 / 94.393 / 91.544 / 45.185 | AXE (wider energy-blend grid, per-trajectory blend ridge, CB weight 1.0 on window 0), regime-change reset (REC), IPC, KD4, KD5 | k3_ke1 | same code path, different weights | exact |
| `lb0924_k3f_kd3` | 09-24 f39128a9 | 82.308 | 94.850 / 81.051 / 94.392 / 91.603 / 45.163 | 3-member fp16 ensemble of soups, CB, AW, AX, gain clip 20, exact mask, KD2, KD3 | k3_ke1 | same code path, different weights | exact |
| `lb0923_b5sp` | 09-23 74f9d98f | 81.999 | 94.778 / 80.716 / 94.337 / 91.432 / 44.260 | one sim-pretrained b5 backbone (fp32); window-0/1 graphs, kernel diet, fp16 width net, scan on windows 0-1 only | b5sp | approximate | same code path, different weights |
| `lb0922_n1_w2` | 09-22 3919f0a7 | 81.828 | 94.704 / 80.398 / 94.376 / 91.131 / 43.939 | W2 widths, whole-step graph | n1_w2 | approximate | approximate |
| `lb0920_n1_m1` | 09-20 b7c6004b | 81.701 | 94.704 / 80.398 / 94.376 / 90.630 / 43.615 | n1 SimVP backbone, M1 widths; also stands for the 09-21 upload v8 (81.670: same predictions, timing-only change) | n1_m1 | approximate | approximate |

Leaderboard values are from `leaderboard.csv`. Uploads before 09-20 (UNet / FNO backbones) used
code paths that are not in the final source and have no preset.

The fidelity terms mean the following. "exact": the CONFIG equals the upload's, and the code
switched off was checked bit-exact against the upload's code when it was added (each preset's
`fidelity.why` names the check; the notes that record it are not published). The
weights are a retrain of the same recipe; GPU training is not bit-exact across GPU models, and
identical uploads already varied by about ±0.03 final through timing (`final_submission/README.md`).
"same code path, different weights": as exact, but the weights differ from the upload's; for
`lb0925` / `lb0924` on the final weights only `ke_net_flow.pt` differs (the 4-seed energy net
instead of one net fitted on the ensemble). "approximate": code the CONFIG cannot revert differs,
or the weights are of another architecture. On the final weights, `lb0923` runs the 3-member
ensemble in TF32 instead of one fp32 b5 backbone, and `lb0922` / `lb0920` run it instead of the n1
backbone. On their own weights, `lb0922` / `lb0920` still differ from their uploads by the fp16
width net inside the whole-step graph (added 09-23, not bit-exact, no flag), and `lb0920` also by
the whole-step graph replacing per-stage graphs (predictions bit-exact, bounds within 1e-6).

`tests/test_presets.py` checks, for every lineage preset whose historic source is in the repository
(`lb0925`, `lb0926`, `lb0927`), that the resolved CONFIG equals the CONFIG of that source key by key
(keys the historic source does not have must be off). For `lb0920` to `lb0924` the same check passed
against their sources during development.

### Component ladder (final weights)

Cumulative: each rung switches on what is listed, in the order the components were developed. All
use the final weights, so the differences are inference-only. The ladder ends at `lb0927_final`
(forgetting factor 0.97 instead of 0.9).

| rung | adds | execution path |
|---|---|---|
| `l00_backbone` | raw 3-member ensemble output, no intervals (sps then uses the scorer's default band) | eager |
| `l01_gainmap` | energy rule, running bias, SPS intervals centred on the prediction | eager |
| `l02_advect` | advected last input frame blended into the fluctuation shape | eager |
| `l03_prior` | (Re, AoA) prior for the running statistics | eager |
| `l04_energy_net` | learned energy net | eager |
| `l05_centre` | decoupled interval centre, two-vector basis | per-stage graphs |
| `l06_bundle` | energy-target blend (weight 0.85), variance-preserving bias | per-stage graphs |
| `l07_cond_width` | learned conditional widths | whole-step graph |
| `l08_cb` | CB: interval centre 0.75 of the way toward the raw backbone output | whole-step graph |
| `l09_aw` | per-trajectory CB weight | whole-step graph |
| `l10_ax` | per-trajectory width multiplier and energy blend (grid 0.75-0.925) | whole-step graph |
| `l11_gm20` | gain clip 5 -> 20 | whole-step graph |
| `l12_msk` | exact scored mask | whole-step graph |
| `l13_axe` | energy-blend grid 0.45-0.925, per-trajectory blend ridge, CB weight 1.0 on window 0 | whole-step graph |
| `l14_rec` | regime-change reset (REC) | whole-step graph + KD6-KD8 |
| `l15_lowke` | low-energy damping | whole-step graph + KD6-KD8 |
| `l16_lsk09` | LS head correction, forgetting factor 0.9 | whole-step graph + KD6-KD8 |
| `l17_lskr09` | LS reset on regime change, forgetting factor 0.9 | whole-step graph + KD6-KD8 |

Execution-only levers stay at their final values throughout and switch themselves off where their
prerequisites are missing. KD6 (and with it KD7 and KD8) needs KD2, KD3, REC, both per-trajectory
blend choices (eb, br), the variance-preserving bias and CB (`kd6_conc.covered()`), so it is off on
rungs l07-l13 and comes on at l14: most of the time change at l14 is KD6-KD8's, not REC's.
`--diff` prints these inert levers. The time subscore of the eager rungs is not comparable with the
graphed ones. The scan schedule is the final one throughout (the lineage presets change it).

### Execution ablation

`t00_no_exec_levers`: the final package with KD2, KD4, KD5, KD6 (and KD7 / KD8), IPC, IPW, BVR and
CDIO off. KD3 stays on because REC needs it. KD3 on with KD2 off was never packaged by the chain, so
this combination is unverified; the preset lists `expect_active: ["rec", "kd3"]` and a GPU run fails
if either is not active. Outputs should equal the final package's except for KD5's fp16
rounding-order differences. IPC, IPW, BVR and CDIO act only when `ttt_step` runs in a worker process
or receives CUDA tensors; the kit's in-process loop with CPU tensors never triggers them, so locally
they change nothing.

## Components

`python experiments/run_preset.py --list` prints all of them; `--off NAME` switches one off. The
kind is `pred` (changes the prediction), `interval` (only the bounds), `time` (bit-exact) or `time~`
(near-exact). The place is `eager` (runs in every path), `graph` (only in the whole-step CUDA graph)
or `cuda` (an execution detail).

Not switchable at inference, because there is no CONFIG flag: the weights (backbone architecture,
soups, ensemble, prior, energy net, width net), the window-0/1 graphs and the kernel diet (both
bit-exact), the fp16 width net inside the whole-step graph (not bit-exact), and the whole-step graph
itself (built whenever its scope holds on a CUDA device).

### How the package reads CONFIG, and the rules `ablate.py` enforces

`get_ttt_model` reads the module-level `CONFIG` when it is called and passes the same dict on as
`model.cfg`; the graph wrappers read it while capturing inside `get_ttt_model`. Overrides are
therefore written into `submission.CONFIG` after import and before `get_ttt_model`; later changes
have no effect. `get_ttt_model` adds `cond_width_ckpt_path` to CONFIG, so every build imports a
fresh copy of the package: modules of the previously loaded package (by file location) and any
module named like a package file are dropped from `sys.modules`, the previous package dir is
removed from `sys.path`, and after `get_ttt_model` every package module is checked to come from the
package dir (its lazy imports included). `run_preset.py` runs several presets in separate processes.

The package's modules do `from submission import ...`, so `submission.py` is imported under the
module name `submission`, as Codabench and `final_submission/solution.py` do. The kit's
`local_eval.py` imports it as `participant_submission`, which would make those modules load a
second, unmodified copy; `run_preset.py` therefore hands `local_eval` the model it built instead of
letting it import the package.

`submission.py` prepends `<package>/../../../src` to `sys.path`, ahead of the package itself, when
that dir has a `load_baseline.py`. `ablate.py` refuses such package locations (for example a package
dir directly inside `final_submission/realpde/submissions/<name>/`); the chain builds its packages at
`<out>/build/w/x/pkg` for the same reason.

A preset is rejected for any of these combinations:

| rule | reason |
|---|---|
| `eb_adapt` without `e_blend_w`; `cb_blend_w` not a list of 20 | `get_ttt_model` crashes |
| `cb_adapt` or `cb_blend_w0` without `cb_blend_w`; `kw_adapt` without `cb_adapt`; `br_adapt` without `eb_adapt` | silently ignored by `get_ttt_model` |
| `lsk_rec_reset` without `lsk_head` and `rec_detect` | silently inert |
| `ctr`, `cond_width` or `cb_blend_w` without `bounds`; `cb_blend_w` without `ctr` | silently inert |
| `rec_detect` without `kd3_step` (which needs `fp16_core`, `mu_vp`, `ctr_basis2`, `blend` "region") | REC lives in the KD3 path |
| any of `kw_adapt`, `eb_adapt`, `br_adapt`, `rec_detect`, `tka_lowke` when the whole-step graph cannot be built | they exist only in that graph; it needs `cond_width`, `bounds`, `ctr`, `advect`, `gain`, `prior`, `e_blend_w` and the energy net (`ke_flow` set, `ke_flow_scale` not 0), none of `uc_fixed`, `uc_guard`, `mean_mode`, `er_online`, and no forced eager path |

The per-stage graphs (`cuda_graph_tail`, `_pre`, `_megatail`, `_bounds`, `cond_width_graph`, and
the whole-step and window-0/1 graphs built from them) hard-code the shipped values of `advect`,
`gain`, `bias`, `bounds`, `prior`, `ke_flow`, `ctr`, `ctr_bias`, `ctr_basis2` and a few unused
knobs, and are captured whenever their state preconditions hold. For a preset that changes any of
those keys, `build_model` replaces these graph classes in the freshly loaded module with disabled
stand-ins before `get_ttt_model` runs, so nothing is captured under that CONFIG and every step runs
the eager code, which reads CONFIG. `force_eager` then checks that no such graph is attached.

Inert execution flags only produce warnings: KD2 needs the three per-trajectory selectors (AW, kw,
eb); KD3 needs the fp16 core; KD6, KD7 and KD8 need what is listed in the ladder section; LSK with a
single non-fp16 backbone switches itself off (`report["lsk_error"]` says so).

`report = ablate.runtime_report(model)` reads back what was built: which eager components exist,
whether the whole-step graph was captured, which graph components and KD levels are live, capture
and self-test errors, and the CUDA RNG probe the package runs after its captures. It uses `getattr`
throughout, so it also works on historic packages whose wrappers lack the newer attributes.
`report["graph_log"]` lists every graph-wrapper construction with its result; `report["problems"]`
lists components the CONFIG asks for that did not come up.

## Shipped weights

`weights/final_retrain_a100/` holds the four weight files of the final model from a `train()` run
of the published final image on an A100 (CC BY-NC 4.0; provenance, hashes and local scores in its
README). They are a retrain of submission 948041's recipe, not that upload's own weights. They
serve every preset whose weight set is `final` (`lb0927`, `lb0926`, the ladder, `t00`); the other
lineage presets need the retrains below. With them, the final model on CPU took about 0.8 s for
`get_ttt_model` and about 0.2 s per window on the kit example data (4 cores; windows 0 and 1 only,
no CUDA graph).

`make_package.py` writes a marker file, `.make_package.json`, into the package it assembles so that it
can recognise and reuse it. To upload that dir as a Codabench submission, zip it without the marker.

## Retraining the weights for a preset

Weight sets and the `retrain_all.sh` flags that build them (from a copy of the training tree, data
in `data/competition/`):

| weights | flags | presets | output dir (model.pth, prior.npz, ke_net_flow.pt) | width net | model.pth arch |
|---|---|---|---|---|---|
| final | none | `lb0927`, `lb0926`, ladder, `t00` | `data/checkpoints/sp/ship_enrx3_soup4_kes4` | `data/checkpoints/b1_3090/ship_condwidth_v2/cond_width_weights.pt` | `b5x3_h192_blk6_rf11` |
| k3_ke1 | `FREEZE_DEFAULTS=0 AXE=1 IPC=1 KD5=1 KD4=1 REC=1` (09-25), `FREEZE_DEFAULTS=0` (09-24), or `KE3=0` | `lb0925`, `lb0924` | `data/checkpoints/sp/ship_enrx3_soup4` | as final | `b5x3_h192_blk6_rf11` |
| b5sp | `BACKBONE=b5sp` | `lb0923` | `data/checkpoints/sp/ship` | as final | `b5_h192_blk6_rf11` |
| n1_w2 | `BACKBONE=n1` | `lb0922` | `data/checkpoints/ship_simvp_s3` | `data/checkpoints/ship_condwidth_v2/cond_width_weights.pt` | `n1_simvp` |
| n1_m1 | `BACKBONE=n1 CONDWIDTH_RECIPE=m1` | `lb0920` | `data/checkpoints/ship_simvp_s3` | `data/checkpoints/ship_condwidth/net_all3.pt` | `n1_simvp` |

`DRY_RUN=1` resolution of each preset's flags in `final_submission/realpde`. A dry run checks the
flag logic and the package-source guards, not that the steps succeed.

| flags | resolution |
|---|---|
| none | resolves, step-10 source present |
| `LSK=0 LSKR=0 KD8=0 LSKR97=0` | resolves, source present |
| `FREEZE_DEFAULTS=0 AXE=1 IPC=1 KD5=1 KD4=1 REC=1` | resolves, source present |
| `FREEZE_DEFAULTS=0` | FATAL: its package source is not in this repository (the `lb0925_fze` flags build the same weights) |
| `KE3=0` | resolves |
| `BACKBONE=b5sp`, `BACKBONE=n1`, `BACKBONE=n1 CONDWIDTH_RECIPE=m1` | resolves; step-10 source (2026-10-06) missing, so `train_preset.sh` runs steps 01-09 only |

Where the step-10 source is missing, only the weights are retrained; the preset's package is then
the final source with those weights (`run_preset.py` switches components off by CONFIG). For the n1
presets the chain would package the 2026-10-06 source (whose n1 path is bit-exact to 2026-10-05),
not the 09-20 / 09-22 upload sources.

`train_preset.sh` does this end to end:

```bash
uv run bash experiments/train_preset.sh lb0925_fze /data /scratch/w_fze --dry-run    # resolution and checks only
uv run bash experiments/train_preset.sh lb0925_fze /data /scratch/w_fze              # about 2 h on one GPU
uv run python experiments/run_preset.py --package /scratch/w_fze/package_lb0925_fze --data DATA lb0925_fze
```

`/data` holds `train_sim/`, `train_real/` and optionally `mean_std_real.pt`, as for
`solution.train()`. The script copies the tree into `WORK_DIR/tree` (and refuses a WORK_DIR it did not
create), links the data, checks the chain's resolved output dir and width-net path
against the preset, runs the chain (all steps if the resolved package source exists in the tree,
else steps 01-09), checks the `arch` key of `model.pth`, and assembles `WORK_DIR/package_<preset>`
from the final source and the four weight files. Paths with whitespace are rejected. A historic
package from step 10 (`WORK_DIR/tree/submissions/retrained_*/submission.zip`) can be unzipped and
evaluated as is with `run_preset.py --package DIR --as-packaged`. Presets that share a weight set
can share a work dir.

## Caveats

Local data is in-sample. Every shipped weight set is trained on all 81 usable `train_real` files,
and the hidden evaluation set is not released, so local scores on `train_real` files are memorised
(the chain's own step 11 calls them a sanity check). Component differences measured this way are
biased, most for components that correct backbone error. The development history measured
components on leave-conditions-out fold pipelines (not published); this tooling does not
build fold packages for the final lineage.

The leaderboard's final-score formula is unpublished. The `fit` column is a least-squares
combination of the five subscores fitted by `composite.py`, at run time, to the 14 rows of
`leaderboard.csv` that have all five subscores and a final score (every recorded upload from 09-04
to 09-27, including the two the board did not keep, scored by their own row). Weights: rel_l2
0.447, tke 0.110, mvpe 0.117, time 0.098, sps 0.244 (sum 1.015). Maximum residual 0.006 in-sample;
leave-one-out maximum 0.025 (the 09-04 row) and rms 0.008; leave-one-out weight ranges are within
±0.02 of these values. The rows cover rel_l2 93.9-94.9, tke 76.9-81.1, mvpe 93.2-94.5, time
78.3-92.5 and sps 36.3-45.2. Two rows carry leverage near 1 (09-04: 0.94, 09-07: 0.96; 09-08: 0.89),
so the time and tke weights rest mostly on them; a near-zero leave-one-out residual on such a row
says little. A result outside the fitted range, or with a leverage above the largest row leverage,
is marked `*` and is not reliable.
In-sample local runs usually fall outside it (rel_l2 near 96), so compare subscores, not `fit`.
`plan_rel` uses the team's planning weights (0.48 / 0.07 / 0.10 / 0.09 / 0.28,
`final_submission/realpde/scripts/lco_analyze.py`); it sits 0.78-0.94 below the leaderboard final on
every row, so use it only for differences between presets. The time subscore is the local
machine's.

CPU runs check plumbing only. On CPU no CUDA graph is built, so the graph-only components (kw, eb,
br, REC, LOWKE) and KD2-KD8 are inactive, and the fp16 backbone core runs in fp32
(`ship_b5x2_models.py` falls back). The package computes on CUDA whenever one is visible, whatever
`--device` says; to force a CPU run on a GPU machine use `CUDA_VISIBLE_DEVICES=` together with
`--device cpu`.

The list of keys the per-stage graphs hard-code (`ablate.GRAPH_SPECIALISED`) was compiled from the
graph modules' own code and comments; ladder rungs l00-l04 rely on it to run eager. The eager code is
what the ungraphed packages up to 09-18 ran; with these weights on a GPU it has not been compared
against a graphed run.

## Tests

```bash
uv run python -m pytest experiments/tests/test_presets.py                        # fast suite
ABLATE_SLOW_TESTS=1 uv run python -m pytest experiments/tests/test_presets.py    # + smoke tests
uv run python experiments/tests/test_presets.py                                  # without pytest
```

CPU only (the test module hides GPUs), no data or trained weights. On a 4-core CPU the fast suite
takes about 6 s and the full suite about 60 s.

The fast suite (17 tests) checks: preset fields; component keys exist and are on in the final
CONFIG; every preset passes the dependency checks; lineage presets equal the historic CONFIGs; the
lineage only switches components off going back in time; each dependency rule fires, including the
KD6 coverage warning; `--list` / `--check` / `--diff` and the CLI argument errors with torch
blocked; the three data layouts; the composite fit; loading a second package drops the first one's
modules and `sys.path` entry; a package with modified code is refused; unknown `expect_active`
names are rejected; the graph stand-ins; `runtime_report` on wrappers without the newer
attributes. Synthetic weights (`tests/dummy_weights.py`: random 3-member b5 ensemble, random prior,
energy net and width net) are written for the loading tests.

The slow suite adds: every preset through `get_ttt_model` on CPU with two trajectories of three
synthetic `(1, 20, 32, 64, 3)` windows (finite outputs, bounds all-or-nothing and ordered, the
runtime report matching the CONFIG, no reported problems, the package dir unchanged), the warning
for a second live model, two presets through the kit's `local_eval.py` via `run_preset.py`, and,
when `weights/final_retrain_a100/` exists, the final model with those weights through
`run_preset.py` and `examples/run_ttt.py` on CPU (finite outputs, bounds present and ordered, no
reported problems). The fast suite also checks that `make_package.py` refuses a corrupted weight
file and a directory it did not make.

Not covered: anything on a GPU (graph capture, the stand-ins inside a real `get_ttt_model`, KD
levels, REC, LOWKE, the per-trajectory selectors, timing), `train_preset.sh` beyond `--dry-run`
and its assembly step, and any score on real evaluation data (the shipped-weights test runs on the
kit's synthetic example data only).
