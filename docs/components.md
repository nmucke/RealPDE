# Components of the final solution

This is a glossary of every component in the final Track 2 solution (Codabench submission 948041, archive
`f11d2c1d`, development-leaderboard final score 82.452). For each component it says what the component does, whether
it changes accuracy/interval scores or only the Time subscore, where it lives (training side, needing a retrain or
refit, or inference side, switched by a `CONFIG` key in `submission.py`), and what was measured when it was added.

Conventions used below:

- "Local" numbers are changes in our local composite score, computed by streaming held-out real trajectories
  through the exact submission code, scoring with the official `scoring.py`, and holding the time subscore fixed.
  The folds hold out whole (Reynolds number, angle of attack) conditions: fold1 is the selection fold, fold0 the
  primary held-out fold, foldR holds out 16 real files at angles that stay in training, chosen to mimic the pattern of the 18 (Re, AoA)
  conditions that exist only in the simulation data (top-Re extrapolation, Re 13,950 interpolation, a high-Re block at
  AoA 15, two single points), Rx is Reynolds extrapolation, and foldS-lo / foldS-hi
  are simulation-only conditions used as surrogates for regimes outside the real training range (foldS-hi is an
  emulated Re 40–53k regime, beyond the hidden set's stated maximum of 27,975).
- Local numbers are increments at the time a component was adopted, on the stack of that time. They are not a
  leave-one-out ablation of the final stack. Leaderboard changes were typically about 0.6 times the fold0 change.
- Leaderboard ("board") deltas are only attributable to a single component when an upload changed only that
  component. Most uploads changed several things; this is stated per component. The uploads are listed in
  `docs/submissions.md` and `experiments/leaderboard.csv`.
- "Off value" is the `CONFIG` value that restores the parent code path. Unless noted, the source comments state
  that the off value reproduces the parent bit for bit. The `CONFIG` dict is at lines 272–496 of
  `final_submission/realpde/submissions/2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk-recr-kd8-r97/source/submission.py`.
- Submission directory names (`final_submission/realpde/submissions/`) start with a
  `YYYY-MM-DD_` prefix that is a sequence label, not an upload date; the labels run past the end of the development
  phase (the final archive is labelled 2026-10-23 and was uploaded on 2026-09-27). Upload dates in this document are
  taken from the development ledger.
- The development used Claude-based AI agents for engineering, independent reviews and verification, under human
  direction. In the development notes, `FR<n>` reports are
  reviews by a separate agent running a different model, which re-ran the evidence and returned GO / NO-GO;
  `V<n>` notes are verification runs on the exact submission zip; `R1<x>` are from-scratch rehearsals of the training
  chain. "c2" names the second machine (an RTX 4090) and the agent session that worked on it.
- Chain flags (`CB=1`, `LSK=1`, …) are the switches of `scripts/retrain_all.sh` in `final_submission/realpde/`.
  Apart from `KE3`, they only select which tracked package source is assembled; they do not change any fit.
- Sources cited as "development notes" (experiment notes and review reports) or "development ledger" (the upload log
  with pre-registered predictions and results) are part of the development history, which is not published; the
  numbers taken from them are given here. The published sources are `final_submission/realpde/docs/METHOD.md`, the
  long method description, and `docs/writeup/final_long.tex`, the 7-page write-up (its section "Validation evidence
  for each component" has the component evidence table).

## Abbreviations

| short name | meaning |
|---|---|
| S3 | training loss: Welsch loss mixed 75/25 between physical and standardised space |
| SP | sim pretraining |
| SOUP4 | uniform weight average of four fine-tunes |
| ENRx, K3 | packed ensemble of soups from different sim-pretrain seeds (K = 2, K = 3) |
| FP16X | fp16 core of the packed ensemble |
| K3F | K = 3 ensemble with the fp16 core |
| ctx-norm | causal context normalisation |
| D2 | bundle of the energy-target blend `e_blend_w` and the variance-preserving mean `mu_vp` |
| ke_net, KE3 | energy nets: the original net, and the 4-seed net written as one dense net |
| GM20 | energy-gain clamp raised to 20 (`gain_max`) |
| M1, W2 | conditional interval-width nets, first version and all-81 cross-fitted refit |
| CB | centre blend of the interval toward the raw ensemble output |
| AW | adaptive (per-trajectory) CB weight, `cb_adapt` |
| AX | adaptive width multiplier κ (`kw_adapt`) and adaptive energy blend (`eb_adapt`) |
| AXE (E4) | AX extension: wider energy-blend grid, adaptive blend ridge (`br_adapt`), window-0 CB weight 1.0 |
| AXG | AX stack with GM20 (an intermediate archive) |
| MSK | exact scored mask (`exact_mask`) |
| REC | regime-change detector and partial state reset |
| LOWKE, TKA | low-energy fluctuation damping (`tka_lowke`) |
| LSK | online least-squares correction of the ensemble head (`lsk_head`) |
| LSKR | reset of the LSK state when REC fires (`lsk_rec_reset`) |
| R97, X5R | LSK forgetting factor ρ = 0.97 instead of 0.9 |
| TFIX | batched convection-speed scan (2026-09-08 time fix) |
| CG3, CG9 | three and nine separately captured CUDA graphs |
| T1 | whole-step CUDA graph |
| T2 | GPU kernel cuts in the whole-step body |
| W01 | CUDA graphs for windows 0 and 1 |
| KD | first kernel diet (with the scan-cadence change and B2) |
| B2 | conditional-width net run in fp16 |
| KD2 … KD8 | later kernel diets, multi-stream schedule (KD6) and fused kernels (KD7, KD8) |
| IPC, IPW, CDIO, BVR | transport levers: one shared output storage, multiprocessing warm-up, direct CUDA I/O, by-value outputs |
| UV | u,v-only by-value payload (not uploaded) |
| FZ-E, FZ-F | the two candidate freeze lineages of 2026-09-24/25, with and without AXE |
| FZ-EIKC | FZ-E + IPW + KE3 + CDIO |
| LCO | leave-conditions-out validation folds |
| sps | the competition's interval score (Safe Prediction Score) |

## Overview

| component | group | affects | side | `CONFIG` key (off value) or chain flag |
|---|---|---|---|---|
| b5 SimVP-gSTA backbone | backbone | accuracy, time | training | chain `BACKBONE` |
| S3 Welsch loss | backbone | accuracy | training | none (training recipe) |
| causal context normalisation | backbone / TTA | accuracy | training + inference | none (part of the architecture) |
| sim pretraining | backbone | accuracy | training | chain `BACKBONE=b5sp*` |
| soup of 4 | backbone | accuracy | training | chain `SOUP_K=4` |
| K=3 packed ensemble | backbone | accuracy, time | training | chain `BACKBONE=b5spx3` |
| fp16 core | backbone | time | inference | `fp16_core` (False) |
| condition prior | TTA | accuracy, sps | training (statistics) + inference | `prior` (False) |
| gain map / energy calibration | TTA | accuracy | inference | `gain` (False) |
| energy net ke_net_flow / KE3 | TTA | accuracy | training (fit) | `ke_flow`; chain `KE3=1` |
| advection blend | TTA | accuracy | inference | `advect` (False) |
| D2 bundle | TTA | accuracy | inference (+ S3 checkpoint) | `e_blend_w` (None), `mu_vp` (False) |
| gain clamp 20 (GM20) | TTA | accuracy | inference | `gain_max` (5.0) |
| adaptive energy blend (AX part B) | TTA | accuracy | inference | `eb_adapt` (None) |
| AXE adaptive blend ridge | TTA | accuracy | inference | `br_adapt` (None) |
| REC | TTA | robustness | inference | `rec_detect` (None) |
| LOWKE / TKA | TTA | accuracy | inference | `tka_lowke` (None) |
| LSK | TTA | accuracy | inference | `lsk_head` (None) |
| LSK REC reset (LSKR) | TTA | robustness | inference | `lsk_rec_reset` (False) |
| LSK ρ 0.97 (R97) | TTA | accuracy | inference | `lsk_head["rho"]` (0.9) |
| decoupled interval centre | intervals | sps | inference | `ctr` (False), `ctr_basis2` |
| conditional widths M1 → W2 | intervals | sps | training (fit) + inference | `cond_width` (False) |
| centre blend CB | intervals | sps | inference | `cb_blend_w` (None) |
| adaptive CB weight AW | intervals | sps | inference | `cb_adapt` (None) |
| adaptive width κ (AX part A) | intervals | sps | inference | `kw_adapt` (None) |
| AXE window-0 CB weight | intervals | sps | inference | `cb_blend_w0` (None) |
| exact scored mask MSK | intervals | sps | inference | `exact_mask` (False) |
| CUDA graphs, whole-step graph, W01 | speed | time | inference | none (built at load, eager fallback) |
| convection-scan fixes (TFIX, cadence) | speed | time | inference | `uc_warm`, `uc_every` |
| T2, KD, B2 | speed | time | inference | none (code paths) |
| KD2, KD3, KD4, KD5, KD6, KD7, KD8 | speed | time | inference | `kd2_adapt`, `kd3_step`, `kd4_w01`, `kd5_near`, `kd6_concurrency`, `kd7`, `kd8` (all False) |
| IPC, IPW, CDIO, BVR | speed | time | inference | `ipc_one_storage`, `ipc_prewarm`, `cuda_direct_io`, `ipc_byvalue` (all False) |

## 1. Backbone and training

### b5 SimVP-gSTA backbone

`b5_h192_blk6_rf11` is a SimVP-style convolutional network with gated spatiotemporal attention (gSTA) blocks:
encoder width 128, hidden width 192, six gSTA blocks, receptive field 11, 3,390,460 parameters. It maps the 20-frame
input window to all 20 output frames in one pass. It replaced the larger `n1_simvp` SimVP variant (7,111,868
parameters), which had itself replaced a ConvNeXt-UNet; before 2026-09-18 every upload used the organizers' FNO
checkpoint. Training side (the chain's `BACKBONE` flag selects the architecture). It affects accuracy and, through its
size, time.

Measured: `n1_simvp` vs the UNet with the same context normalisation and Welsch recipe, pipeline composite fold0 /
fold1 / foldR 81.826 / 82.261 / 82.279 vs 81.498 / 82.015 / 82.099 (+0.33 / +0.25 / +0.18;
development ledger, candidate `2026-09-25_simvp-ctxnorm-welsch-graphed4`). The smaller b5 vs `n1_simvp`, with
the W2 widths on: fold0 +0.035, fold1 +0.052, foldR +0.020 (development notes). On the board, the SimVP switch arrived on 2026-09-20 together with the S3
loss, the D2 bundle and the M1 widths (+0.277 final overall); b5 arrived on 2026-09-23 together with sim pretraining
and the W01/KD time stack (+0.171 overall), so neither is isolated on the board.

### S3 Welsch loss

The training loss is a Welsch (correntropy) loss, a bounded robust loss that limits the influence of large residuals.
"S3" mixes it 75/25 between the physical prediction space and the per-channel standardised space; the two
normalisers are fixed from the first batch of each training stage. Training side, accuracy only. Introduced in two
steps: a single-space Welsch loss first (board 2026-09-19), then the S3 mix (board 2026-09-20).

Measured: single-space Welsch vs MSE on the ctx-norm UNet, pipeline composite fold0 / fold1 / foldR 81.498 / 82.015 /
82.10 vs 80.92 / 81.25 / 81.44 (+0.58 / +0.77 / +0.66; wins on 56/56 held-out files). The
2026-09-19 upload combined this loss with three CUDA graphs: rel_l2 +0.263, tke +0.157, sps +1.291, final +0.962, of
which most of the final change came from time. S3 vs single-space Welsch: fold0 +0.070 on the UNet
(development notes) and +0.027 on `n1_simvp` (development notes, D2 bundle,
rung a).

### Causal context normalisation (ctx-norm)

For each trajectory a wrapper keeps a per-pixel running mean and standard deviation over every frame revealed so far
(inputs and previously revealed targets, never the current target). The two maps are fed to the network as extra
input channels (mean and log-std), and the network output is de-standardised with them. This adapts the model to the
trajectory's own operating point without any gradient step. It is part of the architecture, so it is trained in
(training side) and runs at every inference step; there is no `CONFIG` switch. Introduced together with turning the
earlier online mean correction off (`mean_mode` None instead of "both"), which won on 56/56 held-out files.

Measured: pipeline composite fold0 80.92 vs 80.24 for the same UNet without ctx-norm (+0.68), foldR 81.44 vs 81.16
(+0.28) (development notes). The long method
document calls it the largest single accuracy contribution of the pipeline's development
(`final_submission/realpde/docs/METHOD.md` §4). On the board it arrived with the switch from the organizers' FNO to
our own UNet on 2026-09-18: rel_l2 +0.47, tke +1.01, sps +2.06, but time −1.61 (the UNet was launch-bound on the
eval machine), final +0.73.

### Sim pretraining (SP)

Each backbone is first trained on all 100 `train_sim` files (subsampled to the real 32×64 grid, rescaled to physical
units with U∞ = 1.263·10⁻⁵·Re, pressure masked to zero so the input layout matches real data), AdamW lr 5e-4, a
20k-iteration cosine schedule stopped at 10k. Real fine-tuning (8k iterations on the 81 usable real files) starts from
that snapshot. Training side, accuracy only. The 10k length was chosen on fold1 from {5k, 10k, 20k} by a pre-declared
rule, scored on the backbone's own output without the online pipeline (composite gain over the random-init backbone
+0.128 / +0.155 / +0.148).

Measured, full pipeline vs the matched random-init b5: fold1 +0.105, fold0 +0.184, foldR +0.064, Rx +0.177,
foldS-lo +0.186, foldS-hi −0.181 (write-up evidence table). The foldS-hi loss
is the one systematic negative, in a regime beyond the hidden set's stated range. Board: part of the 2026-09-23 upload
(+0.171 together with b5 and the W01/KD time stack).

### Soup of 4 (SOUP4)

Four real fine-tunes (seeds 0–3) from the same sim-pretrained snapshot are averaged uniformly in weight space (in
fp64, cast to fp32). This only works because the shared pretrained initialisation puts the four in one loss basin: a
soup of random-init fine-tunes lost 5.17 composite points, and souping across different pretrain seeds loses 3.5–4.5
composite points on the backbone's own output (before the online pipeline). Training side, accuracy only; zero inference cost (one set of weights).

Measured vs the single-seed sim-pretrained b5: fold1 +0.022, fold0 +0.075 (0/20 files worse), foldR +0.048, Rx +0.041,
foldS-lo +0.024, foldS-hi −0.001; K=8 added only +0.010 on fold1 (`final_submission/realpde/docs/METHOD.md` §3.3). Board: part of the 2026-09-24 upload, not isolated (the soup-only
archive `2b7dfe3c` was never uploaded).

### K=3 packed ensemble (ENRx / K3)

Steps sim pretrain → four fine-tunes → soup are repeated with three pretrain seeds (0, 1, 2). The three soups are
packed into one batched network (stem concatenation, grouped convolutions, batched 1×1 convolutions; 10,166,380
parameters) whose output is the members' mean in standardised space. Diversity has to come from the pretrain seed:
two soups from the same pretrain gain only +0.004–0.005. Training side. It improves accuracy and costs time (the
third member was about break-even in TF32 until the fp16 core below).

Measured: second member, fold1 +0.080, fold0 +0.091, and on an independent build on a second machine foldR +0.082,
Rx +0.060, foldS-lo +0.073, foldS-hi +0.084; third member (measured together with the fp16 core) fold1 +0.048, fold0
+0.039, foldS-lo +0.026, foldS-hi −0.015 (write-up evidence table). Cost of the second member +0.25 ms/step on a 3090
(development notes). Board: part of the 2026-09-24 upload (+0.309 final for ten components
together).

### fp16 core (FP16X)

Everything inside the packed backbone (stem, downsampling, all gSTA blocks, up/fuse, head) runs in half precision;
context normalisation and the whole pipeline around it stay fp32. Inference side, `CONFIG["fp16_core"] = True`
(off: `False` = TF32 batched-matmul packing). It is a speed component: the composite changes by |Δ| ≤ 0.0002, and the
worst observed headroom to fp16 overflow was 1878×. Alone it missed its own −0.2 ms gate at K=2 (−0.185 ms); together
with the third member it made K=3 0.054–0.06 ms faster than the K=2 TF32 parent, which is why the two shipped
together ("K3F") (development notes).

## 2. Test-time adaptation (accuracy)

None of these components updates network weights at test time. They are causal running statistics, closed-form fits
and per-trajectory selectors computed from already revealed windows.

### Gain map, condition prior and advection blend (the base pipeline)

These date from the first week and still form the core of the online pipeline. The gain map (`gain`, power 0.75)
rescales the backbone's predicted fluctuation per pixel so that its energy matches a running energy target; a running
mean innovation is added as a bias (`bias`). The condition prior (`prior`, `prior.npz`) initialises the running
statistics from a kernel-weighted average of per-training-file statistic maps, with Re and AoA estimated from the
first input window (Re from the free-stream speed, AoA from the geometry of the body mask); it carries 6 pseudo-windows
of decaying weight. The advection blend (`advect`) mixes in a Taylor-advected copy of the last revealed frame, using a
convection speed estimated by a correlation scan. The prior is a training-side fit of statistics (no gradient
training); the rest is inference side. All affect accuracy and the interval statistics.

Measured: the gain-map pipeline on the organizers' FNO was the first recorded board row (2026-09-04, final 79.06),
replacing an EnKF wrapper with the same accuracy at one forward pass instead of 16 and +1.6 sps on the test file
(development notes). On the then-memorised local files the prior was worth +0.5–0.75 sps and the
advected shape +0.03–0.15 sps (development notes). These early numbers were measured on a backbone
that had seen the local files, so they are not held-out estimates.

### Energy net: ke_net, ke_net_flow and KE3

A small CNN predicts the per-pixel, per-channel log fluctuation energy of the coming window from causal features, as a
zero-initialised residual on the gain-map rule; the pipeline then rescales the fluctuation to that energy. `ke_net`
(2026-09-07) used energy summaries as features; `ke_net_flow` (2026-09-16, `CONFIG["ke_flow"] = "ke_net_flow.pt"`)
uses the flow fields themselves. KE3 fits four seeds and writes their mean exactly as one dense network of 4× width
(block-diagonal weights, fp64 check ≤ 5·10⁻⁹), so the package loads it through the unchanged single-net path.
Training side: the net is refit for each backbone; KE3 is the only chain flag that changes a fit (`KE3=1`, chain
step 08). Accuracy (rel_l2, tke and, through the accuracy factor, sps).

Measured: `ke_net` vs the gain map on the board (2026-09-07 vs 2026-09-04): tke +2.0, sps +2.3, rel_l2 −0.10,
matching the local deltas (development notes). KE3 vs a single net on the K=3 stack: fold1 +0.018,
fold0 +0.043, foldR +0.020, foldS-lo +0.057, foldS-hi +0.115; it also halves the energy-fit share of retrain variance;
cost +10 kernels, +0.06 ms (4090), +0.10 ms (3090), +0.11 ms (A100) (development notes). Board: part of the 2026-09-26 upload, not isolated.

### D2 bundle: energy-target blend and variance-preserving mean

Two small changes adopted together with the S3 loss. `e_blend_w = 0.85` sets the energy target to 0.85 × the energy
net's prediction + 0.15 × a causal running mean of the revealed energy (`R_run`). `mu_vp = True` adds the running mean
correction in a variance-preserving way (the fluctuation is rescaled so its temporal variance is unchanged). Inference
side; off values `e_blend_w` None and `mu_vp` False. Accuracy only. A third rung (a wider window-0 interval) was
dropped at the gate (+0.0028 < +0.005).

Measured on fold0 (development notes): S3 +0.027, energy blend +0.0065, `mu_vp` +0.0094;
bundle fold0 / fold1 / foldR 81.868 / 82.353 / 82.365 vs 81.826 / 82.261 / 82.279. Board: part of the 2026-09-20
upload.

### Gain clamp 20 (GM20)

The energy gain applied to the fluctuation is clamped to [1/`gain_max`, `gain_max`]. Raising `gain_max` from 5 to 20
(`CONFIG["gain_max"] = 20.0`, off value 5.0) came from a grid {8, 12, 20} that increased monotonically; an adaptive
per-trajectory clamp was dominated by the fixed value. Inference side, accuracy only, zero kernels.

Measured vs the AX stack: fold1 +0.006, fold0 +0.008, foldR +0.031, Rx +0.019, foldS-lo −0.004, foldS-hi +0.000
(development notes). Board: part of 2026-09-24.

### Adaptive energy blend (AX part B, `eb_adapt`) and the AXE extension (`br_adapt`)

These are per-trajectory selectors. Every step, all candidate values are evaluated in one batched pass; each candidate
is scored on the previous window with the official composite terms (rel_l2, tke, mvpe) computed from the revealed
target; an exponential moving average (ρ = 0.95) of the scores is kept, and the best candidate's output is emitted.
`eb_adapt` chooses the energy-blend weight from {0.45, 0.60, 0.75, 0.85, 0.925} (the AX grid was {0.75, 0.85, 0.925};
AXE widened it). `br_adapt` (AXE) chooses the ridge strength of the advection-blend least-squares fit from
{0.125, 0.25, 0.5, 1.0}. Both change `pred`. Graph-only: the eager fallback keeps the fixed values. Inference side; off
values `eb_adapt` None, `br_adapt` None. Accuracy, mainly as a hedge for regimes outside the training range.

Measured: AX as a whole (this part together with the width multiplier κ below) vs AW: fold1 +0.017, fold0 +0.027,
foldR +0.011, foldS-lo +0.024, foldS-hi +0.046, with κ driving most of the in-range gain and the energy blend most of
the beyond-range gain (development notes). AXE (including the window-0 CB weight below) vs the
same K=3 stack without it: fold1 +0.0002, fold0 +0.009, foldS-lo +0.009, foldS-hi +0.235; ≥ no-AXE on 20/21 regimes of
an unseen-regime stress matrix (development notes).
Cost: AX +60 kernels, +0.155 ms vs AW; AXE about +0.1 ms before later kernel diets. Board: AX in 2026-09-24; AXE in
2026-09-25, where the final changed by +0.004 together with REC, KD5, KD4, IPC and a weight retrain on another GPU.

### Regime-change recovery (REC)

Context normalisation accumulates over the whole stream, so after an abrupt change of free-stream speed inside one
trajectory the pipeline did not recover (a stress-test finding). REC computes d = |log(Ū_window / Ū_running)| from the
mean u of the current input frames; if d > log 1.5 it restarts the context accumulator and the error, centre and blend
moments (pseudo-count 6) and rescales mean-type statistics by the speed ratio. Inference side,
`CONFIG["rec_detect"] = {"thr": 0.405465, "n0": 6.0, "rescale": True, "u_eps": 1e-4}`, off value None. It is a
robustness component: on all 81 real files d never exceeds 0.085, so REC is bit-identical to its absence on every
normal stream.

Measured: 0 fires on fold0/fold1/foldR; on synthetic regime-switch concatenations it fires at the switch and recovers
within 0–2 windows (+17…+22 composite on the affected file); +11 kernels, about +0.01 ms
(development notes). No board effect is expected or isolated (2026-09-25 upload).

### Low-energy fluctuation damping (LOWKE, also called TKA)

In the steady whole-step graph (windows ≥ 2) the emitted prediction's temporal fluctuation is shrunk where the
prediction's own kinetic-energy map K_p is low: g = clamp((K_p / (1.5·mean K_p))^0.25, 0, 1),
pred ← m_t + g·(pred − m_t), with m_t the 20-frame mean. Bounds and all internal state are unchanged. It removes weakly
correlated low-energy fluctuation: rel_l2 and sps rise, tke falls slightly. Inference side,
`CONFIG["tka_lowke"] = {"theta": 1.5, "delta": 0.25, "gmin": 0.0}`, off value None. Accuracy.

Measured (net of its time cost): fold1 +0.022, fold0 +0.033, foldR +0.021, foldS-lo +0.037, foldS-hi +0.035, 64/64
files positive (development notes). The parameters were
chosen on fold1 from an 84-point grid at its edge, and the fold1 gain missed that study's own pre-declared +0.03 build
threshold, so it is labelled post hoc. Board: part of the 2026-09-26 upload (rel_l2 +0.046 and tke −0.022 vs the
previous row, in the predicted direction).

### Online least-squares head correction (LSK)

A per-trajectory recursive ridge regression corrects the packed backbone's raw output before every downstream stage.
At load time the top-8 right singular vectors V of the stacked member head weights are computed (no fit, no extra
weights). Each step the features φ = [Vᵀh; 1] (one fp16 GEMM hooked into the head) and the causal context std s are
stored; when the previous window's truth is revealed, fp64 moments Φ ← ρΦ + Σ s²φφᵀ and B ← ρB + e(sφ)ᵀ are updated
on a side stream and M = B(Φ + λI)⁻¹ is solved without host synchronisation; the correction (Mφ)·s is added to all 20
leads. Window 0 is uncorrected. Inference side, `CONFIG["lsk_head"] = {"r": 8, "rho": 0.97, "lam": 30.0}`, off value
None. Accuracy.

Measured vs the same stack without LSK, config B (λ_rel 30, ρ 0.9): fold1 +0.018, fold0 +0.041, foldR +0.027, Rx +0.034
(write-up evidence table). Config A (λ 10) was the fold1 pick; the independent review report FR33
(development notes, written by a separate agent) recommended B after evaluating it once on five further held-out
sets (fold0, foldR, Rx, a high-Re set, a regime-switch set) because it removed most of A's per-file losses on AoA-0, highest-Re files. Known weak spot:
those files still lose −0.010…−0.016 each. Cost +53 kernels (mostly on a side stream), +0.034…+0.046 ms (3090).
Board: part of the 2026-09-27 upload (+0.033 final with LSKR, R97 and KD8).

### LSK reset on REC (LSKR)

When REC fires, the LSK moments, λ, the count and the applied M are multiplied by REC's keep factor (0 on a reset step,
exactly 1.0 otherwise), so the correction restarts in the new regime. Inference side,
`CONFIG["lsk_rec_reset"] = True`, off value False. Robustness only: bit-identical wherever REC does not fire. Measured
+0.036 on a 21-file synthetic regime-switch set, identical on all 9 non-switch sets; +2 kernels, +0.015 ms (3090)
(development notes). Board: 2026-09-27 upload; no effect expected.

### LSK forgetting factor 0.97 (R97)

`lsk_head["rho"]` 0.9 → 0.97 (a tensor constant, 0 ms). A pre-declared fold1 search over eight LSK-family variants
found only ρ = 0.97 beating ρ = 0.9 by the +0.003 selection margin (+0.0044). Held out once (with LSKR): fold0
+0.0099, foldR +0.0046, foldS-lo +0.0075, foldS-hi +0.033, Rx +0.007, high-Re surrogates +0.003…+0.024, switch set
+0.045. This was below the study's pre-declared fold0 bar of +0.015; the gate was overridden, and the override
disclosed, because every held-out set and every surrogate set for the private-test regimes was non-negative. An independent refit on a second machine
reproduced it on 5/5 sets (fold0 +0.0086, fold1 +0.0038, foldR +0.0032, foldS-lo +0.0069, foldS-hi +0.0316)
(development notes). Inference
side, accuracy. Board: 2026-09-27 upload, not isolated.

## 3. Uncertainty intervals (SPS)

All of these change only `lower` / `upper`. `pred`, and therefore rel_l2 / tke / mvpe, are bit-identical with each of
them on or off. They raise the SPS subscore, which multiplies an interval term with an accuracy factor.

### Decoupled interval centre (ctr, ctr_basis2)

The emitted prediction is deliberately energy-inflated for TKE, which makes its errors larger than they need to be for
interval purposes. The pipeline therefore carries a second, un-inflated centre with its own running bias and
innovation variance, and centres the interval there. `ctr_basis2` uses a two-vector basis (blend shape and advected
fluctuation) with a 2×2 per-pixel least-squares gain. Inference side, `CONFIG["ctr"] = True` (off False),
`ctr_basis2 = True`. Measured +1.5 to +1.9 sps with the other subscores bit-identical, on the memorised local files
(development notes); board as part of the 2026-09-07 `ke_net` upload. A later kernel
audit found the second basis vector slightly negative in range (+0.008…+0.011 when removed) but worth +0.056 on
foldS-hi, so it was kept (development notes).

### Conditional interval widths: M1, then W2

A small CNN predicts the log-ratio between the local error scale and the pipeline's own running error scale, from
causal features (input and prediction statistics, lead index, an EMA of the previous window's error map). The
half-width is the running scale × λ·exp(α·net). M1 (2026-09-20) was fitted on held-out streams of three fold
pipelines; W2 (2026-09-22) is refit on out-of-sample residuals from 4-group cross-fitting over all 81 files (each
group's backbone, prior and energy net trained on the other three groups), with more features. Training side (the
`cond_width_weights.pt` fit in the chain) plus inference side `CONFIG["cond_width"] = True` (off False). The W2 net
is fitted on random-init b5 backbones and shared unchanged across the later backbones; the mismatch was measured at
about 0.

Measured: M1, fold0 sps +0.70, fold1 +0.63, foldR +0.47, composite fold0 82.060 vs 81.868 (+0.19)
(development notes). Board 2026-09-20: about +0.64 of that upload's
+0.954 sps is attributed to M1 from the implied interval factor. W2 vs M1, sps fold deltas +0.344 / +0.357 / +0.289
(development notes); board 2026-09-22, the only interval change in that upload: sps
+0.324 (43.615 → 43.939), worth about +0.09 final.

### Centre blend toward the raw ensemble output (CB)

The emitted bounds are shifted toward the raw ensemble output: lower/upper += w·(raw − centre), with w = 0.75 for every
lead, on u and v. The pipeline's blended centre is more accurate on average, but the raw output is a better centre for
scoring the interval. Inference side, `CONFIG["cb_blend_w"] = [0.75] * 20`, off value None. +3 kernels.

Measured on the K=2 ensemble: fold1 +0.110, fold0 +0.089, foldR +0.114; on a single backbone also Rx +0.082, foldS-lo
+0.138, foldS-hi −0.001, 0/67 files worse outside foldS-hi (development notes). Board: part of 2026-09-24 (sps 44.260 → 45.163 for CB, AW, AX, MSK and the
backbone changes together).

### Adaptive centre-blend weight (AW, `cb_adapt`)

Replaces the fixed w = 0.75 by a per-trajectory choice from {0.30, 0.45, 0.60, 0.75, 0.90}: each candidate is scored on
the previous window with the exact SPS element term, an EMA with ρ = 0.95 is kept, and the best is used from window 2
on. Inference side, `CONFIG["cb_adapt"] = {"grid": [...], "rho": 0.95}`, off value None. Measured vs CB: fold1
+0.0009, fold0 +0.0010, foldS-lo +0.0021, foldS-hi +0.0548 (about 89 % of the per-trajectory oracle there). It is
almost neutral in range and was adopted as a beyond-range hedge. +17 kernels, about +0.065 ms
(development notes).

### Adaptive width multiplier (AX part A, `kw_adapt`)

The W2 half-width is multiplied by κ ∈ {0.85, 0.925, 1, 1.075, 1.15}, chosen separately for leads 1–2 and leads 3–20
by the same previous-window SPS score and EMA rule. Graph-only (κ = 1 in the eager fallback and on windows 0/1).
Inference side, `CONFIG["kw_adapt"]`, off value None. Measured as part of AX (see the energy-blend entry above); κ
carries most of AX's in-range gain.

### Window-0 CB weight (AXE part, `cb_blend_w0`)

On window 0, where no truth has been revealed yet, the CB weight is 1.0 (the raw ensemble output as centre) instead of
0.75. Inference side, `CONFIG["cb_blend_w0"] = 1.0`, off value None (= `cb_blend_w`). Measured only as part of AXE.

### Exact scored mask (MSK)

W2 was fitted with the scorer's own mask (target ≠ 0 in physical units), but at inference the same test was applied to
the normalised target, where a physical zero of v becomes a tiny non-zero value; every PIV-dropout zero of v was
therefore counted as scored in W2's error features and in the AW/AX scoring. The fix uses |prev_target| > 1e-6.
Inference side, `CONFIG["exact_mask"] = True`, off value False. Adopted as a fit/inference consistency fix; it does not
meet the gate written for new features. Measured: fold1 +0.003, fold0 +0.005, foldR +0.004, Rx +0.009, foldS-lo
−0.001, foldS-hi +0.005; +1 kernel (development notes). Board: part of 2026-09-24.

## 4. Speed (Time subscore only)

The time subscore is computed from the timed `ttt_step` / `reset_ttt_state` calls. Near the final operating point one
millisecond per step was worth about 0.7 time points, about 0.06 final points. Every speed component below was adopted
only if its outputs are bit-identical to the reference path (KD5 changes fp16 rounding order only, and B2 changes
bounds numerics at |Δ composite| ≤ 0.0001), it self-tests at model construction (untimed), and it falls back to the
parent path on failure. Local timings are from an RTX 3090 unless noted.

Board step time, as recorded or implied by the time subscore: 8.5 ms (2026-09-04, gain map on the FNO) → 56 ms
(2026-09-07, unbatched scan) → 15.3 ms (2026-09-16, FNO pipeline) → 20.2 ms (2026-09-18, eager UNet) → 7.27 ms (2026-09-19, three graphs) → 7.79 ms
(2026-09-20, nine graphs) → 6.90 ms (2026-09-22, whole-step graph) → about 6.40 ms (2026-09-23) → 6.13 ms (2026-09-24) → 6.22 ms
(2026-09-25) → 4.91 ms (2026-09-26) → 4.84 ms (2026-09-27, final). Source: `final_submission/METHOD.md` §7.

### Convection-scan fixes (TFIX and scan cadence)

The convection-speed scan of the advection blend originally ran 82 correlations per step, each ending in a host sync.
TFIX (2026-09-08) made it one batched gather on the device and ran it on the first 4 windows and then every 4th:
board time 78.29 → 87.90 with identical accuracy (development notes). Later the periodic rescan was
found to be worthless on all five folds (|Δ| ≤ 0.0003) and switched off: `uc_warm` 2, `uc_every` 10⁹ (previously 4 / 4),
about −0.27 ms (development notes). Inference side.

### CUDA graphs and the whole-step graph (T1), windows 0/1 (W01)

The step was first split into separately captured CUDA graphs: two (context normalisation + backbone; advection /
coefficient solve / centre), then a third (energy net), then up to nine (pre-stage statistics, a merged "mega-tail",
mask, bounds, scan, conditional widths). Each graph is checked against the eager code at construction and has an eager
fallback. Three graphs took the board step from 20.2 to 7.27 ms (2026-09-19, together with the Welsch weights). Going
to nine graphs cut launches per step from 218 to about 60 but the board time rose to 7.79 ms (2026-09-20), which
disproved the launch-count cost model. The whole-step graph (T1) then made every step from window 2 on one pinned
host-to-device copy, one graph replay, one device-to-host copy and one sync; W01 gives windows 0 and 1 their own
graphs. Neither has a `CONFIG` key: both are built in `get_ttt_model` and fall back to the previous path if capture or
the self-test fails. Board: whole-step + T2 took 7.79 → 6.90 ms (2026-09-22, about a third of what the cost model had
predicted); W01 + KD arrived with the b5 backbone on 2026-09-23 (about −0.5 ms).

### T2, KD and B2

T2 (built 2026-09-21, uploaded 2026-09-22) cut GPU kernels in the whole-step body (1,147 → 975 per step) with bit-exact fusions. KD (uploaded 2026-09-23)
was a further bit-exact kernel diet of the whole-step graph (992 → 760 kernels per step, −0.32 ms; both counts are on
the n1 backbone, but T2's 975 is a per-step census of the v2 pipeline while KD's counts come from whole-step graph
replays, whose census gave 948 kernels on steps without the convection scan and 1,052 on steps with it; the two
measurements are not directly comparable) shipped together
with the scan-cadence change above and B2, which runs the conditional-width net in fp16 (bounds only,
|Δ composite| ≤ 0.0001, −0.20 ms); the three together were estimated at −0.79 ms on the 3090
(development notes). KD also carried a required fix: a capture-time host synchronisation
had been found to corrupt the process's CUDA RNG state, which would have zeroed every subscore; an RNG probe now
checks it after all captures. No `CONFIG` keys.

### KD2 and KD3

KD2 (`kd2_adapt`) fuses the AW / AX / AXE candidate scoring and selection into fewer batched kernels (about −40
kernels, −0.11 ms). KD3 (`kd3_step`) rewrites the whole-step backbone and the merged energy / prior / advection stage ("mega" stage) with fewer kernels (759 → 664,
−0.11 ms). Both bit-exact; off value False. Board: part of 2026-09-24.

### KD4 and KD5

KD4 (`kd4_w01`) captures the window-0/1 graphs with the scan / prior / pre-stage block on a side stream and private
memory pools (bit-exact, −0.05 ms averaged over a stream). KD5 (`kd5_near`) rewrites the stem and downsampling
convolutions as unfold + GEMM, a depthwise convolution as space-to-batch, and folds W2's first bias into a ones
channel. It is exact in real arithmetic but changes fp16 rounding order: |Δ composite| ≤ 0.0004 on three folds,
−0.115 ms. Off value False. Board: part of 2026-09-25, where the time subscore was 91.54 vs 91.60 the day before, i.e.
no measurable change on the board.

### KD6: multi-stream schedule

The steady whole-step graph is captured with fork/join edges so that work independent of the backbone output (error and
loss bookkeeping, running variances, width-net state updates, selector scoring, input-only preparation) runs on side
streams concurrently with the backbone. Same kernels and same arithmetic; a construction-time self-test compares all
reachable storages over three replays against the serial graph, and a transitive happens-before race checker reports
0 unordered conflicts. `CONFIG["kd6_concurrency"] = True`, off False. −0.42 ms (3090), −0.50 ms (A100)
(development notes).

### KD7 and KD8: exact-rounding fused kernels

KD7 fuses the longest elementwise chains left on the critical path into `torch.cuda.jiterator` kernels written with
round-to-nearest intrinsics so NVRTC cannot contract them into FMAs; the transcendental functions used were compared
with PyTorch over all 2³² float inputs. Bit-identical; compiled and tested at construction; any failure (including a
missing NVRTC) falls back to the KD6 body. −0.12 ms (3090), −0.11 ms (A100). KD8 moves more backbone-independent work
(bias pre-loads, the LSK hook and correction, LOWKE) to side streams and adds six more fused kernels: −0.066…−0.074 ms
(3090), −0.094…−0.108 ms (A100). `CONFIG["kd7"]`, `CONFIG["kd8"]`, off False (development notes). Board: KD6 + KD7 are estimated at about −0.6 ms of the 2026-09-26 drop from 6.22
to 4.91 ms (A100 emulation); KD8 is part of 2026-09-27.

### Transport levers: IPC, IPW, CDIO, BVR

The official harness runs the model in a worker subprocess; how inputs and outputs cross the process boundary is not
public. Four levers each target one possible harness design, return identical values and are inert elsewhere.

- IPC (`ipc_one_storage` "auto", off False): when the input arrived through PyTorch's shared-memory multiprocessing,
  pred / lower / upper are returned as three views of one CPU clone. No measurable board effect on 2026-09-25.
- IPW (`ipc_prewarm` True, off False): in a multiprocessing child, starts PyTorch's resource-sharer listener during
  model construction (untimed) instead of in the first timed step.
- CDIO (`cuda_direct_io` True, off False): only when `ttt_step` receives CUDA tensors, copies them device-to-device and
  returns CUDA clones; −0.44…−0.55 ms with CUDA inputs in emulation, 0 otherwise.
- BVR (`ipc_byvalue` True, off False): in a multiprocessing child, CPU outputs are returned as a zero-copy tensor
  subclass that pickles as plain values instead of file-descriptor handles. A100 emulation: −0.64…−0.77 ms per step in
  by-value harnesses, at most about +0.1 ms in others.

Board: on 2026-09-26 the step fell from 6.22 to 4.91 ms; with about 0.6 ms attributed to KD6 + KD7, about 0.7 ms is
attributed to BVR, i.e. the harness moves outputs by value (development notes).

## Components tried on the board but not in the final

- Asynchronous return (v8, 2026-09-21): returning unsynchronised CUDA tensors, plus shared static I/O buffers. Outputs
  identical to the previous upload; board time was 0.6 ms slower (8.39 vs 7.79 ms), which showed that the official
  timer includes the GPU work. Not kept.
- u,v-only by-value payload (UV): built for the last day, but slower on an A100 in an A/B test (+0.08…+0.17 ms), so it
  was not uploaded (development notes).
- The organizers' FNO checkpoint, gradient test-time training (kit template), EnKF / ES-MDA state and parameter
  updates, the original `ke_net`, and the online mean correction (`mean_mode` "both") were all used in early uploads
  or experiments and later replaced; see `docs/history.md`.
