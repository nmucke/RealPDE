# Codabench submissions

This page lists every Codabench upload of team nmucke to RealPDE Track 2 (competition 17385) during the Development
Phase, and the Final Decision Phase package. The numbers are the same as in `experiments/leaderboard.csv`. The primary
record is the development ledger, which is not published, together with a `manifest.md` for each package. The
package sources of the last three uploads (2026-09-25 to 2026-09-27) are in `final_submission/realpde/submissions/`;
those of earlier uploads are not published and are named here by their directory name only.
Short names (CB, LSK, KD6, …) are expanded in the abbreviation table of `docs/components.md`, which also gives the
fold-level evidence for each component.

Notes on reading it:

- Directory date prefixes are sequence labels, not upload dates (see the conventions in `docs/components.md`).
- The board used Force_Best: a team's best upload counts. "Δ vs best" is the change against the best score on the
  board before the upload; an upload below that best was not kept.
- Codabench ids are given where the repository records them. The archive sha256 prefix identifies the uploaded zip.
- Preset ids refer to `experiments/presets/`; each preset switches components of the final package on or off to
  approximate an upload of the final lineage. Uploads before 2026-09-20 have no preset.
- The time subscore depends on the evaluation machine and its load; step times in ms are converted from it with
  t = 0.72896 s · (100/time − 1)².

## Summary

| date | what changed | final | Δ vs previous | Δ vs best | rank |
|---|---|---|---|---|---|
| 09-03 | starting-kit template (gradient TTT around the organizers' CNO) | not recorded | | | |
| 09-04 | EnKF around the organizers' FNO | not recorded | | | |
| 09-04 | gain map on the frozen FNO | 79.06 | | | |
| 09-07 | learned energy net, separate interval centre, advection, prior | 78.63 | −0.43 | −0.43, not kept | |
| 09-08 | time fix: batched convection-speed scan | 79.574 | +0.944 | +0.514 | 40/84 |
| 09-16 | online mean correction, two-vector centre, flow-field energy net | 79.73 | +0.156 | +0.156 | |
| 09-18 | own UNet backbone with context normalisation | 80.462 | +0.732 | +0.732 | about 23/48 |
| 09-19 | Welsch loss, three CUDA graphs | 81.424 | +0.962 | +0.962 | 14/94 |
| 09-20 | SimVP backbone, S3 loss, D2 bundle, M1 widths, nine graphs | 81.701 | +0.277 | +0.277 | 7/50 |
| 09-21 | asynchronous output return (time only) | 81.670 | −0.031 | −0.031, not kept | |
| 09-22 | whole-step CUDA graph, T2, W2 widths | 81.828 | +0.158 | +0.127 | 7 |
| 09-23 | b5 backbone with sim pretraining, W01, KD | 81.999 | +0.171 | +0.171 | 3 |
| 09-24 | soups, K=3 ensemble, fp16 core, CB, AW, AX, gain_max 20, MSK, KD2, KD3 | 82.308 | +0.309 | +0.309 | 1 |
| 09-25 | AXE, REC, KD5, KD4, IPC; weights retrained on a second machine | 82.311 | +0.004 | +0.004 | 2 |
| 09-26 | KE3, LOWKE, KD6, KD7, IPW, CDIO, BVR | 82.419 | +0.107 | +0.107 | 2 |
| 09-27 | LSK with ρ 0.97 and its REC reset, KD8 | 82.452 | +0.033 | +0.033 | 1 |
| Final Decision Phase | the 09-27 method retrained by the organizers, private test | | | | Track 2 winner |

Ranks are as recorded at the time of each upload; the field moved between uploads, so a rank can fall while the score
rises.

## Uploads

### 2026-09-03: first_test

The starting kit's gradient test-time-training template around the organizers' CNO checkpoint, as a plumbing check.
No leaderboard result is recorded (the manifest's table is empty). Package `2026-09-03_first_test`.

### 2026-09-04: enkf-fno

An ensemble Kalman filter (16 members) around the organizers' frozen FNO checkpoint, with a model-error covariance
learned from innovations and closed-form interval bounds. No leaderboard result is recorded. Package
`2026-09-04_enkf-fno`.

### 2026-09-04: gain-map-fno (79.06)

Archive `f8d0bcab`, package `2026-09-04_gain-map-fno`. The frozen FNO, a per-pixel gain that
rescales the predicted fluctuation to a running energy target, a running bias, and closed-form bounds; one forward
pass per step instead of the EnKF's 16. Scores: rel_l2 94.01, tke 76.89, mvpe 93.30, time 90.24, sps 36.31, final
79.06. First recorded result.

### 2026-09-07: ke_net (78.63, not kept)

Package `2026-09-07_ke_net`. Added a learned energy net (`ke_net`), a separate un-inflated
interval centre, a Taylor-advected copy of the last frame, and an (Re, AoA)-conditioned prior for the running
statistics. Scores: 93.91 / 78.88 / 93.25 / 78.29 / 38.65, final 78.63, −0.43 against 79.06. Against the 09-04 row, tke rose by 2.0
and sps by 2.3, rel_l2 fell by 0.10 and mvpe by 0.05 (the same directions as measured locally), and time fell from 90.24 to 78.29 (about 56 ms per step) because the
convection-speed scan made 82 host synchronisations per step. Not kept. The zip uploaded that day is not identified
(the manifest's sha refers to a later repack).

### 2026-09-08: ke_net with the time fix (79.574)

Codabench 918557. Same weights; the scan became one batched GPU gather, run on the first 4 windows and then every 4th.
Scores: 93.914 / 78.881 / 93.253 / 87.903 / 38.650, final 79.574: +0.944 against the day before, +0.514 against the
best. A time-only change; the accuracy subscores are those of 09-07. Rank 40 of 84.

### 2026-09-16: integrated (79.73)

Archive `cce71d80`, package `2026-09-16_integrated`. An online correction of the window mean,
a two-vector basis for the interval centre, and the flow-field energy net (`ke_net_flow`) in place of `ke_net`.
Scores: 93.94 / 78.76 / 93.75 / 87.33 / 39.31, final 79.73 (+0.156): mvpe +0.50 and sps +0.66; tke −0.12 and time −0.57. Last upload
on the organizers' FNO checkpoint. Two other packages built the same day (`2026-09-16_honest-backbone`,
`2026-09-16_knobs`) have no recorded upload.

### 2026-09-18: unet-ctxnorm-fast (80.462)

Archive `17195bf6`, package `2026-09-20_unet-ctxnorm-fast`. Our own ConvNeXt-UNet backbone,
trained on all 81 usable real files, with causal context normalisation (running per-pixel mean and std of the
revealed frames as input channels and for de-standardising the output); the online mean correction switched off;
three output-identical speed changes. Scores: 94.409 / 79.774 / 94.294 / 85.720 / 41.370, final 80.462 (+0.732).
rel_l2 +0.47, tke +1.01, mvpe +0.54 and sps +2.06 (about 0.6 times the fold0 deltas); time −1.61, from 15.3 to
20.2 ms per step, because the UNet launches many small kernels and the evaluation machine is launch-bound. Rank about
23rd of 48 September teams.

### 2026-09-19: unet-ctxnorm-welsch-graphed2 (81.424)

Codabench 933378, archive `7bb2dfec`, package `2026-09-22_unet-ctxnorm-welsch-graphed2`. The UNet
retrained with a Welsch (bounded robust) loss, and the step split into three CUDA graphs. Scores: 94.672 / 79.931 /
94.273 / 90.922 / 42.661, final 81.424 (+0.962). Time +5.20 (20.2 to 7.27 ms per step) gave most of the gain; rel_l2
+0.26, tke +0.16, sps +1.29, mvpe −0.02. Rank 14 of 94.

### 2026-09-20: v2, SimVP with M1 widths (81.701)

Archive `b7c6004b`, package `2026-09-29_simvp-s3-bundle-condwidth-graphed` (the directory of
the same name under `final_submission/realpde/submissions/` holds only `cond_width_net.py`, which the training
chain's `scripts/w2_convert_ckpt.py` imports), preset `lb0920_n1_m1`. The backbone became a SimVP-style network with gSTA
blocks (`n1_simvp`), trained with the S3 loss (Welsch mixed 75/25 between physical and standardised space); the D2
bundle (energy target blended 0.85/0.15 with the running revealed energy, variance-preserving mean); the first
conditional interval-width net (M1); nine CUDA graphs. Scores: 94.704 / 80.398 / 94.376 / 90.630 / 43.615, final
81.701 (+0.277). sps +0.95, of which about +0.64 is attributed to M1; tke +0.47; rel_l2 only +0.03 (about 0.2 times
the local estimate); time −0.29 (7.79 ms per step) although launches per step fell from 218 to about 60. Rank 7 of 50.

### 2026-09-21: v8, asynchronous return (81.670, not kept)

Archive `f404f679`, package `2026-10-02_simvp-s3-bundle-condwidth-async-v8`; preset
`lb0920_n1_m1` stands for it (same weights and predictions). Outputs returned as unsynchronised CUDA tensors, static
buffers shared. Scores: 94.704108 / 80.397982 / 94.375714 / 90.311719 / 43.614643, final 81.670 (−0.031). A time-only
change: accuracy and sps were identical to 09-20 to every digit, and time was 0.6 ms slower (8.39 ms). This showed
that the official timer includes GPU work and that run-to-run noise on the board sits in the time subscore. Not kept.

### 2026-09-22: whole-step graph and W2 widths (81.828)

Codabench 939485, archive `3919f0a7`, package `2026-10-03_simvp-s3-bundle-condwidth-v2-wholestep-t2`,
preset `lb0922_n1_w2`. Built on the second machine (RTX 4090). Each step from window 2 on became one CUDA-graph replay
(T1), with GPU kernel cuts (T2, 1,147 to 975 kernels), and the width net was refitted on cross-fitted out-of-sample
residuals over all 81 files (W2). Predictions identical to 09-20. Scores: 94.704108 / 80.397982 / 94.375714 /
91.130797 / 43.938907, final 81.828: +0.158 against 09-21, +0.127 against the best (81.701). sps +0.324 from W2
alone; time +0.50 (6.90 ms per step). Rank 7.

### 2026-09-23: SP-b5 (81.999)

Archive `74f9d98f`, package `2026-10-08_simvp-b5sp-s3-bundle-condwidth-v2-wholestep-t2-w01-kd`,
preset `lb0923_b5sp`. A smaller SimVP variant (b5, 3.39 M parameters) pretrained on the 100 simulation files and
fine-tuned on the 81 real files; CUDA graphs for windows 0 and 1 (W01); a kernel diet (KD) with the periodic
convection rescan switched off and the width net in fp16. Scores: 94.778040 / 80.716160 / 94.336535 / 91.431588 /
44.260186, final 81.999 (+0.171). rel_l2 +0.074, tke +0.318, sps +0.321, time +0.301 (about 6.40 ms per step), mvpe
−0.040. The b5-only and time-only archives were never uploaded, so the gain is not split between backbone, sim
pretraining and speed. Rank 3.

### 2026-09-24: K3F-KD3 (82.308)

Archive `f39128a9`, package `2026-10-17_simvp-b5spx3-soup4-fp16-cbadapt-ax-gm20-msk-kd3-s3-bundle-condwidth-v2-wholestep-t2-w01-kd`,
preset `lb0924_k3f_kd3`. The backbone became a packed ensemble of three members, each a weight average of four
fine-tunes from its own sim-pretrain seed, with the packed core in fp16. The intervals gained the centre blend toward
the raw ensemble output (CB) with a per-trajectory adaptive weight (AW), a per-trajectory width multiplier and energy
blend (AX), and the exact scored mask (MSK); the energy-gain clamp was raised to 20; two kernel diets (KD2, KD3)
offset the added cost. Scores: 94.850321 / 81.050726 / 94.391939 / 91.603345 / 45.163361, final 82.308 (+0.309). sps
+0.90, tke +0.33, rel_l2 +0.07, time +0.17 (6.13 ms per step). Ten components in one upload; the intermediate
candidates were not uploaded. Rank 1.

### 2026-09-25: FZ-E, retrained on the second machine (82.311)

Codabench 944304, archive `53f7865c`; code = `2026-10-19_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-rec-ipc-s3-bundle-condwidth-v2-wholestep-t2-w01-kd`
(in `final_submission/realpde/submissions/`), preset `lb0925_fze`. Added the beyond-range
extension of the adaptive choices (AXE: wider energy-blend grid, adaptive blend ridge, CB weight 1.0 on window 0), the
regime-change reset (REC), kernel changes KD5 and KD4, and one-storage outputs (IPC). The zip was built from a
fresh-clone retrain on the RTX 4090 (development notes). Scores: 94.862633 / 81.031615 /
94.392779 / 91.544250 / 45.184824, final 82.311 (+0.004). rel_l2 +0.012, sps +0.021, tke −0.019, time −0.059
(6.22 ms per step): no measurable change, in line with the prediction (82.31). AXE and REC act mainly outside the
training range (REC never fires on normal streams), and the IPC lever did not apply to the official harness. The accuracy subscores were within ±0.02 of the prediction made for the
RTX 3090 lineage, which served as a check of retrain variance. Rank 2.

### 2026-09-26: KE3, LOWKE, KD6, KD7 and transport levers (82.419)

Archive `fa948762`, source dir `2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke`
(in `final_submission/realpde/submissions/`), preset `lb0926_fa948762`. Added the 4-seed energy net written as one dense network (KE3),
low-energy fluctuation damping (LOWKE), a multi-stream schedule of the whole-step graph (KD6), exact-rounding fused
kernels (KD7), and three transport levers: a multiprocessing warm-up at construction (IPW), direct CUDA I/O (CDIO) and
by-value outputs (BVR). RTX 3090 weights. Scores: 94.908958 / 81.009206 / 94.395393 / 92.417334 / 45.196247, final
82.419 (+0.107). Mostly time: +0.873 (6.22 to 4.91 ms per step), of which about 0.7 ms is attributed to by-value
outputs (the harness moves outputs by value) and about 0.6 ms to KD6 + KD7. rel_l2 +0.046 and tke −0.022, the
expected signature of LOWKE; sps +0.011. Rank 2.

### 2026-09-27: final (82.452)

Codabench 948041, archive `f11d2c1d`, source dir `2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk-recr-kd8-r97`
(in `final_submission/realpde/submissions/`), preset `lb0927_final`. Added the online least-squares correction of the ensemble head (LSK) with
forgetting factor 0.97, its reset when REC fires, and a further fusion layer (KD8); weights identical to 09-26.
Scores: 94.939383 / 81.016714 / 94.499950 / 92.464050 / 45.210544, final 82.452 (+0.033), with every subscore up:
mvpe +0.105, rel_l2 +0.030, time +0.047 (4.84 ms per step), sps +0.014, tke +0.008. First on the development
leaderboard at the time (np-user 82.448, doomduke2 82.370). A last candidate with a u,v-only output payload was
dropped that morning after an A100 test showed it slower. Source: `final_submission/README.md`.

### Final Decision Phase: the 948041 method retrained by the organizers

The package in `final_submission/` builds the method of submission 948041 with one command
(`solution.train(data_dir, output_dir, device)`, which runs `realpde/scripts/retrain_all.sh` with its default flags).
The organizers retrained the shortlisted methods from their submitted code and evaluated them on a private test set
with unseen parameter regimes. This entry won Track 2. No private-test scores are recorded in this repository.

## Which additions moved the score

Measured against the best score on the board before each upload, the largest steps were the Welsch loss with three
CUDA graphs (+0.962 on 2026-09-19, mostly from time), our own backbone with context normalisation (+0.732 on
2026-09-18, accuracy and sps up with a time loss), the time fix of the convection scan (+0.514 on 2026-09-08), and the
ensemble-and-intervals upload of 2026-09-24 (+0.309, mostly sps). The SimVP backbone with the M1 widths (+0.277), sim
pretraining with the b5 backbone and the W01/KD speed changes (+0.171), the whole-step graph with W2 (+0.127) and the
KE3/LOWKE/transport upload (+0.107, mostly time) followed. The last two uploads added +0.004 and +0.033. Most uploads
bundled several changes, so these board deltas are not per-component attributions; fold-level numbers for individual
components, measured when each was adopted, are in `docs/components.md` and in the evidence table of `docs/writeup/final_long.tex`.
