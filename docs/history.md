# How the solution evolved

This is a short account of team nmucke's Track 2 entry (Long-Term Test-Time Adaptation) from the first upload on
2026-09-03 to the last development-phase upload on 2026-09-27. Per-upload scores are in
`docs/submissions.md` and `experiments/leaderboard.csv`; the components named here are described in `docs/components.md`. The primary record
was the development ledger (pre-registered predictions and results for every upload) and the experiment notes, which
are not published. Short names such as CB, LSK or KD6 are expanded where they first appear below and are listed in
the abbreviation table of `docs/components.md`. sps is the competition's interval score (Safe Prediction Score).

Submission directory date prefixes are sequence labels, not upload dates (see the conventions in
`docs/components.md`); upload dates below come from the ledger. From 2026-09-18 on, every upload was preceded by a written prediction of its scores, so that the
leaderboard result could be read as a test of how local evidence transfers.

## Early September: data assimilation around the organizers' checkpoints

The first upload (2026-09-03) was the starting kit's gradient test-time-training template around the organizers'
CNO checkpoint, a plumbing check. The kit's per-step gradient update turned out to lose accuracy, which was traced to
BatchNorm running statistics drifting under batch-size-1 updates. The next line of work treated the problem as data
assimilation: an ensemble Kalman filter around the frozen FNO checkpoint, with a model-error covariance learned from
innovations and closed-form interval bounds (`2026-09-04_enkf-fno`), followed by ES-MDA with the FNO's last layer in
the filter state. Neither added anything over the additive model-error term: with a fully observed state, the analysis
step amounts to a reset onto the revealed truth, and a full least-squares refit of the last layer on one window did not
transfer to the next.

What did work was much cheaper. The gain map (`2026-09-04_gain-map-fno`, final 79.06, the first recorded board row)
keeps the frozen FNO, rescales its predicted fluctuation per pixel to a running energy target, adds a running bias
and emits closed-form bounds, with one forward pass per step instead of 16. The following week added a Taylor-advected
copy of the last frame, an (Re, AoA)-conditioned prior for the running statistics, a learned energy net (`ke_net`) and
a decoupled interval centre. The 2026-09-07 upload of this pipeline lost time (78.63 final) because the
convection-speed scan synchronised with the host 82 times per step; the batched scan re-uploaded on 2026-09-08 scored
79.57 (rank 40 of 84). On 2026-09-16 the `integrated` package (online mean correction, a two-vector interval centre
and a flow-field energy net, `ke_net_flow`) reached 79.73.

## Mid-September: held-out validation and our own backbones

A research pass on 2026-09-15 found that every local real file was inside the organizers' fine-tuning set, so all
local scores until then were memorised training-set scores; local numbers sat about 3 rel_l2 points and 14 sps points
above the leaderboard. Two consequences shaped the rest of the competition. Local decisions moved to
leave-conditions-out (LCO) folds that hold out whole (Re, AoA) conditions (fold1 for selection, fold0 as the primary
held-out fold, foldR for unseen Reynolds numbers at seen angles, mimicking the conditions that exist only in the simulation data, Rx
for Reynolds extrapolation, and simulation-only foldS sets as surrogates for unseen regimes).
And the backbone became our own, trained only on the released data.

The first own backbone was a ConvNeXt-UNet. Adding causal context normalisation (running per-pixel mean and std of
the revealed frames as input channels and for de-standardising the output) and switching the online mean correction
off gave fold0 80.92 vs 79.57 for the FNO pipeline. Uploaded on 2026-09-18 as `unet-ctxnorm-fast`, it scored 80.462:
accuracy and sps rose about as expected (about 0.6 times the fold0 deltas, a transfer factor used for planning from
then on), but the time subscore fell because the many-small-kernel UNet was launch-bound on the evaluation machine
(20.2 ms per step). The next day's upload combined a Welsch training loss with three CUDA graphs and reached 81.424
(14th of 94); the graphs brought the board step time to 7.27 ms.

On 2026-09-20 the backbone changed to a SimVP-style network with gSTA blocks (`n1_simvp`), trained with the S3 mixed
Welsch loss, together with two small energy-pipeline changes (the D2 bundle) and a learned conditional-width net (M1)
for the intervals. The upload scored 81.701 (7th of 50 listed). The interval gain transferred at about 0.9 times the
local estimate, the rel_l2 gain at only about 0.2 times, and the time got worse (7.79 ms) although kernel launches per
step had dropped from 218 to about 60.

## Timing: what the evaluation machine actually measures

The next three uploads were mostly about time. The launch-count model had failed, so a cost model was fitted on the
eval-box points instead. The 2026-09-21 upload (v8) returned outputs as unsynchronised CUDA tensors; its outputs were
bit-identical to the day before, and its time was 0.6 ms slower, which showed that the official timer includes GPU
work. It also showed that the accuracy and sps subscores are deterministic on the board and that run-to-run noise sits
in the time subscore. The 2026-09-22 upload, built on a second machine (RTX 4090), made each step one CUDA-graph replay
(the whole-step graph), cut GPU kernels (T2) and refitted the conditional widths on out-of-sample residuals over all 81
files (W2). It scored 81.828: sps +0.324 from W2 alone, and time 6.90 ms, about a third of the saving the cost model had
predicted.

## The last five days: sim pretraining, souping, ensembling and per-trajectory selectors

On 2026-09-23 a smaller SimVP variant (b5, 3.39 M parameters) was sim-pretrained on all 100 simulation files and
fine-tuned on the 81 real files, shipped together with graphs for the first two windows (W01) and a kernel diet (KD).
The upload scored 81.999 (3rd), within 0.001 of its prediction.

The 2026-09-24 upload changed the most at once. The backbone became a packed ensemble of three members, each a uniform
weight average of four fine-tunes from its own sim-pretrain seed, with the packed core in fp16. The intervals gained a
centre blend toward the raw ensemble output (CB), a per-trajectory adaptive blend weight (AW), an adaptive width
multiplier (part of AX) and a mask fix (MSK). On the accuracy side, AX also made the energy blend adaptive per
trajectory, and the energy-gain clamp was widened from 5 to 20 (gain_max 20). Two more kernel diets (KD2, KD3) offset most of the added cost. Several intermediate candidates were built that night; each
was checked by an independent review report and a verification run on the exact archive (both done by Claude-based AI
agents) and superseded before the upload. It scored 82.308 and moved the entry to first place. Under the
rules written in advance, sps above 44.85 meant the lineage was frozen and later uploads only added components on top
of it.

The 2026-09-25 upload added a beyond-training-range extension of the selectors (AXE), a regime-change detector (REC),
two more execution changes (KD5, KD4) and a transport lever (IPC). Its weights came from a from-scratch retrain on the
second machine, so the row (82.311, 2nd) also served as a check of retrain variance: every accuracy subscore was within ±0.02
of the prediction made for the first machine's (RTX 3090) lineage. The 2026-09-26 upload added a 4-seed energy net written as one dense network (KE3),
low-energy fluctuation damping (LOWKE), a multi-stream schedule of the whole-step graph (KD6), exact-rounding fused
kernels (KD7) and three transport levers: a multiprocessing warm-up at model construction (IPW), a direct path for
CUDA inputs (CDIO) and by-value output transport (BVR). The step time fell from 6.22 to 4.91 ms; about 0.7 ms of that is
attributed to BVR, which means the evaluation harness moves outputs by value. The score was 82.419 (2nd).

The last upload, on 2026-09-27, added an online least-squares correction of the ensemble head (LSK), its reset on a
regime change, a further fusion layer (KD8) and a forgetting factor of 0.97 instead of 0.9 for LSK. The forgetting
factor missed its study's own pre-declared fold0 bar (+0.0099 against +0.015) and was adopted as a disclosed override
because it was non-negative on every held-out and surrogate set; an independent refit on the second machine reproduced
it on 5/5 sets. A last candidate (a u,v-only output payload) was dropped that morning after an A100 test showed it was
slower. The upload, Codabench submission 948041 (archive `f11d2c1d`), scored 82.452 against a prediction of 82.455 and
was first on the development leaderboard at the time of submission (np-user 82.448, doomduke2 82.370).

The per-upload scores, deltas and ranks are tabulated in `docs/submissions.md`.

Over the whole phase the final score went from 79.06 (first recorded row) to 82.452. Measured against the best score
on the board at the time, the largest steps were the Welsch loss with CUDA graphs (+0.96, mostly time), our own backbone
with context normalisation (+0.73), the time fix of the convection scan on 2026-09-08 (+0.51) and the 2026-09-24
ensemble-and-intervals upload (+0.31). From 2026-09-24 on, the daily gains were small (+0.004 to +0.107) and
came as much from time as from accuracy.

## What the final solution is

The Final Decision Phase package in `final_submission/` builds the lineage of submission 948041 with one command. The
model is the packed K=3 ensemble of sim-pretrained, souped b5 backbones with an fp16 core, plus three fitted
auxiliaries: the condition prior, the KE3 energy net and the W2 width net. Around it runs a causal, gradient-free
pipeline: context normalisation, REC, LSK, energy calibration with the per-trajectory energy-blend and ridge
selectors, LOWKE, and intervals from W2 with the adaptive width multiplier and the adaptive centre blend. The whole
step is one CUDA-graph replay scheduled over several streams with exact fused kernels, and outputs are returned by
value. Changes in the local composite when each component was added are tabulated in the evidence table of
`docs/writeup/final_long.tex` and summarised in `docs/components.md`; the backbone recipe (sim pretraining, soup, ensemble) and the
centre blend gave the largest held-out gains, the per-trajectory selectors were close to neutral in range and larger
on the beyond-range surrogates, and the execution work took the board step from 7.8 to 4.8 ms.

Several directions did not make it into the final: gradient test-time training, EnKF and ES-MDA updates, CRPS-trained
ensembles, training losses aimed directly at TKE, divergence-free projection, output-side reduced models, and further
interval-width warps.

## Final Decision Phase

The development-leaderboard scores above are on hidden validation data. At the end of the development phase the
organizers shortlisted the top entries, retrained each method from scratch from its submitted training code, and
evaluated the retrained models on a private test set with unseen parameter regimes. The package in
`final_submission/` (the lineage of submission 948041) went through that process, and the entry won Track 2.
