"""P2/2026-09-27 -- D2 "free-rider bundle" (S3 loss + H1 energy-target blend
w=0.85 + variance-preserving mu; E1 window-0 sigma schedule DROPPED per D2's
own gate, marginal +0.0028 < the +0.005 keep bar) ported onto the graphed5
8-graph package (experiments/c16t_launch/, 50.85 launches/step). Base file:
`submissions/2026-09-26_simvp-ctxnorm-welsch-graphed5/submission.py`
(sha256 of the parent archive `8fe033f5...`), unmodified except for:

  * `CONFIG["e_blend_w"] = 0.85`, `CONFIG["mu_vp"] = True` (new knobs; ported
    verbatim from experiments/d2_bundle/fold0_c/submission.py, diffed against
    experiments/c11a_backbones2/fold0_n1simvp/submission.py for the clean
    patch -- see experiments/p2_bundle_pkg/bundle.diff).
  * `GainMapImprovedModel`: new state `self.R_run`/`self.n_R` (causal plain
    running mean of revealed target ke maps, cleared in `reset()`), new method
    `_update_R`, blend applied in `_g2_base`, variance-preserving mu applied
    in `step()`'s eager output-assembly branch.
  * `cuda_graph_megatail.py` (mega graph, window >= 2): the SAME e_tgt blend
    inlined into `_forward_body` (R_run passed in as a NEW fresh-per-step
    input buffer, `self._R_run` -- R_run's own update/seeding stays entirely
    EAGER in `step()`, never inside a capture, so no counter/EMA-scalar
    machinery is needed for it) + the same mu_vp block replacing the plain
    `out = out + new_mu`.
  * `cuda_graph_tail.py` (tail graph, the window-1 / mega-fallback path): the
    e_tgt blend already happens EAGERLY upstream (this graph receives
    `gamma` pre-blended, unchanged plumbing) -- only the mu_vp block is
    ported into its `_forward_body`.
  * `cuda_graph_bounds.py`/`cuda_graph_scan_uc.py`: UNCHANGED (E1 sig_sched,
    the only bundle piece that touched `bounds()`, was dropped).

Weights: retrained from scratch with `scripts/ship_simvp_finetune_s3.py`
(S3 loss, all 81 usable real files, never `train_real/7575_0.h5`); prior.npz/
ke_net_flow.pt refit on train+val with the unchanged `ship_simvp_fit_*`
scripts. See `submissions/2026-09-27_simvp-s3-bundle-graphed5/manifest.md`
for the full chain, evidence tables and sha256s.

Everything below (docstring included) is graphed5's own submission.py,
reproduced verbatim for context except where noted above.

---

c4p_packaging -- shipping candidate #2, `submissions/2026-09-20_unet-ctxnorm/`:
the c3b UNet `integrated` pipeline with a CAUSAL in-context-standardised
(RevIN-style) backbone, trained on ALL 81 usable real files (not an LCO fold),
packaged for submission. Builds on candidate #1
(`submissions/2026-09-19_unet-backbone/`, plain `unet_local`) and the C4-R1
family (`experiments/c4r1_ctxnorm/`, `mean_mode=None` chosen by held-out
inspection, unanimous on 36/36 files across fold0+foldR) with the reviewer's
constant-caching latency fix (`experiments/c4r1_review/latency_fix_check.py`,
`experiments/c4p_packaging/bit_identity_check.py`) applied.

Copied from `experiments/c4r1_ctxnorm/fold0_meanoff/submission.py`
(itself copied from `experiments/c3b_unet_pipeline/fold0/submission.py` per
the family rule: "copy ... with only the backbone construction swapped and
reset_ttt_state also clearing the wrapper state") with these changes on top:

  * `import ship_ctxnorm_models as _CTX` instead of `import c4r1_models as
    _CTX` -- `src/ship_ctxnorm_models.py` is a FIXED copy of the wrapper
    (`src/c4r1_models.py` is never edited, family rule): the original
    `CtxNormWrapper.forward`/`causal_ms`/`build_extra_maps` rebuilt five
    small constant tensors via `torch.as_tensor` on EVERY `ttt_step`
    (measured 26.09 ms mean/step); the fix builds them once, as registered
    buffers, at construction (measured 6.21 ms mean/step, same GPU/load).
    Proved bit-identical to the original THROUGH THE FULL PIPELINE (not just
    the wrapper in isolation): `torch.equal` on both predictions and bounds,
    168 streamed steps (>= 100 required) across all 4 `data/real/test`
    trajectories, `experiments/c4p_packaging/bit_identity_check.py`.
  * `CONFIG["unet_ckpt"]` -> `data/checkpoints/ship/unet_ctxnorm_all81.pth`:
    the ctx-norm backbone retrained with `scripts/c4r1_finetune.py` (its
    `--train` already accepts several directories, `nargs="+"`, and merges/
    dedupes/filters `7575_0.h5` exactly like `finetune_fno.collect()` -- no
    wrapper script needed for this step) on ALL 81 usable real files
    (`data/real/{train,val,test}`), matching the convention candidate #1 and
    the organizers' own `sim_real_fno_fp16.pth` used.
  * `prior.npz` / `ke_net_flow.pt` next to this file are refit on
    `data/real/train` + `data/real/val` (77 files, test never opened) with
    NEW thin wrappers `scripts/ship_ctxnorm_fit_prior.py` /
    `scripts/ship_ctxnorm_fit_ke_flow.py` -- merge of `c4r1_fit_prior.py` /
    `c4r1_fit_ke_flow.py`'s STATEFUL/causal (batch-1, sequential,
    `wrapper.reset()` per file) forward with `ship_unet_fit_prior.py` /
    `ship_unet_fit_ke_flow.py`'s "all-files mode" (`--files` takes several
    directories, merged), using the FIXED wrapper throughout.
  * `mean_mode` stays `None` (c4r1's winning config on both LCO folds).

Everything below this point (docstring included) is c4r1_ctxnorm/fold0's own
text (itself carrying c3b's own text) describing the rest of the pipeline and
its own backbone-swap diff, reproduced verbatim for context -- NOT touched by
this candidate beyond the changes listed above:

---

Original header (`experiments/c4r1_ctxnorm/fold0_meanoff/submission.py`)
follows, unmodified below this line.

---

c4r1_ctxnorm/fold0 -- the c3b UNet `integrated` pipeline with a CAUSAL
in-context-standardised (RevIN-style) backbone (`src/c4r1_models.py`) instead
of the plain `unet_local`, on LCO fold0.

Copied from experiments/c3b_unet_pipeline/fold0/submission.py (family rule:
"copy ... with only the backbone construction swapped and reset_ttt_state
also clearing the wrapper state") with exactly those two changes:

  * `get_ttt_model` builds `c4r1_models.CtxNormWrapper` (a stateful wrapper
    around `c4r1_models.CtxNormUNet`, itself a `unet_local` with a 4-channel-
    extended stem) instead of the plain `unet_local`, loading the
    `scripts/c4r1_finetune.py` checkpoint named by `C3B_UNET_CKPT` (env,
    `_resolve_unet_ckpt` reused unchanged -- it is just a path resolver) /
    `CONFIG["unet_ckpt"]`. The wrapper's `forward(x)` already does
    accumulate-then-predict internally (see its docstring), so the single
    `raw = self.base(x)` call site inside `GainMapImprovedModel.step` (S1 of
    the original c3b docstring below) needs no change at all.
  * `ExperimentTTT.reset_ttt_state` additionally calls the wrapper's own
    `reset()` (clears its causal S1/S2/n) next to `self.model.reset()` (the
    pipeline's own online-adaptation state) -- both are trajectory-scoped.
  * `prior.npz` / `ke_net_flow.pt` next to this file are refit for the
    ctx-norm backbone on fold0 TRAIN only (`scripts/c4r1_fit_prior.py`,
    `scripts/c4r1_fit_ke_flow.py` -- STATEFUL adaptations of the c3b fit
    scripts, since the frozen forecast is now causal/sequential, not a
    chunked one-shot forward; see experiments/c4r1_ctxnorm/notes.md).

Everything below this point (docstring included) is c3b's own unmodified
text describing the rest of the pipeline (advect/gain/bias/spread/ke-net/
bounds), which this task did not touch -- reproduced verbatim for context:

Copied from experiments/lco_baseline/fold0/submission.py (itself an unmodified
copy of experiments/integrated/submission.py except the path-depth fix) with
ONLY the backbone construction changed:

  * `get_ttt_model` builds `c1d_models.build("unet_local")` instead of
    `load_baseline.build_model("fno")`, and loads the checkpoint named by
    `C3B_UNET_CKPT` (env) / `CONFIG["unet_ckpt"]` (single config point, per
    task instruction -- a better UNet checkpoint drops in by changing one
    string). Default: `data/checkpoints/c1d/unet_local_fold0.pth` (recipe B on
    fold0 TRAIN, but sim-pretrained for only 1500 it -- see
    experiments/c1d_capacity/notes.md and experiments/c3b_unet_pipeline/notes.md).
  * `prior.npz` / `ke_net_flow.pt` next to this file are refit for this UNet
    checkpoint on fold0 TRAIN only (`scripts/c3b_fit_prior.py`,
    `scripts/c3b_fit_ke_flow.py` -- thin adaptations of the `lco_fit_*`
    scripts that swap the FNO builder for the UNet one; see notes.md).
  * The ke_net_flow features (`src/ke_net_flow.py:features`) are computed
    purely from the input window, the backbone's OWN predicted fluctuation and
    energies -- no FNO-internal activations are consumed anywhere in the
    downstream pipeline -- so no feature-source change was needed beyond
    swapping which backbone produces `fl_f` / `e_f`.

Nothing else in the pipeline logic (advect/blend, gain rule, online mean
correction, decoupled interval centre, bounds) was touched. See
`experiments/c3b_unet_pipeline/fold0_v1/submission.py` for V1 (energy net off,
== `experiments/c2_diagnostics/stage_b_meanonly`'s CONFIG diff applied here).

Original header (experiments/integrated/submission.py) follows, unmodified.

---

The merge of the two 2026-09-16 tracks, on top of experiments/ke_net:

  * experiments/knobs_honest: the online window-mean correction
    (`mean_mode = "both"`, `mean_ridge = 0.25`) and the two-vector interval
    centre basis (`ctr_basis2`). Every other knob of that fork keeps the
    default that reproduces ke_net bit-for-bit.
  * experiments/tke_gate: the learned flow-field energy net
    (`ke_flow = "ke_net_flow.pt"`) REPLACES `ke_net`'s energy correction, so
    `ke_net = None` and `ke_net_flow.pt` ships instead of `ke_net.pt`.

Both branches are residual: with `ke_flow = None`, `mean_mode = None` and
`ctr_basis2 = False` this file is experiments/ke_net bit-for-bit (verified in
notes.md), so the ke_net numbers remain the reference for every row here.

The energy net is fitted on the *features of the backbone it runs on*
(`src/tke_energy_fit.py --ckpt <backbone>`, ~20 min): refitting it is worth
+1.5 tke / +0.4 sps on clean test, so `ke_net_flow.pt` must be re-fitted
whenever `model.pth` changes (notes.md, "Re-packaging with a new backbone").

Original header follows.

gain_map_improved -- RealPDE Track 2 LTTTA submission.

Scaffolded by scripts/create_experiment.py (forked from experiments/gain_map). This
file is the experiment: CONFIG below is the whole configuration, and it is what
ships in the zip.

Runs in two places unchanged:
  * in place, `python src/local_eval.py --submission experiments/gain_map_improved \
        --data starting_kit/example_data` (kit modules resolved from src/,
    checkpoint resolved from data/competition/baseline_checkpoints/);
  * from the packaged zip, where scripts/create_submission.py has placed the
    imported kit modules and `model.pth` next to this file.

Contract (docs/track2.md): every tensor is (1, 20, 32, 64, 3) in normalized
space, `ttt_step` and `reset_ttt_state` are timed, and `info["lower"]` /
`info["upper"]` are all-or-nothing for the entire run.

Idea: gain_map's three online per-pixel maps (fluctuation gain, bias, spread),
plus two things the stream itself can tell us -- the wake's propagation speed
and the flow parameters -- used where each provably helps:

  * propagation speed -> the *shape* of the fluctuation. The vortex street
    convects downstream at Uc px/frame, read off every input window as the
    shift that best maps frame t onto frame t+tau (grid scan, running average
    over the trajectory; 0.80 px/frame = 0.74 U_inf at Re 11k / AoA 0, 1.36 =
    0.60 U_inf at Re 24k / AoA 15). Taylor's hypothesis makes the last input
    frame, shifted by Uc (k+1), a forecast of lead k with the right *phase*:
    its correlation with the truth is 0.94 at lead 0 against 0.75 for the FNO,
    and it decays with lead time -- slowly at low Re (0.4 at lead 19), within
    two frames at high Re. It exists only where the source column lies
    downstream of the airfoil / laser shadow. The fluctuation shape is
    a * (F(x) - mean_t F) + b * advected with (a, b) per lead and channel the
    ridge least-squares fit to the revealed targets (per-element sufficient
    statistics, pooled separately over pixels with and without an advected
    value), then rescaled per pixel and window to the energy below.
  * energy -> gain_map's rule, untouched: per (pixel, channel) the output's
    temporal variance this window is var_t(F') times the running ratio of the
    input's energy to the forecast's, to the power 0.75. Measured on the two test
    trajectories, the FNO's window-to-window energy modulation tracks the true
    TKE better than anything the propagation speed offers (the advected
    pattern's energy adds nothing at high Re), and the per-window TKE varies by
    ~50 %, so the energy is not the blend's to set: the least-squares blend is
    rel_l2-optimal but zeroes the fluctuation where nothing predicts the phase,
    which is exactly where TKE still needs the energy. Splitting shape and
    energy keeps the TKE map identical to gain_map's and improves the phase
    where it can.
  * parameters -> the starting point of the running statistics. (Re, AoA) are
    read off the first input window (src/parameter_estimation.py) and select,
    by a Gaussian kernel in (AoA, log Re), the training files whose converged
    maps (mean innovation, residual variance, input and forecast energies;
    scripts/fit_gain_prior.py) seed the bias, spread and gain with the weight
    of a few windows, so the first windows of every trajectory -- where
    gain_map's statistics start from nothing -- are no longer the worst.
  * bias / spread: gain_map's running mean and pooled variance of the
    innovation; SPS interval at the closed-form score-optimal width.

This fork (ke_net) adds two ways of improving that one number -- the per-(pixel,
channel) energy the fluctuation is rescaled to -- and nothing else:

  * `ke_net`: an offline-trained residual correction to the energy rule
    (src/ke_net.py). The net predicts a log-space correction on top of
    log[var_t(F') (runx/runf)^0.75], with its last layer zero-initialised, and is
    fitted on data/real/train (selected on data/real/val, test never touched)
    with the tke metric itself as the loss. `ke_net_scale = 0` is the shipped
    rule bit-for-bit. `ke_net_online` additionally takes a few Adam steps per
    revealed window on the just-observed (features -> true ke map) pair.
  * `er_online`: the online-tuned energy rule. A small candidate grid of
    multiplicative corrections (amplitude A, modulation exponent q) is scored on
    every revealed window by the *exact* realised TKE the candidate would have
    produced, and the running argmin is used for the next window. With `ke_net`
    on, the candidates multiply the net's energy target rather than the rule's.

It also decouples the interval's centre from the returned prediction (`ctr`, from
sps_ideas/sps_3). Only `pred` enters the accuracy factors, so the bounds may be
centred anywhere -- and the shipped `pred` is deliberately energy-inflated for TKE,
which makes the innovation the intervals are built from louder than it needs to be.
So carry a second centre `const + g * basis`, with `const` the FNO/blend mean and
`basis` the un-inflated least-squares fluctuation (the blend shape *before* the
energy rescaling), and emit `centre +- k*(sigma2) sigma2` while returning `pred`
untouched. Writing `a = y - const`, every member's innovation is `a - g basis`, so
one pair of running means (M0, M1) and three pooled second moments (Saa, Sab, Sbb)
give the bias `M0 - g M1` and the variance `Saa - 2 g Sab + g^2 Sbb` of *every* `g`
in closed form -- including the per-(pixel, channel) least-squares `g* = Sab/Sbb`,
which minimises sigma2 and hence (the interval factor being monotone decreasing in
sigma) maximises SPS. `ctr = False` is the pred-centred baseline.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch

CONFIG = {
    "base": "unet_local",      # c3b: src/c1d_models.py MODEL_NAMES entry (was "fno")
    # c3b: SINGLE config point for the backbone checkpoint -- swap in a better
    # UNet checkpoint by changing this string (or setting env C3B_UNET_CKPT,
    # which wins). Relative paths resolve against the repo root.
    "unet_ckpt": "data/checkpoints/ship_simvp_s3/simvp_s3_all81.pth",  # P2: S3-trained n1_simvp, all 81 usable real files (dev fallback only; packaged model.pth wins, env C3B_UNET_CKPT wins over both)
    "ckpt": "sim_real_fno_fp16.pth",            # unused by c3b's get_ttt_model; kept for header parity
    "backbone_ckpt": None,         # dev-only override (see _resolve_ckpt); env RPDE_BACKBONE_CKPT wins
    "adapt": False,                # kept for the notes.md summary: no gradient TTT here
    # --- propagation-speed fluctuation source ---
    "advect": True,                # blend in the Taylor-advected last input frame
    "uc_max": 4.0,                 # px/frame: top of the convection-speed scan grid
    "uc_step": 0.1,                # scan resolution (parabolic refinement on top)
    "uc_taus": (2, 3),             # frame lags the scan correlates over
    "uc_fixed": None,              # px/frame: bypass the scan with a constant (diagnostics)
    "uc_warm": 2,                  # P7 stack: scan cadence off (KD-R GO) -- scan every window until this many are seen ...
    "uc_every": 1000000000,        # ... then every uc_every-th window; large finite int stands in for infinity (graph never rescans past uc_warm)
    "blend": "region",             # "region": (a, b) per lead & channel, split valid/invalid | "pixel": per element on top
    "blend_ridge": 1.0,            # ridge toward (a, b) = (1, 0) [gain_map], relative to the local energies; 0 = pure LS
    "blend_pixel_n0": 8.0,         # "pixel" only: pseudo-windows of ridge toward the region solution (fades as 1/n)
    "blend_smooth": True,          # "pixel" only: 3x3 box-smooth the sufficient statistics before solving
    # --- gain / bias / spread ---
    "gain": True,                  # gain_map's rule: per-window energy = var_t(F') * (E_input / E_F)^power, running per pixel
    "gain_power": 0.75,            # gamma = ratio^(power/2); < 1 shrinks the gain toward 1
    "gain_max": 20.0,              # clip gamma to [1/gain_max, gain_max] (AE: 20.0 shipped, was 5.0)
    # FP16X (experiments/fp16x_core): packed-backbone core precision. False = the shipped TF32 bmm packing
    # (bit-exact). True / "all" = fp16 stem..head (fp32-accumulating tensor-core GEMMs, GroupNorm stats in
    # fp32 inside torch, ctx-norm + de-standardisation fp32; 2 casts per step, inside the backbone graph).
    # "blocks" = only the 6 translator blocks in fp16. A single-b5 checkpoint is run as a K=1 pack when on.
    "fp16_core": True,     # K3F: shipped ON (fp16 core, scope "all"; experiments/k3f_build)
    "shape_cap": 2.0,              # blend shape scaled up to this factor to reach the window energy; F' fills the rest
    # --- learned TKE-map predictor (src/ke_net.py, scripts/fit_ke_net.py) ---
    #   log E_out = log[ ef (runx/runf)^gain_power ] + ke_net_scale * net(features)
    # The net's last layer is zero-initialised and it is trained as a residual on the
    # rule, so ke_net = None (or ke_net_scale = 0) is gain_map_improved bit-for-bit.
    "ke_net": None,            # integrated: off -- the flow net below replaces it
    "ke_net_scale": 1.0,           # shrink knob on the log correction; 0 = the rule, 1 = the fit
    "ke_net_clip": 1.5,            # cap on |log correction| (a factor 4.5 in energy)
    "ke_net_online": False,        # False | int: Adam steps per revealed window on the
                                   # previous window's features vs the revealed ke map.
                                   # Timed; reset_ttt_state restores the offline weights.
    "ke_net_online_lr": 1e-4,      # lr of those steps
    # --- learned flow-field correction to the energy (src/ke_net_flow.py) ---
    # When set, the flow net's energy map REPLACES ke_net's for the u, v channels
    # (both are residuals on the same rule, so applying both would double-correct);
    # `ke_flow_blend` < 1 takes a geometric blend in log-energy space,
    #   log E_out = (1 - blend) * log E_ke_net + blend * log E_ke_flow.
    "ke_flow": "ke_net_flow.pt",   # src/ke_net_flow.py weights next to this file
    "ke_flow_scale": 1.0,          # shrink knob on the flow net's log correction
    "ke_flow_blend": 1.0,          # 1 = the flow net alone, 0 = ke_net alone
    # --- online-tuned energy rule (idea_1), applied on top of the above ---
    #   E_out = E_base * A * (kf/runf)^(q-1) * (kx/runx)^w
    # with E_base the rule's energy, or the net's when ke_net is on -- so the candidate
    # family is always "amplitude x modulation-shrinkage around the current energy
    # target", and (q, w, A) = (1, 0, 1) leaves that target untouched. (q, w, A) is
    # chosen online per trajectory by the realised score on the revealed windows.
    "er_online": False,            # False = whatever the energy target already is
    "er_q": (0.7, 0.85, 1.0),      # exponent on the FNO's window-to-window energy modulation
    "er_w": (0.0,),                # weight on the input window's own modulation (persistence)
    "er_a": (1.0, 1.05, 1.1),      # global amplitude. Capped at 1.1 on purpose: letting the
                                   # search reach 1.15-1.2 buys ~0 extra TKE and costs
                                   # 0.2-0.6 SPS, because a louder prediction means a louder
                                   # innovation and hence a wider optimal interval.
    "er_warm": (1.0, 0.0, 1.0),    # used until the first target is revealed
    "er_obj": "tke",               # "tke" = ke-map rel-L2 only | "both" = + er_lam * field rel-L2
    "er_lam": 1.4,                 # ratio of the two subscores' d(score)/d(error) at their operating points
    "er_exact": True,              # score candidates on the reconstructed output ke (incl. the bias
                                   # term's own temporal variance); False = the bare energy rule
    # --- (Re, AoA)-conditioned prior for the running statistics ---
    "prior": True,                 # start mu / var / energies from prior.npz maps if the file exists
    "prior_windows": 6.0,          # pseudo-windows the prior counts for before the stream takes over
    "prior_sigma_aoa": 2.5,        # deg: kernel width over the training files' AoA
    "prior_sigma_re": 0.15,        # log-Re kernel width (0.15 ~ 16 % in Re, one grid step is ~10 %)
    "bias": True,                  # add the running mean innovation
    "bounds": True,                # SPS intervals from the running innovation variance
    "pool_t": True,                # innovation variance pooled over the 20 frames: True | False | "sep" (pooled x lead profile)
    "alpha_min": 0.05,             # floor on the running-average gain max(1/(n+prior_n), alpha_min)
    "prior_n": 1.0,                # pseudo-count of the initial statistics
    "k_scale": 1.0,                # multiplier on the optimal interval half-width

    # --- decoupled interval centre (sps_ideas/sps_3) ---
    # Only `pred` enters the SPS accuracy factors; the bounds may be centred anywhere.
    # The shipped `pred` is energy-inflated for TKE, which makes its innovation louder
    # than necessary. When `ctr` is on we carry a SECOND, un-inflated centre with its
    # own running bias and innovation variance, and put the interval around that while
    # returning `pred` unchanged -- so rel_l2 / tke / mvpe are bit-identical either way.
    "ctr": True,                   # False = baseline (bounds centred on pred)
    "ctr_w": 1.0,                  # weight of the un-inflated LS blend shape zc vs the shipped fluc
    "ctr_g": "auto_px",            # float shrink on the centre's fluctuation (0 = FNO mean only),
                                   # or "auto" / "auto_px": the running least-squares gain,
                                   # pooled over the field / per (pixel, channel)
    "ctr_g_ridge": 0.1,            # pseudo-weight pulling the online gain toward 1 (val-selected)
    "ctr_g_clip": (0.0, 1.25),     # clamp on the online gain
    "ctr_bias": True,              # give the centre its own running bias map
    "ctr_k_scale": 1.0,            # multiplier on the centre's optimal half-width

    # --- knobs_honest: every default below reproduces ke_net bit-for-bit -------
    # (1) prior life: how long the (Re, AoA) prior outweighs the stream
    "prior_windows_e": None,       # separate pseudo-count for the energy statistics (None = prior_windows)
    "prior_windows_c": None,       # separate pseudo-count for the centre moments (None = prior_windows)
    # (2) convection-scan instability guard
    "uc_guard": None,              # None | "median" | "shrink" | "both"
    "uc_guard_k": 4,               # most recent scans the stability test looks at
    "uc_guard_cv": 0.25,           # median-absolute-deviation / median above which the scan is unstable
    "uc_guard_peak": 0.0,          # scan peak correlation below which the scan is "flat" (0 = off)
    "uc_guard_w": 0.0,             # weight kept on the Taylor branch while unstable
    # (3) mean / bias predictor
    "mu_alpha_min": None,          # gain floor of the bias map only (float or per-channel); None = alpha_min
    "mean_mode": None,             # c4r1 meanoff variant: online mean/bias correction OFF (was "both")
    "mean_ridge": 0.25,            # pseudo-weight toward "no correction" (slope) / toward the FNO mean (ls)
    "mean_clip": 1.0,              # clamp on the slope correction coefficient
    # (4) interval centre basis
    # (5) incoherent-band energy floor (experiments/gain_map_improved/tke_ideas/idea_4.md)
    "band_split": None,            # None = off; int kc = cut the fluctuation at DFT mode kc
    "hi_scale": 2.0,               # the injected high-band energy, in units of the input window's own
    "ctr_lead": False,             # per-lead centre gain (second moments not pooled over t)
    "ctr_basis2": True,            # two-vector basis (blend shape, advected fluctuation), 2x2 per-pixel LS
    "seed": 0,

    # --- D2 free-rider bundle: S3 loss (checkpoint-level, see unet_ckpt below) +
    # H1 energy-target blend + variance-preserving mu (experiments/d2_bundle/notes.md).
    # E1 window-0 sigma schedule was DROPPED at the D2 gate (marginal +0.0028 < the
    # +0.005 bar) and is NOT ported here. Ported verbatim from
    # experiments/d2_bundle/fold0_c/submission.py (diffed against
    # experiments/c11a_backbones2/fold0_n1simvp/submission.py for the clean patch).
    "e_blend_w": 0.85,              # D2 rung (c): e_tgt = w*Khat + (1-w)*R_run (fold0-TRAIN optimum)
    "mu_vp": True,                  # D2 rung (c): variance-preserving mu

    # --- P3: M1 learned conditional interval widths (experiments/m1_cond_widths/,
    # experiments/p3_condwidth_pkg/). A small (~22k param) conv net multiplies the
    # pipeline's OWN half-width by lambda*exp(alpha*net(causal features)); centres
    # and predictions are UNTOUCHED (only bounds() half-width changes). Trained
    # OFFLINE, out-of-sample, on held-out streams of this exact bundle's own fold
    # pipelines (scripts/ship_condwidth_fit.py) -- see cond_width_net.py (vendored,
    # no repo imports) for the online causal-feature construction. Default OFF:
    # behaviour with cond_width=False is bit-identical to the 09-27 bundle
    # (graphed5+D2), including still using the graphed bounds() replay.
    "cond_width": True,

    # --- CB (experiments/cb_centre_blend): per-lead convex blend of the EMITTED interval centre with the
    # raw backbone output: bounds are shifted by w_l * (raw - ctr_out) (half-widths, pred and every piece
    # of internal state untouched). None = off = bit-exact with the parent. Otherwise a list of 20 floats
    # (one per lead, applied to u and v; the unmeasured 3rd channel gets 0) or a (20, 2) nested list.
    # SHIPPED 2026-10-10 (CBP): one global w = 0.75 for every lead (flat optimum; honest per-fold fits within
    # 0.005 of it on fold0 / fold1 / foldR -- experiments/cb_centre_blend/notes.md). A plain scalar is NOT
    # accepted by get_ttt_model (it needs 20 values), hence the 20-vector.
    "cb_blend_w": [0.75] * 20,
    # BR (experiments/br_beyond, arm 3): CB weight used on WINDOW 0 only (no truth revealed yet); None = cb_blend_w.
    "cb_blend_w0": 1.0,   # BR arm 3 (window-0 CB fallback; shipped ON in AXE)
    # --- AW (experiments/aw_adaptive_cb): causal per-trajectory CB weight (cb_adapt.py). None = off =
    # bit-exact with the fixed-w CB above. Otherwise {"grid": [...], "rho": float}: from window 2 of a
    # trajectory, w = grid[argmax S], S <- rho S + (score each grid w would have earned on the previous
    # window); windows 0/1 keep cb_blend_w. Requires cb_blend_w (its lead-0 u value is the tie-break w).
    "cb_adapt": {"grid": [0.30, 0.45, 0.60, 0.75, 0.90], "rho": 0.95},   # AW C3 (shipped ON)
    # --- AX (experiments/ax_adaptive, ax_adapt.py): two more causal per-trajectory choices, whole-step graph only
    # (windows >= 2; the eager / W01 paths keep the fixed values). None = off = bit-exact with the parent.
    # kw_adapt (needs cb_adapt): {"grid": [...], "groups": [[leads], ...], "rho": r} -- W2 half-width x kappa per
    #   lead group, kappa = grid[argmax S], S <- rho S + exact sps term of each kappa on the previous window.
    # eb_adapt: {"grid": [...], "rho": r} -- e_blend_w per trajectory: all grid values evaluated each step, the one
    #   with the best EMA of the revealed-window composite J (rel_l2 / tke / mvpe) is emitted (from window 3 on).
    "kw_adapt": {"grid": [0.85, 0.925, 1.0, 1.075, 1.15], "groups": [[0, 1], list(range(2, 20))], "rho": 0.95},   # AX A2 (shipped ON)
    "eb_adapt": {"grid": [0.45, 0.60, 0.75, 0.85, 0.925], "rho": 0.95},   # AX B1 + AXE extended grid (shipped ON)
    # AXE (experiments/axe_extended, axe_adapt.py): blend_ridge per trajectory, factorised with eb_adapt (requires it):
    # {"grid": [...], "rho": r}; windows 0/1 and the tie prior use CONFIG["blend_ridge"]. None = off (bit-exact).
    "br_adapt": {"grid": [0.125, 0.25, 0.5, 1.0], "rho": 0.95},   # AXE E4 (shipped ON)
    "cond_width_ckpt": "cond_width_weights.pt",   # next to this file; env P3_CONDWIDTH_CKPT wins
    # --- MSK (experiments/msk_mask): scored mask on the revealed target (W2 error map, AW / AX scorers).
    # False = `pt != 0.0` (parent, bit-exact); True = `|pt| > 1e-6` (= the scorer's / W2 training's exact mask).
    "exact_mask": True,    # MSK shipped ON (experiments/mskp_pkg)
    # --- KD2 (experiments/kd2_adapt_diet, kd2_adapt.py): AW + AX (+ AXE) selectors and their CB / MSK glue fused in
    # the whole-step graph (same decisions, fewer kernels per step). False = the parent code path, bit for bit.
    "kd2_adapt": True,
    # --- KD3 (experiments/kd3_step_diet, kd3_step.py): bit-exact kernel diet of the whole-step graph's backbone
    # (bias-prefilled GEMMs, in-place fp16 input / output assembly) and mega bodies (stacked moment state). False = KD2.
    "kd3_step": True,
    # --- IPC (experiments/ipc_outproc): graphed steps return pred / lower / upper as three disjoint contiguous views of
    # ONE fresh CPU clone (one storage per step) instead of three clones -- same values. "auto" = only when the input
    # arrived through torch's shared-memory multiprocessing reduction (fd strategy). False = KD3 bit for bit.
    "ipc_one_storage": "auto",
    "ipc_prewarm": True,          # c2 IPW: untimed out-of-process warm-up in get_ttt_model (mp child + fd strategy only)
    # HX2 (experiments/hx2_harness): CUDA-input fast path in the whole-step graph (active ONLY when the caller passes CUDA
    # tensors; CPU / numpy inputs take the unchanged path). False = FZ-E bit for bit.
    "cuda_direct_io": True,
    # --- KD5 (experiments/kd5_more, kd3_step.py): NEAR-EXACT variants inside the KD3 whole-step body: stem / down convs
    # as unfold + GEMM, dw2 as a space-to-batch depthwise conv, W2 c1 bias as a ones channel. False = KD3, bit for bit.
    "kd5_near": True,
    # --- KD4 (experiments/kd4_w01_diet, kd4_w01.py): W01's window-0 / window-1 graphs captured through dieted hooks
    # (KD3 backbone, no wrapper copies) with the scan / prior / PRE block on a side stream (fork / join, same ops on
    # the same values). False = KD3.
    "kd4_w01": True,
    # --- REC (experiments/rec_recovery, rec_detect.py): causal regime-change detector (window mean velocity vs the
    # ctx-norm running mean, |log ratio| > thr) + partial state reset / velocity-scale rescale inside the whole-step
    # graph. None = off = the parent bit for bit; on normal streams the flag never fires (outputs bit-identical).
    "rec_detect": {"thr": 0.405465, "n0": 6.0, "rescale": True, "u_eps": 1e-4},
}

# Official train_real per-channel statistics (mean_std_real.pt). The evaluator
# normalises the input with the *input* stats and the target with the *target*
# stats; SPS measures interval width against SIGMA_GLOBAL in physical units.
_MEAN_IN = (0.15496086, -0.00051399, 0.0)
_STD_IN = (0.09680565, 0.01596068, 1.0)
_MEAN_TGT = (0.15496257, -0.00051779, 0.0)
_STD_TGT = (0.09681041, 0.01596364, 1.0)
_SIGMA_GLOBAL = 0.0563870259

_HERE = os.path.dirname(os.path.abspath(__file__))
# NOTE (lco_baseline): this file is an unmodified copy of experiments/integrated/submission.py
# (commit ca20bb5), except for this one path-depth fix. The original assumes it lives at
# <repo>/experiments/<name>/ (two levels under the repo root) and walks up two pardirs; this
# copy lives at <repo>/experiments/lco_baseline/fold0/ (three levels), so it needs a third.
# No pipeline/model logic changed -- see experiments/lco_baseline/notes.md.
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))

# Packaged: load_baseline.py / rpde_baselines/ sit next to this file. Dev: they
# live in the repo's src/. _HERE is inserted last so the packaged copy wins.
for _p in (os.path.join(_REPO, "src"), _HERE):
    if os.path.isfile(os.path.join(_p, "load_baseline.py")) and _p not in sys.path:
        sys.path.insert(0, _p)

import ke_net as K  # noqa: E402
import ke_net_flow  # noqa: E402
import c1d_models as _C1D  # noqa: E402  (c3b: UNet backbone builder)
import ship_ctxnorm_models as _CTX  # noqa: E402  (c4p: latency-fixed causal in-context standardisation wrapper)
import ship_simvp_models as _SIMVP  # noqa: E402  (c13p: N1SimVP backbone + CtxNormUNet-compatible wrapper)
import ship_b1_models as _B1  # noqa: E402  (P8: vendored B1 backbone b5_h192_blk6_rf11)
import ship_b5x2_models as _B5X2  # noqa: E402  (EN-S: packed K=2 b5 ensemble)
from cuda_graph_ctxnorm import GraphedCtxNormWrapper  # noqa: E402  (c6t: launch-cost pass)
from cuda_graph_tail import GraphedTailWrapper  # noqa: E402  (c6t2: launch-cost pass, phase 2)
from cuda_graph_keflow import GraphedKeFlowWrapper  # noqa: E402  (c8t: launch-cost pass, phase 3)
from cuda_graph_pre import GraphedPreWrapper  # noqa: E402  (c10t: launch-cost pass, phase 4)
from cuda_graph_mask import GraphedMaskWrapper  # noqa: E402  (c12t: launch-cost pass, phase 5)
from cuda_graph_megatail import GraphedMegaWrapper  # noqa: E402  (c12t: launch-cost pass, phase 5)
from cuda_graph_bounds import GraphedBoundsWrapper  # noqa: E402  (c16t: launch-cost pass, phase 6)
from cuda_graph_scan_uc import GraphedScanUcWrapper  # noqa: E402  (c16t: launch-cost pass, phase 6)
from load_baseline import load_checkpoint_state  # noqa: E402
from parameter_estimation import estimate_aoa, estimate_re  # noqa: E402
import cond_width_net as _CW  # noqa: E402  (P3: learned conditional interval widths)
from cond_width_graph import GraphedCondWidthWrapper  # noqa: E402  (P4: launch-cost pass, cond_width)
from cuda_graph_wholestep import GraphedWholeStepWrapper  # noqa: E402  (T1: one graph per step)
from w01_warmup_graph import GraphedWarmupWrapper  # noqa: E402  (W01: window-0/1 graphs)
from cb_adapt import CBAdapt, cb_adapt_select  # noqa: E402  (AW: causal per-trajectory CB weight)
from ax_adapt import KWAdapt, EBAdapt  # noqa: E402  (AX: causal per-trajectory width multiplier / energy blend)
from axe_adapt import BRAdapt  # noqa: E402  (AXE: causal per-trajectory blend_ridge)


def _resolve_cond_width_ckpt(submission_dir: str) -> str:
    """Single config/env point for the P3 conditional-width net weights --
    same resolution order/pattern as `_resolve_unet_ckpt`: env override wins,
    else the packaged file next to `submission_dir` (the archive's own
    extraction dir, as the eval harness passes it to `get_ttt_model`), else
    `CONFIG["cond_width_ckpt"]` resolved relative to the repo root (dev
    convenience only, mirrors every other checkpoint resolver in this file)."""
    env = os.environ.get("P3_CONDWIDTH_CKPT")
    if env:
        path = env if os.path.isabs(env) else os.path.join(_REPO, env)
        if not os.path.isfile(path):
            raise FileNotFoundError("cond_width checkpoint override not found: %s" % path)
        return path
    packaged = os.path.join(submission_dir, os.path.basename(CONFIG["cond_width_ckpt"]))
    if os.path.isfile(packaged):
        return packaged
    path = os.path.join(_REPO, CONFIG["cond_width_ckpt"])
    if not os.path.isfile(path):
        raise FileNotFoundError("cond_width checkpoint not found: %s" % path)
    return path


def _resolve_unet_ckpt(submission_dir: str) -> str:
    """c4p: the single config/env point for the UNet backbone checkpoint --
    PACKAGED-model.pth-aware (c3e_packaging's fix, carried forward here: the
    fold0_meanoff/c4r1 version of this function -- reproduced above this
    edit in the copied docstring text -- never checked ``submission_dir`` at
    all, because that experiment was "dev-only, never packaged for
    submission". This candidate IS packaged, so without this fix the archive
    would try to resolve ``CONFIG["unet_ckpt"]`` (a repo-relative path)
    against a ``_REPO`` that does not exist on the eval platform and crash
    at ``get_ttt_model`` -- exactly the bug ``c3e_packaging`` fixed for the
    plain-UNet candidate; this mirrors that fix, and ``_resolve_ckpt``'s own
    packaged-``model.pth``-first pattern for the FNO checkpoint, verbatim.

    ``C3B_UNET_CKPT`` (env) wins outright (dev override, e.g. pointing at a
    different local checkpoint) over the packaged ``<submission_dir>/model.pth``,
    which in turn wins over ``CONFIG["unet_ckpt"]`` (repo-relative dev
    fallback, used when running in place / unpackaged). A named-but-missing
    override path is an error; a missing packaged file falls through to the
    dev fallback instead of erroring immediately, so this function still
    works unpackaged (in the repo tree) exactly as before.
    """
    override = os.environ.get("C3B_UNET_CKPT")
    if override:
        path = override if os.path.isabs(override) else os.path.join(_REPO, override)
        if not os.path.isfile(path):
            raise FileNotFoundError("UNet backbone checkpoint override not found: %s" % path)
        return path
    packaged = os.path.join(submission_dir, "model.pth")
    if os.path.isfile(packaged):
        return packaged
    name = CONFIG.get("unet_ckpt")
    if not name:
        raise FileNotFoundError("no UNet checkpoint configured: set C3B_UNET_CKPT "
                                "or CONFIG['unet_ckpt']")
    path = name if os.path.isabs(name) else os.path.join(_REPO, name)
    if not os.path.isfile(path):
        raise FileNotFoundError("UNet backbone checkpoint not found: %s" % path)
    return path


def _resolve_ckpt(submission_dir: str) -> str:
    """model.pth from the archive root, else the dev checkpoint tree.

    Development override: ``RPDE_BACKBONE_CKPT`` (or ``CONFIG["backbone_ckpt"]``)
    names a checkpoint to load instead, absolute or relative to the repo root.
    Both formats that ``load_checkpoint_state`` knows work (fp16-packed and the
    kit's fp32 ``model_state_dict`` layout), so a locally fine-tuned backbone can
    be streamed through the unmodified pipeline. Unset in a packaged submission,
    so shipping behaviour is untouched; a named-but-missing path is an error.
    """
    override = os.environ.get("RPDE_BACKBONE_CKPT") or CONFIG.get("backbone_ckpt")
    if override:
        path = override if os.path.isabs(override) else os.path.join(_REPO, override)
        if not os.path.isfile(path):
            raise FileNotFoundError("backbone override not found: %s" % path)
        return path
    packaged = os.path.join(submission_dir, "model.pth")
    if os.path.isfile(packaged):
        return packaged
    for sub in ("sim_real_ft", "sim_pretrain"):
        dev = os.path.join(_REPO, "data", "competition", "baseline_checkpoints",
                           sub, CONFIG["ckpt"])
        if os.path.isfile(dev):
            return dev
    raise FileNotFoundError(
        "no model.pth in %s and no %s under data/competition/baseline_checkpoints/"
        % (submission_dir, CONFIG["ckpt"]))


def _load_prior(device: str):
    """prior.npz next to this file (scripts/fit_gain_prior.py), as device tensors, or None.

    Per training file: mean innovation (physical, fp16), pooled residual variance and
    the input / forecast energies, plus its (Re, AoA). Converted here to the
    normalised spaces the online statistics live in (target stats for the
    innovation and the forecast energy, input stats for the input energy).
    """
    path = os.path.join(_HERE, "prior.npz")
    if not os.path.isfile(path):
        return None
    import numpy as np
    z = np.load(path)
    std_t = torch.tensor(_STD_TGT[:2]); std_i = torch.tensor(_STD_IN[:2])
    pad = lambda t: torch.cat([t, torch.zeros_like(t[..., :1])], dim=-1)   # p channel
    return {
        "re": torch.tensor(z["re"], dtype=torch.float32, device=device),
        "aoa": torch.tensor(z["aoa"], dtype=torch.float32, device=device),
        "mu": pad(torch.tensor(z["mu"].astype(np.float32)) / std_t).to(device),            # (F, T, H, W, 3)
        "var": pad(torch.tensor(z["var"]) / std_t ** 2).unsqueeze(1).to(device),         # (F, 1, H, W, 3)
        "e_true": pad(torch.tensor(z["e_true"]) / std_i ** 2).unsqueeze(1).to(device),
        "e_pred": pad(torch.tensor(z["e_pred"]) / std_t ** 2).unsqueeze(1).to(device),
    }


class _KeFlow:
    """The learned flow-field correction to the energy rule (src/ke_net_flow.py).

    Lifted unchanged from experiments/ke_net_flow/submission.py: the net sees the last
    input frames and the FNO fluctuation as scale-free fields, so it can resolve *where*
    the wake is and what state it is in, which no summary of the energy can. The head is
    zero-initialised and the parameterisation is residual, so `scale = 0` is the rule.

    Holds the per-trajectory state its features need (the previous window's energy maps
    and the window counter), cleared by `reset`.
    """

    def __init__(self, net, cfg: Dict[str, Any], scale: float):
        self.net, self.cfg, self.scale = net, cfg, float(scale)
        # c8t (launch-cost, phase 3): set by get_ttt_model AFTER a real window
        # (n >= 1) has populated ex_prev/ef_prev (GraphedKeFlowWrapper needs
        # real, correctly-shaped state to capture against, exactly like
        # GraphedTailWrapper). None here means "use the eager path below";
        # reset() must NOT touch this attribute (same contract as
        # GainMapImprovedModel._tail).
        self._graph = None
        # W01: `leads` as ONE device LongTensor built at the first (untimed, get_ttt_model
        # warm-up) call and reused -- `features` does `ffl.index_select(1, leads_idx)`
        # either way (same indices, same values); a per-call CPU->device tensor is not
        # capturable inside the window-0 CUDA graph (w01_warmup_graph.py).
        self._leads_dev = None
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.ex_prev = self.ef_prev = None

    def __call__(self, x, fl_f, e_x, e_f, runx, runf, mask):
        """Per-(pixel, channel) target energy of the output fluctuation, (1, 1, H, W, 3)."""
        if self._graph is not None and self._graph.should_use():
            # c8t (launch-cost, phase 3): ONE CUDAGraph replay in place of the
            # feature build + KeFlowNet forward + correction below -- bit-exact
            # transcription (unmodified `ke_net_flow.features`/
            # `apply_correction`/`self.net` calls on static buffers), see
            # cuda_graph_keflow.py's module docstring for the full argument.
            # Gated to n >= 1 (never window 0 of a trajectory, where
            # ex_prev/ef_prev are None -- structurally different code path
            # below), a vanishing fraction of steps on any real (stride-20)
            # trajectory.
            return self._graph.forward(x, fl_f, e_x, e_f, runx, runf, mask)
        c = self.cfg
        ex, ef = e_x[:, 0, ..., :2], e_f[:, 0, ..., :2]
        xw, ffl = x[..., :2], fl_f[..., :2]
        rx, rf = runx[:, 0, ..., :2], runf[:, 0, ..., :2]
        halves = None
        if c["summary"]:
            halves = (xw[:, :10].var(1, unbiased=False), xw[:, 10:].var(1, unbiased=False),
                      ffl[:, :10].var(1, unbiased=False), ffl[:, 10:].var(1, unbiased=False))
        wfrac = torch.full((1,), min(self.n, 20) / 20.0, device=x.device, dtype=x.dtype)
        if self._leads_dev is None and len(c["leads"]):
            self._leads_dev = torch.as_tensor(list(c["leads"]), device=x.device, dtype=torch.long)
        feat = ke_net_flow.features(
            xw, ffl, ex, ef, rx, rf, mask[None], wfrac,
            k_in=c["k_in"], leads=(self._leads_dev if self._leads_dev is not None else c["leads"]), summary=c["summary"], halves=halves,
            ex_prev=self.ex_prev, ef_prev=self.ef_prev)
        if c.get("lead_e") or c.get("phase"):
            # nets fitted by src/tke_energy_fit.py carry the extra state channels
            import tke_energy_feats  # noqa: PLC0415  (diagnostic nets only)
            feat = torch.cat([feat, tke_energy_feats.extra_features(
                xw, ffl, ef, mask[None], bool(c.get("lead_e")), bool(c.get("phase")),
                int(c.get("n_blocks", 4)), c.get("phase_kind", "probe"))], 1)
        e_new = ke_net_flow.apply_correction(
            self.net(feat), ke_net_flow.rule_log_energy(ef, rx, rf), mask[None], self.scale)
        self.ex_prev, self.ef_prev, self.n = ex, ef, self.n + 1
        out = e_f.clone()                       # p channel keeps the rule's value
        out[:, 0, ..., :2] = e_new
        return out


def _load_ke_flow(cfg: Dict[str, Any], device: str):
    """The trained flow net named by CONFIG['ke_flow'], next to this file, or None.

    KES (`runs/briefs/KE_seed_ensemble.md`, Stage 1/2): if the checkpoint is a
    grouped-conv packed K-seed ensemble (saved by
    `src/kes_ke_flow_packed.py:save_packed`, which stamps a top-level "k" key that
    `ke_net_flow.save_ke_flow_net`'s plain-net checkpoints never have), load it via
    `kes_ke_flow_packed.load_packed` instead of `ke_net_flow.load_ke_flow_net` --
    `KeFlowNetPacked` exposes the same `.stem.weight` / `forward(feat) -> (1,H,W,2)`
    contract `_KeFlow`, `GraphedKeFlowWrapper`, and `cuda_graph_megatail.py`'s
    `self._kf_net(feat)` call all rely on generically (none of them hand-transcribe
    the net's own layers), so this is otherwise an unmodified drop-in.
    """
    name = cfg.get("ke_flow")
    if not name or float(cfg.get("ke_flow_scale", 1.0)) == 0.0:
        return None
    path = name if os.path.isabs(name) else os.path.join(_HERE, name)
    if not os.path.isfile(path):
        raise FileNotFoundError("ke_flow weights not found: %s" % path)
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "k" in raw:
        import kes_ke_flow_packed as _KESP  # noqa: PLC0415  (KES Stage 1 packed net)
        net = _KESP.load_packed(path, device)
        ncfg = dict(raw["cfg"])
    else:
        net, ncfg = ke_net_flow.load_ke_flow_net(path, device)
    return _KeFlow(net, ncfg, cfg["ke_flow_scale"])


def _load_ke_net(device: str, cfg: Dict[str, Any]):
    """The trained TKE-map predictor next to this file (scripts/fit_ke_net.py), or None.

    Resolved exactly like prior.npz, so scripts/create_submission.py packages it with the
    rest of the experiment. A named-but-missing file is an error rather than a silent
    fallback to the rule: a submission that quietly loses its correction is worse than one
    that fails the preflight.
    """
    name = cfg.get("ke_net")
    if not name or not float(cfg["ke_net_scale"]):
        return None
    return K.load_ke_net(os.path.join(_HERE, name), device)


# --------------------------------------------------------------------------- #
# SPS-optimal interval width for a Gaussian error.
# --------------------------------------------------------------------------- #
def _optimal_k_table(n: int = 512, a_min: float = 1e-3, a_max: float = 1e2):
    """k*(a) maximising exp(-a k) * erf(k / sqrt 2), a = 2 sigma_phys / SIGMA_GLOBAL.

    For an element with Gaussian error of std sigma and a symmetric interval of
    half-width k sigma, the expected SPS element score is
    P(inside) * exp(-width / SIGMA_GLOBAL) = erf(k/sqrt2) * exp(-a k). The
    maximiser solves a erf(k/sqrt2) = sqrt(2/pi) exp(-k^2/2); solved once by
    bisection on a log grid of `a` (untimed, at construction).
    """
    log_a = torch.linspace(math.log(a_min), math.log(a_max), n, dtype=torch.float64)
    ks = []
    for a in log_a.exp().tolist():
        lo, hi = 0.0, 10.0
        for _ in range(60):
            k = 0.5 * (lo + hi)
            g = a * math.erf(k / math.sqrt(2.0)) - math.sqrt(2.0 / math.pi) * math.exp(-0.5 * k * k)
            if g > 0:
                hi = k
            else:
                lo = k
        ks.append(0.5 * (lo + hi))
    return log_a.float(), torch.tensor(ks, dtype=torch.float32)


def _ke(x: torch.Tensor) -> torch.Tensor:
    """Per-pixel, per-channel temporal variance of a (1, T, H, W, C) window: (1, 1, H, W, C)."""
    return x.var(dim=1, keepdim=True, unbiased=False)


# --------------------------------------------------------------------------- #
# Streamwise shift with linear interpolation (Taylor's hypothesis).
# --------------------------------------------------------------------------- #
class _Shifter:
    """Shift fields along W by a fractional number of columns, source-validity aware.

    A pixel at column c shifted by d takes its value from column c - d (linear
    interpolation). The value is valid when the whole path c - d .. c lies in
    the domain and downstream of the last masked pixel (airfoil body / laser
    shadow) upstream of c on that row, so nothing is advected through the
    airfoil or from outside the field of view.
    """

    def __init__(self, w: int, device: str):
        self.w = w
        self.cols = torch.arange(w, device=device, dtype=torch.float32)

    def last_obstacle(self, mask: torch.Tensor) -> torch.Tensor:
        """(H, W) column index of the last masked pixel at or upstream of each column, -1 if none."""
        idx = torch.where(mask, self.cols.expand_as(mask), torch.full_like(self.cols, -1.0).expand_as(mask))
        return torch.cummax(idx, dim=-1).values

    def __call__(self, f: torch.Tensor, d: float, obstacle: torch.Tensor):
        """f (..., H, W, C) shifted by d columns; returns (shifted, valid (H, W))."""
        src = self.cols - d
        i0 = torch.floor(src)
        wgt = (src - i0).view(-1, 1)
        i0 = i0.long()
        valid = (i0.view(1, -1) >= obstacle.long() + 1) & (i0 >= 0).view(1, -1)
        i0c = i0.clamp(0, self.w - 1)
        i1c = (i0 + 1).clamp(0, self.w - 1)
        out = (1.0 - wgt) * f[..., i0c, :] + wgt * f[..., i1c, :]
        return out, valid


class GainMapImprovedModel:
    """F(x) blended with the advected input, then online gain, bias and spread."""

    def __init__(self, base: torch.nn.Module, device: str, cfg: Dict[str, Any]):
        self.base = base.to(device).eval()
        self.device = device
        self.cfg = cfg
        self.log_a, self.k_tab = (t.to(device) for t in _optimal_k_table())
        self.std_tgt = torch.tensor(_STD_TGT, device=device)
        self.mean_in = torch.tensor(_MEAN_IN, device=device)
        self.std_in = torch.tensor(_STD_IN, device=device)
        self.mean_tgt = torch.tensor(_MEAN_TGT, device=device)
        # gain floor of the bias map only: scalar or per-channel, as a broadcastable map
        f = cfg.get("mu_alpha_min")
        self.mu_floor = None if f is None else torch.tensor(
            [float(f)] * 3 if not isinstance(f, (tuple, list)) else list(f) + [0.0] * (3 - len(f)),
            device=device).view(1, 1, 1, 1, 3)
        self.shift = _Shifter(64, device)
        self.uc_grid = torch.arange(0.0, float(cfg["uc_max"]) + 1e-6, float(cfg["uc_step"]), device=device)
        # c5t (speed): _scan_uc's per-tau shift geometry (the unclamped column
        # index i0, the interpolation weight wgt, and the clamped i0c/i1c) is a
        # function only of self.shift.cols / self.uc_grid / tau -- all fixed
        # right here at construction, never of any window's data -- so it is
        # bit-identical on every call for a given tau. `_scan_uc` used to
        # recompute it (floor + long-cast + 2x clamp over a (U, W) grid) on
        # every one of its own calls; precomputed once per configured tau
        # instead and reused (`self._uc_geom`), which `_scan_uc` only builds
        # `valid` and the correlation sums from the actual window data.
        self._uc_geom: Dict[Any, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for _tau in cfg["uc_taus"]:
            _u = self.uc_grid.numel()
            _src = self.shift.cols.view(1, -1) - self.uc_grid.view(-1, 1) * _tau
            _i0 = torch.floor(_src)
            _wgt = (_src - _i0).view(1, 1, 1, _u, self.shift.w, 1)
            _i0 = _i0.long()
            _i0c = _i0.clamp(0, self.shift.w - 1)
            _i1c = (_i0 + 1).clamp(0, self.shift.w - 1)
            self._uc_geom[_tau] = (_i0, _wgt, _i0c, _i1c)
        self.prior = _load_prior(device) if cfg["prior"] else None
        # the learned energy correction, plus the weights reset_ttt_state restores
        self.ke = _load_ke_net(device, cfg)
        self.ke_flow = _load_ke_flow(cfg, device)
        self.ke_steps = int(cfg["ke_net_online"] or 0) if self.ke is not None else 0
        self.ke_offline = ({k: v.detach().clone() for k, v in self.ke.state_dict().items()}
                           if self.ke_steps else None)
        self.ke_opt = None
        # online energy-rule candidates, as (K, 1, 1, 1, 1, 1) tensors ready to broadcast
        self.er_cands = [(float(q), float(w), float(a))
                         for q in cfg["er_q"] for w in cfg["er_w"] for a in cfg["er_a"]]
        v = lambda j: torch.tensor([c[j] for c in self.er_cands], device=device).view(-1, 1, 1, 1, 1, 1)
        self.er_q, self.er_w, self.er_a = v(0), v(1), v(2)
        # c6t2 (launch-cost, phase 2): set by get_ttt_model AFTER a dummy
        # warm-up trajectory has populated self.S/mu/M0/... (GraphedTailWrapper
        # needs real, non-None, correctly-shaped state to capture against).
        # None here means "use the eager path", exactly like
        # GraphedCtxNormWrapper's own `.ok = False` fallback -- reset() below
        # (and every later reset_ttt_state()) must NOT touch this attribute.
        self._tail = None
        # (c10t: launch-cost pass, phase 4) same story for `self._pre`
        # (GraphedPreWrapper: _update_error + _update_ctr + _update_blend,
        # the PRE-stage graph). Set by get_ttt_model after the SAME dummy
        # warm-up trajectory used for `self._tail` (that warm-up's window 1
        # already runs the PRE-stage updates eagerly, populating exactly the
        # state this graph needs -- see cuda_graph_pre.py's module docstring
        # and get_ttt_model's comment). reset() must NOT touch this attribute.
        self._pre = None
        # (c12t: launch-cost pass, phase 5) `self._mega` merges PRE +
        # energy/ke_net_flow + TAIL into ONE CUDA graph for window >= 2 (the
        # same window range `self._pre` already used) -- see
        # cuda_graph_megatail.py's module docstring. Strictly additive: when
        # None or not `.should_use()`, step() falls back to the existing
        # `self._pre`/`self._tail`/`ke_flow._graph` chain above, unchanged.
        # `self._maskg` graphs `_mask` (needed every step, including window
        # 0, so it has no window-based gating). Neither touched by reset().
        self._mega = None
        self._maskg = None
        # (c16t: launch-cost pass, phase 6) `self._scang` graphs `_scan_uc`
        # (pure function of (fl_x, obstacle, mask), no window gating -- see
        # cuda_graph_scan_uc.py's module docstring). `self._boundsg` graphs
        # `bounds()` (called from ExperimentTTT.ttt_step, not step() -- see
        # cuda_graph_bounds.py's module docstring). Neither touched by
        # reset().
        self._scang = None
        self._boundsg = None
        # (P3: learned conditional interval widths) loaded ONCE here (not in
        # reset()) -- same lifetime as self.ke/self.ke_flow/self.prior above.
        # cfg["cond_width_ckpt_path"] is set by get_ttt_model (submission_dir
        # is not known at __init__ time from cfg alone); falls back to the
        # repo-relative dev path via _resolve_cond_width_ckpt if absent (dev
        # convenience, mirrors every other checkpoint in this file).
        self.cw = None
        _CW.EXACT_MASK = bool(cfg.get("exact_mask", False))   # MSK: before any capture
        if cfg.get("cond_width"):
            ckpt_path = cfg.get("cond_width_ckpt_path") or _resolve_cond_width_ckpt(_REPO)
            self.cw = _CW.CondWidthState(ckpt_path, device)
        # (P4: launch-cost pass) `self._cwg` -- GraphedCondWidthWrapper,
        # folds cond_width's net forward + causal feature construction into
        # 2 CUDA graphs (update_graph/apply_graph). Set by get_ttt_model
        # AFTER self.cw is loaded above. UNLIKE `_tail`/`_pre`/`_mega`/
        # `_maskg`/`_scang`/`_boundsg` (never touched by reset()), this
        # wrapper's `_live` one-time-seed flag IS cleared by reset() below,
        # per the task's own "reset by host-side copy_ in reset_ttt_state"
        # option -- without this, a stale `_live=True` left over from a
        # PRIOR trajectory would make the NEXT trajectory's window 0/1
        # incorrectly try to replay apply_graph against unseeded persistent
        # buffers (cw.reset() clears the read side but not this flag).
        # Cheap: 0 GPU launches, a plain python bool write.
        self._cwg = None
        self.reset()

    def reset(self) -> None:
        if self.cw is not None:
            self.cw.reset()
        if self._cwg is not None:
            self._cwg._live = False
            self._cwg._ctr_old_valid = False
        self.n = 0                  # innovations seen
        self.n_e = 0                # energy pairs seen
        self.n_uc = 0               # convection-speed estimates seen
        self.n_win = 0              # windows seen in this trajectory
        self.n_b = 0                # blend pairs seen
        self.uc = None              # convection speed, px/frame (0-d tensor on the device)
        self.params: Optional[Tuple[float, float]] = None   # (Re, AoA) read off the first window
        self.prev_out = self.prev_fl = self.prev_adv = self.prev_mf = self.prev_mask = None
        self.n_w = 0                # windows seen in this trajectory (the net's wfrac feature)
        self.mu = self.var = self.prof = self.e_true = self.e_pred = None
        # D2/H1 free-rider bundle: causal PLAIN running mean of revealed target ke
        # maps (self.R_run), seeded from the window-0 input's own ke (self.n_R=1) in
        # the eager branch of step() -- guaranteed non-None by the time any graph
        # (mega/tail, both gated to window >= 1) ever reads it. Seeding/clearing is
        # UNCONDITIONAL (not gated on cfg["e_blend_w"]): a few cheap extra host-side
        # ops even when the blend is off, in exchange for R_run always being a valid,
        # correctly-shaped tensor input for the graphed megatail wrapper -- dead
        # value when e_blend_w is None, never read.
        self.R_run = None
        self.n_R = 0
        # the alternative centre, carried as the two components y - const and the
        # fluctuation basis, so any shrink g has a closed-form bias and variance
        self.M0 = self.M1 = self.Saa = self.Sab = self.Sbb = None
        self.n_c = 0
        self.prev_const = self.prev_basis = self.ctr_out = self.ctr_var = None
        self.uc_hist = []           # scanned convection speeds (0-d tensors), for the stability guard
        self.uc_eff = None          # the speed actually advected with (guard may freeze it)
        self.adv_w = None           # weight on the Taylor branch (guard may shrink it)
        self.M2 = self.S02 = self.S12 = self.S22 = None   # second centre basis
        self.prev_basis2 = None
        self.Ec = self.Ey = self.Sdd = self.Sdy = None    # window-mean regression
        self.prev_cdev = self.prev_c_raw = self.prev_cx = None
        self.mx_bias = None         # running (target mean - input mean)
        self.run_xhi = None         # running high-band energy of the input window
        self.Mcc = self.Mxx = self.Mcx = self.Mcy = self.Mxy = None   # mean_mode "ls"
        self.n_m = 0
        self.S = None               # blend sufficient statistics: dict of (1, T, H, W, C) maps
        self.ke_ex_prev = self.ke_ef_prev = None    # the net's lagged-energy features
        if self.ke_flow is not None:
            self.ke_flow.reset()
        self.ke_prev = None         # (features, log rule energy, mask) of the emitted window
        self.g2b = None             # energy gain before the online candidate correction
        self.er_acc = None          # (K,) accumulated realised error of every candidate
        self.er_cur = tuple(float(t) for t in self.cfg["er_warm"])
        self.er_prev = None         # what the candidate scorer needs about the last window
        if self.ke_steps:           # undo whatever the previous trajectory adapted
            self.ke.load_state_dict(self.ke_offline)
            self.ke_opt = torch.optim.Adam(self.ke.parameters(),
                                           lr=float(self.cfg["ke_net_online_lr"]))

    def _a(self, n: int) -> float:
        return max(1.0 / (n + float(self.cfg["prior_n"])), float(self.cfg["alpha_min"]))

    def _w01_loss(self, t: torch.Tensor):
        """adapt_loss as a python float (a host sync). W01's window-1 capture overrides this
        on the instance to keep the 0-dim tensor instead (no sync inside the capture)."""
        return float(t)

    # ------------------------------------------------------------------ masks
    def _mask(self, x: torch.Tensor) -> torch.Tensor:
        """(H, W) pixels that are exactly zero (physical u and v) in every frame."""
        phys = x[0, ..., :2] * self.std_in[:2] + self.mean_in[:2]
        return (phys.abs() < 1e-6).all(-1).all(0)

    # ------------------------------------------------------- convection speed
    def _scan_uc(self, fl: torch.Tensor, obstacle: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Shift (px/frame) that best maps frame t onto frame t+tau inside the window.

        Correlation over valid, unmasked pixels of the shifted earlier frames with
        the later frames, summed over the lags in `uc_taus`, maximised on the grid
        and refined by a parabola through the three best points. All candidates are
        gathered in one batched indexing op and the result stays on the device: no
        host sync per candidate (the per-candidate loop this replaces cost ~30 ms per
        step on a GPU, four times the FNO forward).
        """
        f = fl[..., :2]
        grid = self.uc_grid
        u, w = grid.numel(), self.shift.w
        keep = (~mask).view(1, 1, *mask.shape, 1, 1).to(f.dtype)          # (1, 1, H, W, 1, 1)
        obs = obstacle.long().unsqueeze(0)                                  # (1, H, W)
        num = den_a = den_b = torch.zeros(u, device=f.device, dtype=f.dtype)
        for tau in self.cfg["uc_taus"]:
            # c5t (speed): i0/wgt/i0c/i1c are a pure function of (grid, cols, tau)
            # -- none of which depend on this window's data -- precomputed once
            # at construction (see __init__) instead of every call.
            i0, wgt, i0c, i1c = self._uc_geom[tau]
            valid = (i0.unsqueeze(1) >= obs + 1) & (i0 >= 0).unsqueeze(1)   # (U, H, W)
            a = f[:, :-tau]                                                 # (1, T', H, W, 2)
            shifted = (1.0 - wgt) * a[:, :, :, i0c, :] + wgt * a[:, :, :, i1c, :]   # (1, T', H, U, W, 2)
            b = f[:, tau:].unsqueeze(3)                                     # (1, T', H, 1, W, 2)
            k = valid.permute(1, 0, 2).reshape(1, 1, mask.shape[0], u, w, 1).to(f.dtype) * keep.view(1, 1, mask.shape[0], 1, w, 1)
            num = num + (shifted * b * k).sum(dim=(0, 1, 2, 4, 5))
            den_a = den_a + (shifted * shifted * k).sum(dim=(0, 1, 2, 4, 5))
            den_b = den_b + (b * b * k).sum(dim=(0, 1, 2, 4, 5))
        s = num / (den_a * den_b + 1e-30).sqrt()
        k = s.argmax()
        km, kp = (k - 1).clamp(0, u - 1), (k + 1).clamp(0, u - 1)
        den = s[km] - 2.0 * s[k] + s[kp]
        interior = (k > 0) & (k < u - 1) & (den < 0)
        off = torch.where(interior, 0.5 * (s[km] - s[kp]) / torch.where(den < 0, den, -torch.ones_like(den)),
                          torch.zeros_like(den))
        return grid[k] + off * float(self.cfg["uc_step"]), s[k]

    def _scan_uc_dispatch(self, fl: torch.Tensor, obstacle: torch.Tensor, mask: torch.Tensor):
        """c16t (launch-cost, phase 6): ONE CUDAGraph replay in place of
        `_scan_uc` -- bit-exact transcription (see cuda_graph_scan_uc.py's
        module docstring for the full argument: no window gating needed,
        `_scan_uc` has no counter-driven state). Falls back to the unmodified
        eager `_scan_uc` when `self._scang` is None or its own capture
        failed. The HOST-VISIBLE firing schedule (uc_warm/uc_every) that
        decides WHETHER to call this at all lives unchanged at each of the
        three call sites below -- this dispatch only chooses graphed-vs-eager
        for a call that is already happening.
        """
        if self._scang is not None and self._scang.should_use():
            return self._scang.forward(fl, obstacle, mask)
        return self._scan_uc(fl, obstacle, mask)

    def _update_uc(self, scan) -> None:
        """Merge a new scan into the running speed, then apply the stability guard.

        The scan is noisy on a few trajectories (25425_0: uc 0.84 +- 0.67 over windows,
        `notes/research_2026-09-15/probe_dmd.md`), and a wrong speed advects the last
        input frame into the wrong place -- worse than not advecting at all. The guard
        calls a scan unstable when the recent scans disagree (MAD / median above
        `uc_guard_cv`) or the objective is flat (peak correlation below
        `uc_guard_peak`), and then freezes the speed at the running median and/or
        shrinks the Taylor branch to `uc_guard_w`. Everything stays on the device: no
        host sync, and the cost is a median of <= `uc_guard_k` scalars.
        """
        uc, peak = scan
        if self.uc is None:
            self.uc = uc
        else:
            self.n_uc += 1
            self.uc = self.uc + self._a(self.n_uc) * (uc - self.uc)
        mode = self.cfg["uc_guard"]
        if not mode:
            self.uc_eff, self.adv_w = self.uc, None
            return
        self.uc_hist.append(uc.detach())
        hist = torch.stack(self.uc_hist[-int(self.cfg["uc_guard_k"]):])
        med = hist.median()
        bad = torch.zeros((), device=uc.device, dtype=torch.bool)
        if hist.numel() >= 2:
            mad = (hist - med).abs().median()
            bad = mad / med.abs().clamp_min(1e-2) > float(self.cfg["uc_guard_cv"])
        if float(self.cfg["uc_guard_peak"]) > 0.0:
            bad = bad | (peak < float(self.cfg["uc_guard_peak"]))
        self.uc_eff = torch.where(bad, med, self.uc) if mode in ("median", "both") else self.uc
        self.adv_w = (torch.where(bad, torch.full_like(med, float(self.cfg["uc_guard_w"])),
                                  torch.ones_like(med))
                      if mode in ("shrink", "both") else None)

    def _advect(self, fl_last: torch.Tensor, obstacle: torch.Tensor, mask: torch.Tensor, t: int):
        """Last input frame's fluctuation shifted by uc*(k+1) for k < t, in one gather:
        (1, T, H, W, C), and the validity (1, T, H, W, 1)."""
        w = self.shift.w
        uc = self.uc if self.uc_eff is None else self.uc_eff
        d = uc * torch.arange(1, t + 1, device=fl_last.device, dtype=fl_last.dtype)   # (T,)
        src = self.shift.cols.view(1, -1) - d.view(-1, 1)                                   # (T, W)
        i0 = torch.floor(src)
        wgt = (src - i0).view(1, 1, t, w, 1)
        i0 = i0.long()
        valid = (i0.unsqueeze(1) >= obstacle.long().unsqueeze(0) + 1) & (i0 >= 0).unsqueeze(1) & (~mask).unsqueeze(0)  # (T, H, W)
        i0c, i1c = i0.clamp(0, w - 1), (i0 + 1).clamp(0, w - 1)
        adv = (1.0 - wgt) * fl_last[:, :, i0c, :] + wgt * fl_last[:, :, i1c, :]              # (1, H, T, W, C)
        adv = adv.permute(0, 2, 1, 3, 4) * valid.view(1, t, *mask.shape, 1).to(fl_last.dtype)
        adv[..., 2] = 0.0
        return adv, valid.view(1, t, *mask.shape, 1)

    # ------------------------------------------------------------------ blend
    def _update_blend(self, fl: torch.Tensor, adv: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> None:
        keep = (~mask).view(1, 1, *mask.shape, 1).to(fl.dtype)
        cur = {"ff": fl * fl, "aa": adv * adv, "fa": fl * adv, "fy": fl * y, "ay": adv * y}
        cur = {k: v * keep for k, v in cur.items()}
        if self.S is None:
            self.S = cur
            return
        self.n_b += 1
        a = self._a(self.n_b)
        self.S = {k: self.S[k] + a * (cur[k] - self.S[k]) for k in cur}

    @staticmethod
    def _smooth(m: torch.Tensor) -> torch.Tensor:
        """3x3 box average over (H, W) of a (1, T, H, W, C) map, edge-aware."""
        t, h, w, c = m.shape[1:]
        flat = m.permute(0, 1, 4, 2, 3).reshape(t * c, 1, h, w)
        out = torch.nn.functional.avg_pool2d(flat, 3, stride=1, padding=1, count_include_pad=False)
        return out.reshape(1, t, c, h, w).permute(0, 1, 3, 4, 2)

    @staticmethod
    def _solve(ff, aa, fa, fy, ay, lam: float, with_b: bool, a0=1.0, b0=0.0):
        """Ridge least squares y ~ a f + b adv toward (a0, b0), weight lam x own energies.

        Cauchy-Schwarz gives det >= ff aa (1 - (1+lam)^-2) > 0 whenever lam > 0; the
        clamp guards float32 and lam = 0.
        """
        p11 = ff * (1.0 + lam)
        r1 = fy + lam * ff * a0
        if not with_b:
            return r1 / (p11 + 1e-30), torch.zeros_like(r1)
        p22 = aa * (1.0 + lam)
        r2 = ay + lam * aa * b0
        det = (p11 * p22 - fa * fa).clamp_min(1e-3 * p11 * p22 + 1e-30)
        return (p22 * r1 - fa * r2) / det, (p11 * r2 - fa * r1) / det

    def _coefficients(self, valid: torch.Tensor, mask: torch.Tensor):
        """(a, b) maps of shape (1, T, H, W, C) from the running statistics.

        Ridge least squares toward gain_map's (a, b) = (1, 0), weight `blend_ridge`
        relative to the pooled energies, so that where the revealed targets say the
        blend explains nothing the FNO fluctuation is kept rather than zeroed:
            [ff (1+l), fa; fa, aa (1+l)] [a; b] = [fy + l ff; ay].
        """
        T = valid.shape[1]
        shape = (1, T, *mask.shape, 3)
        if self.S is None:
            return torch.ones(shape, device=self.device), torch.zeros(shape, device=self.device)
        lam = float(self.cfg["blend_ridge"])
        keep = (~mask).view(1, 1, *mask.shape, 1)
        v = (valid & keep).to(self.S["ff"].dtype)
        i = (~valid & keep).to(self.S["ff"].dtype)
        pooled = {}
        # T2 (coefsum, bit-exact -- experiments/t2_gpu_compute/notes.md S2): the
        # 10 per-key pooled sums as ONE multiply and ONE reduction over a
        # stacked tensor (was 10 reduce_kernel<512,1> launches).
        w2 = torch.stack([v, i], 0)                                   # (2,1,T,H,W,3)
        keys = ("ff", "aa", "fa", "fy", "ay")
        st = torch.stack([self.S[k] for k in keys], 0)                # (5,1,T,H,W,3)
        allsums = (st.unsqueeze(1) * w2.unsqueeze(0)).sum(dim=(4, 5), keepdim=True)
        for j, (region, with_b) in enumerate((("v", True), ("i", False))):
            pooled[region] = self._solve(*[allsums[k, j] for k in range(5)], lam=lam, with_b=with_b)
        a = torch.where(valid, pooled["v"][0], pooled["i"][0])
        b = torch.where(valid, pooled["v"][1], torch.zeros_like(pooled["v"][1]))
        if self.cfg["blend"] == "pixel":
            # per element, ridge toward the region solution with n0 pseudo-windows
            S = self.S
            if self.cfg["blend_smooth"]:
                S = {k: self._smooth(v_) for k, v_ in S.items()}
            lam_p = float(self.cfg["blend_pixel_n0"]) / float(self.n_b + 1)
            ap, bp = self._solve(*(S[k] for k in ("ff", "aa", "fa", "fy", "ay")), lam=lam_p, with_b=True, a0=a, b0=b)
            a = ap
            b = torch.where(valid, bp, torch.zeros_like(bp))
        a[..., 2] = 1.0
        b[..., 2] = 0.0
        return a, b

    # ------------------------------------------------------------------- gain
    def _update_energy(self, e_true: torch.Tensor, e_pred: torch.Tensor) -> None:
        if self.e_true is None:
            self.e_true, self.e_pred = e_true.clone(), e_pred.clone()
            return
        self.n_e += 1
        a = self._a(self.n_e)
        self.e_true = self.e_true + a * (e_true - self.e_true)
        self.e_pred = self.e_pred + a * (e_pred - self.e_pred)

    def _gamma(self) -> torch.Tensor:
        g_max = float(self.cfg["gain_max"])
        ratio = (self.e_true + 1e-8) / (self.e_pred + 1e-8)
        return ratio.pow(0.5 * float(self.cfg["gain_power"])).clamp(1.0 / g_max, g_max)

    def _update_R(self, y: torch.Tensor) -> None:
        """D2/H1 free-rider bundle: causal PLAIN running mean (1/n, not EMA --
        matches `scripts/h1_stage0.py`'s `R_k` candidate exactly) of REVEALED
        target ke maps. Called on the just-revealed `prev_target` before this
        window's own `kx` is available; `self.R_run` is seeded from `kx`
        itself at window 0 (see `step()`), matching `h1_stage0.py`'s
        `M[0] = ke of the raw input window` causal seed. Entirely eager, plain
        Python-int counter -- never called from inside a CUDA graph capture,
        so no scalar-tensor/EMA machinery is needed here (contrast the
        counters cuda_graph_pre.py/cuda_graph_megatail.py DO need, which are
        read INSIDE a captured region)."""
        ky = _ke(y)
        self.n_R += 1
        self.R_run = self.R_run + (ky - self.R_run) / self.n_R

    # ----------------------------------------------- learned / online energy target
    def _g2_base(self, x, fl_f, e_f, kx, mask) -> torch.Tensor:
        """Energy gain E_out / e_f before the online candidate correction, (1, 1, H, W, C).

        gain_map's calibration `(runx/runf)^power` in energy units, times the learned
        log-space correction on the u, v channels. The net's last layer is
        zero-initialised and it is fitted as a residual on the rule, so with the net off
        (or `ke_net_scale = 0`) this is the shipped rule.
        """
        g2 = ((self.e_true + 1e-8) / (self.e_pred + 1e-8)).pow(float(self.cfg["gain_power"]))
        if self.ke_flow is not None:
            # the flow net predicts the energy target directly (rule + residual); express
            # it as a gain on e_f so the clip and both branches downstream are untouched
            e_tgt = self.ke_flow(x, fl_f, kx, e_f, self.e_true, self.e_pred, mask)
            w = self.cfg.get("e_blend_w")
            if w is not None and self.R_run is not None:
                # D2/H1 energy-target blend: e_tgt = w*Khat + (1-w)*R, R the
                # causal running mean of revealed target ke (self.R_run,
                # updated in step()).
                e_tgt = float(w) * e_tgt + (1.0 - float(w)) * self.R_run
            g_flow = (e_tgt[..., :2] + 1e-10) / (e_f[..., :2] + 1e-10)
            if self.ke is None:
                g2 = g2.clone()
                g2[..., :2] = g_flow
                return g2
        if self.ke is None:
            return g2
        wfrac = torch.full((1,), min(self.n_w, 20) / 20.0, device=x.device, dtype=x.dtype)
        feat, ex, ef = K.window_features(x, fl_f, kx, e_f, self.e_true, self.e_pred,
                                         mask.unsqueeze(0), wfrac,
                                         self.ke_ex_prev, self.ke_ef_prev)
        corr = K.predict_log_energy(self.ke, feat, mask.unsqueeze(0),
                                    float(self.cfg["ke_net_scale"]), float(self.cfg["ke_net_clip"]))
        self.ke_ex_prev, self.ke_ef_prev = ex, ef
        if self.ke_steps:
            self.ke_prev = (feat, K.rule_log_energy(ef, self.e_true[:, 0, ..., :2],
                                                    self.e_pred[:, 0, ..., :2],
                                                    float(self.cfg["gain_power"])),
                            mask.unsqueeze(0))
        g2 = g2.clone()
        g2[..., :2] = g2[..., :2] * corr.unsqueeze(1).exp()
        if self.ke_flow is not None:
            # both nets are residuals on the same rule: blend them in log-energy space
            b = float(self.cfg.get("ke_flow_blend", 1.0))
            g2[..., :2] = g2[..., :2].clamp_min(1e-12).log().mul(1.0 - b).add(
                g_flow.clamp_min(1e-12).log().mul(b)).exp()
        return g2

    def _gamma_ext(self, x, fl_f, e_f, kx, mask) -> torch.Tensor:
        """gamma with the learned correction and the online candidate applied.

        E_out = e_f * g2_base * A * (e_f/runf)^(q-1) * (kx/runx)^w; the candidate factor is
        1 for (q, w, A) = (1, 0, 1), which is what `er_warm` and `er_online = False` use.
        """
        g_max = float(self.cfg["gain_max"])
        g2 = self.g2b = self._g2_base(x, fl_f, e_f, kx, mask)
        if self.cfg["er_online"]:
            q, w, a = self.er_cur
            if (q, w, a) != (1.0, 0.0, 1.0):
                g2 = g2 * a * ((e_f + 1e-10) / (self.e_pred + 1e-10)).pow(q - 1.0) \
                            * ((kx + 1e-10) / (self.e_true + 1e-10)).pow(w)
        return g2.clamp(1.0 / g_max ** 2, g_max ** 2).sqrt()

    def _ke_online_update(self, y: torch.Tensor) -> None:
        """`ke_net_online` Adam steps on the window whose target has just been revealed.

        The features of that window are cached, so this is a few forward/backward passes
        of a 5.5k-parameter conv net on one sample -- the same metric loss the offline fit
        minimises, on the single pair the stream just handed us.
        """
        if self.ke_prev is None:
            return
        feat, log_rule, mask = self.ke_prev
        ey = _ke(y)[:, 0, ..., :2]
        with torch.enable_grad():
            for _ in range(self.ke_steps):
                loss = K.metric_loss(self.ke, feat, log_rule, ey, mask,
                                     float(self.cfg["ke_net_scale"]), float(self.cfg["ke_net_clip"]))
                self.ke_opt.zero_grad()
                loss.backward()
                self.ke_opt.step()

    # ------------------------------------------------- online energy-rule selection
    def _e_cands(self, P) -> torch.Tensor:
        """(K, 1, 1, H, W, C) output energy of every candidate for the cached window."""
        g_max = float(self.cfg["gain_max"])
        mod = ((P["kf"] + 1e-10) / (P["rf"] + 1e-10)).pow(self.er_q - 1.0)
        per = ((P["kx"] + 1e-10) / (P["rx"] + 1e-10)).pow(self.er_w)
        g2 = (P["g2b"] * self.er_a * mod * per).clamp(1.0 / g_max ** 2, g_max ** 2)
        return P["kf"] * g2

    def _update_er(self, y: torch.Tensor) -> None:
        """Score every candidate on the window whose target has just been revealed.

        The emitted output was `const + fluc + mu` with `const` constant in t, so a
        candidate that wants energy E instead of the realised var_t(fluc) simply scales
        `fluc` by c = sqrt(E / var_t(fluc)); its temporal variance is then
        c^2 vF + 2 c cFM + vM in closed form, and its field residual is R + (c-1) fluc.
        No extra forward pass and no per-candidate copy of the field.
        """
        P = self.er_prev
        if P is None:
            return
        std2 = (self.std_tgt[:2] ** 2).view(1, 1, 1, 1, 1, 2)
        c = (self._e_cands(P) / P["vF"].clamp_min(1e-12)).clamp_min(0.0).sqrt()   # (K,1,1,H,W,C)
        var = c * c * P["vF"] + 2.0 * c * P["cFM"] + P["vM"]                      # (K,1,1,H,W,C)
        ke_c = 0.5 * (var[..., :2] * std2).sum(-1)                                # (K,1,1,H,W)
        ke_y = 0.5 * (_ke(y)[..., :2] * std2[0]).sum(-1)                          # (1,1,H,W)
        num = (ke_c - ke_y).flatten(1).pow(2).sum(1)
        err = (num / ke_y.pow(2).sum().clamp_min(1e-16)).sqrt()                   # (K,)
        if self.cfg["er_obj"] == "both":
            R = P["out"] - y
            SRR = (R * R).sum(dim=1, keepdim=True)
            SRF = (R * P["fluc"]).sum(dim=1, keepdim=True)
            SFF = (P["fluc"] * P["fluc"]).sum(dim=1, keepdim=True)
            d = c - 1.0
            q = ((SRR + 2.0 * d * SRF + d * d * SFF)[..., :2] * std2).clamp_min(0.0)
            den = ((y[..., :2] * self.std_tgt[:2]) ** 2).sum().clamp_min(1e-16)
            err = err + float(self.cfg["er_lam"]) * (q.flatten(1).sum(1) / den).sqrt()
        self.er_acc = err if self.er_acc is None else self.er_acc + err
        self.er_cur = self.er_cands[int(self.er_acc.argmin())]

    def _shape(self, zc, fl_f, e_z, e_f, c_zf, e_tgt):
        """Fluctuation with per-(pixel, channel) temporal energy e_tgt from the blend's shape zc.

        zc is scaled by up to `shape_cap`; if that does not reach e_tgt (the blend has
        (nearly) zeroed the fluctuation because nothing predicts the phase there) the
        remainder comes from the FNO's own fluctuation fl_f, whose energy is the right
        one: s solves s^2 e_f + 2 s cov(gamma zc, fl_f) = e_tgt - gamma^2 e_z. Where the
        blend is (1, 0) this reduces exactly to gain_map's gamma * fl_f.
        """
        gamma = ((e_tgt + 1e-10) / (e_z + 1e-10)).sqrt().clamp(0.0, float(self.cfg["shape_cap"]))
        d_e = (e_tgt - gamma * gamma * e_z).clamp_min(0.0)
        c = gamma * c_zf
        s = ((c * c + e_f * d_e).clamp_min(0.0).sqrt() - c) / (e_f + 1e-10)
        return gamma * zc + s.clamp(0.0, float(self.cfg["gain_max"])) * fl_f

    # ------------------------------------------------- incoherent-band energy floor
    def _band_floor(self, fluc: torch.Tensor, fl_x: torch.Tensor) -> torch.Tensor:
        """Re-energise the fluctuation's incoherent band to a floor read off the input.

        `idea_4` measured that the whole TKE error lives in DFT modes <= 4 and that the
        prediction is amplitude-deficient there; injecting the missing energy into the
        *incoherent* band instead raises the amplitude at a fraction of the rel_l2 cost,
        because that band's phase is unpredictable anyway and contributes almost nothing
        to the pointwise error. The two bands are exactly orthogonal over the 20 frames,
        so the per-pixel TKE map is the sum of their energies and nothing else changes.
        Deterministic rescaling of the predictor's own high band (the `noise_hi`
        synthesis of idea_4 is worth another +0.1 TKE and is not implemented here).
        """
        kc = int(self.cfg["band_split"])
        F = torch.fft.rfft(fluc, dim=1)
        F[:, :kc + 1] = 0.0                                  # keep only modes > kc
        hi = torch.fft.irfft(F, n=fluc.shape[1], dim=1)
        lo = fluc - hi
        Fx = torch.fft.rfft(fl_x, dim=1)
        Fx[:, :kc + 1] = 0.0
        x_hi = _ke(torch.fft.irfft(Fx, n=fl_x.shape[1], dim=1)) * (self.std_in / self.std_tgt) ** 2
        if self.run_xhi is None:
            self.run_xhi = x_hi.clone()
        else:
            self.run_xhi = self.run_xhi + self._a(self.n_e) * (x_hi - self.run_xhi)
        tgt = self.run_xhi * float(self.cfg["hi_scale"])
        c = ((tgt + 1e-12) / (_ke(hi) + 1e-12)).sqrt().clamp(0.0, float(self.cfg["gain_max"]))
        return lo + c * hi

    # ------------------------------------------------------------ bias/spread
    def _update_error(self, err: torch.Tensor) -> None:
        pool = self.cfg["pool_t"]
        if self.mu is None:
            self.mu = torch.zeros_like(err)
            self.var = _ke(err).clone() if pool else _ke(err).expand_as(err).clone()
            self.prof = torch.ones_like(err.mean(dim=(2, 3), keepdim=True))
        self.n += 1
        a = self._a(self.n)
        d = err - self.mu
        am = a if self.mu_floor is None else torch.clamp(self.mu_floor, min=a)
        self.mu = self.mu + am * d
        d2 = d * d
        self.var = (1.0 - a) * (self.var + a * (d2.mean(dim=1, keepdim=True) if pool else d2))
        if pool == "sep":  # lead-time profile pooled over the field, (1, T, 1, 1, C)
            self.prof = self.prof + a * (d2.mean(dim=(2, 3), keepdim=True) - self.prof)

    def _var(self) -> torch.Tensor:
        if self.cfg["pool_t"] == "sep":
            return self.var * self.prof / self.prof.mean(dim=1, keepdim=True)
        return self.var

    # --------------------------------------------------- window-mean correction
    def _to_tgt(self, x: torch.Tensor) -> torch.Tensor:
        """A window normalised with the *input* statistics, in *target*-normalised units."""
        return (x * self.std_in + self.mean_in - self.mean_tgt) / self.std_tgt

    def _update_mean(self, y: torch.Tensor) -> None:
        """Running statistics of the window mean, for `mean_mode`.

        `rel_l2` on unseen trajectories is set by the window mean plus the bias map
        (`notes/research_2026-09-15/transfer_gap.md` 6.3), so the mean deserves its own
        online model. Two families, both per (pixel, channel):

        * "slope": ridge regression of the revealed window mean on the deviation of the
          FNO/blend window mean from its own running mean. Only the *slope* is applied
          (`(slope - 1) * dev`, zero-mean over windows), so it composes with the bias map
          `mu` instead of fighting it for the level.
        * "persist" / "ls": the input window's own mean is a second candidate predictor
          (the target frames follow the input frames, so in a statistically stationary
          wake it is a strong one). "persist" uses it plus a running offset; "ls" solves
          the per-pixel 2-term ridge least squares between it and the FNO mean, ridged
          toward the FNO mean, so `mean_ridge -> inf` is the shipped behaviour.
        """
        mode = self.cfg["mean_mode"]
        ym = y.mean(dim=1, keepdim=True)
        c = self.prev_c_raw
        cx = self.prev_cx
        if c is None:
            return
        self.n_m += 1
        al = self._a(self.n_m)
        if mode == "both":
            for m in ("slope", "ls"):
                self.cfg["mean_mode"] = m
                self.n_m -= 1
                self._update_mean(y)
            self.cfg["mean_mode"] = "both"
            return
        if mode == "slope":
            if self.Ec is None:
                self.Ec, self.Ey = c.clone(), ym.clone()
                self.Sdd = torch.zeros_like(c)
                self.Sdy = torch.zeros_like(c)
                return
            dc, dy = c - self.Ec, ym - self.Ey
            self.Ec = self.Ec + al * dc
            self.Ey = self.Ey + al * dy
            self.Sdd = (1.0 - al) * (self.Sdd + al * dc * dc)
            self.Sdy = (1.0 - al) * (self.Sdy + al * dc * dy)
            return
        if mode == "persist":
            d = ym - cx
            self.mx_bias = d.clone() if self.mx_bias is None else self.mx_bias + al * (d - self.mx_bias)
            return
        if mode == "ls":
            cur = (c * c, cx * cx, c * cx, c * ym, cx * ym)
            if self.Mcc is None:
                self.Mcc, self.Mxx, self.Mcx, self.Mcy, self.Mxy = (t.clone() for t in cur)
                return
            self.Mcc = self.Mcc + al * (cur[0] - self.Mcc)
            self.Mxx = self.Mxx + al * (cur[1] - self.Mxx)
            self.Mcx = self.Mcx + al * (cur[2] - self.Mcx)
            self.Mcy = self.Mcy + al * (cur[3] - self.Mcy)
            self.Mxy = self.Mxy + al * (cur[4] - self.Mxy)

    def _mean_correct(self, const: torch.Tensor, cx: torch.Tensor) -> torch.Tensor:
        """`const` (constant in t) after the `mean_mode` correction."""
        mode = self.cfg["mean_mode"]
        if mode == "both":   # the two corrections are additive deltas on the same base
            self.cfg["mean_mode"] = "slope"
            out = self._mean_correct(const, cx)
            self.cfg["mean_mode"] = "ls"
            out = out + self._mean_correct(const, cx) - const
            self.cfg["mean_mode"] = "both"
            return out
        r = float(self.cfg["mean_ridge"])
        cl = float(self.cfg["mean_clip"])
        if mode == "slope":
            if self.Sdd is None:
                return const
            slope = (self.Sdy + r * self.Sdd) / (self.Sdd * (1.0 + r) + 1e-20)
            return const + (slope - 1.0).clamp(-cl, cl) * (const - self.Ec)
        if mode == "persist":
            if self.mx_bias is None:
                return const
            return cx + self.mx_bias
        if mode == "ls":
            if self.Mcc is None:
                return const
            p11 = self.Mcc * (1.0 + r)
            p22 = self.Mxx * (1.0 + r)
            r1 = self.Mcy + r * self.Mcc
            r2 = self.Mxy
            det = (p11 * p22 - self.Mcx * self.Mcx).clamp_min(1e-3 * p11 * p22 + 1e-30)
            w1 = (p22 * r1 - self.Mcx * r2) / det
            w2 = (p11 * r2 - self.Mcx * r1) / det
            return w1.clamp(0.0, 1.0 + cl) * const + w2.clamp(-cl, cl) * cx
        return const

    # ------------------------------------------- the alternative centre's statistics
    def _update_ctr(self, y: torch.Tensor) -> None:
        """Running statistics of the centre family `const + g * basis`.

        Writing `a = y - const` and `b = basis`, every member's innovation is
        `a - g b`, so one pair of running means (M0, M1) and the three pooled second
        moments (Saa, Sab, Sbb) give the bias `M0 - g M1` and the pooled variance
        `Saa - 2 g Sab + g^2 Sbb` of *every* g in closed form -- including the online
        least-squares choice `g = Sab / Sbb`.  Three elementwise products and three
        temporal means: ~1 ms.
        """
        a0 = y - self.prev_const
        b0 = self.prev_basis
        # pooled over the 20 leads by default; `ctr_lead` keeps the moments per lead,
        # so the shrink (and the interval width) may differ between lead 1 and lead 20.
        lead = bool(self.cfg["ctr_lead"])
        m = (lambda t: t) if lead else (lambda t: t.mean(dim=1, keepdim=True))
        two = bool(self.cfg["ctr_basis2"])
        c0 = self.prev_basis2 if two else None
        if self.M0 is None:
            self.M0 = torch.zeros_like(a0)
            self.M1 = torch.zeros_like(b0)
            self.Saa = m(_ke(a0).expand_as(a0) if lead else _ke(a0)).clone()
            self.Sab = torch.zeros_like(self.Saa)
            self.Sbb = torch.zeros_like(self.Saa)
        if two and self.M2 is None:
            self.M2 = torch.zeros_like(b0)
            self.S02 = torch.zeros_like(self.Saa)
            self.S12 = torch.zeros_like(self.Saa)
            self.S22 = torch.zeros_like(self.Saa)
        self.n_c += 1
        al = self._a(self.n_c)
        da, db = a0 - self.M0, b0 - self.M1
        self.M0 = self.M0 + al * da
        self.M1 = self.M1 + al * db
        self.Saa = (1.0 - al) * (self.Saa + al * m(da * da))
        self.Sab = (1.0 - al) * (self.Sab + al * m(da * db))
        self.Sbb = (1.0 - al) * (self.Sbb + al * m(db * db))
        if two:
            dc = c0 - self.M2
            self.M2 = self.M2 + al * dc
            self.S02 = (1.0 - al) * (self.S02 + al * m(da * dc))
            self.S12 = (1.0 - al) * (self.S12 + al * m(db * dc))
            self.S22 = (1.0 - al) * (self.S22 + al * m(dc * dc))

    def _ctr_gain(self) -> torch.Tensor:
        """The shrink g applied to the centre's fluctuation.

        A constant, or the running least-squares gain `Sab / Sbb` -- the g that minimises
        the centre's innovation variance, and so (the interval factor being monotone
        decreasing in sigma) the g that maximises SPS.  Ridged toward 1 with the g = 0
        variance Saa as the reference scale, which is exactly 1 before any basis has
        been seen.
        """
        g = self.cfg["ctr_g"]
        if not isinstance(g, str):
            return torch.full((1, 1, 1, 1, 1), float(g),
                              device=self.Saa.device, dtype=self.Saa.dtype)
        r = float(self.cfg["ctr_g_ridge"])
        lo, hi = self.cfg["ctr_g_clip"]
        if g == "auto":                                    # one gain for the whole field
            num = self.Sab.sum() + r * self.Saa.sum()
            den = self.Sbb.sum() + r * self.Saa.sum()
            return (num / den.clamp_min(1e-20)).clamp(float(lo), float(hi)) \
                .reshape(1, 1, 1, 1, 1)
        num = self.Sab + r * self.Saa                      # per (pixel, channel)
        den = self.Sbb + r * self.Saa
        return (num / den.clamp_min(1e-20)).clamp(float(lo), float(hi))

    def _ctr_gain2(self):
        """(g1, g2) of the two-vector centre `const + g1 b1 + g2 b2`, per pixel/channel.

        Same closed form one dimension up: minimising `Saa - 2 g.S0 + g' S g` plus the
        ridge `r Saa (|g1 - 1|^2 + |g2|^2)` -- so `mean_ridge -> inf` gives the shipped
        (g1, g2) = (1, 0) and a 2x2 symmetric solve gives the optimum. b1 is the shipped
        basis (the least-squares blend shape), b2 the advected fluctuation alone, so the
        pair spans the FNO and Taylor directions separately.
        """
        r = float(self.cfg["ctr_g_ridge"])
        lo, hi = self.cfg["ctr_g_clip"]
        p11 = self.Sbb + r * self.Saa
        p22 = self.S22 + r * self.Saa
        r1 = self.Sab + r * self.Saa
        r2 = self.S02
        det = (p11 * p22 - self.S12 * self.S12).clamp_min(1e-3 * p11 * p22 + 1e-30)
        g1 = ((p22 * r1 - self.S12 * r2) / det).clamp(float(lo), float(hi))
        g2 = ((p11 * r2 - self.S12 * r1) / det).clamp(-float(hi), float(hi))
        return g1, g2

    # ------------------------------------------------------------------ prior
    def _apply_prior(self, x: torch.Tensor) -> None:
        """Start the running statistics from the kernel-weighted prior maps of the
        training files nearest in (AoA, log Re) to what the first window says."""
        state = x[0].cpu().numpy()
        re, aoa = float(estimate_re(state)), float(estimate_aoa(state))
        self.params = (re, aoa)
        self._apply_prior_maps(aoa, math.log(max(re, 1.0)))

    def _apply_prior_maps(self, aoa, log_re) -> None:
        """Device part of `_apply_prior`. `aoa` / `log_re`: python floats (eager path) or
        float32 0-dim device tensors holding the same values rounded to float32 (W01's
        window-0 graph) -- `t - float` and `t - f32_tensor` are the same float32
        subtraction (checked: scripts/w01_scalar_check.py, 20k draws, 0 mismatches)."""
        p = self.prior
        d_aoa = (p["aoa"] - aoa) / float(self.cfg["prior_sigma_aoa"])
        d_re = (p["re"].clamp_min(1.0).log() - log_re) / float(self.cfg["prior_sigma_re"])
        w = torch.exp(-0.5 * (d_aoa * d_aoa + d_re * d_re))
        w = w / w.sum().clamp_min(1e-12)
        mix = lambda m: torch.einsum("f,f...->...", w, m).unsqueeze(0)
        self.mu, self.var = mix(p["mu"]), mix(p["var"])
        self.prof = torch.ones_like(self.mu.mean(dim=(2, 3), keepdim=True))
        self.e_true, self.e_pred = mix(p["e_true"]), mix(p["e_pred"])
        n0 = float(self.cfg["prior_windows"])
        n0e = self.cfg["prior_windows_e"]
        n0c = self.cfg["prior_windows_c"]
        self.n = n0
        self.n_e = n0 if n0e is None else float(n0e)
        n0c = n0 if n0c is None else float(n0c)
        if self.cfg["ctr"]:
            # the prior maps are of the *raw FNO* innovation, which is what the
            # un-inflated centre is closer to; seed (M0, Saa) from them and leave the
            # basis moments at zero, so g starts at the ridge's 1.
            var0 = self.var.expand_as(self.mu).clone() if self.cfg["ctr_lead"] else self.var.clone()
            self.M0, self.Saa, self.n_c = self.mu.clone(), var0, n0c
            self.M1 = torch.zeros_like(self.M0)
            self.Sab = torch.zeros_like(self.Saa)
            self.Sbb = torch.zeros_like(self.Saa)
            if self.cfg["ctr_basis2"]:
                self.M2 = torch.zeros_like(self.M0)
                self.S02 = torch.zeros_like(self.Saa)
                self.S12 = torch.zeros_like(self.Saa)
                self.S22 = torch.zeros_like(self.Saa)

    # ------------------------------------------------------------------- step
    def step(self, x: torch.Tensor, prev_target: Optional[torch.Tensor]):
        loss = None
        # c12t (launch-cost, phase 5): `use_mega` uses the SAME gate
        # `self._pre` itself uses (self.model.S is not None, checked here at
        # the TOP of step() -- see cuda_graph_megatail.py's module
        # docstring). When true, the ENTIRE PRE-stage update below (err's
        # EMA bookkeeping) is deferred to `self._mega.forward(...)` after
        # mask/base(x)/scan_uc (a values-preserving reordering: PRE's math
        # has no data dependency on `raw`/`fl_f`/`mf`, the same argument
        # cuda_graph_tail.py already used to reorder energy_gain). `err` and
        # `blend_y` are still computed eagerly here either way (also needed
        # for `loss` and for the `self._pre`/eager fallback branches).
        use_mega = self._mega is not None and self._mega.should_use()
        err = None
        blend_y = None
        if prev_target is not None and self.prev_out is not None:
            err = prev_target - self.prev_out
            loss = self._w01_loss((err ** 2).mean())
            blend_y = prev_target - self.prev_mf
            if use_mega:
                pass  # deferred to self._mega.forward(...) below
            elif self._pre is not None and self._pre.should_use():
                # c10t (launch-cost, phase 4): _update_error + _update_ctr +
                # _update_blend replaced by ONE CUDAGraph replay -- bit-exact
                # transcription, see cuda_graph_pre.py's module docstring for
                # the full argument (gate, EMA-tensor-scalar bit-exactness,
                # scope). `should_use()` is False on window 0 AND window 1 of
                # every trajectory (self.S is None until _update_blend's own
                # seed call, which this graph cannot express -- see
                # docstring), exactly where the ORIGINAL eager code already
                # takes a structurally different branch (the blend seed, or
                # the dead `is None` branches for mu/M0) -- not a new
                # behavioural condition, only a faster implementation of the
                # EMA branch the eager code already selects from window 2 on.
                self._pre.forward(err, prev_target, blend_y, self.prev_fl, self.prev_adv,
                                   self.prev_mask, self.prev_const, self.prev_basis, self.prev_basis2)
            else:
                self._update_error(err)
                if self.cfg["ctr"] and self.prev_const is not None:
                    self._update_ctr(prev_target)
                if self.cfg["advect"]:
                    self._update_blend(self.prev_fl, self.prev_adv, blend_y, self.prev_mask)
            if self.cfg["mean_mode"]:
                self._update_mean(prev_target)
            if self.cfg.get("e_blend_w") is not None:
                self._update_R(prev_target)   # D2/H1: causal running mean of revealed target ke
            if self.cw is not None and self.ctr_out is not None:
                # P3: update the previous-window error-map / EMA state EAGERLY,
                # BEFORE self.ctr_out is reassigned below for the new window --
                # same "compute now, from state as it stood one window ago"
                # pattern as _update_R just above. Physical-unit residual
                # against the centre actually shipped for `prev_target`'s
                # window (self.ctr_out, target-normalised -> physical via
                # std_tgt/mean_tgt, no mean-shift needed for `half`'s own
                # dimensionless multiplier -- see cond_width_net.py).
                #
                # (P4-v2, follow-up pass) from the 2nd update of the
                # trajectory onward (cw.ema_err[0] already non-None), the
                # update math is DEFERRED to the merged update+apply graph,
                # replayed once from ExperimentTTT.ttt_step AFTER step()
                # returns (values-preserving reorder, see cond_width_graph.py's
                # module docstring for the full argument) -- only a PYTHON
                # REFERENCE save here (`self._cwg._ctr_old = self.ctr_out`,
                # zero GPU cost), capturing the value `self.ctr_out` holds
                # RIGHT NOW, before it is reassigned below for the new
                # window. Window 1's FIRST-EVER update (EMA seed, `prev_ema
                # is None`) always stays eager (rare, cheap; the graph does
                # not express that branch) -- and, the moment it completes,
                # arms `self._cwg` for every subsequent window via `seed()`.
                if self._cwg is not None and self._cwg.should_use():
                    self._cwg._ctr_old = self.ctr_out
                    self._cwg._ctr_old_valid = True
                else:
                    for _c in (0, 1):
                        _pt_phys = prev_target[..., _c] * self.std_tgt[_c] + self.mean_tgt[_c]
                        _ctr_phys = self.ctr_out[..., _c] * self.std_tgt[_c] + self.mean_tgt[_c]
                        self.cw.update_channel(_c, _pt_phys, _ctr_phys)
                    if self._cwg is not None:
                        self._cwg._ctr_old_valid = False  # this step took the eager branch
                    if self._cwg is not None and not self._cwg._live and self.cw.ema_err[0] is not None:
                        self._cwg.seed()

        if prev_target is not None:
            if self.ke_steps:
                self._ke_online_update(prev_target)
            if self.cfg["er_online"]:
                self._update_er(prev_target)

        if self.prior is not None and self.params is None:
            self._apply_prior(x)

        T = x.shape[1]
        # c12t (launch-cost, phase 5), task item (d): `_mask` graphed
        # unconditionally (no window gating needed -- see
        # cuda_graph_mask.py's module docstring).
        prev_mask_for_mega = self.prev_mask
        mask = (self._maskg.forward(x) if (self._maskg is not None and self._maskg.should_use())
                else self._mask(x))
        raw = self.base(x)
        self.cb_raw = raw
        mf = raw.mean(dim=1, keepdim=True)
        fl_f = raw - mf

        if use_mega:
            # c12t (launch-cost, phase 5): PRE + energy/ke_net_flow + TAIL,
            # ONE CUDAGraph replay -- bit-exact transcription, see
            # cuda_graph_megatail.py's module docstring for the full
            # argument (gate, scope, EMA-tensor-scalar bit-exactness, why
            # merging beats sharing buffers between separately-captured
            # graphs). `prev_mask_for_mega` = self.prev_mask AS OF THE TOP
            # of this call (before this window's own `mask` was computed
            # above) -- exactly what the eager PRE-stage / self._pre would
            # have read as `self.prev_mask`.
            obstacle = self.shift.last_obstacle(mask)
            mx = x.mean(dim=1, keepdim=True)
            fl_x = x - mx
            if self.cfg["uc_fixed"] is not None:
                self.uc = float(self.cfg["uc_fixed"])
            elif (self.uc is None or self.n_win < int(self.cfg["uc_warm"])
                  or self.n_win % int(self.cfg["uc_every"]) == 0):
                self._update_uc(self._scan_uc_dispatch(fl_x, obstacle, mask))
            adv, const, basis, b2, out, ctr, ctr_var = self._mega.forward(
                x, mask, prev_mask_for_mega, fl_f, mf, err, prev_target, blend_y,
                self.prev_fl, self.prev_adv, self.prev_const, self.prev_basis, self.prev_basis2,
                self.uc, self.uc_eff, self.R_run)
            self.prev_const, self.prev_basis = const.detach(), basis.detach()
            self.prev_basis2 = b2.detach()
            self.ctr_out, self.ctr_var = ctr.detach(), ctr_var
        elif self._tail is not None and self._tail.should_use():
            # c6t2 (launch-cost, phase 2): the graphed tail replaces advect +
            # coefficients-solve + shape/const/fluc + output assembly (incl.
            # bias) + ctr-apply below with ONE CUDAGraph replay -- bit-exact
            # transcription, see cuda_graph_tail.py's module docstring for
            # the full argument (in particular: WHY energy_gain -- computed
            # here, eager, UNCHANGED, just reordered a few lines earlier,
            # which is safe because it has no data dependency on
            # advect/coefficients' output -- is not itself graphed).
            # `should_use()` is False on window 0 of every trajectory (self.S
            # is None there), so the block below only ever runs from window 1
            # onward, exactly where the ORIGINAL branches it replaces
            # (`self.S is not None`, `self.M0/M2 is not None`) are guaranteed
            # true anyway (prior-seeded / set by this step's own PRE-stage
            # update above) -- this is not a new behavioural condition, only
            # a faster implementation of the one the eager branches already
            # select in that regime.
            # convection-speed scan: UNCHANGED, eager, same condition as the
            # original inline block below -- `_advect` (inside the graph)
            # reads self.uc/self.uc_eff, which this call updates in place, so
            # it MUST run before the graph replay, not be skipped by it.
            obstacle = self.shift.last_obstacle(mask)
            mx = x.mean(dim=1, keepdim=True)
            fl_x = x - mx
            if self.cfg["uc_fixed"] is not None:
                self.uc = float(self.cfg["uc_fixed"])
            elif (self.uc is None or self.n_win < int(self.cfg["uc_warm"])
                  or self.n_win % int(self.cfg["uc_every"]) == 0):
                self._update_uc(self._scan_uc_dispatch(fl_x, obstacle, mask))

            kx = _ke(x)
            e_f = _ke(fl_f)
            self._update_energy(kx, e_f)
            self.g2b = None
            gamma = self._gamma_ext(x, fl_f, e_f, kx, mask)
            adv, const, basis, b2, out, ctr, ctr_var = self._tail.forward(
                x, mask, fl_f, mf, e_f, gamma)
            self.prev_const, self.prev_basis = const.detach(), basis.detach()
            self.prev_basis2 = b2.detach()
            self.ctr_out, self.ctr_var = ctr.detach(), ctr_var
        else:
            if self.cfg["advect"]:
                obstacle = self.shift.last_obstacle(mask)
                mx = x.mean(dim=1, keepdim=True)
                fl_x = x - mx
                if self.cfg["uc_fixed"] is not None:
                    self.uc = float(self.cfg["uc_fixed"])
                elif (self.uc is None or self.n_win < int(self.cfg["uc_warm"])
                      or self.n_win % int(self.cfg["uc_every"]) == 0):
                    self._update_uc(self._scan_uc_dispatch(fl_x, obstacle, mask))
                adv, valid = self._advect(fl_x[:, -1], obstacle, mask, T)
                a, b = self._coefficients(valid, mask)
                if self.adv_w is not None:
                    b = b * self.adv_w
                z = a * fl_f + b * adv
            else:
                adv = None
                z = fl_f

            # out = const (constant in t) + fluc (zero temporal mean) + mu; the online energy
            # rule only ever rescales `fluc`, so keeping the three apart makes the candidate
            # scorer exact and free.
            const, fluc = mf, torch.zeros_like(fl_f)
            zc_ls = None                 # the un-inflated least-squares fluctuation, if any
            kx = _ke(x)
            if self.R_run is None:
                # D2/H1: seed the causal running mean R from the input window's own
                # energy at window 0 -- matches h1_stage0.py's causal M[0] exactly
                # (no revealed target exists yet, so the input window IS the
                # history). Unconditional on cfg["e_blend_w"] -- see reset()'s
                # comment for why (keeps the graphed megatail wrapper's R_run
                # input always valid).
                self.R_run = kx.clone()
                self.n_R = 1
            if self.cfg["gain"]:
                # gain_map's energy rule: the output's per-(pixel, channel) temporal energy this
                # window is var_t(F') * gamma^2 with gamma from the running ratio of the input
                # window's energy to the forecast's (the current window included) -- corrected by
                # the learned map and/or the online candidate when either is switched on.
                e_f = _ke(fl_f)
                self._update_energy(kx, e_f)
                self.g2b = None
                if self.ke is None and self.ke_flow is None and not self.cfg["er_online"]:
                    gamma = self._gamma()          # untouched: gain_map_improved bit-for-bit
                else:
                    gamma = self._gamma_ext(x, fl_f, e_f, kx, mask)
                if self.cfg["advect"] and self.S is not None:
                    # the blend supplies the *shape*: rescale it to that energy, per window
                    mz = z.mean(dim=1, keepdim=True)
                    zc = zc_ls = z - mz
                    const = mf + mz
                    fluc = self._shape(zc, fl_f, _ke(zc), e_f, (zc * fl_f).mean(dim=1, keepdim=True), gamma * gamma * e_f)
                else:
                    fluc = gamma * fl_f
            elif self.cfg["advect"]:
                mz = z.mean(dim=1, keepdim=True)
                const, fluc = mf + mz, z - mz
            else:
                const, fluc = mf, raw - mf
            if self.cfg["band_split"] and self.cfg["gain"] and self.cfg["advect"]:
                fluc = self._band_floor(fluc, fl_x)
            if self.cfg["mean_mode"]:
                # the window mean gets its own online model; the raw mean and the input
                # window's mean are cached for the regression the next revealed target feeds
                cx = self._to_tgt(x.mean(dim=1, keepdim=True))
                self.prev_c_raw, self.prev_cx = const.detach(), cx.detach()
                const = self._mean_correct(const, cx)
            mu = None
            if self.cfg["bias"] and self.mu is not None and self.cfg.get("mu_vp", False):
                # D2/H1 variance-preserving mu: mu's own zero-mean-over-t
                # fluctuation otherwise adds UNCOUNTED variance on top of
                # fluc's already-imposed energy target (measured -0.11 tke on
                # the shipped config). Demean mu (mu' = mu - mean_t mu) and
                # rescale fluc by the closed-form c solving
                # var_t(c*fluc + mu') == var_t(fluc) (the already-achieved,
                # imposition-exact target) -- mu itself is added back
                # UNCHANGED (original, not demeaned) below.
                mu = self.mu
                mu_p = mu - mu.mean(dim=1, keepdim=True)
                e_tgt_ach = _ke(fluc)
                vM = _ke(mu_p)
                cFM = (fluc * mu_p).mean(dim=1, keepdim=True)
                disc = (cFM * cFM - e_tgt_ach * vM + e_tgt_ach * e_tgt_ach).clamp_min(0.0)
                c = ((-cFM + disc.sqrt()) / e_tgt_ach.clamp_min(1e-12)).clamp_min(0.0)
                fluc = c * fluc
            elif self.cfg["bias"] and self.mu is not None:
                mu = self.mu
            out = const + fluc
            if mu is not None:
                out = out + mu

            # ---- the decoupled interval centre ------------------------------------
            # `out` is what we return and what the accuracy factors see; `ctr_out` is only
            # ever the middle of the interval.  Its fluctuation is a blend of the shipped
            # (energy-inflated) one and the un-inflated least-squares one, shrunk by g, and
            # its bias and spread come from the centre family's own running statistics.
            self.ctr_out = self.ctr_var = None
            if self.cfg["ctr"]:
                w = float(self.cfg["ctr_w"])
                basis = fluc if (zc_ls is None or w == 0.0) else (1.0 - w) * fluc + w * zc_ls
                self.prev_const, self.prev_basis = const.detach(), basis.detach()
                two = bool(self.cfg["ctr_basis2"])
                b2 = None
                if two:
                    b2 = (torch.zeros_like(basis) if adv is None
                          else adv - adv.mean(dim=1, keepdim=True))
                    self.prev_basis2 = b2.detach()
                if self.M0 is not None:
                    if two and self.M2 is not None:
                        g, g2 = self._ctr_gain2()
                        ctr = const + g * basis + g2 * b2
                        if self.cfg["ctr_bias"]:
                            ctr = ctr + self.M0 - g * self.M1 - g2 * self.M2
                        elif mu is not None:
                            ctr = ctr + mu
                        self.ctr_var = (self.Saa - 2.0 * g * self.Sab - 2.0 * g2 * self.S02
                                        + g * g * self.Sbb + 2.0 * g * g2 * self.S12
                                        + g2 * g2 * self.S22).clamp_min(1e-12)
                        self.ctr_out = ctr.detach()
                    else:
                        g = self._ctr_gain()
                        ctr = const + g * basis
                        if self.cfg["ctr_bias"]:
                            ctr = ctr + self.M0 - g * self.M1
                        elif mu is not None:
                            ctr = ctr + mu
                        self.ctr_var = (self.Saa - 2.0 * g * self.Sab
                                     + g * g * self.Sbb).clamp_min(1e-12)
                        self.ctr_out = ctr.detach()

        if self.cfg["er_online"] and self.cfg["gain"]:
            mu_c = torch.zeros_like(fluc) if mu is None else mu - mu.mean(dim=1, keepdim=True)
            self.er_prev = {
                "kf": e_f, "kx": kx, "rx": self.e_true.clone(), "rf": self.e_pred.clone(),
                "g2b": self.g2b, "vF": _ke(fluc), "vM": _ke(mu_c),
                "cFM": (fluc * mu_c).mean(dim=1, keepdim=True) if mu is not None else torch.zeros_like(e_f),
            }
            if not self.cfg["er_exact"]:
                self.er_prev["vM"] = torch.zeros_like(e_f)
                self.er_prev["cFM"] = torch.zeros_like(e_f)
            if self.cfg["er_obj"] == "both":
                self.er_prev["out"] = out.detach()
                self.er_prev["fluc"] = fluc.detach()

        self.n_w += 1
        self.prev_out = out.detach()
        self.prev_fl, self.prev_adv, self.prev_mf, self.prev_mask = fl_f.detach(), adv, mf.detach(), mask
        self.n_win += 1
        return out.detach(), loss

    def bounds(self, mean: torch.Tensor, x: torch.Tensor,
               var: Optional[torch.Tensor] = None,
               k_scale: Optional[float] = None,
               pred_out: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score-optimal interval around `mean`. `var` / `k_scale` override the
        prediction's own spread, so the decoupled centre can bring its own.

        `pred_out` (P3, only used when `cfg["cond_width"]`): the pipeline's
        OWN returned prediction for this window (`out` in `ttt_step`, target-
        normalised) -- the `pred` feature input `cond_width_net.py`'s causal
        features are defined against (m1_common's own convention: features
        use the shipped `pred`, not necessarily the same tensor as `mean`/
        `ctr_out`, which can be a decoupled centre). Only measured channels
        (u, v) are rescaled; `mean`/predictions are never touched here.
        """
        if var is None:
            var = self._var() if self.var is not None else _ke(x)   # first window: input variance
        std = var.expand_as(mean).sqrt()
        a = 2.0 * std * self.std_tgt / _SIGMA_GLOBAL
        pos = (a.clamp_min(1e-12).log() - self.log_a[0]) / (self.log_a[1] - self.log_a[0])
        k = self.k_tab[pos.round().long().clamp(0, len(self.k_tab) - 1)]
        half = float(self.cfg["k_scale"] if k_scale is None else k_scale) * k * std
        if self.cfg.get("cond_width") and self.cw is not None and pred_out is not None:
            sig_pipe_phys = std[..., :2] * self.std_tgt[:2]
            half = self.cw.apply(half, x, pred_out, sig_pipe_phys,
                                 self.std_in, self.mean_in, self.std_tgt, self.mean_tgt)
        return mean - half, mean + half


class ExperimentTTT:
    """c3b: internal compute device may differ from the harness's `device`.

    `local_eval_with_analytics.Normalizer` (shared infra, not ours to edit) is
    never moved off CPU, so a CUDA-returning `ttt_step` crashes its
    `postprocess_pred` (device-mismatch bug documented in
    experiments/lco_baseline/notes.md). Rather than force the whole pipeline
    onto CPU threads -- expensive on this shared 32-core box under heavy
    contention (orchestrator instruction 2026-09-18) -- compute always runs on
    CUDA when available (`compute_device`, independent of whatever `device`
    string the harness constructed us with), and `ttt_step` converts back to
    the caller's own input device (`in_device`, read off the actual tensor it
    is handed) before returning -- so CPU-harness callers (e.g.
    `scripts/lco_run_dump.py`'s default `--device cpu`) still get CPU tensors
    back, satisfying the Normalizer, while none of the backbone forward or the
    online-statistics tensor ops run on the CPU.
    """

    def __init__(self, base: torch.nn.Module, device: str, cfg: Dict[str, Any]):
        compute_device = "cuda" if torch.cuda.is_available() else device
        self.model = GainMapImprovedModel(base, compute_device, cfg)
        self.device = compute_device
        self.cfg = cfg
        # T1: whole-step CUDA graph (cuda_graph_wholestep.py); None = v2 path.
        self._whole = None
        # W01: window-0 / window-1 CUDA graphs (w01_warmup_graph.py); None = v2 path.
        self._warm = None

    def reset_ttt_state(self) -> None:
        if self._whole is not None:
            self._whole.reset()
        if self._warm is not None:
            self._warm.reset()
        self.model.reset()
        if getattr(self.model, "cb_adapt", None) is not None:
            self.model._cb_d = self.model._cb_wd = self.model._cb_S = None
        # c4r1: also clear the backbone wrapper's own causal (S1, S2, n) state --
        # trajectory-scoped exactly like self.model.reset() above, but a
        # separate object (GainMapImprovedModel.base is the CtxNormWrapper).
        reset_fn = getattr(self.model.base, "reset", None)
        if callable(reset_fn):
            reset_fn()

    @torch.no_grad()
    def ttt_step(self, input_norm, prev_target_norm=None):
        # T1: from window 2 of a trajectory on, the whole step is ONE CUDA
        # graph replay (see cuda_graph_wholestep.py); windows 0/1 and every
        # fallback case run the unchanged v2 code below.
        if (self._whole is not None and prev_target_norm is not None
                and self._whole.should_use()):
            return self._whole.ttt_step(input_norm, prev_target_norm)
        # W01: windows 0 / 1 of a trajectory as one graph replay each (None = not
        # applicable here -> the unchanged code below; the wrapper then stays off for
        # the rest of the trajectory).
        if self._warm is not None and self._warm.phase >= 0:
            r = self._warm.try_step(input_norm, prev_target_norm)
            if r is not None:
                return r
        x_in = torch.as_tensor(input_norm)
        in_device = x_in.device
        x = x_in.to(self.device)
        y = torch.as_tensor(prev_target_norm).to(self.device) if prev_target_norm is not None else None
        cb_ad = getattr(self.model, "cb_adapt", None)
        cb_wsel = None
        if (cb_ad is not None and y is not None and getattr(self.model, "_cb_d", None) is not None
                and self.model.ctr_out is not None):
            cb_wsel = cb_adapt_select(cb_ad, self.model._cb_S, y, self.model.ctr_out,
                                      self.model._cb_d, self.model._cb_wd)
        out, loss = self.model.step(x, y)
        info: Dict[str, Any] = {"adapt_loss": loss}
        if self.cfg["bounds"]:
            m = self.model
            # P4-v2 (was: P3 "cond_width is not folded into the graphed
            # bounds() replay"): from the 2nd window of cond_width state
            # onward, `m._cwg.combined_step` replays ONE merged CUDA graph
            # doing the deferred update (see step()'s cw block above, which
            # only saved `m._cwg._ctr_old` by reference) THEN bounds()'s live
            # branch fused with CondWidthState.apply's REAL branch --
            # bit-exact transcription (values-preserving reorder), see
            # cond_width_graph.py's module docstring. Window 0 (neutral) and
            # window 1 (first-ever update, EMA seed) still take the fully
            # eager `m.bounds(..., pred_out=out)` branch (unchanged from P3).
            cw_on = bool(self.cfg.get("cond_width"))
            if self.cfg["ctr"] and m.ctr_out is not None:
                # c16t (launch-cost, phase 6): ONE CUDAGraph replay in place
                # of bounds()'s live branch -- bit-exact transcription, see
                # cuda_graph_bounds.py's module docstring for the full
                # argument (var/k_scale always non-None here, so bounds()'s
                # own `if var is None` branch is dead code this graph does
                # not need to express).
                if (not cw_on) and m._boundsg is not None and m._boundsg.should_use():
                    info["lower"], info["upper"] = m._boundsg.forward(m.ctr_out, m.ctr_var)
                elif cw_on and m._cwg is not None and m._cwg.combined_should_use():
                    info["lower"], info["upper"] = m._cwg.combined_step(
                        y, m._cwg._ctr_old, m.ctr_out, m.ctr_var, x, out)
                else:
                    info["lower"], info["upper"] = m.bounds(
                        m.ctr_out, x, m.ctr_var, float(self.cfg["ctr_k_scale"]),
                        pred_out=(out if cw_on else None))
            else:
                info["lower"], info["upper"] = m.bounds(out, x, pred_out=(out if cw_on else None))
            cbw = getattr(m, "cb_w", None)
            if cbw is not None and m.ctr_out is not None and "lower" in info:
                # CB: shift the emitted interval toward the raw backbone output
                d = m.cb_raw - m.ctr_out
                if cb_ad is not None and y is not None:
                    # AW: store this window's (d, pre-shift width) for the next window's score.
                    # Window 0 (y is None) stores nothing: windows 0 AND 1 keep the fixed CB
                    # weight; window 1 is the first evidence, used from window 2 on.
                    m._cb_wd = torch.sub(info["upper"][..., :2], info["lower"][..., :2])
                    m._cb_d = d
                    if getattr(m, "_cb_S", None) is None:
                        m._cb_S = cb_ad.S0.clone()
                    if cb_wsel is not None:
                        cbw = cb_wsel
                if y is None and getattr(m, "cb_w0", None) is not None:
                    cbw = m.cb_w0            # BR arm 3: window-0 fallback weight (output only)
                info["lower"] = torch.addcmul(info["lower"], cbw, d)
                info["upper"] = torch.addcmul(info["upper"], cbw, d)
        out = out.to(in_device)
        if "lower" in info:
            info["lower"] = info["lower"].to(in_device)
            info["upper"] = info["upper"].to(in_device)
        return out, info


def get_ttt_model(submission_dir: str, device: str):
    """Entry point. Called once before the stream; not timed.

    c3b: builds the ConvNeXt U-Net (`unet_local`) instead of the FNO. Same
    (B, 20, 32, 64, 3) normalised in / out contract (src/unet_convnext.py
    docstring; verified against `experiments/lco_baseline/fold0/submission.py`'s
    `raw = self.base(x)` call site, which this pipeline is otherwise identical
    to), `cond=False` (built unconditioned, matching how
    `data/checkpoints/c1d/unet_local_fold0.pth` was trained -- see
    experiments/c1d_capacity/notes.md) so no (Re, AoA) params argument needed.

    Compute device: always CUDA when available regardless of `device` (see
    `ExperimentTTT` docstring); the CPU-thread cap below (orchestrator
    instruction) covers whatever CPU-side glue (masking, host<->device
    transfer) remains either way.
    """
    torch.set_num_threads(4)
    torch.manual_seed(int(CONFIG["seed"]))
    compute_device = "cuda" if torch.cuda.is_available() else device
    # c4r1: base = a fresh unet_local, extended + wrapped by CtxNormUNet, then
    # given causal (S1, S2, n) state by CtxNormWrapper -- ckpt is the
    # c4r1_finetune.py payload ({"model_state_dict": ..., "ctx_const": ...}),
    # not a plain state dict, so loaded directly rather than through
    # load_checkpoint_state (which only knows the plain/fp16-packed formats).
    ckpt_path = _resolve_unet_ckpt(submission_dir)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    # c13p: dispatch the backbone CLASS from an explicit "arch" metadata key
    # the packer writes into the checkpoint payload (scripts/ship_simvp_pack_fp16.py)
    # -- absent for every pre-existing checkpoint (this package's own model.pth
    # included, when built from the UNet recipe), so the ORIGINAL 3-line
    # unet_local construction below is byte-for-byte unchanged and still the
    # default path; only a checkpoint explicitly tagged "n1_simvp" takes the
    # new branch. `ctx_unet` ends up as either a `CtxNormUNet` or a
    # `ship_simvp_models.GenericCore` -- both expose the identical
    # `.forward(xs, extra)` contract `CtxNormWrapper` (and the CUDA-graph
    # capture in cuda_graph_ctxnorm.py, which only ever calls
    # `w.ctx_unet(xs, extra)`) already treat as a black box, so nothing below
    # this block needed to change.
    arch = payload.get("arch") if isinstance(payload, dict) else None
    if arch == "n1_simvp":
        raw_core = _SIMVP.N1SimVP()
        ctx_unet = _SIMVP.GenericCore(raw_core)
        ctx_unet.load_state_dict(state)
    elif arch == _B1.ARCH_B5:
        # P8: B1 smaller backbone (SimVP1Stage enc128/hidden192/6 blocks/RF 11),
        # same GenericCore contract as the n1 branch above -> everything below
        # (CtxNormWrapper, every CUDA graph) treats it as the same black box.
        if CONFIG.get("fp16_core"):
            ctx_unet = _B5X2.build_packed_fp16_core([state], CONFIG["fp16_core"])   # FP16X: K=1 pack
        else:
            ctx_unet = _B1.build_b5_core()
            ctx_unet.load_state_dict(state)
    elif arch in (_B5X2.ARCH_B5X2, _B5X2.ARCH_B5X3):   # K3: b5x3 = same packed builder, K = 3
        # EN-S: packed K=2 b5 seed ensemble (bmmtf32 packing, member-mean in standardised
        # space); payload["member_states"] = the two members' b1_finetune.py state dicts.
        # Same GenericCore contract -> CtxNormWrapper / every CUDA graph unchanged.
        ctx_unet = _B5X2.build_b5x2_core(payload["member_states"], fp16_core=CONFIG.get("fp16_core", False))
    else:
        raw_unet = _C1D.build("unet_local")
        ctx_unet = _CTX.CtxNormUNet(raw_unet)     # extends the stem BEFORE loading
        ctx_unet.load_state_dict(state)
    base = _CTX.CtxNormWrapper(ctx_unet.to(compute_device), payload["ctx_const"])
    base = base.to(compute_device)
    if compute_device == "cuda":
        # c6t (launch-cost): replace c5t's O3 dummy-forward warm-up with a
        # CUDA-graph capture of the same call site (`base(x)` inside
        # `GainMapImprovedModel.step`) -- captures accumulate + standardise +
        # UNet forward + un-standardise as ONE graph, replayed every step
        # instead of launched kernel-by-kernel (see cuda_graph_ctxnorm.py for
        # the full correctness argument). The capture's own warm-up already
        # pays the cudnn/CUDA cold-start cost here, at construction (untimed,
        # per docs/track2.md: "Model construction / checkpoint loading not
        # timed"), so it fully subsumes O3's benefit -- no separate dummy
        # forward needed in the success path.
        graphed = GraphedCtxNormWrapper(base)
        if graphed.ok:
            base = graphed
        else:
            # try/except lives inside GraphedCtxNormWrapper.__init__, never
            # in the timed path; on any capture failure (unsupported
            # driver/CUDA graphs, etc.) fall back to c5t's own O3 eager
            # warm-up so the cold-start cost still lands here, untimed, even
            # without the launch-count reduction.
            with torch.no_grad():
                dummy = torch.zeros(1, 20, 32, 64, 3, device=compute_device)
                base(dummy)
                base.reset()
    if CONFIG.get("cond_width") and not CONFIG.get("cond_width_ckpt_path"):
        # P3: single resolution point, submission_dir-aware (mirrors
        # _resolve_unet_ckpt's own call site) -- done once, before
        # GainMapImprovedModel.__init__ loads the net.
        CONFIG["cond_width_ckpt_path"] = _resolve_cond_width_ckpt(submission_dir)
    exp = ExperimentTTT(base, device, CONFIG)
    exp.model.cb_w = None
    if CONFIG.get("cb_blend_w") is not None:
        _w = torch.as_tensor(CONFIG["cb_blend_w"], dtype=torch.float32)
        if _w.dim() == 1:
            _w = torch.stack([_w, _w], dim=-1)
        _w = torch.cat([_w, torch.zeros(_w.shape[0], 1)], dim=-1)      # (20, 3), channel 2 = 0
        exp.model.cb_w = _w.reshape(1, 20, 1, 1, 3).to(exp.device)
    exp.model.cb_w0 = None
    if CONFIG.get("cb_blend_w0") is not None and exp.model.cb_w is not None:
        _w0 = torch.full((20,), float(CONFIG["cb_blend_w0"]), dtype=torch.float32)
        _w0 = torch.stack([_w0, _w0, torch.zeros(20)], dim=-1)          # (20, 3), channel 2 = 0
        exp.model.cb_w0 = _w0.reshape(1, 20, 1, 1, 3).to(exp.device)
    exp.model.cb_adapt = None
    if CONFIG.get("cb_adapt") is not None and exp.model.cb_w is not None:
        _m = exp.model
        exp.model.cb_adapt = CBAdapt(CONFIG["cb_adapt"]["grid"], CONFIG["cb_adapt"]["rho"],
                                     float(CONFIG["cb_blend_w"][0] if not isinstance(CONFIG["cb_blend_w"][0], (list, tuple))
                                           else CONFIG["cb_blend_w"][0][0]),
                                     _m.std_tgt, _m.mean_tgt, _SIGMA_GLOBAL, exp.device)
        _m._cb_d = _m._cb_wd = _m._cb_S = None
    exp.model.kw_adapt = None
    exp.model.eb_adapt = None
    if CONFIG.get("kw_adapt") is not None and exp.model.cb_adapt is not None:
        _c = CONFIG["kw_adapt"]
        exp.model.kw_adapt = KWAdapt(_c["grid"], _c["groups"], _c["rho"], exp.model.std_tgt, _SIGMA_GLOBAL, 20,
                                     exp.device)
    if CONFIG.get("eb_adapt") is not None:
        _c = CONFIG["eb_adapt"]
        exp.model.eb_adapt = EBAdapt(_c["grid"], _c["rho"], exp.model.std_tgt, exp.model.mean_tgt, 32, 64,
                                     exp.device, b_fixed=float(CONFIG["e_blend_w"]))
    exp.model.br_adapt = None
    if CONFIG.get("br_adapt") is not None and exp.model.eb_adapt is not None:
        _c = CONFIG["br_adapt"]
        exp.model.br_adapt = BRAdapt(_c["grid"], _c["rho"], exp.model.eb_adapt, exp.device,
                                     br_fixed=float(CONFIG["blend_ridge"]))
    if compute_device == "cuda":
        # c6t2 (launch-cost, phase 2): GraphedTailWrapper needs real,
        # correctly-shaped, non-None running-statistics state to warm up and
        # capture against (self.S / self.mu / self.M0... are all None right
        # after construction -- reset() just set them so). Run a 2-window
        # DUMMY trajectory through the model's own EAGER path (untimed, model
        # construction / checkpoint loading is explicitly excluded from the
        # Time metric per docs/track2.md) to populate that state with
        # something real and correctly shaped -- window 0 (prev_target=None)
        # takes the same "self.S is None" branch a real trajectory's first
        # window does, and window 1 (a real, non-None prev_target) runs the
        # PRE-stage updates (_update_error/_update_ctr/_update_blend) that
        # set self.S/M0/M1/Saa/... for the first time -- then reset_ttt_state
        # wipes every bit of it (both the pipeline's own state AND the
        # ctx-norm graph's causal accumulator) before the real stream starts,
        # exactly like c6t's own dummy-forward warm-up already does for the
        # UNet region.
        try:
            m = exp.model
            dummy_x = torch.zeros(1, 20, 32, 64, 3, device=compute_device)
            dummy_y = torch.zeros(1, 20, 32, 64, 3, device=compute_device)
            with torch.no_grad():
                m.step(dummy_x, None)
                m.step(dummy_x, dummy_y)
            # c8t (launch-cost, phase 3): GraphedKeFlowWrapper captures
            # _KeFlow.__call__ (ke_net_flow feature build + forward), the
            # largest remaining eager stage. Same 2-window dummy warm-up
            # already leaves m.ke_flow.n == 2 / ex_prev, ef_prev populated
            # (_KeFlow.__call__ runs unconditionally every step(), both dummy
            # windows included) -- nothing extra needed to prime it. Own
            # try/except so a capture failure here can never prevent the tail
            # graph above from being attached (and vice versa).
            try:
                if m.ke_flow is not None:
                    kf_graph = GraphedKeFlowWrapper(m.ke_flow)
                    if kf_graph.ok:
                        m.ke_flow._graph = kf_graph
            except Exception:  # noqa: BLE001 - capture failure -> eager fallback
                pass
            tail = GraphedTailWrapper(m)
            if tail.ok:
                m._tail = tail
            # c10t (launch-cost, phase 4): GraphedPreWrapper (_update_error +
            # _update_ctr + _update_blend) captures against the SAME
            # already-populated dummy-warm-up state as GraphedTailWrapper
            # above (window 1 of that warm-up runs the PRE-stage EMA
            # branch eagerly, leaving self.S/mu/M0/.../prev_const/basis/
            # basis2 exactly as a real trajectory's window 1 would). Own
            # try/except so a capture failure here can never prevent the
            # ctxnorm/tail/keflow graphs above from being attached (and
            # vice versa) -- same independence contract as the keflow block.
            try:
                pre = GraphedPreWrapper(m)
                if pre.ok:
                    m._pre = pre
            except Exception:  # noqa: BLE001 - capture failure -> eager fallback
                pass
            # c12t (launch-cost, phase 5): `GraphedMegaWrapper` captures
            # against the SAME already-populated dummy-warm-up state as
            # `GraphedPreWrapper`/`GraphedTailWrapper`/`GraphedKeFlowWrapper`
            # above (window 1 of that warm-up runs the PRE-stage EMA branch
            # AND the eager energy/ke_net_flow call, leaving self.S/mu/M0/
            # .../e_true/e_pred/ke_flow.ex_prev/ef_prev exactly as a real
            # trajectory's window 1 would). Own try/except so a capture
            # failure here can never prevent (or be prevented by) any other
            # graph in this file -- `self._pre`/`self._tail`/`ke_flow._graph`
            # remain fully independent fallbacks (see step()'s `use_mega`
            # branch: never removes the existing chain, only skips it when
            # this faster path is available).
            try:
                mega = GraphedMegaWrapper(m)
                if mega.ok:
                    m._mega = mega
            except Exception:  # noqa: BLE001 - capture failure -> eager/pre+tail fallback
                pass
            # `GraphedMaskWrapper` has no window-based precondition (pure
            # function of x/std_in/mean_in, all available since __init__) --
            # own try/except for the same independence reason as above.
            try:
                maskg = GraphedMaskWrapper(m)
                if maskg.ok:
                    m._maskg = maskg
            except Exception:  # noqa: BLE001 - capture failure -> eager _mask fallback
                pass
            # c16t (launch-cost, phase 6): `GraphedScanUcWrapper` captures
            # `_scan_uc` -- a pure function of (fl_x, obstacle, mask), no
            # window-based precondition (same reasoning as `GraphedMaskWrapper`
            # above) -- own try/except for the same independence reason as
            # every other graph in this file.
            try:
                scang = GraphedScanUcWrapper(m)
                if scang.ok:
                    m._scang = scang
            except Exception:  # noqa: BLE001 - capture failure -> eager _scan_uc fallback
                pass
            # c16t (launch-cost, phase 6): `GraphedBoundsWrapper` captures
            # `bounds()`'s live branch (called from ExperimentTTT.ttt_step,
            # not step() -- see cuda_graph_bounds.py's module docstring).
            # Captures against `m.ctr_out`/`m.ctr_var`, already populated,
            # correctly-shaped, non-None by window 0 of the SAME dummy
            # warm-up trajectory above (the (Re, AoA) prior seeds them before
            # window 0's own ctr section of `step()` ever runs). Own
            # try/except so a capture failure here can never prevent (or be
            # prevented by) any other graph in this file.
            try:
                boundsg = GraphedBoundsWrapper(m)
                if boundsg.ok:
                    m._boundsg = boundsg
            except Exception:  # noqa: BLE001 - capture failure -> eager bounds() fallback
                pass
            # (P4: launch-cost pass) `GraphedCondWidthWrapper` -- folds
            # cond_width's net forward + causal feature construction into 2
            # CUDA graphs (update_graph/apply_graph). Only relevant when
            # `cfg["cond_width"]` is on (m.cw is not None); own try/except so
            # a capture failure here can never prevent (or be prevented by)
            # any other graph in this file. Needs no specific warm-up state
            # from the dummy trajectory above -- its own `_capture()`
            # synthesizes deterministic dummy data for every buffer
            # (shapes/dtypes/addresses only matter at capture time, see
            # cond_width_graph.py) and is re-seeded for real the first time
            # it is actually used in a real trajectory (`seed()`).
            try:
                if m.cw is not None:
                    cwg = GraphedCondWidthWrapper(m)
                    if cwg.ok:
                        m._cwg = cwg
            except Exception:  # noqa: BLE001 - capture failure -> eager cond_width fallback
                pass
            # T1 (2026-10-01): ONE whole-step CUDA graph composed from the
            # capture bodies of the wrappers above (see
            # cuda_graph_wholestep.py). Requires every composed wrapper to
            # have captured; own try/except, never in the timed path.
            try:
                whole = GraphedWholeStepWrapper(exp)
                if whole.ok:
                    exp._whole = whole
            except Exception:  # noqa: BLE001 - capture failure -> v2 path
                pass
            # W01 (2026-09-22): window-0 / window-1 graphs, captured from the unchanged
            # eager code of those windows (w01_warmup_graph.py). Own try/except; the
            # wrapper self-tests graph == eager on a dummy trajectory before enabling.
            try:
                warm = GraphedWarmupWrapper(exp)
                if warm.ok0:
                    exp._warm = warm
            except Exception:  # noqa: BLE001 - capture failure -> v2 path
                pass
        except Exception:  # noqa: BLE001 - any warm-up/capture failure -> eager fallback
            pass
        finally:
            exp.reset_ttt_state()
        # P7 stack (KD-R finding, `experiments/kd_review/report.md` -- a CUDA-graph
        # capture failure that happens INSIDE `torch.cuda.graph` can poison the
        # process' CUDA RNG for the rest of the run, so every later `ttt_step`
        # raises even though construction itself "succeeded" via the try/except
        # fallbacks above). Construction-end probe: after ALL captures, a single
        # cheap device randn, own try/except, result logged as a diagnostic
        # attribute on the model only -- never touches `info`/outputs.
        try:
            torch.randn(1, device=compute_device)
            m._p7_rng_probe_ok = True
        except Exception as e:  # noqa: BLE001
            m._p7_rng_probe_ok = False
            m._p7_rng_probe_err = repr(e)
    # c2 IPW (2026-09-25): out-of-process warm-up, untimed. When this process is a multiprocessing child using torch's
    # file-descriptor sharing (the harness's worker), the FIRST tensor it sends back pays a one-time ~100 ms (starting
    # multiprocessing's resource_sharer listener + first shared-memory / fd handling) inside the first timed step. Pickle one
    # tiny tensor with torch's ForkingPickler here instead (share_memory_ + DupFd registration starts the listener; nothing
    # is sent, no connection is made). Values / outputs are untouched; any failure is ignored.
    if CONFIG.get("ipc_prewarm", False):
        try:
            import multiprocessing as _mpw
            import torch.multiprocessing as _tmpw
            from multiprocessing.reduction import ForkingPickler as _FPw
            if _mpw.parent_process() is not None and _tmpw.get_sharing_strategy() == "file_descriptor":
                _w = _FPw.dumps(torch.zeros(8))
                del _w
        except Exception:  # noqa: BLE001 - warm-up only
            pass
    return exp
