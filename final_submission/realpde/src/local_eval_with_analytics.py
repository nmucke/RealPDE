#!/usr/bin/env python3
"""local_eval.py plus diagnostic plots.

Same streaming protocol and the same five subscores as the kit's local_eval.py
(whose data pipeline it imports rather than re-implements, so the numbers stay
identical), with a per-window record kept along the way and written out as
figures and metrics.json.

The plots answer the questions the five scalars cannot:

  01_stream_overview   errors, adapt_loss and per-step time along the stream,
                       with trajectory boundaries -- is adaptation helping or
                       drifting, and where does Time go?
  02_lead_time_error   error vs frame within the 20-frame window -- how fast the
                       autoregressive rollout degrades.
  03_spatial_error     mean |error| per grid point for u and v, unscored points
                       (target == 0: outside the PIV field of view or inside the
                       airfoil) masked out the way the scorer masks them.
  04_field_snapshot    target / prediction / signed error for one window.
  05_tke_maps          the tke_score ingredient: per-point kinetic energy of the
                       prediction against the target.
  06_mvpe_profiles     the mvpe_score ingredient: time-averaged u, v at the four
                       wake probe stations, predicted against measured.
  07_adaptation        per-window error distribution, and error against position
                       within the trajectory -- the TTT learning curve.
  08_sps_bounds        interval width in sigma units, coverage, and the three
                       SPS branches -- participant bounds or the scorer default.

Usage:
    python src/local_eval_with_analytics.py --submission experiments/<name> \\
        --data starting_kit/example_data [--outdir DIR] [--snapshot worst]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import local_eval as kit  # noqa: E402  (kit data pipeline: build_stream, Normalizer, ...)
import scoring as official  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

# --------------------------------------------------------------------------- #
# Palette: categorical slots 1-3, one-hue sequential ramp, blue<->red diverging
# with a neutral gray midpoint. Text keeps ink colors; grid and reference lines
# stay recessive.
# --------------------------------------------------------------------------- #
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
RED = "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
SURFACE, GRID = "#fcfcfb", "#e5e4e0"
SEQ = LinearSegmentedColormap.from_list(
    "rp_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "rp_div", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#7d1a1a"])

SUBSCORES = ["rel_l2_score", "tke_score", "mvpe_score", "time_score", "sps_score"]


def use_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True,
        "axes.axisbelow": True,
        "legend.frameon": False, "lines.linewidth": 2.0,
        "figure.titlesize": 11, "axes.titlesize": 9.5,
    })


def finish(fig, path: Path, dpi: int) -> Path:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Streaming run -- the kit loop, with a per-window record kept.
# --------------------------------------------------------------------------- #
def run_stream(submission_dir: Path, data_dir: Path, device: str) -> dict:
    stats_path = data_dir / "mean_std_real.pt"
    if not stats_path.exists():
        raise SystemExit(f"Missing {stats_path}.")

    stream = kit.build_stream(data_dir)
    normalizer = kit.Normalizer(stats_path)
    module = kit.import_submission(submission_dir)
    model = module.get_ttt_model(str(submission_dir), device)
    for attr in ("reset_ttt_state", "ttt_step"):
        if not hasattr(model, attr):
            raise SystemExit(f"get_ttt_model() returned an object lacking {attr}().")

    n_traj = len({s["sim_id"] for s in stream})
    print(f"[eval] {len(stream)} steps over {n_traj} trajectories, batch size 1")

    times, preds, tgts, lowers, uppers = [], [], [], [], []
    losses, sim_ids, time_ids, firsts = [], [], [], []
    bounds_active, prev_pair, expected = None, None, None

    for step in stream:
        inp = step["input"].unsqueeze(0)
        tgt = step["target"].unsqueeze(0)

        reset_elapsed = 0.0
        if step["is_first"]:
            t_reset = perf_counter()
            model.reset_ttt_state()
            reset_elapsed = perf_counter() - t_reset
            prev_pair = None

        inp_norm, tgt_norm = normalizer.preprocess(inp, tgt)
        prev_target = prev_pair[1] if prev_pair is not None else None

        t0 = perf_counter()
        pred_norm, info = model.ttt_step(inp_norm, prev_target)
        times.append(perf_counter() - t0 + reset_elapsed)

        prev_pair = (inp_norm.detach(), tgt_norm.detach())  # cached AFTER timing

        pred_norm = torch.as_tensor(pred_norm)
        expected = expected or tuple(tgt_norm.shape)
        if tuple(pred_norm.shape) != expected:
            raise SystemExit(f"ttt_step returned shape {tuple(pred_norm.shape)}, "
                             f"expected {expected}.")
        if not isinstance(info, dict) or "adapt_loss" not in info:
            raise SystemExit('ttt_step must return (pred, info) with info["adapt_loss"].')

        has_lower, has_upper = "lower" in info, "upper" in info
        if has_lower != has_upper:
            raise SystemExit("ttt_step returned only one of info['lower'] / "
                             "info['upper']; bounds are all-or-nothing.")
        if has_lower and (info["lower"] is None or info["upper"] is None):
            raise SystemExit("info['lower'] / info['upper'] set to None; leave the "
                             "keys out entirely instead.")
        if bounds_active is None:
            bounds_active = has_lower
        elif has_lower != bounds_active:
            raise SystemExit("ttt_step started or stopped returning SPS bounds "
                             "mid-run; supply them on every step or on none.")

        preds.append(normalizer.postprocess_pred(pred_norm.detach())
                     .squeeze(0).cpu().numpy().astype(np.float32))
        tgts.append(tgt.squeeze(0).cpu().numpy().astype(np.float32))
        if has_lower:
            lo, up = torch.as_tensor(info["lower"]), torch.as_tensor(info["upper"])
            if tuple(lo.shape) != expected or tuple(up.shape) != expected:
                raise SystemExit("SPS bounds must match the prediction shape.")
            lowers.append(normalizer.postprocess_pred(lo.detach())
                          .squeeze(0).cpu().numpy().astype(np.float32))
            uppers.append(normalizer.postprocess_pred(up.detach())
                          .squeeze(0).cpu().numpy().astype(np.float32))

        loss = info.get("adapt_loss")
        losses.append(float(loss) if loss is not None else np.nan)
        sim_ids.append(step["sim_id"])
        time_ids.append(step["time_id"])
        firsts.append(bool(step["is_first"]))

    return {
        "pred": np.stack(preds), "target": np.stack(tgts),
        "lower": np.stack(lowers) if bounds_active else None,
        "upper": np.stack(uppers) if bounds_active else None,
        "times": np.asarray(times), "adapt_loss": np.asarray(losses),
        "sim_ids": sim_ids, "time_ids": time_ids, "is_first": np.asarray(firsts),
        "bounds_active": bool(bounds_active),
    }


def save_run(run: dict, path: Path) -> None:
    arrays = {k: run[k] for k in ("pred", "target", "times", "adapt_loss", "is_first")
              if run[k] is not None}
    if run["bounds_active"]:
        arrays["lower"], arrays["upper"] = run["lower"], run["upper"]
    arrays["sim_ids"] = np.array(run["sim_ids"])
    arrays["time_ids"] = np.array(run["time_ids"])
    np.savez_compressed(path, **arrays)


def load_run(path: Path) -> dict:
    """Replot from a saved run without re-running the (slow, timed) stream."""
    z = np.load(path, allow_pickle=False)
    has_bounds = "lower" in z.files
    return {
        "pred": z["pred"], "target": z["target"],
        "lower": z["lower"] if has_bounds else None,
        "upper": z["upper"] if has_bounds else None,
        "times": z["times"], "adapt_loss": z["adapt_loss"],
        "sim_ids": [str(v) for v in z["sim_ids"]],
        "time_ids": [int(v) for v in z["time_ids"]],
        "is_first": z["is_first"], "bounds_active": has_bounds,
    }


def compute_metrics(run: dict) -> dict:
    pred, tgt = run["pred"], run["target"]
    c = official.measured_channels(tgt)
    rel = official.rel_l2_per_sample(pred, tgt, c)
    tke = official.tke_rel_l2_per_sample(pred, tgt, c)
    mvpe = official.mvpe_rel_l2_per_sample(pred, tgt)
    mean_t = float(np.mean(run["times"]))
    sps, coverage = official.aggregate_sps(pred, tgt, c,
                                           lower=run["lower"], upper=run["upper"])
    return {
        "channels": int(c), "per_window": {"rel_l2": rel, "tke": tke, "mvpe": mvpe},
        "mean_step_s": mean_t, "coverage": coverage,
        "subscores": {
            "rel_l2_score": official.score_error(float(np.mean(rel))),
            "tke_score": official.score_error(float(np.mean(tke))),
            "mvpe_score": official.score_error(float(np.mean(mvpe))),
            "time_score": official.score_time(mean_t),
            "sps_score": official.score_sps(sps),
        },
    }


# --------------------------------------------------------------------------- #
# Shared helpers for the figures.
# --------------------------------------------------------------------------- #
def traj_starts(run: dict) -> list[int]:
    return [i for i, f in enumerate(run["is_first"]) if f]


def mark_trajectories(ax, run: dict, label: bool = False) -> None:
    for i in traj_starts(run)[1:]:
        ax.axvline(i - 0.5, color=MUTED, lw=0.8, ls=":", zorder=0)
    starts = traj_starts(run)
    if label and len(starts) > 1:
        for i in starts:
            ax.annotate(run["sim_ids"][i].replace(".h5", ""), (i, 0.98),
                        xycoords=("data", "axes fraction"), color=MUTED,
                        fontsize=7, ha="left", va="top")


def scored_mask(tgt: np.ndarray, c: int) -> np.ndarray:
    """The scorer ignores target == 0: outside the PIV window or inside the body."""
    return tgt[..., :c] != 0.0


def field_extent(h: int, w: int) -> tuple:
    return (0, w, h, 0)


def imshow(ax, data, cmap=SEQ, *, vmin=None, vmax=None, title="", cbar_label=""):
    h, w = data.shape
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper",
                   extent=field_extent(h, w), interpolation="nearest", aspect="equal")
    ax.set_title(title)
    ax.grid(False)
    ax.set_xticks([]), ax.set_yticks([])
    cax = make_axes_locatable(ax).append_axes("right", size="3.5%", pad=0.08)
    cb = ax.figure.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7, color=GRID)
    cb.outline.set_edgecolor(GRID)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=7, color=INK2)
    return im


# --------------------------------------------------------------------------- #
# Figures.
# --------------------------------------------------------------------------- #
def plot_stream_overview(run, met, out: Path, dpi: int) -> Path:
    pw = met["per_window"]
    x = np.arange(len(pw["rel_l2"]))
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)

    ax = axes[0]
    for series, color, name in ((pw["rel_l2"], C1, "rel L2"),
                                (pw["tke"], C2, "TKE rel L2"),
                                (pw["mvpe"], C3, "MVPE rel L2")):
        ax.plot(x, series, color=color, label=name)
    span = np.concatenate([pw["rel_l2"], pw["tke"], pw["mvpe"]])
    span = span[span > 0]
    if span.size and span.max() / span.min() > 20:
        ax.set_yscale("log")            # one axis, log -- never a second y-scale
        ax.set_ylabel("relative error (log)")
    else:
        ax.set_ylabel("relative error")
    ax.set_title("Per-window error along the stream (lower is better)", loc="left")
    ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0), borderaxespad=0)
    mark_trajectories(ax, run, label=True)

    ax = axes[1]
    ax.plot(x, run["adapt_loss"], color=C1, label="adapt_loss")
    if np.all(np.isnan(run["adapt_loss"])):
        ax.annotate("no adaptation reported (adapt_loss was None every step)",
                    (0.5, 0.5), xycoords="axes fraction", ha="center", color=MUTED)
    ax.set_ylabel("adapt_loss")
    ax.set_title("Adaptation loss on the revealed previous pair", loc="left")
    mark_trajectories(ax, run)

    ax = axes[2]
    ms = run["times"] * 1e3
    ax.plot(x, ms, color=C1, label="per-step time")
    ax.axhline(ms.mean(), color=MUTED, lw=1.2, ls="--")
    ax.annotate(f"mean {ms.mean():.0f} ms", (x[0], ms.mean()), color=INK2,
                fontsize=8, ha="left", va="bottom")
    ref = official.T_NUMERICAL_SEC * 1e3
    if ms.max() > ref * 0.2:
        ax.axhline(ref, color=RED, lw=1.2, ls="--")
        ax.annotate(f"t_numerical {ref:.0f} ms (r = 1)", (x[-1], ref), color=RED,
                    fontsize=8, ha="right", va="bottom")
    ax.set_ylabel("ms")
    ax.set_xlabel("stream step")
    ax.set_title("Per-step wall time (resets charged to the following step)", loc="left")
    mark_trajectories(ax, run)

    fig.suptitle("Stream overview", x=0.005, ha="left", fontweight="bold")
    fig.tight_layout()
    return finish(fig, out / "01_stream_overview.png", dpi)


def plot_lead_time(run, met, out: Path, dpi: int) -> Path:
    pred, tgt, c = run["pred"], run["target"], met["channels"]
    n, t = pred.shape[0], pred.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    names = ["u", "v", "p"]

    ax = axes[0]
    for ch, color in zip(range(c), (C1, C2, C3)):
        p = pred[..., ch].reshape(n, t, -1)
        g = tgt[..., ch].reshape(n, t, -1)
        err = (np.linalg.norm(p - g, axis=2)
               / np.linalg.norm(g, axis=2).clip(min=1e-8))          # (N, T)
        lo, mid, hi = np.percentile(err, [25, 50, 75], axis=0)
        frames = np.arange(1, t + 1)
        ax.fill_between(frames, lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(frames, err.mean(axis=0), color=color, label=f"{names[ch]} (mean)")
        ax.plot(frames, mid, color=color, lw=1.0, ls="--")
    ax.set_xlabel("frame within the predicted window (lead time)")
    ax.set_ylabel("relative error")
    ax.set_title("Error vs lead time -- band: IQR over windows, dashed: median",
                 loc="left")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper left")

    ax = axes[1]
    mask = scored_mask(tgt, c)
    abs_err = np.abs(pred[..., :c] - tgt[..., :c])
    per_frame = np.array([abs_err[:, i][mask[:, i]].mean() for i in range(t)])
    ax.plot(np.arange(1, t + 1), per_frame, color=C1)
    growth = per_frame[-1] / max(per_frame[0], 1e-12)
    ax.annotate(f"x{growth:.2f} from first to last frame", (0.97, 0.95),
                xycoords="axes fraction", color=INK2, fontsize=8,
                ha="right", va="top")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("frame within the predicted window")
    ax.set_ylabel("mean |error| (scored points)")
    ax.set_title("Absolute error growth over the rollout", loc="left")

    fig.suptitle("Lead-time degradation", x=0.005, ha="left", fontweight="bold")
    fig.tight_layout()
    return finish(fig, out / "02_lead_time_error.png", dpi)


def plot_spatial_error(run, met, out: Path, dpi: int) -> Path:
    pred, tgt, c = run["pred"], run["target"], met["channels"]
    mask = scored_mask(tgt, c)
    names = ["u", "v", "p"]
    fig, axes = plt.subplots(1, c + 1, figsize=(4.4 * (c + 1), 2.7))
    axes = np.atleast_1d(axes)

    for ch in range(c):
        err = np.abs(pred[..., ch] - tgt[..., ch])
        m = mask[..., ch]
        count = m.sum(axis=(0, 1))
        total = np.where(m, err, 0.0).sum(axis=(0, 1))
        field = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        imshow(axes[ch], field, SEQ, title=f"mean |error| -- {names[ch]}",
               cbar_label="raw units")

    m_all = mask.any(axis=-1)
    seen = m_all.mean(axis=(0, 1))
    imshow(axes[c], seen, SEQ, vmin=0, vmax=1,
           title="fraction of frames scored (target != 0)", cbar_label="fraction")

    fig.suptitle("Where the error lives (white = never scored: outside the PIV "
                 "window or inside the airfoil)", x=0.005, ha="left", fontweight="bold")
    fig.tight_layout()
    return finish(fig, out / "03_spatial_error.png", dpi)


def pick_window(met, choice: str) -> int:
    rel = met["per_window"]["rel_l2"]
    if choice == "worst":
        return int(np.argmax(rel))
    if choice == "best":
        return int(np.argmin(rel))
    if choice == "median":
        return int(np.argsort(rel)[len(rel) // 2])
    idx = int(choice)
    return idx if idx >= 0 else len(rel) + idx


def plot_field_snapshot(run, met, out: Path, dpi: int, window: int, frame: int) -> Path:
    pred, tgt, c = run["pred"], run["target"], met["channels"]
    frame = frame if frame >= 0 else pred.shape[1] + frame
    names = ["u", "v", "p"]
    fig, axes = plt.subplots(c, 3, figsize=(13.2, 2.9 * c), squeeze=False)

    for ch in range(c):
        t_field = tgt[window, frame, ..., ch]
        p_field = pred[window, frame, ..., ch]
        both = np.concatenate([t_field.ravel(), p_field.ravel()])
        lo, hi = float(both.min()), float(both.max())
        # Diverging only when the field genuinely straddles zero; a single-signed
        # field reads as magnitude and gets the one-hue ramp.
        if lo < 0 < hi and min(abs(lo), hi) > 0.15 * max(abs(lo), hi):
            lim = max(abs(lo), hi) or 1.0
            kw = dict(cmap=DIV, vmin=-lim, vmax=lim)
        else:
            kw = dict(cmap=SEQ, vmin=lo, vmax=hi)
        imshow(axes[ch][0], t_field, title=f"target {names[ch]}", **kw)
        imshow(axes[ch][1], p_field, title=f"prediction {names[ch]}", **kw)
        diff = np.where(t_field != 0.0, p_field - t_field, np.nan)  # unscored -> blank
        dlim = float(np.nanmax(np.abs(diff))) or 1.0
        imshow(axes[ch][2], diff, DIV, vmin=-dlim, vmax=dlim,
               title=f"prediction - target ({names[ch]}), max |err| {dlim:.3g}")

    rel = met["per_window"]["rel_l2"][window]
    fig.suptitle(f"Window {window} ({run['sim_ids'][window].replace('.h5','')}, "
                 f"t0 = {run['time_ids'][window]}), frame {frame + 1}/{pred.shape[1]} "
                 f"-- window rel L2 {rel:.4f}",
                 x=0.005, ha="left", fontweight="bold")
    fig.tight_layout()
    return finish(fig, out / "04_field_snapshot.png", dpi)


def plot_tke_maps(run, met, out: Path, dpi: int, window: int) -> Path:
    pred, tgt, c = run["pred"], run["target"], met["channels"]
    ke_p = official.kinetic_energy(pred[window:window + 1, ..., :c])[0]
    ke_t = official.kinetic_energy(tgt[window:window + 1, ..., :c])[0]
    vmax = float(max(ke_t.max(), ke_p.max())) or 1.0
    diff = ke_p - ke_t
    dlim = float(np.abs(diff).max()) or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 2.7))
    imshow(axes[0], ke_t, SEQ, vmin=0, vmax=vmax, title="target TKE")
    imshow(axes[1], ke_p, SEQ, vmin=0, vmax=vmax, title="predicted TKE")
    imshow(axes[2], diff, DIV, vmin=-dlim, vmax=dlim, title="predicted - target")
    fig.suptitle(f"Turbulent kinetic energy, window {window} "
                 f"-- window TKE rel L2 {met['per_window']['tke'][window]:.4f}",
                 x=0.005, ha="left", fontweight="bold")
    fig.tight_layout()
    return finish(fig, out / "05_tke_maps.png", dpi)


def probe_geometry(h: int, w: int, sub_s: int = 2):
    """The probe grid scoring.py's MVPE uses: 4 streamwise stations x 9 rows."""
    d, center_x, center_y, n_probe = 16, 10, 32, 9
    probe_center_y = int(center_y / sub_s)
    interval_y = min(2, int(h / (n_probe + 1)))
    probe_y = [probe_center_y + interval_y * j
               for j in range(-(n_probe - 1) // 2, n_probe - (n_probe - 1) // 2)]
    probe_y = [y for y in probe_y if 0 <= y < h]
    xs = []
    for i in range(4):
        if int((2 * d + center_x) / sub_s) < w:
            probe_x = int(((i + 1) * d + center_x) / sub_s)
        else:
            probe_x = int((0.5 * (i + 2) * d + center_x) / sub_s)
        if 0 <= probe_x < w:
            xs.append(probe_x)
    return probe_y, xs


def plot_mvpe_profiles(run, met, out: Path, dpi: int) -> Path:
    pred, tgt = run["pred"], run["target"]
    h, w = pred.shape[2], pred.shape[3]
    probe_y, probe_x = probe_geometry(h, w)
    if not probe_y or not probe_x:
        return None
    fig, axes = plt.subplots(2, len(probe_x), figsize=(3.1 * len(probe_x), 6.4),
                             squeeze=False, sharey=True)
    for col, px in enumerate(probe_x):
        for row, name in enumerate(("u", "v")):
            ax = axes[row][col]
            p = pred[:, :, probe_y, px, row].mean(axis=1).mean(axis=0)
            t = tgt[:, :, probe_y, px, row].mean(axis=1).mean(axis=0)
            ax.plot(t, probe_y, color=MUTED, lw=2.5, label="measured")
            ax.plot(p, probe_y, color=C1, lw=2.0, label="predicted")
            ax.invert_yaxis()
            ax.set_title(f"x = {px}   ({name})")
            if col == 0:
                ax.set_ylabel("probe row (grid y)")
            ax.set_xlabel(f"time-averaged {name}")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis="x", labelsize=7)
    axes[0][0].legend(loc="best")
    fig.suptitle("Mean velocity profiles at the wake probe stations, averaged over "
                 f"the stream -- MVPE rel L2 {np.mean(met['per_window']['mvpe']):.4f}",
                 x=0.005, ha="left", fontweight="bold")
    fig.tight_layout()
    return finish(fig, out / "06_mvpe_profiles.png", dpi)


def plot_adaptation(run, met, out: Path, dpi: int) -> Path:
    rel = met["per_window"]["rel_l2"]
    starts = traj_starts(run) + [len(rel)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    ax = axes[0]
    ax.hist(rel, bins=min(30, max(5, len(rel) // 2)), color=C1, alpha=0.85)
    ax.axvline(float(np.mean(rel)), color=MUTED, ls="--", lw=1.2)
    ax.annotate(f"mean {np.mean(rel):.4f}", (float(np.mean(rel)), 0.95),
                xycoords=("data", "axes fraction"), color=INK2, fontsize=8, ha="left")
    ax.set_xlabel("per-window rel L2")
    ax.set_ylabel("windows")
    ax.set_title("Error distribution over the stream", loc="left")
    ax.grid(axis="x")

    ax = axes[1]
    longest = 0
    for a, b in zip(starts[:-1], starts[1:]):
        seg = rel[a:b]
        longest = max(longest, len(seg))
        ax.plot(np.arange(len(seg)), seg, color=MUTED, lw=1.0, alpha=0.7)
    curve = np.full(longest, np.nan)
    for k in range(longest):
        vals = [rel[a + k] for a, b in zip(starts[:-1], starts[1:]) if a + k < b]
        curve[k] = np.mean(vals) if vals else np.nan
    n_traj = len(starts) - 1
    ax.plot(np.arange(longest), curve, color=C1,
            label="mean over trajectories" if n_traj > 1 else "trajectory")
    ax.set_xlabel("step within trajectory (0 = first window after reset)")
    ax.set_ylabel("rel L2")
    ax.set_title("Does adaptation improve with more revealed targets?", loc="left")
    ax.legend(loc="best")

    fig.suptitle("Adaptation behaviour", x=0.005, ha="left", fontweight="bold")
    fig.tight_layout()
    return finish(fig, out / "07_adaptation.png", dpi)


def plot_sps(run, met, out: Path, dpi: int) -> Path:
    pred, tgt, c = run["pred"], run["target"], met["channels"]
    p, t = pred[..., :c], tgt[..., :c]
    if run["bounds_active"]:
        lower, upper = run["lower"][..., :c], run["upper"][..., :c]
        source = "participant bounds (info['lower'] / info['upper'])"
    else:
        interval = 0.1 * np.abs(p)
        lower, upper = p - interval / 2.0, p + interval / 2.0
        source = "scorer default band (no bounds returned): 0.1 * |pred|"

    sigma = official.SIGMA_GLOBAL
    scored = t != 0.0
    inside = (t >= lower) & (t <= upper)
    nil = (upper - lower) / sigma

    per_window_cov = np.array([
        inside[i][scored[i]].mean() if scored[i].any() else np.nan
        for i in range(p.shape[0])])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]
    ax.plot(np.arange(len(per_window_cov)), per_window_cov * 100, color=C1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("stream step")
    ax.set_ylabel("coverage (%)")
    ax.set_title("Per-window coverage of the true field", loc="left")
    mark_trajectories(ax, run)

    ax = axes[1]
    widths = nil[scored]
    ax.hist(widths, bins=40, color=C1, alpha=0.85)
    ax.axvline(float(widths.mean()), color=MUTED, ls="--", lw=1.2)
    ax.annotate(f"mean {widths.mean():.2f} sigma\nreward factor "
                f"exp(-width) = {np.exp(-widths.mean()):.3f}",
                (0.97, 0.9), xycoords="axes fraction", ha="right", va="top",
                color=INK2, fontsize=8)
    ax.set_xlabel("interval width / sigma")
    ax.set_ylabel("elements")
    ax.set_title("Interval width -- SPS pays exp(-width) when covered", loc="left")
    ax.grid(axis="x")

    ax = axes[2]
    rng = np.random.default_rng(0)
    flat_w, flat_e = nil[scored], np.abs(p - t)[scored]
    flat_in = inside[scored]
    if flat_w.size > 20000:
        sel = rng.choice(flat_w.size, 20000, replace=False)
        flat_w, flat_e, flat_in = flat_w[sel], flat_e[sel], flat_in[sel]
    ax.scatter(flat_e[flat_in], flat_w[flat_in], s=4, color=C1, alpha=0.3,
               label="covered", linewidths=0)
    ax.scatter(flat_e[~flat_in], flat_w[~flat_in], s=4, color=RED, alpha=0.3,
               label="missed", linewidths=0)
    ax.set_xlabel("|prediction - target| (raw units)")
    ax.set_ylabel("interval width / sigma")
    ax.set_title("Calibration: width against actual error", loc="left")
    ax.legend(loc="upper left", markerscale=3)

    fig.suptitle(f"Safe Prediction Score -- {source}; coverage "
                 f"{met['coverage'] * 100:.1f}%, sps_score "
                 f"{met['subscores']['sps_score']:.2f}",
                 x=0.005, ha="left", fontweight="bold")
    fig.tight_layout()
    return finish(fig, out / "08_sps_bounds.png", dpi)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", default=str(HERE),
                    help="directory containing submission.py")
    ap.add_argument("--data", default=str(HERE / "example_data"),
                    help="data directory with test_real/ and mean_std_real.pt")
    ap.add_argument("--outdir", default=None,
                    help="where to write figures + metrics.json "
                         "(default: <submission>/analytics)")
    ap.add_argument("--snapshot", default="worst",
                    help="window for the field/TKE snapshots: worst | best | median "
                         "| an index (default: worst)")
    ap.add_argument("--frame", type=int, default=-1,
                    help="frame within that window for the field snapshot "
                         "(default: -1, the hardest lead time)")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--save-run", default=None,
                    help="also write the raw run (predictions, targets, times) to this "
                         ".npz so the figures can be redrawn without re-running")
    ap.add_argument("--from-run", default=None,
                    help="skip the stream and redraw the figures from a --save-run .npz")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    submission_dir = Path(args.submission).resolve()
    data_dir = Path(args.data).resolve()
    out = Path(args.outdir).resolve() if args.outdir else submission_dir / "analytics"
    out.mkdir(parents=True, exist_ok=True)

    if args.from_run:
        run = load_run(Path(args.from_run).resolve())
        print(f"[eval] replotting from {args.from_run} "
              f"({run['pred'].shape[0]} windows; the stream was not re-run)")
    else:
        run = run_stream(submission_dir, data_dir, args.device)
        if args.save_run:
            save_run(run, Path(args.save_run).resolve())
            print(f"[eval] saved raw run to {args.save_run}")
    met = compute_metrics(run)

    print(f"[eval] prediction stack shape {run['pred'].shape}")
    print(f"[eval] mean per-step time: {met['mean_step_s'] * 1e3:.2f} ms "
          f"(total {run['times'].sum():.3f} s)")
    print(f"[eval] SPS bounds: "
          f"{'participant-supplied' if run['bounds_active'] else 'none -> scorer default band'}")
    print("[eval] subscores (same formulas as the leaderboard; local data):")
    for name in SUBSCORES:
        print(f"[eval]     {name:13s} {met['subscores'][name]:7.3f}")

    rel = met["per_window"]["rel_l2"]
    worst = np.argsort(rel)[::-1][:3]
    print("[eval] worst windows: " + ", ".join(
        f"{run['sim_ids'][i].replace('.h5','')}@t{run['time_ids'][i]} "
        f"(rel L2 {rel[i]:.4f})" for i in worst))

    use_style()
    window = pick_window(met, args.snapshot)
    figures = [
        plot_stream_overview(run, met, out, args.dpi),
        plot_lead_time(run, met, out, args.dpi),
        plot_spatial_error(run, met, out, args.dpi),
        plot_field_snapshot(run, met, out, args.dpi, window, args.frame),
        plot_tke_maps(run, met, out, args.dpi, window),
        plot_mvpe_profiles(run, met, out, args.dpi),
        plot_adaptation(run, met, out, args.dpi),
        plot_sps(run, met, out, args.dpi),
    ]
    figures = [f for f in figures if f is not None]

    payload = {
        "submission": str(submission_dir),
        "data": f"cached:{args.from_run}" if args.from_run else str(data_dir),
        "n_steps": int(len(rel)), "n_trajectories": int(sum(run["is_first"])),
        "channels": met["channels"], "bounds_active": run["bounds_active"],
        "mean_step_ms": met["mean_step_s"] * 1e3, "coverage": met["coverage"],
        "subscores": met["subscores"],
        "per_window": {
            "sim_id": run["sim_ids"], "time_id": [int(t) for t in run["time_ids"]],
            "is_first": [bool(b) for b in run["is_first"]],
            "rel_l2": [float(v) for v in rel],
            "tke": [float(v) for v in met["per_window"]["tke"]],
            "mvpe": [float(v) for v in met["per_window"]["mvpe"]],
            "step_ms": [float(v * 1e3) for v in run["times"]],
            "adapt_loss": [None if np.isnan(v) else float(v) for v in run["adapt_loss"]],
        },
        "figures": [f.name for f in figures],
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(f"[eval] wrote {len(figures)} figures + metrics.json to {out}")
    for f in figures:
        print(f"[eval]     {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
