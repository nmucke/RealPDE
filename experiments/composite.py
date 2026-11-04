"""Estimate of the leaderboard final_score from the five subscores.

The leaderboard's combination formula is unpublished. This fits final ~ w . (rel_l2, tke, mvpe,
time, sps), no intercept, by least squares over the rows of experiments/leaderboard.csv that have
all five subscores and a final score (every recorded upload, including those the board did not
keep, scored by their own row). It also reports the in-sample and leave-one-out residuals and the
leverage of each row.

The rows span a narrow range (sps 36-45, time 78-92), and two rows (09-04 and 09-07) carry
leverage near 0.95: the time and tke weights rest mostly on them. The estimate is therefore meant
for subscores inside the fitted range; `estimate` flags values outside it and points with a
leverage above the largest row leverage.

    python experiments/composite.py        # weights, residuals, leverages, ranges
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "leaderboard.csv")
KEYS = ("rel_l2", "tke", "mvpe", "time", "sps")
# the team's planning weights (final_submission/realpde/scripts/lco_analyze.py); they sit about
# 0.8 below the leaderboard scale on every row, so use them for differences between presets only
W_PLAN = {"rel_l2": 0.48, "tke": 0.07, "mvpe": 0.10, "time": 0.09, "sps": 0.28}


def rows(path: str = CSV) -> Tuple[List[Dict], int]:
    """(usable rows, number of CSV rows skipped because a subscore or the final score is missing / malformed)."""
    out, skipped = [], 0
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                out.append({"date": r["upload_date"], "x": [float(r[k]) for k in KEYS], "y": float(r["final_score"])})
            except (KeyError, TypeError, ValueError):
                skipped += 1
    return out, skipped


def _lstsq(data):
    import numpy as np
    A = np.array([d["x"] for d in data])
    y = np.array([d["y"] for d in data])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def fit(path: str = CSV) -> Dict:
    """Weights, residuals (in-sample and leave-one-out), leverages and the subscore ranges of the fit rows."""
    import numpy as np
    data, skipped = rows(path)
    X = np.array([d["x"] for d in data])
    w = _lstsq(data)
    loo_w = [_lstsq(data[:i] + data[i + 1:]) for i in range(len(data))]
    XtX_inv = np.linalg.inv(X.T @ X)
    lev = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    return {"n": len(data), "skipped": skipped, "weights": dict(zip(KEYS, map(float, w))),
            "dates": [d["date"] for d in data],
            "in_sample": [float(np.dot(w, d["x"]) - d["y"]) for d in data],
            "loo": [float(np.dot(wi, d["x"]) - d["y"]) for wi, d in zip(loo_w, data)],
            "loo_weight_range": {k: (float(min(v[i] for v in loo_w)), float(max(v[i] for v in loo_w)))
                                 for i, k in enumerate(KEYS)},
            "leverage": [float(h) for h in lev], "max_leverage": float(lev.max()), "XtX_inv": XtX_inv.tolist(),
            "range": {k: (float(X[:, i].min()), float(X[:, i].max())) for i, k in enumerate(KEYS)}}


def estimate(scores: Dict[str, float], f: Dict) -> Dict:
    """{"fit", "extrapolated": subscores outside the fit range, "leverage", "high_leverage", "plan_rel"}."""
    import numpy as np
    s = {k: scores[k + "_score"] if k + "_score" in scores else scores[k] for k in KEYS}
    x = np.array([s[k] for k in KEYS])
    h = float(x @ np.array(f["XtX_inv"]) @ x)
    return {"fit": sum(f["weights"][k] * s[k] for k in KEYS),
            "extrapolated": [k for k in KEYS if not f["range"][k][0] <= s[k] <= f["range"][k][1]],
            "leverage": h, "high_leverage": h > f["max_leverage"],
            "plan_rel": sum(W_PLAN[k] * s[k] for k in KEYS)}


def main() -> None:
    f = fit()
    print("least-squares fit over %d rows of %s (%d rows skipped: missing or malformed values)"
          % (f["n"], os.path.relpath(CSV), f["skipped"]))
    print("  weights: " + "  ".join("%s %.4f" % (k, v) for k, v in f["weights"].items())
          + "  (sum %.3f)" % sum(f["weights"].values()))
    print("  leave-one-out weight ranges: " + "  ".join("%s %.3f-%.3f" % (k, *v) for k, v in f["loo_weight_range"].items()))
    print("  %-10s %9s %9s %9s" % ("upload", "in-sample", "LOO", "leverage"))
    for d, a, b, h in zip(f["dates"], f["in_sample"], f["loo"], f["leverage"]):
        print("  %-10s %+9.4f %+9.4f %9.3f" % (d, a, b, h))
    mx = lambda v: max(abs(t) for t in v)  # noqa: E731
    rms = lambda v: (sum(t * t for t in v) / len(v)) ** 0.5  # noqa: E731
    print("  max |residual|: in-sample %.4f, LOO %.4f; LOO rms %.4f" % (mx(f["in_sample"]), mx(f["loo"]), rms(f["loo"])))
    print("  subscore ranges of the fit rows: " + "  ".join("%s %.1f-%.1f" % (k, *v) for k, v in f["range"].items()))
    print("  rows with leverage near 1 pin the weights they mostly determine; a LOO residual near 0 on such a "
          "row does not mean it is predicted well.")


if __name__ == "__main__":
    main()
