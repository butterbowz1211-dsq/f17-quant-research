# Research project author: Gilmour Jia
"""Apply F17 weight-lab coefficients to the full eligible cross-section.

The lab fits on complete-case cross-sections (all 17 factors present).  A live
score must also rank stocks with missing factors.  This script standardises each
factor among the stocks where it is observed on that date, fills the
cross-sectional mean (0) for missing factors, and applies the fitted weight
vector.  It then compares rank IC and top/bottom-decile long-short returns,
gross and net of 25 bp one-way cost, with the production available-case F17
score.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"workspace/source_workspace/quant_factor_ashare/ml_policy")
ART = ROOT / "artifacts"
CACHE = ART / "f17_panel_cache.npz"
DEPS = Path(r"workspace/review_workspace/quant_factor_ashare_review/.deps")
if str(DEPS) not in sys.path:
    sys.path.append(str(DEPS))
for _d in (DEPS / "pyarrow", ROOT / "tools" / "pyarrow_dlls"):
    if _d.is_dir():
        os.add_dll_directory(str(_d))
import pyarrow  # noqa: F401
import pandas as pd  # noqa: F811

ART.mkdir(parents=True, exist_ok=True)
H = 21


def rank_ic(score, label):
    ok = np.isfinite(score) & np.isfinite(label)
    if ok.sum() < 50:
        return np.nan
    a = pd.Series(score[ok]).rank(method="average").to_numpy()
    b = pd.Series(label[ok]).rank(method="average").to_numpy()
    a = (a - a.mean()) / (a.std(ddof=0) + 1e-12)
    b = (b - b.mean()) / (b.std(ddof=0) + 1e-12)
    return float((a * b).mean())


def hac(x, lags=3):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return {"n": n, "mean": None, "t": None}
    e = x - x.mean()
    lrv = float(e @ e / n)
    for k in range(1, min(lags, n - 1) + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1)) * float(e[k:] @ e[:-k] / n)
    se = math.sqrt(max(lrv, 0.0) / n)
    return {"n": n, "mean": float(x.mean()),
            "t": float(x.mean() / se) if se > 0 else None}


def full_score(Xj, base, w):
    """Standardise each factor on its observed base and dot with w."""
    s = np.zeros(len(Xj), float)
    for k in range(Xj.shape[1]):
        m = base & np.isfinite(Xj[:, k])
        if m.sum() < 50:
            continue
        v = Xj[m, k].astype(float)
        mu, sd = v.mean(), v.std(ddof=0)
        if sd > 1e-12:
            s[m] += w[k] * (Xj[m, k].astype(float) - mu) / sd
    return s[base]


def ls_stats(S, Y, elig, dates, cost_bp):
    rets, turns, grosses = [], [], []
    prev_t = prev_b = None
    for j in range(len(dates)):
        m = elig[j] & np.isfinite(Y[j]) & np.isfinite(S[j])
        if m.sum() < 100:
            rets.append(np.nan); turns.append(np.nan); grosses.append(np.nan)
            continue
        idx = np.flatnonzero(m)
        order = idx[np.argsort(-S[j, idx], kind="stable")]
        qn = max(20, int(len(order) * 0.1))
        top, bot = set(order[:qn].tolist()), set(order[-qn:].tolist())
        gross = float(Y[j][list(top)].mean() - Y[j][list(bot)].mean())
        if prev_t is None:
            turn = 2.0
        else:
            turn = len(top - prev_t) / len(top) + len(bot - prev_b) / len(bot)
        rets.append(gross - cost_bp * 1e-4 * turn)
        turns.append(turn)
        grosses.append(gross)
        prev_t, prev_b = top, bot
    return np.asarray(rets), np.asarray(turns), np.asarray(grosses)


def perf(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return {"n": len(x), "ann": None, "sharpe": None, "hit": None}
    ann = float(np.prod(1.0 + x) ** (252.0 / (H * len(x))) - 1.0)
    sd = x.std(ddof=1) * math.sqrt(252.0 / H)
    sharpe = float(x.mean() / x.std(ddof=1) * math.sqrt(252.0 / H)) if x.std(ddof=1) > 0 else None
    return {"n": len(x), "ann": ann, "sharpe": sharpe, "hit": float((x > 0).mean())}


def main():
    z = np.load(CACHE, allow_pickle=True)
    X = z["X"].astype(np.float32)
    Y = z["Y"].astype(np.float64)
    elig = z["elig"].astype(bool)
    score_prod = z["score"].astype(np.float64)
    dates = pd.to_datetime([str(s) for s in z["dates"]])
    syms = z["syms"]

    meta = json.loads((ART / "f17_weight_lab_full_weights.json").read_text(encoding="utf-8"))
    models = ["equal", "ols", "ridge", "ridge_eq", "lasso", "enet", "kozak", "brandt"]
    weights = {m: {} for m in models}
    for m in models:
        for y, info in meta["info"].get(m, {}).items():
            weights[m][int(y)] = np.asarray(info["w"], float)

    n = len(dates)
    S = {m: np.full((n, X.shape[1]), np.nan, dtype=np.float32) for m in models}
    S["production_f17"] = score_prod.astype(np.float32)
    for j in range(n):
        base = elig[j] & np.isfinite(Y[j])
        if base.sum() < 100:
            continue
        yr = int(dates[j].year)
        for m in models:
            w = weights[m].get(yr)
            if w is None:
                continue
            S[m][j, base] = full_score(X[j], base, w)

    rows = []
    for split, lo, hi in [("development_2016_2022", "2016-01-01", "2022-12-31"),
                           ("validation_2023_2026", "2023-01-01", "2026-12-31")]:
        msk = (dates >= lo) & (dates <= hi)
        series = {}
        for m in ["production_f17"] + models:
            ics = []
            for j in np.flatnonzero(msk):
                mm = elig[j] & np.isfinite(Y[j]) & np.isfinite(S[m][j])
                if mm.sum() >= 50:
                    ics.append(rank_ic(S[m][j][mm], Y[j][mm]))
            st = hac(ics)
            series[m] = np.asarray(ics, float)
            delta = {}
            if m != "equal" and "equal" in series:
                d = series[m] - series["equal"]
                dh = hac(d)
                delta = {"delta_ic_vs_equal": dh["mean"], "delta_t_vs_equal": dh["t"]}
            for cost in (0.0, 25.0):
                rets, turns, gross = ls_stats(S[m], Y, elig, dates, cost)
                r = rets[msk]
                p = perf(r)
                rows.append({
                    "split": split, "model": m, "cost_bp": cost,
                    "n_ic": st["n"], "ic": st["mean"], "ic_t": st["t"],
                    "ls_n": p["n"], "ls_ann": p["ann"], "ls_sharpe": p["sharpe"],
                    "ls_hit": p["hit"],
                    "turnover": float(np.nanmean(turns[msk])),
                    **delta,
                })
    out = pd.DataFrame(rows)
    out.to_csv(ART / "f17_weight_deploy_metrics.csv", index=False, encoding="utf-8-sig")
    (ART / "f17_weight_deploy_metrics.json").write_text(
        out.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
