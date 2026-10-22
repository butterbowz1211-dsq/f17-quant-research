# Research project author: Gilmour Jia
"""Beta/alpha decomposition of score-to-portfolio rules against the EW pool.

The 2022 fold lost 25-29% for every long-only rule, which looks like a mapping
failure but is more likely market beta.  This script regresses each rule's daily
net return on the equal-weight all-pool return produced by the *same* hard
executor, so execution frictions and cash drag are common to both sides.

    python tools/beta_decompose.py --fold 0
    python tools/beta_decompose.py --fold 1

Inputs are the CSVs written by
    python tools/engine_from_scores.py --fold N --save-daily ...
No model is trained and no score is recomputed here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))

from run_policy import perf_metrics  # noqa: E402


def newey_west(y: np.ndarray, x: np.ndarray, lags: int = 5):
    """OLS with Newey-West HAC standard errors; returns slope, intercept, t."""
    X = np.column_stack([np.ones_like(x), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    n = len(y)
    lags = min(lags, max(0, n - 2))
    S = np.zeros((2, 2))
    for lag in range(lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        if lag == 0:
            u = X * e[:, None]
            S += w * (u.T @ u)
        else:
            a = (X[lag:] * e[lag:, None]).T
            bb = (X[:-lag] * e[:-lag, None]).T
            S += w * (a @ bb.T + bb @ a.T)
    cov = XtX_inv @ S @ XtX_inv / max(1, n)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    r2 = 1.0 - float(e @ e) / float(((y - y.mean()) ** 2).sum())
    return dict(alpha_daily=float(b[0]), beta=float(b[1]),
                alpha_t=float(b[0] / se[0]) if se[0] > 0 else 0.0,
                beta_t=float(b[1] / se[1]) if se[1] > 0 else 0.0,
                r2=r2, n=int(n))


def load_daily(path: Path) -> pd.Series:
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")["net"].astype(float)


def decompose(p: pd.Series, b: pd.Series, label: str) -> dict:
    idx = p.index.intersection(b.index)
    p, b = p.loc[idx], b.loc[idx]
    fit = newey_west(p.to_numpy(), b.to_numpy())
    pv, bv = p.to_numpy(), b.to_numpy()
    up, down = bv > 0, bv < 0
    beta_up = (np.cov(pv[up], bv[up])[0, 1] / np.var(bv[up], ddof=1)
               if up.sum() > 2 else np.nan)
    beta_dn = (np.cov(pv[down], bv[down])[0, 1] / np.var(bv[down], ddof=1)
               if down.sum() > 2 else np.nan)
    cap_up = pv[up].mean() / bv[up].mean() if up.sum() and bv[up].mean() else np.nan
    cap_dn = pv[down].mean() / bv[down].mean() if down.sum() and bv[down].mean() else np.nan
    hedged = pv - fit["beta"] * bv
    hm = perf_metrics(hedged)
    pm = perf_metrics(pv)
    out = dict(rule=label, **fit, beta_up=float(beta_up), beta_dn=float(beta_dn),
               up_capture=float(cap_up), down_capture=float(cap_dn),
               alpha_ann=fit["alpha_daily"] * 252.0,
               raw_cagr=pm["cagr"], raw_sharpe=pm["sharpe"],
               raw_maxdd=pm["max_drawdown"],
               hedged_cagr=hm["cagr"], hedged_sharpe=hm["sharpe"],
               hedged_maxdd=hm["max_drawdown"],
               hedged_total=hm["total_return"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--benchmark", default="ew_all.csv")
    ap.add_argument("--lags", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    ddir = Path(args.dir) if args.dir else ART / ("engine_daily_fold%d" % args.fold)
    bench = load_daily(ddir / args.benchmark)
    files = sorted(p for p in ddir.glob("*.csv") if p.name != args.benchmark)

    rows = []
    for path in files:
        p = load_daily(path)
        rows.append(decompose(p, bench, path.stem))
    df = pd.DataFrame(rows).sort_values("alpha_ann", ascending=False)

    print("fold%d  benchmark=%s  n=%d" % (args.fold, args.benchmark, len(bench)))
    print("%-24s %7s %8s %8s %8s %8s %7s %7s %8s %8s"
          % ("rule", "beta", "b_down", "upCap", "dnCap", "alpha%", "a-t",
             "R2", "hCAGR", "hSharpe"))
    for _, r in df.iterrows():
        print("%-24s %7.3f %8.3f %8.2f %8.2f %+8.2f %7.2f %7.2f %+7.1f%% %8.2f"
              % (r["rule"], r["beta"], r["beta_dn"], r["up_capture"],
                 r["down_capture"], r["alpha_ann"] * 100, r["alpha_t"],
                 r["r2"], r["hedged_cagr"] * 100, r["hedged_sharpe"]))

    # Calendar-year split, which is what 2021 vs 2022 actually asks.
    years = sorted(set(bench.index.year))
    yearly = []
    for y in years:
        b = bench[bench.index.year == y]
        for path in files:
            p = load_daily(path)
            p = p[p.index.year == y]
            if len(p) < 30:
                continue
            fit = newey_west(p.to_numpy(), b.loc[b.index.intersection(p.index)].to_numpy(),
                             lags=args.lags)
            yearly.append(dict(year=int(y), rule=path.stem, **fit,
                               alpha_ann=fit["alpha_daily"] * 252.0))
    ydf = pd.DataFrame(yearly)

    out = args.out or str(ART / ("beta_decompose_fold%d.json" % args.fold))
    Path(out).write_text(json.dumps(dict(
        fold=args.fold, benchmark=args.benchmark,
        full=df.to_dict(orient="records"),
        yearly=ydf.to_dict(orient="records")), indent=2), encoding="utf-8")
    print("[beta] wrote %s" % out)
    return df


if __name__ == "__main__":
    main()
