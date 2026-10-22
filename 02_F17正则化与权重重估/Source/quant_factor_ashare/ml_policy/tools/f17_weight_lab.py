# Research project author: Gilmour Jia
"""Fit F17 factor weights with regularisation from the asset-pricing literature.

The baseline F17 score is the equal-weight average of 17 standardised,
industry/size-neutral factor panels.  This lab fits signed linear composites on
the same panels and compares them with equal weights out of sample.

Methods
-------
* OLS: unregularised cross-sectional regression of the next 21-day return rank
  on the 17 factor exposures.
* Ridge / ridge-to-equal: L2 penalty, with the prior mean either zero or the
  equal-weight vector (the F17 prior).
* Lasso / adaptive lasso / elastic net: L1 and L1+L2 penalties.  The adaptive
  lasso uses a first-stage ridge estimate to set feature-specific penalties,
  following the adaptive-LASSO idea used by Freyberger, Neuhierl and Weber
  (2020, RFS) for characteristic selection.
* Kozak-L2: SDF coefficients b = (Sigma + gamma I)^-1 mu on the 17
  characteristic long-short factor returns, with gamma chosen by 3-fold
  cross-validation of the factor-return R2 (Kozak, Nagel and Santosh, 2020,
  JFE, eqs. 18-22, 26-30).
* Brandt policy: maximise the in-sample Sharpe ratio of the parametric
  portfolio w_i = 1/N + (1/N) theta'x_i, then shrink theta toward equal weights
  (Brandt, Santa-Clara and Valkanov, 2009, RFS, eqs. 3, 5-6, 7 and section
  2.3.5).  The fitted theta is used as a cross-sectional score.

All models are refit annually on a five-year rolling window.  A training label
must exit strictly before 1 January of the test year.  Hyper-parameters are
chosen by three contiguous time-series folds inside the training window, never
on the 2023-2026 validation period.

Run with the project venv, e.g.

    python tools/f17_weight_lab.py --years 2016 2017 2018 --models equal ols ridge lasso

"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(r"workspace/source_workspace\quant_factor_ashare")
DEPS = Path(r"workspace/review_workspace\quant_factor_ashare_review\.deps")
if str(DEPS) not in sys.path:
    sys.path.append(str(DEPS))
for _d in (DEPS / "pyarrow", ROOT_DIR / "ml_policy" / "tools" / "pyarrow_dlls"):
    if _d.is_dir():
        os.add_dll_directory(str(_d))
import pyarrow  # noqa: F401  (pandas 3 pickles use Arrow-backed columns)
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ART = ROOT / "artifacts"
CACHE = ART / "f17_panel_cache.npz"
PREP = ROOT.parent / "data" / "prep"

NAMES = ["TURN", "MAX5", "INTRADAY", "ILLIQ", "COMMOVE", "TURN_SHOCK",
         "REV_1W", "ISKEW", "IVOL", "OVERNIGHT", "LEAD_LAG", "CO_SKEW",
         "BETA", "REV_LT", "SIZE", "HIGH52", "RES_MOM"]
K = len(NAMES)
H = 21
STEP = 21
LAM_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
LASSO_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
EN_GRID = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
KOZAK_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
BRANDT_SHRINK = [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------- basic stats
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


def zscore(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std(ddof=0) + 1e-12)


def rank_z(v):
    r = pd.Series(v).rank(method="average").to_numpy()
    return zscore(r)


def winsor_z(v, lo=1.0, hi=99.0):
    v = np.asarray(v, float)
    a, b = np.nanpercentile(v, [lo, hi])
    return zscore(np.clip(v, a, b))


# ------------------------------------------------------------------- loading
def load_cache():
    z = np.load(CACHE, allow_pickle=True)
    X = z["X"].astype(np.float32)
    Y = z["Y"].astype(np.float64)
    elig = z["elig"].astype(bool)
    dates = pd.to_datetime([str(s) for s in z["dates"]])
    pos = z["pos"].astype(int)
    full = pd.read_pickle(PREP / f"{NAMES[0]}.pkl").index
    exit_dates = pd.DatetimeIndex([full[p + H + 1] for p in pos])
    return X, Y, elig, dates, exit_dates


def cs_sample(X, Y, elig, j, target="rank"):
    """Eligible, complete-case cross-section with standardised factors."""
    m = elig[j] & np.isfinite(Y[j]) & np.isfinite(X[j]).all(axis=1)
    x = X[j, m, :].astype(np.float64)
    x = (x - x.mean(0)) / (x.std(0) + 1e-12)
    y = Y[j, m].astype(np.float64)
    if target == "rank":
        y = rank_z(y)
    elif target == "winsor":
        y = winsor_z(y)
    else:
        y = zscore(y)
    return x, y, m


# ------------------------------------------------------------------ fitting
def fit_ridge(xt, yt, lam, prior=None):
    n = len(yt)
    G = xt.T @ xt / n
    c = xt.T @ yt / n
    if prior is not None:
        c = c + lam * prior
    return np.linalg.solve(G + lam * np.eye(xt.shape[1]), c)


def fit_ols(xt, yt):
    return np.linalg.lstsq(xt, yt, rcond=None)[0]


def _lasso(xt, yt, alpha, penalty=None, l1_ratio=1.0, max_iter=20000):
    from sklearn.linear_model import ElasticNet
    if penalty is None:
        xx = xt
        scale = np.ones(xt.shape[1])
    else:
        scale = np.asarray(penalty, float)
        xx = xt * scale[None, :]
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False,
                       max_iter=max_iter, tol=1e-6, selection="cyclic")
    model.fit(xx, yt)
    coef = np.asarray(model.coef_, float)
    if penalty is not None:
        coef = coef * scale
    return coef


def fit_adaptive(xt, yt, alpha, w0, gamma=1.0):
    scale = np.abs(w0) ** gamma
    scale = np.maximum(scale, 1e-6)
    return _lasso(xt, yt, alpha, penalty=scale)


# --------------------------------------------------------------- train folds
def time_folds(n, k=3):
    """Expanding-window folds: each validation block uses only earlier dates."""
    bounds = np.linspace(0, n, k + 1).astype(int)
    out = []
    for i in range(1, k):
        val = np.arange(bounds[i], bounds[i + 1])
        tr = np.arange(0, bounds[i])
        if len(tr) and len(val):
            out.append((tr, val))
    return out


def train_arrays(X, Y, elig, hist, target="rank"):
    xs, ys = [], []
    for j in hist:
        x, y, _ = cs_sample(X, Y, elig, j, target=target)
        if len(y) >= 100:
            xs.append(x)
            ys.append(y)
    if not xs:
        return None, None
    return np.vstack(xs), np.concatenate(ys)


def cv_alpha(X, Y, elig, hist, fit_fn, grid, target="rank"):
    """Pick a hyper-parameter by mean validation rank IC over expanding folds."""
    folds = time_folds(len(hist), 3)
    if len(folds) < 2:
        return grid[len(grid) // 2], []
    prepared = []
    for tr, va in folds:
        xt, yt = train_arrays(X, Y, elig, [hist[i] for i in tr], target=target)
        if xt is None or len(yt) < 500:
            continue
        prepared.append((xt, yt, [hist[i] for i in va]))
    if not prepared:
        return grid[len(grid) // 2], []
    score = []
    for g in grid:
        ics = []
        for xt, yt, va_dates in prepared:
            w = fit_fn(xt, yt, g)
            for j in va_dates:
                xv, _, m = cs_sample(X, Y, elig, j, target=target)
                if m.sum() < 100:
                    continue
                ic = rank_ic(xv @ w, Y[j, m])
                if np.isfinite(ic):
                    ics.append(ic)
        score.append(float(np.mean(ics)) if ics else -9.9)
    best = int(np.argmax(score))
    return grid[best], score


# ------------------------------------------------------------------- Kozak
def factor_returns(X, Y, elig, q=0.2):
    n = len(Y)
    F = np.full((n, K), np.nan)
    for j in range(n):
        base = elig[j] & np.isfinite(Y[j])
        for k in range(K):
            m = base & np.isfinite(X[j, :, k])
            if m.sum() < 100:
                continue
            v = X[j, m, k]
            order = np.argsort(v, kind="stable")
            qn = max(5, int(len(order) * q))
            lo, hi = order[:qn], order[-qn:]
            F[j, k] = Y[j, m][hi].mean() - Y[j, m][lo].mean()
    return F


def kozak_weights(F, hist, gamma):
    f = F[hist]
    f = f[np.isfinite(f).all(axis=1)]
    if len(f) < 12:
        return None
    mu = f.mean(0)
    Sig = np.cov(f, rowvar=False)
    return np.linalg.solve(Sig + gamma * np.eye(K), mu)


def kozak_cv(F, hist, grid):
    folds = time_folds(len(hist), 3)
    if len(folds) < 2:
        return grid[len(grid) // 2], []
    scores = []
    for g in grid:
        r2s = []
        for tr, va in folds:
            tr_dates = [hist[i] for i in tr]
            va_dates = [hist[i] for i in va]
            b = kozak_weights(F, tr_dates, g)
            if b is None:
                continue
            fva = F[va_dates]
            fva = fva[np.isfinite(fva).all(axis=1)]
            if len(fva) < 6:
                continue
            mu2 = fva.mean(0)
            S2 = np.cov(fva, rowvar=False)
            res = mu2 - S2 @ b
            den = float(mu2 @ mu2)
            if den > 1e-12:
                r2s.append(1.0 - float(res @ res) / den)
        scores.append(float(np.mean(r2s)) if r2s else -9.9)
    best = int(np.argmax(scores))
    return grid[best], scores


# ------------------------------------------------------------------ Brandt
def brandt_moments(X, Y, elig, hist):
    G, req = [], []
    for j in hist:
        x, _, m = cs_sample(X, Y, elig, j, target="raw")
        if m.sum() < 100:
            continue
        r = Y[j, m]
        G.append(x.T @ r / len(r))
        req.append(float(r.mean()))
    if not G:
        return None, None
    return np.vstack(G), np.asarray(req)


def brandt_fit(G, req, starts=None):
    from scipy.optimize import minimize

    def neg_sharpe(v):
        nv = np.linalg.norm(v)
        if nv < 1e-10:
            return 1e6
        th = v / nv
        rp = req + G @ th
        sd = rp.std(ddof=1)
        if sd <= 1e-12:
            return 1e6
        return -float(rp.mean() / sd)

    starts = starts or []
    best = None
    for s in starts:
        s = np.asarray(s, float)
        s = s / (np.linalg.norm(s) + 1e-12)
        res = minimize(neg_sharpe, s, method="BFGS",
                       options={"maxiter": 2000, "gtol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        return np.ones(K) / math.sqrt(K)
    v = best.x
    return v / (np.linalg.norm(v) + 1e-12)


def brandt_shrink_cv(G, req, grid):
    folds = time_folds(len(req), 3)
    if len(folds) < 2:
        return grid[-1], []
    eq = np.ones(K) / math.sqrt(K)
    scores = []
    for a in grid:
        ss = []
        for tr, va in folds:
            th = brandt_fit(G[tr], req[tr],
                            starts=[eq, np.ones(K), np.arange(1, K + 1, dtype=float)])
            th = (1 - a) * th + a * eq
            rp = req[va] + G[va] @ th
            sd = rp.std(ddof=1)
            if sd > 1e-12:
                ss.append(float(rp.mean() / sd))
        scores.append(float(np.mean(ss)) if ss else -9.9)
    best = int(np.argmax(scores))
    return grid[best], scores


# ------------------------------------------------------------------ main
MODELS = ["equal", "ols", "ridge", "ridge_eq", "lasso", "adaptive", "enet", "kozak", "brandt"]


def fit_year(model, X, Y, elig, hist, F, target="rank"):
    """Return (weights, info)."""
    if model == "equal":
        return np.ones(K) / K, {}
    if model == "ols":
        xt, yt = train_arrays(X, Y, elig, hist, target)
        return fit_ols(xt, yt), {}
    if model in ("ridge", "ridge_eq"):
        prior = np.ones(K) / K if model == "ridge_eq" else None
        fn = lambda xt, yt, g: fit_ridge(xt, yt, g, prior=prior)
        best, scores = cv_alpha(X, Y, elig, hist, fn, LAM_GRID, target)
        xt, yt = train_arrays(X, Y, elig, hist, target)
        return fit_ridge(xt, yt, best, prior=prior), {"lam": best, "cv": scores}
    if model == "lasso":
        fn = lambda xt, yt, g: _lasso(xt, yt, g)
        best, scores = cv_alpha(X, Y, elig, hist, fn, LASSO_GRID, target)
        xt, yt = train_arrays(X, Y, elig, hist, target)
        return _lasso(xt, yt, best), {"alpha": best, "cv": scores}
    if model == "adaptive":
        fn0 = lambda xt, yt, g: fit_ridge(xt, yt, g)
        lam, _ = cv_alpha(X, Y, elig, hist, fn0, LAM_GRID, target)
        xt, yt = train_arrays(X, Y, elig, hist, target)
        w0 = fit_ridge(xt, yt, lam)
        fn = lambda x, y, g: fit_adaptive(x, y, g, w0)
        best, scores = cv_alpha(X, Y, elig, hist, fn, LASSO_GRID, target)
        return fit_adaptive(xt, yt, best, w0), {"lam0": lam, "alpha": best, "cv": scores}
    if model == "enet":
        fn = lambda xt, yt, g: _lasso(xt, yt, g, l1_ratio=0.5)
        best, scores = cv_alpha(X, Y, elig, hist, fn, EN_GRID, target)
        xt, yt = train_arrays(X, Y, elig, hist, target)
        return _lasso(xt, yt, best, l1_ratio=0.5), {"alpha": best, "l1_ratio": 0.5, "cv": scores}
    if model == "kozak":
        best, scores = kozak_cv(F, hist, KOZAK_GRID)
        w = kozak_weights(F, hist, best)
        if w is None:
            return np.ones(K) / K, {"gamma": best, "fallback": True, "cv": scores}
        return w, {"gamma": best, "cv": scores}
    if model == "brandt":
        G, req = brandt_moments(X, Y, elig, hist)
        if G is None:
            return np.ones(K) / K, {"fallback": True}
        eq = np.ones(K) / math.sqrt(K)
        th = brandt_fit(G, req, starts=[eq, np.ones(K), np.arange(1, K + 1, dtype=float)])
        a, scores = brandt_shrink_cv(G, req, BRANDT_SHRINK)
        th = (1 - a) * th + a * eq
        return th, {"shrink": a, "cv": scores}
    raise ValueError(model)


def ls_stats(S, Y, elig, dates, cost_bp=0.0):
    """Top/bottom decile long-short, 21-day non-overlapping returns."""
    rets, turns = [], []
    prev_t = prev_b = None
    for j in range(len(dates)):
        m = elig[j] & np.isfinite(Y[j]) & np.isfinite(S[j])
        if m.sum() < 100:
            rets.append(np.nan)
            turns.append(np.nan)
            continue
        idx = np.flatnonzero(m)
        sc = S[j, idx]
        order = idx[np.argsort(-sc, kind="stable")]
        qn = max(20, int(len(order) * 0.1))
        top, bot = set(order[:qn].tolist()), set(order[-qn:].tolist())
        y = Y[j]
        gross = float(y[list(top)].mean() - y[list(bot)].mean())
        if prev_t is None:
            turn = 2.0
        else:
            turn = (len(top - prev_t) / len(top) + len(bot - prev_b) / len(bot))
        rets.append(gross - cost_bp * 1e-4 * turn)
        turns.append(turn)
        prev_t, prev_b = top, bot
    rets = np.asarray(rets, float)
    return rets, np.asarray(turns, float)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=list(range(2016, 2027)))
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--target", choices=["rank", "winsor", "raw"], default="rank")
    ap.add_argument("--out", default=str(ART / "f17_weight_lab"))
    args = ap.parse_args(argv)

    X, Y, elig, dates, exit_dates = load_cache()
    F = factor_returns(X, Y, elig)
    n = len(dates)
    print(f"loaded {X.shape}, dates {dates.min().date()}..{dates.max().date()}, "
          f"factor returns finite {np.isfinite(F).all(axis=1).mean():.3f}", flush=True)

    S = {m: np.full((n, X.shape[1]), np.nan, dtype=np.float32) for m in args.models}
    info = {m: {} for m in args.models}

    for year in args.years:
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        start = cutoff - pd.DateOffset(years=5)
        hist = [j for j in range(n)
                if start <= dates[j] < cutoff and exit_dates[j] < cutoff]
        test = [j for j in range(n) if dates[j].year == year]
        if len(hist) < 24:
            print(f"skip {year}: {len(hist)} training dates", flush=True)
            continue
        print(f"year {year}: {len(hist)} train dates, {len(test)} test dates", flush=True)
        for model in args.models:
            w, meta = fit_year(model, X, Y, elig, hist, F, target=args.target)
            w = np.asarray(w, float)
            print(f"    {model} {year} weight shape {w.shape}", flush=True)
            info[model][str(year)] = {"w": [float(v) for v in np.ravel(w)], **meta}
            for j in test:
                m = elig[j] & np.isfinite(Y[j]) & np.isfinite(X[j]).all(axis=1)
                if m.sum() == 0:
                    continue
                x = X[j, m, :].astype(np.float64)
                x = (x - x.mean(0)) / (x.std(0) + 1e-12)
                S[model][j, m] = x @ w
            print(f"  {model:10s} " + " ".join(f"{v:+.3f}" for v in w), flush=True)

    # equal score on the same cross-sections (not the production F17 score,
    # which averages available factors rather than complete cases)
    for j in range(n):
        m = elig[j] & np.isfinite(Y[j]) & np.isfinite(X[j]).all(axis=1)
        if m.sum():
            S["equal"][j, m] = X[j, m, :].mean(axis=1)

    outbase = Path(args.out)
    outbase.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(outbase) + "_scores.npz",
                        **{m: S[m] for m in args.models},
                        dates=np.array([str(d.date()) for d in dates]),
                        syms=np.load(CACHE, allow_pickle=True)["syms"])
    (Path(str(outbase) + "_weights.json")).write_text(
        json.dumps({"models": args.models, "names": NAMES, "info": info},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for split, lo, hi in [("development_2016_2022", "2016-01-01", "2022-12-31"),
                          ("validation_2023_2026", "2023-01-01", "2026-12-31")]:
        msk = (dates >= lo) & (dates <= hi)
        base_ic = None
        for model in args.models:
            ics, lss = [], []
            for j in np.flatnonzero(msk):
                mm = elig[j] & np.isfinite(Y[j]) & np.isfinite(S[model][j])
                ics.append(rank_ic(S[model][j][mm], Y[j][mm]))
            lss, turns = ls_stats(S[model], Y, elig, dates)
            ic = np.asarray(ics, float)
            ls = np.asarray(lss, float)[msk]
            st = hac(ic)
            if model == "equal":
                base_ic = ic.copy()
            d = hac(ic - base_ic) if base_ic is not None and len(ic) == len(base_ic) else {"mean": None, "t": None}
            ls = ls[np.isfinite(ls)]
            rows.append({
                "split": split, "model": model, "n_ic": st["n"],
                "ic": st["mean"], "ic_t": st["t"],
                "delta_ic": d["mean"], "delta_t": d["t"],
                "ls_mean_21d": float(ls.mean()) if len(ls) else None,
                "ls_ann": float(ls.mean() * 12) if len(ls) else None,
                "ls_sharpe": float(ls.mean() / ls.std(ddof=1) * math.sqrt(12)) if len(ls) > 2 and ls.std(ddof=1) > 0 else None,
                "ls_hit": float((ls > 0).mean()) if len(ls) else None,
                "ls_turnover": float(np.nanmean(turns[msk])) if len(turns) else None,
            })
    df = pd.DataFrame(rows)
    df.to_csv(str(outbase) + "_metrics.csv", index=False)
    print("\n" + df.round(4).to_string(index=False), flush=True)
    (Path(str(outbase) + "_metrics.json")).write_text(
        json.dumps({"target": args.target, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[lab] wrote {outbase}_scores.npz, _weights.json, _metrics.csv", flush=True)


if __name__ == "__main__":
    main()
