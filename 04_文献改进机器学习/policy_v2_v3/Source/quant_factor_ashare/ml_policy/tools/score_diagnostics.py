# Research project author: Gilmour Jia
"""Tier 1: does the policy carry cross-sectional information at all?

Two questions, one run:

1. What is the policy's own rank IC against the realised 21-day forward return,
   by year, and how persistent is the ranking?
2. Does a *per-stock* supervised model on exactly the same features do better?
   This is the control that separates "our training lost the signal" from "the
   feature set has no signal".  The policy is supervised with one reward per
   decision date, so its effective sample size is the number of periods (~70 in
   fold 0).  Ridge and a random forest get one target per stock instead, so their
   effective sample size is stock-days (~56,000).  Same inputs, same labels, same
   test dates.

Everything downstream of a score is a per-date 1-D vector aligned with that
date's eligible names, because mixing a full date-by-symbol matrix with a
truncated cross-section is the bug that already cost this run once.

The F17 score is measured on the same dates and labels as a third yardstick.

    python tools/score_diagnostics.py --folds 0 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))

from qlp.data import build_prep  # noqa: E402
from qlp.dataset import (DatasetConfig, assemble, make_sample,  # noqa: E402
                          standardize_state)
from qlp.policy import PolicyConfig  # noqa: E402
from qlp.splits import expanding_folds  # noqa: E402

import run_policy as rp  # noqa: E402

F17_CACHE = Path(
    r"workspace/review_workspace/quant_factor_ashare_review/linkage_round2/score_F17.pkl")


# ------------------------------------------------------------------ ranks ---

def rank_ic(x, y):
    """Spearman correlation on the names where both are known."""
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 30:
        return float("nan")
    # pandas 3 hands back a read-only view here; in-place centring needs a copy.
    rx = pd.Series(x[m]).rank().to_numpy(dtype=float).copy()
    ry = pd.Series(y[m]).rank().to_numpy(dtype=float).copy()
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def summarise(ics, years):
    a = np.array(ics, dtype=float)
    ok = np.isfinite(a)
    if ok.sum() < 3:
        return dict(n=int(ok.sum()))
    a = a[ok]
    mean = float(a.mean())
    sd = float(a.std(ddof=1))
    per_year = 12.0                      # 21 trading days between decisions
    yrs = np.array([y for v, y in zip(ics, years) if np.isfinite(v)])
    by_year = {}
    for y in sorted(set(int(v) for v in years)):
        sel = yrs == y
        if sel.sum() >= 2:
            by_year[y] = dict(n=int(sel.sum()), ic=float(a[sel].mean()))
    return dict(n=int(a.size), ic_mean=mean, ic_sd=sd,
                ic_ir=float(mean / sd * np.sqrt(per_year)) if sd > 1e-12 else 0.0,
                t_stat=float(mean / (sd / np.sqrt(a.size))) if sd > 1e-12 else 0.0,
                hit=float((a > 0).mean()), by_year=by_year)


def lag1_autocorr(rows, ids_list):
    out = []
    pairs = zip(zip(rows[:-1], ids_list[:-1]), zip(rows[1:], ids_list[1:]))
    for (r1, i1), (r2, i2) in pairs:
        _, a1, a2 = np.intersect1d(i1, i2, return_indices=True)
        if a1.size < 30:
            continue
        c = rank_ic(r1[a1], r2[a2])
        if np.isfinite(c):
            out.append(c)
    return (float(np.mean(out)) if out else float("nan")), len(out)


def decile_spread(rows, fwds):
    tops, bots = [], []
    for r, f in zip(rows, fwds):
        m = np.isfinite(r) & np.isfinite(f)
        if int(m.sum()) < 50:
            continue
        rv, fv = r[m], f[m]
        k = max(1, rv.size // 10)
        order = np.argsort(-rv, kind="stable")
        tops.append(float(np.mean(fv[order[:k]])))
        bots.append(float(np.mean(fv[order[-k:]])))
    if not tops:
        return dict(n=0)
    t, b = np.array(tops), np.array(bots)
    return dict(n=int(t.size), top=float(t.mean()), bottom=float(b.mean()),
                spread=float(t.mean() - b.mean()))


def topn_overlap(rows, ids_list, k=30):
    ov = []
    pairs = zip(zip(rows[:-1], ids_list[:-1]), zip(rows[1:], ids_list[1:]))
    for (r1, i1), (r2, i2) in pairs:
        def top(r, i):
            m = np.isfinite(r)
            if int(m.sum()) < k:
                return set()
            return set(i[m][np.argsort(-r[m], kind="stable")[:k]].tolist())

        a, b = top(r1, i1), top(r2, i2)
        if a and b:
            ov.append(len(a & b) / float(k))
    return (float(np.mean(ov)) if ov else float("nan")), len(ov)


def block(rows, ids_list, fwds, years, test_mask):
    ics = [rank_ic(r, f) for r, f in zip(rows, fwds)]
    t_ics = [v for v, tm in zip(ics, test_mask) if tm]
    t_years = [y for y, tm in zip(years, test_mask) if tm]
    return dict(all=summarise(ics, years), test=summarise(t_ics, t_years),
                autocorr=lag1_autocorr(rows, ids_list),
                decile=decile_spread(rows, fwds),
                top30=topn_overlap(rows, ids_list, 30))


# --------------------------------------------------------- policy scores ---

def dump_scores(policy, a, t_list, device):
    """Raw logits (pre-softmax, pre-cap), one aligned vector per decision date."""
    policy.eval()
    rows, ids_list, fwds, samples = [], [], [], []
    with torch.no_grad():
        for t in t_list:
            t = int(t)
            s = make_sample(a, t, slot=t // max(1, int(a.rb)))
            x = torch.from_numpy(s.feat).to(device)
            rows.append(policy.logits(x)[:-1].cpu().numpy().astype(np.float64))
            ids_list.append(s.ids)
            fwds.append(s.returns)
            samples.append(s)
    return rows, ids_list, fwds, samples


# ------------------------------------------------- per-stock supervision ---

def demean_within(samples, values):
    """Centre the target inside each decision date.

    The question is whether a name can be ranked against its peers on the same
    day.  Leaving the market's average move in the target lets a model score
    well by forecasting the market, which is not what a cross-sectional
    portfolio needs.
    """
    out = values.astype(np.float64).copy()
    i = 0
    for s in samples:
        n = len(s)
        v = out[i:i + n]
        ok = np.isfinite(v)
        if ok.any():
            out[i:i + n] = np.where(ok, v - v[ok].mean(), 0.0)
        i += n
    return out


def pool(samples):
    X = np.vstack([s.feat for s in samples]).astype(np.float64)
    y = np.concatenate([s.returns for s in samples]).astype(np.float64)
    return X, demean_within(samples, y)


def split_pred(pred, samples):
    out, i = [], 0
    for s in samples:
        n = len(s)
        out.append(pred[i:i + n])
        i += n
    return out


def per_stock_baseline(train_s, valid_s, eval_s, seed=0, verbose=True):
    """Ridge and random forest, one target per stock on the same features.

    This is the architecture control.  Both models see exactly the design matrix
    the policy sees; the only thing that changes is the granularity of the
    supervision, from one reward per decision date to one label per stock.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    Xtr, ytr = pool(train_s)
    Xva, _ = pool(valid_s)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    if verbose:
        print("      per-stock fit: %d rows x %d features" % Xtr.shape, flush=True)

    best = None
    for alpha in (1.0, 10.0, 100.0, 1000.0):
        model = Ridge(alpha=alpha).fit(Xtr, ytr)
        pred = split_pred(model.predict(Xva), valid_s)
        ics = [rank_ic(p, s.returns) for p, s in zip(pred, valid_s)]
        sc = float(np.nanmean(ics)) if np.isfinite(ics).any() else float("nan")
        if verbose:
            print("        ridge alpha %6.1f -> valid IC %+0.4f" % (alpha, sc), flush=True)
        if best is None or (np.isfinite(sc) and sc > best[1]):
            best = (alpha, sc)

    ridge = Ridge(alpha=best[0]).fit(Xtr, ytr)
    rf = RandomForestRegressor(n_estimators=120, max_depth=10,
                               min_samples_leaf=50, n_jobs=-1, random_state=seed)
    t0 = time.time()
    rf.fit(Xtr, ytr)
    if verbose:
        print("        random forest fitted in %.0fs" % (time.time() - t0), flush=True)

    Xev = (np.vstack([s.feat for s in eval_s]).astype(np.float64) - mu) / sd
    out = {}
    for name, model in (("ridge", ridge), ("rf", rf)):
        out[name] = split_pred(model.predict(Xev), eval_s)
    return out, dict(ridge_alpha=best[0], ridge_valid_ic=best[1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rebalance", type=int, default=21)
    ap.add_argument("--first-year", type=int, default=2013)
    ap.add_argument("--cap", type=float, default=0.03)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default=str(ART / "score_diagnostics.json"))
    args = ap.parse_args(argv)

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print("[diag] device=%s" % device, flush=True)
    prep = build_prep()
    t0 = time.time()
    a = assemble(prep, DatasetConfig(rebalance=args.rebalance))
    print("[diag] assembled in %.1fs" % (time.time() - t0), flush=True)
    folds = expanding_folds(a.dates, first_year=args.first_year)

    cfg = PolicyConfig(n_own=int(a.own.shape[2]) + 1,
                       n_neighbour=int(a.peer.shape[2]),
                       n_market=int(a.mkt.shape[1]), n_state=3,
                       cap=args.cap, temperature=1.0)
    n_sym = len(a.symbols)

    f17 = None
    if F17_CACHE.exists():
        f17 = pd.read_pickle(F17_CACHE).reindex(index=pd.DatetimeIndex(a.dates),
                                                columns=pd.Index(a.symbols))
        print("[diag] F17 yardstick loaded", flush=True)

    report = dict(config=dict(vars(args)), folds=[])
    for fidx in args.folds:
        fold = folds[fidx]
        print("[diag] fold %d: %s" % (fidx, fold.label()), flush=True)
        dates = pd.DatetimeIndex(a.dates)
        tr = np.flatnonzero((dates >= fold.train_start) & (dates <= fold.train_end))
        va = np.flatnonzero((dates >= fold.valid_start) & (dates <= fold.valid_end))
        te = np.flatnonzero((dates >= fold.test_start) & (dates <= fold.test_end))
        a.state_scale = standardize_state(a.mkt, slice(int(tr[0]), int(tr[-1]) + 1))
        cache = {}
        tr_t = a.decision_index(int(tr[0]), int(tr[-1]) + 1)
        va_t = a.decision_index(int(va[0]), int(va[-1]) + 1)
        train_s = rp.build_samples(a, tr_t, cache)
        valid_s = rp.build_samples(a, va_t, cache)

        models, seed_info = [], []
        for seed in args.seeds:
            res = rp.train_one_seed(train_s, valid_s, cfg, seed=int(seed),
                                    epochs=args.epochs, lr=args.lr,
                                    batch=args.batch, patience=args.patience,
                                    temp_end=0.1, rb=args.rebalance,
                                    n_symbols=n_sym, device=device, verbose=False)
            models.append(res["policy"])
            seed_info.append(dict(seed=int(seed), best_epoch=res["best_epoch"],
                                  valid_score=res["best_valid_score"],
                                  valid_ir=res["best_valid_ir"],
                                  valid_breadth=(res["best_valid"] or {}).get("mean_breadth")))
            print("   seed %d: epoch %d score %+.3f IR %+.3f breadth %.0f"
                  % (seed, res["best_epoch"], res["best_valid_score"],
                     res["best_valid_ir"],
                     (res["best_valid"] or {}).get("mean_breadth", 0.0)), flush=True)

        rep = rp.train_one_seed(train_s, valid_s, cfg, seed=int(args.seeds[0]),
                                epochs=args.epochs, lr=args.lr, batch=args.batch,
                                patience=args.patience, temp_end=0.1,
                                rb=args.rebalance, n_symbols=n_sym,
                                device=device, verbose=False)
        repro = dict(seed=int(args.seeds[0]), first_epoch=seed_info[0]["best_epoch"],
                     repeat_epoch=rep["best_epoch"],
                     first_score=seed_info[0]["valid_score"],
                     repeat_score=rep["best_valid_score"])
        print("   reproducibility: epoch %d vs %d, score %+.4f vs %+.4f"
              % (repro["first_epoch"], repro["repeat_epoch"],
                 repro["first_score"], repro["repeat_score"]), flush=True)

        ev_lo = int(tr[0])
        ev_hi = int(min(int(te[-1]), len(dates) - int(args.rebalance) - 1))
        t_eval = a.decision_index(ev_lo, ev_hi + 1)
        per_seed, ids_list, fwds, eval_s = [], None, None, None
        for m in models:
            rows, ids_list, fwds, eval_s = dump_scores(m, a, t_eval, device)
            per_seed.append(rows)
        ens = [np.mean([per_seed[j][k] for j in range(len(models))], axis=0)
               for k in range(len(t_eval))]

        years = [int(pd.Timestamp(a.dates[t]).year) for t in t_eval]
        test_mask = [bool(pd.Timestamp(fold.test_start) <= pd.Timestamp(a.dates[t])
                          <= pd.Timestamp(fold.test_end)) for t in t_eval]

        blocks = dict(ensemble=block(ens, ids_list, fwds, years, test_mask))
        for j, info in enumerate(seed_info):
            blocks["seed%d" % info["seed"]] = block(per_seed[j], ids_list, fwds,
                                                    years, test_mask)
        if f17 is not None:
            f17_rows = [f17.iloc[int(t)].to_numpy(dtype=np.float64)[ids]
                        for t, ids in zip(t_eval, ids_list)]
            blocks["F17"] = block(f17_rows, ids_list, fwds, years, test_mask)

        extra = {}
        if not args.no_baseline:
            try:
                base, info = per_stock_baseline(train_s, valid_s, eval_s, seed=0)
                for name, rows in base.items():
                    blocks[name] = block(rows, ids_list, fwds, years, test_mask)
                extra["per_stock"] = info
            except Exception as exc:
                print("      per-stock baseline failed: %r" % (exc,), flush=True)
                extra["per_stock_error"] = repr(exc)

        report["folds"].append(dict(index=fidx, label=fold.label(),
                                    contamination=("CONTAMINATED"
                                                    if fold.contaminated else "untouched"),
                                    test_window=[str(pd.Timestamp(fold.test_start).date()),
                                                 str(pd.Timestamp(fold.test_end).date())],
                                    n_decisions=int(len(t_eval)),
                                    n_train=int(len(train_s)), n_valid=int(len(valid_s)),
                                    seeds=seed_info, reproducibility=repro, extra=extra,
                                    blocks=blocks))
        print(_format_fold(report["folds"][-1]), flush=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str),
                                  encoding="utf-8")

    Path(args.out).write_text(json.dumps(report, indent=2, default=str),
                              encoding="utf-8")
    print()
    print(_format_table(report))
    print("[diag] wrote " + args.out)
    return report


def _format_fold(f):
    lines = ["   IC on the test window, ranked by source:"]
    lines.append("      %-10s %8s %8s %8s %8s %10s %9s"
                 % ("source", "IC", "IC-IR", "t-stat", "hit", "lag1-rho", "top30-ovl"))
    for name, b in f["blocks"].items():
        t = b["test"]
        if not t.get("n"):
            continue
        lines.append("      %-10s %+8.4f %8.3f %8.2f %8.2f %+10.3f %9.3f"
                     % (name, t["ic_mean"], t["ic_ir"], t["t_stat"], t["hit"],
                        b["autocorr"][0], b["top30"][0]))
    d = f["blocks"]["ensemble"]["decile"]
    if d.get("n"):
        lines.append("      ensemble decile spread per 21d: %+.4f (top %+.4f / bottom %+.4f)"
                     % (d["spread"], d["top"], d["bottom"]))
    if f.get("extra", {}).get("per_stock"):
        e = f["extra"]["per_stock"]
        lines.append("      per-stock ridge alpha %.1f (valid IC %+.4f)"
                     % (e["ridge_alpha"], e["ridge_valid_ic"]))
    return chr(10).join(lines)


def _format_table(report):
    out = ["", "fold  source      IC      IC-IR   t-stat   hit   lag1-rho  top30-ovl"]
    for f in report["folds"]:
        for name, b in f["blocks"].items():
            t = b["test"]
            if not t.get("n"):
                continue
            out.append("%-5s %-11s %+.4f %7.3f %8.2f %6.2f %+9.3f %10.3f"
                       % (f["index"], name, t["ic_mean"], t["ic_ir"], t["t_stat"],
                          t["hit"], b["autocorr"][0], b["top30"][0]))
    return chr(10).join(out)


if __name__ == "__main__":
    main()
