# Research project author: Gilmour Jia
"""Read a trained holdout ensemble out into pre-registered portfolios.

The model is trained once; this tool never re-fits or tunes anything.  It reads
the per-seed score/weight tensors saved by run_policy.py and turns them into
five portfolios on the same eligible rows:

  policy_softmax_all  the model's own dense softmax weights (cap applied per seed)
  ew_top500           equal weight in the 500 highest ensemble scores
  ew_top10pct         equal weight in the top score decile
  vw_top10pct         market-cap weight in the same top decile
  ew_all              equal weight in every eligible name
  vw_all              market-cap weight in every eligible name

All six go through the same hard event-driven executor and the same 2023+
window, so the comparison is a read-out comparison, not a model comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qlp import weights as wt
from qlp.data import build_prep, load_panel
from qlp.splits import holdout_fold
from run_policy import engine_inputs, hold_forward, ledger_metrics, slice_ledger


def _load_seed(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _ensemble_scores(arrays: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    t = arrays[0]["t"].astype(np.int64)
    for a in arrays[1:]:
        if not np.array_equal(a["t"].astype(np.int64), t):
            raise ValueError("score files were saved on different decision dates")
    stack = np.stack([a["z"].astype(np.float64) for a in arrays], axis=0)
    counts = np.isfinite(stack).sum(axis=0)
    total = np.nansum(stack, axis=0)
    out = np.full(counts.shape, np.nan, dtype=np.float64)
    np.divide(total, counts, out=out, where=counts > 0)
    return t, out


def _project_row(row: np.ndarray, cap: float, total: float | None = None) -> np.ndarray:
    nz = np.flatnonzero(row > 0)
    out = np.zeros_like(row, dtype=np.float64)
    if nz.size == 0:
        return out
    if total is None:
        total = float(row[nz].sum())
    out[nz] = wt.project_capped_simplex(
        row[nz].astype(np.float64), np.zeros(nz.size),
        np.full(nz.size, cap), total=float(total))[0]
    return out


def _top_k(score: np.ndarray, universe: np.ndarray, k: int) -> np.ndarray:
    active = np.flatnonzero(np.isfinite(score) & universe)
    if active.size == 0 or k <= 0:
        return np.zeros(0, dtype=np.int64)
    k = min(int(k), int(active.size))
    order = np.argpartition(-score[active], k - 1)[:k]
    return active[order]


def _market_cap(prep: dict) -> pd.DataFrame:
    panel = load_panel()
    # Only the score window needs shares; the full panel is 13.4M rows and the
    # pivot is the memory peak of this tool.
    panel = panel.loc[panel["date"] >= pd.Timestamp("2022-01-01")]
    shares = panel.pivot_table(index="date", columns="symbol",
                               values="outstanding_share", aggfunc="last",
                               observed=True)
    shares = shares.reindex(index=pd.DatetimeIndex(prep["dates"]),
                            columns=pd.Index(prep["symbols"]))
    mcap = prep["raw_close"] * shares
    return mcap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="policy_holdout_2013_2020")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--cap", type=float, default=0.03)
    ap.add_argument("--capital", type=float, default=1e7)
    ap.add_argument("--rebalance", type=int, default=21)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-vw", action="store_true",
                    help="skip the market-cap read-out (useful before the panel is loadable)")
    args = ap.parse_args()

    art = ROOT / "artifacts"
    paths = [art / ("%s_fold%d_seed%d_scores.npz" % (args.tag, args.fold, s))
             for s in args.seeds]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("missing score files: " + ", ".join(missing))

    arrays = [_load_seed(p) for p in paths]
    t_all, score_all = _ensemble_scores(arrays)
    prep = build_prep()
    dates = pd.DatetimeIndex(prep["dates"])
    symbols = pd.Index(prep["symbols"])
    universe = prep["universe"].to_numpy(dtype=bool)
    n_dates, n_sym = len(dates), len(symbols)
    fold = holdout_fold(dates, first_year=2013, train_end=2020,
                        valid_years=2, test_start=2023)

    # The score files start in 2022 so run_policy could reuse one graph walk.
    # The read-out is a 2023+ test and must start with cash on the first test
    # decision, not inherit a 2022 book that was part of model selection.
    keep = dates[t_all] >= fold.test_start
    keep_idx = np.flatnonzero(keep)
    t_list, score = t_all[keep], score_all[keep]
    for a in arrays:
        for key in ("z", "w", "rets", "t"):
            a[key] = a[key][keep_idx]
    print("[readout] %d test decision dates x %d symbols, %d seeds"
          % (t_list.size, n_sym, len(paths)), flush=True)
    mcap = None if args.no_vw else _market_cap(prep)

    # The executor only needs the price axes and the point-in-time universe.
    a = SimpleNamespace(dates=dates.to_numpy(), symbols=symbols,
                        universe=universe, rb=args.rebalance,
                        with_prev_weight=False)

    grids = {name: np.zeros((n_dates, n_sym), dtype=np.float64)
             for name in ("policy_softmax_all", "ew_top500", "ew_top10pct",
                          "vw_top10pct", "ew_all", "vw_all")}
    coverage = {"n_dates": int(t_list.size), "mean_eligible": 0.0,
                "decile_names": [], "top500_names": []}
    eligible_counts = []
    decile_counts = []
    top500_counts = []

    for i, t in enumerate(t_list):
        t = int(t)
        active = np.isfinite(score[i]) & universe[t]
        n_active = int(active.sum())
        if n_active == 0:
            continue
        eligible_counts.append(n_active)
        decile_k = max(1, int(np.floor(0.1 * n_active)))
        top_k = min(500, n_active)
        decile_counts.append(decile_k)
        top500_counts.append(top_k)

        # 1) model's own dense softmax, cap-projected per seed then averaged.
        policy_rows = []
        for s in arrays:
            w = s["w"][i].astype(np.float64)
            policy_rows.append(_project_row(w, args.cap, total=float(w.sum())))
        grids["policy_softmax_all"][t] = np.mean(policy_rows, axis=0)

        # 2) equal weight top 500 by the ensemble score.
        idx = _top_k(score[i], universe[t], top_k)
        if idx.size:
            grids["ew_top500"][t, idx] = 1.0 / idx.size

        # 3) equal weight top decile.
        idx = _top_k(score[i], universe[t], decile_k)
        if idx.size:
            grids["ew_top10pct"][t, idx] = 1.0 / idx.size

        # 4) market-cap weight in the same top decile.
        if not args.no_vw:
            idx = _top_k(score[i], universe[t], decile_k)
            if idx.size:
                cap_row = np.asarray(mcap.iloc[t, idx], dtype=np.float64)
                bad = ~np.isfinite(cap_row) | (cap_row <= 0)
                if bad.any():
                    cap_row[bad] = np.nanmedian(cap_row)
                if np.isfinite(cap_row).all() and cap_row.sum() > 0:
                    raw = np.zeros(n_sym, dtype=np.float64)
                    raw[idx] = cap_row / cap_row.sum()
                    grids["vw_top10pct"][t] = _project_row(raw, args.cap, total=1.0)

        # 5) equal weight every eligible name.
        idx = np.flatnonzero(active)
        grids["ew_all"][t, idx] = 1.0 / idx.size

        # 6) market-cap weight every eligible name: the fair cap-weighted
        # benchmark, since ew_all is a small-cap-heavy 1/N book.
        if not args.no_vw:
            cap_row = np.asarray(mcap.iloc[t, idx], dtype=np.float64)
            bad = ~np.isfinite(cap_row) | (cap_row <= 0)
            if bad.any():
                cap_row[bad] = np.nanmedian(cap_row)
            if np.isfinite(cap_row).all() and cap_row.sum() > 0:
                raw = np.zeros(n_sym, dtype=np.float64)
                raw[idx] = cap_row / cap_row.sum()
                grids["vw_all"][t] = _project_row(raw, args.cap, total=1.0)

    if eligible_counts:
        coverage["mean_eligible"] = float(np.mean(eligible_counts))
    coverage["decile_names"] = [int(v) for v in decile_counts]
    coverage["top500_names"] = [int(v) for v in top500_counts]

    print("[readout] test %s..%s" % (fold.test_start, fold.test_end), flush=True)

    results = {}
    for name, grid in grids.items():
        start = time.time()
        carry = hold_forward(grid, t_list, args.rebalance)
        led = engine_inputs(prep, a, carry, cap=args.cap, capital=args.capital,
                            partial=1.0, band=0.0, price_guard=0.0, strict=False)
        seg = slice_ledger(led, fold.test_start, fold.test_end)
        met = ledger_metrics(seg)
        daily = seg["daily"]
        csv = art / ("%s_%s_daily.csv" % (args.tag, name))
        daily.to_csv(csv, index=True)
        results[name] = {"metrics": met, "daily_csv": csv.name,
                         "elapsed_s": round(time.time() - start, 1)}
        print("  %-20s CAGR %7.2f%%  Sharpe %6.3f  maxDD %6.2f%%  "
              "turnover %6.0f%%  cost %5.0fbp  names %5.1f  cash %.3f"
              % (name, 100.0 * met.get("cagr", 0.0), met.get("sharpe", 0.0),
                 100.0 * met.get("max_drawdown", 0.0),
                 100.0 * met.get("annual_two_way_turnover", 0.0),
                 met.get("total_cost_bp_of_capital", 0.0),
                 met.get("mean_holdings", 0.0),
                 met.get("mean_cash_weight", 0.0)), flush=True)

    report = {
        "tag": args.tag, "fold": args.fold, "seeds": args.seeds,
        "cap": args.cap, "capital": args.capital, "rebalance": args.rebalance,
        "test_start": str(fold.test_start.date()),
        "test_end": str(fold.test_end.date()),
        "contaminated": bool(fold.contaminated),
        "coverage": coverage, "readouts": results,
    }
    out = Path(args.out) if args.out else art / ("%s_readout.json" % args.tag)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[readout] wrote %s" % out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
