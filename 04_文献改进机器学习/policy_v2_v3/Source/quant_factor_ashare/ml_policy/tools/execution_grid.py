# Research project author: Gilmour Jia
"""Tier 0: attribute the difference between runs to execution or to the model.

The problem this solves.  Two runs of the same protocol produced +17.9% and
+1.9% on the same year, but they differed in *two* things at once: the
execution rule (full adjustment with no no-trade band vs half adjustment with
a 25bp band) and the trained models (different seeds stopped at different
epochs, holding 27 vs 156 names at the validation read-out).  Neither number
can be quoted until those two are pulled apart.

This script holds the targets fixed and varies only the execution rule, then
does the same for the equal-weight benchmark so the comparison stays fair.

    python tools/execution_grid.py --fold 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))

from qlp import execution as ex  # noqa: E402
from qlp.data import build_prep  # noqa: E402
from qlp.splits import expanding_folds  # noqa: E402

import run_policy as rp  # noqa: E402

# (label, carry file) -- all evaluated on the same dates with the same engine.
CARRIES = [
    ("policy p1 (full adj run)", "p1_fold0_carry.npy"),
    ("policy final k=all", "final_fold0_kall_carry.npy"),
    ("policy final k=30", "final_fold0_k30_carry.npy"),
    ("equal weight (fully invested)", "beq_fold0_carry.npy"),
]

# (partial_adjustment, no_trade_band)
EXEC = [(1.0, 0.0), (1.0, 0.0025), (0.5, 0.0), (0.5, 0.0025)]


def digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--first-year", type=int, default=2013)
    ap.add_argument("--cap", type=float, default=0.03)
    ap.add_argument("--capital", type=float, default=10_000_000.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    prep = build_prep()
    dates = pd.DatetimeIndex(prep["dates"])
    cols = pd.Index([str(s) for s in prep["symbols"]])
    folds = expanding_folds(dates, first_year=args.first_year)
    fold = folds[args.fold]

    # Only the window the carry actually describes, so the engine is not asked
    # to walk three years of an empty book.
    keep = dates >= pd.Timestamp("%d-01-01" % args.first_year)
    idx = dates[keep]
    open_ = pd.DataFrame(prep["open"], index=dates, columns=cols).loc[idx]
    close = pd.DataFrame(prep["close"], index=dates, columns=cols).loc[idx]
    adv = pd.DataFrame(prep["amount_lag"], index=dates, columns=cols).loc[idx]
    uni = pd.DataFrame(prep["universe"], index=dates, columns=cols).fillna(False)
    eligible = uni.loc[idx]
    lk = pd.DataFrame(prep["limit_locked"], index=dates, columns=cols).fillna(False)
    locked_flag = lk.loc[idx]
    tradable = eligible & (~locked_flag) & np.isfinite(open_) & (open_ > 0)
    locked = pd.DataFrame(False, index=idx, columns=cols)

    rows = []
    for label, fname in CARRIES:
        path = ART / fname
        if not path.exists():
            print("  MISSING %s" % fname, flush=True)
            continue
        carry_full = np.load(path).astype(np.float64)
        carry = carry_full[keep]
        dep = carry.sum(axis=1)
        print("[%s] hash %s  max names %d  mean deployed %.3f"
              % (label, digest(carry_full), int((carry > 1e-4).sum(axis=1).max()),
                 float(dep.mean())), flush=True)
        targets = pd.DataFrame(carry, index=idx, columns=cols)
        for partial, band in EXEC:
            t0 = time.time()
            cfg = ex.ExecutionConfig(capital=args.capital, cap=args.cap,
                                     partial_adjustment=partial,
                                     no_trade_weight=band, price_guard=0.0,
                                     allow_stale=True)
            led = ex.run_target_weight_engine(open_, close, targets, eligible,
                                              tradable, locked, adv, cfg)
            seg = rp.slice_ledger(led, fold.test_start, fold.test_end)
            m = rp.ledger_metrics(seg)
            row = dict(carry=label, partial=partial, band=band,
                       cagr=m["cagr"], vol=m["vol"], sharpe=m["sharpe"],
                       maxdd=m["max_drawdown"],
                       turnover=m["annual_two_way_turnover"],
                       cost_bp=m["total_cost_bp_of_capital"],
                       cash=m["mean_cash_weight"],
                       holdings=m["mean_holdings"],
                       seconds=round(time.time() - t0, 1))
            rows.append(row)
            print("   partial %.1f band %5.4f -> cagr %+.3f sharpe %+.3f"
                  " turn %5.0f%% cost %4.0f  (%.0fs)"
                  % (partial, band, m["cagr"], m["sharpe"],
                     m["annual_two_way_turnover"] * 100,
                     m["total_cost_bp_of_capital"], time.time() - t0), flush=True)

    out = Path(args.out) if args.out else ART / ("exec_grid_fold%d.json" % args.fold)
    out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    print()
    print("%-32s %5s %6s %8s %8s %9s %8s %7s"
          % ("carry", "part", "band", "CAGR", "Sharpe", "turnover", "cost bp", "cash"))
    for r in rows:
        print("%-32s %5.1f %6.4f %+8.3f %+8.3f %8.0f%% %8.0f %7.3f"
              % (r["carry"][:32], r["partial"], r["band"], r["cagr"], r["sharpe"],
                 r["turnover"] * 100, r["cost_bp"], r["cash"]))
    print()
    print("wrote " + str(out))
    return rows


if __name__ == "__main__":
    main()
