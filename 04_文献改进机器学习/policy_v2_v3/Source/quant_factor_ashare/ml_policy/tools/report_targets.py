# Research project author: Gilmour Jia
"""Evaluate pre-computed target weights with the hard executor.

Why this exists: every policy number must be comparable to a baseline that
ran through the *same* engine, the same universe and the same period.  The
cheapest way to guarantee that is to make the reporting path itself shared.

Usage:
    python tools/report_targets.py --spec p1 --fold 0

It reads artifacts/<spec>_fold<k>_carry.npy (the seed-averaged targets),
runs the ledger, and writes artifacts/<spec>_fold<k>_metrics.json plus a
ledger CSV, so a report can be checked without retraining.
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
sys.path.insert(0, str(ROOT))

from qlp import execution as ex  # noqa: E402
from qlp.data import build_prep  # noqa: E402
from qlp.splits import expanding_folds  # noqa: E402

import run_policy as rp  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="tag used when training")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--first-year", type=int, default=2013)
    ap.add_argument("--cap", type=float, default=0.03)
    ap.add_argument("--partial", type=float, default=1.0)
    ap.add_argument("--band", type=float, default=0.0)
    ap.add_argument("--capital", type=float, default=10_000_000.0)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)

    prep = build_prep()
    dates = pd.DatetimeIndex(prep["dates"])
    cols = pd.Index([str(s) for s in prep["symbols"]])
    folds = expanding_folds(dates, first_year=args.first_year)
    fold = folds[args.fold]

    carry_path = ROOT / "artifacts" / ("%s_fold%d_carry.npy" % (args.spec, args.fold))
    if not carry_path.exists():
        raise SystemExit("missing %s" % carry_path)
    carry = np.load(carry_path).astype(np.float64)
    if carry.shape != (len(dates), len(cols)):
        raise SystemExit("carry grid shape %s does not match the panel" % (carry.shape,))

    open_ = pd.DataFrame(prep["open"], index=dates, columns=cols)
    close = pd.DataFrame(prep["close"], index=dates, columns=cols)
    adv = pd.DataFrame(prep["amount_lag"], index=dates, columns=cols)
    eligible = pd.DataFrame(prep["universe"], index=dates, columns=cols).fillna(False)
    locked_flag = pd.DataFrame(prep["limit_locked"], index=dates, columns=cols).fillna(False)
    tradable = eligible & (~locked_flag) & np.isfinite(open_) & (open_ > 0)
    locked = pd.DataFrame(False, index=dates, columns=cols)
    targets = pd.DataFrame(carry, index=dates, columns=cols)

    cfg = ex.ExecutionConfig(capital=args.capital, cap=args.cap,
                             partial_adjustment=args.partial,
                             no_trade_weight=args.band, price_guard=0.0,
                             allow_stale=not args.strict)
    led = ex.run_target_weight_engine(open_, close, targets, eligible, tradable,
                                      locked, adv, cfg)
    seg = rp.slice_ledger(led, fold.test_start, fold.test_end)
    val = rp.slice_ledger(led, fold.valid_start, fold.valid_end)
    out = dict(spec=args.spec, fold=args.fold, label=fold.label(),
               contamination="CONTAMINATED" if fold.contaminated else "untouched",
               config=vars(args),
               test=rp.ledger_metrics(seg), validation=rp.ledger_metrics(val))
    tag = "%s_fold%d" % (args.spec, args.fold)
    (ROOT / "artifacts" / (tag + "_metrics.json")).write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    seg["daily"].to_csv(ROOT / "artifacts" / (tag + "_daily.csv"))
    print(json.dumps({k: v for k, v in out.items() if k != "config"},
                     indent=2, default=str))
    return out


if __name__ == "__main__":
    main()
