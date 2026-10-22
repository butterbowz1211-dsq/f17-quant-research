# Research project author: Gilmour Jia
"""Control: is the policy's return information, or is it just its turnover?

Tier 0 showed the policy beats an equal-weight book of the same names by about
3.6 points a year at full rebalancing -- but it trades 2.5 times as much.  Since
both books earn their returns by being repeatedly pushed back to target, "return
scales with turnover" is a live alternative to "return comes from selection".

This script destroys the selection while preserving everything else: for each
decision date it keeps the exact multiset of target weights and the number of
names, and permutes which name gets which weight.  Same dispersion, same
rebalancing, same turnover, no information.

    python tools/shuffle_control.py
"""
from __future__ import annotations

import argparse
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

CARRIES = [("policy p1", "p1_fold0_carry.npy"),
           ("policy final k=all", "final_fold0_kall_carry.npy"),
           ("policy final k=30", "final_fold0_k30_carry.npy")]


def decision_rows(carry):
    """Rows where the target actually changes; the rest are carried copies."""
    nz = np.flatnonzero((carry > 0).sum(axis=1) > 0)
    if nz.size == 0:
        return []
    dec = [int(nz[0])]
    for a, b in zip(nz[:-1], nz[1:]):
        if not np.array_equal(carry[a], carry[b]):
            dec.append(int(b))
    return dec


def shuffled_carry(carry, seed=0):
    """Same weights, same names held, wrong names."""
    rng = np.random.default_rng(seed)
    dec = decision_rows(carry)
    out = np.zeros_like(carry)
    for i, d in enumerate(dec):
        stop = dec[i + 1] if i + 1 < len(dec) else carry.shape[0]
        row = carry[d].copy()
        pos = np.flatnonzero(row > 0)
        if pos.size:
            row[pos] = rng.permutation(row[pos])
        out[d:stop] = row
    return out, len(dec)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--first-year", type=int, default=2013)
    ap.add_argument("--cap", type=float, default=0.03)
    ap.add_argument("--partial", type=float, default=1.0)
    ap.add_argument("--band", type=float, default=0.0)
    ap.add_argument("--capital", type=float, default=10_000_000.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    prep = build_prep()
    dates = pd.DatetimeIndex(prep["dates"])
    cols = pd.Index([str(s) for s in prep["symbols"]])
    folds = expanding_folds(dates, first_year=args.first_year)
    fold = folds[args.fold]
    keep = dates >= pd.Timestamp("%d-01-01" % args.first_year)
    idx = dates[keep]

    open_ = pd.DataFrame(prep["open"], index=dates, columns=cols).loc[idx]
    close = pd.DataFrame(prep["close"], index=dates, columns=cols).loc[idx]
    adv = pd.DataFrame(prep["amount_lag"], index=dates, columns=cols).loc[idx]
    eligible = pd.DataFrame(prep["universe"], index=dates, columns=cols).fillna(False).loc[idx]
    lk = pd.DataFrame(prep["limit_locked"], index=dates, columns=cols).fillna(False).loc[idx]
    tradable = eligible & (~lk) & np.isfinite(open_) & (open_ > 0)
    locked = pd.DataFrame(False, index=idx, columns=cols)

    rows = []
    for label, fname in CARRIES:
        path = ART / fname
        if not path.exists():
            print("  MISSING %s" % fname, flush=True)
            continue
        real = np.load(path).astype(np.float64)
        shuf, n_dec = shuffled_carry(real, seed=17)
        for tag, carry in (("real", real), ("shuffled", shuf)):
            t0 = time.time()
            targets = pd.DataFrame(carry[keep], index=idx, columns=cols)
            cfg = ex.ExecutionConfig(capital=args.capital, cap=args.cap,
                                     partial_adjustment=args.partial,
                                     no_trade_weight=args.band, price_guard=0.0,
                                     allow_stale=True)
            led = ex.run_target_weight_engine(open_, close, targets, eligible,
                                              tradable, locked, adv, cfg)
            seg = rp.slice_ledger(led, fold.test_start, fold.test_end)
            m = rp.ledger_metrics(seg)
            rows.append(dict(carry=label, variant=tag, decisions=n_dec,
                             cagr=m["cagr"], vol=m["vol"], sharpe=m["sharpe"],
                             maxdd=m["max_drawdown"],
                             turnover=m["annual_two_way_turnover"],
                             cost_bp=m["total_cost_bp_of_capital"],
                             seconds=round(time.time() - t0, 1)))
            print("[%s / %-8s] cagr %+.3f  sharpe %+.3f  turn %5.0f%%  cost %4.0f  (%.0fs)"
                  % (label, tag, m["cagr"], m["sharpe"],
                     m["annual_two_way_turnover"] * 100,
                     m["total_cost_bp_of_capital"], time.time() - t0), flush=True)

    out = Path(args.out) if args.out else ART / ("shuffle_control_fold%d.json" % args.fold)
    out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print()
    print("%-24s %-9s %8s %8s %9s %8s"
          % ("carry", "variant", "CAGR", "Sharpe", "turnover", "cost bp"))
    for r in rows:
        print("%-24s %-9s %+8.3f %+8.3f %8.0f%% %8.0f"
              % (r["carry"][:24], r["variant"], r["cagr"], r["sharpe"],
                 r["turnover"] * 100, r["cost_bp"]))
    print()
    print("wrote " + str(out))
    return rows


if __name__ == "__main__":
    main()
