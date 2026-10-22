# Research project author: Gilmour Jia
"""Same-protocol baselines: equal weight and top-N rank-and-hold.

Every baseline here is built from the *same* point-in-time universe the policy
uses and executed by the *same* engine, so a comparison is about the weighting
rule and nothing else.  This is the comparison the earlier F17 run could not
make: that one used a different universe (LSY instead of top-1000 turnover)
and a different preprocessing chain, so any difference would have been
unattributable.

Usage:
    python run_baselines.py --kind equal --folds 0 1 --tag beq
    python run_baselines.py --kind topn --n 50 --folds 0 1 --tag btop50
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
sys.path.insert(0, str(HERE))

from qlp.baselines import equal_weight_grid, topn_grid  # noqa: E402
from qlp.data import build_prep  # noqa: E402
from qlp.splits import expanding_folds  # noqa: E402

import run_policy as rp  # noqa: E402

SCORE_CACHE = Path(
    r"workspace/review_workspace/quant_factor_ashare_review/linkage_round2/score_F17.pkl")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["equal", "topn", "f17"], default="equal")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--cash", type=float, default=0.0,
                    help="fraction of NAV held back before weighting (equal only)")
    ap.add_argument("--buffer", type=float, default=0.6)
    ap.add_argument("--folds", type=int, nargs="*", default=[0])
    ap.add_argument("--rebalance", type=int, default=21)
    ap.add_argument("--first-year", type=int, default=2013)
    ap.add_argument("--cap", type=float, default=0.03)
    ap.add_argument("--partial", type=float, default=1.0)
    ap.add_argument("--band", type=float, default=0.0)
    ap.add_argument("--capital", type=float, default=10_000_000.0)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    prep = build_prep()
    dates = pd.DatetimeIndex(prep["dates"])
    folds = expanding_folds(dates, first_year=args.first_year)

    # A light stand-in for the full dataset: the baseline rules only need the
    # universe, the tradability mask and the axes.
    class Panel:
        pass
    a = Panel()
    a.dates = dates
    a.symbols = np.asarray([str(s) for s in prep["symbols"]])
    uni = pd.DataFrame(prep["universe"], index=dates, columns=a.symbols).fillna(False)
    sus = pd.DataFrame(prep["suspended"], index=dates, columns=a.symbols).fillna(False)
    lock = pd.DataFrame(prep["limit_locked"], index=dates, columns=a.symbols).fillna(False)
    a.universe = uni.to_numpy()
    a.tradable = (uni & (~sus) & (~lock)).to_numpy()
    a.rb = args.rebalance
    a.warmup = 260

    score = None
    if args.kind == "f17":
        if not SCORE_CACHE.exists():
            raise SystemExit("missing %s" % SCORE_CACHE)
        score = pd.read_pickle(SCORE_CACHE).reindex(index=dates, columns=a.symbols)
        score = score.to_numpy(dtype=np.float64)

    report = dict(tag=args.tag, kind=args.kind, config=vars(args), folds=[])
    for k in args.folds:
        fold = folds[k]
        lo = int(np.flatnonzero(dates >= fold.train_start)[0])
        hi = int(min(np.flatnonzero(dates <= fold.test_end)[-1],
                     len(dates) - a.rb - 1))
        t_eval = np.arange(max(lo, a.warmup), hi, a.rb, dtype=np.int64)
        t0 = time.time()
        if args.kind == "equal":
            grid, diag = equal_weight_grid(a, t_eval, cap=args.cap, cash=args.cash)
        else:
            grid = topn_grid(a, score, t_eval, n=args.n, cap=args.cap,
                             buffer=args.buffer)
            diag = []
        carry = rp.hold_forward(grid, t_eval, a.rb)
        open_ = pd.DataFrame(prep["open"], index=dates, columns=a.symbols)
        close = pd.DataFrame(prep["close"], index=dates, columns=a.symbols)
        adv = pd.DataFrame(prep["amount_lag"], index=dates, columns=a.symbols)
        eligible = pd.DataFrame(a.universe, index=dates, columns=a.symbols)
        tradable = pd.DataFrame(a.tradable, index=dates, columns=a.symbols)
        locked = pd.DataFrame(False, index=dates, columns=a.symbols)
        targets = pd.DataFrame(carry, index=dates, columns=a.symbols)
        cfg = rp.ex.ExecutionConfig(capital=args.capital, cap=args.cap,
                                    partial_adjustment=args.partial,
                                    no_trade_weight=args.band, price_guard=0.0,
                                    allow_stale=not args.strict)
        led = rp.ex.run_target_weight_engine(open_, close, targets, eligible,
                                             tradable, locked, adv, cfg)
        seg = rp.slice_ledger(led, fold.test_start, fold.test_end)
        m = rp.ledger_metrics(seg)
        report["folds"].append(dict(index=k, label=fold.label(),
                                    contamination=("CONTAMINATED"
                                                    if fold.contaminated else "untouched"),
                                    n_decisions=int(len(t_eval)), test=m,
                                    seconds=round(time.time() - t0, 1)))
        tag = "%s_fold%d" % (args.tag, k)
        np.save(HERE / "artifacts" / (tag + "_carry.npy"), carry.astype(np.float32))
        seg["daily"].to_csv(HERE / "artifacts" / (tag + "_daily.csv"))
        print("[%s fold %d] %s cagr %.3f vol %.3f sharpe %.3f maxdd %.3f turnover %.1f%%"
              % (args.kind, k, report["folds"][-1]["contamination"], m["cagr"], m["vol"],
                 m["sharpe"], m["max_drawdown"],
                 m["annual_two_way_turnover"] * 100), flush=True)

    out = Path(args.out) if args.out else HERE / "artifacts" / ("%s_report.json" % args.tag)
    out.write_text(json.dumps(rp._jsonable(report), indent=2), encoding="utf-8")
    print("[run] wrote " + str(out), flush=True)
    return report


if __name__ == "__main__":
    main()
