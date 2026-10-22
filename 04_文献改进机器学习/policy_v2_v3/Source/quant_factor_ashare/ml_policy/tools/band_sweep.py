# Research project author: Gilmour Jia
"""Relative no-trade band sweep on saved scores, no retraining.

Builds each rule's target grid once, then reruns only the execution ledger for
each band.  The band is a fraction of the *target weight*, so a 20% position
moving 1pp is a 5% relative gap and a 1% position moving 1pp is 100%.

    python tools/band_sweep.py --fold 0 --rules ew_top500 policy_softmax_all \
        --bands 0 0.1 0.25 0.5 0.75 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import score_sweep as ss  # noqa: E402
import run_policy as rp  # noqa: E402
from engine_from_scores import load_rows, spec_for  # noqa: E402
from qlp.data import build_prep  # noqa: E402
from qlp.splits import expanding_folds  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--rules", nargs="*", default=["ew_top500", "policy_softmax_all"])
    ap.add_argument("--bands", type=float, nargs="*",
                    default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--rebalance", type=int, default=21)
    ap.add_argument("--cap", type=float, default=0.03)
    ap.add_argument("--cost-bp", type=float, default=25.0)
    ap.add_argument("--capital", type=float, default=10_000_000.0)
    ap.add_argument("--partial", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    per_seed = [load_rows(ART / ("score_sweep_fold%d_seed%d.npz" % (args.fold, s)))
                for s in args.seeds]
    n_sym = int(max(row["ids"].max() for rows in per_seed for row in rows)) + 1
    t_eval = np.asarray([row["t"] for row in per_seed[0]], dtype=np.int64)

    prep = build_prep()
    dates = np.asarray(prep["dates"])
    a = types.SimpleNamespace(dates=dates, symbols=np.asarray(prep["symbols"]),
                              universe=np.asarray(prep["universe"], dtype=bool))
    fold = expanding_folds(dates, first_year=2013)[args.fold]

    rows_out = []
    for name in args.rules:
        spec, top_n = spec_for(name)
        grid = np.zeros((len(dates), n_sym), dtype=np.float64)
        for rows in per_seed:
            for row in rows:
                w = ss.score_to_weights(spec, row["z"], row["cash"])[:row["ids"].size]
                w, _ = ss.apply_cap(w, args.cap)
                grid[int(row["t"]), row["ids"]] += w / len(per_seed)
        grid = rp.select_top(grid, t_eval, top_n, args.cap)
        carry = rp.hold_forward(grid, t_eval, args.rebalance)
        for br in args.bands:
            led = rp.engine_inputs(prep, a, carry, cap=args.cap,
                                   capital=args.capital, partial=args.partial,
                                   band=0.0, band_rel=br, price_guard=0.0,
                                   strict=False)
            seg = rp.slice_ledger(led, fold.test_start, fold.test_end)
            m = rp.ledger_metrics(seg)
            row = dict(rule=name, band_rel=float(br), top_n=top_n,
                       cagr=float(m["cagr"]), sharpe=float(m["sharpe"]),
                       vol=float(m["vol"]), maxdd=float(m["max_drawdown"]),
                       turnover=float(m["annual_two_way_turnover"]),
                       cost_bp=float(m["total_cost_bp_of_capital"]),
                       holdings=float(m["mean_holdings"]),
                       cash=float(m["mean_cash_weight"]))
            rows_out.append(row)
            print("%-20s band %.2f  cagr %+6.2f%%  sharpe %+5.2f  maxdd %6.2f%%"
                  "  turn %5.0f%%  cost %3.0f bp  names %5.1f  cash %.2f"
                  % (name, br, row["cagr"] * 100, row["sharpe"],
                     row["maxdd"] * 100, row["turnover"] * 100,
                     row["cost_bp"], row["holdings"], row["cash"]), flush=True)
            del led, seg
        del grid, carry

    out = args.out or str(ART / ("band_sweep_fold%d.json" % args.fold))
    Path(out).write_text(json.dumps(dict(fold=args.fold, rows=rows_out), indent=2),
                         encoding="utf-8")
    print("[band] wrote %s" % out)
    return rows_out


if __name__ == "__main__":
    main()
