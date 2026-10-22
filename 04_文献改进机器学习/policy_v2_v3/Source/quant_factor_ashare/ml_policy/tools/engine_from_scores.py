# Research project author: Gilmour Jia
"""Run the hard execution ledger on saved scores, without retraining.

score_sweep.py's fast accounting is the right tool for scanning 60+ read-out
rules, but its engine shortlist averaged three sparse portfolios and then forgot
to re-apply the holding count.  This script re-reads the saved logits, applies
the rule and the *exact* top-N, and runs the same ledger used by run_policy.py.

No model is trained here.  The only inputs are the persisted scores in
artifacts/score_sweep_fold*.npz, the point-in-time price panel, and the rule.

    python tools/engine_from_scores.py --fold 0 --rules ew_top300 ew_top200 ...
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import score_sweep as ss  # noqa: E402
import run_policy as rp  # noqa: E402
from qlp.data import build_prep  # noqa: E402
from qlp.splits import expanding_folds  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    d = np.load(path)
    z, r, t, cash = d["z"], d["rets"], d["t"], d["cash"]
    rows = []
    for i in range(z.shape[0]):
        ids = np.flatnonzero(np.isfinite(z[i]))
        rows.append(dict(t=int(t[i]), ids=ids,
                         z=z[i, ids].astype(np.float64),
                         cash=float(cash[i]),
                         rets=r[i, ids].astype(np.float64)))
    return rows


def spec_for(name: str):
    """Map a rule name to (spec, top_n)."""
    if name == "policy_softmax_all":
        return "policy", 0
    m = re.match(r"^(softmax|ew|pow|zpos)_top(\d+)(?:_T([0-9.]+))?(?:_p([0-9.]+))?$", name)
    if m:
        kind, k, temp, power = m.groups()
        k = int(k)
        if kind == "ew":
            return ("ew", k), k
        if kind == "softmax":
            return ("softmax", float(temp), k), k
        if kind == "pow":
            return ("pow", k, float(power)), k
        if kind == "zpos":
            return ("zpos", k), k
    m = re.match(r"^(softmax|zpos)_all(?:_T([0-9.]+))?$", name)
    if m:
        kind, temp = m.groups()
        if kind == "zpos":
            return ("zpos", None), 0
        return ("softmax", float(temp), None), 0
    if name in ("ew_all", "ew_top1000"):
        return ("ew", None), 0
    raise ValueError("unknown rule %r" % name)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--rules", nargs="*", default=[
        "policy_softmax_all", "ew_all", "ew_top500", "ew_top300", "ew_top200",
        "ew_top100", "ew_top30", "softmax_top100_T0.03", "softmax_all_T1",
        "zpos_all", "zpos_top300"])
    ap.add_argument("--rebalance", type=int, default=21)
    ap.add_argument("--cap", type=float, default=0.03)
    ap.add_argument("--cost-bp", type=float, default=25.0)
    ap.add_argument("--capital", type=float, default=10_000_000.0)
    ap.add_argument("--partial", type=float, default=1.0)
    ap.add_argument("--band", type=float, default=0.0)
    ap.add_argument("--band-rel", type=float, default=0.0)
    ap.add_argument("--save-daily", action="store_true")
    ap.add_argument("--daily-dir", default=None)
    ap.add_argument("--agg", choices=["weight", "rank"], default="weight",
                    help="weight: average seed weights; rank: average seed ranks first")
    ap.add_argument("--out", default=str(ART / "engine_from_scores_fold0.json"))
    args = ap.parse_args(argv)

    paths = [ART / ("score_sweep_fold%d_seed%d.npz" % (args.fold, s))
             for s in args.seeds]
    per_seed = [load_rows(p) for p in paths]
    n_sym = int(max(row["ids"].max() for rows in per_seed for row in rows)) + 1
    t_eval = np.asarray([row["t"] for row in per_seed[0]], dtype=np.int64)

    prep = build_prep()
    dates = np.asarray(prep["dates"])
    # A minimal stand-in for Assembled: engine_inputs only needs these three.
    a = types.SimpleNamespace(dates=dates, symbols=np.asarray(prep["symbols"]),
                              universe=np.asarray(prep["universe"], dtype=bool))
    folds = expanding_folds(dates, first_year=2013)
    fold = folds[args.fold]

    out = []
    for name in args.rules:
        spec, top_n = spec_for(name)
        grid = np.zeros((len(dates), n_sym), dtype=np.float64)
        if args.agg == "rank" and top_n and spec != "policy":
            from scipy.stats import rankdata
            for j in range(len(per_seed[0])):
                ids = per_seed[0][j]["ids"]
                ranks = np.vstack([rankdata(rows[j]["z"]) / max(1, len(ids))
                                   for rows in per_seed])
                avg = ranks.mean(axis=0)
                keep = np.argsort(-avg, kind="stable")[:top_n]
                w = np.zeros(len(ids), dtype=np.float64)
                w[keep] = 1.0 / float(top_n)
                grid[int(per_seed[0][j]["t"]), ids] = w
        else:
            for rows in per_seed:
                for row in rows:
                    w = ss.score_to_weights(spec, row["z"], row["cash"])[:row["ids"].size]
                    w, _ = ss.apply_cap(w, args.cap)
                    grid[int(row["t"]), row["ids"]] += w / len(per_seed)
        grid = rp.select_top(grid, t_eval, top_n, args.cap)
        carry = rp.hold_forward(grid, t_eval, args.rebalance)
        led = rp.engine_inputs(prep, a, carry, cap=args.cap,
                               capital=args.capital, partial=args.partial,
                               band=args.band, band_rel=args.band_rel,
                               price_guard=0.0, strict=False)
        seg = rp.slice_ledger(led, fold.test_start, fold.test_end)
        if args.save_daily:
            ddir = (Path(args.daily_dir) if args.daily_dir
                    else ART / ("engine_daily_fold%d" % args.fold))
            ddir.mkdir(parents=True, exist_ok=True)
            daily = seg["daily"].copy()
            daily.insert(0, "date", daily.index)
            daily.to_csv(ddir / (name + ".csv"), index=False)
        m = rp.ledger_metrics(seg)
        row = dict(rule=name, top_n=top_n, cagr=float(m["cagr"]),
                   sharpe=float(m["sharpe"]), vol=float(m["vol"]),
                   maxdd=float(m["max_drawdown"]),
                   turnover=float(m["annual_two_way_turnover"]),
                   cost_bp=float(m["total_cost_bp_of_capital"]),
                   holdings=float(m["mean_holdings"]),
                   cash=float(m["mean_cash_weight"]))
        out.append(row)
        print("%-24s cagr %+7.3f  sharpe %+6.3f  maxdd %6.3f  turn %5.0f%%"
              "  cost %4.0f bp  names %5.1f  cash %.3f"
              % (name, row["cagr"], row["sharpe"], row["maxdd"],
                 row["turnover"] * 100, row["cost_bp"], row["holdings"],
                 row["cash"]), flush=True)
        del grid, carry, led

    Path(args.out).write_text(json.dumps(dict(fold=args.fold, rules=out), indent=2),
                              encoding="utf-8")
    print("[engine] wrote %s" % args.out, flush=True)
    return out


if __name__ == "__main__":
    main()
