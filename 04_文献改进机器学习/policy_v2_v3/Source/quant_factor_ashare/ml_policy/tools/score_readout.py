# Research project author: Gilmour Jia
"""Extra read-out controls for the score -> portfolio sweep.

The main sweep (tools/score_sweep.py) covers equal-weight top-N, softmax, power
and positive-part rules.  This script re-reads the *same* saved logits and adds
the two controls the literature makes essential:

  * random top-N : same holding count, same rebalancing dates, random names.
    Without it a top-N number cannot be distinguished from "small portfolio in
    a rising market".
  * tilt around EW : w = (1-lambda) * EW + lambda * softmax(score / T), and a
    linear rank tilt.  Brandt-Santa-Clara-Valkanov's weight function is a tilt
    around a benchmark, not a dense softmax; this tests whether a small tilt
    keeps the IC that the dense softmax washes out.

It also re-computes rank IC and decile returns from the saved scores, so the
mapping table and the score diagnostic are on exactly the same rows.

Usage:
    python tools/score_readout.py --fold 0 --seeds 0 1 2
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
sys.path.insert(0, str(HERE))

import score_sweep as ss  # noqa: E402
from qlp.data import build_prep  # noqa: E402


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


def run_rule(rows, weight_fn, cap, cost_bp, n_symbols):
    """Same accounting as score_sweep.run_rule, with a callable read-out."""
    held = np.zeros(int(n_symbols), dtype=np.float64)
    net, turn, breadth = [], [], []
    for row in rows:
        w = np.asarray(weight_fn(row["z"]), dtype=np.float64)
        stock = w[:row["ids"].size]
        stock, _ = ss.apply_cap(stock, cap)
        prev = held[row["ids"]]
        buys = float(np.clip(stock - prev, 0.0, None).sum())
        drops = float(np.clip(prev - stock, 0.0, None).sum())
        left = held.copy()
        left[row["ids"]] = 0.0
        t = buys + drops + float(left.sum())
        net.append(float((stock * row["rets"]).sum()) - (cost_bp / 1e4) * t)
        turn.append(t)
        breadth.append(float((stock > 1e-4).sum()))
        held = np.zeros(int(n_symbols), dtype=np.float64)
        held[row["ids"]] = stock
    return dict(net=np.asarray(net), turn=np.asarray(turn),
                breadth=float(np.mean(breadth)) if breadth else 0.0)


def tilt_softmax(lam: float, temperature: float):
    def fn(z):
        ew = np.full(z.size, 1.0 / z.size)
        sm = ss.rule_softmax(z, temperature, None)
        return (1.0 - lam) * ew + lam * sm
    return fn


def tilt_rank(lam: float):
    def fn(z):
        n = z.size
        order = np.argsort(np.argsort(z, kind="stable"), kind="stable")
        u = (order + 0.5) / n - 0.5          # uniform on [-0.5, 0.5)
        w = np.clip(1.0 + 2.0 * lam * u, 0.0, None)
        return w / w.sum()
    return fn


def rank_ic(z, r):
    if z.size < 20:
        return np.nan
    rz = np.argsort(np.argsort(z, kind="stable"), kind="stable").astype(float)
    rr = np.argsort(np.argsort(r, kind="stable"), kind="stable").astype(float)
    rz -= rz.mean()
    rr -= rr.mean()
    den = np.sqrt((rz ** 2).sum() * (rr ** 2).sum())
    return float((rz * rr).sum() / den) if den > 1e-15 else np.nan


def decile(z, r):
    n = z.size
    k = max(1, n // 10)
    order = np.argsort(-z, kind="stable")
    return float(r[order[:k]].mean()), float(r[order[-k:]].mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--cap", type=float, default=0.03)
    ap.add_argument("--cost-bp", type=float, default=25.0)
    ap.add_argument("--random-draws", type=int, default=20)
    ap.add_argument("--out", default=str(ART / "score_readout.json"))
    args = ap.parse_args(argv)

    paths = [ART / ("score_sweep_fold%d_seed%d.npz" % (args.fold, s))
             for s in args.seeds]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("missing score files: %s" % ", ".join(missing))
    per_seed = [load_rows(p) for p in paths]
    n_sym = int(max((row["ids"].max() for rows in per_seed for row in rows),
                    default=0)) + 1

    prep = build_prep()
    dates = pd.DatetimeIndex(prep["dates"])
    rows0 = per_seed[0]
    in_test = np.array([bool(pd.Timestamp(dates[row["t"]]).year == 2021)
                        if args.fold == 0 else
                        bool(pd.Timestamp(dates[row["t"]]).year == 2022)
                        for row in rows0])
    print("[readout] fold %d  %d decisions  %d in test  %d symbols"
          % (args.fold, len(rows0), int(in_test.sum()), n_sym), flush=True)

    # Score-level diagnostics on exactly these rows.
    ics = np.array([rank_ic(row["z"], row["rets"]) for row in rows0])
    tops, bots = zip(*[decile(row["z"], row["rets"]) for row in rows0])
    print("[readout] score IC mean %+.4f  IR %+.2f  hit %.2f"
          % (np.nanmean(ics), np.nanmean(ics) / np.nanstd(ics),
             float(np.nanmean(ics > 0))))
    print("[readout] decile top %+.4f  bottom %+.4f  spread %+.4f (per 21d)"
          % (np.mean(tops), np.mean(bots), np.mean(tops) - np.mean(bots)))

    specs = []
    for k in (30, 50, 75, 100, 150, 200, 300, 500, 900):
        specs.append(("ew_top%d" % k, ss.rule_ew, dict(k=k), 1.0))
        specs.append(("ew_top%d_cap%g" % (k, args.cap), ss.rule_ew, dict(k=k),
                      args.cap))
        cap_n = max(args.cap, 1.0 / k)
        specs.append(("ew_top%d_cap1/N" % k, ss.rule_ew, dict(k=k), cap_n))
    for lam in (0.1, 0.25, 0.5, 0.75, 1.0):
        specs.append(("tilt_sm_l%g_T1" % lam, tilt_softmax(lam, 1.0), {}, 1.0))
        specs.append(("tilt_sm_l%g_T1_cap%g" % (lam, args.cap),
                      tilt_softmax(lam, 1.0), {}, args.cap))
    for lam in (0.25, 0.5, 1.0):
        specs.append(("tilt_rank_l%g" % lam, tilt_rank(lam), {}, 1.0))
        specs.append(("tilt_rank_l%g_cap%g" % (lam, args.cap),
                      tilt_rank(lam), {}, args.cap))

    results = {}
    for name, fn, kw, cap in specs:
        weight_fn = (lambda z, fn=fn, kw=kw: fn(z, **kw)) if kw else fn
        nets, turns, breadths = [], [], []
        for rows in per_seed:
            built = run_rule(rows, weight_fn, cap, args.cost_bp, n_sym)
            nets.append(built["net"])
            turns.append(built["turn"][in_test].mean())
            breadths.append(built["breadth"])
        test = [summarise(n[in_test]) for n in nets]
        results[name] = dict(
            cagr=float(np.mean([x["cagr"] for x in test])),
            sharpe=float(np.mean([x["sharpe"] for x in test])),
            vol=float(np.mean([x["vol"] for x in test])),
            hit=float(np.mean([x["hit"] for x in test])),
            turn=float(np.mean(turns)),
            breadth=float(np.mean(breadths)),
            seeds=[round(x["cagr"], 4) for x in test],
        )

    # Random top-N: same dates, same count, random names.
    for k in (30, 100, 200):
        cagrs, sharpes = [], []
        for draw in range(args.random_draws):
            rng = np.random.default_rng(1000 + draw)
            nets = []
            for rows in per_seed:
                rnd = [dict(row, z=rng.normal(size=row["z"].size)) for row in rows]
                nets.append(run_rule(rnd, lambda z, k=k: ss.rule_ew(z, k),
                                     1.0, args.cost_bp, n_sym)["net"])
            test = [summarise(n[in_test]) for n in nets]
            cagrs.append(float(np.mean([x["cagr"] for x in test])))
            sharpes.append(float(np.mean([x["sharpe"] for x in test])))
        results["random_top%d" % k] = dict(
            cagr=float(np.mean(cagrs)), cagr_sd=float(np.std(cagrs, ddof=1)),
            sharpe=float(np.mean(sharpes)), draws=args.random_draws)

    # Print sorted by CAGR.
    print("\n%-28s %8s %8s %8s %8s %8s" %
          ("rule", "cagr", "sharpe", "vol", "turn", "names"))
    for name, v in sorted(results.items(), key=lambda kv: -kv[1]["cagr"]):
        print("%-28s %+8.3f %+8.3f %8.3f %8.1f%% %8.1f"
              % (name, v["cagr"], v["sharpe"], v.get("vol", float("nan")),
                 v.get("turn", float("nan")) * 100, v.get("breadth", float("nan"))))

    out = dict(fold=args.fold, seeds=args.seeds, in_test=int(in_test.sum()),
               score=dict(ic_mean=float(np.nanmean(ics)),
                          ic_ir=float(np.nanmean(ics) / np.nanstd(ics)),
                          decile_top=float(np.mean(tops)),
                          decile_bottom=float(np.mean(bots)),
                          decile_spread=float(np.mean(tops) - np.mean(bots))),
               results=results)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n[readout] wrote %s" % args.out, flush=True)
    return out


def summarise(net):
    net = np.asarray(net, dtype=np.float64)
    n = net.size
    if n == 0:
        return dict(cagr=0.0, sharpe=0.0, vol=0.0, hit=0.0)
    per_year = 252.0 / 21.0
    growth = float(np.prod(1.0 + net))
    cagr = growth ** (per_year / n) - 1.0 if growth > 0 else -1.0
    vol = float(np.std(net, ddof=1) * np.sqrt(per_year)) if n > 1 else 0.0
    return dict(cagr=float(cagr), vol=vol,
                sharpe=float(cagr / vol) if vol > 1e-12 else 0.0,
                hit=float((net > 0).mean()))


if __name__ == "__main__":
    main()
