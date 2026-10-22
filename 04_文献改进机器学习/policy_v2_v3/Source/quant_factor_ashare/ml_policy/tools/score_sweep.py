# Research project author: Gilmour Jia
"""Score -> portfolio read-out sweep.

The policy produces two separable things: a cross-sectional score and a map from
that score to portfolio weights.  tools/score_diagnostics.py measures the score
(rank IC ~0.099 in 2021).  tools/shuffle_control.py shows that the *weights*
carry no information once you permute them inside a date.  Those two results can
both be true only if the map, not the score, is where the information dies.

This module isolates the map.  It trains the same seeds with the same
hyper-parameters as tools/score_diagnostics.py -- verified by the selected epoch,
which is the reproducibility check -- then persists the raw logits and re-reads
those *same* logits through a grid of allocation rules.  No network changes
between rows of the table.

Three families of read-out are compared:

  equal weight top-N      the read-out most of the literature uses with a score
  softmax (ours)          dense, temperature-controlled, cash as a peer node
  positive-part / power   weight proportional to the score or its rank

Each rule is evaluated twice: once with the fast vectorised accounting (gross
minus a weight-space cost, same convention as the training surrogate) and, for
the shortlist, once through the hard executor in qlp.execution.  Only the second
is a reportable number; the first exists so a 40-rule grid can be scanned without
40 ledger passes.

    python tools/score_sweep.py --folds 0 1
    python tools/score_sweep.py --folds 0 --engine 8      # shortlist only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))

from qlp import weights as wt  # noqa: E402
from qlp.data import build_prep  # noqa: E402
from qlp.dataset import (DatasetConfig, assemble, make_sample,  # noqa: E402
                          standardize_state)
from qlp.policy import PolicyConfig  # noqa: E402
from qlp.splits import expanding_folds  # noqa: E402

import run_policy as rp  # noqa: E402

TORCH_DTYPE = torch.float32


# ------------------------------------------------------------------- rules ---

def _keep(s: np.ndarray, k: int | None) -> np.ndarray:
    """Boolean mask of the k highest scores (all of them when k is None)."""
    if k is None or k >= s.size:
        return np.ones(s.size, dtype=bool)
    k = max(1, int(k))
    pos = np.argpartition(-s, k - 1)[:k]
    m = np.zeros(s.size, dtype=bool)
    m[pos] = True
    return m


def rule_ew(s: np.ndarray, k: int | None = None) -> np.ndarray:
    m = _keep(s, k)
    w = np.zeros(s.size, dtype=np.float64)
    w[m] = 1.0 / float(m.sum())
    return w


def rule_softmax(s: np.ndarray, temperature: float, k: int | None = None) -> np.ndarray:
    """Softmax restricted to the kept names.

    Subtracting the kept-set maximum is not cosmetic: at temperature 0.05 the raw
    exp() of a logit near 30 overflows float64 and the whole vector becomes NaN,
    which then looks like "no trade" instead of "bug".
    """
    m = _keep(s, k)
    w = np.zeros(s.size, dtype=np.float64)
    if not m.any():
        return w
    z = s[m]
    z = (z - z.max()) / max(1e-12, float(temperature))
    e = np.exp(np.clip(z, -700.0, 0.0))
    w[m] = e / e.sum()
    return w


def rule_zpos(s: np.ndarray, k: int | None = None, floor: float = 0.0) -> np.ndarray:
    """Positive part of the cross-sectional z-score, renormalised.

    Standardising on the *full* eligible set rather than on the kept subset keeps
    the sign convention meaningful: a name is long only if it beats the average
    name that day, not merely the average name in the top-N.
    """
    sd = float(np.std(s))
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros(s.size, dtype=np.float64)
    z = (s - float(np.mean(s))) / sd
    m = _keep(s, k)
    w = np.zeros(s.size, dtype=np.float64)
    w[m] = np.clip(z[m] - floor, 0.0, None)
    tot = w.sum()
    return w / tot if tot > 1e-12 else w


def rule_pow(s: np.ndarray, k: int | None, power: float) -> np.ndarray:
    """Weight proportional to (k - rank)^power: a smooth equal-weight-to-top-1 glide."""
    m = _keep(s, k)
    w = np.zeros(s.size, dtype=np.float64)
    idx = np.flatnonzero(m)
    order = idx[np.argsort(-s[idx])]
    deg = (order.size - np.arange(order.size)).astype(np.float64) ** float(power)
    w[order] = deg
    return w / w.sum()


def apply_cap(w: np.ndarray, cap: float) -> tuple[np.ndarray, bool]:
    """Hard capped-simplex projection on the support of the rule's weights.

    Projecting over all n names instead would let the cap push weight into names
    the rule never selected -- a 30-name book at a 3% cap would come back holding
    900 names at 0.01% -- so the support is frozen exactly as
    run_policy.select_top freezes it.
    """
    nz = np.flatnonzero(w > 0.0)
    if nz.size == 0:
        return w, True
    total = float(w[nz].sum())
    sub, ok = wt.project_capped_simplex(w[nz], np.zeros(nz.size),
                                        np.full(nz.size, float(cap)), total=total)
    out = np.zeros_like(w)
    out[nz] = sub
    return out, bool(ok)


# ---------------------------------------------------------------- builder ---

def build_rules() -> list[tuple[str, object]]:
    """The grid.  Ordered from the densest read-out to the sparsest."""
    rules: list[tuple[str, object]] = []
    rules.append(("policy_softmax_all", "policy"))
    for t in (1.0, 0.5, 0.2, 0.1, 0.05):
        rules.append(("softmax_all_T%g" % t, ("softmax", t, None)))
    for k in (30, 100):
        for t in (0.3, 0.1, 0.03):
            rules.append(("softmax_top%d_T%g" % (k, t), ("softmax", t, k)))
    for k in (10, 20, 30, 50, 100, 200, 300, 500, 1000):
        rules.append(("ew_top%d" % k, ("ew", k)))
    rules.append(("ew_all", ("ew", None)))
    for k in (30, 100):
        for p in (0.5, 1.0, 2.0):
            rules.append(("pow_top%d_p%g" % (k, p), ("pow", k, p)))
    rules.append(("zpos_all", ("zpos", None)))
    for k in (30, 100, 300):
        rules.append(("zpos_top%d" % k, ("zpos", k)))
    return rules


def score_to_weights(spec, z: np.ndarray, cash: float) -> np.ndarray:
    """Map one date's scaled stock logits onto weights over that date's names."""
    if spec == "policy":
        full = np.concatenate([z, np.asarray([cash], dtype=np.float64)])
        return np.exp(full - full.max()) / np.exp(full - full.max()).sum()
    kind = spec[0]
    if kind == "softmax":
        return rule_softmax(z, spec[1], spec[2])
    if kind == "ew":
        return rule_ew(z, spec[1])
    if kind == "pow":
        return rule_pow(z, spec[1], spec[2])
    if kind == "zpos":
        return rule_zpos(z, spec[1])
    raise ValueError("unknown rule %r" % (spec,))


# ------------------------------------------------------------- accounting ---

def run_rule(rows: list[dict], spec, cap: float, cost_bp: float,
             n_symbols: int) -> dict:
    """Walk one rule through every decision date, charging weight-space costs.

    The carried-holdings vector is global (indexed by symbol), so a name that
    leaves the cross-section and returns is charged for the sale, exactly as
    qlp.policy.portfolio_growth does during training.
    """
    held = np.zeros(n_symbols, dtype=np.float64)
    net, gross, turn, breadth, maxw, dep, cashw, names = ([] for _ in range(8))
    cap_ok = True
    for row in rows:
        ids = row["ids"]
        z = row["z"]
        w = score_to_weights(spec, z, row["cash"])
        stock = w[:ids.size]
        cash = float(w[ids.size]) if w.size > ids.size else 0.0
        stock, ok = apply_cap(stock, cap)
        cap_ok = cap_ok and bool(ok)
        prev = held[ids]
        buys = float(np.clip(stock - prev, 0.0, None).sum())
        drops = float(np.clip(prev - stock, 0.0, None).sum())
        left = held.copy()
        left[ids] = 0.0
        dropped_out = float(left.sum())
        t = buys + drops + dropped_out
        g = float((stock * row["rets"]).sum())
        c = (cost_bp / 1e4) * t
        net.append(g - c)
        gross.append(g)
        turn.append(t)
        breadth.append(float((stock > 1e-4).sum()))
        names.append(float((stock > 0).sum()))
        maxw.append(float(stock.max()) if stock.size else 0.0)
        dep.append(float(stock.sum()))
        cashw.append(1.0 - float(stock.sum()))
        held = np.zeros(n_symbols, dtype=np.float64)
        held[ids] = stock
    return dict(net=np.asarray(net), gross=np.asarray(gross), turn=np.asarray(turn),
                breadth=float(np.mean(breadth)) if breadth else 0.0,
                names=float(np.mean(names)) if names else 0.0,
                max_weight=float(np.max(maxw)) if maxw else 0.0,
                deployed=float(np.mean(dep)) if dep else 0.0,
                cash=float(np.mean(cashw)) if cashw else 0.0,
                cap_ok=cap_ok)


def summarise(net: np.ndarray, rb: int) -> dict:
    net = np.asarray(net, dtype=np.float64)
    n = int(net.size)
    if n == 0:
        return dict(n=0)
    per_year = 252.0 / float(rb)
    growth = float(np.prod(1.0 + net))
    cagr = growth ** (per_year / n) - 1.0 if growth > 0 else -1.0
    vol = float(np.std(net, ddof=1) * np.sqrt(per_year)) if n > 1 else 0.0
    return dict(n=n, cagr=float(cagr), vol=vol,
                sharpe=float(cagr / vol) if vol > 1e-12 else 0.0,
                mean_bp=float(net.mean() * 1e4), hit=float((net > 0).mean()))


def paired_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = d.size
    if n < 3:
        return float("nan"), float(d.mean() * 1e4) if n else float("nan")
    sd = float(np.std(d, ddof=1))
    t = float(np.mean(d) / (sd / np.sqrt(n))) if sd > 1e-15 else float("nan")
    return t, float(d.mean() * 1e4)


# ------------------------------------------------------------------ scores ---

@torch.no_grad()
def dump_logits(policy, a, t_list, device):
    """Scaled stock logits plus the cash logit, one aligned row per date."""
    policy.eval()
    rows = []
    for t in t_list:
        t = int(t)
        s = make_sample(a, t, slot=t // max(1, int(a.rb)))
        x = torch.from_numpy(s.feat).to(device)
        full = policy.logits(x).cpu().numpy().astype(np.float64)
        rows.append(dict(t=t, ids=s.ids.copy(), z=full[:-1], cash=float(full[-1]),
                         rets=s.returns.astype(np.float64)))
    return rows


def save_rows(path: Path, rows: list[dict], n_symbols: int) -> None:
    """Dense, NaN-padded so pandas-free loading stays simple."""
    T, N = len(rows), n_symbols
    z = np.full((T, N), np.nan, dtype=np.float32)
    r = np.full((T, N), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        z[i, row["ids"]] = row["z"]
        r[i, row["ids"]] = row["rets"]
    np.savez_compressed(path, z=z, rets=r,
                        t=np.asarray([row["t"] for row in rows], dtype=np.int32),
                        cash=np.asarray([row["cash"] for row in rows], dtype=np.float64))


def load_rows(path: Path, ids_rows: list[dict]) -> list[dict]:
    d = np.load(path)
    z, r, cash = d["z"], d["rets"], d["cash"]
    out = []
    for i, row in enumerate(ids_rows):
        ids = row["ids"]
        out.append(dict(t=row["t"], ids=ids, z=z[i, ids].astype(np.float64),
                        cash=float(cash[i]), rets=r[i, ids].astype(np.float64)))
    return out


# -------------------------------------------------------------------- main ---

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
    ap.add_argument("--temp-end", type=float, default=0.1)
    ap.add_argument("--cap", type=float, default=0.03)
    ap.add_argument("--cap-alt", type=float, default=1.0,
                    help="second cap setting, 1.0 means uncapped")
    ap.add_argument("--cost-bp", type=float, default=25.0)
    ap.add_argument("--band", type=float, default=0.0)
    ap.add_argument("--partial", type=float, default=1.0)
    ap.add_argument("--capital", type=float, default=10_000_000.0)
    ap.add_argument("--price-guard", type=float, default=0.0)
    ap.add_argument("--engine", type=int, default=0,
                    help="run the hard ledger for this many top rules")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse artifacts/score_sweep_fold*.npz instead of training")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default=str(ART / "score_sweep.json"))
    args = ap.parse_args(argv)

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print("[sweep] device=%s torch=%s" % (device, torch.__version__), flush=True)
    prep = build_prep()
    t0 = time.time()
    a = assemble(prep, DatasetConfig(rebalance=args.rebalance))
    print("[sweep] assembled in %.1fs: %d dates x %d symbols"
          % (time.time() - t0, len(a.dates), len(a.symbols)), flush=True)
    n_sym = len(a.symbols)
    folds = expanding_folds(a.dates, first_year=args.first_year)

    cfg = PolicyConfig(n_own=int(a.own.shape[2]) + 1,
                       n_neighbour=int(a.peer.shape[2]),
                       n_market=int(a.mkt.shape[1]), n_state=3,
                       cap=args.cap, temperature=1.0, cost_bp=args.cost_bp)
    if cfg.n_input != a.n_input:
        raise ValueError("policy expects %d inputs, dataset provides %d"
                         % (cfg.n_input, a.n_input))

    rules = build_rules()
    caps = [(args.cap, "cap%g" % args.cap)]
    if args.cap_alt != args.cap:
        caps.append((args.cap_alt, "cap%g" % args.cap_alt))

    report = dict(config=dict(vars(args)), rules=[r[0] for r in rules], folds=[])
    for fidx in args.folds:
        fold = folds[fidx]
        print("[sweep] fold %d: %s" % (fidx, fold.label()), flush=True)
        dates = pd.DatetimeIndex(a.dates)
        tr = np.flatnonzero((dates >= fold.train_start) & (dates <= fold.train_end))
        va = np.flatnonzero((dates >= fold.valid_start) & (dates <= fold.valid_end))
        te = np.flatnonzero((dates >= fold.test_start) & (dates <= fold.test_end))
        a.state_scale = standardize_state(a.mkt, slice(int(tr[0]), int(tr[-1]) + 1))
        cache: dict = {}
        train_s = rp.build_samples(a, a.decision_index(int(tr[0]), int(tr[-1]) + 1), cache)
        valid_s = rp.build_samples(a, a.decision_index(int(va[0]), int(va[-1]) + 1), cache)
        ev_hi = int(min(int(te[-1]), len(dates) - int(args.rebalance) - 1))
        t_eval = a.decision_index(int(tr[0]), ev_hi + 1)

        t_start = time.time()
        per_seed_rows, seed_info = [], []
        ids_rows = [dict(t=int(t), ids=make_sample(
            a, int(t), slot=int(t) // int(a.rb)).ids) for t in t_eval]
        paths = [ART / ("score_sweep_fold%d_seed%d.npz" % (fidx, int(s)))
                 for s in args.seeds]
        if args.reuse and all(p.exists() for p in paths):
            per_seed_rows = [load_rows(p, ids_rows) for p in paths]
            print("[sweep] reused %d score files for fold %d"
                  % (len(paths), fidx), flush=True)
        else:
            for seed in args.seeds:
                res = rp.train_one_seed(train_s, valid_s, cfg, seed=int(seed),
                                        epochs=args.epochs, lr=args.lr,
                                        batch=args.batch, patience=args.patience,
                                        temp_end=args.temp_end, rb=args.rebalance,
                                        n_symbols=n_sym, device=device, verbose=False)
                rows = dump_logits(res["policy"], a, t_eval, device)
                per_seed_rows.append(rows)
                seed_info.append(dict(seed=int(seed), best_epoch=int(res["best_epoch"]),
                                      best_valid_score=float(res["best_valid_score"]),
                                      valid_breadth=float((res["best_valid"] or {})
                                                          .get("mean_breadth", 0.0)),
                                      temperature=float(res["policy"].cfg.temperature)))
                print("   seed %d: epoch %2d  valid score %+.3f  valid breadth %6.1f"
                      % (seed, res["best_epoch"], res["best_valid_score"],
                         (res["best_valid"] or {}).get("mean_breadth", 0.0)), flush=True)
                save_rows(paths[len(per_seed_rows) - 1], rows, n_sym)
                torch.save(res["policy"].state_dict(),
                           ART / ("score_sweep_fold%d_seed%d.pt" % (fidx, int(seed))))
            print("[sweep] scores saved for %d seeds (%.0fs)"
                  % (len(per_seed_rows), time.time() - t_start), flush=True)

        in_test = np.array([bool(fold.test_start <= pd.Timestamp(a.dates[int(row["t"])])
                                 <= fold.test_end) for row in per_seed_rows[0]])
        in_hold = np.array([bool(fold.valid_start <= pd.Timestamp(a.dates[int(row["t"])])
                                 <= fold.test_end) for row in per_seed_rows[0]])
        market = [float(np.nanmean(row["rets"])) for row in per_seed_rows[0]]

        results = {}
        for label, spec in rules:
            for cap, cap_tag in caps:
                per = []
                for rows in per_seed_rows:
                    built = run_rule(rows, spec, cap, args.cost_bp, n_sym)
                    per.append(dict(
                        test=summarise(built["net"][in_test], args.rebalance),
                        hold=summarise(built["net"][in_hold], args.rebalance),
                        breadth=built["breadth"], names=built["names"],
                        max_weight=built["max_weight"], deployed=built["deployed"],
                        cash=built["cash"], turn=float(built["turn"][in_test].mean()),
                        cap_ok=built["cap_ok"]))
                t_list, ex_list = [], []
                for j in range(len(per_seed_rows)):
                    b = per_seed_rows[j]
                    built = run_rule(b, spec, cap, args.cost_bp, n_sym)
                    ref = run_rule(b, ("ew", None), cap, args.cost_bp, n_sym)
                    tt, ex = paired_t(built["net"][in_test], ref["net"][in_test])
                    t_list.append(tt)
                    ex_list.append(ex)
                    del built, ref
                agg = dict(
                    cagr=float(np.mean([p["test"]["cagr"] for p in per])),
                    cagr_sd=float(np.std([p["test"]["cagr"] for p in per], ddof=1))
                    if len(per) > 1 else 0.0,
                    sharpe=float(np.mean([p["test"]["sharpe"] for p in per])),
                    vol=float(np.mean([p["test"]["vol"] for p in per])),
                    mean_bp=float(np.mean([p["test"]["mean_bp"] for p in per])),
                    hit=float(np.mean([p["test"]["hit"] for p in per])),
                    hold_cagr=float(np.mean([p["hold"]["cagr"] for p in per])),
                    breadth=float(np.mean([p["breadth"] for p in per])),
                    names=float(np.mean([p["names"] for p in per])),
                    max_weight=float(np.mean([p["max_weight"] for p in per])),
                    deployed=float(np.mean([p["deployed"] for p in per])),
                    cash=float(np.mean([p["cash"] for p in per])),
                    turn=float(np.mean([p["turn"] for p in per])),
                    cap_ok=bool(all(p["cap_ok"] for p in per)),
                    excess_bp=float(np.nanmean(ex_list)),
                    excess_t=float(np.nanmean(t_list)),
                )
                results["%s|%s" % (label, cap_tag)] = agg
        report["folds"].append(dict(index=fidx, label=fold.label(),
                                    contamination=("CONTAMINATED"
                                                   if fold.contaminated else "untouched"),
                                    n_decisions=int(len(t_eval)),
                                    n_test=int(in_test.sum()),
                                    tests=[str(pd.Timestamp(a.dates[int(t)]).date())
                                           for t in np.asarray(t_eval)[in_test]],
                                    seeds=seed_info, results=results,
                                    market_mean_bp=float(np.mean(market) * 1e4)))
        if args.engine:
            report["folds"][-1]["engine"] = _engine_shortlist(
                results, rules, per_seed_rows, a, prep, fold, t_eval, args, n_sym, caps)

        _print_fold(report["folds"][-1], caps)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str),
                                  encoding="utf-8")

    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("[sweep] wrote %s" % args.out, flush=True)
    return report


def _engine_shortlist(results, rules, per_seed_rows, a, prep, fold, t_eval,
                      args, n_sym, caps) -> list[dict]:
    """Run the hard ledger on the anchor plus the best few rules.

    The seed grids are averaged first and the cap is applied to the average, so
    this is the same read-out protocol run_policy.evaluate_targets uses; a ledger
    per seed would triple the cost and could not be compared with the reported
    numbers.
    """
    cap = args.cap
    tag = "|cap%g" % cap
    ranked = sorted(((k[:-len(tag)], v["cagr"]) for k, v in results.items()
                     if k.endswith(tag)), key=lambda kv: -kv[1])
    wanted = [n for n, _ in ranked[:max(1, int(args.engine))]]
    if "policy_softmax_all" not in wanted:
        wanted.append("policy_softmax_all")
    spec_by_name = {name: spec for name, spec in rules}
    out = []
    for name in wanted:
        spec = spec_by_name[name]
        grid = np.zeros((len(a.dates), n_sym), dtype=np.float64)
        for rows in per_seed_rows:
            for row in rows:
                w = score_to_weights(spec, row["z"], row["cash"])[:row["ids"].size]
                w, _ = apply_cap(w, cap)
                grid[int(row["t"]), row["ids"]] += w / len(per_seed_rows)
        grid = rp.select_top(grid, t_eval, 0, cap)
        carry = rp.hold_forward(grid, t_eval, args.rebalance)
        led = rp.engine_inputs(prep, a, carry, cap=cap, capital=args.capital,
                               partial=args.partial, band=args.band,
                               price_guard=args.price_guard, strict=False)
        m = rp.ledger_metrics(rp.slice_ledger(led, fold.test_start, fold.test_end))
        row = dict(rule=name, cagr=float(m["cagr"]), sharpe=float(m["sharpe"]),
                   vol=float(m["vol"]), maxdd=float(m["max_drawdown"]),
                   turnover=float(m["annual_two_way_turnover"]),
                   cost_bp=float(m["total_cost_bp_of_capital"]),
                   holdings=float(m["mean_holdings"]),
                   cash=float(m["mean_cash_weight"]))
        out.append(row)
        print("    [engine] %-22s cagr %+7.3f  sharpe %+6.3f  maxdd %6.3f"
              "  turn %5.0f%%  cost %4.0f bp  names %5.1f"
              % (name, row["cagr"], row["sharpe"], row["maxdd"],
                 row["turnover"] * 100, row["cost_bp"], row["holdings"]), flush=True)
        del grid, carry, led
    return out


def _print_fold(fold_rep: dict, caps) -> None:
    for _, cap_tag in caps:
        rows = [(k.split("|")[0], v) for k, v in fold_rep["results"].items()
                if k.endswith("|" + cap_tag)]
        rows.sort(key=lambda kv: -kv[1]["cagr"])
        print("\n  [%s] %s  test window %d decisions  (%s)"
              % (cap_tag, fold_rep["contamination"], fold_rep["n_test"],
                 fold_rep["label"]), flush=True)
        print("    %-22s %7s %7s %7s %8s %7s %8s %7s %7s"
              % ("rule", "cagr", "sd", "sharpe", "mean_bp", "names", "maxw",
                 "deploy", "excess_t"), flush=True)
        for name, v in rows:
            print("    %-22s %+7.3f %7.3f %+7.3f %+8.1f %7.1f %8.4f %7.3f %+7.2f"
                  % (name, v["cagr"], v["cagr_sd"], v["sharpe"], v["mean_bp"],
                     v["names"], v["max_weight"], v["deployed"], v["excess_t"]),
                  flush=True)
    print("", flush=True)


if __name__ == "__main__":
    main()
