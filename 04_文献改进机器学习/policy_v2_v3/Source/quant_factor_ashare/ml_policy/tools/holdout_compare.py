# Research project author: Gilmour Jia
"""Compare the holdout read-outs against two eligible books.

Reads the daily ledgers written by run_policy.py and tools/holdout_readout.py,
computes headline metrics plus CAPM-style beta/alpha against both the
equal-weight (ew_all) and market-cap-weight (vw_all) eligible books, and writes
one JSON and one markdown table.  No model or portfolio choice lives here;
this is a reporting pass only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _perf(net: np.ndarray) -> dict:
    net = np.asarray(net, dtype=float)
    net = net[np.isfinite(net)]
    if net.size == 0:
        return dict(n_days=0)
    nav = np.cumprod(1.0 + net)
    total = float(nav[-1] - 1.0)
    years = net.size / 252.0
    cagr = float((nav[-1]) ** (1.0 / years) - 1.0) if nav[-1] > 0 else -1.0
    vol = float(net.std(ddof=1) * np.sqrt(252.0)) if net.size > 1 else 0.0
    sharpe = float(cagr / vol) if vol > 1e-12 else 0.0
    peak = np.maximum.accumulate(nav)
    dd = float(np.min(nav / peak - 1.0))
    return dict(n_days=int(net.size), total_return=total, cagr=cagr,
                vol=vol, sharpe=sharpe, max_drawdown=dd,
                calmar=float(cagr / abs(dd)) if dd < 0 else 0.0)


def _alpha_beta(net: np.ndarray, bench: np.ndarray) -> dict:
    ok = np.isfinite(net) & np.isfinite(bench)
    y, x = net[ok], bench[ok]
    if y.size < 3 or np.std(x) < 1e-12:
        return dict(alpha_ann=0.0, alpha_t=0.0, beta=0.0, r2=0.0, n=0)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(1, y.size - 2)
    s2 = float(resid @ resid / dof)
    xtx_inv = np.linalg.inv(X.T @ X)
    se_alpha = float(np.sqrt(max(1e-30, s2 * xtx_inv[0, 0])))
    alpha = float(coef[0])
    beta = float(coef[1])
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - (resid @ resid) / ss_tot) if ss_tot > 1e-30 else 0.0
    return dict(alpha_ann=alpha * 252.0,
                alpha_t=alpha / se_alpha if se_alpha > 0 else 0.0,
                beta=beta, r2=r2, n=int(y.size))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="policy_holdout_2013_2020")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    art = ROOT / "artifacts"

    readout_path = art / (args.tag + "_readout.json")
    readout = json.loads(readout_path.read_text(encoding="utf-8"))
    test_start = pd.Timestamp(readout["test_start"])
    test_end = pd.Timestamp(readout["test_end"])

    series = {}
    for name in ("policy_softmax_all", "ew_top500", "ew_top10pct",
                 "vw_top10pct", "ew_all", "vw_all"):
        path = art / ("%s_%s_daily.csv" % (args.tag, name))
        d = pd.read_csv(path, index_col=0, parse_dates=True)
        d = d.loc[(d.index >= test_start) & (d.index <= test_end)]
        series[name] = d["net"].astype(float)

    # The model's own two operating points come from the training report's
    # ledgers, so the read-out comparison includes what run_policy deployed.
    for suffix, name in (("kall", "model_dense"), ("k500", "model_top500")):
        path = art / ("%s_fold0_%s_daily.csv" % (args.tag, suffix))
        if path.exists():
            d = pd.read_csv(path, index_col=0, parse_dates=True)
            d = d.loc[(d.index >= test_start) & (d.index <= test_end)]
            series[name] = d["net"].astype(float)

    bench_ew = series["ew_all"]
    bench_vw = series["vw_all"]
    rows = {}
    for name, net in series.items():
        row = _perf(net.to_numpy())
        if name != "ew_all":
            row.update(_alpha_beta(net.to_numpy(), bench_ew.to_numpy()))
        else:
            row.update(dict(alpha_ann=0.0, alpha_t=0.0, beta=1.0, r2=1.0,
                            n=int(net.size)))
        if name != "vw_all":
            ab = _alpha_beta(net.to_numpy(), bench_vw.to_numpy())
        else:
            ab = dict(alpha_ann=0.0, alpha_t=0.0, beta=1.0, r2=1.0,
                      n=int(net.size))
        row.update(dict(alpha_vw_ann=ab["alpha_ann"], alpha_vw_t=ab["alpha_t"],
                        beta_vw=ab["beta"], r2_vw=ab["r2"]))
        rows[name] = row

    lead = ["model_top500", "model_dense", "policy_softmax_all", "ew_top500",
            "ew_top10pct", "vw_top10pct", "ew_all", "vw_all"]
    order = [k for k in lead if k in rows] + [k for k in rows if k not in lead]

    def fmt(v, pct=False):
        if pct:
            return "%.2f%%" % (100.0 * v)
        return "%.3f" % v

    lines = ["| portfolio | CAGR | Sharpe | maxDD | Calmar | alpha/ew | t | beta/ew | alpha/vw | t | beta/vw |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in order:
        r = rows[name]
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                     % (name, fmt(r["cagr"], True), fmt(r["sharpe"]),
                        fmt(r["max_drawdown"], True), fmt(r["calmar"]),
                        fmt(r["alpha_ann"], True), fmt(r["alpha_t"]),
                        fmt(r["beta"]), fmt(r["alpha_vw_ann"], True),
                        fmt(r["alpha_vw_t"]), fmt(r["beta_vw"])))
    table = "\n".join(lines)
    report = dict(tag=args.tag, test_start=readout["test_start"],
                  test_end=readout["test_end"], contaminated=readout["contaminated"],
                  rows=rows, table=table)
    out = Path(args.out) if args.out else art / (args.tag + "_compare.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table)
    print("\n[compare] wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
