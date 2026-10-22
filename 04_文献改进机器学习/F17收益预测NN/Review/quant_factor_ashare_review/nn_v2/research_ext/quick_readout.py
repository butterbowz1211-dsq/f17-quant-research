# Research project author: Gilmour Jia
"""Fast per-year readout of frozen predictions; no full-ledger gate."""
from pathlib import Path
import sys
import json
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import runtime  # noqa: F401
from metrics import rank_ic


def stats(r):
    r = np.asarray(r, float)
    nav = np.cumprod(1+r)
    peak = np.maximum.accumulate(nav)
    return {"ann_return": float(nav[-1] ** (252/(22*len(r))) - 1), "sharpe": float(r.mean()/r.std(ddof=1)*np.sqrt(252/22)),
            "max_drawdown": float((nav/peak-1).min()), "mean_22d": float(r.mean()), "hit": float((r > 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="derived/research_model_v4")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    p = pd.read_csv(ROOT/args.run/f"predictions_{args.year}.csv")
    daily, rows = [], []
    for d, g in p.groupby("formation_time", sort=True):
        m = g.label_reason.eq("ok") & np.isfinite(g[["NN","Ridge","F17","label"]]).all(axis=1)
        good = g[m]
        if len(good) < 100:
            continue
        rec = {"date": d, "n": len(good), "label": float(good.label.mean())}
        for name in ("NN","Ridge","F17"):
            rec[name+"_ic"] = rank_ic(good[name], good.label)
            rec[name+"_top"] = float(good.nlargest(args.top, name).label.mean())
        daily.append(rec)
    daily = pd.DataFrame(daily)
    out = {"year": args.year, "dates": len(daily), "rows": int(p.shape[0])}
    for name in ("NN","Ridge","F17"):
        ic = daily[name+"_ic"].dropna()
        out[name] = {"mean_ic": float(ic.mean()), "t": float(ic.mean()/(ic.std(ddof=1)/np.sqrt(len(ic)))),
                     "hit": float((ic > 0).mean())}
        out[name].update(stats(daily[name+"_top"]))
    out["equal_weight_all"] = stats(daily.label)
    out["NN_minus_F17_ic"] = out["NN"]["mean_ic"] - out["F17"]["mean_ic"]
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out:
        (ROOT/args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    daily.to_csv(ROOT/args.run/f"quick_readout_{args.year}.csv", index=False)


if __name__ == "__main__":
    main()
