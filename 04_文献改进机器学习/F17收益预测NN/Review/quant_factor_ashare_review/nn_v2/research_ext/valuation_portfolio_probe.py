# Research project author: Gilmour Jia
"""Quick top-50 portfolio probe: F17 versus F17 plus Baidu value composite."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import runtime  # noqa: F401
import pyarrow  # noqa: E402
pyarrow._hotfix_installed = True

OUT = ROOT / "research_ext/valuation_ic"


def z(s):
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else s * 0.0


def stats(r, label, periods):
    r = np.asarray(r, float)
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(nav)
    mdd = float((nav / peak - 1.0).min())
    ann = float(nav[-1] ** (252.0 / (22.0 * periods)) - 1.0)
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(252.0 / 22.0)) if r.std(ddof=1) > 0 else None
    return {"label": label, "periods": periods, "mean_22d": float(r.mean()), "ann_return": ann,
            "ann_vol": float(r.std(ddof=1) * np.sqrt(252.0 / 22.0)), "sharpe": sharpe,
            "max_drawdown": mdd, "hit": float((r > 0).mean()), "nav": float(nav[-1])}


def main():
    with np.load(ROOT / "derived/research_training_v4/samples.npz", allow_pickle=False) as zz:
        dates = pd.to_datetime(zz["formation_time"])
        off = zz["offsets"]
        y = zz["y"]
        f17 = zz["f17_score"]
        syms = zz["symbols"]
    rows = []
    for i, d in enumerate(dates):
        a, b = off[i], off[i + 1]
        rows.append(pd.DataFrame({"date": d, "symbol": syms[a:b], "y": y[a:b], "f17": f17[a:b]}))
    df = pd.concat(rows, ignore_index=True)
    val = pd.read_parquet(ROOT / "derived/valuation_at_formation_v1/valuation.parquet")
    df = df.merge(val, on=["date", "symbol"], how="left", validate="many_to_one")
    for c in ("pe_ttm", "pb", "ps_ttm"):
        df["inv_" + c] = 1.0 / df[c].where(df[c] > 0)
    df["EP"] = df.inv_pe_ttm
    df["BP"] = df.inv_pb
    df["SP"] = df.inv_ps_ttm
    df["DIV"] = df.div_yield_ttm

    out = []
    for d, g in df.groupby("date", sort=True):
        g = g[np.isfinite(g.y) & np.isfinite(g.f17)]
        if len(g) < 100:
            continue
        ranks = g[["EP", "BP", "SP", "DIV"]].rank(pct=True)
        g = g.copy()
        g["value"] = ranks.mean(axis=1, skipna=True)
        rec = {"date": d, "n": len(g), "all": float(g.y.mean()), "f17": float(g.nlargest(50, "f17").y.mean())}
        rec["value"] = float(g.nlargest(50, "value").y.mean()) if g.value.notna().sum() >= 50 else np.nan
        zf = z(g.f17)
        zv = z(g.value)
        for lam in (0.1, 0.25, 0.5):
            g["combo"] = zf + lam * zv
            rec[f"combo_{lam}"] = float(g.nlargest(50, "combo").y.mean())
        out.append(rec)
    frame = pd.DataFrame(out)
    frame.to_csv(OUT / "valuation_top50_by_date.csv", index=False)
    labels = ["all", "f17", "value", "combo_0.1", "combo_0.25", "combo_0.5"]
    res = [stats(frame[l].dropna(), l, len(frame[l].dropna())) for l in labels]
    pd.DataFrame(res).to_csv(OUT / "valuation_top50_stats.csv", index=False)
    pd.DataFrame(res).to_json(OUT / "valuation_top50_stats.json", orient="records", force_ascii=False, indent=2)
    print(pd.DataFrame(res).to_string(index=False))


if __name__ == "__main__":
    main()
