# Research project author: Gilmour Jia
"""Cross-sectional informativeness of Baidu valuation fields on v4 formation dates.

Descriptive only: no model selection, no training-package modification.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import runtime  # noqa: F401  (bootstraps pyarrow)
import pyarrow  # noqa: E402
pyarrow._hotfix_installed = True  # pandas 3.0.6 tries an unregister that pyarrow 26 lacks

SAMPLES = ROOT / "derived/research_training_v4/samples.npz"
VALUATION = ROOT / "derived/valuation_at_formation_v1/valuation.parquet"
OUT = ROOT / "research_ext/valuation_ic"
OUT.mkdir(parents=True, exist_ok=True)
HORIZON = 22


def zscore(s):
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else s * 0.0


def main():
    with np.load(SAMPLES, allow_pickle=False) as zz:
        dates = pd.to_datetime(zz["formation_time"])
        offsets = zz["offsets"]
        y = zz["y"]
        f17 = zz["f17_score"]
        symbols = zz["symbols"]
    rows = []
    for i, d in enumerate(dates):
        a, b = offsets[i], offsets[i + 1]
        rows.append(pd.DataFrame({"date": d, "symbol": symbols[a:b], "y": y[a:b], "f17": f17[a:b]}))
    sample = pd.concat(rows, ignore_index=True)
    val = pd.read_parquet(VALUATION)
    df = sample.merge(val, on=["date", "symbol"], how="left", validate="many_to_one")

    df["EP"] = np.where(df.pe_ttm > 0, 1.0 / df.pe_ttm, np.nan)
    df["EP_STATIC"] = np.where(df.pe > 0, 1.0 / df.pe, np.nan)
    df["BP"] = np.where(df.pb > 0, 1.0 / df.pb, np.nan)
    df["SP"] = np.where(df.ps_ttm > 0, 1.0 / df.ps_ttm, np.nan)
    df["DIV"] = df.div_yield_ttm
    df["SIZE_TOTAL"] = -np.log(df.total_cap.where(df.total_cap > 0))
    df["SIZE_FLOAT"] = -np.log(df.float_cap.where(df.float_cap > 0))
    df["FREE_RATIO"] = df.free_float / df.float_shares.where(df.float_shares > 0)
    df["SHARE_RATIO"] = df.float_shares / df.total_shares.where(df.total_shares > 0)

    factors = ["EP", "EP_STATIC", "BP", "SP", "DIV", "SIZE_TOTAL", "SIZE_FLOAT", "FREE_RATIO", "SHARE_RATIO"]
    records = []
    deciles = []
    for d, g in df.groupby("date", sort=True):
        base = g[np.isfinite(g.y)]
        if len(base) < 50:
            continue
        # each factor is ranked on its own available set; y fixed to valid labels
        r17 = base.f17.rank(pct=True)
        ry = base.y.rank(pct=True)
        rec = {"date": d, "n": len(base), "f17_ic": r17.corr(ry)}
        z_parts = []
        for f in factors:
            m = np.isfinite(base[f])
            sub = base[m]
            if len(sub) < 50:
                rec[f + "_ic"] = np.nan
                rec[f + "_partial"] = np.nan
                rec[f + "_n"] = int(len(sub))
                continue
            rf = sub[f].rank(pct=True)
            ryy = sub.y.rank(pct=True)
            rec[f + "_ic"] = rf.corr(ryy)
            # partial: residualize factor rank on F17 rank, then correlate with y rank
            r17s = sub.f17.rank(pct=True)
            ok = np.isfinite(r17s)
            if ok.sum() >= 50:
                X = np.column_stack([np.ones(ok.sum()), r17s[ok].to_numpy()])
                beta = np.linalg.lstsq(X, rf[ok].to_numpy(), rcond=None)[0]
                resid = rf[ok] - X @ beta
                rec[f + "_partial"] = np.corrcoef(resid, ryy[ok].to_numpy())[0, 1]
            else:
                rec[f + "_partial"] = np.nan
            rec[f + "_n"] = int(len(sub))
            # decile monotonicity for the four valuation ratios
            if f in ("EP", "BP", "SP", "DIV"):
                q = pd.qcut(rf, 10, labels=False, duplicates="drop")
                tmp = pd.DataFrame({"q": q, "y": ryy}).groupby("q").y.mean()
                deciles.append({"date": d, "factor": f, "top": float(tmp.iloc[-1]), "bottom": float(tmp.iloc[0]),
                                "spread": float(tmp.iloc[-1] - tmp.iloc[0]), "n": len(sub)})
                z_parts.append(zscore(sub[f]))
        # equal-weight valuation composite: rank-average of available fields, no imputation
        comp_cols = ["EP", "BP", "SP", "DIV"]
        ranks = base[comp_cols].rank(pct=True)
        comp = ranks.mean(axis=1, skipna=True)
        rec["VALUE_ic"] = comp.corr(ry)
        # combined with F17, weight lambda on cross-sectional z of the composite
        zf = zscore(base.f17)
        for lam in (0.1, 0.25, 0.5, 1.0):
            combo = zf + lam * zscore(comp)
            rec[f"combo_{lam}_ic"] = combo.corr(ry)
        rec["n_value"] = int(comp.notna().sum())
        records.append(rec)

    daily = pd.DataFrame(records)
    daily.to_csv(OUT / "valuation_ic_by_date.csv", index=False)
    pd.DataFrame(deciles).to_csv(OUT / "valuation_deciles.csv", index=False)

    def block(frame):
        out = {"dates": int(len(frame)), "mean_n": float(frame.n.mean())}
        for f in ["f17"] + factors + ["VALUE"]:
            col = f + "_ic"
            if col not in frame:
                continue
            s = frame[col].dropna()
            out[f] = {"ic": float(s.mean()), "ir": float(s.mean() / s.std(ddof=1) * np.sqrt(len(s))) if len(s) > 1 and s.std(ddof=1) > 0 else None,
                      "t": float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) if len(s) > 1 and s.std(ddof=1) > 0 else None,
                      "hit": float((s > 0).mean()), "dates": int(len(s))}
            p = frame[f + "_partial"].dropna() if f + "_partial" in frame else pd.Series(dtype=float)
            if len(p) > 1:
                out[f]["partial_ic"] = float(p.mean())
                out[f]["partial_t"] = float(p.mean() / (p.std(ddof=1) / np.sqrt(len(p))))
        for lam in (0.1, 0.25, 0.5, 1.0):
            col = f"combo_{lam}_ic"
            if col in frame:
                s = frame[col].dropna()
                out[f"combo_{lam}"] = {"ic": float(s.mean()), "t": float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) if len(s) > 1 else None}
        return out

    dev = daily[daily.date < "2023-01-01"]
    vall = daily[daily.date >= "2023-01-01"]
    summary = {"samples": int(len(df)), "valuation_rows": int(len(val)),
               "dates": int(daily.date.nunique()), "first": str(daily.date.min().date()), "last": str(daily.date.max().date()),
               "all": block(daily), "dev_2013_2022": block(dev), "validation_2023_2025": block(vall)}
    (OUT / "valuation_ic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
