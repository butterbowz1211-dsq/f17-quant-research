# Research project author: Gilmour Jia
"""Post-hoc value overlay on frozen v4 predictions.

No retraining. The NN score is frozen; the valuation composite is a
cross-sectional rank average of EP/BP/SP/DIV. Lambda is selected only on
2021-2022 and evaluated on 2023-2025. Top-50 readout is a research
diagnostic, not an execution-accepted backtest.
"""
from pathlib import Path
import sys
import json
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import runtime  # noqa: F401,E402
import pyarrow  # noqa: E402
pyarrow._hotfix_installed = True  # pandas 3.0.6 + pyarrow 26

from metrics import rank_ic  # noqa: E402

OUT = ROOT / "research_ext/overlay_v4"
OUT.mkdir(parents=True, exist_ok=True)
VALUE_COLS = ["EP", "BP", "SP", "DIV"]
LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
COSTS_BPS = [15.0, 25.0]


def load_frame():
    frames = []
    for year in range(2021, 2026):
        p = pd.read_csv(ROOT / f"derived/research_model_v4/predictions_{year}.csv")
        frames.append(p)
    p = pd.concat(frames, ignore_index=True)
    p["date"] = pd.to_datetime(p["formation_time"])
    p = p[p.label_reason.eq("ok")].copy()
    p = p[np.isfinite(p[["NN", "F17", "label"]].to_numpy()).all(axis=1)]
    val = pd.read_parquet(ROOT / "derived/valuation_at_formation_v1/valuation.parquet")
    val["date"] = pd.to_datetime(val["date"])
    val = val.drop_duplicates(["date", "symbol"])
    df = p.merge(val, on=["date", "symbol"], how="left", validate="many_to_one")
    df["EP"] = np.where(df.pe_ttm > 0, 1.0 / df.pe_ttm, np.nan)
    df["BP"] = np.where(df.pb > 0, 1.0 / df.pb, np.nan)
    df["SP"] = np.where(df.ps_ttm > 0, 1.0 / df.ps_ttm, np.nan)
    df["DIV"] = df.div_yield_ttm
    return df


def add_value(df):
    out = []
    for d, g in df.groupby("date", sort=True):
        g = g.copy()
        ranks = g[VALUE_COLS].rank(pct=True)
        g["VALUE"] = ranks.mean(axis=1, skipna=True)
        g["value_fields"] = ranks.notna().sum(axis=1)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def top_set(g, score, n=50):
    return set(g.nlargest(n, score).symbol.tolist())


def turnover(prev, cur):
    if not prev or not cur:
        return np.nan
    return 1.0 - len(prev & cur) / float(len(cur))


def stats(r, label, periods=None, turnovers=None, cost_bps=0.0):
    r = np.asarray(r, float)
    if turnovers is not None and cost_bps:
        r = r - np.asarray(turnovers, float) * cost_bps / 10000.0
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(nav)
    n = len(r)
    return {
        "label": label,
        "periods": int(n),
        "mean_22d": float(r.mean()),
        "ann_return": float(nav[-1] ** (252.0 / (22.0 * n)) - 1.0),
        "ann_vol": float(r.std(ddof=1) * np.sqrt(252.0 / 22.0)) if n > 1 else None,
        "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(252.0 / 22.0)) if n > 1 and r.std(ddof=1) > 0 else None,
        "max_drawdown": float((nav / peak - 1.0).min()),
        "hit": float((r > 0).mean()),
        "nav": float(nav[-1]),
        "mean_turnover": float(np.mean(turnovers)) if turnovers is not None and len(turnovers) else None,
        "cost_bps_one_way": float(cost_bps),
    }


def block_stats(daily, lam, start, end, top_sets_nn, top_sets_combo):
    s = daily[(daily.date >= start) & (daily.date < end)].reset_index(drop=True)
    if s.empty:
        return {}
    rec = {"dates": int(len(s)), "mean_n": float(s.n.mean())}
    for name, col, sets in [
        ("NN_matched", "nn_top", top_sets_nn),
        ("F17_matched", "f17_top", None),
        ("VALUE", "value_top", None),
        (f"combo_lam_{lam}", f"combo_{lam}_top", top_sets_combo),
    ]:
        # Drop the first period for every series so all stats cover the same dates.
        r = s[col].to_numpy()[1:]
        tos = None
        if sets is not None:
            tos = [turnover(sets[s.date.iloc[i - 1]], sets[s.date.iloc[i]]) for i in range(1, len(s))]
        rec[name] = stats(r, name, turnovers=tos, cost_bps=0.0)
        if sets is not None:
            rec[name + "_turnover"] = float(np.mean(tos)) if tos else None
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = add_value(load_frame())
    coverage = {
        "prediction_rows": int(len(df)),
        "with_value": int(df.VALUE.notna().sum()),
        "value_coverage": float(df.VALUE.notna().mean()),
        "dates": int(df.date.nunique()),
        "first": str(df.date.min().date()),
        "last": str(df.date.max().date()),
        "mean_rows_per_date": float(df.groupby("date").size().mean()),
    }

    daily, top_sets = [], {f"combo_{l}": {} for l in LAMBDAS}
    sets_nn, sets_f17, sets_value = {}, {}, {}
    for d, g in df.groupby("date", sort=True):
        g = g[np.isfinite(g.NN) & np.isfinite(g.F17) & np.isfinite(g.label) & np.isfinite(g.VALUE)].copy()
        if len(g) < 100:
            continue
        g["nn_rank"] = g.NN.rank(pct=True)
        g["value_rank"] = g.VALUE.rank(pct=True)
        rec = {
            "date": d, "n": int(len(g)),
            "all": float(g.label.mean()),
            "nn_ic": rank_ic(g.NN, g.label),
            "f17_ic": rank_ic(g.F17, g.label),
            "value_ic": rank_ic(g.VALUE, g.label),
            "nn_top": float(g.nlargest(50, "NN").label.mean()),
            "f17_top": float(g.nlargest(50, "F17").label.mean()),
            "value_top": float(g.nlargest(50, "VALUE").label.mean()),
            "value_fields_mean": float(g.value_fields.mean()),
            "median_total_cap": float(g.total_cap.median()),
        }
        sets_nn[d] = top_set(g, "NN")
        sets_f17[d] = top_set(g, "F17")
        sets_value[d] = top_set(g, "VALUE")
        for lam in LAMBDAS:
            g["score"] = g.nn_rank + lam * g.value_rank
            rec[f"combo_{lam}_ic"] = rank_ic(g.score, g.label)
            rec[f"combo_{lam}_top"] = float(g.nlargest(50, "score").label.mean())
            top_sets[f"combo_{lam}"][d] = top_set(g, "score")
        daily.append(rec)
    daily = pd.DataFrame(daily).sort_values("date").reset_index(drop=True)

    dev = daily[daily.date < "2023-01-01"]
    test = daily[daily.date >= "2023-01-01"]
    grid = []
    for lam in LAMBDAS:
        grid.append({
            "lambda": lam,
            "dev_ic": float(dev[f"combo_{lam}_ic"].mean()),
            "dev_top": float(dev[f"combo_{lam}_top"].mean()),
            "test_ic": float(test[f"combo_{lam}_ic"].mean()),
            "test_top": float(test[f"combo_{lam}_top"].mean()),
        })
    grid = pd.DataFrame(grid)
    # Select by validation rank IC; tie -> smaller lambda.
    best = grid.sort_values(["dev_ic", "lambda"], ascending=[False, True]).iloc[0]
    lam = float(best["lambda"])

    blocks = {
        "dev_2021_2022": block_stats(daily, lam, "2021-01-01", "2023-01-01", sets_nn, top_sets[f"combo_{lam}"]),
        "test_2023_2025": block_stats(daily, lam, "2023-01-01", "2026-01-01", sets_nn, top_sets[f"combo_{lam}"]),
        "all_2021_2025": block_stats(daily, lam, "2021-01-01", "2026-01-01", sets_nn, top_sets[f"combo_{lam}"]),
    }

    # Net-cost sensitivity on the test block: same cost model for every variant.
    net = []
    for name, col, sets in [
        ("NN_matched", "nn_top", sets_nn),
        ("F17_matched", "f17_top", sets_f17),
        (f"combo_lam_{lam}", f"combo_{lam}_top", top_sets[f"combo_{lam}"]),
    ]:
        for bps in COSTS_BPS:
            tos = [turnover(sets[test.date.iloc[i - 1]], sets[test.date.iloc[i]]) for i in range(1, len(test))]
            net.append(stats(test[col].to_numpy()[1:], f"{name}_net_{bps}bp", turnovers=tos, cost_bps=bps))

    # Paired descriptive test on the selected test series.
    d = test[f"combo_{lam}_top"].to_numpy() - test["nn_top"].to_numpy()
    paired = {
        "periods": int(len(d)),
        "mean_combo_minus_nn_22d": float(np.mean(d)),
        "t_combo_minus_nn": float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d)))) if len(d) > 1 and np.std(d, ddof=1) > 0 else None,
        "combo_wins": int((d > 0).sum()),
        "nn_wins": int((d < 0).sum()),
    }

    # Composition of top-50 for selected combo versus NN, test block.
    comp = []
    for d, g in df.groupby("date", sort=True):
        if d < pd.Timestamp("2023-01-01"):
            continue
        g = g[np.isfinite(g.NN) & np.isfinite(g.VALUE) & np.isfinite(g.label)].copy()
        if len(g) < 100:
            continue
        g["nn_rank"] = g.NN.rank(pct=True)
        g["value_rank"] = g.VALUE.rank(pct=True)
        g["score"] = g.nn_rank + lam * g.value_rank
        for name, col in [("NN", "NN"), ("combo", "score")]:
            top = g.nlargest(50, col)
            comp.append({
                "date": d, "variant": name, "n": len(top),
                "median_total_cap": float(top.total_cap.median()),
                "median_EP": float(top.EP.median()),
                "median_BP": float(top.BP.median()),
                "median_DIV": float(top.DIV.median()),
                "mean_value_rank": float(top.value_rank.mean()),
            })
    comp = pd.DataFrame(comp)

    # Yearly table for selected lambda.
    yearly = []
    for year in range(2021, 2026):
        s = daily[daily.date.dt.year == year]
        if s.empty:
            continue
        rec = {"year": year, "dates": int(len(s)), "n": float(s.n.mean())}
        for name, col in [("NN", "nn_top"), ("F17", "f17_top"), ("VALUE", "value_top"), (f"combo_{lam}", f"combo_{lam}_top")]:
            rec[name] = stats(s[col], name)
        for name, col in [("NN_ic", "nn_ic"), ("F17_ic", "f17_ic"), ("VALUE_ic", "value_ic"), (f"combo_{lam}_ic", f"combo_{lam}_ic")]:
            rec[name] = float(s[col].mean())
        yearly.append(rec)
    yearly = pd.DataFrame(yearly)

    daily.to_csv(out / "daily_overlay.csv", index=False)
    grid.to_csv(out / "lambda_grid.csv", index=False)
    yearly.to_csv(out / "yearly_overlay.csv", index=False)
    comp.to_csv(out / "top50_composition.csv", index=False)
    pd.DataFrame([blocks[k]["NN_matched"] for k in blocks]).to_csv(out / "block_nn.csv", index=False)
    summary = {
        "coverage": coverage,
        "lambda_selected": lam,
        "lambda_selection": "max mean daily rank IC on 2021-2022; tie -> smaller lambda",
        "grid": grid.to_dict(orient="records"),
        "blocks": blocks,
        "net_sensitivity_test": net,
        "paired_test_combo_minus_nn": paired,
        "yearly": yearly.to_dict(orient="records"),
        "composition_test": comp.groupby("variant").median(numeric_only=True).reset_index().to_dict(orient="records"),
        "note": "Frozen v4 NN; valuation composite is rank(EP,BP,SP,DIV). Top-50 equal weight, 22-day label, gross unless cost_bps stated. Research diagnostic only; not execution accepted.",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["coverage", "lambda_selected", "blocks", "net_sensitivity_test", "paired_test_combo_minus_nn"]}, ensure_ascii=False, indent=2, default=float))
    print("\nGRID\n", grid.to_string(index=False))
    print("\nYEARLY\n", yearly.to_string(index=False))
    print("\nCOMP\n", summary["composition_test"])


if __name__ == "__main__":
    main()
