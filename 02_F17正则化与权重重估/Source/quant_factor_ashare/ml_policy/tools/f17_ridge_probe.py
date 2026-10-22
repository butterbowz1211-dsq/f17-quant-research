# Research project author: Gilmour Jia
"""Does linear regularization improve the F17 factor score? Paired test.

F17 itself is an equal-weight average of 17 standardized, industry/size-neutral
factor panels; it has no fitted weights and therefore no regularization.  This
probe fits signed linear composites on exactly the same 17 panels and compares
them with equal weights on the same complete-case cross-sections.

Design:
  * decision dates: every 21st trading day from the panel origin
  * label: buy next open, exit open t+22 (forward_open_return, h=21)
  * annual refit; training labels must exit strictly before Jan 1
  * 5-year rolling window, at least 24 sampled dates
  * ridge toward zero and ridge toward the equal-weight prior
  * rank IC and paired HAC t versus equal weight, on a common sample
"""
from pathlib import Path
import json
import math
import os
import sys
import numpy as np

ROOT = Path(r"workspace/source_workspace\quant_factor_ashare")
DEPS = Path(r"workspace/review_workspace\quant_factor_ashare_review\.deps")
if str(DEPS) not in sys.path:
    sys.path.append(str(DEPS))
for d in (DEPS / "pyarrow", ROOT / "ml_policy" / "tools" / "pyarrow_dlls"):
    if d.is_dir():
        os.add_dll_directory(str(d))
import pyarrow  # noqa: F401  (pandas 3 pickles use Arrow-backed columns)
import pandas as pd

PREP = ROOT / "data" / "prep"
OUT = ROOT / "ml_policy" / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)

NAMES = ["TURN", "MAX5", "INTRADAY", "ILLIQ", "COMMOVE", "TURN_SHOCK",
         "REV_1W", "ISKEW", "IVOL", "OVERNIGHT", "LEAD_LAG", "CO_SKEW",
         "BETA", "REV_LT", "SIZE", "HIGH52", "RES_MOM"]
LAMS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
HORIZON = 21
STEP = 21


def ranks(x):
    return pd.Series(x).rank(method="average").to_numpy()


def rank_ic(score, label):
    ok = np.isfinite(score) & np.isfinite(label)
    if ok.sum() < 50:
        return np.nan
    a = ranks(score[ok])
    b = ranks(label[ok])
    a = (a - a.mean()) / (a.std(ddof=0) + 1e-12)
    b = (b - b.mean()) / (b.std(ddof=0) + 1e-12)
    return float((a * b).mean())


def hac(x, lags=3):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return {"n": n, "mean": None, "t": None}
    e = x - x.mean()
    lrv = float(e @ e / n)
    for k in range(1, min(lags, n - 1) + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1)) * float(e[k:] @ e[:-k] / n)
    se = math.sqrt(max(lrv, 0.0) / n)
    return {"n": n, "mean": float(x.mean()),
            "t": float(x.mean() / se) if se > 0 else None}


def load_panels():
    first = pd.read_pickle(PREP / f"{NAMES[0]}.pkl")
    dates = first.index
    syms = first.columns
    pos = np.arange(0, len(dates) - (HORIZON + 1), STEP)
    x = np.empty((len(pos), len(syms), len(NAMES)), dtype=np.float32)
    for k, name in enumerate(NAMES):
        f = pd.read_pickle(PREP / f"{name}.pkl").reindex(index=dates, columns=syms)
        x[:, :, k] = f.to_numpy(dtype=np.float32)[pos]
        print(f"loaded {name:11s} {f.shape}", flush=True)
    return dates, syms, pos, x


def load_label(dates, syms, pos):
    panel = pd.read_pickle(ROOT / "data" / "panel.pkl")
    op = panel.pivot(index="date", columns="symbol", values="open")
    del panel
    op = op.reindex(index=dates, columns=syms)
    fwd = op.shift(-(HORIZON + 1)) / op.shift(-1) - 1.0
    return fwd.to_numpy(dtype=np.float64)[pos]


def main():
    dates, syms, pos, X = load_panels()
    Y = load_label(dates, syms, pos)
    complete = np.isfinite(X).all(axis=2)
    valid = complete & np.isfinite(Y)
    print(f"sample dates {len(pos)}; complete-case mean {complete.mean():.3f}; "
          f"with label {valid.sum(axis=1).mean():.1f} stocks/date", flush=True)

    moments = []
    for j, p in enumerate(pos):
        m = valid[j]
        if m.sum() < 100:
            continue
        xx = X[j, m, :].astype(np.float64)
        xx = (xx - xx.mean(0)) / (xx.std(0) + 1e-12)
        yy = Y[j, m]
        yy = ranks(yy)
        yy = (yy - yy.mean()) / (yy.std(ddof=0) + 1e-12)
        n = len(yy)
        moments.append((dates[p], dates[p + HORIZON + 1],
                        xx.T @ xx / n, xx.T @ yy / n, n))
    print(f"moments {len(moments)}", flush=True)

    w_eq = np.ones(len(NAMES)) / len(NAMES)
    models = {"equal": w_eq}
    for lam in LAMS:
        models[f"ridge0_{lam:g}"] = ("zero", lam)
        models[f"ridgeeq_{lam:g}"] = ("equal", lam)

    def fit(hist, spec):
        G = np.mean([m[2] for m in hist], axis=0)
        c = np.mean([m[3] for m in hist], axis=0)
        if isinstance(spec, np.ndarray):
            return spec
        prior, lam = spec
        b = c + (lam * w_eq if prior == "equal" else 0.0)
        return np.linalg.solve(G + lam * np.eye(len(c)), b)

    records = []
    for year in range(2015, dates.max().year + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        hist = [m for m in moments
                if cutoff - pd.DateOffset(years=5) <= m[0] < cutoff and m[1] < cutoff]
        if len(hist) < 24:
            print(f"skip {year}: {len(hist)} training dates", flush=True)
            continue
        js = [j for j, p in enumerate(pos) if dates[p].year == year]
        for name, spec in models.items():
            w = fit(hist, spec)
            ics = [rank_ic(X[j, valid[j], :] @ w, Y[j, valid[j]]) for j in js]
            for j, ic in zip(js, ics):
                if np.isfinite(ic):
                    records.append({"year": year, "date": str(dates[pos[j]].date()),
                                    "model": name, "ic": ic})
        print(f"year {year}: {len(hist)} train dates, {len(js)} test dates", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(OUT / "f17_ridge_probe_ic.csv", index=False)
    rows = []
    for split, lo, hi in [("development_2015_2022", "2015-01-01", "2022-12-31"),
                          ("validation_2023_2026", "2023-01-01", "2026-12-31")]:
        sub = df[(df.date >= lo) & (df.date <= hi)]
        base = sub[sub.model == "equal"].set_index("date").ic
        for name in models:
            s = sub[sub.model == name].set_index("date").ic
            s = s.reindex(base.index)
            st = hac(s.to_numpy())
            d = hac((s - base).to_numpy())
            rows.append({"split": split, "model": name, "n": st["n"],
                         "ic": st["mean"], "t": st["t"],
                         "delta_ic": d["mean"], "delta_t": d["t"]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "f17_ridge_probe_summary.csv", index=False)
    print("\n" + out.round(4).to_string(index=False), flush=True)
    (OUT / "f17_ridge_probe_summary.json").write_text(
        json.dumps({"names": NAMES, "lams": LAMS, "rows": rows,
                    "n_sample_dates": int(len(pos)),
                    "mean_complete_stocks": float(valid.sum(axis=1).mean())},
                   ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
