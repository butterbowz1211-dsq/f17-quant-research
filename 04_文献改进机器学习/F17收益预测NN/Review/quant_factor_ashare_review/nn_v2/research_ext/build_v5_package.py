# Research project author: Gilmour Jia
"""Build an experimental 21-factor package from frozen v4 labels plus Baidu valuation.

Does not modify the frozen v4 package or any top-level nn_v2 source file.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import runtime  # noqa: F401  (imports pyarrow before pandas)
import pyarrow  # noqa: E402
pyarrow._hotfix_installed = True
import json  # noqa: E402
import shutil  # noqa: E402
import argparse  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

V4_FACTORS = ["TURN", "MAX5", "INTRADAY", "ILLIQ", "COMMOVE", "TURN_SHOCK", "REV_1W",
              "ISKEW", "IVOL", "OVERNIGHT", "LEAD_LAG", "CO_SKEW", "BETA", "REV_LT",
              "SIZE", "HIGH52", "RES_MOM"]
NEW_FACTORS = ["EP", "BP", "SP", "DIV"]
FACTORS = V4_FACTORS + NEW_FACTORS
N = len(FACTORS)


def preprocess_n(raw, eligible, log_cap, industry):
    """Same winsorize/neutralize/standardize contract as data.preprocess, generalized."""
    raw = np.asarray(raw, float)
    eligible = np.asarray(eligible, bool)
    cap = np.asarray(log_cap, float)
    ind = np.asarray(industry).astype(str)
    if raw.ndim != 2 or raw.shape[1] != N:
        raise ValueError("Wrong factor count")
    if eligible.sum() < 30:
        raise ValueError("Too few eligible stocks")
    clean = np.full_like(raw, np.nan)
    for j in range(N):
        m = eligible & np.isfinite(raw[:, j]) & np.isfinite(cap)
        if m.sum() < 30:
            raise ValueError(f"Factor {FACTORS[j]} has insufficient usable coverage")
        v, size, group = raw[m, j].copy(), cap[m].copy(), ind[m]
        median = np.median(v)
        mad = np.median(np.abs(v - median)) * 1.4826
        v = np.clip(v, median - 3 * mad, median + 3 * mad)
        vr, sr = v.copy(), size.copy()
        for key in np.unique(group):
            g = group == key
            vr[g] -= v[g].mean() if g.sum() >= 5 else v.mean()
            sr[g] -= size[g].mean() if g.sum() >= 5 else size.mean()
        denom = sr @ sr
        if denom > 1e-12:
            vr -= (vr @ sr / denom) * sr
        sd = vr.std(ddof=1)
        clean[m, j] = (vr - vr.mean()) / sd if sd > 1e-12 else 0.0
    missing = ~np.isfinite(clean)
    med = np.nanmedian(clean[eligible], axis=0)
    if not np.isfinite(med).all():
        raise ValueError("Whole factor missing; refuse training date")
    filled = np.where(missing, med, clean)
    x = np.concatenate([filled, missing.astype(float)], axis=1).astype(np.float32)
    x[~eligible] = np.nan
    count = np.isfinite(clean).sum(axis=1)
    score = np.divide(np.nansum(clean, axis=1), count,
                      out=np.full(len(raw), np.nan), where=count > 0)
    return clean, x, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="derived/research_training_v4")
    ap.add_argument("--output", default="derived/research_training_v5")
    ap.add_argument("--valuation", default="derived/valuation_at_formation_v1/valuation.parquet")
    ap.add_argument("--normalized", default="derived/research_normalized_v6")
    args = ap.parse_args()
    src, out = ROOT/args.source, ROOT/args.output
    if out.exists():
        raise SystemExit(f"output exists: {out}")
    out.mkdir(parents=True)

    with np.load(src/"samples.npz", allow_pickle=False) as z:
        formation = pd.to_datetime(z["formation_time"])
        entry = z["entry_time"].copy()
        label_end = z["label_end_time"].copy()
        offsets = z["offsets"].copy()
        y = z["y"].copy()
        reason = z["reason"].copy()
        symbols = z["symbols"].copy()
        f17 = z["f17_score"].copy()
    dates = pd.DatetimeIndex(formation)
    old = pd.read_parquet(src/"factors_raw.parquet")
    old = old[["date", "symbol"] + V4_FACTORS]
    val = pd.read_parquet(ROOT/args.valuation)
    val = val[["date", "symbol", "pe_ttm", "pb", "ps_ttm", "div_yield"]].copy()
    val["EP"] = np.where(val.pe_ttm > 0, 1.0/val.pe_ttm, np.nan)
    val["BP"] = np.where(val.pb > 0, 1.0/val.pb, np.nan)
    val["SP"] = np.where(val.ps_ttm > 0, 1.0/val.ps_ttm, np.nan)
    val["DIV"] = val.div_yield
    val = val[["date", "symbol"] + NEW_FACTORS]
    raw = old.merge(val, on=["date", "symbol"], how="left", validate="one_to_one")
    if len(raw) != len(old):
        raise ValueError("Valuation merge changed row count")

    # Eligibility and float cap, formation dates only.
    frames = []
    for p in sorted((ROOT/args.normalized).glob("*.parquet")):
        q = pd.read_parquet(p, columns=["date", "symbol", "eligible", "float_cap"])
        q = q[q.date.isin(dates)]
        frames.append(q)
    state = pd.concat(frames, ignore_index=True)
    if state.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate normalized keys")
    industry = pd.read_pickle(runtime.SOURCE/"data/industry_clean.pkl").set_index("symbol").industry.replace("", "未知")
    if industry.index.has_duplicates:
        raise ValueError("Duplicate industry keys")

    xs, cleans, scores = [], [], []
    for i, d in enumerate(dates):
        a, b = offsets[i], offsets[i+1]
        ss = symbols[a:b]
        r = raw[raw.date == d].set_index("symbol").reindex(ss)
        st = state[state.date == d].set_index("symbol").reindex(ss)
        if r[FACTORS].shape != (len(ss), N) or st.eligible.isna().any():
            raise ValueError(f"Alignment failed on {d}")
        elig = st.eligible.to_numpy(bool) & np.isfinite(st.float_cap.to_numpy(float))
        log_cap = np.log(st.float_cap.to_numpy(float))
        ind = industry.reindex(ss).fillna("未知").to_numpy()
        clean, x, score = preprocess_n(r[FACTORS].to_numpy(float), elig, log_cap, ind)
        xs.append(x)
        cleans.append(clean)
        scores.append(score)

    x = np.concatenate(xs).astype(np.float32)
    clean = np.concatenate(cleans).astype(np.float32)
    if not np.isfinite(x).all() or x.shape != (len(y), N*2):
        raise ValueError("Nonfinite or wrong-shape features")
    np.savez_compressed(out/"samples.npz", formation_time=formation.to_numpy(), entry_time=entry,
                        label_end_time=label_end, offsets=offsets, x=x, y=y, reason=reason,
                        symbols=symbols, f17_score=f17, features=np.asarray(FACTORS))
    raw = raw.sort_values(["date", "symbol"]).reset_index(drop=True)
    clean_frame = pd.DataFrame(clean.astype(float), columns=FACTORS)
    clean_frame.insert(0, "symbol", symbols)
    clean_frame.insert(0, "date", np.repeat(formation.to_numpy(), np.diff(offsets)))
    raw.to_parquet(out/"factors_raw.parquet", index=False)
    clean_frame.to_parquet(out/"factors_clean.parquet", index=False)
    for name in ("calendar.csv", "security_coverage.csv", "label_coverage.csv",
                 "skipped_dates.json", "sample_quality.parquet", "source_manifest.json", "authorization.json"):
        shutil.copy2(src/name, out/name)

    report = json.loads((src/"quality_report.json").read_text(encoding="utf-8"))
    report["rows"] = int(len(y))
    report["valid_labels"] = int(((reason == "ok") & np.isfinite(y)).sum())
    report["securities"] = int(len(np.unique(symbols)))
    report["factor_count"] = N
    report["factor_names"] = FACTORS
    report["checks"]["finite_features"] = bool(np.isfinite(x).all())
    report["checks"]["all_17_factors"] = True
    report["new_factors"] = {"EP": "1/pe_ttm; loss-makers blank", "BP": "1/pb",
                             "SP": "1/ps_ttm", "DIV": "annual dividend yield percent"}
    report["notes"] = report.get("notes", []) + ["v5 adds four Baidu valuation factors; labels and universe unchanged from v4."]
    (out/"quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    acceptance = json.loads((src/"research_acceptance.json").read_text(encoding="utf-8"))
    (out/"research_acceptance.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2), encoding="utf-8")

    from spec import code_hash, file_hash
    manifest = json.loads((src/"package_manifest.json").read_text(encoding="utf-8"))
    manifest["preparation_code"] = code_hash()
    manifest["files"] = {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out/"package_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "rows": len(y), "factors": N, "x": list(x.shape),
                      "finite": bool(np.isfinite(x).all()), "files": len(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
