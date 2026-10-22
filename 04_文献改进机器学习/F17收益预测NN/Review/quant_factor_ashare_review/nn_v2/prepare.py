# Research project author: Gilmour Jia
"""Build model samples from a verified, normalized source package.

This is intentionally separate from repair_sources.py, whose output retains
unresolved gaps and cannot be used as a formally accepted dataset.
"""
import runtime
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data import validate_market, decision_indices, preprocess, forward_label
from factors import build_factors
from spec import FACTORS, FEATURES, SPEC, write_json, file_hash, code_hash


def prepare(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("New output directory required; never overwrite a prepared dataset")
    evidence = json.loads((source/"source_acceptance.json").read_text(encoding="utf-8"))
    if evidence.get("status") != "PASS":
        raise ValueError("Verified source acceptance required before factor rebuild")
    # Bind source verification to exact inputs, not a free-floating PASS flag.
    for name in ("market.parquet","calendar.csv","actions.json"):
        if evidence.get("files",{}).get(name) != file_hash(source/name):
            raise ValueError(f"Unverified source version: {name}")
    frame = pd.read_parquet(source/"market.parquet")
    validate_market(frame)
    required = {"date","symbol","exchange","security_type","float_shares","turnover","listing_date","is_st","industry",
                "status","adjustment_verified","limit_up_price","limit_down_price","no_limit"}
    if not required <= set(frame):
        raise ValueError(f"Missing verified fields: {sorted(required-set(frame))}")
    frame = frame[frame["exchange"].isin(["SH","SZ"]) & frame["security_type"].eq("A")].copy()
    if frame.empty:
        raise ValueError("No verified SH/SZ A shares")
    dates = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(source/"calendar.csv").date),name="date")
    indices = decision_indices(dates)
    symbols = pd.Index(sorted(frame.symbol.unique()),name="symbol")
    def wide(col):
        return frame.pivot(index="date",columns="symbol",values=col).reindex(index=dates,columns=symbols)
    mats = {c: wide(c) for c in frame.columns if c not in ("date","symbol")}
    mats["float_cap"] = mats["raw_close"]*mats["float_shares"]
    mats["amount"] = mats["raw_amount"]
    mats["traded"] = (mats["status"] == "trading") & (mats["raw_volume"] > 0)
    # Listing ages use the full known calendar; warmup must not infer listing
    # from the first quote. Pre-calendar listings conservatively start at zero.
    listing = mats["listing_date"].apply(pd.to_datetime).to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(dates.to_numpy(dtype="datetime64[ns]"), listing)
    age = np.arange(len(dates))[:,None] - positions
    base = (mats["traded"] & mats["is_st"].eq(False) & (age >= 250) &
            (mats["float_cap"] > 0))
    ranks = mats["float_cap"].where(base).rank(axis=1,method="first",pct=True)
    formation_up = mats["raw_open"] >= mats["limit_up_price"]-1e-8
    limits_known = mats["no_limit"].eq(True) | (mats["limit_up_price"].notna() & mats["limit_down_price"].notna())
    eligible = base & (ranks > .30) & (~formation_up | mats["no_limit"].eq(True)) & limits_known
    print("Rebuilding the fixed 17 factors on verified inputs",flush=True)
    factors = build_factors(mats)
    output.mkdir(parents=True)
    rows, xs, ys, reasons, syms, f17s, offsets = [], [], [], [], [], [], [0]
    coverage = []
    raw_records, clean_records = [], []
    for t in indices:
        if dates[t] < pd.Timestamp("2013-01-01"):
            continue
        mask = eligible.iloc[t].to_numpy(bool)
        raw = np.column_stack([factors[f].iloc[t].to_numpy(float) for f in FACTORS])
        clean, x, score = preprocess(raw,mask,np.log(mats["float_cap"].iloc[t].to_numpy(float)),
                                     mats["industry"].iloc[t].fillna("未知").to_numpy())
        y, reason, entry, end = forward_label(mats["adjusted_open"].to_numpy(float),dates,t,
                                             mats["status"].fillna("unknown").to_numpy(),
                                             mats["adjustment_verified"].eq(True).to_numpy())
        count = int(mask.sum())
        rows.append((dates[t].to_datetime64(),entry,end))
        xs.append(x[mask]); ys.append(y[mask]); reasons.append(reason[mask])
        syms.append(symbols.to_numpy(dtype=str)[mask]); f17s.append(score[mask])
        offsets.append(offsets[-1]+count)
        for what, target in ((raw,raw_records),(clean,clean_records)):
            df = pd.DataFrame(what[mask],columns=FACTORS)
            df.insert(0,"symbol",symbols[mask])
            df.insert(0,"date",dates[t])
            target.append(df)
        for reason_code in np.unique(reason[mask]):
            coverage.append({"date":str(dates[t].date()),"reason":str(reason_code),
                             "count":int((reason[mask] == reason_code).sum()),"eligible":count})
    if not rows:
        raise ValueError("No usable formation dates")
    times = np.asarray(rows,dtype="datetime64[ns]")
    np.savez_compressed(output/"samples.npz",formation_time=times[:,0],entry_time=times[:,1],
                        label_end_time=times[:,2],offsets=offsets,x=np.concatenate(xs),y=np.concatenate(ys),
                        reason=np.concatenate(reasons),symbols=np.concatenate(syms),f17_score=np.concatenate(f17s),
                        features=np.asarray(FEATURES))
    pd.concat(raw_records,ignore_index=True).to_parquet(output/"factors_raw.parquet",index=False)
    pd.concat(clean_records,ignore_index=True).to_parquet(output/"factors_clean.parquet",index=False)
    pd.DataFrame(coverage).to_csv(output/"label_coverage.csv",index=False)
    # Explicit open-time permissions, no intraday future bars used.
    # status must mean the exchange's open-time trading status; volume above was
    # used for formation eligibility only. Execution never depends on day's volume.
    buy = (mats["status"] == "trading") & limits_known & (mats["no_limit"].eq(True) | (mats["raw_open"] < mats["limit_up_price"]-1e-8))
    sell = (mats["status"] == "trading") & limits_known & (mats["no_limit"].eq(True) | (mats["raw_open"] > mats["limit_down_price"]+1e-8))
    adv = mats["raw_amount"].rolling(20,min_periods=20).mean().shift(1)
    market = pd.concat({c:mats[c].stack() for c in ("raw_open","raw_close")},axis=1)
    for c,v in (("can_buy",buy),("can_sell",sell),("adv_lag",adv)):
        market[c] = v.stack()
    market.reset_index().to_parquet(output/"market.parquet",index=False)
    (output/"actions.json").write_bytes((source/"actions.json").read_bytes())
    write_json(output/"package_manifest.json",{
        "version":"nn_v2_verified_input_v1","preparation_code":code_hash(),
        "source_acceptance":evidence,"files":{p.name:file_hash(p) for p in output.iterdir() if p.is_file()}})
    write_json(output/"acceptance.json",{"status":"REVIEW_REQUIRED",
        "checks":{},"note":"Review coverage, security omissions and corporate ledgers; never auto-pass."})


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    prepare(a.source,a.output)
