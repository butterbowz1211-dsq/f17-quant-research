# Research project author: Gilmour Jia
"""Bounded-memory preparation from observed sources under an explicit waiver.

This never creates market.parquet/actions.json or marks execution PASS.
"""
import runtime
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data import decision_indices, preprocess, split_samples, validate_keys
from factors import build_factors
from research import MODE, AUTHORIZATION, CONTRACT, research_label, audit_sample_times
from spec import FACTORS, FEATURES, file_hash, write_json, code_hash
from data_mcp.normalize import CODE_CHANGES, bars
from supplement import load_raw as load_supplement_raw
from supplement import load_master as load_supplement_master
from supplement import merge_metadata
from baidu_supplement import load_status as load_baidu_status

ROOT = runtime.ROOT
FLOATS = ["adjusted_open", "adjusted_high", "adjusted_close", "amount", "turnover", "float_cap"]
RAW = ["raw_open", "raw_high", "raw_low", "raw_close", "raw_pre_close", "raw_volume", "raw_amount"]
ADJ = ["adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"]


def read_record(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    if not record.get("summary", {}).get("ok") or not record.get("result", {}).get("data"):
        return None
    return record


def active_code_mask(frame):
    mask = pd.Series(True,index=frame.index)
    for change in CODE_CHANGES:
        date = pd.Timestamp(change["effective"])
        mask &= ~((frame.symbol.eq(change["new"]) & (frame.date < date)) |
                  (frame.symbol.eq(change["old"]) & (frame.date >= date)))
    return mask


def supplemental(snapshot,supplement=None,baidu=None):
    """Completed snapshot responses plus optional BaoStock/Baidu daily status."""
    raw_frames, adj_frames, files = [], [], {}
    candidates = list(snapshot.glob("*.json"))
    candidates += [ROOT/"data_mcp/probe_baostock_extended/qfq_600000.json"]
    evidence = ROOT/"derived/mcp_evidence_v1/daily_distinct_candidates.parquet"
    raw_frames.append(pd.read_parquet(evidence))
    files[str(evidence)] = file_hash(evidence)
    for p in sorted(candidates):
        rec = read_record(p) if p.name not in ("summary.json", "tools.json", "initialize.json") else None
        if rec is None or rec.get("tool") != "get_history_k_data":
            continue
        files[str(p)] = file_hash(p)
        if rec["arguments"].get("adjustflag", "3") == "3":
            raw_frames.append(bars(rec))
        else:
            df = pd.DataFrame(rec["result"]["data"])
            if not df.code.eq(rec["arguments"]["code"]).all():
                raise ValueError("Invalid supplemental adjustment source")
            df["date"] = pd.to_datetime(df.date).astype("datetime64[ns]")
            if not df.date.between(rec["arguments"]["start_date"], rec["arguments"]["end_date"]).all():
                raise ValueError("Supplemental dates outside request")
            df["symbol"] = df.code.str.replace(".", "", regex=False)
            # Successor queries backfill predecessor raw prices with flag=3.
            # Those inactive-code rows must be removed before price acceptance.
            df = df[active_code_mask(df)].copy()
            if not df.adjustflag.eq("2").all():
                raise ValueError("Unadjusted rows inside an active-code adjusted history")
            validate_keys(df)
            out = df[["date", "symbol"]].copy()
            for c in ("open", "high", "low", "close"):
                out["adjusted_"+c] = pd.to_numeric(df[c], errors="coerce")
            adj_frames.append(out)
            # Adjustment flags affect prices, not these volume/status fields.
            # Do not relabel adjusted OHLC as unadjusted execution prices.
            meta = df[["date","symbol"]].copy()
            meta["raw_volume"] = pd.to_numeric(df.volume,errors="coerce")
            meta["raw_amount"] = pd.to_numeric(df.amount,errors="coerce")
            meta["turnover"] = pd.to_numeric(df.turn,errors="coerce")/100
            meta["is_st"] = df.isST.map({"1":True,"0":False}).astype("boolean")
            meta["daily_status"] = df.tradestatus.map({"1":"trading","0":"suspended"})
            raw_frames.append(meta)
    if supplement is not None:
        raw_frames.append(load_supplement_raw(supplement))
        for name in ("manifest.json","coverage.csv","security_master.parquet"):
            p = Path(supplement)/name
            if p.exists():
                files[str(p)] = file_hash(p)
    # Explicit source precedence; absent fields never erase earlier evidence.
    raw = raw_frames[0].set_index(["date","symbol"])
    for f in raw_frames[1:]:
        raw = f.set_index(["date","symbol"]).combine_first(raw)
    raw = raw.reset_index()
    status = None
    if baidu is not None:
        status = load_baidu_status(baidu)
        for name in ("manifest.json","coverage.csv","security_master.parquet"):
            p = Path(baidu)/name
            if p.exists():
                files[str(p)] = file_hash(p)
    adj = pd.concat(adj_frames, ignore_index=True) if adj_frames else pd.DataFrame(columns=["date","symbol"]+ADJ)
    if adj.duplicated(["date","symbol"]).any():
        raise ValueError("Overlapping adjusted sources: select one complete history per code")
    return raw, adj, status, files


def metadata(supplement=None):
    basic_path = ROOT/"data_mcp/probe_tushare_v1/stock_basic_L.json"
    # Existing local response only. No Tushare requests are made.
    listed = pd.DataFrame(read_record(basic_path)["result"]["data"])
    listed["symbol"] = listed.ts_code.str[7:].str.lower()+listed.ts_code.str[:6]
    listing = pd.Series(pd.to_datetime(listed.list_date, format="%Y%m%d", errors="coerce").to_numpy(),
                        index=listed.symbol).to_dict()
    master_path = ROOT/"derived/mcp_evidence_v1/security_master_candidates.csv"
    master = pd.read_csv(master_path)
    exits = {}
    for row in master.itertuples():
        if pd.notna(row.ipoDate):
            listing[row.symbol] = pd.Timestamp(row.ipoDate)
        if pd.notna(row.outDate):
            exits[row.symbol] = pd.Timestamp(row.outDate)
    industry_path = runtime.SOURCE/"data/industry_clean.pkl"
    ind = pd.read_pickle(industry_path).set_index("symbol").industry.replace("", "未知")
    if ind.index.has_duplicates:
        raise ValueError("Duplicate industry keys")
    sources = {str(p):file_hash(p) for p in (basic_path,master_path,industry_path)}
    if supplement is not None:
        master = load_supplement_master(supplement)
        listing, exits = merge_metadata(listing,exits,master)
        sources[str(Path(supplement)/"security_master.parquet")] = file_hash(
            Path(supplement)/"security_master.parquet")
    return listing, exits, ind, sources


def coherent_prices(frame):
    """Quarantine malformed bars; absorb independent 0.01 vendor rounding only."""
    frame[ADJ] = frame[ADJ].astype(float)
    raw, adj = frame[RAW[:4]].to_numpy(float), frame[ADJ].to_numpy(float)
    def invalid(a):
        o,h,l,c = a.T
        finite = np.isfinite(a).all(axis=1)
        return finite & ((a <= 0).any(axis=1) | (h+1e-5 < np.maximum(o,c)) |
                         (l-1e-5 > np.minimum(o,c)) | (l > h))
    raw_bad, adj_bad = invalid(raw), invalid(adj)
    factor = np.divide(adj[:,3], raw[:,3], out=np.full(len(frame),np.nan), where=raw[:,3]>0)
    implied = raw*factor[:,None]
    # Both adjusted close and each adjusted OHLC are independently rounded.
    tolerance = .015 + .015*np.abs(np.divide(raw,raw[:,3,None],
                                            out=np.zeros_like(raw),where=raw[:,3,None]>0))
    inconsistent = (np.isfinite(raw).all(axis=1) & np.isfinite(adj).all(axis=1) &
                    (np.abs(adj-implied) > tolerance).any(axis=1))
    unit_bad = ((frame.raw_volume > 0) & (frame.raw_amount > 0) &
                ((frame.raw_amount/frame.raw_volume < frame.raw_low*.95) |
                 (frame.raw_amount/frame.raw_volume > frame.raw_high*1.05))).to_numpy()
    negative = ((frame.raw_volume < 0) | (frame.raw_amount < 0)).to_numpy()
    bad = raw_bad | adj_bad | inconsistent | unit_bad | negative
    # Use coherent raw OHLC scaled by vendor adjustment multiplier, not VWAP.
    good = np.isfinite(implied).all(axis=1) & ~bad & (factor > 0)
    frame.loc[good, ADJ] = implied[good]
    frame.loc[bad, ADJ] = np.nan
    return bad, {"bad_raw_ohlc":int(raw_bad.sum()),"bad_adjusted_ohlc":int(adj_bad.sum()),
                 "incoherent_adjustment":int(inconsistent.sum()),"units":int(unit_bad.sum()),
                 "negative_volume_amount":int(negative.sum()),"quarantined":int(bad.sum())}


def normalize(union, work, snapshot, supplement=None, baidu=None):
    work.mkdir(parents=True, exist_ok=False)
    raw, adj, status, hashes = supplemental(snapshot, supplement, baidu)
    listing, exits, industry, meta_hashes = metadata(supplement)
    hashes.update(meta_hashes)
    manifest_path = union/"manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes[str(manifest_path)] = file_hash(manifest_path)
    observed, all_dates = {}, set()
    for p in sorted(union.glob("*.parquet")):
        if file_hash(p) != source_manifest["files"][p.name]["sha256"]:
            raise ValueError(f"Source checksum changed: {p}")
        hashes[str(p)] = file_hash(p)
        mini = pd.read_parquet(p, columns=["date","symbol"])
        all_dates.update(mini.date.unique())
        for symbol, first in mini.groupby("symbol").date.min().items():
            observed[symbol] = min(observed.get(symbol,first),first)
    dates = pd.DatetimeIndex(sorted(all_dates), name="date")
    decision_indices(dates)
    calendar_paths = [ROOT/"data_mcp/probe_baostock_extended/calendar_2008_2013.json"]
    calendar_paths += list(snapshot.glob("calendar_*.json"))
    calendar_checks = []
    for p in calendar_paths:
        rec = read_record(p)
        if rec is None:
            continue
        cal = pd.DataFrame(rec["result"]["data"])
        vendor_dates = pd.DatetimeIndex(pd.to_datetime(cal.loc[cal.is_trading_day.eq("1"),"calendar_date"]))
        start = max(pd.Timestamp(rec["arguments"]["start_date"]), dates[0])
        end = min(pd.Timestamp(rec["arguments"]["end_date"]), dates[-1])
        if set(dates[(dates >= start) & (dates <= end)]) != set(vendor_dates[(vendor_dates >= start) & (vendor_dates <= end)]):
            raise ValueError(f"Source calendar disagrees with BaoStock: {p}")
        hashes[str(p)] = file_hash(p)
        calendar_checks.append({"start":str(start.date()),"end":str(end.date()),"match":True})
    daily_rows, clean_stats, coverage = 0, [], []
    normalized_hashes = {}
    adj_symbols = set(adj.symbol.unique())
    for path in sorted(union.glob("*.parquet")):
        year = int(path.stem)
        f = pd.read_parquet(path, columns=["date","symbol"]+RAW+ADJ+
                            ["outstanding_share","turnover","is_st","st_snapshot_age_days"])
        f[ADJ] = f[ADJ].astype(float)
        if status is not None:
            bai = status[status.date.dt.year == year].copy()
            bai = bai.rename(columns={c:c+"_bai" for c in bai.columns
                                      if c not in ("date","symbol")})
            f = f.merge(bai,on=["date","symbol"],how="left",validate="one_to_one")
        for cols, supplement in ((RAW+["turnover","is_st","daily_status"],raw),
                                  (ADJ,adj)):
            sub = supplement[supplement.date.dt.year == year]
            available = [c for c in cols if c in sub]
            f = f.merge(sub[["date","symbol"]+available],on=["date","symbol"],how="outer",
                        suffixes=("", "_bao"),validate="one_to_one")
            for col in available:
                if col+"_bao" in f:
                    if col in ADJ:
                        # Never splice the old adjusted series into a selected
                        # BaoStock history when the new provider has a gap.
                        chosen = f.symbol.isin(adj_symbols)
                        f.loc[chosen,col] = f.loc[chosen,col+"_bao"]
                    else:
                        f[col] = f[col+"_bao"].combine_first(f[col])
            if cols[0] == "raw_open":
                f["daily_st"] = f.get("is_st_bao",pd.Series(np.nan,index=f.index)).notna()
            f = f.drop(columns=[c for c in f if c.endswith("_bao")])
        if "is_st_bai" in f:
            # Baidu daily rows are complete for 2010-2025 and take precedence.
            f["daily_st"] = f["is_st_bai"].notna()
            f["is_st"] = f["is_st_bai"].combine_first(f["is_st"])
            for c in ("float_shares","total_shares","free_float","limit_up","limit_down"):
                if c+"_bai" in f:
                    f[c] = f[c+"_bai"]
            f["turnover"] = f["turnover_bai"].combine_first(f["turnover"])
            f["daily_status"] = f["daily_status_bai"].combine_first(f["daily_status"])
            f = f.drop(columns=[c for c in f if c.endswith("_bai")])
        f = f[f.symbol.str.match(r"^(sh60|sh688|sz00|sz30)\d+") &
              f.date.isin(dates) & f.symbol.ne("sh689009")].copy()
        inactive = ~active_code_mask(f)
        f = f[~inactive].sort_values(["date","symbol"]).reset_index(drop=True)
        validate_keys(f)
        bad, stat = coherent_prices(f)
        stat.update(year=year, rows=len(f), inactive_alias_rows=int(inactive.sum()))
        clean_stats.append(stat)
        f["daily_status"] = f.daily_status.fillna("unknown")
        exit_date = pd.to_datetime(f.symbol.map(exits))
        f["terminal"] = f.date.ge(exit_date) & exit_date.notna()
        # Code changes are not interpreted as a zero terminal cash settlement.
        f["boundary"] = False
        for change in CODE_CHANGES:
            d = pd.Timestamp(change["effective"])
            f.loc[f.symbol.eq(change["old"]), "terminal"] = False
            # Boundaries are represented later even when there is no quote.
        f["suspended"] = f.daily_status.eq("suspended")
        f["traded"] = (f.raw_volume.gt(0) & f.raw_amount.gt(0) &
                       f.raw_close.gt(0) & ~f.suspended & ~f.terminal & ~bad)
        f["st_source"] = np.where(f.daily_st, 1, np.where(f.is_st.notna(),2,0)).astype("int8")
        for c in ("float_shares","limit_up"):
            if c not in f:
                f[c] = np.nan
        exact_shares = f.float_shares.where(f.float_shares.gt(0))
        snapshot_shares = f.outstanding_share.where(f.outstanding_share.gt(0))
        base_shares = exact_shares.combine_first(snapshot_shares)
        implied = f.raw_volume / f.turnover.where(f.turnover > 0)
        shares = base_shares.combine_first(implied)
        f["cap_proxy"] = base_shares.isna()
        f["float_cap"] = f.raw_close * shares
        f["amount"] = f.raw_amount
        # A first-quote fallback delays eligibility; never backdate a new listing.
        known_listing = pd.to_datetime(f.symbol.map(listing))
        first_quote = pd.to_datetime(f.symbol.map(observed))
        chosen_listing = known_listing.fillna(first_quote)
        f["listing_proxy"] = known_listing.isna()
        age = dates.get_indexer(f.date) - dates.searchsorted(chosen_listing.to_numpy())
        base = f.traded & f.is_st.eq(False) & (age >= 250) & f.float_cap.gt(0)
        rank = f.float_cap.where(base).groupby(f.date).rank(method="first",pct=True)
        rate = np.where(f.symbol.str.startswith("sh688") |
                        (f.symbol.str.startswith("sz30") & f.date.ge("2020-08-24")),.20,.10)
        proxy_upper = pd.Series(np.floor(f.raw_pre_close.to_numpy()*(1+rate)*100+.5)/100,
                                index=f.index)
        upper = f.limit_up.where(f.limit_up.gt(0), proxy_upper)
        f["limit_proxy"] = ~f.limit_up.gt(0)
        limit_known = f.limit_up.gt(0) | f.raw_pre_close.gt(0)
        f["eligible"] = base & rank.gt(.30) & limit_known & (f.raw_open < upper-1e-6)
        f["bad_price"] = bad
        # Quarantined/absent close stays NaN. No filling historical price gaps.
        save_cols = ["date","symbol"]+FLOATS+["eligible","traded","suspended","terminal",
                     "st_source","cap_proxy","listing_proxy","bad_price","st_snapshot_age_days"]
        dest = work/path.name
        f[save_cols].to_parquet(dest,index=False)
        normalized_hashes[dest.name] = file_hash(dest)
        for s,g in f.groupby("symbol"):
            coverage.append({"symbol":s,"year":year,"observed_rows":len(g),
                             "adjusted_rows":int(g.adjusted_close.notna().sum()),
                             "eligible_rows":int(g.eligible.sum()),
                             "daily_st_rows":int(g.st_source.eq(1).sum()),
                             "snapshot_st_rows":int(g.st_source.eq(2).sum()),
                             "unknown_st_rows":int(g.st_source.eq(0).sum()),
                             "cap_proxy_rows":int(g.cap_proxy.sum()),
                             "listing_proxy_rows":int(g.listing_proxy.sum())})
        daily_rows += len(f)
        print(f"normalized {year}: {len(f):,} rows; quarantined {stat['quarantined']}",flush=True)
        del f
        gc.collect()
    write_json(work/"manifest.json",{"sources":hashes,"normalized_files":normalized_hashes,
                                    "adjusted_BaoStock_symbols":sorted(adj_symbols),
                                    "code_changes":CODE_CHANGES,"statistics":clean_stats,
                                    "calendar_checks":calendar_checks})
    pd.DataFrame(coverage).to_csv(work/"security_coverage.csv",index=False)
    return dates, industry, exits


def prepare(union, work, snapshot, output, supplement=None, baidu=None, end_year=2025):
    if output.exists() or work.exists():
        raise ValueError("Use new version directories")
    dates, industry, exits = normalize(union,work,snapshot,supplement,baidu)
    manifest = json.loads((work/"manifest.json").read_text(encoding="utf-8"))
    coverage = pd.read_csv(work/"security_coverage.csv")
    symbols = pd.Index(sorted(coverage.symbol.unique()),name="symbol")
    shape = (len(dates), len(symbols))
    arrays = {c:np.full(shape,np.nan,dtype=np.float32) for c in FLOATS}
    bools = {c:np.zeros(shape,dtype=bool) for c in
             ("eligible","traded","suspended","terminal","boundary","cap_proxy","listing_proxy","bad_price")}
    st_source = np.zeros(shape,dtype=np.int8)
    for p in sorted(work.glob("*.parquet")):
        f = pd.read_parquet(p)
        ij = dates.get_indexer(f.date), symbols.get_indexer(f.symbol)
        for c in arrays:
            arrays[c][ij] = f[c].to_numpy(np.float32)
        for c in bools:
            if c in f:
                bools[c][ij] = f[c].to_numpy(bool)
        st_source[ij] = f.st_source.to_numpy(np.int8)
        del f
    for s, end in exits.items():
        if s in symbols and not any(x["old"] == s for x in CODE_CHANGES):
            bools["terminal"][dates >= end, symbols.get_loc(s)] = True
    for change in CODE_CHANGES:
        pos = dates.searchsorted(pd.Timestamp(change["effective"]))
        if pos < len(dates):
            for symbol in (change["old"],change["new"]):
                if symbol in symbols:
                    bools["boundary"][pos,symbols.get_loc(symbol)] = True
    output.mkdir(parents=True)
    pd.DataFrame({"date":dates}).to_csv(output/"calendar.csv",index=False)
    coverage.to_csv(output/"security_coverage.csv",index=False)
    write_json(output/"source_manifest.json",manifest)
    write_json(output/"authorization.json",{"user_instruction":AUTHORIZATION,"mode":MODE,
               "supersedes":"strict acceptance prerequisite for training only",
               "date":"2026-10-20","execution_accepted":False})
    rows, xs, ys, reasons, syms, f17s, offsets = [], [], [], [], [], [], [0]
    label_counts, skipped, raw_records, clean_records, flags = [], [], [], [], []
    indices = decision_indices(dates)
    m_all = {k:pd.DataFrame(v,index=dates,columns=symbols) for k,v in arrays.items()}
    m_all["traded"] = pd.DataFrame(bools["traded"],index=dates,columns=symbols)
    industry = industry.reindex(symbols).fillna("未知")
    # One pass over the full panel. Rebuilding a separate 800-session window for
    # every year recomputed the same rolling statistics 13 times.
    m_all["industry"] = industry
    print(f"rebuilding all 17 factors once over {dates[0].date()} to {dates[-1].date()}",flush=True)
    factors_all = build_factors(m_all)
    del m_all
    gc.collect()
    for year in range(2013,end_year+1):
        selected = [int(t) for t in indices if dates[t].year == year]
        if not selected:
            continue
        for t in selected:
            mask = bools["eligible"][t]
            raw = np.column_stack([factors_all[f].iloc[t].to_numpy(float) for f in FACTORS])
            with np.errstate(divide="ignore",invalid="ignore"):
                log_cap = np.log(arrays["float_cap"][t].astype(float))
            try:
                clean, x, score = preprocess(raw,mask,log_cap,industry.to_numpy())
            except ValueError as e:
                skipped.append({"date":str(dates[t].date()),"reason":str(e),
                                "eligible":int(mask.sum()),
                                "finite_factors":dict(zip(FACTORS,np.isfinite(raw[mask]).sum(axis=0).tolist()))})
                continue
            y, reason, entry, end = research_label(arrays["adjusted_open"],dates,t,
                        bools["suspended"],bools["terminal"],bools["boundary"])
            rows.append((dates[t].to_datetime64(),entry,end))
            xs.append(x[mask]); ys.append(y[mask]); reasons.append(reason[mask])
            syms.append(symbols.to_numpy(dtype=str)[mask]); f17s.append(score[mask])
            offsets.append(offsets[-1]+int(mask.sum()))
            for values, records in ((raw,raw_records),(clean,clean_records)):
                f = pd.DataFrame(values[mask],columns=FACTORS)
                f.insert(0,"symbol",symbols[mask]); f.insert(0,"date",dates[t])
                records.append(f)
            for r in np.unique(reason[mask]):
                label_counts.append({"date":str(dates[t].date()),"reason":str(r),
                                     "count":int((reason[mask]==r).sum()),"eligible":int(mask.sum())})
            q = pd.DataFrame({"date":dates[t],"symbol":symbols[mask],
                              "st_source":st_source[t,mask],"cap_proxy":bools["cap_proxy"][t,mask],
                              "listing_proxy":bools["listing_proxy"][t,mask],
                              "adjustment_verified":False,"label_basis":"vendor_adjustment_research_proxy"})
            flags.append(q)
    del factors_all
    gc.collect()
    times = np.asarray(rows,dtype="datetime64[ns]")
    np.savez_compressed(output/"samples.npz",formation_time=times[:,0],entry_time=times[:,1],
                        label_end_time=times[:,2],offsets=offsets,x=np.concatenate(xs),y=np.concatenate(ys),
                        reason=np.concatenate(reasons),symbols=np.concatenate(syms),f17_score=np.concatenate(f17s),
                        features=np.asarray(FEATURES))
    pd.concat(raw_records,ignore_index=True).to_parquet(output/"factors_raw.parquet",index=False)
    pd.concat(clean_records,ignore_index=True).to_parquet(output/"factors_clean.parquet",index=False)
    pd.concat(flags,ignore_index=True).to_parquet(output/"sample_quality.parquet",index=False)
    pd.DataFrame(label_counts).to_csv(output/"label_coverage.csv",index=False)
    write_json(output/"skipped_dates.json",skipped)
    from data import load_samples
    samples = load_samples(output/"samples.npz")
    audit_sample_times(samples,dates)
    folds = []
    for year in range(2021,end_year+1):
        tr, va, final, outer = split_samples(samples,year)
        folds.append({"year":year,"train_dates":len(tr),"validation_dates":len(va),
                      "final_dates":len(final),"outer_dates":len(outer),
                      "train_labels":sum(int(s.valid.sum()) for s in tr),
                      "latest_train_label":str(max(s.label_end_time for s in tr)),
                      "latest_validation_label":str(max(s.label_end_time for s in va)),
                      "first_outer_formation":str(outer[0].formation_time)})
    all_symbols = set(np.concatenate(syms))
    supplement_summary = None
    if supplement is not None:
        sm = json.loads((Path(supplement)/"manifest.json").read_text(encoding="utf-8"))
        supplement_summary = {key:sm.get(key) for key in
            ("version","symbols","downloaded_symbols","empty_symbols",
             "not_downloaded_symbols","rows")}
        supplement_summary["complete"] = bool(sm.get("complete",
            sm.get("not_downloaded_symbols") == 0))
    if baidu is not None:
        bm = json.loads((Path(baidu)/"manifest.json").read_text(encoding="utf-8"))
        supplement_summary = {key:bm.get(key) for key in
            ("version","symbols","rows","complete")}
        supplement_summary["source"] = "Baidu stock_full daily status"
    eligible_mask = bools["eligible"]
    st_counts = {name:int((st_source[eligible_mask] == code).sum())
                 for name,code in (("daily",1),("snapshot",2),("unknown",0))}
    eligible_total = int(eligible_mask.sum())
    write_json(output/"quality_report.json",{
        "mode":MODE,"execution_accepted":False,
        "checks":dict.fromkeys(["finite_features","unique_keys","label_endpoints",
                               "purge_all_folds","all_17_factors","source_hashes"],True),
        "formation_dates":len(samples),"rows":offsets[-1],"securities":len(all_symbols),
        "valid_labels":sum(int(s.valid.sum()) for s in samples),
        "first_formation":str(samples[0].formation_time),"last_formation":str(samples[-1].formation_time),
        "folds":folds,"skipped_dates":skipped,
        "known_symbols_never_eligible":sorted(set(symbols)-all_symbols),
        "eligible_st_source":st_counts,
        "eligible_rows":eligible_total,
        "supplement":supplement_summary,
        "limitations":["Incomplete delisting/merger terminal consideration: affected labels excluded, selection bias remains.",
            "Vendor adjusted prices, no independently reconciled corporate-action ledger.",
            f"Daily ST/status covers {st_counts['daily']:,}/{eligible_total:,} eligible rows; "
            f"{st_counts['snapshot']:,} still use backward month-end snapshots and "
            f"{st_counts['unknown']:,} are unknown.",
            "Static industry; source share history not fully PIT verified; BaoStock turnover-implied share proxy where needed.",
            "Formation limit screen uses vendor daily limit prices where available; board-rate preclose proxy otherwise.",
            "Float cap uses vendor float shares where available; turnover-implied proxy otherwise.",
            "Code effective dates enforced; price histories are not spliced across incompatible vendor adjustment bases.",
            "Historical 2021-2026 research is contaminated; this is not a pristine independent test.",
            "Execution performance and superiority over original F17 cannot be certified from this training package."]})
    write_json(output/"package_manifest.json",{
        "version":CONTRACT["version"],"training_contract":CONTRACT,"preparation_code":code_hash(),
        "files":{p.name:file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}})
    write_json(output/"research_acceptance.json",{
        "status":"RESEARCH_TRAINING_AUTHORIZED_WITH_LIMITATIONS","execution_accepted":False,
        "basis":"Latest explicit user instruction plus numerical/schema/time-boundary checks",
        "strict_gate":"unchanged; requires independent data/execution acceptance"})
    print(json.dumps({"prepared":str(output),"formation_dates":len(samples),
                      "rows":offsets[-1],"valid_labels":sum(int(s.valid.sum()) for s in samples)},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--union",type=Path,default=ROOT/"derived/source_union_v1")
    ap.add_argument("--work",type=Path,required=True)
    ap.add_argument("--snapshot",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--supplement",type=Path,default=None,
                    help="Optional local BaoStock supplement directory")
    ap.add_argument("--baidu",type=Path,default=None,
                    help="Optional Baidu stock_full daily status supplement")
    ap.add_argument("--end-year",type=int,default=2025,
                    help="Last formation year to materialize")
    a = ap.parse_args()
    prepare(a.union,a.work,a.snapshot,a.output,a.supplement,a.baidu,a.end_year)
