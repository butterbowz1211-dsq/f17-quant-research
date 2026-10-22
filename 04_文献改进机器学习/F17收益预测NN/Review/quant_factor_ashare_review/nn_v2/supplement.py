# Research project author: Gilmour Jia
"""Incremental BaoStock security master and daily-status supplement.

The supplement is independent of the frozen research package. It downloads
one history per security, keeps a resumable ledger, and materializes compact
year partitions plus a coverage table. It never marks execution accepted and
never rewrites an existing research version directory.
"""
import runtime
import argparse
import hashlib
import json
import os
import socket
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = runtime.ROOT
DEFAULT = ROOT/"derived"/"supplement_v1"
START = "2010-01-01"
END = "2026-09-28"
PAGE = 2000
FIELDS = ["date","code","open","high","low","close","preclose","volume",
          "amount","turn","tradestatus","isST"]
RAW_MAP = {"open":"raw_open","high":"raw_high","low":"raw_low",
           "close":"raw_close","preclose":"raw_pre_close",
           "volume":"raw_volume","amount":"raw_amount"}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    os.replace(tmp,path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_parquet(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    frame.to_parquet(tmp,index=False)
    os.replace(tmp,path)


def read_result(rs):
    """Consume every row in the current response and any later pages.

    BaoStock's next() does not advance cur_row_num; get_row_data() does. A
    first response can also contain more than one nominal page. Both cases are
    handled without relying on next() for the in-page cursor.
    """
    rows = []
    while True:
        rows.extend(rs.data[rs.cur_row_num:])
        rs.cur_row_num = len(rs.data)
        if len(rs.data) != PAGE:
            break
        if rs.error_code != "0":
            break
        if not rs.next():
            break
    return rows


def normalize_bars(rows,fields,code):
    if not rows:
        return pd.DataFrame(columns=["date","symbol"]+list(RAW_MAP.values())+
                            ["turnover","is_st","daily_status","code"])
    frame = pd.DataFrame(rows,columns=fields)
    if not frame.code.eq(code).all():
        raise ValueError(f"BaoStock returned a foreign code for {code}")
    out = pd.DataFrame({"date":pd.to_datetime(frame.date,errors="coerce"),
                        "symbol":code.replace(".",""),"code":code})
    for src,dst in RAW_MAP.items():
        out[dst] = pd.to_numeric(frame[src],errors="coerce")
    out["turnover"] = pd.to_numeric(frame.turn,errors="coerce")/100.0
    out["is_st"] = frame.isST.map({"1":True,"0":False}).astype("boolean")
    out["daily_status"] = frame.tradestatus.map({"1":"trading","0":"suspended"})
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _query(code,start,end,adjustflag):
    import baostock as bs
    socket.setdefaulttimeout(60)
    rs = bs.query_history_k_data_plus(code,",".join(FIELDS),start,end,
                                     frequency="d",adjustflag=str(adjustflag))
    if rs is None:
        raise RuntimeError("BaoStock returned no result object")
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock {rs.error_code}: {rs.error_msg}")
    return normalize_bars(read_result(rs),rs.fields,code)


def query_bars(code,start=START,end=END,adjustflag="3",retries=5):
    """Query one full history; a shorter history is valid, not a lost page.

    BaoStock returns complete history through its page protocol. A delisted or
    suspended security legitimately ends before the requested end date, so the
    response must not be split into synthetic sub-windows just because its last
    observation is before END.
    """
    last = None
    for attempt in range(1,retries+1):
        try:
            return _query(code,start,end,adjustflag)
        except Exception as exc:
            last = exc
            time.sleep(min(8,1.5*attempt))
            try:
                import baostock as bs
                socket.setdefaulttimeout(60)
                try:
                    bs.logout()
                except Exception:
                    pass
                bs.login()
            except Exception:
                pass
    raise RuntimeError(f"{code}: {last}")


def _master_frame(rows,fields):
    frame = pd.DataFrame(rows,columns=fields)
    out = pd.DataFrame({"code":frame.code,"symbol":frame.code.str.replace(".","",regex=False),
                        "name":frame.code_name,"ipo":pd.to_datetime(frame.ipoDate,errors="coerce"),
                        "out":pd.to_datetime(frame.outDate,errors="coerce"),
                        "type":frame.type.astype("string"),"status":frame.status.astype("string")})
    out["is_stock"] = out.type.eq("1")
    out["listed"] = out.status.eq("1")
    return out.sort_values("symbol").reset_index(drop=True)


def fetch_master(out=DEFAULT):
    import baostock as bs
    socket.setdefaulttimeout(60)
    bs.login()
    try:
        rs = bs.query_stock_basic()
        if rs.error_code != "0":
            raise RuntimeError(f"BaoStock {rs.error_code}: {rs.error_msg}")
        rows = read_result(rs)
        frame = _master_frame(rows,rs.fields)
    finally:
        bs.logout()
    atomic_parquet(frame,Path(out)/"security_master.parquet")
    write_json(Path(out)/"master_manifest.json",{"requested":now(),"rows":len(frame),
               "stocks":int(frame.is_stock.sum()),"listed":int(frame.listed.sum()),
               "fields":rs.fields,"sha256":sha256(Path(out)/"security_master.parquet")})
    return frame


def load_master(out=DEFAULT):
    path = Path(out)/"security_master.parquet"
    if not path.exists():
        return fetch_master(out)
    return pd.read_parquet(path)


def _meta_path(directory,symbol):
    return directory/"_meta"/f"{symbol}.json"


def _download_one(task):
    code,start,end,out_dir,adjustflag = task
    directory = Path(out_dir)/("bars" if str(adjustflag)=="3" else "qfq")
    symbol = code.replace(".","")
    meta_path = _meta_path(directory,symbol)
    target = directory/f"{symbol}.parquet"
    if target.exists() and meta_path.exists():
        try:
            meta = read_json(meta_path)
            if meta.get("end") == end:
                return {"symbol":symbol,"status":"skipped","rows":int(meta.get("rows",0)),
                        "first":meta.get("first"),"last":meta.get("last")}
        except Exception:
            pass
    try:
        frame = query_bars(code,start,end,adjustflag)
        if len(frame) == 0:
            write_json(meta_path,{"symbol":symbol,"code":code,"start":start,"end":end,
                                   "adjustflag":str(adjustflag),"rows":0,"downloaded":now()})
            return {"symbol":symbol,"status":"empty","rows":0}
        atomic_parquet(frame,target)
        meta = {"symbol":symbol,"code":code,"start":start,"end":end,
                "adjustflag":str(adjustflag),"rows":len(frame),
                "first":str(frame.date.min().date()),"last":str(frame.date.max().date()),
                "sha256":sha256(target),"downloaded":now()}
        write_json(meta_path,meta)
        return {"symbol":symbol,"status":"ok","rows":len(frame),
                "first":meta["first"],"last":meta["last"]}
    except Exception as exc:
        return {"symbol":symbol,"status":"error","error":str(exc),
                "trace":traceback.format_exc(limit=3)}


def _worker_init():
    import baostock as bs
    socket.setdefaulttimeout(60)
    bs.login()


def _worker_exit():
    import baostock as bs
    try:
        bs.logout()
    except Exception:
        pass


def download(out=DEFAULT,workers=4,start=START,end=END,limit=0,only=None):
    out = Path(out)
    master = load_master(out)
    stocks = master[master.is_stock].copy()
    if only:
        stocks = stocks[stocks.symbol.isin(set(only))]
    tasks = [(row.code,start,end,str(out),"3") for row in stocks.itertuples()]
    if limit:
        tasks = tasks[:int(limit)]
    ledger = out/"download_ledger.jsonl"
    ledger.parent.mkdir(parents=True,exist_ok=True)
    ok = skipped = errors = 0
    started = time.time()
    with ledger.open("a",encoding="utf-8") as log:
        import multiprocessing as mp
        with mp.Pool(workers,initializer=_worker_init,maxtasksperchild=40) as pool:
            for i,result in enumerate(pool.imap_unordered(_download_one,tasks,chunksize=1),1):
                log.write(json.dumps({"at":now(),**result},ensure_ascii=False)+"\n")
                log.flush()
                if result["status"] == "ok":
                    ok += 1
                elif result["status"] == "empty":
                    skipped += 1
                elif result["status"] == "skipped":
                    skipped += 1
                else:
                    errors += 1
                if i % 25 == 0 or i == len(tasks):
                    elapsed = time.time()-started
                    rate = i/elapsed if elapsed else 0
                    print(f"[supplement] {i}/{len(tasks)} ok={ok} skipped={skipped} "
                          f"errors={errors} {rate:.2f}/s elapsed={elapsed:.0f}s",flush=True)
    summary = {"requested":now(),"tasks":len(tasks),"ok":ok,"skipped":skipped,
               "errors":errors,"start":start,"end":end,"workers":workers,
               "elapsed_seconds":round(time.time()-started,1)}
    write_json(out/"download_summary.json",summary)
    return summary


def build(out=DEFAULT):
    """Materialize year partitions and a ledger covering every known security."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = Path(out)
    bars = sorted((out/"bars").glob("*.parquet"))
    if not bars:
        raise ValueError("No downloaded bars to build")
    master = load_master(out)
    stocks = master[master.is_stock].copy()
    known = set(stocks.symbol.astype(str))
    year_dir = out/"by_year"
    year_dir.mkdir(parents=True,exist_ok=True)
    writers, bar_cov = {}, {}
    for path in bars:
        frame = pd.read_parquet(path)
        # Different batches infer int64 vs float64 for volume; pin the schema
        # before appending to a shared year file.
        frame["date"] = pd.to_datetime(frame.date)
        frame["symbol"] = frame.symbol.astype(str)
        frame["code"] = frame.code.astype(str)
        for col in RAW_MAP.values():
            frame[col] = pd.to_numeric(frame[col],errors="coerce").astype("float64")
        frame["turnover"] = pd.to_numeric(frame.turnover,errors="coerce").astype("float64")
        frame["is_st"] = frame.is_st.astype("boolean")
        frame["daily_status"] = frame.daily_status.astype("string")
        symbol = path.stem
        bar_cov[symbol] = {"symbol":symbol,"rows":len(frame),
                           "first_date":frame.date.min(),"last_date":frame.date.max(),
                           "trading_days":int(frame.daily_status.eq("trading").sum()),
                           "suspended_days":int(frame.daily_status.eq("suspended").sum()),
                           "st_days":int(frame.is_st.fillna(False).sum()),
                           "unknown_status":int(frame.daily_status.isna().sum()),
                           "status":"ok"}
        frame = frame.assign(year=frame.date.dt.year)
        for year,part in frame.groupby("year",sort=True):
            target = year_dir/f"{int(year)}.parquet"
            if year not in writers:
                writers[year] = pq.ParquetWriter(target,pa.Table.from_pandas(part,preserve_index=False).schema)
            writers[year].write_table(pa.Table.from_pandas(part,preserve_index=False))
    for writer in writers.values():
        writer.close()
    coverage = []
    for symbol in sorted(known | set(bar_cov)):
        if symbol in bar_cov:
            coverage.append(bar_cov[symbol])
        else:
            meta = out/"bars"/"_meta"/f"{symbol}.json"
            status = "empty" if meta.exists() else "not_downloaded"
            coverage.append({"symbol":symbol,"rows":0,"first_date":pd.NaT,
                             "last_date":pd.NaT,"trading_days":0,"suspended_days":0,
                             "st_days":0,"unknown_status":0,"status":status})
    cov = pd.DataFrame(coverage).sort_values("symbol")
    cov.to_csv(out/"coverage.csv",index=False)
    master.to_csv(out/"security_master.csv",index=False)
    files = {p.name:sha256(p) for p in sorted(year_dir.glob("*.parquet"))}
    status = cov.status.value_counts().to_dict()
    downloaded = int(cov.status.eq("ok").sum())
    empty = int(status.get("empty",0))
    missing = int(status.get("not_downloaded",0))
    complete = downloaded + empty == len(cov)
    write_json(out/"manifest.json",{"version":"supplement_v1","source":"BaoStock",
               "start":START,"end":END,"fields":FIELDS,"symbols":len(cov),
               "downloaded_symbols":downloaded,
               "empty_symbols":empty,
               "not_downloaded_symbols":missing,
               "complete":complete,
               "rows":int(cov.rows.sum()),"year_files":files,
               "security_master":sha256(out/"security_master.parquet"),
               "coverage":sha256(out/"coverage.csv"),
               "execution_accepted":False,"note":"Daily status/ST supplement; not an execution acceptance."})
    return {"symbols":len(cov),"downloaded":downloaded,"empty":empty,
            "not_downloaded":missing,"complete":complete,
            "rows":int(cov.rows.sum()),"years":sorted(writers)}


def load_raw(out=DEFAULT):
    """Return the canonical raw/status columns used by prepare_research."""
    out = Path(out)
    files = sorted((out/"by_year").glob("*.parquet"))
    if not files:
        files = sorted((out/"bars").glob("*.parquet"))
    frames = [pd.read_parquet(p) for p in files]
    if not frames:
        raise ValueError("Supplement has no bars")
    frame = pd.concat(frames,ignore_index=True)
    keep = ["date","symbol"]+list(RAW_MAP.values())+["turnover","is_st","daily_status"]
    return frame[keep].sort_values(["date","symbol"]).reset_index(drop=True)


def merge_metadata(listing,exits,master):
    listing, exits = dict(listing), dict(exits)
    for row in master.itertuples():
        if pd.notna(row.ipo):
            listing[row.symbol] = pd.Timestamp(row.ipo)
        if pd.notna(row.out):
            exits[row.symbol] = pd.Timestamp(row.out)
    return listing, exits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command",choices=["master","download","build"])
    ap.add_argument("--out",type=Path,default=DEFAULT)
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--start",default=START)
    ap.add_argument("--end",default=END)
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--symbols",default="")
    a = ap.parse_args()
    if a.command == "master":
        frame = fetch_master(a.out)
        print(json.dumps({"rows":len(frame),"stocks":int(frame.is_stock.sum())},ensure_ascii=False))
    elif a.command == "download":
        only = [s for s in a.symbols.split(",") if s] or None
        print(json.dumps(download(a.out,a.workers,a.start,a.end,a.limit,only),ensure_ascii=False))
    else:
        print(json.dumps(build(a.out),ensure_ascii=False))


if __name__ == "__main__":
    main()
