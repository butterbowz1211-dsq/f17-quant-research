# Research project author: Gilmour Jia
"""Build a compact daily-status supplement from the Baidu stock_full zip.

Source: workspace/source_workspace/_baidu_extract/stock_full.zip
Output: nn_v2/derived/baidu_supplement_v1/{by_year/*.parquet,manifest.json,...}

Parallel per-symbol extraction. This is a data-quality supplement, not an
execution acceptance. It supplies daily ST, exact limit prices, float/total/
free shares and turnover for 2010-2025. Prices stay in source_union_v1.
"""
import runtime
import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = runtime.ROOT
ZIP = Path("workspace/source_workspace/_baidu_extract/stock_full.zip")
OUT = ROOT / "derived" / "baidu_supplement_v1"
START = pd.Timestamp("2010-01-01")
END = pd.Timestamp("2025-12-31")

COLS = {
    "股票代码": "code",
    "股票名称": "name",
    "交易日": "date",
    "收盘价": "close",
    "前收盘价": "pre_close",
    "成交量（手）": "volume_lots",
    "成交额（千元）": "amount_k",
    "换手率（%）": "turnover_pct",
    "总股本（万股）": "total_shares_wan",
    "流通股本（万股）": "float_shares_wan",
    "自由流通股本（万）": "free_float_wan",
    "复权因子": "adj_factor",
    "当日涨停价": "limit_up",
    "当日跌停价": "limit_down",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
}

SCHEMA = pa.schema([
    ("date", pa.timestamp("ns")),
    ("symbol", pa.string()),
    ("name", pa.string()),
    ("is_st", pa.bool_()),
    ("daily_status", pa.string()),
    ("float_shares", pa.float64()),
    ("total_shares", pa.float64()),
    ("free_float", pa.float64()),
    ("turnover", pa.float64()),
    ("limit_up", pa.float64()),
    ("limit_down", pa.float64()),
    ("close", pa.float64()),
    ("pre_close", pa.float64()),
    ("adj_factor", pa.float64()),
])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def symbol_of(code):
    code = str(code).strip().upper()
    if "." not in code:
        return None
    num, ex = code.split(".", 1)
    if ex not in ("SH", "SZ") or not re.fullmatch(r"\d{6}", num):
        return None
    return ("sh" if ex == "SH" else "sz") + num


def transform(df):
    df = df.rename(columns=COLS)
    if not {"code", "name", "date"}.issubset(df.columns):
        raise ValueError("missing code/name/date columns")
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    out["symbol"] = df["code"].map(symbol_of)
    out["name"] = df["name"].astype("string")
    name = out["name"].fillna("")
    out["is_st"] = name.str.contains(r"\*?ST", case=False, regex=True, na=False)
    vol = pd.to_numeric(df.get("volume_lots"), errors="coerce")
    out["daily_status"] = np.where(vol.fillna(0) > 0, "trading", "suspended")
    for src, dst, scale in (
        ("float_shares_wan", "float_shares", 1e4),
        ("total_shares_wan", "total_shares", 1e4),
        ("free_float_wan", "free_float", 1e4),
        ("limit_up", "limit_up", 1.0),
        ("limit_down", "limit_down", 1.0),
        ("close", "close", 1.0),
        ("pre_close", "pre_close", 1.0),
        ("adj_factor", "adj_factor", 1.0),
    ):
        out[dst] = pd.to_numeric(df.get(src), errors="coerce") * scale
    out["turnover"] = pd.to_numeric(df.get("turnover_pct"), errors="coerce") / 100.0
    out = out[out["symbol"].notna() & out["date"].between(START, END)].copy()
    out = out[out["symbol"].str.match(r"^(sh60|sh688|sz00|sz30)\d+$", na=False)]
    out = out.drop_duplicates(["date", "symbol"]).sort_values(["date", "symbol"])
    for c in ("float_shares", "total_shares", "free_float", "turnover",
              "limit_up", "limit_down", "close", "pre_close", "adj_factor"):
        out[c] = out[c].astype("float64")
    out["is_st"] = out["is_st"].astype("bool")
    out["daily_status"] = out["daily_status"].astype("string")
    out["name"] = out["name"].astype("string")
    out["symbol"] = out["symbol"].astype("string")
    return out


def csv_members(z):
    for name in z.namelist():
        base = name.rsplit("/", 1)[-1]
        if "__MACOSX" in name or base.startswith("._") or not base.lower().endswith(".csv"):
            continue
        if not re.fullmatch(r"\d{6}\.(SH|SZ)\.csv", base):
            continue
        yield name, base[:-4]


_Z = None
_TMP = None


def _init(zip_path, tmp):
    global _Z, _TMP
    _Z = zipfile.ZipFile(zip_path)
    _TMP = Path(tmp)


def _work(task):
    member, code = task
    try:
        with _Z.open(member) as f:
            raw = pd.read_csv(f, encoding="utf-8-sig")
        part = transform(raw)
        if part.empty:
            return code, 0, None, None, None
        part.to_parquet(_TMP / f"{code}.parquet", index=False)
        return (code, len(part), str(part.date.min().date()),
                str(part.date.max().date()), str(part.name.iloc[-1]))
    except Exception as exc:
        return code, -1, repr(exc), None, None


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    tmp = OUT / "_parts"
    tmp.mkdir()
    z = zipfile.ZipFile(ZIP)
    members = list(csv_members(z))
    z.close()
    print(f"[baidu] {len(members)} symbols from {ZIP.name}", flush=True)
    rows = 0
    errors = []
    cov = {}
    with mp.Pool(8, initializer=_init, initargs=(str(ZIP), str(tmp))) as pool:
        for i, res in enumerate(pool.imap_unordered(_work, members, chunksize=1), 1):
            code, n, first, last, name = res
            if n < 0:
                errors.append({"code": code, "error": first})
            elif n > 0:
                rows += n
                s = symbol_of(code)
                cov[s] = {"symbol": s, "name": name, "first_date": first,
                          "last_date": last, "rows": n}
            if i % 250 == 0:
                print(f"[baidu] {i}/{len(members)} rows={rows:,} errors={len(errors)}", flush=True)
    by_year = OUT / "by_year"
    by_year.mkdir()
    writers = {}
    for p in sorted(tmp.glob("*.parquet")):
        part = pd.read_parquet(p)
        part["year"] = part.date.dt.year
        for year, g in part.groupby("year", sort=True):
            g = g.drop(columns=["year"])
            table = pa.Table.from_pandas(g, schema=SCHEMA, preserve_index=False)
            if year not in writers:
                writers[year] = pq.ParquetWriter(by_year / f"{int(year)}.parquet", SCHEMA)
            writers[year].write_table(table)
    for w in writers.values():
        w.close()
    shutil.rmtree(tmp)
    frame = pd.DataFrame(cov.values()).sort_values("symbol")
    frame.to_csv(OUT / "coverage.csv", index=False)
    frame.to_parquet(OUT / "security_master.parquet", index=False)
    files = {p.name: sha256(p) for p in sorted(by_year.glob("*.parquet"))}
    manifest = {
        "version": "baidu_supplement_v1",
        "source": "Baidu stock_full.zip",
        "source_sha256": sha256(ZIP),
        "start": str(START.date()),
        "end": str(END.date()),
        "fields": [f.name for f in SCHEMA],
        "symbols": int(len(frame)),
        "rows": int(rows),
        "year_files": files,
        "complete": len(errors) == 0,
        "errors": errors[:50],
        "note": "Daily ST/limit/float/turnover supplement; not an execution acceptance.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("symbols", "rows", "complete")}, ensure_ascii=False), flush=True)


def load_status(path=OUT):
    path = Path(path)
    files = sorted((path / "by_year").glob("*.parquet"))
    if not files:
        raise ValueError(f"No Baidu supplement files in {path}")
    frames = [pd.read_parquet(p) for p in files]
    frame = pd.concat(frames, ignore_index=True)
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


if __name__ == "__main__":
    build()
