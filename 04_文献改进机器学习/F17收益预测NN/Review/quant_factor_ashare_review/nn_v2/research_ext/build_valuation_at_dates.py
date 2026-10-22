# Research project author: Gilmour Jia
"""Extract point-in-time valuation fields on the frozen formation dates.

Reads the Baidu stock_full zip, keeps only rows whose date is a formation date,
and writes a compact parquet. This is a research extension, not an execution
acceptance, and it does not modify the frozen v4 package.
"""
import io
import json
import multiprocessing as mp
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime  # noqa: E402  bootstraps pyarrow/DLL paths

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

ROOT = runtime.ROOT
ZIP = Path("workspace/source_workspace/_baidu_extract/stock_full.zip")
SAMPLES = ROOT / "derived/research_training_v4/samples.npz"
OUT = ROOT / "derived/valuation_at_formation_v1"

COLS = {
    "股票代码": "code",
    "交易日": "date",
    "市盈率（亏损为空）": "pe",
    "市盈率（TTM，亏损为空）": "pe_ttm",
    "市净率": "pb",
    "市销率": "ps",
    "市销率（TTM）": "ps_ttm",
    "股息率（%）": "div_yield",
    "股息率（%，TTM）": "div_yield_ttm",
    "总股本（万股）": "total_shares_wan",
    "流通股本（万股）": "float_shares_wan",
    "自由流通股本（万）": "free_float_wan",
    "总市值（万元）": "total_cap_wan",
    "流通市值（万元）": "float_cap_wan",
}

SCHEMA = pa.schema([
    ("date", pa.timestamp("ns")),
    ("symbol", pa.string()),
    ("pe", pa.float64()),
    ("pe_ttm", pa.float64()),
    ("pb", pa.float64()),
    ("ps", pa.float64()),
    ("ps_ttm", pa.float64()),
    ("div_yield", pa.float64()),
    ("div_yield_ttm", pa.float64()),
    ("total_shares", pa.float64()),
    ("float_shares", pa.float64()),
    ("free_float", pa.float64()),
    ("total_cap", pa.float64()),
    ("float_cap", pa.float64()),
])

_DATES = None
_Z = None
_TMP = None


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
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    out["symbol"] = df["code"].map(symbol_of)
    for src, dst, scale in (
        ("pe", "pe", 1.0), ("pe_ttm", "pe_ttm", 1.0),
        ("pb", "pb", 1.0), ("ps", "ps", 1.0), ("ps_ttm", "ps_ttm", 1.0),
        ("div_yield", "div_yield", 0.01), ("div_yield_ttm", "div_yield_ttm", 0.01),
        ("total_shares_wan", "total_shares", 1e4),
        ("float_shares_wan", "float_shares", 1e4),
        ("free_float_wan", "free_float", 1e4),
        ("total_cap_wan", "total_cap", 1e4),
        ("float_cap_wan", "float_cap", 1e4),
    ):
        out[dst] = pd.to_numeric(df.get(src), errors="coerce") * scale
    out = out[out["symbol"].notna() & out["date"].isin(_DATES)].copy()
    out = out[out["symbol"].str.match(r"^(sh60|sh688|sz00|sz30)\d+$", na=False)]
    out = out.drop_duplicates(["date", "symbol"]).sort_values(["date", "symbol"])
    for c in SCHEMA.names[2:]:
        out[c] = out[c].astype("float64")
    out["symbol"] = out["symbol"].astype("string")
    return out


def members(z):
    for name in z.namelist():
        base = name.rsplit("/", 1)[-1]
        if "__MACOSX" in name or base.startswith("._") or not base.lower().endswith(".csv"):
            continue
        if re.fullmatch(r"\d{6}\.(SH|SZ)\.csv", base):
            yield name, base[:-4]


def init(zip_path, dates, tmp):
    global _Z, _DATES, _TMP
    _Z = zipfile.ZipFile(zip_path)
    _DATES = dates
    _TMP = Path(tmp)


def work(task):
    member, code = task
    try:
        with _Z.open(member) as f:
            raw = pd.read_csv(f, encoding="utf-8-sig")
        part = transform(raw)
        if part.empty:
            return code, 0, None
        part.to_parquet(_TMP / f"{code}.parquet", index=False)
        return code, len(part), None
    except Exception as exc:
        return code, -1, repr(exc)


def main():
    if OUT.exists():
        raise ValueError(f"New output directory required: {OUT}")
    with np.load(SAMPLES, allow_pickle=False) as z:
        dates = pd.to_datetime(z["formation_time"]).normalize().unique()
    date_set = set(pd.DatetimeIndex(dates))
    print(f"[valuation] {len(date_set)} formation dates", flush=True)
    OUT.mkdir(parents=True)
    tmp = OUT / "_parts"
    tmp.mkdir()
    with zipfile.ZipFile(ZIP) as z:
        tasks = list(members(z))
    print(f"[valuation] {len(tasks)} symbol files", flush=True)
    errors, rows, kept = [], 0, 0
    with mp.Pool(8, initializer=init, initargs=(str(ZIP), date_set, str(tmp))) as pool:
        for i, (code, n, err) in enumerate(pool.imap_unordered(work, tasks, chunksize=1), 1):
            if n < 0:
                errors.append({"code": code, "error": err})
            else:
                kept += int(n > 0)
                rows += n
            if i % 500 == 0:
                print(f"[valuation] {i}/{len(tasks)} rows={rows:,} errors={len(errors)}", flush=True)
    files = sorted(tmp.glob("*.parquet"))
    tables = [pq.read_table(p) for p in files]
    table = pa.concat_tables(tables).cast(SCHEMA)
    OUT.mkdir(exist_ok=True)
    pq.write_table(table, OUT / "valuation.parquet", compression="zstd")
    frame = table.to_pandas()
    summary = {
        "version": "valuation_at_formation_v1",
        "source": str(ZIP),
        "source_zip_bytes": ZIP.stat().st_size,
        "formation_dates": int(len(date_set)),
        "symbols_kept": int(kept),
        "rows": int(len(frame)),
        "first": str(frame.date.min().date()),
        "last": str(frame.date.max().date()),
        "coverage_by_field": {c: float(frame[c].notna().mean()) for c in SCHEMA.names[2:]},
        "errors": errors[:20],
        "complete": len(errors) == 0,
        "note": "Valuation fields on formation dates only; no execution acceptance.",
    }
    (OUT / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    import shutil
    shutil.rmtree(tmp)
    print(json.dumps({k: summary[k] for k in ("symbols_kept", "rows", "complete")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
