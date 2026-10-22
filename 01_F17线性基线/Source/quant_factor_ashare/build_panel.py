# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
统一数据面板构建 (build_panel.py)

输入
----
data/sina_parts/*.pkl   逐股前复权行情 (新浪) —— 价量 / 流通股本 / 换手率
data/raw_daily/*.pkl    Tushare 全市场原始行情 —— 真实成交价 + 交叉校验
data/st_snapshots.pkl   月末时点 ST 快照 (baostock)
data/industry.pkl       baostock 行业分类

输出
----
data/panel.pkl          精简长表, 每股每日一行

关于"原始成交价"从哪来 —— 这一步是规模因子的地基
--------------------------------------------------
流通市值必须用**真实成交价**乘流通股本。但新浪给的是前复权价(qfq)，它被归一化到
最新价，历史行直接乘股本是错的（茅台 2015 年的 qfq 价只有真实价的几分之一）。

三个来源，优先级从高到低：
  1. Tushare 原始行情 close        —— 精确，但受 50次/分钟 限流，需 85 分钟才能下完
  2. 成交均价 VWAP = amount/volume —— 逐行精确（与收盘价差异通常 <2%），无需额外请求
  3. (已废弃) 用全历史复权因子中位数回填 —— 会把 2010 年的因子套到 2026 年，误差 30-50%

因此本脚本优先用 1，缺失时自动回落到 2。两者都拿得到时还会做一致性校验，
用于发现数据源错误（这是双通道抓取最大的附加价值）。

体积控制
--------
面板列已精简到必需项 (12 列, float32 为主)。5367 只股票 × 全历史约 1300 万行、
约 700MB。派生字段(ret / vwap / float_mktcap / days_listed)一律在
ashare_factor_strategy.py 的 Panels 里由基础字段算出，不落盘 —— 既省磁盘也省内存。
"""
import os
import glob
import argparse
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

F32 = "float32"

# 最终落盘的列（顺序即存储顺序）
KEEP_COLS = ["symbol", "date", "open", "high", "low", "close",
             "volume", "amount", "outstanding_share", "turnover",
             "raw_close", "is_st"]


def ts_code_to_sina(ts_code):
    """Tushare '000001.SZ' -> 新浪 'sz000001'"""
    try:
        num, ex = str(ts_code).split(".")
        return f"{ex.lower()}{num}"
    except Exception:
        return None


# ------------------------------------------------------------------ 1. 新浪逐股
def load_sina(limit=0):
    files = sorted(glob.glob(os.path.join(DATA, "sina_parts", "*.pkl")))
    if limit:
        files = files[:limit]
    print(f"读取新浪逐股文件: {len(files)}", flush=True)
    frames, bad = [], 0
    t0 = time.time()
    for i, f in enumerate(files, 1):
        try:
            df = pd.read_pickle(f)
            if df is None or len(df) == 0:
                bad += 1
                continue
            frames.append(df)
        except Exception:
            bad += 1
        if i % 1500 == 0:
            print(f"  {i}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    out = pd.concat(frames, ignore_index=True)
    # 统一 datetime 精度为 ns（pandas 3.0 下不同来源会推断出 s / us / ns 三种精度）
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    print(f"  合并 {len(out):,} 行 (坏文件 {bad})", flush=True)
    return out


# ------------------------------------------------------------------ 2. Tushare 原始
def load_tushare_raw():
    files = sorted(f for f in glob.glob(os.path.join(DATA, "raw_daily", "*.pkl"))
                   if not f.endswith("_failed.pkl"))
    if not files:
        print("  [提示] 暂无 Tushare 原始行情，raw_close 将全部回落到 VWAP 估计", flush=True)
        return None
    frames = [pd.read_pickle(f)[["ts_code", "trade_date", "close"]] for f in files]
    out = pd.concat(frames, ignore_index=True)
    out["symbol"] = out["ts_code"].map(ts_code_to_sina)
    out["date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d").astype("datetime64[ns]")
    out = out.dropna(subset=["symbol"]).rename(columns={"close": "ts_close"})
    out = out[["symbol", "date", "ts_close"]]
    print(f"  Tushare 原始行情 {len(out):,} 行, {out.date.nunique()} 交易日 "
          f"({out.date.min().date()} ~ {out.date.max().date()})", flush=True)
    return out


# ------------------------------------------------------------------ 3. ST / 行业
def load_st():
    p = os.path.join(DATA, "st_snapshots.pkl")
    if not os.path.exists(p):
        print("  [警告] 无 ST 快照", flush=True)
        return None
    st = pd.read_pickle(p)
    st["date"] = pd.to_datetime(st["date"]).astype("datetime64[ns]")
    return st


def load_industry():
    p = os.path.join(DATA, "industry.pkl")
    if not os.path.exists(p):
        print("  [警告] 无行业分类", flush=True)
        return None
    return pd.read_pickle(p)


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 只股票(冒烟测试)")
    args = ap.parse_args()

    print("=" * 70)
    print("[1/5] 读取新浪逐股行情")
    px = load_sina(args.limit)

    print("[2/5] 读取 Tushare 全市场原始行情")
    ts = load_tushare_raw()

    print("[3/5] 派生与合并")
    px = px.sort_values(["symbol", "date"]).reset_index(drop=True)
    n0 = len(px)

    # ---- 成交均价 VWAP：逐行精确的原始价估计 ----
    # amount 单位:元, volume 单位:股 -> vwap 单位:元/股, 与真实成交价同量纲
    vol = pd.to_numeric(px["volume"], errors="coerce")
    amt = pd.to_numeric(px["amount"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = (amt / vol.replace(0, np.nan)).astype(F32)
    # 明显异常的 VWAP（数据错误）置空
    vwap = vwap.where(np.isfinite(vwap) & (vwap > 0))

    if ts is not None:
        px = px.merge(ts, on=["symbol", "date"], how="left")
        matched = px["ts_close"].notna().mean()
        print(f"  与 Tushare 匹配率: {matched*100:.2f}%", flush=True)
        # ---- 双通道一致性校验 ----
        both = px["ts_close"].notna() & vwap.notna() & (vwap > 0)
        if both.sum() > 1000:
            rel = ((px.loc[both, "ts_close"] - vwap[both]).abs() / vwap[both])
            print(f"  【数据校验】Tushare收盘 vs 新浪VWAP 相对偏差: "
                  f"中位数 {rel.median()*100:.3f}%  "
                  f"95分位 {rel.quantile(0.95)*100:.2f}%  "
                  f"超5%的占比 {(rel>0.05).mean()*100:.2f}%", flush=True)
        raw = px["ts_close"].where(px["ts_close"].notna(), vwap)
        src = np.where(px["ts_close"].notna(), "tushare", "vwap")
    else:
        raw = vwap
        src = np.full(len(px), "vwap", dtype=object)
    px["raw_close"] = raw.astype(F32)

    # ---- 复权因子 = 前复权价 / 真实价（≤1, 末端≈1）----
    adj = (px["close"] / px["raw_close"].replace(0, np.nan)).astype(F32)
    n_bad_adj = int((adj.notna() & ((adj > 1.5) | (adj <= 0))).sum())
    print(f"  raw_close 来源: Tushare {int((src=='tushare').sum()):,} 行 / "
          f"VWAP {int((src=='vwap').sum()):,} 行", flush=True)
    print(f"  复权因子异常值(>1.5 或 ≤0): {n_bad_adj:,} 行", flush=True)
    del adj, vwap, vol, amt, raw, src

    print("[4/5] ST 快照（月末 → 日频前向填充）")
    st = load_st()
    if st is not None:
        st = st[st["symbol"].astype(str).str.match(r"^(sh6|sz0|sz3)")]
        st = st[["date", "symbol", "is_st"]].drop_duplicates(["symbol", "date"])
        st["symbol"] = st["symbol"].astype(str)
        left = (px[["symbol", "date"]].reset_index()
                  .assign(symbol=px["symbol"].astype(str))
                  .sort_values("date"))
        right = st.sort_values("date")
        merged = pd.merge_asof(left, right, on="date", by="symbol",
                               direction="backward")
        merged = merged.sort_values("index")
        px["is_st"] = merged["is_st"].fillna(False).to_numpy()
        cov = merged["is_st"].notna().mean()
        print(f"  ST 覆盖率 {cov*100:.1f}%   ST 占比 {px['is_st'].mean()*100:.2f}%",
              flush=True)
        del left, right, merged, st
    else:
        px["is_st"] = False

    print("[5/5] 落盘")
    out = px[KEEP_COLS].copy()
    # 数值统一 float32（因子排序用不到 float64 精度，内存减半）
    for c in ["open", "high", "low", "close", "volume", "amount",
              "outstanding_share", "turnover", "raw_close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(F32)
    out["symbol"] = out["symbol"].astype("category")
    out["is_st"] = out["is_st"].astype(bool)

    path = os.path.join(DATA, "panel.pkl")
    out.to_pickle(path)
    mb = os.path.getsize(path) / 1e6

    # 行业单独存（每股一条，不随日期变化）
    ind = load_industry()
    if ind is not None:
        ind[["symbol", "industry"]].to_pickle(os.path.join(DATA, "industry_clean.pkl"))

    print("=" * 70)
    print(f"面板: {len(out):,} 行 × {out.shape[1]} 列  ->  {path}  ({mb:.0f} MB)")
    print(f"股票 {out.symbol.nunique():,}   交易日 {out.date.nunique():,}   "
          f"区间 {out.date.min().date()} ~ {out.date.max().date()}")
    print(f"内存占用(float32): 约 {out.memory_usage(deep=True).sum()/1e6:.0f} MB "
          f"(原始 {n0:,} 行)")
    print("\n样例:")
    print(out.tail(3).to_string())


if __name__ == "__main__":
    main()
