# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
benchmark_windows.py — 基准的多窗口夏普（回答「基金夏普 2.6」的含义）
================================================================================
问题
----
用户指出「市面上很多基金近一年夏普能达到 2.6」。
但夏普是**窗口依赖**的，且短窗口的估计标准误极大。关键追问是：

    同期**市场本身**的夏普是多少？

若 2024-09 之后 A 股出现大幅上涨，市场自身的 12 个月夏普就可能达到 2~3，
那么「基金夏普 2.6」可能只是 beta，而不是选股能力。

本脚本给出：
1. 各基准（等权全市场 / 流通市值加权 / 沪深300 / 中证500 / 中证1000）
   在 近3/6/12/24/36 个月 的夏普
2. 夏普的近似标准误（Lo 2002）：se ≈ sqrt((1 + SR²/2)/n)，n 为年数
3. 与策略对比 —— 关键是**超额夏普**而非绝对夏普
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
PPY, RF = 252, 0.02

WINDOWS = [("近 3 个月", 63), ("近 6 个月", 126), ("近 12 个月", 252),
           ("近 24 个月", 504), ("近 36 个月", 756)]


def sh(seg):
    seg = pd.Series(seg).dropna()
    if len(seg) < 15:
        return None
    vol = seg.std() * np.sqrt(PPY)
    sr = (seg.mean() * PPY - RF) / vol
    eq = (1 + seg).cumprod()
    yrs = len(seg) / PPY
    return {"CAGR": float(eq.iloc[-1] ** (1 / yrs) - 1), "AnnVol": float(vol),
            "Sharpe": float(sr),
            "se": float(np.sqrt((1 + 0.5 * sr ** 2) / yrs)),
            "MaxDD": float((eq / eq.cummax() - 1).min())}


def main():
    md = pd.read_pickle(os.path.join(DATA, "market_daily.pkl"))
    series = {"流通市值加权市场": md["mkt_vw"].dropna(),
              "等权全市场": md["mkt_ew"].dropna()}
    try:
        import akshare as ak
        for code, nm in [("sh000300", "沪深300"), ("sh000905", "中证500"),
                         ("sh000852", "中证1000")]:
            try:
                d = ak.stock_zh_index_daily(symbol=code)
                d["date"] = pd.to_datetime(d["date"])
                s = pd.to_numeric(d["close"], errors="coerce")
                r = s.pct_change().dropna()
                r.index = pd.to_datetime(d["date"])[1:]
                series[nm] = r
            except Exception as e:
                print(f"  {nm}: 获取失败 {str(e)[:60]}")
    except Exception:
        pass

    print("=" * 104)
    print("基准的多窗口夏普（含 Lo 2002 近似标准误）")
    print("=" * 104)
    out = {}
    print(f"  {'基准':18s} {'窗口':10s} {'年化':>9s} {'波动':>8s} {'夏普':>7s} "
          f"{'±se':>6s} {'回撤':>9s}")
    print("  " + "-" * 90)
    for nm, r in series.items():
        out[nm] = {}
        for wn, nd in WINDOWS:
            seg = r.iloc[-nd:] if len(r) >= nd else r
            m = sh(seg)
            if not m:
                continue
            out[nm][wn] = {k: round(v, 4) for k, v in m.items()}
            print(f"  {nm:18s} {wn:10s} {m['CAGR']:>9.2%} {m['AnnVol']:>8.2%} "
                  f"{m['Sharpe']:>7.3f} {m['se']:>6.2f} {m['MaxDD']:>9.2%}")
        print()

    with open(os.path.join(OUT, "benchmark_windows.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[已保存] {os.path.join(OUT,'benchmark_windows.json')}")


if __name__ == "__main__":
    main()
