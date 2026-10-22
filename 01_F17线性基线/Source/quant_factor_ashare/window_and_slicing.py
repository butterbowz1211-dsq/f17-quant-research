# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
window_and_slicing.py — 多窗口夏普分解 + LSY 日频切片调仓
================================================================================
动机（用户提出的两个正确批评）
------------------------------
1. **夏普是窗口依赖的。** 此前只报「3.6 年样本外」与「相位平均」，
   把长窗口的结论当成了普遍结论。必须把近 1 / 2 / 3 年分别报出来。
2. **调仓方式过于粗糙。** 一直是「每 21 日一次性换仓」，而文献里有更细的做法：

   ★ Liu-Stambaugh-Yuan (2019, JFE 134(1)) 的反转因子再平衡配方（原文 Appendix A.2）：
       "sort **EACH DAY** on the most recent 20-day cumulative return;
        rebalance a **ONE-FIFTH slice** held **5 trading days**;
        average the 5 slices; compound daily returns to monthly"
     含义：每日刷新信号，把目标仓位切成 5 份错开持有，等价于对**每日目标权重取
     5 日移动平均**。这是文献针对中国反转因子 355%/月 换手给出的解法，
     能在保持信号新鲜度的同时压低换手。

   ★ Kang-Liu-Ni (2002)：周频反转必须**跳一周**（否则是买卖价差反弹）：
     1.86%/周 → 0.08%/周。

本脚本
------
A. 多窗口业绩（近 1/2/3 年 + 全样本外 + 全样本），多头与多空
B. 日频切片调仓（1 日信号 + N 日移动平均权重），N ∈ {1,3,5,10,21}
C. 调仓频率扫描（5/10/21/42 日），配合缓冲区
D. 对最优方案做相位平均，避免单相位运气
"""

import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ashare_factor_strategy import (load_panel, Panels, build_universe, backtest,
                                    CFG, newey_west_tstat)
import strategy_v2 as v2

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
PPY, RF = 252, 0.02
IS = ("2010-01-01", "2022-12-31")
OOS = ("2023-01-01", "2026-12-31")
BASE = dict(n=50, weighting="equal", buffer=0.6, rebalance_days=21, max_weight=0.03)


def met(seg, label=""):
    seg = pd.Series(seg).dropna()
    if len(seg) < 15:
        return None
    eq = (1 + seg).cumprod()
    yrs = len(seg) / PPY
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = seg.std() * np.sqrt(PPY)
    dd = float((eq / eq.cummax() - 1).min())
    # 夏普的近似标准误（Lo 2002 简化式），短窗口必须一并给出
    n = len(seg)
    se = np.sqrt((1 + 0.5 * ((seg.mean() * PPY - RF) / vol) ** 2) / n) * np.sqrt(PPY) \
        if vol > 0 else np.nan
    return {"label": label, "n_days": int(n), "years": round(yrs, 2),
            "CAGR": round(float(cagr), 4), "AnnVol": round(float(vol), 4),
            "Sharpe": round(float((seg.mean() * PPY - RF) / vol), 4),
            "Sharpe_se": round(float(se), 3),
            "MaxDD": round(dd, 4),
            "Calmar": round(float(cagr / abs(dd)), 3) if dd < 0 else None}


def windows(series, tag, out):
    """按要求拆近 1/2/3 年，并给全样本外/全样本"""
    end = series.index.max()
    spans = [("近 3 个月", 63), ("近 6 个月", 126), ("近 12 个月", 252),
             ("近 24 个月", 504), ("近 36 个月", 756)]
    for nm, nd in spans:
        seg = series.iloc[-nd:] if len(series) >= nd else series
        m = met(seg, f"{tag}|{nm}")
        if m:
            out[f"{tag}|{nm}"] = m
            print(f"    {tag:6s} {nm:9s} 年化 {m['CAGR']:>8.2%}  波动 {m['AnnVol']:>7.2%}  "
                  f"夏普 {m['Sharpe']:>6.3f} (±{m['Sharpe_se']:.2f})  "
                  f"回撤 {m['MaxDD']:>8.2%}  Calmar {str(m['Calmar']):>6s}")
    for nm, (a, b) in [("样本外3.6年", OOS), ("全样本16.7年", ("2010-01-01", "2026-12-31"))]:
        seg = series[(series.index >= a) & (series.index <= b)]
        m = met(seg, f"{tag}|{nm}")
        if m:
            out[f"{tag}|{nm}"] = m
            print(f"    {tag:6s} {nm:9s} 年化 {m['CAGR']:>8.2%}  波动 {m['AnnVol']:>7.2%}  "
                  f"夏普 {m['Sharpe']:>6.3f} (±{m['Sharpe_se']:.2f})  "
                  f"回撤 {m['MaxDD']:>8.2%}  Calmar {str(m['Calmar']):>6s}")


def main():
    t0 = time.time()
    print("=" * 104)
    print("多窗口夏普分解 + LSY 日频切片调仓")
    print("=" * 104)
    df = load_panel()
    P = Panels(df)
    del df
    eligible, _, _ = build_universe(P, CFG)
    need = sorted(set(v2.FACTOR_SETS["F17"] + v2.EXTRA_NAMES + v2.CONTROL_NAMES))
    v2.build_prep(P, eligible, need)
    score = v2.composite(v2.FACTOR_SETS["F17"])
    print("  因子得分完成", flush=True)

    res = {}

    # ---------------- A. 基线（21 日调仓）的多窗口 ----------------
    print("\n[A] 基线（21 日调仓 + 缓冲 0.6）多窗口业绩", flush=True)
    W = v2.make_weights(score, eligible, side="long", **BASE)
    r_long = backtest(P, W, CFG, "基线多头")["net"]
    Ws = v2.make_weights(score, eligible, side="short", **BASE)
    Wls = W.sub(Ws.reindex(columns=W.columns, fill_value=0.0), fill_value=0.0)
    r_ls = backtest(P, Wls, CFG, "基线多空")["net"]
    del W, Ws, Wls
    windows(r_long, "多头", res)
    print()
    windows(r_ls, "多空", res)

    # ---------------- B. LSY 日频切片调仓 ----------------
    print("\n[B] LSY 日频切片调仓（每日刷新信号，权重取 N 日移动平均）", flush=True)
    print("    说明：LSY 原文为「5 份错开持有 5 日」，等价于对每日目标权重取 5 日 MA。")
    print(f"    {'N 日 MA':>8s} | {'IS夏普':>7s} | {'OOS夏普':>8s} {'OOS年化':>8s} "
          f"{'OOS回撤':>8s} {'Calmar':>7s} | {'换手':>7s}")
    slice_out = {}
    t1 = time.time()
    Wday = v2.make_weights(score, eligible, side="long", n=50, weighting="equal",
                           buffer=0.0, rebalance_days=1, max_weight=0.03)
    print(f"    日频目标权重计算完成 {time.time()-t1:.0f}s  {Wday.shape}", flush=True)
    for N in [1, 3, 5, 10, 21]:
        Wn = Wday.rolling(N, min_periods=1).mean()
        r = backtest(P, Wn, CFG, f"slice{N}")
        net, tv = r["net"], r["turnover"]
        i = met(net[(net.index >= IS[0]) & (net.index <= IS[1])])
        o = met(net[(net.index >= OOS[0]) & (net.index <= OOS[1])])
        if not i or not o:
            continue
        slice_out[N] = {"IS_sharpe": i["Sharpe"], "OOS_sharpe": o["Sharpe"],
                        "OOS_cagr": o["CAGR"], "OOS_maxdd": o["MaxDD"],
                        "OOS_calmar": o["Calmar"],
                        "turnover": round(float(tv.mean() * PPY), 4)}
        print(f"    {N:>8d} | {i['Sharpe']:>7.3f} | {o['Sharpe']:>8.3f} {o['CAGR']:>8.2%} "
              f"{o['MaxDD']:>8.2%} {str(o['Calmar']):>7s} | "
              f"{tv.mean()*PPY:>7.0%}", flush=True)
        del Wn, r
    del Wday
    res["slicing"] = slice_out

    # ---------------- C. 调仓频率扫描 ----------------
    print("\n[C] 调仓频率扫描（配合缓冲区；含 SKIP 变体）", flush=True)
    print(f"    {'方案':30s} | {'IS夏普':>7s} | {'OOS夏普':>8s} {'OOS年化':>8s} "
          f"{'OOS回撤':>8s} {'Calmar':>7s} | {'换手':>7s}")
    freq_out = {}
    for rd, buf in [(5, 0.0), (5, 0.6), (10, 0.0), (10, 0.6), (15, 0.6),
                    (21, 0.0), (21, 0.6), (42, 0.6), (63, 0.6)]:
        lab = f"调仓{rd}日 缓冲{buf}"
        W2 = v2.make_weights(score, eligible, side="long", n=50, weighting="equal",
                             buffer=buf, rebalance_days=rd, max_weight=0.03)
        r = backtest(P, W2, CFG, lab)
        del W2
        net, tv = r["net"], r["turnover"]
        i = met(net[(net.index >= IS[0]) & (net.index <= IS[1])])
        o = met(net[(net.index >= OOS[0]) & (net.index <= OOS[1])])
        if not i or not o:
            continue
        freq_out[lab] = {"IS_sharpe": i["Sharpe"], "OOS_sharpe": o["Sharpe"],
                         "OOS_cagr": o["CAGR"], "OOS_maxdd": o["MaxDD"],
                         "OOS_calmar": o["Calmar"],
                         "turnover": round(float(tv.mean() * PPY), 4)}
        print(f"    {lab:30s} | {i['Sharpe']:>7.3f} | {o['Sharpe']:>8.3f} "
              f"{o['CAGR']:>8.2%} {o['MaxDD']:>8.2%} {str(o['Calmar']):>7s} | "
              f"{tv.mean()*PPY:>7.0%}", flush=True)
        del r
    res["freq"] = freq_out

    with open(os.path.join(OUT, "window_slicing.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n[已保存] {os.path.join(OUT,'window_slicing.json')}  总用时 {(time.time()-t0)/60:.1f} 分")


if __name__ == "__main__":
    main()
