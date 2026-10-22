# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
phase_multiwindow.py — 多窗口结果的相位稳健性 + 滚动夏普
================================================================================
为什么必须做
------------
上一轮发现「近 24 个月多头夏普 1.72、三项达标」，但那只是**相位 0 的单点值**。
本轮之前已经吃过两次亏：
  · 单相位 1.017 的缓冲区改进 → 相位平均后只剩 0.860
  · 单相位 +0.125 的 α 加权改进 → 相位 0 上其实是 −0.011
因此**任何多窗口结论都必须给出相位分布**，否则可能又是一个相位运气。

本脚本
------
1. 对 11 个调仓相位，分别计算 近3/6/12/24/36 个月的夏普、回撤、Calmar
   → 给出每个窗口的**相位均值、标准差、区间**，以及「三项达标」的相位占比
2. 滚动 252 日夏普的时间序列（看是否依赖特定时段）
3. 按窗口给出 DSR（多重检验校正后的显著性）
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

from ashare_factor_strategy import (load_panel, Panels, build_universe, backtest,
                                    CFG, probabilistic_sharpe, deflated_sharpe)
import strategy_v2 as v2

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
PPY, RF = 252, 0.02
REC = dict(n=50, weighting="equal", buffer=0.6, rebalance_days=21, max_weight=0.03)
PHASES = list(range(0, 21, 2))          # 11 个相位
WINDOWS = [("近3个月", 63), ("近6个月", 126), ("近12个月", 252),
           ("近24个月", 504), ("近36个月", 756)]


def one(seg):
    seg = pd.Series(seg).dropna()
    if len(seg) < 15:
        return None
    eq = (1 + seg).cumprod()
    yrs = len(seg) / PPY
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = seg.std() * np.sqrt(PPY)
    sr = (seg.mean() * PPY - RF) / vol
    dd = float((eq / eq.cummax() - 1).min())
    return {"Sharpe": float(sr), "CAGR": float(cagr), "AnnVol": float(vol),
            "MaxDD": dd, "Calmar": float(cagr / abs(dd)) if dd < 0 else None}


def main():
    df = load_panel()
    P = Panels(df)
    del df
    eligible, _, _ = build_universe(P, CFG)
    need = sorted(set(v2.FACTOR_SETS["F17"] + v2.EXTRA_NAMES + v2.CONTROL_NAMES))
    v2.build_prep(P, eligible, need)
    score = v2.composite(v2.FACTOR_SETS["F17"])
    print("因子得分完成", flush=True)

    print(f"\n[1] 相位扫描（{len(PHASES)} 个相位）", flush=True)
    nets = {}
    for off in PHASES:
        s2 = score.iloc[off:] if off else score
        el2 = eligible.reindex(s2.index)
        Wl = v2.make_weights(s2, el2, side="long", **REC)
        rl = backtest(P, Wl, CFG, f"L{off}")["net"]
        Ws = v2.make_weights(s2, el2, side="short", **REC)
        Wls = Wl.sub(Ws.reindex(columns=Wl.columns, fill_value=0.0), fill_value=0.0)
        rls = backtest(P, Wls, CFG, f"LS{off}")["net"]
        del Wl, Ws, Wls
        nets[("多头", off)] = rl
        nets[("多空", off)] = rls
        print(f"    相位 {off:>2d} 完成", flush=True)

    print("\n" + "=" * 116)
    print("多窗口 × 相位稳健性")
    print("=" * 116)
    out = {}
    for mode in ["多头", "多空"]:
        print(f"\n  --- {mode} ---")
        print(f"  {'窗口':>10s} {'夏普均值':>9s} {'标准差':>8s} {'min':>7s} {'max':>7s} "
              f"{'回撤均值':>9s} {'回撤最差':>9s} {'Calmar均值':>11s} "
              f"{'三项达标相位占比':>16s}")
        win_out = {}
        for wn, nd in WINDOWS:
            sh, dd, ca, ok = [], [], [], 0
            n = 0
            for off in PHASES:
                ser = nets[(mode, off)]
                seg = ser.iloc[-nd:] if len(ser) >= nd else ser
                m = one(seg)
                if not m:
                    continue
                n += 1
                sh.append(m["Sharpe"]); dd.append(m["MaxDD"])
                if m["Calmar"] is not None:
                    ca.append(m["Calmar"])
                s_ok = m["Sharpe"] > 1.2
                d_ok = m["MaxDD"] > -0.15
                c_ok = (m["Calmar"] or -9) > 0.9
                ok += int(s_ok and d_ok and c_ok)
            if not sh:
                continue
            win_out[wn] = {
                "sharpe_mean": round(float(np.mean(sh)), 4),
                "sharpe_sd": round(float(np.std(sh)), 4),
                "sharpe_min": round(float(np.min(sh)), 4),
                "sharpe_max": round(float(np.max(sh)), 4),
                "maxdd_mean": round(float(np.mean(dd)), 4),
                "maxdd_worst": round(float(np.min(dd)), 4),
                "calmar_mean": round(float(np.mean(ca)), 3) if ca else None,
                "n_phases": n,
                "pct_all3_ok": round(ok / n, 3)}
            w = win_out[wn]
            print(f"  {wn:>10s} {w['sharpe_mean']:>9.3f} {w['sharpe_sd']:>8.3f} "
                  f"{w['sharpe_min']:>7.3f} {w['sharpe_max']:>7.3f} "
                  f"{w['maxdd_mean']:>9.2%} {w['maxdd_worst']:>9.2%} "
                  f"{str(w['calmar_mean']):>11s} {w['pct_all3_ok']:>15.0%}")
        out[mode] = win_out

    # ---------------- 滚动 252 日夏普 ----------------
    print("\n" + "=" * 116)
    print("滚动 252 日夏普（相位 0）")
    print("=" * 116)
    roll = {}
    for mode in ["多头", "多空"]:
        r = nets[(mode, 0)]
        mu = r.rolling(252).mean() * PPY - RF
        sd = r.rolling(252).std() * np.sqrt(PPY)
        rs = (mu / sd).dropna()
        roll[mode] = rs
        q = rs.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"  {mode:6s} n={len(rs)}  中位 {q.iloc[2]:>6.2f}  "
              f"P25 {q.iloc[1]:>6.2f}  P75 {q.iloc[3]:>6.2f}  "
              f"P5 {q.iloc[0]:>6.2f}  P95 {q.iloc[4]:>6.2f}  "
              f"<-0 占比 {float((rs < 0).mean()):.1%}")
        # 按年给出滚动夏普均值
        byy = rs.groupby(rs.index.year).mean()
        print("         年度滚动夏普均值：" +
              "  ".join(f"{y}:{v:+.2f}" for y, v in byy.items()))

    # ---------------- DSR（逐窗口） ----------------
    print("\n" + "=" * 116)
    print("逐窗口 DSR（试验分布 n=334, 方差 0.0682；N 取 50 与 334）")
    print("=" * 116)
    var = 0.0682
    dsr_out = {}
    print(f"  {'组合':6s}{'窗口':>10s}{'夏普':>9s}{'PSR':>8s}{'DSR(N=50)':>11s}"
          f"{'DSR(N=334)':>12s}")
    for mode in ["多头", "多空"]:
        ser = nets[(mode, 0)]
        for wn, nd in WINDOWS:
            seg = ser.iloc[-nd:] if len(ser) >= nd else ser
            m = one(seg)
            if not m:
                continue
            sr = m["Sharpe"]
            psr = probabilistic_sharpe(sr / np.sqrt(PPY), len(seg),
                                       float(seg.skew()), float(seg.kurtosis() + 3.0))
            d50 = deflated_sharpe(seg, n_trials=50, sr_variance=var)
            d334 = deflated_sharpe(seg, n_trials=334, sr_variance=var)
            dsr_out[f"{mode}|{wn}"] = {"Sharpe": round(sr, 4), "PSR": round(float(psr), 4),
                                       "DSR_N50": round(float(d50["DSR"]), 4),
                                       "DSR_N334": round(float(d334["DSR"]), 4),
                                       "SR0_N50": round(float(d50["SR0"]), 4)}
            print(f"  {mode:6s}{wn:>10s}{sr:>9.2f}{psr:>8.4f}"
                  f"{d50['DSR']:>11.4f}{d334['DSR']:>12.4f}")

    with open(os.path.join(OUT, "phase_multiwindow.json"), "w", encoding="utf-8") as fh:
        json.dump({"phase_windows": out, "dsr": dsr_out,
                   "roll_sharpe_stats": {m: {"median": round(float(s.median()), 3),
                                             "p25": round(float(s.quantile(0.25)), 3),
                                             "p75": round(float(s.quantile(0.75)), 3),
                                             "pct_negative": round(float((s < 0).mean()), 4)}
                                         for m, s in roll.items()}},
                  fh, ensure_ascii=False, indent=2, default=str)
    # 保存滚动夏普供画图
    pd.DataFrame(roll).to_pickle(os.path.join(OUT, "rolling_sharpe.pkl"))
    print(f"\n[已保存] {os.path.join(OUT,'phase_multiwindow.json')}")


if __name__ == "__main__":
    main()
