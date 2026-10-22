# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
final_multiwindow.py — 最终多窗口业绩（回答用户的核心问题）
================================================================================
用户提出的正确批评
------------------
「为何是不可能达到的数字？市面上很多基金近一年夏普能到 2.6。
  你可以把最终报告窗口分别定格在近一年、近两年和近三年分别讨论。」

我此前的错误：用 3.6 年样本外窗口的结论否定了 1 年窗口的目标，
并用被动指数去标定主动策略。夏普是窗口依赖的，必须分窗口报告。

同时补上第三个有文献依据的合成方法（Fama-MacBeth 多变量权重）的完整结果。

本脚本产出
----------
1. 多窗口业绩表：近 3/6/12/24/36 个月 + 样本外 + 全样本
   对象：等权多头/多空、FM 加权多头/多空、等权全市场、沪深300
2. 每个夏普附 **Lo (2002) 近似标准误** —— 短窗口的夏普必须以区间形式呈现
3. 超额夏普（策略 − 基准）—— 这比绝对夏普更有信息量
4. 按窗口逐一对目标（夏普>1.2 / 回撤<15% / Calmar>0.9）核定
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

from ashare_factor_strategy import load_panel, Panels, build_universe, backtest, CFG
import strategy_v2 as v2

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
PPY, RF = 252, 0.02
REC = dict(n=50, weighting="equal", buffer=0.6, rebalance_days=21, max_weight=0.03)
WINDOWS = [("近3个月", 63), ("近6个月", 126), ("近12个月", 252),
           ("近24个月", 504), ("近36个月", 756)]


def metrics(seg):
    seg = pd.Series(seg).dropna()
    if len(seg) < 15:
        return None
    eq = (1 + seg).cumprod()
    yrs = len(seg) / PPY
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = seg.std() * np.sqrt(PPY)
    sr = (seg.mean() * PPY - RF) / vol
    dd = float((eq / eq.cummax() - 1).min())
    se = float(np.sqrt((1 + 0.5 * sr ** 2) / yrs))
    return {"years": round(yrs, 2), "CAGR": round(float(cagr), 4),
            "AnnVol": round(float(vol), 4), "Sharpe": round(float(sr), 4),
            "Sharpe_se": round(se, 3), "Sharpe_lo": round(float(sr - 1.96 * se), 3),
            "Sharpe_hi": round(float(sr + 1.96 * se), 3),
            "MaxDD": round(dd, 4),
            "Calmar": round(float(cagr / abs(dd)), 3) if dd < 0 else None}


def main():
    df = load_panel()
    P = Panels(df)
    del df
    eligible, _, _ = build_universe(P, CFG)
    names = v2.FACTOR_SETS["F17"]
    need = sorted(set(names + v2.EXTRA_NAMES + v2.CONTROL_NAMES))
    v2.build_prep(P, eligible, need)

    # FM 权重（样本内估计，读上一步结果）
    fp = os.path.join(OUT, "fm_weights.json")
    if os.path.exists(fp):
        with open(fp, encoding="utf-8") as fh:
            wj = json.load(fh)["weights"]
        key = "FM岭λ=0.2"
        w_fm = np.array([max(wj[key][n], 0.0) for n in names])
        w_fm = w_fm / w_fm.sum() if w_fm.sum() > 0 else np.ones(len(names)) / len(names)
    else:
        w_fm = np.ones(len(names)) / len(names)
    print("FM 权重（截断负值后归一化，仅显示前 8）：")
    for n, w in sorted(zip(names, w_fm), key=lambda x: -x[1])[:8]:
        print(f"    {n:11s} {w:.4f}")

    SC = {"等权": v2.composite(names),
          "FM加权": v2.composite(names, weights=w_fm)}

    print("\n构造组合（相位 0）...", flush=True)
    series = {}
    for sn, sc in SC.items():
        Wl = v2.make_weights(sc, eligible, side="long", **REC)
        rl = backtest(P, Wl, CFG, f"{sn}_L")["net"]
        Ws = v2.make_weights(sc, eligible, side="short", **REC)
        Wls = Wl.sub(Ws.reindex(columns=Wl.columns, fill_value=0.0), fill_value=0.0)
        rls = backtest(P, Wls, CFG, f"{sn}_LS")["net"]
        del Wl, Ws, Wls
        series[f"{sn}·多头"] = rl
        series[f"{sn}·多空"] = rls
        print(f"    {sn} 完成", flush=True)

    Wb = v2.universe_weights(eligible)
    series["等权全市场"] = backtest(P, Wb, CFG, "基准")["net"]
    try:
        import akshare as ak
        d = ak.stock_zh_index_daily(symbol="sh000300")
        d["date"] = pd.to_datetime(d["date"])
        s = pd.to_numeric(d["close"], errors="coerce")
        r = s.pct_change().dropna()
        r.index = pd.to_datetime(d["date"])[1:]
        series["沪深300"] = r
    except Exception as e:
        print(f"  沪深300 获取失败: {str(e)[:60]}")

    # ---------------- 主表 ----------------
    print("\n" + "=" * 118)
    print("多窗口业绩（相位 0；夏普后括号内为 Lo 2002 近似 95% 区间）")
    print("=" * 118)
    out = {}
    hdr = f"  {'组合':16s}" + "".join(f"{w:>17s}" for w, _ in WINDOWS) + \
        f"{'样本外3.6年':>17s}{'全样本':>17s}"
    print(hdr)
    print("  " + "-" * 114)
    for nm, ser in series.items():
        row = {}
        cells = []
        for wn, nd in WINDOWS:
            seg = ser.iloc[-nd:] if len(ser) >= nd else ser
            m = metrics(seg)
            row[wn] = m
            cells.append(f"{m['Sharpe']:>7.2f}" if m else f"{'—':>7s}")
            if m:
                cells[-1] += f"[{m['Sharpe_lo']:>5.2f},{m['Sharpe_hi']:>5.2f}]"[:10]
        for lab, cond in [("样本外3.6年", (ser.index >= "2023-01-01")),
                          ("全样本", (ser.index >= "2010-01-01"))]:
            seg = ser[cond]
            m = metrics(seg)
            row[lab] = m
            cells.append(f"{m['Sharpe']:>7.2f}[{m['Sharpe_lo']:>5.2f},{m['Sharpe_hi']:>5.2f}]" if m else f"{'—':>17s}")
        out[nm] = row
        print(f"  {nm:16s}" + "".join(f"{c:>17s}" for c in cells))

    # ---------------- 超额夏普 ----------------
    print("\n" + "=" * 118)
    print("超额夏普（策略 − 等权全市场）")
    print("=" * 118)
    bench = series["等权全市场"]
    print(f"  {'组合':16s}" + "".join(f"{w:>12s}" for w, _ in WINDOWS) +
          f"{'样本外3.6年':>13s}")
    for nm in ["等权·多头", "等权·多空", "FM加权·多头", "FM加权·多空"]:
        ser = series[nm]
        cells = []
        for wn, nd in WINDOWS:
            a = metrics(ser.iloc[-nd:] if len(ser) >= nd else ser)
            b = metrics(bench.iloc[-nd:] if len(bench) >= nd else bench)
            cells.append(f"{a['Sharpe']-b['Sharpe']:>+12.2f}" if a and b else f"{'—':>12s}")
        a = metrics(ser[ser.index >= "2023-01-01"])
        b = metrics(bench[bench.index >= "2023-01-01"])
        print(f"  {nm:16s}" + "".join(cells) + f"{a['Sharpe']-b['Sharpe']:>+13.2f}")

    # ---------------- 目标核定（逐窗口） ----------------
    print("\n" + "=" * 118)
    print("目标逐窗口核定（夏普>1.2 / 回撤<15% / Calmar>0.9）")
    print("=" * 118)
    print(f"  {'组合':16s}{'窗口':>12s}{'夏普':>9s}{'达标':>7s}{'回撤':>10s}{'达标':>7s}"
          f"{'Calmar':>9s}{'达标':>7s}{'三项':>8s}")
    verdict = {}
    for nm in ["等权·多头", "等权·多空", "FM加权·多头", "FM加权·多空"]:
        ser = series[nm]
        for wn, nd in WINDOWS:
            m = metrics(ser.iloc[-nd:] if len(ser) >= nd else ser)
            if not m:
                continue
            a = "✓" if m["Sharpe"] > 1.2 else "✗"
            b = "✓" if m["MaxDD"] > -0.15 else "✗"
            c = "✓" if (m["Calmar"] or -9) > 0.9 else "✗"
            n_ok = sum(x == "✓" for x in (a, b, c))
            verdict[f"{nm}|{wn}"] = {"Sharpe": m["Sharpe"], "MaxDD": m["MaxDD"],
                                     "Calmar": m["Calmar"], "n_ok": n_ok}
            print(f"  {nm:16s}{wn:>12s}{m['Sharpe']:>9.2f}{a:>7s}{m['MaxDD']:>10.2%}"
                  f"{b:>7s}{str(m['Calmar']):>9s}{c:>7s}{n_ok:>6d}/3")
        print()

    with open(os.path.join(OUT, "final_multiwindow.json"), "w", encoding="utf-8") as fh:
        json.dump({"windows": out, "verdict": verdict,
                   "fm_weights": dict(zip(names, np.round(w_fm, 5).tolist()))},
                  fh, ensure_ascii=False, indent=2, default=str)
    print(f"[已保存] {os.path.join(OUT,'final_multiwindow.json')}")


if __name__ == "__main__":
    main()
