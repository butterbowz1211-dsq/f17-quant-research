# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
leg_decomposition.py — 改进来源分解 + DSR 敏感性 + 年度分解
================================================================================
三个问题
--------
1. **CH-4 α 加权带来的改进，来自多头腿还是空头腿？**
   Stambaugh-Yu-Yuan (2012) 发现异象效应 94% 来自空头腿、多头腿 t=0.62（为零）。
   若我们的改进同样只来自空头腿，那么「多头口径无法受益」就是结构性的，
   而不是我们没找对权重方式。这直接决定多头还有没有救。

2. **DSR 对「试验次数」假设有多敏感？**
   DSR 的 SR0 取决于试验夏普的方差与次数 N。N 的选取有主观性，
   因此把 N ∈ {10, 20, 50, 100, 312} 全部报出来，不挑一个好看的。

3. 年度分解（检验是否依赖某一年）。
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ashare_factor_strategy import (load_panel, Panels, build_universe, backtest,
                                    CFG, newey_west_tstat, probabilistic_sharpe,
                                    deflated_sharpe)
import strategy_v2 as v2

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
PPY, RF = 252, 0.02
IS = ("2010-01-01", "2022-12-31")
OOS = ("2023-01-01", "2026-12-31")
REC = dict(n=50, weighting="equal", buffer=0.6, rebalance_days=21, max_weight=0.03)


def met(seg):
    seg = pd.Series(seg).dropna()
    if len(seg) < 20:
        return None
    eq = (1 + seg).cumprod()
    cagr = eq.iloc[-1] ** (PPY / len(seg)) - 1
    vol = seg.std() * np.sqrt(PPY)
    dd = float((eq / eq.cummax() - 1).min())
    return {"Sharpe": float((seg.mean() * PPY - RF) / vol), "CAGR": float(cagr),
            "AnnVol": float(vol), "MaxDD": dd,
            "Calmar": float(cagr / abs(dd)) if dd < 0 else None}


def main():
    df = load_panel()
    P = Panels(df)
    del df
    eligible, _, _ = build_universe(P, CFG)
    need = sorted(set(v2.FACTOR_SETS["F17"] + v2.EXTRA_NAMES + v2.CONTROL_NAMES))
    v2.build_prep(P, eligible, need)
    names = v2.FACTOR_SETS["F17"]
    with open(os.path.join(OUT, "longonly_improve.json"), encoding="utf-8") as fh:
        wv = json.load(fh)["weights_spread"]
    w = np.array([wv[n] for n in names]); w = w / w.sum()

    SC = {"等权": v2.composite(names),
          "CH4alpha加权": v2.composite(names, weights=w)}

    # ---------------- 1. 分腿分解 ----------------
    print("=" * 100)
    print("1  改进来源分解：多头腿 vs 空头腿")
    print("=" * 100)
    legs = {}
    for sn, sc in SC.items():
        Wl = v2.make_weights(sc, eligible, side="long", **REC)
        Ws = v2.make_weights(sc, eligible, side="short", **REC)
        rl = backtest(P, Wl, CFG, f"{sn}_L")["net"]
        rs = backtest(P, Ws, CFG, f"{sn}_S")["net"]
        # 自融资价差
        Wls = Wl.sub(Ws.reindex(columns=Wl.columns, fill_value=0.0), fill_value=0.0)
        rls = backtest(P, Wls, CFG, f"{sn}_LS")["net"]
        del Wl, Ws, Wls
        legs[sn] = {"long": rl, "short": rs, "ls": rls}
        print(f"\n  --- {sn} ---")
        for leg, ser in [("多头腿", rl), ("空头腿", rs), ("多空价差", rls)]:
            for lab, (a, b) in [("样本内", IS), ("样本外", OOS)]:
                m = met(ser[(ser.index >= a) & (ser.index <= b)])
                if m:
                    print(f"    {leg:8s} {lab}  年化 {m['CAGR']:>8.2%}  波动 {m['AnnVol']:>7.2%}  "
                          f"夏普 {m['Sharpe']:>6.3f}  回撤 {m['MaxDD']:>8.2%}")

    print("\n  改进的分解（样本外，CH4α加权 − 等权）：")
    for leg in ["long", "short", "ls"]:
        a = met(legs["等权"][leg][(legs["等权"][leg].index >= OOS[0]) &
                                 (legs["等权"][leg].index <= OOS[1])])
        b = met(legs["CH4alpha加权"][leg][(legs["CH4alpha加权"][leg].index >= OOS[0]) &
                                         (legs["CH4alpha加权"][leg].index <= OOS[1])])
        if a and b:
            print(f"    {leg:6s}  夏普 {a['Sharpe']:>6.3f} → {b['Sharpe']:>6.3f}  "
                  f"(Δ {b['Sharpe']-a['Sharpe']:>+6.3f})   "
                  f"年化 {a['CAGR']:>7.2%} → {b['CAGR']:>7.2%} "
                  f"(Δ {(b['CAGR']-a['CAGR'])*100:>+5.2f}pp)")

    # ---------------- 2. DSR 对 N 的敏感性 ----------------
    print("\n" + "=" * 100)
    print("2  DSR 对「试验次数 N」的敏感性")
    print("=" * 100)
    trial = []
    for f in ["ls_improve.json", "longonly_improve.json", "v2_phase.json",
              "neutral_experiment.json", "v2_tune.json", "v2_grid.json",
              "v2_beta.json", "calibration.json"]:
        fp = os.path.join(OUT, f)
        if not os.path.exists(fp):
            continue
        try:
            with open(fp, encoding="utf-8") as fh:
                j = json.load(fh)
            stack = [j]
            while stack:
                o = stack.pop()
                if isinstance(o, dict):
                    for k, v in o.items():
                        if k in ("OOS_sharpe_mean", "Sharpe") and isinstance(v, (int, float)):
                            trial.append(float(v))
                        elif isinstance(v, dict):
                            ov = v.get("OOS")
                            if isinstance(ov, dict) and ov.get("Sharpe") is not None:
                                trial.append(float(ov["Sharpe"]))
                            stack.append(v)
        except Exception:
            pass
    ts = pd.Series(trial).replace([np.inf, -np.inf], np.nan).dropna()
    var = float(ts.var(ddof=1))
    print(f"  试验夏普样本：n={len(ts)}  均值 {ts.mean():.3f}  标准差 {ts.std():.3f}  方差 {var:.4f}")
    print(f"  注：这 {len(ts)} 个多为同一策略的相位/参数变体，高度相关；")
    print(f"      「独立试验数」远小于此，故下表把 N 从 10 取到 {len(ts)}。\n")
    r_ls = legs["CH4alpha加权"]["ls"]
    o = r_ls[(r_ls.index >= OOS[0]) & (r_ls.index <= OOS[1])]
    f_ = r_ls[(r_ls.index >= "2010-01-01") & (r_ls.index <= "2026-12-31")]
    tbl = {}
    print(f"  {'N':>5s} {'SR0(年化)':>10s} {'OOS DSR':>9s} {'全样本 DSR':>11s}")
    for N in [10, 20, 50, 100, len(ts)]:
        d1 = deflated_sharpe(o, n_trials=N, sr_variance=var)
        d2 = deflated_sharpe(f_, n_trials=N, sr_variance=var)
        tbl[N] = {"SR0": round(float(d1["SR0"]), 4), "DSR_oos": round(float(d1["DSR"]), 4),
                  "DSR_full": round(float(d2["DSR"]), 4)}
        print(f"  {N:>5d} {d1['SR0']:>10.3f} {d1['DSR']:>9.4f} {d2['DSR']:>11.4f}")
    sr_o = (o.mean() * PPY - RF) / (o.std() * np.sqrt(PPY))
    print(f"\n  参考：样本外夏普 {sr_o:.3f}  NW t={newey_west_tstat(o, lags=5)[1]:+.2f}")
    print(f"        PSR={probabilistic_sharpe(sr_o/np.sqrt(PPY), len(o), float(o.skew()), float(o.kurtosis()+3.0)):.4f}")

    # ---------------- 3. 年度分解 ----------------
    print("\n" + "=" * 100)
    print("3  年度分解（检验是否依赖某一年）")
    print("=" * 100)
    print(f"  {'年份':6s} {'多头':>10s} {'多空':>10s} {'基准':>10s}")
    wb = v2.universe_weights(eligible)
    rb = backtest(P, wb, CFG, "基准")["net"]
    yr = {}
    for y in range(2010, 2027):
        row = {}
        for nm, ser in [("多头", legs["CH4alpha加权"]["long"]),
                        ("多空", legs["CH4alpha加权"]["ls"]),
                        ("基准", rb)]:
            m = ser[ser.index.year == y]
            row[nm] = float((1 + m).prod() - 1) if len(m) > 20 else None
        yr[y] = row
        def fm(v):
            return f"{v:>10.2%}" if v is not None else f"{'—':>10s}"
        print(f"  {y:6d} {fm(row['多头'])} {fm(row['多空'])} {fm(row['基准'])}")

    with open(os.path.join(OUT, "leg_decomposition.json"), "w", encoding="utf-8") as fh:
        json.dump({"legs": {sn: {lg: met(s[(s.index >= OOS[0]) & (s.index <= OOS[1])])
                                 for lg, s in v.items()} for sn, v in legs.items()},
                   "dsr_sensitivity": {str(k): v for k, v in tbl.items()},
                   "trial_stats": {"n": int(len(ts)), "mean": round(float(ts.mean()), 4),
                                   "sd": round(float(ts.std()), 4), "var": round(var, 4)},
                   "yearly": yr}, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n[已保存] {os.path.join(OUT,'leg_decomposition.json')}")


if __name__ == "__main__":
    main()
