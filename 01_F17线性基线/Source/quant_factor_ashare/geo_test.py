# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
geo_test.py — 地理联动因子的 IC 与组合增量检验
================================================================================
背景
----
GEO_LINK（同城市·不同行业其他公司过去 1 月均值收益，等权）与既有因子的
截面相关性几乎为零：
    vs COMMOVE +0.011   vs LEAD_LAG +0.000   vs TURN +0.022   vs MAX5 +0.047
这与 Parsons-Sabbatucci-Titman (2020, RFS) 报告的性质一致
（效应对分析师覆盖度平坦 → 与标准因子正交）。

**这是本项目加入的第一个真正独立的新信息**（CGO 与 MAX5 相关 0.44、
ABN_TURN/VTURN 与 TURN 同族，均已被证明无增量）。

检验
----
1. 各 GEO 因子的样本内/样本外 RankIC
2. 单独看它的多空组合对 CH-4 的 alpha（是否独立于中国三/四因子）
3. 加进组合后的相位扫描：F17 vs F18（多头与多空）
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
                                    CFG, newey_west_tstat)
import strategy_v2 as v2

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
PPY, RF = 252, 0.02
IS = ("2010-01-01", "2022-12-31")
OOS = ("2023-01-01", "2026-12-31")
REC = dict(n=50, weighting="equal", buffer=0.6, rebalance_days=21, max_weight=0.03)
PHASES = list(range(0, 21, 3))
GEO = ["GEO_LINK", "GEO_CITY", "GEO_PROV", "GEO_IND"]


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
    print("=" * 100)
    print("地理联动因子：IC 与组合增量")
    print("=" * 100)
    df = load_panel()
    P = Panels(df)
    del df
    eligible, _, _ = build_universe(P, CFG)
    need = sorted(set(v2.FACTOR_SETS["F17"] + v2.EXTRA_NAMES + v2.CONTROL_NAMES + GEO))
    v2.build_prep(P, eligible, need)

    print("\n[1] 各 GEO 因子的 RankIC")
    ic_tbl = {}
    for n in GEO:
        r = v2.rank_ic_report(P, eligible, n)
        ic_tbl[n] = r
        i2, o2 = r.get("IS", {}), r.get("OOS", {})
        print(f"    {n:10s} 样本内 IC={i2.get('IC')} (t={i2.get('t')}, ICIR={i2.get('ICIR')}, "
              f"正IC占比={i2.get('pos_rate')})  |  样本外 IC={o2.get('IC')} (t={o2.get('t')})")
    print("\n    对照（既有因子）：")
    for n in ["COMMOVE", "TURN", "MAX5", "ILLIQ", "LEAD_LAG"]:
        r = v2.rank_ic_report(P, eligible, n)
        ic_tbl[n] = r
        i2, o2 = r.get("IS", {}), r.get("OOS", {})
        print(f"    {n:10s} 样本内 IC={i2.get('IC')}  |  样本外 IC={o2.get('IC')}")

    print("\n[2] 组合对比（相位扫描 7 个相位）", flush=True)
    SETS = {"F17": v2.FACTOR_SETS["F17"],
            "F17+GEO_LINK": v2.FACTOR_SETS["F17"] + ["GEO_LINK"],
            "F17+GEO_PROV": v2.FACTOR_SETS["F17"] + ["GEO_PROV"]}
    ch = pd.read_pickle(os.path.join(DATA, "ch3.pkl")).shift(1)
    res = {}
    for sn, fl in SETS.items():
        sc = v2.composite(fl)
        for mode in ["多头", "多空"]:
            iss, ooss, dds, cgs, keep0 = [], [], [], [], None
            for off in PHASES:
                s2 = sc.iloc[off:] if off else sc
                el2 = eligible.reindex(s2.index)
                Wl = v2.make_weights(s2, el2, side="long", **REC)
                if mode == "多空":
                    Ws = v2.make_weights(s2, el2, side="short", **REC)
                    Wl = Wl.sub(Ws.reindex(columns=Wl.columns, fill_value=0.0),
                                fill_value=0.0)
                    del Ws
                r = backtest(P, Wl, CFG, f"{sn}|{mode}|{off}")["net"]
                del Wl
                i = met(r[(r.index >= IS[0]) & (r.index <= IS[1])])
                o = met(r[(r.index >= OOS[0]) & (r.index <= OOS[1])])
                if i: iss.append(i["Sharpe"])
                if o: ooss.append(o["Sharpe"]); dds.append(o["MaxDD"]); cgs.append(o["CAGR"])
                if off == 0:
                    keep0 = r
                del r
            if not ooss:
                continue
            mm = keep0.resample("ME").apply(lambda x: (1 + x).prod() - 1)
            sub = mm[(mm.index >= OOS[0]) & (mm.index <= OOS[1])]
            d = pd.concat({"y": sub, "MKT": ch["MKT"], "SMB": ch["SMB"],
                           "VMG": ch["VMG"], "PMO": ch["PMO"]}, axis=1).dropna()
            ch4 = {}
            if len(d) > 24:
                m = sm.OLS(d["y"], sm.add_constant(d[["MKT", "SMB", "VMG", "PMO"]])
                           ).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
                ch4 = {"alpha_a_pct": round(((1 + float(m.params["const"])) ** 12 - 1) * 100, 2),
                       "t": round(float(m.tvalues["const"]), 2),
                       "r2": round(float(m.rsquared), 3)}
            res[f"{sn}|{mode}"] = {
                "IS_sharpe_mean": round(float(np.mean(iss)), 4),
                "OOS_sharpe_mean": round(float(np.mean(ooss)), 4),
                "OOS_sharpe_sd": round(float(np.std(ooss)), 4),
                "OOS_sharpe_min": round(float(np.min(ooss)), 4),
                "OOS_sharpe_max": round(float(np.max(ooss)), 4),
                "OOS_cagr_mean": round(float(np.mean(cgs)), 4),
                "OOS_maxdd_mean": round(float(np.mean(dds)), 4),
                "OOS_calmar_mean": round(float(np.mean(cgs) / abs(np.mean(dds))), 3),
                "ch4_oos": ch4, "n_phases": len(ooss), "n_factors": len(fl)}
            t = res[f"{sn}|{mode}"]
            print(f"    {sn:14s} {mode}  IS {t['IS_sharpe_mean']:.3f} | "
                  f"OOS {t['OOS_sharpe_mean']:.3f}±{t['OOS_sharpe_sd']:.3f} "
                  f"[{t['OOS_sharpe_min']:.3f},{t['OOS_sharpe_max']:.3f}]  "
                  f"年化 {t['OOS_cagr_mean']:>7.2%}  回撤 {t['OOS_maxdd_mean']:>7.2%}  "
                  f"Calmar {t['OOS_calmar_mean']:.2f}  "
                  f"CH4α {ch4.get('alpha_a_pct', float('nan')):>6.2f}% "
                  f"(t={ch4.get('t', float('nan')):>4.2f})", flush=True)

    with open(os.path.join(OUT, "geo_test.json"), "w", encoding="utf-8") as fh:
        json.dump({"ic": ic_tbl, "results": res}, fh, ensure_ascii=False, indent=2,
                  default=str)
    print(f"\n[已保存] {os.path.join(OUT,'geo_test.json')}")


if __name__ == "__main__":
    main()
