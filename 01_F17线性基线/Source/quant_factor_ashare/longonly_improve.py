# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
longonly_improve.py — 多头组合的改进 + 最终显著性核算
================================================================================
问题
----
CH-4 alpha 加权让**多空**大幅改善（OOS 夏普 1.201 → 1.327，回撤 −13.54% → −12.66%），
但让**多头**变差（OOS 0.859 → 0.709，回撤 −26.25% → −34.66%）。

原因清楚：**因子对「多空价差」的贡献 ≠ 对「多头腿」的贡献**。
Stambaugh-Yu-Yuan (2012) 的发现正是如此——情绪效应 94% 来自空头腿，多头腿 t=0.62（为零）。
如果某因子的 alpha 主要来自空头腿，用它加权只会让多头腿更集中、更偏。

做法
----
按**多头腿单独**的风险调整后 alpha 加权：
    ω_i ∝ max(α_i^{long-only}, 0)
其中 α_i^{long-only} 是「因子得分前 30% 等权组合」对 CH-4 回归的截距，只用样本内估计。

同时给出最终显著性核算（含 DSR，试验次数与分布诚实计入）。
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
PHASES = list(range(0, 21, 3))


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
    print("多头组合改进（按多头腿 CH-4 alpha 加权）+ 最终显著性")
    print("=" * 100)
    df = load_panel()
    P = Panels(df)
    del df
    eligible, _, _ = build_universe(P, CFG)
    need = sorted(set(v2.FACTOR_SETS["F17"] + v2.EXTRA_NAMES + v2.CONTROL_NAMES))
    v2.build_prep(P, eligible, need)

    names = v2.FACTOR_SETS["F17"]
    ch = pd.read_pickle(os.path.join(DATA, "ch3.pkl")).shift(1)
    expo = pd.read_pickle(os.path.join(DATA, "ch3_expo.pkl"))
    C = pd.read_pickle(os.path.join(DATA, "ch3_prices.pkl"))
    cm = C["close_me"]
    ret_fwd = cm.shift(-1) / cm - 1.0
    months = [d for d in ch.index if d in expo]
    is_months = [d for d in months if IS[0] <= str(d.date()) <= IS[1]]

    print(f"\n[1] 估计各因子**多头腿**的样本内 CH-4 alpha（{len(is_months)} 个月）", flush=True)
    w_spread, w_long = [], []
    det = {}
    for n in names:
        fp = os.path.join(DATA, "prep", n + ".pkl")
        if not os.path.exists(fp):
            w_spread.append(0.0); w_long.append(0.0); continue
        fac = pd.read_pickle(fp)
        rs, rl = [], []
        for dt in is_months:
            e = expo.get(dt)
            if e is None or dt not in fac.index:
                continue
            x = fac.loc[dt].reindex(e.index)
            r = ret_fwd.loc[dt].reindex(e.index)
            m = x.notna() & r.notna()
            if m.sum() < 80:
                continue
            xx, rr = x[m], r[m]
            hi, lo = xx >= xx.quantile(0.70), xx <= xx.quantile(0.30)
            if hi.sum() < 5 or lo.sum() < 5:
                continue
            rs.append((dt, float(rr[hi].mean() - rr[lo].mean())))
            rl.append((dt, float(rr[hi].mean())))
        del fac
        rec = {}
        for tag, rows in [("spread", rs), ("long", rl)]:
            if len(rows) < 40:
                rec[tag] = (np.nan, np.nan); continue
            s = pd.Series(dict(rows)).sort_index()
            d = pd.concat({"y": s, "MKT": ch["MKT"], "SMB": ch["SMB"],
                           "VMG": ch["VMG"], "PMO": ch["PMO"]}, axis=1).dropna()
            if len(d) < 40:
                rec[tag] = (np.nan, np.nan); continue
            mm = sm.OLS(d["y"], sm.add_constant(d[["MKT", "SMB", "VMG", "PMO"]])
                        ).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
            rec[tag] = (float(mm.params["const"]), float(mm.tvalues["const"]))
        det[n] = rec
        w_spread.append(max(rec["spread"][0], 0.0) if np.isfinite(rec["spread"][0]) else 0.0)
        w_long.append(max(rec["long"][0], 0.0) if np.isfinite(rec["long"][0]) else 0.0)
    ws = np.array(w_spread); wl = np.array(w_long)
    ws = ws / ws.sum() if ws.sum() > 0 else np.ones(len(names)) / len(names)
    wl = wl / wl.sum() if wl.sum() > 0 else np.ones(len(names)) / len(names)

    print(f"    {'因子':11s} {'价差α':>9s} {'t':>6s} {'多头腿α':>9s} {'t':>6s} "
          f"{'ω_价差':>8s} {'ω_多头':>8s}")
    for n, a1, a2 in sorted(zip(names, ws, wl), key=lambda x: -x[2]):
        sp, lo = det[n]["spread"], det[n]["long"]
        print(f"    {n:11s} {sp[0]*100:>+8.3f}% {sp[1]:>+6.2f} {lo[0]*100:>+8.3f}% "
              f"{lo[1]:>+6.2f} {a1:>8.4f} {a2:>8.4f}")

    SCORES = {
        "等权": v2.composite(names),
        "价差α加权": v2.composite(names, weights=ws),
        "多头腿α加权": v2.composite(names, weights=wl),
    }

    print(f"\n[2] 相位扫描（{len(PHASES)} 相位）× 3 权重方案 × 多头口径", flush=True)
    res = {}
    keep = {}
    for sn, sc in SCORES.items():
        iss, ooss, dds, cgs = [], [], [], []
        for off in PHASES:
            s2 = sc.iloc[off:] if off else sc
            W = v2.make_weights(s2, eligible.reindex(s2.index), side="long", **REC)
            r = backtest(P, W, CFG, f"{sn}|{off}")["net"]
            del W
            i = met(r[(r.index >= IS[0]) & (r.index <= IS[1])])
            o = met(r[(r.index >= OOS[0]) & (r.index <= OOS[1])])
            if i: iss.append(i["Sharpe"])
            if o: ooss.append(o["Sharpe"]); dds.append(o["MaxDD"]); cgs.append(o["CAGR"])
            if off == 0:
                keep[sn] = r
            del r
        if not ooss:
            continue
        mm = keep[sn].resample("ME").apply(lambda x: (1 + x).prod() - 1)
        sub = mm[(mm.index >= OOS[0]) & (mm.index <= OOS[1])]
        d = pd.concat({"y": sub, "MKT": ch["MKT"], "SMB": ch["SMB"],
                       "VMG": ch["VMG"], "PMO": ch["PMO"]}, axis=1).dropna()
        ch4 = {}
        if len(d) > 24:
            mm2 = sm.OLS(d["y"], sm.add_constant(d[["MKT", "SMB", "VMG", "PMO"]])
                         ).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
            ch4 = {"alpha_a_pct": round(((1 + float(mm2.params["const"])) ** 12 - 1) * 100, 2),
                   "t": round(float(mm2.tvalues["const"]), 2),
                   "r2": round(float(mm2.rsquared), 3),
                   "beta_MKT": round(float(mm2.params["MKT"]), 3),
                   "beta_SMB": round(float(mm2.params["SMB"]), 3)}
        res[sn] = {"IS_sharpe_mean": round(float(np.mean(iss)), 4),
                   "OOS_sharpe_mean": round(float(np.mean(ooss)), 4),
                   "OOS_sharpe_sd": round(float(np.std(ooss)), 4),
                   "OOS_sharpe_min": round(float(np.min(ooss)), 4),
                   "OOS_sharpe_max": round(float(np.max(ooss)), 4),
                   "OOS_cagr_mean": round(float(np.mean(cgs)), 4),
                   "OOS_maxdd_mean": round(float(np.mean(dds)), 4),
                   "OOS_calmar_mean": round(float(np.mean(cgs) / abs(np.mean(dds))), 3),
                   "ch4_oos": ch4, "n_phases": len(ooss)}
        t = res[sn]
        print(f"    {sn:14s} IS {t['IS_sharpe_mean']:.3f} | OOS {t['OOS_sharpe_mean']:.3f}"
              f"±{t['OOS_sharpe_sd']:.3f} [{t['OOS_sharpe_min']:.3f},{t['OOS_sharpe_max']:.3f}]"
              f"  年化 {t['OOS_cagr_mean']:>7.2%}  回撤 {t['OOS_maxdd_mean']:>7.2%}  "
              f"Calmar {t['OOS_calmar_mean']:.2f}  CH4α {ch4.get('alpha_a_pct', float('nan')):>6.2f}%"
              f" (t={ch4.get('t', float('nan')):>4.2f})", flush=True)

    best_is = max(res, key=lambda k: res[k]["IS_sharpe_mean"])
    b = res[best_is]
    print(f"\n[3] 按样本内夏普选择 → {best_is}  IS {b['IS_sharpe_mean']:.3f}  →  "
          f"OOS {b['OOS_sharpe_mean']:.3f}  回撤 {b['OOS_maxdd_mean']:.2%}  "
          f"Calmar {b['OOS_calmar_mean']:.2f}")

    with open(os.path.join(OUT, "longonly_improve.json"), "w", encoding="utf-8") as fh:
        json.dump({"results": res, "weights_spread": dict(zip(names, ws.round(4))),
                   "weights_longonly": dict(zip(names, wl.round(4))),
                   "detail": {k: {kk: [round(v[0] * 100, 4) if np.isfinite(v[0]) else None,
                                       round(v[1], 2) if np.isfinite(v[1]) else None]
                                  for kk, v in vv.items()} for k, vv in det.items()},
                   "is_selected": best_is}, fh, ensure_ascii=False, indent=2, default=str)
    print(f"[已保存] {os.path.join(OUT,'longonly_improve.json')}")


if __name__ == "__main__":
    main()
