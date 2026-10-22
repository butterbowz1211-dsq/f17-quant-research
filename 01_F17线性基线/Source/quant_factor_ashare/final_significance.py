# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
final_significance.py — 最终推荐配置的显著性核算
================================================================================
推荐配置（多空口径）
--------------------
    因子集   F17（17 个）
    因子权重 样本内 CH-4 **价差** alpha 加权（只用 2010-2022 估计，无前视）
    持仓     多空各 50 名等权，单票上限 3%
    缓冲区   0.6
    调仓     每 21 个交易日

核算内容
--------
· Newey-West t（Bartlett，lag=5）
· PSR / DSR（试验次数与夏普分布诚实计入）
· 相位稳健性（7 个调仓相位的均值、标准差、区间）
· CH-4 alpha
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


def main():
    df = load_panel()
    P = Panels(df)
    del df
    eligible, _, _ = build_universe(P, CFG)
    need = sorted(set(v2.FACTOR_SETS["F17"] + v2.EXTRA_NAMES + v2.CONTROL_NAMES))
    v2.build_prep(P, eligible, need)
    names = v2.FACTOR_SETS["F17"]

    with open(os.path.join(OUT, "longonly_improve.json"), encoding="utf-8") as fh:
        ws = json.load(fh)["weights_spread"]
    w = np.array([ws[n] for n in names])
    w = w / w.sum()
    sc = v2.composite(names, weights=w)
    print("[1] 用样本内 CH-4 价差 alpha 权重合成得分完成")

    ch = pd.read_pickle(os.path.join(DATA, "ch3.pkl")).shift(1)

    # 相位扫描（多空）
    sharpes, keep = [], {}
    for off in PHASES:
        s2 = sc.iloc[off:] if off else sc
        el2 = eligible.reindex(s2.index)
        Wl = v2.make_weights(s2, el2, side="long", **REC)
        Ws = v2.make_weights(s2, el2, side="short", **REC)
        Wls = Wl.sub(Ws.reindex(columns=Wl.columns, fill_value=0.0), fill_value=0.0)
        del Wl, Ws
        r = backtest(P, Wls, CFG, f"LS|{off}")["net"]
        del Wls
        o = r[(r.index >= OOS[0]) & (r.index <= OOS[1])]
        if len(o) > 50:
            sharpes.append(float((o.mean() * PPY - RF) / (o.std() * np.sqrt(PPY))))
        if off == 0:
            keep["net"] = r
        del r
    r_ls = keep["net"]
    print(f"[2] 相位扫描完成：OOS 夏普 均值 {np.mean(sharpes):.3f} "
          f"标准差 {np.std(sharpes):.3f} 区间 [{np.min(sharpes):.3f}, {np.max(sharpes):.3f}]")

    out = {"phase": {"mean": round(float(np.mean(sharpes)), 4),
                     "sd": round(float(np.std(sharpes)), 4),
                     "min": round(float(np.min(sharpes)), 4),
                     "max": round(float(np.max(sharpes)), 4),
                     "n": len(sharpes), "offsets": PHASES}}

    # 试验分布（用于 DSR）
    trial = list(sharpes)
    for f in ["ls_improve.json", "longonly_improve.json", "v2_phase.json",
              "neutral_experiment.json", "v2_tune.json", "v2_grid.json", "v2_beta.json"]:
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
                            if isinstance(ov, dict) and "Sharpe" in ov and ov["Sharpe"] is not None:
                                trial.append(float(ov["Sharpe"]))
                            stack.append(v)
        except Exception:
            pass
    ts = pd.Series(trial).replace([np.inf, -np.inf], np.nan).dropna()
    sr_var = float(ts.var(ddof=1)) if len(ts) > 1 else 0.0
    n_trials = int(len(ts))
    print(f"[3] 试验分布：n={n_trials}  均值 {ts.mean():.3f}  标准差 {ts.std():.3f}  "
          f"方差 {sr_var:.4f}")

    sig = {}
    for lab, ser, (s0, e0) in [("多空|样本外", r_ls, OOS),
                               ("多空|全样本", r_ls, ("2010-01-01", "2026-12-31"))]:
        seg = ser[(ser.index >= s0) & (ser.index <= e0)]
        t = newey_west_tstat(seg, lags=5)[1]
        sr = (seg.mean() * PPY - RF) / (seg.std() * np.sqrt(PPY))
        psr = probabilistic_sharpe(sr / np.sqrt(PPY), len(seg),
                                   float(seg.skew()), float(seg.kurtosis() + 3.0))
        dsr = deflated_sharpe(seg, n_trials=n_trials, sr_variance=sr_var)
        eq = (1 + seg).cumprod()
        cagr = eq.iloc[-1] ** (PPY / len(seg)) - 1
        mdd = float((eq / eq.cummax() - 1).min())
        sig[lab] = {"Sharpe": round(float(sr), 4), "NW_t": round(float(t), 3),
                    "PSR": round(float(psr), 4), "DSR": round(float(dsr["DSR"]), 4),
                    "SR0_ann": round(float(dsr["SR0"]), 4),
                    "CAGR": round(float(cagr), 4), "AnnVol": round(float(seg.std() * np.sqrt(PPY)), 4),
                    "MaxDD": round(mdd, 4),
                    "Calmar": round(float(cagr / abs(mdd)), 3) if mdd < 0 else None,
                    "n_days": int(len(seg))}
        v = sig[lab]
        print(f"    {lab:12s} 年化 {v['CAGR']:>7.2%}  波动 {v['AnnVol']:>6.2%}  "
              f"夏普 {v['Sharpe']:>6.3f}  NW t={v['NW_t']:>+6.2f}  回撤 {v['MaxDD']:>7.2%}  "
              f"Calmar {str(v['Calmar']):>5s}  PSR={v['PSR']:.4f}  "
              f"DSR={v['DSR']:.4f} (SR0={v['SR0_ann']:.3f})")

    # CH-4
    mm = r_ls.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    for lab, (s0, e0) in [("样本外", OOS), ("全样本", ("2010-01-01", "2026-12-31"))]:
        sub = mm[(mm.index >= s0) & (mm.index <= e0)]
        d = pd.concat({"y": sub, "MKT": ch["MKT"], "SMB": ch["SMB"],
                       "VMG": ch["VMG"], "PMO": ch["PMO"]}, axis=1).dropna()
        m = sm.OLS(d["y"], sm.add_constant(d[["MKT", "SMB", "VMG", "PMO"]])
                   ).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        a = float(m.params["const"])
        sig[f"CH4|{lab}"] = {"alpha_a_pct": round(((1 + a) ** 12 - 1) * 100, 3),
                             "alpha_m_pct": round(a * 100, 4),
                             "t": round(float(m.tvalues["const"]), 2),
                             "betas": {c: round(float(m.params[c]), 3)
                                       for c in ["MKT", "SMB", "VMG", "PMO"]},
                             "r2": round(float(m.rsquared), 4), "n_months": int(len(d))}
        v = sig[f"CH4|{lab}"]
        print(f"    CH-4 {lab}: α={v['alpha_a_pct']:>+7.2f}%/年  t={v['t']:>+5.2f}  "
              f"R²={v['r2']:.3f}  β={v['betas']}")

    out["significance"] = sig
    out["n_trials"] = n_trials
    out["trial_var"] = round(sr_var, 4)
    with open(os.path.join(OUT, "final_significance.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n[已保存] {os.path.join(OUT,'final_significance.json')}")


if __name__ == "__main__":
    main()
