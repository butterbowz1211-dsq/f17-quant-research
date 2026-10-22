# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
dsr_phase_range.py — 逐相位 DSR（给出最差相位的下界，而非只报相位 0）
================================================================================
为什么必须做
------------
DSR = 0.9719 是**相位 0** 的结果，而相位 0 是 11 个相位里夏普最高的一档。
若只报它，就重复了本项目之前犯过两次的错误（单相位冒充当期结论）。

本脚本对 11 个调仓相位分别计算 近 24 个月窗口 的 DSR，给出：
    中位相位 DSR / **最差相位 DSR** / 达标相位占比
以此为「多重检验校正」这一目标给出**下界式**的诚实结论。

试验分布取自 dsr_family.py 的精确收集结果（114 个多头配置），
门槛 SR0 = 0.349。
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
OUT = os.path.join(HERE, "output")
PPY, RF = 252, 0.02
REC = dict(n=50, weighting="equal", buffer=0.6, rebalance_days=21, max_weight=0.03)
PHASES = list(range(0, 21, 2))          # 11 个相位


def main():
    from ashare_factor_strategy import (load_panel, Panels, build_universe, backtest,
                                        CFG, probabilistic_sharpe, deflated_sharpe)
    import strategy_v2 as v2

    with open(os.path.join(OUT, "dsr_family.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    fam = d["trial_stats"]["long"]
    N, VAR = int(fam["n"]), float(fam["var"])
    pooled = d["trial_stats"]["pooled"]
    print(f"试验分布（多头族）：N={N}  方差={VAR:.4f}  标准差={fam['sd']:.4f}")
    print(f"试验分布（合并）  ：N={pooled['n']}  方差={pooled['var']:.4f}")

    df = load_panel()
    P = Panels(df)
    del df
    eligible, _, _ = build_universe(P, CFG)
    need = sorted(set(v2.FACTOR_SETS["F17"] + v2.EXTRA_NAMES + v2.CONTROL_NAMES))
    v2.build_prep(P, eligible, need)
    score = v2.composite(v2.FACTOR_SETS["F17"])

    print(f"\n逐相位计算（近 24 个月 = 504 个交易日）", flush=True)
    rows = []
    for off in PHASES:
        s2 = score.iloc[off:] if off else score
        W = v2.make_weights(s2, eligible.reindex(s2.index), side="long", **REC)
        r = backtest(P, W, CFG, f"ph{off}")["net"]
        del W
        seg = r.iloc[-504:].dropna()
        if len(seg) < 100:
            continue
        vol = seg.std() * np.sqrt(PPY)
        sr = (seg.mean() * PPY - RF) / vol
        psr = probabilistic_sharpe(sr / np.sqrt(PPY), len(seg),
                                   float(seg.skew()), float(seg.kurtosis() + 3.0))
        d_fam = deflated_sharpe(seg, n_trials=N, sr_variance=VAR)
        d_all = deflated_sharpe(seg, n_trials=int(pooled["n"]), sr_variance=float(pooled["var"]))
        eq = (1 + seg).cumprod()
        mdd = float((eq / eq.cummax() - 1).min())
        rows.append({"phase": off, "sharpe": round(float(sr), 4),
                     "PSR": round(float(psr), 4),
                     "DSR_family": round(float(d_fam["DSR"]), 4),
                     "DSR_pooled": round(float(d_all["DSR"]), 4),
                     "MaxDD": round(mdd, 4)})
        v = rows[-1]
        print(f"  相位 {off:>2d}  夏普 {v['sharpe']:>6.3f}  PSR {v['PSR']:.4f}  "
              f"DSR(本族) {v['DSR_family']:.4f}  DSR(合并) {v['DSR_pooled']:.4f}  "
              f"回撤 {v['MaxDD']:>7.2%}", flush=True)
        del r

    S = pd.DataFrame(rows)
    print("\n" + "=" * 88)
    print("汇总（这是「多重检验校正」目标的诚实下界）")
    print("=" * 88)
    for col in ["sharpe", "PSR", "DSR_family", "DSR_pooled"]:
        a = S[col].to_numpy()
        print(f"  {col:12s} 中位 {np.median(a):>7.4f}  均值 {a.mean():>7.4f}  "
              f"**最差 {a.min():>7.4f}**  最好 {a.max():>7.4f}  "
              f">0.95 的相位占比 {float((a > 0.95).mean()):>5.0%}")
    a = S["MaxDD"].to_numpy()
    print(f"  {'MaxDD':12s} 中位 {np.median(a):>7.2%}  均值 {a.mean():>7.2%}  "
          f"**最差 {a.min():>7.2%}**  <15% 的相位占比 {float((a > -0.15).mean()):>5.0%}")

    ok_sharpe = (S["sharpe"] > 1.2).mean()
    ok_dd = (S["MaxDD"] > -0.15).mean()
    print(f"\n  夏普>1.2 的相位占比      {ok_sharpe:.0%}")
    print(f"  回撤<15% 的相位占比      {ok_dd:.0%}")

    with open(os.path.join(OUT, "dsr_phase_range.json"), "w", encoding="utf-8") as fh:
        json.dump({"per_phase": rows, "N_family": N, "var_family": VAR,
                   "summary": {c: {"median": round(float(np.median(S[c])), 4),
                                   "min": round(float(S[c].min()), 4),
                                   "pct_gt_095": round(float((S[c] > 0.95).mean()), 3)}
                               for c in ["sharpe", "PSR", "DSR_family", "DSR_pooled"]}},
                  fh, ensure_ascii=False, indent=2)
    print(f"\n[已保存] {os.path.join(OUT,'dsr_phase_range.json')}")


if __name__ == "__main__":
    main()
