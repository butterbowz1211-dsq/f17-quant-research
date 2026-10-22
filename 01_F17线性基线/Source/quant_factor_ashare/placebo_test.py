# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
placebo_test.py — 安慰剂检验：不依赖「试验次数 N」的显著性检验
================================================================================
为什么需要它
------------
DSR（Bailey & López de Prado 2014）是多重检验校正的标准工具，但它有一个
**实践上无法客观确定的输入**：试验次数 N 与试验夏普的方差。报告 DSR=0.91 还是 0.94，
取决于我声称自己认真试过多少个配置 —— 这本质上是自证的。

**安慰剂检验绕开了这个问题。** 做法：

    在每个调仓日，把因子得分的数值在**截面内随机打乱**（保留得分的分布、
    保留股票池、保留换手与成本结构、保留组合构建规则），然后重跑完整策略。
    重复 B 次，得到「无预测能力时」的业绩分布。

    真实策略在这个分布中的位置即为 p 值。

它回答的问题比 DSR 更直接：
    **「按我的得分选股，是否显著优于从同一股票池里随机选股？」**

对照组设计
----------
· 打乱后组合的**风格暴露会消失**（随机选股 ≈ 市场组合），因此不能只比原始夏普，
  否则会把 beta 差异算成 alpha。故同时报告：
    1. 原始夏普（含 beta）
    2. **CH-4 alpha**（剔除中国市场三/四因子后）—— 这是主判据
· 同时给 DSR 作对照，说明两种方法结论是否一致。
"""

import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ashare_factor_strategy import (load_panel, Panels, build_universe, backtest,
                                    CFG, deflated_sharpe)
import strategy_v2 as v2

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
PPY, RF = 252, 0.02
IS = ("2010-01-01", "2022-12-31")
OOS = ("2023-01-01", "2026-12-31")
WIN24 = 504
REC = dict(n=50, weighting="equal", buffer=0.6, rebalance_days=21, max_weight=0.03)
B = 200


def sharpe(seg):
    seg = pd.Series(seg).dropna()
    if len(seg) < 30:
        return np.nan
    vol = seg.std() * np.sqrt(PPY)
    return float((seg.mean() * PPY - RF) / vol) if vol > 0 else np.nan


def ch4_alpha(seg, ch):
    mm = seg.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    d = pd.concat({"y": mm, "MKT": ch["MKT"], "SMB": ch["SMB"],
                   "VMG": ch["VMG"], "PMO": ch["PMO"]}, axis=1).dropna()
    if len(d) < 24:
        return np.nan, np.nan
    m = sm.OLS(d["y"], sm.add_constant(d[["MKT", "SMB", "VMG", "PMO"]])
               ).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return float(m.params["const"]) * 12 * 100, float(m.tvalues["const"])


def main():
    t0 = time.time()
    print("=" * 104)
    print(f"安慰剂检验：截面随机打乱 × {B} 次")
    print("=" * 104)
    df = load_panel()
    P = Panels(df)
    del df
    eligible, _, _ = build_universe(P, CFG)
    need = sorted(set(v2.FACTOR_SETS["F17"] + v2.EXTRA_NAMES + v2.CONTROL_NAMES))
    v2.build_prep(P, eligible, need)
    score = v2.composite(v2.FACTOR_SETS["F17"])
    ch = pd.read_pickle(os.path.join(DATA, "ch3.pkl")).shift(1)
    print("  因子得分完成", flush=True)

    # ---- 真实策略 ----
    def run(sc):
        W = v2.make_weights(sc, eligible, side="long", **REC)
        r = backtest(P, W, CFG, "run")["net"]
        del W
        return r

    r_true = run(score)
    true_full = sharpe(r_true)
    true_oos = sharpe(r_true[(r_true.index >= OOS[0]) & (r_true.index <= OOS[1])])
    true_24 = sharpe(r_true.iloc[-WIN24:])
    a_full = ch4_alpha(r_true, ch)
    a_oos = ch4_alpha(r_true[(r_true.index >= OOS[0]) & (r_true.index <= OOS[1])], ch)
    print(f"\n  真实策略：全样本夏普 {true_full:.3f}  样本外 {true_oos:.3f}  "
          f"近24月 {true_24:.3f}")
    print(f"            CH-4 α(全样本) {a_full[0]:+.2f}%/年 (t={a_full[1]:.2f})  "
          f"CH-4 α(样本外) {a_oos[0]:+.2f}%/年 (t={a_oos[1]:.2f})")

    # ---- 安慰剂：逐日截面打乱得分 ----
    print(f"\n  开始 {B} 次截面随机打乱 ...", flush=True)
    Sv = score.to_numpy(dtype=np.float64, copy=True)
    rows_idx = np.arange(Sv.shape[0])
    res = {"full_sharpe": [], "oos_sharpe": [], "sh24": [],
           "ch4_full_a": [], "ch4_oos_a": []}
    for b in range(B):
        rng = np.random.default_rng(10000 + b)
        S2 = Sv.copy()
        for i in rows_idx[::3]:          # 每 3 日打乱一次即可（调仓在 21 日网格上）
            v = np.isfinite(S2[i])
            idx = np.flatnonzero(v)
            if idx.size > 10:
                S2[i, idx] = S2[i, rng.permutation(idx)]
        sc2 = pd.DataFrame(S2, index=score.index, columns=score.columns)
        r2 = run(sc2)
        del sc2
        res["full_sharpe"].append(sharpe(r2))
        res["oos_sharpe"].append(sharpe(r2[(r2.index >= OOS[0]) & (r2.index <= OOS[1])]))
        res["sh24"].append(sharpe(r2.iloc[-WIN24:]))
        res["ch4_full_a"].append(ch4_alpha(r2, ch)[0])
        res["ch4_oos_a"].append(
            ch4_alpha(r2[(r2.index >= OOS[0]) & (r2.index <= OOS[1])], ch)[0])
        del r2, S2
        if (b + 1) % 20 == 0:
            el = time.time() - t0
            print(f"    [{b+1}/{B}] 用时 {el/60:.1f} 分  预计剩余 "
                  f"{el/(b+1)*(B-b-1)/60:.1f} 分", flush=True)

    def rep(key, actual):
        arr = np.array([x for x in res[key] if np.isfinite(x)])
        if arr.size < 10:
            return {}
        p = float((arr >= actual).mean())
        return {"n": int(arr.size), "mean": round(float(arr.mean()), 4),
                "sd": round(float(arr.std()), 4),
                "p95": round(float(np.percentile(arr, 95)), 4),
                "max": round(float(arr.max()), 4),
                "actual": round(float(actual), 4), "p_value": round(p, 4),
                "z": round(float((actual - arr.mean()) / (arr.std() + 1e-12)), 2)}

    print("\n" + "=" * 104)
    print("安慰剂分布 vs 真实策略")
    print("=" * 104)
    out = {}
    for key, lab, actual in [
            ("full_sharpe", "全样本夏普", true_full),
            ("oos_sharpe", "样本外夏普", true_oos),
            ("sh24", "近24月夏普", true_24),
            ("ch4_full_a", "CH-4 α 全样本(%/年)", a_full[0]),
            ("ch4_oos_a", "CH-4 α 样本外(%/年)", a_oos[0])]:
        r = rep(key, actual)
        out[lab] = r
        if not r:
            continue
        print(f"  {lab:22s} 真实 {r['actual']:>8.3f}   安慰剂均值 {r['mean']:>8.3f} "
              f"(sd {r['sd']:.3f})  95 分位 {r['p95']:>8.3f}  最大 {r['max']:>8.3f}  "
              f"**p = {r['p_value']:.4f}**  z = {r['z']:+.2f}")

    # ---- DSR 对照 ----
    print("\n  DSR 对照（说明两种方法是否一致）：")
    seg_full = r_true
    seg_24 = r_true.iloc[-WIN24:]
    for lab, seg, N in [("全样本 N=50", seg_full, 50), ("全样本 N=334", seg_full, 334),
                        ("近24月 N=50", seg_24, 50), ("近24月 N=334", seg_24, 334)]:
        d = deflated_sharpe(seg, n_trials=N, sr_variance=0.0682)
        print(f"    {lab:16s} DSR={d['DSR']:.4f}  (SR0={d['SR0']:.3f})")
        out[f"DSR_{lab}"] = {"DSR": round(float(d["DSR"]), 4),
                             "SR0": round(float(d["SR0"]), 4)}

    with open(os.path.join(OUT, "placebo_test.json"), "w", encoding="utf-8") as fh:
        json.dump({"placebo": out,
                   "raw_distributions": {k: [round(float(x), 4) for x in v
                                             if np.isfinite(x)]
                                         for k, v in res.items()},
                   "true": {"full_sharpe": true_full, "oos_sharpe": true_oos,
                            "sh24": true_24,
                            "ch4_full_alpha": a_full[0], "ch4_full_t": a_full[1],
                            "ch4_oos_alpha": a_oos[0], "ch4_oos_t": a_oos[1]},
                   "B": B}, fh, ensure_ascii=False, indent=2)
    print(f"\n[已保存] {os.path.join(OUT,'placebo_test.json')}  总用时 {(time.time()-t0)/60:.1f} 分")


if __name__ == "__main__":
    main()
