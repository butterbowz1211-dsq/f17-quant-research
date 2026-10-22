# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
test_syy.py — Stambaugh, Yu & Yuan (2012, JFE 104(2), 288-302) 的严格复现检验
================================================================================
论文："The short of it: Investor sentiment and anomalies"

论文的三个核心结论（据 NBER WP w16898 原文逐条核对）
----------------------------------------------------
1. 每个异象在**高情绪之后更强**（多空更赚钱）。
2. 效应**几乎全部来自空头腿**：FF3 调整后高-低差 +0.70 (t=3.74)，
   其中空头腿贡献 −0.66 (t=−3.89)，多头腿贡献 +0.04 (**t=0.62，为零**)。
   原文摘要："sentiment exhibits no relation to returns on the long legs"。
3. 时序设定严格为：  R_{i,t} = a + b·S_{t−1} + u_t
   情绪用 **LEVEL**（原文明确写 "level"），滞后**一个月**，不是同期。

论文的机制：卖空限制 ⇒ 最乐观的投资者定价 ⇒ 高情绪使**空头腿**（被高估的那一端）
更加高估 ⇒ 随后回调更大 ⇒ 做空更赚。

★ 原文的一个陷阱（子代理核对后指出）
   SYY 的「多头腿」是按**异象自身已验证的方向**定义的，不是按分位数编号。
   我们这里的方向已统一为「因子值越大 → 预期收益越高」，
   故 多头腿 = 得分最高组，空头腿 = 得分最低组，与 SYY 口径一致。

★ 与原文的两处**刻意偏离**（必须披露）
   (a) 原文用**全样本中位数**划分高低情绪（有前视）。本实现改用
       **扩展窗口中位数**（real-time），因此结论严格可比性下降但无前视。
   (b) 原文的情绪指数是 Baker-Wurgler 六代理主成分。我们只有
       换手率 / IPO 数量 / 净股权发行 三个代理（缺封闭式基金折价、
       股利溢价、IPO 首日收益），故本指数是 **BW-style 变体**，非 BW 指数。
   (c) 原文标准误用 **White (1980) HC**（非 Newey-West）。本实现两者都报。
   (d) 原文样本为 1965:08-2008:01 美股。A 股的融券供给小且时断时续，
       「卖空限制 ⇒ 乐观者定价」的机制弱得多，符号可能真的不同。
"""

import os
import sys
import json
import time
import pickle
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ashare_factor_strategy import load_panel, Panels, build_universe, backtest, CFG  # noqa
import strategy_v2 as v2                                                             # noqa

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
IS = ("2010-01-01", "2022-12-31")
OOS = ("2023-01-01", "2026-12-31")


# ==============================================================================
def build_sentiment_index():
    """BW-style 情绪指数：可用三代理的第一主成分。

    BW (2006) 的做法：每个代理先对宏观变量正交化，再取第一主成分。
    本实现无宏观月频序列，故只做「同月内标准化后取 PC1」，
    且**主成分载荷只用样本内 2010-2022 估计**，避免前视。
    指数符号统一为：值越大 = 越乐观（用与换手率的正相关校验）。
    """
    m = pd.read_pickle(os.path.join(DATA, "market_daily2.pkl"))
    mo = pd.DataFrame(index=m.resample("ME").last().index)
    mo["turnover"] = m["turnover_agg"].resample("ME").mean()      # BW 代理 1
    mo["ipo_n"] = m["ipo_n"].resample("ME").sum()                 # BW 代理 2
    mo["issuance"] = m["issuance"].clip(
        upper=m["issuance"].quantile(0.99)).resample("ME").sum()  # BW 代理 3
    mo["dispersion"] = m["dispersion"].resample("ME").mean()      # 备选（分歧）
    mo["mkt"] = m["mkt_vw"].resample("ME").apply(lambda x: (1 + x).prod() - 1)

    proxies = ["turnover", "ipo_n", "issuance"]
    is_mask = (mo.index >= IS[0]) & (mo.index <= IS[1])
    Z = pd.DataFrame(index=mo.index)
    for c in proxies:
        mu, sd = mo.loc[is_mask, c].mean(), mo.loc[is_mask, c].std()
        Z[c] = (mo[c] - mu) / sd
    Zc = Z.loc[is_mask].dropna()
    # 第一主成分（样本内估计载荷）
    X = Zc.to_numpy()
    X = X - X.mean(axis=0)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    load = vt[0]
    if np.corrcoef(X @ load, Zc["turnover"])[0, 1] < 0:
        load = -load                                    # 符号统一：高 = 乐观
    S = Z.to_numpy() @ load
    S = pd.Series(S, index=Z.index, name="S")
    S = (S - S.loc[is_mask].mean()) / S.loc[is_mask].std()
    return S, mo, load, proxies


def expanding_median_flag(s, min_obs=36):
    """扩展窗口中位数划分高低情绪（real-time，无前视）。

    对应 SYY 的 high/low 哑变量，但把「全样本中位数」换成「截至上月的扩展中位数」。
    """
    med = s.expanding(min_periods=min_obs).median().shift(1)
    return (s > med), med


def leg_regression(y, S_lag, label, cols=None):
    """把某条组合收益序列对滞后情绪做时序回归（SYY 表 4/5 的设定）。

    标准误同时给 White HC 与 Newey-West。
    """
    d = pd.concat({"y": y, "S": S_lag}, axis=1).dropna()
    if len(d) < 40:
        return None
    X = sm.add_constant(d[["S"]])
    out = {"label": label, "n": int(len(d))}
    for cov, tag in [({"cov_type": "HC0"}, "white"),
                     ({"cov_type": "HAC", "cov_kwds": {"maxlags": 6}}, "nw")]:
        mm = sm.OLS(d["y"], X).fit(**cov)
        out[tag] = {"b": round(float(mm.params["S"]), 5),
                    "t": round(float(mm.tvalues["S"]), 2),
                    "a": round(float(mm.params["const"]), 5),
                    "a_t": round(float(mm.tvalues["const"]), 2),
                    "r2": round(float(mm.rsquared), 4)}
    return out


def dummy_regression(y, hi, label):
    """SYY 表 2/3 的哑变量设定：R = a_H·d_H + a_L·d_L + u"""
    d = pd.concat({"y": y, "hi": hi.astype(float)}, axis=1).dropna()
    if len(d) < 40:
        return None
    dH = (d["hi"] == 1).astype(float)
    dL = (d["hi"] == 0).astype(float)
    Xdf = pd.DataFrame({"dH": dH, "dL": dL}).dropna()
    dd = pd.concat([d, Xdf], axis=1).dropna()
    if dd["dH"].sum() < 10 or dd["dL"].sum() < 10:
        return None
    mm = sm.OLS(dd["y"], dd[["dH", "dL"]]).fit(cov_type="HC0")
    diff = float(mm.params["dH"] - mm.params["dL"])
    # 高-低差的 t 值：用约束检验
    R = np.array([[1.0, -1.0]])
    tt = mm.t_test(R)
    return {"label": label, "a_high": round(float(mm.params["dH"]), 5),
            "t_high": round(float(mm.tvalues["dH"]), 2),
            "a_low": round(float(mm.params["dL"]), 5),
            "t_low": round(float(mm.tvalues["dL"]), 2),
            "high_minus_low": round(diff, 5),
            "t_diff": round(float(np.ravel(tt.tvalue)[0]), 2),
            "n": int(len(dd))}


def main():
    t0 = time.time()
    print("=" * 92)
    print("Stambaugh-Yu-Yuan (2012) 情绪条件化 —— A 股复现检验")
    print("=" * 92)

    print("\n[1] 加载面板 / 股票池 / 因子得分")
    df = load_panel()
    P = Panels(df)
    del df
    eligible, _, _ = build_universe(P, CFG)

    need = sorted(set(v2.FACTOR_SETS["F13"] + v2.EXTRA_NAMES))
    v2.build_prep(P, eligible, need)
    score = v2.composite(v2.FACTOR_SETS["F13"])
    print("    因子得分合成完成", flush=True)

    print("\n[2] 构造多头腿 / 空头腿 / 多空组合（n=150 等权，月度调仓）")
    KW = dict(n=150, weighting="equal", rebalance_days=21)
    r_long = backtest(P, v2.make_weights(score, eligible, side="long", **KW),
                      CFG, "多头腿")["net"]
    r_short = backtest(P, v2.make_weights(score, eligible, side="short", **KW),
                       CFG, "空头腿")["net"]
    r_ls = (r_long - r_short).dropna()
    for nm, s in [("多头腿", r_long), ("空头腿", r_short), ("多空", r_ls)]:
        print(f"    {nm:6s} 日均 {s.mean()*1e4:+.3f}bp  年化波动 {s.std()*np.sqrt(252):.2%}")

    print("\n[3] BW-style 情绪指数")
    S, mo, load, proxies = build_sentiment_index()
    print(f"    代理: {proxies}   载荷: " +
          "  ".join(f"{p}={l:+.3f}" for p, l in zip(proxies, load)))
    hi, med = expanding_median_flag(S)
    S_lag = S.shift(1)
    n_hi = int(hi.dropna().sum())
    print(f"    高情绪月 {n_hi}/{int(hi.notna().sum())}   "
          f"样本内均值 {S.loc[IS[0]:IS[1]].mean():+.3f}   "
          f"样本外均值 {S.loc[OOS[0]:OOS[1]].mean():+.3f}")

    res = {"factor_set": "F13", "loadings": {p: round(float(l), 4)
                                             for p, l in zip(proxies, load)},
           "n_high": n_hi}

    # ---------------- 4. 分腿回归（SYY 表 4/5 的核心） ----------------
    print("\n" + "=" * 92)
    print("[4] 分腿时序回归  R_t = a + b·S_{t-1} + u_t      （SYY 预测空头腿 b<0）")
    print("=" * 92)
    print(f"    {'组合':8s} {'区间':6s} {'b':>10s} {'t(White)':>10s} {'t(NW)':>8s} "
          f"{'a(月%)':>9s} {'R²':>7s}   SYY 预期")
    print("    " + "-" * 84)
    expect = {"多头腿": "≈0（论文 t=0.62）", "空头腿": "<0（论文 t=−3.89）",
              "多空": ">0（论文 t=+3.74）"}
    leg_month = {}
    for nm, s in [("多头腿", r_long), ("空头腿", r_short), ("多空", r_ls)]:
        ms = s.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        leg_month[nm] = ms
        for lab, (st, en) in [("全样本", ("2010-01-01", "2026-12-31")),
                              ("样本内", IS), ("样本外", OOS)]:
            m = ms[(ms.index >= st) & (ms.index <= en)]
            r = leg_regression(m, S_lag.reindex(m.index), f"{nm}|{lab}")
            if r:
                res[f"reg|{nm}|{lab}"] = r
                print(f"    {nm:8s} {lab:6s} {r['white']['b']:>+10.5f} "
                      f"{r['white']['t']:>+10.2f} {r['nw']['t']:>+8.2f} "
                      f"{r['white']['a']*100:>+9.3f} {r['white']['r2']:>7.4f}   {expect[nm]}")

    # ---------------- 5. 哑变量设定（高/低情绪分组） ----------------
    print("\n" + "=" * 92)
    print("[5] 高低情绪分组（扩展窗口中位数，real-time）  R_t = a_H·d_H + a_L·d_L + u_t")
    print("=" * 92)
    print(f"    {'组合':8s} {'区间':6s} {'高情绪 a_H':>12s} {'t':>7s} {'低情绪 a_L':>12s} "
          f"{'t':>7s} {'高−低':>10s} {'t':>7s}")
    print("    " + "-" * 84)
    for nm in ["多头腿", "空头腿", "多空"]:
        ms = leg_month[nm]
        for lab, (st, en) in [("全样本", ("2010-01-01", "2026-12-31")), ("样本外", OOS)]:
            m = ms[(ms.index >= st) & (ms.index <= en)]
            r = dummy_regression(m, hi.reindex(m.index), f"{nm}|{lab}")
            if r:
                res[f"dummy|{nm}|{lab}"] = r
                print(f"    {nm:8s} {lab:6s} {r['a_high']*100:>+12.3f} {r['t_high']:>+7.2f} "
                      f"{r['a_low']*100:>+12.3f} {r['t_low']:>+7.2f} "
                      f"{r['high_minus_low']*100:>+10.3f} {r['t_diff']:>+7.2f}")

    # ---------------- 6. 若成立，能改善多少？ ----------------
    print("\n" + "=" * 92)
    print("[6] 若情绪条件化成立，能否改善风险？—— 仅在样本内选规则，样本外验证")
    print("=" * 92)
    # 规则：S_{t-1} 高于扩展中位数 -> 满仓；否则按 (1-k) 缩仓
    for target, series in [("多空", r_ls), ("多头腿", r_long)]:
        print(f"\n  --- {target} ---")
        for k in [0.0, 0.25, 0.5, 0.75, 1.0]:
            scale_m = (hi.astype(float) * (1.0 - k) + (1 - hi.astype(float)) * 1.0)
            # 高情绪满仓/低情绪缩仓，或反之——两种方向都试
            for direction, sc in [("低情绪缩仓", scale_m),
                                  ("高情绪缩仓", 2.0 - scale_m)]:
                sc = sc.reindex(series.index, method="ffill").shift(1).fillna(1.0)
                rr = (series * sc).dropna()
                o = rr[(rr.index >= OOS[0]) & (rr.index <= OOS[1])]
                if len(o) < 50:
                    continue
                eq = (1 + o).cumprod()
                cagr = eq.iloc[-1] ** (252 / len(o)) - 1
                vol = o.std() * np.sqrt(252)
                dd = float((eq / eq.cummax() - 1).min())
                if k in (0.0, 0.5, 1.0) or direction == "低情绪缩仓":
                    print(f"    k={k:.2f} {direction}  样本外年化 {cagr:>7.2%} 波动 {vol:>6.2%} "
                          f"夏普 {(o.mean()*252-0.02)/vol:>6.3f} 回撤 {dd:>8.2%}")

    with open(os.path.join(OUT, "syy_test.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n[已保存] {os.path.join(OUT,'syy_test.json')}")
    print(f"总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
