# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
risk_attribution.py — 回撤归因与 beta 对冲可行性分析
================================================================================
背景
----
dm_diagnostics.py 已证明：Daniel & Moskowitz (2016) 的 panic 状态机制**不适用于**
本策略（γ_int 符号相反且 t=0.90 不显著）。原因是本策略不含动量因子，
而 DM 的机制专属于动量崩溃（熊市中过去的输家变成深度价外看涨期权）。

因此回撤必须另找来源。本脚本回答四个问题：

  Q1  最大回撤发生在什么时间？期间市场跌了多少？多少来自 beta、多少来自 alpha？
  Q2  滚动 beta 是否稳定？市场中性化后 alpha 还剩多少？
  Q3  用股指期货对冲 beta 能改善多少？考虑基差成本（A 股 IC/IM 常年贴水）后还剩多少？
  Q4  哪些变量能预测本策略的月度收益（用于设计真正有效的风险控制）？

关键概念
--------
· 多头的收益可分解为   r_p = α + β·r_m + ε
  对冲后   r_hedge = r_p − β·r_m ≈ α + ε
  对冲把「市场方向风险」剥离，只留下选股能力。
· 但 A 股股指期货长期贴水（basis<0），做空期货要付出基差成本：
    空头期货收益 = −r_spot + b ，  b = 年化基差
  故对冲组合净收益 = r_p − β·(r_spot) + β·b ， b<0 时为净支出。
  这一点必须计入，否则会高估对冲策略。
"""

import os
import sys
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")

RF = 0.02
PPY = 252


# ==============================================================================
def perf(r, name="", rf=RF):
    r = pd.Series(r).dropna()
    if len(r) < 20:
        return {"name": name}
    eq = (1 + r).cumprod()
    yrs = len(r) / PPY
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(PPY)
    sharpe = (r.mean() * PPY - rf) / vol if vol > 0 else np.nan
    dd = eq / eq.cummax() - 1
    mdd = float(dd.min())
    downside = r[r < 0].std() * np.sqrt(PPY)
    return {
        "name": name,
        "CAGR": round(float(cagr), 4),
        "AnnVol": round(float(vol), 4),
        "Sharpe": round(float(sharpe), 4),
        "MaxDD": round(mdd, 4),
        "Calmar": round(float(cagr / abs(mdd)), 3) if mdd < 0 else None,
        "Sortino": round(float((r.mean() * PPY - rf) / downside), 3) if downside > 0 else None,
        "WinRate": round(float((r > 0).mean()), 4),
        "Skew": round(float(r.skew()), 3),
        "n": int(len(r)),
        "start": str(r.index[0].date()),
        "end": str(r.index[-1].date()),
    }


def dd_table(r):
    """列出最大的 5 个回撤区间"""
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1
    troughs = []
    in_dd = False
    start = None
    for d, v in dd.items():
        if not in_dd and v < -1e-9:
            in_dd, start = True, d
        elif in_dd and v >= -1e-9:
            seg = dd.loc[start:d]
            troughs.append((start, seg.idxmin(), seg.min()))
            in_dd = False
    if in_dd:
        seg = dd.loc[start:]
        troughs.append((start, seg.idxmin(), seg.min()))
    troughs.sort(key=lambda t: t[2])
    rows = []
    for s, t, m in troughs[:6]:
        rows.append({
            "peak_date": str(s.date()), "trough_date": str(t.date()),
            "depth": round(float(m), 4),
            "days": int(r.loc[s:t].shape[0]),
        })
    return rows


# ==============================================================================
def main():
    print("=" * 78)
    print("回撤归因 / beta 对冲可行性 / 风险状态预测")
    print("=" * 78)

    mkt = pd.read_pickle(os.path.join(DATA, "market_daily.pkl"))
    with open(os.path.join(OUT, "returns_LSY.pkl"), "rb") as fh:
        R = pickle.load(fh)

    LONG = "多头组合(可交易)"
    LS = "多空组合(学术)"

    r_long = R[LONG]["net"].dropna()
    r_ls = R[LS]["net"].dropna()
    rm = mkt["mkt_vw"].reindex(r_long.index)
    rm_ew = mkt["mkt_ew"].reindex(r_long.index)

    result = {}

    # ---------------- Q1 全样本 beta 分解 ----------------
    print("\n" + "=" * 78)
    print("Q1  全样本 beta 分解与回撤归因")
    print("=" * 78)

    df = pd.concat({"p": r_long, "m": rm, "mew": rm_ew}, axis=1).dropna()
    X = sm.add_constant(df["m"])
    m_full = sm.OLS(df["p"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    a, b = m_full.params["const"], m_full.params["m"]
    resid = m_full.resid
    print(f"\n  多头 vs 市值加权市场（全样本 {len(df)} 日）")
    print(f"    α = {a*PPY:+.4%}/年 (t={m_full.tvalues['const']:.2f})   "
          f"β = {b:.3f} (t={m_full.tvalues['m']:.2f})")
    print(f"    市场部分 β·σ_m = {b*df['m'].std()*np.sqrt(PPY):.4f}  "
          f"特质部分 σ_ε = {resid.std()*np.sqrt(PPY):.4f}  "
          f"合计 = {df['p'].std()*np.sqrt(PPY):.4f}  相关系数 = {df['p'].corr(df['m']):.4f}")
    result["full_beta"] = {"alpha_ann": round(float(a * PPY), 4),
                           "beta": round(float(b), 4),
                           "beta_t": round(float(m_full.tvalues["m"]), 2),
                           "alpha_t": round(float(m_full.tvalues["const"]), 2),
                           "idio_vol_ann": round(float(resid.std() * np.sqrt(PPY)), 4),
                           "corr_mkt": round(float(df["p"].corr(df["m"])), 4),
                           "r2": round(float(m_full.rsquared), 4)}

    # 分样本
    for lab, s, e in [("样本内 2010-2022", "2010-01-01", "2022-12-31"),
                      ("样本外 2023-2026", "2023-01-01", "2026-12-31")]:
        d = df.loc[s:e]
        mm = sm.OLS(d["p"], sm.add_constant(d["m"])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        print(f"    {lab}:  α={mm.params['const']*PPY:+.4%}/年 (t={mm.tvalues['const']:.2f})  "
              f"β={mm.params['m']:.3f}  R²={mm.rsquared:.4f}")
        result[f"beta_{lab}"] = {"alpha_ann": round(float(mm.params["const"] * PPY), 4),
                                 "beta": round(float(mm.params["m"]), 4),
                                 "alpha_t": round(float(mm.tvalues["const"]), 2)}

    # ---- 最大回撤 ----
    print("\n  【多头组合最大的 6 个回撤区间】")
    dds = dd_table(r_long)
    for d in dds:
        s, t = pd.Timestamp(d["peak_date"]), pd.Timestamp(d["trough_date"])
        mr = (1 + rm.loc[s:t]).prod() - 1
        br = b * mr
        print(f"    {d['peak_date']} → {d['trough_date']}  "
              f"回撤 {d['depth']:+.2%}  ({d['days']} 日)   "
              f"同期市场 {mr:+.2%}   β贡献约 {br:+.2%}   残差 {d['depth']-br:+.2%}")
        d["mkt_ret"] = round(float(mr), 4)
        d["beta_part"] = round(float(br), 4)
        d["resid_part"] = round(float(d["depth"] - br), 4)
    result["long_drawdowns"] = dds

    print("\n  【多空组合最大的 4 个回撤区间】")
    dds_ls = dd_table(r_ls)
    for d in dds_ls[:4]:
        s, t = pd.Timestamp(d["peak_date"]), pd.Timestamp(d["trough_date"])
        mr = (1 + rm.loc[s:t]).prod() - 1
        print(f"    {d['peak_date']} → {d['trough_date']}  回撤 {d['depth']:+.2%} "
              f"({d['days']} 日)  同期市场 {mr:+.2%}")
        d["mkt_ret"] = round(float(mr), 4)
    result["ls_drawdowns"] = dds_ls

    # ---------------- Q2/Q3 滚动 beta 与对冲 ----------------
    print("\n" + "=" * 78)
    print("Q2/Q3  滚动 beta 对冲（含基差成本）")
    print("=" * 78)

    # 用滞后 1 日的 252 日滚动 beta，保证时点正确
    bet = (df["p"].rolling(252).cov(df["m"]) / df["m"].rolling(252).var()).shift(1)
    bet = bet.clip(0.0, 1.6)
    print(f"\n  滚动 beta（252日，滞后1日）:  均值 {bet.mean():.3f}  "
          f"标准差 {bet.std():.3f}  min {bet.min():.3f}  max {bet.max():.3f}  "
          f"P10 {bet.quantile(.1):.3f}  P90 {bet.quantile(.9):.3f}")
    result["rolling_beta"] = {"mean": round(float(bet.mean()), 3),
                              "std": round(float(bet.std()), 3),
                              "p10": round(float(bet.quantile(.1)), 3),
                              "p90": round(float(bet.quantile(.9)), 3)}

    print("\n  【对冲后业绩】空头期货收益 = −β·r_spot + β·b，b 为年化基差")
    print(f"    {'方案':28s} {'年化':>8s} {'波动':>8s} {'夏普':>7s} {'最大回撤':>9s} {'Calmar':>7s}")
    hedge_rows = []
    for basis in [0.0, -0.03, -0.06, -0.10]:
        rb = bet.reindex(df.index).fillna(0.0)
        r_h = df["p"] - rb * df["m"] + rb * (basis / PPY)
        p = perf(r_h.dropna(), f"对冲 β·mkt (基差 {basis:+.1%})")
        hedge_rows.append(p)
        print(f"    {'β对冲 基差'+format(basis, '+.1%'):28s} {p['CAGR']:>8.2%} {p['AnnVol']:>8.2%} "
              f"{p['Sharpe']:>7.3f} {p['MaxDD']:>9.2%} {str(p['Calmar']):>7s}")
    result["hedge_scenarios"] = hedge_rows

    # 固定 beta=1 的简化对冲（更贴近实际操作：按 1:1 名义本金对冲）
    print()
    hedge_rows2 = []
    for basis in [0.0, -0.03, -0.06, -0.10]:
        r_h = df["p"] - df["m"] + (basis / PPY)
        p = perf(r_h.dropna(), f"名义1:1对冲 (基差 {basis:+.1%})")
        hedge_rows2.append(p)
        print(f"    {'名义1:1 基差'+format(basis, '+.1%'):28s} {p['CAGR']:>8.2%} {p['AnnVol']:>8.2%} "
              f"{p['Sharpe']:>7.3f} {p['MaxDD']:>9.2%} {str(p['Calmar']):>7s}")
    result["hedge_1to1"] = hedge_rows2

    # ---------------- Q3b 降低 beta 目标 ----------------
    print("\n  【部分对冲：把组合 beta 压到目标值】")
    part_rows = []
    for btar in [0.9, 0.8, 0.7, 0.5]:
        rb = bet.reindex(df.index).fillna(0.0)
        k = (1 - btar / rb.replace(0, np.nan)).clip(0, 1)   # 对冲比例
        r_p = df["p"] - k * rb * df["m"]
        p = perf(r_p.dropna(), f"目标β={btar}")
        part_rows.append(p)
        print(f"    {'目标β='+str(btar):28s} {p['CAGR']:>8.2%} {p['AnnVol']:>8.2%} "
              f"{p['Sharpe']:>7.3f} {p['MaxDD']:>9.2%} {str(p['Calmar']):>7s}")
    result["partial_hedge"] = part_rows

    # ---------------- Q4 月度收益可预测性 ----------------
    print("\n" + "=" * 78)
    print("Q4  哪些状态变量能预测本策略的月度收益？")
    print("=" * 78)

    ml = r_long.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    mm = rm.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    md = mkt.resample("ME").last()

    feats = pd.DataFrame(index=ml.index)
    feats["mkt_24m"] = (1 + mm).rolling(24).apply(np.prod, raw=True) - 1       # DM 的 IB 变量（连续版）
    feats["mkt_vol126"] = mkt["mkt_vw"].rolling(126).std().resample("ME").last() * np.sqrt(PPY)
    feats["mkt_12m"] = (1 + mm).rolling(12).apply(np.prod, raw=True) - 1
    feats["mkt_1m"] = mm
    feats["own_12m"] = (1 + ml).rolling(12).apply(np.prod, raw=True) - 1
    feats["own_vol60"] = r_long.rolling(60).std().resample("ME").last() * np.sqrt(PPY)
    feats["own_dd"] = ((1 + r_long).cumprod() /
                       (1 + r_long).cumprod().cummax() - 1).resample("ME").last()
    feats["turnover_lvl"] = md["turn"].pct_change(12)
    feats["n_stock"] = md["n_stock"].pct_change(12)

    pred = feats.shift(1)      # 全部滞后一个月，只用可得信息
    for c in pred.columns:
        d = pd.concat({"y": ml, "x": pred[c]}, axis=1).dropna()
        if len(d) < 40:
            continue
        m = sm.OLS(d["y"], sm.add_constant(d["x"])).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        sig = "  <<<" if abs(m.tvalues["x"]) > 1.96 else ""
        print(f"    {c:14s}  β={m.params['x']:+.4f}  t={m.tvalues['x']:+.2f}  "
              f"R²={m.rsquared:.4f}  n={len(d)}{sig}")

    # 多元：DM 风格 + 自身状态
    X = pred[["mkt_24m", "mkt_vol126", "own_12m", "own_vol60", "own_dd"]].dropna()
    y = ml.reindex(X.index)
    m = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    print(f"\n    多元回归 (n={len(y)}, R²={m.rsquared:.4f})")
    for c in ["const"] + list(X.columns):
        print(f"      {c:14s} β={m.params[c]:+.4f}  t={m.tvalues[c]:+.2f}")
    result["predictors"] = {"r2": round(float(m.rsquared), 4), "n": int(len(y)),
                            "coef": {c: [round(float(m.params[c]), 4), round(float(m.tvalues[c]), 2)]
                                     for c in ["const"] + list(X.columns)}}

    # ---------------- 换手与成本 ----------------
    print("\n" + "=" * 78)
    print("换手与成本")
    print("=" * 78)
    for k in [LONG, LS]:
        g, n, c, tv = R[k]["gross"], R[k]["net"], R[k]["cost"], R[k]["turnover"]
        print(f"  {k}:  年化换手 {tv.mean()*PPY:.2%}  "
              f"年均成本拖累 {c.mean()*PPY:.4%}  "
              f"毛夏普 {perf(g)['Sharpe']:.3f} → 净夏普 {perf(n)['Sharpe']:.3f}  "
              f"差 {perf(g)['Sharpe']-perf(n)['Sharpe']:.3f}")
        result[f"cost_{k}"] = {
            "ann_turnover": round(float(tv.mean() * PPY), 4),
            "ann_cost_drag": round(float(c.mean() * PPY), 4),
            "gross_sharpe": perf(g)["Sharpe"], "net_sharpe": perf(n)["Sharpe"]}

    with open(os.path.join(OUT, "risk_attribution.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\n[已保存] {os.path.join(OUT,'risk_attribution.json')}")


if __name__ == "__main__":
    main()
