# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
dm_diagnostics.py — Daniel & Moskowitz (2016, JFE) "Momentum Crashes" 方法论诊断
================================================================================
目的：把 DM 论文的检验原样搬到我们的 A 股截面上，回答三个问题

  Q1  「panic 状态」能否预测我们策略的条件均值 μ？
      DM 的预测回归（Table 5）：
        R_t = γ0 + γB·IB,t-1 + γσ2·σ²m,t-1 + γint·(IB,t-1 · σ²m,t-1) + ε_t
      其中 IB,t-1 = 1 若过去 24 个月市场累计收益为负
           σ²m,t-1 = 截至 t-1 末 126 个交易日的市场日收益方差

  Q2  我们的组合是否具备 DM 所说的「期权式」payoff？
      Henriksson-Merton (1981) 上下行 beta 不对称检验：
        R_t = [α0+αB·IB] + [β0 + IB·(βB + ĨU·βB,U)]·Rm,t + ε_t
      ĨU,t = 1 若当月市场收益为正

  Q3  DM 的最优权重公式能否真的改善业绩？
      w*_t = (1/2λ)·μ_t / σ²_t                                    （DM Eq.5）
      注意：μ_t<0 时 w*<0，即 panic 状态下**反手做空**。
      这是与我们此前失败的「波动率目标化」最本质的区别——
      波动率目标化只是 Eq.5 在「夏普比恒为常数」下的特例。

文献出处
  Daniel, K. & Moskowitz, T. (2016). Momentum crashes.
  Journal of Financial Economics 122(2), 221-247.
  Appendix C 给出 Eq.5 的推导；Appendix D 给出 GJR-GARCH 波动率预测。
  NBER WP 20439 全文已下载至 papers/nber20439.txt。
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
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

# DM 用 CRSP 市值加权指数；A 股我们同时构造流通市值加权与等权两条市场序列做对照
MKTS = ["mkt_vw", "mkt_ew"]


# ==============================================================================
# 1  市场序列：从面板就地构造（避免依赖外部指数接口）
# ==============================================================================
def build_market():
    """构造日频市场序列并缓存。

    流通市值加权收益 r_vw,t = Σ w_i,t-1 r_i,t / Σ w_i,t-1，w = 流通市值。
    权重用 t-1 期市值（时点正确，避免用当日市值加当日收益的前视偏差）。
    """
    cache = os.path.join(DATA, "market_daily.pkl")
    if os.path.exists(cache):
        m = pd.read_pickle(cache)
        print(f"  [缓存] market_daily.pkl  {m.shape}  {m.index[0].date()} ~ {m.index[-1].date()}")
        return m

    from ashare_factor_strategy import Panels, load_panel
    t0 = time.time()
    print("  加载面板并构造宽表 ...", flush=True)
    df = load_panel()
    P = Panels(df)
    del df

    traded = P.traded
    r = P.ret.where(traded)
    w = P.float_mktcap.shift(1).where(traded)      # 滞后一期市值做权重

    den = w.sum(axis=1, min_count=1)
    mkt_vw = (w * r).sum(axis=1, min_count=1) / den
    mkt_ew = r.mean(axis=1)

    out = pd.DataFrame({
        "mkt_vw": mkt_vw,
        "mkt_ew": mkt_ew,
        "amt": P.amount.where(traded).sum(axis=1),
        "turn": (w * P.turnover.where(traded)).sum(axis=1, min_count=1) / den,
        "n_stock": traded.sum(axis=1),
    })
    out.index.name = "date"
    out.to_pickle(cache)
    print(f"  [新建] market_daily.pkl  {out.shape}  {time.time()-t0:.0f}s")
    return out


# ==============================================================================
# 2  收益序列工具
# ==============================================================================
def to_monthly(r):
    """日收益 -> 月末收益（复利）"""
    return r.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)


def max_dd(equity):
    dd = equity / equity.cummax() - 1.0
    return float(dd.min()), dd


def perf(r, rf_annual=0.02, periods_per_year=252):
    r = r.dropna()
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std() * np.sqrt(periods_per_year)
    sharpe = (r.mean() * periods_per_year - rf_annual) / vol if vol > 0 else np.nan
    mdd, _ = max_dd(eq)
    return {
        "CAGR": round(float(cagr), 4),
        "AnnVol": round(float(vol), 4),
        "Sharpe": round(float(sharpe), 4),
        "MaxDD": round(float(mdd), 4),
        "Calmar": round(float(cagr / abs(mdd)), 3) if mdd < 0 else None,
        "n_days": int(len(r)),
    }


def hm_regress(y, rm, ib, iu, lags=5, label=""):
    """Henriksson-Merton 上下行 beta 不对称检验（DM Eq.2 / Table 5）

    R_t = [α0 + αB·IB] + [β0 + IB·(βB + ĨU·βB,U)]·Rm,t + ε_t
    展开后的回归元：1, IB, Rm, IB·Rm, IB·IU·Rm
    """
    X = pd.DataFrame({
        "const": 1.0,
        "IB": ib.astype(float),
        "Rm": rm,
        "IB_Rm": ib.astype(float) * rm,
        "IB_IU_Rm": ib.astype(float) * iu.astype(float) * rm,
    }).dropna()
    y = y.reindex(X.index).dropna()
    X = X.reindex(y.index)
    if len(y) < 30:
        return None
    m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    b = m.params
    down_beta_bear = b["Rm"] + b["IB_Rm"]                      # 熊市下行 beta
    up_beta_bear = down_beta_bear + b["IB_IU_Rm"]              # 熊市上行 beta
    return {
        "label": label,
        "n": int(len(y)),
        "alpha0_ann": round(float(b["const"]) * 12, 4),
        "alphaB_ann": round(float(b["IB"]) * 12, 4),
        "t_alphaB": round(float(m.tvalues["IB"]), 2),
        "beta0": round(float(b["Rm"]), 3),
        "t_beta0": round(float(m.tvalues["Rm"]), 2),
        "betaB": round(float(b["IB_Rm"]), 3),
        "t_betaB": round(float(m.tvalues["IB_Rm"]), 2),
        "betaBU": round(float(b["IB_IU_Rm"]), 3),
        "t_betaBU": round(float(m.tvalues["IB_IU_Rm"]), 2),
        "down_beta_bear": round(float(down_beta_bear), 3),
        "up_beta_bear": round(float(up_beta_bear), 3),
        "up_minus_down_bear": round(float(b["IB_IU_Rm"]), 3),
        "r2": round(float(m.rsquared), 4),
    }


def dm_predict_regress(y, ib, sig2, lags=5, label=""):
    """DM Table 5 预测回归

    R_t = γ0 + γB·IB + γσ2·σ²m + γint·IB·σ²m + ε
    注意 t 统计量对 σ² 的量纲不敏感（缩放回归元不改变 t 值），
    因此显著性结论与单位选择无关；经济幅度另按分位数差单独报告。
    """
    X = pd.DataFrame({
        "const": 1.0, "IB": ib.astype(float),
        "sig2": sig2, "IB_sig2": ib.astype(float) * sig2,
    }).dropna()
    y = y.reindex(X.index).dropna()
    X = X.reindex(y.index)
    if len(y) < 30:
        return None
    # (1)(2)(3)(4) 四列，对应 DM Table 5 的逐步加入
    cols = {
        "col1_IBonly": ["const", "IB"],
        "col2_sig2only": ["const", "sig2"],
        "col3_both": ["const", "IB", "sig2"],
        "col4_interact": ["const", "IB", "sig2", "IB_sig2"],
    }
    res = {"label": label, "n": int(len(y))}
    for k, cc in cols.items():
        m = sm.OLS(y, X[cc]).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        res[k] = {f"{p}": [round(float(m.params[p]), 6), round(float(m.tvalues[p]), 2)]
                  for p in cc}
        res[k]["r2"] = round(float(m.rsquared), 4)
    # 全样本均值，用于解释 IBσ2 的经济幅度
    res["sig2_mean_bear"] = float(sig2[ib == 1].mean()) if (ib == 1).any() else None
    res["n_bear_months"] = int((ib == 1).sum())
    res["n_months"] = int(len(ib))
    return res


# ==============================================================================
# 3  主流程
# ==============================================================================
def main():
    print("=" * 78)
    print("Daniel & Moskowitz (2016) Momentum Crashes —— 方法论诊断")
    print("=" * 78)

    print("\n[1] 市场序列")
    mkt = build_market()
    mkt = mkt.dropna(subset=MKTS)
    print(f"      交易日 {len(mkt)}  {mkt.index[0].date()} ~ {mkt.index[-1].date()}")

    print("\n[2] 策略日收益")
    with open(os.path.join(OUT, "returns_LSY.pkl"), "rb") as fh:
        R = pickle.load(fh)
    strat = {k: v["net"].dropna() for k, v in R.items()}
    for k, v in strat.items():
        print(f"      {k}: {len(v)} 日  {v.index[0].date()} ~ {v.index[-1].date()}")

    # ---- 月末对齐 ----
    print("\n[3] 构造 DM panic 状态变量")
    mmkt = mkt[MKTS].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    daily_mkt = mkt[MKTS]

    months = mmkt.index

    def make_state(mcol):
        """返回 (IB, sig2) 月度序列，均只用截至该月末的信息"""
        mret = mmkt[mcol]
        # 过去 24 个月累计收益（含当月）→ 该月末状态
        cum24 = (1 + mret).rolling(24).apply(np.prod, raw=True) - 1
        IB = (cum24 < 0).astype(float)
        IB[cum24.isna()] = np.nan
        # 截至该月末 126 个交易日的日收益方差（年化，便于阅读）
        d = daily_mkt[mcol]
        sig2 = d.rolling(126).var().resample("ME").last() * 252.0
        return IB, sig2

    out = {"market_used": {}, "panels": {}}

    for mcol in MKTS:
        IB, sig2 = make_state(mcol)
        print(f"\n  --- 市场序列 = {mcol} ---")
        print(f"      熊市月占比 {float(IB.mean()):.1%}  ({int(IB.sum())}/{int(IB.notna().sum())})")
        print(f"      σ² 年化 中位数 {float(sig2.median()):.4f}  "
              f"熊市均值 {float(sig2[IB == 1].mean()):.4f}  非熊市均值 {float(sig2[IB == 0].mean()):.4f}")
        out["market_used"][mcol] = {
            "bear_month_pct": round(float(IB.mean()), 4),
            "n_bear": int(IB.sum()),
            "sig2_median_ann": round(float(sig2.median()), 4),
            "sig2_mean_bear": round(float(sig2[IB == 1].mean()), 4),
            "sig2_mean_nonbear": round(float(sig2[IB == 0].mean()), 4),
        }

        # IB 与 sig2 都滞后一个月：预测第 t 月收益，用 t-1 月末的状态
        IB_lag = IB.shift(1)
        sig2_lag = sig2.shift(1)
        rm = mmkt[mcol]
        iu = (rm > 0).astype(float)     # 当月市场上行指示（同期）

        for sname, sr in strat.items():
            ms = to_monthly(sr)
            idx = ms.index.intersection(IB_lag.dropna().index)
            ms = ms.reindex(idx)
            ib = IB_lag.reindex(idx)
            s2 = sig2_lag.reindex(idx)
            rmv = rm.reindex(idx)
            iuv = iu.reindex(idx)

            key = f"{mcol}|{sname}"
            pr = dm_predict_regress(ms, ib, s2, label=key)
            hr = hm_regress(ms, rmv, ib, iuv, label=key)
            out["panels"][key] = {"dm_table5": pr, "henriksson_merton": hr}

            print(f"\n    [{sname}]  DM Table 5 预测回归 (n={pr['n']} 月)")
            for ck in ["col1_IBonly", "col2_sig2only", "col3_both", "col4_interact"]:
                cc = pr[ck]
                terms = "  ".join(
                    f"{p}={v[0]}(t={v[1]})" for p, v in cc.items() if p != "r2")
                print(f"      {ck:16s} R²={cc['r2']:.4f}  {terms}")
            if hr:
                print(f"    [{sname}]  Henriksson-Merton 上下行 beta")
                print(f"      β0={hr['beta0']}(t={hr['t_beta0']})  "
                      f"βB={hr['betaB']}(t={hr['t_betaB']})  "
                      f"βB,U={hr['betaBU']}(t={hr['t_betaBU']})")
                print(f"      熊市: 下行β={hr['down_beta_bear']}  上行β={hr['up_beta_bear']}  "
                      f"差={hr['up_minus_down_bear']}")

    # ---- 保存 ----
    with open(os.path.join(OUT, "dm_diagnostics.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n[已保存] {os.path.join(OUT,'dm_diagnostics.json')}")


if __name__ == "__main__":
    main()
