# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
neutral_experiment.py — 组合层面的风格暴露控制实验
================================================================================
动机（由 CH-3/CH-4 回归直接指出）
---------------------------------
CH-4 回归显示多头组合带有**巨大的风格暴露**：
    β_MKT = +0.924    β_SMB = −0.722    β_VMG = +0.108    R² = 0.927
即：它本质上是一个「大盘 + 低 beta」的市值暴露组合。

**关键认识**：因子层面的市值中性化**不能**约束组合层面的市值暴露。
中性化只是让「因子值与市值在截面上不相关」，但按中性化后的得分选出的**前 50 名**，
其市值分布仍可与全市场显著不同。要控制它，必须**在组合权重上加约束**。

方法
----
在候选池内对权重施加指数倾斜并解出倾斜系数：

    w_i ∝ base_i · exp(−λ·p_i)      s.t.  Σ w_i·p_i = target

其中 p_i 为个股的截面分位（0=最小/最低，1=最大/最高）。组合的平均分位关于 λ 单调，
用二分法求解。分别对**市值分位**与**EP 分位**做（先单独、后联合）。

判据
----
不是看收益高低，而是看：
  1. 目标暴露是否真的被打到（回归验证，而非假设）
  2. CH-4 alpha 是否保留
  3. **夏普是否改善** —— 这才是目标
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
                                    CFG, fast_rank_pct)
import strategy_v2 as v2

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
PPY, RF = 252, 0.02
IS = ("2010-01-01", "2022-12-31")
OOS = ("2023-01-01", "2026-12-31")


def tilt_weights(base, pct, lam):
    """w ∝ base · exp(−λ·pct)，返回归一化权重（未加上限时的形状）"""
    v = base * np.exp(-lam * pct)
    s = v.sum()
    return v / s if s > 0 else base


def make_weights_expo(score, eligible, size_pct, ep_pct, n=50, pool=200,
                      target_size=None, target_ep=None, rebalance_days=21,
                      max_weight=0.03):
    """带市值/EP 分位目标约束的组合构建"""
    rebal = score.index[::rebalance_days]
    rows = {}
    for dt in rebal:
        s = score.loc[dt].where(eligible.loc[dt]).dropna()
        if len(s) < 20:
            continue
        cand = s.nlargest(min(pool, len(s)))
        thr = cand.quantile(0.25)
        base = (cand - thr).clip(lower=0.0)
        if float(base.sum()) <= 0:
            base = pd.Series(1.0, index=cand.index)
        base = base / base.sum()
        bv = base.to_numpy(dtype=float)

        # 个股市值分位（在候选池内重算，避免依赖外部面板的时点）
        if dt in size_pct.index:
            sp = size_pct.loc[dt].reindex(cand.index)
        else:
            sp = None
        epv = None
        if target_ep is not None and dt in ep_pct.index:
            epv = ep_pct.loc[dt].reindex(cand.index)

        def solve(get_p, target, bv, pv):
            ok = np.isfinite(pv)
            if ok.sum() < 10:
                return bv
            bb, pp = bv[ok], pv[ok]
            def meanp(lam):
                w = bb * np.exp(-lam * pp)
                t = w.sum()
                return float(w @ pp / t) if t > 0 else np.nan
            if not np.isfinite(meanp(0.0)) or abs(meanp(0.0) - target) < 1e-4:
                return bv
            lo, hi = -50.0, 50.0
            flo, fhi = meanp(lo), meanp(hi)
            if not (np.isfinite(flo) and np.isfinite(fhi)):
                return bv
            # meanp 关于 λ 单调递减
            if target >= flo:
                lam = lo
            elif target <= fhi:
                lam = hi
            else:
                for _ in range(70):
                    mid = 0.5 * (lo + hi)
                    if meanp(mid) > target:
                        lo = mid
                    else:
                        hi = mid
                lam = 0.5 * (lo + hi)
            out = bv.copy()
            out[ok] = bb * np.exp(-lam * pp)
            t = out.sum()
            return out / t if t > 0 else bv

        wv = bv.copy()
        if target_size is not None and sp is not None:
            wv = solve(None, target_size, wv, sp.to_numpy(dtype=float))
        if target_ep is not None and epv is not None:
            wv = solve(None, target_ep, wv, epv.to_numpy(dtype=float))
        w = pd.Series(wv, index=cand.index).nlargest(min(n, len(cand)))
        w = v2._apply_cap(w, max_weight)
        rows[dt] = w
    W = pd.DataFrame(rows).T.fillna(0.0)
    return W.reindex(columns=score.columns, fill_value=0.0)


def main():
    print("=" * 92)
    print("组合层面风格暴露控制实验")
    print("=" * 92)

    df = load_panel()
    P = Panels(df)
    del df
    eligible, _, _ = build_universe(P, CFG)
    need = sorted(set(v2.FACTOR_SETS["F17"] + v2.EXTRA_NAMES + v2.CONTROL_NAMES))
    v2.build_prep(P, eligible, need)
    score = v2.composite(v2.FACTOR_SETS["F17"])

    # 市值分位：每个调仓日在**全股票池**内计算（与 CH-3 的 SMB 口径一致）
    print("\n[1] 构造市值/EP 分位面板")
    rebal = score.index[::21]
    size_pct = fast_rank_pct(P.float_mktcap.where(eligible)).reindex(rebal)
    expo = pd.read_pickle(os.path.join(DATA, "ch3_expo.pkl"))
    ep_rows = {}
    for dt in rebal:
        src = [d for d in expo if d <= dt]
        if not src:
            continue
        e = expo[src[-1]]
        ep_rows[dt] = e["EP"].rank(pct=True)
    ep_pct = pd.DataFrame(ep_rows).T.reindex(columns=P.symbols)
    print(f"    调仓日 {len(rebal)}  市值分位 {size_pct.shape}  EP 分位 {ep_pct.shape}")

    print("\n[2] 组合层面约束实验")
    VARIANTS = {
        "无约束(基准)": dict(),
        "市值分位→0.50": dict(target_size=0.50),
        "市值分位→0.60": dict(target_size=0.60),
        "市值分位→0.40": dict(target_size=0.40),
        "市值0.50+EP0.50": dict(target_size=0.50, target_ep=0.50),
        "EP分位→0.50": dict(target_ep=0.50),
    }
    ch = pd.read_pickle(os.path.join(DATA, "ch3.pkl")).shift(1)
    res = {}
    print(f"  {'方案':18s} | {'IS夏普':>7s} {'OOS夏普':>8s} {'OOS年化':>8s} {'OOS回撤':>8s} "
          f"{'Calmar':>7s} | {'CH-4 α(年%)':>12s} {'t':>6s} {'β_SMB':>7s} {'β_VMG':>7s} {'R²':>6s}")
    print("  " + "-" * 116)
    for lab, kw in VARIANTS.items():
        W = make_weights_expo(score, eligible, size_pct, ep_pct, n=50, pool=200,
                              rebalance_days=21, max_weight=0.03, **kw)
        r = backtest(P, W, CFG, lab)["net"]
        del W
        def met(seg):
            seg = seg.dropna()
            if len(seg) < 20:
                return {}
            eq = (1 + seg).cumprod()
            cagr = eq.iloc[-1] ** (PPY / len(seg)) - 1
            vol = seg.std() * np.sqrt(PPY)
            dd = float((eq / eq.cummax() - 1).min())
            return {"Sharpe": (seg.mean() * PPY - RF) / vol, "CAGR": cagr,
                    "MaxDD": dd, "Calmar": cagr / abs(dd) if dd < 0 else np.nan}
        i = met(r[(r.index >= IS[0]) & (r.index <= IS[1])])
        o = met(r[(r.index >= OOS[0]) & (r.index <= OOS[1])])
        # CH-4 回归（样本外）
        mm = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        sub = mm[(mm.index >= OOS[0]) & (mm.index <= OOS[1])]
        d = pd.concat({"y": sub, "MKT": ch["MKT"], "SMB": ch["SMB"],
                       "VMG": ch["VMG"], "PMO": ch["PMO"]}, axis=1).dropna()
        a_t, bs, r2 = np.nan, {}, np.nan
        if len(d) > 24:
            m = sm.OLS(d["y"], sm.add_constant(d[["MKT", "SMB", "VMG", "PMO"]])
                       ).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
            a_t = float(m.tvalues["const"])
            bs = {c: float(m.params[c]) for c in ["MKT", "SMB", "VMG", "PMO"]}
            r2 = float(m.rsquared)
        res[lab] = {"IS": i, "OOS": o, "ch4_oos": {"alpha_m": float(m.params["const"]) if len(d) > 24 else None,
                                                  "t": a_t, "betas": bs, "r2": r2}}
        print(f"  {lab:18s} | {i.get('Sharpe', float('nan')):>7.3f} "
              f"{o.get('Sharpe', float('nan')):>8.3f} {o.get('CAGR', float('nan')):>8.2%} "
              f"{o.get('MaxDD', float('nan')):>8.2%} {o.get('Calmar', float('nan')):>7.3f} | "
              f"{(float(m.params['const'])*12*100 if len(d) > 24 else float('nan')):>12.2f} "
              f"{a_t:>6.2f} {bs.get('SMB', float('nan')):>7.3f} "
              f"{bs.get('VMG', float('nan')):>7.3f} {r2:>6.3f}", flush=True)

    with open(os.path.join(OUT, "neutral_experiment.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n[已保存] {os.path.join(OUT,'neutral_experiment.json')}")


if __name__ == "__main__":
    main()
