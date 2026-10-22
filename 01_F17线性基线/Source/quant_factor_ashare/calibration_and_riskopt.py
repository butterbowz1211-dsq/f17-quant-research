# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
calibration_and_riskopt.py — 目标可达性标定 + 风险模型优化
================================================================================
两个目的
--------
A. **标定**：Sharpe > 1.2 对 A 股「多头」到底是否可达？
   若主流指数与公募长期水平都远低于 1.2，那么该目标本身不现实，
   继续优化只是在追一个不可能的靶子。这一步决定后续是否值得投入。

B. **风险模型优化**（此前完全没试过的杠杆）：
   一直用等权。组合波动 19%、市场 23.7%，这 4.7pp 的差距全靠选股得来，
   **完全没有利用协方差信息**。金融学里降低组合波动最标准的手段就是
   用风险模型优化权重：
        max_w  w'μ − λ·w'Σw     s.t. Σw = 1, 0 ≤ w ≤ cap
   其中 μ 是因子得分（alpha 代理），Σ 是收缩估计的协方差矩阵。
   若能在大致保留 alpha 的同时把波动降下来，夏普就能上台阶。

   实现要点：
   · Σ 用**滚动 252 日**日收益估计，并对每个调仓日重新估计（时点正确）；
   · 用 Ledoit-Wolf 式收缩：Σ = (1−δ)·S + δ·diag(S)，δ=0.2，避免奇异；
   · 求解用 scipy SLSQP，约束 Σw=1、0 ≤ w ≤ cap。
"""

import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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


def calibrate(P, eligible, rb):
    """标定：主流宽基指数与自建市场组合的夏普"""
    print("\n" + "=" * 96)
    print("A  目标可达性标定")
    print("=" * 96)
    out = {}
    # 自建：流通市值加权 / 等权全市场（与策略同一股票池口径）
    for nm, W in [("流通市值加权市场", None),
                  ("等权全市场", v2.universe_weights(eligible))]:
        if W is None:
            mc = P.float_mktcap.where(eligible)
            W = mc.div(mc.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        r = backtest(P, W, CFG, nm)["net"]
        for lab, (s, e) in [("样本内", IS), ("样本外", OOS)]:
            m = met(r[(r.index >= s) & (r.index <= e)])
            if m:
                out[f"{nm}|{lab}"] = m
                print(f"  {nm:16s} {lab}  年化 {m['CAGR']:>7.2%}  波动 {m['AnnVol']:>6.2%}  "
                      f"夏普 {m['Sharpe']:>6.3f}  回撤 {m['MaxDD']:>8.2%}")
    # 外部宽基指数（新浪通道）
    try:
        import akshare as ak
        for code, nm in [("sh000300", "沪深300"), ("sh000905", "中证500"),
                         ("sh000852", "中证1000"), ("sh000985", "中证全指")]:
            try:
                d = ak.stock_zh_index_daily(symbol=code)
                d["date"] = pd.to_datetime(d["date"])
                s = pd.to_numeric(d["close"], errors="coerce")
                r = s.pct_change().dropna()
                r.index = pd.to_datetime(d["date"])[1:]
                rec = {}
                for lab, (a, b) in [("样本内", IS), ("样本外", OOS)]:
                    seg = r[(r.index >= a) & (r.index <= b)]
                    if len(seg) < 100:
                        continue
                    eq = (1 + seg).cumprod()
                    cagr = eq.iloc[-1] ** (PPY / len(seg)) - 1
                    vol = seg.std() * np.sqrt(PPY)
                    dd = float((eq / eq.cummax() - 1).min())
                    rec[lab] = {"CAGR": round(float(cagr), 4), "AnnVol": round(float(vol), 4),
                                "Sharpe": round(float((seg.mean() * PPY - RF) / vol), 4),
                                "MaxDD": round(dd, 4)}
                    print(f"  {nm:16s} {lab}  年化 {cagr:>7.2%}  波动 {vol:>6.2%}  "
                          f"夏普 {(seg.mean()*PPY-RF)/vol:>6.3f}  回撤 {dd:>8.2%}")
                out[nm] = rec
            except Exception as ex:
                print(f"  {nm}: 获取失败 {type(ex).__name__}: {str(ex)[:70]}")
    except Exception as ex:
        print(f"  akshare 不可用: {ex}")
    return out


def solve_mv(mu, S, cap, lam):
    """max w'mu − lam·w'Sw  s.t. sum w = 1, 0 <= w <= cap"""
    n = len(mu)
    x0 = np.full(n, 1.0 / n)
    x0 = np.minimum(x0, cap)
    x0 = x0 / x0.sum()

    def obj(w):
        return -(w @ mu) + lam * float(w @ S @ w)

    def grad(w):
        return -mu + 2.0 * lam * (S @ w)

    bnds = [(0.0, cap)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0,
             "jac": lambda w: np.ones_like(w)}]
    try:
        res = minimize(obj, x0, jac=grad, bounds=bnds, constraints=cons,
                       method="SLSQP", options={"maxiter": 120, "ftol": 1e-9})
        w = np.clip(res.x, 0, cap)
        return w / w.sum() if w.sum() > 0 else x0
    except Exception:
        return x0


def make_weights_mv(score, eligible, P, n=50, pool=200, lam=3.0, cap=0.03,
                    delta=0.2, rebalance_days=21, cov_win=252):
    """风险模型优化权重：在候选池内用收缩协方差最大化 alpha − λ·方差"""
    ret = P.ret
    cols = score.columns
    rebal = score.index[::rebalance_days]
    rows = {}
    t0 = time.time()
    for k, dt in enumerate(rebal):
        s = score.loc[dt].where(eligible.loc[dt]).dropna()
        if len(s) < 20:
            continue
        cand = s.nlargest(min(pool, len(s)))
        if len(cand) < 10:
            continue
        pos = P.dates.get_loc(dt) if dt in P.dates else None
        if pos is None or pos < 60:
            continue
        lo = max(0, pos - cov_win)
        R = ret.iloc[lo:pos].reindex(columns=cand.index).to_numpy(dtype=np.float64)
        R = np.nan_to_num(R, nan=0.0)
        S = np.cov(R, rowvar=False)
        S = np.atleast_2d(S)
        dg = np.diag(np.diag(S))
        S = (1 - delta) * S + delta * dg
        S = S + np.eye(len(cand)) * 1e-8
        mu = cand.to_numpy(dtype=np.float64)
        mu = (mu - mu.mean()) / (mu.std() + 1e-12)     # 标准化，使 λ 有稳定含义
        wv = solve_mv(mu, S, cap, lam)
        w = pd.Series(wv, index=cand.index).nlargest(min(n, len(cand)))
        w = v2._apply_cap(w, cap)
        rows[dt] = w
        if (k + 1) % 40 == 0:
            print(f"      优化进度 {k+1}/{len(rebal)}  {time.time()-t0:.0f}s", flush=True)
    W = pd.DataFrame(rows).T.fillna(0.0)
    return W.reindex(columns=cols, fill_value=0.0)


def main():
    print("=" * 96)
    print("目标可达性标定 + 风险模型优化")
    print("=" * 96)
    df = load_panel()
    P = Panels(df)
    del df
    eligible, _, _ = build_universe(P, CFG)
    need = sorted(set(v2.FACTOR_SETS["F17"] + v2.EXTRA_NAMES + v2.CONTROL_NAMES))
    v2.build_prep(P, eligible, need)
    score = v2.composite(v2.FACTOR_SETS["F17"])

    out = {"calibration": calibrate(P, eligible, None), "riskopt": {}}
    with open(os.path.join(OUT, "calibration.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 96)
    print("B  风险模型优化（此前未试过的杠杆）")
    print("=" * 96)
    CFGS = {
        "等权 n50 (基准)": None,
        "MV n50 pool100 λ=1": dict(n=50, pool=100, lam=1.0),
        "MV n50 pool100 λ=3": dict(n=50, pool=100, lam=3.0),
        "MV n50 pool200 λ=1": dict(n=50, pool=200, lam=1.0),
        "MV n50 pool200 λ=3": dict(n=50, pool=200, lam=3.0),
        "MV n50 pool200 λ=10": dict(n=50, pool=200, lam=10.0),
    }
    print(f"  {'方案':22s} | {'IS夏普':>7s} {'IS波动':>7s} | {'OOS夏普':>8s} {'OOS年化':>8s} "
          f"{'OOS波动':>8s} {'OOS回撤':>8s} {'Calmar':>7s}")
    print("  " + "-" * 100)
    for lab, kw in CFGS.items():
        t0 = time.time()
        if kw is None:
            W = v2.make_weights(score, eligible, side="long", **REC)
        else:
            W = make_weights_mv(score, eligible, P, cap=0.03, rebalance_days=21, **kw)
        r = backtest(P, W, CFG, lab)["net"]
        del W
        i = met(r[(r.index >= IS[0]) & (r.index <= IS[1])])
        o = met(r[(r.index >= OOS[0]) & (r.index <= OOS[1])])
        if not i or not o:
            continue
        out["riskopt"][lab] = {"IS": i, "OOS": o, "params": kw}
        print(f"  {lab:22s} | {i['Sharpe']:>7.3f} {i['AnnVol']:>7.2%} | "
              f"{o['Sharpe']:>8.3f} {o['CAGR']:>8.2%} {o['AnnVol']:>8.2%} "
              f"{o['MaxDD']:>8.2%} {str(o['Calmar']):>7s}   ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(OUT, "calibration.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n[已保存] {os.path.join(OUT,'calibration.json')}")


if __name__ == "__main__":
    main()
