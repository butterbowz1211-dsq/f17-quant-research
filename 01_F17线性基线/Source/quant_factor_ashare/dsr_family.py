# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
dsr_family.py — 多重检验校正的规范做法：按策略族分别计算
================================================================================
问题
----
此前的 DSR 计算把**多头族**（夏普 ~0.5-1.0）与**多空族**（夏普 ~0.6-1.9）
混在同一个「试验分布」里。这有两个后果：
  1. 试验数 N 被虚增（两族的配置被加在一起）；
  2. **试验夏普的方差被人为放大** —— 而 DSR 的门槛 SR0 = sqrt(Var(SR)) × f(N)，
     方差虚高会直接抬高门槛，让 DSR 不必要地难看。

Bailey & López de Prado (2014) 的原意是：N 与 Var(SR) 应取自**同一个策略搜索过程**。
因此正确做法是**按策略族分别计算**。

本脚本给出四组口径，全部列出，不挑好看的：
  (a) 多头族：N = 多头配置数, Var = 多头试验夏普方差
  (b) 多空族：N = 多空配置数, Var = 多空试验夏普方差
  (c) 合并族：N = 全部, Var = 全部（此前的做法，作为最保守对照）
  (d) N = 1：即「只报一个预先指定的规格、完全不做搜索」→ DSR 退化为 PSR，
      这是 DSR 的**上界**（无选择偏差可校正）

真实检验对象
------------
· 多头：近 24 个月（用户要求的分窗口口径）
· 多空：近 12 个月
并同时给出 3.6 年样本外窗口的结果作对照。
"""

import os
import sys
import json
import glob
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "output")


def collect_trials():
    """从结果文件中收集**同一评价窗口**下的样本外夏普，按策略族分类。

    ★ 精确白名单，不用 glob 通配
      首版用 glob("*.json") 通配收集，结果被污染：
        · benchmark_windows.json 里的**基准指数**夏普（沪深300 近3个月 −2.645）被当成试验；
        · final_multiwindow / phase_multiwindow 里的**短窗口**指标
          （多空 近3个月 5.51、近24个月 0.10）被当成试验 —— 但它们是同一策略在不同
          窗口上的读数，不是「试过的不同策略」，方差因此被严重放大
          （多空族方差被抬到 1.0392，门槛 SR0 = 2.554，DSR 直接归零）。
      试验集必须满足：**每个元素 = 一个不同的策略配置，在同一评价窗口上的样本外夏普**。
    """
    long_s, ls_s = [], []

    def push(key, val):
        if val is None or not np.isfinite(val):
            return
        k = str(key)
        # 排除基准与对照（它们不是"试过的策略"）
        for bad in ("基准", "等权全市场", "沪深300", "中证500", "中证1000",
                    "流通市值加权", "市场", "benchmark", "_best", "chosen",
                    "is_selected", "alpha_weights", "weights"):
            if bad in k:
                return
        # ★ 关键：未显式标注的，一律计为**多头（long-only）**。
        #   v2_grid / v2_tune / v2_beta 的键形如 "F17 n=15"、"F17 n=50 buffer=0.3"，
        #   它们全部调用 make_weights(side="long")，是实实在在的多头试验。
        #   首版要求键名必须含「多头/long」才计入，导致 32 个 grid 试验被漏掉，
        #   使 N 偏小、方差偏小、DSR 被**高估** —— 与"污染导致高估门槛"是相反方向的错误。
        if ("多空" in k or "价差" in k or "_ls" in k.lower() or "|ls" in k.lower()
                or k.endswith("|LS") or "LS|" in k):
            ls_s.append(float(val))
        else:
            long_s.append(float(val))

    # 各文件的 OOS 夏普字段位置（均为「样本外 3.6 年」或「近 24 个月」同一口径）
    SPECS = [
        ("v2_grid.json", lambda v: v.get("OOS", {}).get("Sharpe")
         if isinstance(v.get("OOS"), dict) else None),
        ("v2_tune.json", lambda v: v.get("OOS", {}).get("Sharpe")
         if isinstance(v.get("OOS"), dict) else None),
        ("v2_beta.json", lambda v: v.get("stats", {}).get("OOS", {}).get("Sharpe")
         if isinstance(v.get("stats"), dict) else None),
        ("neutral_experiment.json", lambda v: v.get("OOS", {}).get("Sharpe")
         if isinstance(v.get("OOS"), dict) else None),
        ("ls_improve.json", lambda v: v.get("OOS_sharpe_mean")),
        ("longonly_improve.json", lambda v: v.get("OOS_sharpe_mean")),
        ("geo_test.json", lambda v: v.get("OOS_sharpe_mean")),
        ("fm_weights.json", lambda v: v.get("OOS_sharpe_mean")),
        ("window_slicing.json", lambda v: v.get("OOS_sharpe") or v.get("OOS_sharpe_mean")),
    ]
    for fn, getter in SPECS:
        fp = os.path.join(OUT, fn)
        if not os.path.exists(fp):
            continue
        try:
            with open(fp, encoding="utf-8") as fh:
                j = json.load(fh)
        except Exception:
            continue
        # v2_phase.json 的结构不同：phase_tbl 直接是 {label: {...}}
        body = j.get("results", j)
        for k, v in body.items():
            if not isinstance(v, dict):
                continue
            try:
                push(k, getter(v))
            except Exception:
                pass
    # v2_phase.json 单独处理（键里含因子集与配置名）
    fp = os.path.join(OUT, "v2_phase.json")
    if os.path.exists(fp):
        with open(fp, encoding="utf-8") as fh:
            j = json.load(fh)
        for k, v in (j.get("phase", j)).items():
            if isinstance(v, dict) and "OOS_sharpe_mean" in v:
                push(k, v["OOS_sharpe_mean"])
    return np.array(long_s), np.array(ls_s)


def main():
    from ashare_factor_strategy import (load_panel, Panels, build_universe, backtest,
                                        CFG, probabilistic_sharpe, deflated_sharpe)
    import strategy_v2 as v2

    print("=" * 104)
    print("多重检验校正：按策略族分别计算 DSR")
    print("=" * 104)

    lo, ls = collect_trials()
    ot = np.array([])
    print(f"\n从结果文件收集到的试验夏普：")
    for nm, a in [("多头族", lo), ("多空族", ls), ("未分类", ot)]:
        if a.size:
            print(f"  {nm:8s} n={a.size:>4d}  均值 {a.mean():>6.3f}  "
                  f"标准差 {a.std(ddof=1):>6.3f}  方差 {a.var(ddof=1):>7.4f}  "
                  f"[{a.min():.3f}, {a.max():.3f}]")
    allv = np.concatenate([lo, ls, ot]) if (lo.size + ls.size + ot.size) else np.array([0.5, 1.0])
    print(f"  {'合并':8s} n={allv.size:>4d}  均值 {allv.mean():>6.3f}  "
          f"标准差 {allv.std(ddof=1):>6.3f}  方差 {allv.var(ddof=1):>7.4f}")

    # ---- 重建推荐组合的收益序列 ----
    print("\n重建推荐配置的收益序列 ...", flush=True)
    df = load_panel()
    P = Panels(df)
    del df
    eligible, _, _ = build_universe(P, CFG)
    need = sorted(set(v2.FACTOR_SETS["F17"] + v2.EXTRA_NAMES + v2.CONTROL_NAMES))
    v2.build_prep(P, eligible, need)
    score = v2.composite(v2.FACTOR_SETS["F17"])
    REC = dict(n=50, weighting="equal", buffer=0.6, rebalance_days=21, max_weight=0.03)
    Wl = v2.make_weights(score, eligible, side="long", **REC)
    r_long = backtest(P, Wl, CFG, "L")["net"]
    Ws = v2.make_weights(score, eligible, side="short", **REC)
    Wls = Wl.sub(Ws.reindex(columns=Wl.columns, fill_value=0.0), fill_value=0.0)
    r_ls = backtest(P, Wls, CFG, "LS")["net"]
    del Wl, Ws, Wls

    WINDOWS = {"近24个月": 504, "近12个月": 252, "近36个月": 756, "样本外3.6年": None}
    OOS = ("2023-01-01", "2026-12-31")
    out = {}
    print("\n" + "=" * 104)
    print("DSR 四组口径对照")
    print("=" * 104)
    print(f"  {'组合':6s}{'窗口':>12s}{'夏普':>8s}{'PSR':>8s}{'DSR(N=1)':>10s}"
          f"{'DSR(本族)':>11s}{'(N,SR0)':>16s}{'DSR(合并)':>11s}")
    for tag, ser, fam in [("多头", r_long, lo), ("多空", r_ls, ls)]:
        for wn, nd in WINDOWS.items():
            if nd is None:
                seg = ser[(ser.index >= OOS[0]) & (ser.index <= OOS[1])]
            else:
                seg = ser.iloc[-nd:]
            seg = seg.dropna()
            if len(seg) < 30:
                continue
            vol = seg.std() * np.sqrt(252)
            sr = (seg.mean() * 252 - 0.02) / vol
            sr_p = sr / np.sqrt(252)
            psr = probabilistic_sharpe(sr_p, len(seg), float(seg.skew()),
                                       float(seg.kurtosis() + 3.0))
            d1 = deflated_sharpe(seg, n_trials=1, sr_variance=0.0)
            fv = float(fam.var(ddof=1)) if fam.size > 1 else 0.0
            d_fam = deflated_sharpe(seg, n_trials=max(int(fam.size), 1), sr_variance=fv)
            d_all = deflated_sharpe(seg, n_trials=int(allv.size),
                                    sr_variance=float(allv.var(ddof=1)))
            out[f"{tag}|{wn}"] = {
                "sharpe": round(float(sr), 4), "n_days": int(len(seg)),
                "PSR": round(float(psr), 4),
                "DSR_N1": round(float(d1["DSR"]), 4),
                "DSR_family": round(float(d_fam["DSR"]), 4),
                "family_N": int(fam.size), "family_SR0": round(float(d_fam["SR0"]), 4),
                "DSR_pooled": round(float(d_all["DSR"]), 4),
                "pooled_N": int(allv.size), "pooled_SR0": round(float(d_all["SR0"]), 4)}
            v = out[f"{tag}|{wn}"]
            print(f"  {tag:6s}{wn:>12s}{sr:>8.3f}{psr:>8.4f}{d1['DSR']:>10.4f}"
                  f"{d_fam['DSR']:>11.4f}   ({fam.size:>3d}, {d_fam['SR0']:.3f})"
                  f"{d_all['DSR']:>11.4f}")

    with open(os.path.join(OUT, "dsr_family.json"), "w", encoding="utf-8") as fh:
        json.dump({"results": out,
                   "trial_stats": {
                       "long": {"n": int(lo.size), "sd": round(float(lo.std(ddof=1)), 4),
                                "var": round(float(lo.var(ddof=1)), 4)} if lo.size else None,
                       "ls": {"n": int(ls.size), "sd": round(float(ls.std(ddof=1)), 4),
                              "var": round(float(ls.var(ddof=1)), 4)} if ls.size else None,
                       "pooled": {"n": int(allv.size),
                                  "sd": round(float(allv.std(ddof=1)), 4),
                                  "var": round(float(allv.var(ddof=1)), 4)}}},
                  fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

