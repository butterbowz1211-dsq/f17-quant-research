# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
debug_ch3.py — 诊断 CH-3 中 VMG / PMO 符号与量级异常
================================================================================
现象：MKT 1.44%/月、SMB 1.48%/月 尚属合理，但
      VMG = −2.96%/月 (t=−8.25)   —— LSY 原文是 +1.14%
      PMO = −7.09%/月 (t=−22.00)  —— LSY 原文约 +1.0%/月
t 值达 −22 说明是**系统性构建问题**，不是抽样噪声。本脚本用缓存数据逐层拆解。
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

C = pd.read_pickle(os.path.join(DATA, "ch3_prices.pkl"))
close_me, raw_me, mcap_me = C["close_me"], C["raw_me"], C["mcap_me"]
ret_me, abn_turn = C["ret_me"], C["abn_turn"]
expo = pd.read_pickle(os.path.join(DATA, "ch3_expo.pkl"))
ch = pd.read_pickle(os.path.join(DATA, "ch3.pkl"))

print("=" * 88)
print("1  原始分布检查（判断是否有极端值污染）")
print("=" * 88)
allEP, allBM, allAT = [], [], []
for dt, e in expo.items():
    allEP.append(e["EP"].dropna())
    allBM.append(e["BM"].dropna())
    allAT.append(e["abn_turn"].dropna())
EP = pd.concat(allEP); BM = pd.concat(allBM); AT = pd.concat(allAT)
for nm, s in [("EP", EP), ("BM", BM), ("abn_turn", AT)]:
    q = s.quantile([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
    print(f"  {nm:9s} n={len(s):>9,}  mean={s.mean():>9.4f}  sd={s.std():>8.4f}")
    print(f"            p0.1%={q.iloc[0]:>9.4f} p1%={q.iloc[1]:>8.4f} p5%={q.iloc[2]:>8.4f} "
          f"p25%={q.iloc[3]:>8.4f} 中位={q.iloc[4]:>8.4f}")
    print(f"            p75%={q.iloc[5]:>8.4f} p95%={q.iloc[6]:>8.4f} p99%={q.iloc[7]:>8.4f} "
          f"p99.9%={q.iloc[8]:>9.4f}")

print()
print("=" * 88)
print("2  简单 30/70 分位市值加权价差（不走 2×3 分组，直接看符号）")
print("=" * 88)
rows = []
for dt in sorted(expo.keys()):
    e = expo[dt]
    ok = e["mcap"].notna() & (e["mcap"] > 0)
    if ok.sum() < 100:
        continue
    q30 = e.loc[ok, "mcap"].quantile(0.30)
    e = e[ok & (e["mcap"] > q30)]
    r = ret_me.loc[dt].reindex(e.index)
    m = r.notna()
    e, r = e[m], r[m]
    if len(e) < 80:
        continue
    w = e["mcap"] / e["mcap"].sum()

    def spread(col, lo_is_high=False):
        x = e[col]
        if x.notna().sum() < 50:
            return np.nan
        hi = x >= x.quantile(0.70)
        lo = x <= x.quantile(0.30)
        def vw(mask):
            mm = mask & r.notna()
            if mm.sum() < 3:
                return np.nan
            ww = w[mm] / w[mm].sum()
            return float((ww * r[mm]).sum())
        a, b = vw(hi), vw(lo)
        return (b - a) if lo_is_high else (a - b)

    rows.append((dt, spread("EP"), spread("BM"), spread("abn_turn", lo_is_high=True),
                 float((w * r).sum()), int(len(e))))

S = pd.DataFrame(rows, columns=["date", "EP_hi_minus_lo", "BM_hi_minus_lo",
                                "ABNTURN_lo_minus_hi", "MKT", "n"]).set_index("date")
print(f"  {'价差':22s} {'均值(月)':>10s} {'年化':>10s} {'标准差':>9s} {'t':>7s}")
for c in ["EP_hi_minus_lo", "BM_hi_minus_lo", "ABNTURN_lo_minus_hi"]:
    s = S[c].dropna()
    t = s.mean() / s.std() * np.sqrt(len(s))
    print(f"  {c:22s} {s.mean()*100:>9.3f}% {((1+s.mean())**12-1)*100:>9.2f}% "
          f"{s.std()*100:>8.3f}% {t:>7.2f}")

print()
print("=" * 88)
print("3  与策略因子的对照：EP / abn_turn 的横截面 RankIC（21日前瞻）")
print("=" * 88)
mdates = list(S.index)
fwd = close_me.shift(-1) / close_me - 1          # 月末到下一月末
for nm, col in [("EP", "EP"), ("BM", "BM"), ("abn_turn", "abn_turn")]:
    ics = []
    for i, dt in enumerate(mdates[:-1]):
        e = expo.get(dt)
        if e is None:
            continue
        f = fwd.loc[dt].reindex(e.index)
        x = e[col]
        m = x.notna() & f.notna()
        if m.sum() < 80:
            continue
        ics.append(x[m].rank().corr(f[m].rank()))
    if ics:
        s = pd.Series(ics)
        t = s.mean() / s.std() * np.sqrt(len(s))
        print(f"  {nm:9s} RankIC={s.mean():+.4f}  t={t:+.2f}  n={len(s)}  "
              f"正IC占比={(s>0).mean():.3f}")

print()
print("=" * 88)
print("4  保存的 CH-3 因子序列的前后段对照（看是否被某一段主导）")
print("=" * 88)
for c in ch.columns:
    if c == "n":
        continue
    s = ch[c].dropna()
    a, b = s[s.index < "2018-01-01"], s[s.index >= "2018-01-01"]
    print(f"  {c:5s}  全样本 {s.mean()*100:>+7.3f}%  |  "
          f"2010-2017 {a.mean()*100:>+7.3f}% (n={len(a)})  |  "
          f"2018-2026 {b.mean()*100:>+7.3f}% (n={len(b)})")
