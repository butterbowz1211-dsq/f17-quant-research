# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""性能剖析 (profile_pipeline.py) —— 定位全量流程的耗时热点"""
import os, time, sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ashare_factor_strategy as S

def t(label):
    class T:
        def __enter__(self): self.t0 = time.perf_counter(); return self
        def __exit__(self, *a):
            print(f"  {label:<46s} {time.perf_counter()-self.t0:8.2f}s", flush=True)
    return T()

print("=" * 76)
with t("加载面板 + 构建宽表 Panels"):
    df = S.load_panel()
    P = S.Panels(df)
print(f"     面板 {len(df):,} 行, 宽表 {len(P.dates)}×{len(P.symbols)}")
del df

with t("build_universe (含 detect_limit)"):
    eligible, lu, ld = S.build_universe(P, S.CFG)

with t("Ctx (共享中间量)"):
    ctx = S.Ctx(P)

raw = pd.read_pickle(os.path.join(S.DATA, "factors_LSY", "TURN.pkl"))
print(f"     因子 TURN 覆盖 {raw.notna().mean().mean()*100:.1f}%")

with t("  winsorize_mad"):
    x1 = S.winsorize_mad(raw.where(eligible), 3.0)
with t("  neutralize (FWL)"):
    x2 = S.neutralize(x1, P, True, True, eligible)
with t("  standardize"):
    x3 = S.standardize(x2)
print()

fwd = S.forward_return(P.close, S.CFG.ic_forward_days)
m_is = (P.dates >= pd.Timestamp(S.CFG.full_start)) & (P.dates <= pd.Timestamp(S.CFG.selection_end))
m_oos = (P.dates >= pd.Timestamp(S.CFG.bt_start)) & (P.dates <= pd.Timestamp(S.CFG.bt_end))
f_is, e_is = fwd.loc[m_is], eligible.loc[m_is]
pre_is = x3.where(eligible).loc[m_is]

with t("  rank_ic (样本内)"):
    ic = S.rank_ic(pre_is, f_is)
with t("  quantile_backtest"):
    Q = S.quantile_backtest(pre_is, f_is, 10)
with t("  fama_macbeth"):
    fm = S.fama_macbeth(pre_is, f_is, e_is)
with t("  fast_rank_pct 单次"):
    S.fast_rank_pct(pre_is)
print()

# 回测
score = pre_is.copy()
with t("build_weights (多头)"):
    W = S.build_weights(score, eligible, "long", S.CFG)
with t("backtest 主引擎 (逐日漂移)"):
    r = S.backtest(P, W, S.CFG, "t")
with t("backtest_vectorized"):
    S.backtest_vectorized(P, W, S.CFG, "t")
print()

with t("stationary_bootstrap (n=2000)"):
    S.stationary_bootstrap(r["net"].dropna(), 2000, 21, 42)
print()
print("外推估算全流程 (24 因子, 2 个窗口):")
print("  说明: 上表中 prepare_factor 各步 × 24, evaluate 各步 × 24 × 2")
