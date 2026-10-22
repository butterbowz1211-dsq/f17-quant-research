# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
build_market_ext.py — 扩展市场级状态序列（情绪 / 流动性 / 宽度）
================================================================================
为什么要这一步
--------------
risk_attribution.py 的结论：
  · Daniel & Moskowitz (2016) 的「熊市×波动」panic 状态对本策略无效
    （γ_int 符号相反、t=0.90）—— 因为本策略不含动量。
  · 唯一显著的月度收益预测变量是「市场参与股票数的12个月变化」(t=-3.03)。
  · 本策略的真实风险是「做空投机股被逼空」+「小盘流动性踩踏」，
    这正是 Stambaugh-Yu-Yuan (2012) 情绪条件化异常收益所预测的方向。

因此需要构造 A 股的情绪/流动性状态变量。可用的代理（全部由本地面板导出）：

  turnover_agg   总成交额 / 总流通市值          —— Baker-Wurgler 的 turnover 代理
                                                也是 A 股最经典的情绪指标
  ipo_n          当月新上市股票数               —— BW 的 IPO 数量代理
  issuance       净股权发行 / 总流通市值         —— BW 的 equity share in new issues 代理
                                                （流通股本变化 × 价格 / 总市值）
  dispersion     日收益截面标准差               —— 意见分歧 / 套利限制
  small_big      小盘 minus 大盘 相对强弱        —— 规模风格状态
  illiq_agg      聚合成交额加权的 Amihud 非流动性 —— 市场流动性状态
  comove_ew      等权组合方差 / 个股平均方差     —— 平均相关性（联动性）
  breadth        上涨股票占比                    —— 市场宽度

输出 data/market_daily2.pkl（保留原 market_daily.pkl 不动）
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def main():
    t0 = time.time()
    from ashare_factor_strategy import Panels, load_panel

    print("加载面板 ...", flush=True)
    df = load_panel()
    P = Panels(df)
    del df

    traded = P.traded
    r = P.ret.where(traded)
    fmc = P.float_mktcap
    w = fmc.shift(1).where(traded)           # 滞后一期市值做权重（时点正确）
    den = w.sum(axis=1, min_count=1)

    out = pd.DataFrame(index=P.dates)

    # ---- 基础 ----
    out["mkt_vw"] = (w * r).sum(axis=1, min_count=1) / den
    out["mkt_ew"] = r.mean(axis=1)

    # ---- 情绪 1：总成交额 / 总流通市值（换手率口径） ----
    amt = P.amount.where(traded).sum(axis=1)
    total_fmc = fmc.where(traded).sum(axis=1)
    out["turnover_agg"] = amt / total_fmc.replace(0, np.nan)

    # ---- 情绪 2：当月新上市股票数（IPO 数量代理） ----
    # days_listed < 21 视为当月新上市
    new_list = (P.days_listed < 21) & traded
    out["ipo_n"] = new_list.sum(axis=1).astype(float)

    # ---- 情绪 3：净股权发行 / 总市值（BW 的 equity share in new issues 代理）----
    osh = P.outstanding_share.where(traded)
    d_osh = osh.diff()
    # 只在全市场层面聚合，正的部分视为发行、负的部分视为回购/注销
    net_issue = (d_osh.clip(lower=0) * P.raw_close.where(traded)).sum(axis=1, min_count=1)
    out["issuance"] = net_issue / total_fmc.replace(0, np.nan)

    # ---- 分歧：日收益截面标准差 ----
    out["dispersion"] = r.std(axis=1)

    # ---- 规模风格：小盘 vs 大盘 ----
    rank_pct = fmc.where(traded).rank(axis=1, pct=True)
    small = (rank_pct <= 0.30)
    big = (rank_pct >= 0.70)
    out["small_ret"] = (r.where(small)).mean(axis=1)
    out["big_ret"] = (r.where(big)).mean(axis=1)
    out["small_big"] = out["small_ret"] - out["big_ret"]

    # ---- 流动性：聚合成交额加权 Amihud ----
    ill = (P.ret.abs() / P.amount.replace(0, np.nan)).where(traded)
    out["illiq_agg"] = (ill * w).sum(axis=1, min_count=1) / den

    # ---- 平均相关性：等权组合方差 / 个股平均方差 ----
    v_ew = out["mkt_ew"].rolling(60).var()
    v_avg = r.rolling(60).var().mean(axis=1)
    out["comove_ew"] = v_ew / v_avg.replace(0, np.nan)

    # ---- 市场宽度 ----
    n_valid = r.notna().sum(axis=1)
    out["breadth"] = (r > 0).sum(axis=1) / n_valid.replace(0, np.nan)

    out["n_stock"] = traded.sum(axis=1).astype(float)
    out["amt"] = amt
    out.index.name = "date"

    p = os.path.join(DATA, "market_daily2.pkl")
    out.to_pickle(p)
    print(f"[已保存] {p}   {out.shape}   {time.time()-t0:.0f}s")
    print(out.describe().T[["mean", "std", "min", "max"]].to_string(
        float_format=lambda x: f"{x:.6g}"))


if __name__ == "__main__":
    main()
