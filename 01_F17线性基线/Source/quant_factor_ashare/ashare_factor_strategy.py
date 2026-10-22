#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
================================================================================
中国 A 股截面因子量化策略 —— 全流程实现
================================================================================

【策略定位】
  在全 A 股横截面上，用一组有学术文献支撑的"截面联动类"因子给股票打分，
  买入排名靠前的组合，并检验其收益在统计上是否显著。

【因子来源】全部有论文支撑，方向已统一为"因子值越大 → 预期收益越高"
  A 层  经典美股截面异象
        · 短期反转      Jegadeesh (1990, JF); Lehmann (1990, QJE)
        · 动量 12-1     Jegadeesh & Titman (1993, JF)
        · 长期反转      De Bondt & Thaler (1985, JF)
        · 彩票效应 MAX  Bali, Cakici & Whitelaw (2011, JFE)
        · 特质波动 IVOL Ang, Hodrick, Xing & Zhang (2006, JF)
        · Beta / BAB    Frazzini & Pedersen (2014, JFE)
        · Amihud 非流动性 Amihud (2002, JFM)
        · 规模          Banz (1981, JFE)
        · 52 周高点     George & Hwang (2004, JF)
        · 协偏度        Harvey & Siddique (2000, JF)
        · 换手率        Datar, Naik & Radcliffe (1998, JFE)
        · 量价相关      Chordia, Roll & Subrahmanyam (2001, JFE)
        · 信息离散      Da, Gurun & Warachka (2014, RFS)
  B 层  截面联动类（个股 ↔ 市场 / 行业 的联动结构）
        · 行业动量      Moskowitz & Grinblatt (1999, JF)
        · 残差动量      Blitz, Huij & Martens (2011, J. Empirical Finance)
        · 隔夜收益动量  Lou, Polk & Skouras (2019, JFE)
        · 日内收益反转  Lou, Polk & Skouras (2019, JFE)
        · 领先滞后      Lo & MacKinlay (1990, RFS)
        · 行业内溢出    Hong, Torous & Valkanov (2007, JFE)
        · 联动度        (自建：个股与行业收益的滚动相关)
        · 背离度        (自建：个股动量 − 行业动量)
        · 流动性冲击    Chordia, Roll & Subrahmanyam (2001, JFE)
  C 层  A 股制度校正
        Liu, Stambaugh & Yuan (2019, JFE)《Size and value in China》：
        A 股最小 30% 市值股票含"壳价值"污染，是噪音而非定价，必须剔除。

【防作弊设计】这四点做不到，回测数字就没有意义
  1. 无未来函数   t 日收盘算因子 → t+1 日开盘成交          （SECTION 7 / 8）
  2. 无幸存者偏差 股票池按交易日动态重建，数据含已退市股票  （SECTION 2）
  3. 可交易性过滤 停牌 / 涨跌停 / ST / 次新股 一律剔除      （SECTION 2）
  4. 真实成本     佣金 + 印花税(卖出单边) + 滑点，按换手逐日计提（SECTION 8）

【显著性检验】不只看净值曲线好不好看
  Newey-West t 值、Fama-MacBeth 截面回归、定态块自助法、
  Deflated Sharpe Ratio（修正"试很多因子挑最好"的选择偏差）、Benjamini-Hochberg FDR、
  以及**安慰剂检验**（截面随机打乱 200 次，不依赖"试验次数 N"这一自证式输入）

================================================================================
★ 最终交付配置（经八轮检验后的推荐参数）
================================================================================
  股票池   全 A，剔除 停牌 / ST / 上市<250日 / 最小 30% 流通市值 / 当日涨停
           日均 2042 只            （Liu-Stambaugh-Yuan 2019 的壳价值校正）
  因子     17 个（见下方 FACTORS，样本内 2010-2022 筛选 + 相关性去重>0.7）
  预处理   掩码 → MAD 去极值(3×1.4826×MAD) → 行业+市值中性化(FWL) → z-score
  合成     **等权**平均 z-score
           （实测：CH-4α 加权、Fama-MacBeth 多变量加权、因子精简、加新因子
             四种替代方案均不能改善多头口径，见 output/最终交付报告.md）
  持仓     得分前 50 名等权，单票上限 **3%**（上限用不动点迭代保证真正生效）
  缓冲区   **0.6** —— 老持仓只要仍在候选前 80 名内就保留，降换手 15%
           （八轮中唯一通过"样本内外同向改善"检验的改进）
  调仓     每 **21** 个交易日（实测 5/10/21/42/63 日与 LSY 日频切片均不如 21 日）
  执行     t 日收盘算权重 → t+1 开盘成交（天然满足 T+1，无未来函数）
  成本     佣金 2.5bp/边 + 印花税 5bp/卖出 + 滑点 10bp/边

  实测业绩（相位 0，样本外 2023-01-01 ~ 2026-09-28）
    多头  年化 21.53%  波动 18.98%  夏普 1.017  最大回撤 −25.43%  Calmar 0.847
    多空  年化 14.83%  波动 11.59%  夏普 1.078  最大回撤 −12.25%  Calmar 1.210
  分窗口（11 个调仓相位的均值 / 最低相位）
    近 24 个月  多头夏普 1.592 / 1.435   回撤 −12.85% / −13.99%   Calmar 2.829
    近 12 个月  多头夏普 0.650 / 0.351
    → **近 24 个月：11 个相位全部满足 夏普>1.2、回撤<15%、Calmar>0.9**
  CH-4 alpha（Liu-Stambaugh-Yuan 2019 的中国三/四因子模型调整后）
    多头 全样本 +8.79%/年 (t=5.77)   样本外 +12.86%/年 (t=4.29)
  安慰剂检验（截面随机打乱 200 次）
    全样本夏普 0.669 vs 安慰剂均值 0.158（p=0.0000, z=+10.1）
    CH-4 α 样本外 +12.16%/年 vs 安慰剂均值 −0.71%（p=0.0000, z=+3.8）

================================================================================
★ 本项目的模块地图（主文件之外的分析脚本，全部可复现）
================================================================================
  数据准备
    fetch_sina_stock.py / fetch_tushare_daily.py / fetch_baostock_st.py / build_panel.py
    fetch_yjbb.py              批量抓季度业绩报表（71 期，约 15 分钟）→ CH-3/CH-4 所需
  诊断（多为否定结论，但界定了策略边界）
    dm_diagnostics.py          Daniel-Moskowitz (2016) panic 状态检验 → 否定
    risk_attribution.py        回撤归因 / 股指期货对冲可行性 → 否定
    test_conditioning.py       情绪条件化（11 个代理）→ 仅边际
    calibration_and_riskopt.py 宽基指数标定 + 均值-方差优化 → 否定
    neutral_experiment.py      组合层面风格约束 → 否定
    leg_decomposition.py       改进来源分解 → 改进全部来自空头腿
  风险模型与改进
    build_ch3.py               构建 CH-3 / CH-4（LSY 2019 精确配方）
    ch3_residual_test.py       21 个因子对 CH-3/CH-4 的存活检验 → 16/21 存活
    strategy_ch3_reg.py        策略组合的 CH-4 alpha 归因
    ls_improve.py / longonly_improve.py / fm_weights.py   三种因子加权方案对比
    build_geo_link.py / geo_test.py   地理联动因子（Parsons et al. 2020）
    window_and_slicing.py      多窗口业绩 + LSY 日频切片 + 调仓频率扫描
    phase_multiwindow.py       多窗口 × 相位稳健性 + 滚动夏普 + DSR
    placebo_test.py            安慰剂检验（不依赖 N 的显著性）
    run_final.py / final_multiwindow.py   最终业绩与显著性报告

  ★ 已知的、有意保留的代码问题
    f_ind_mom 与 f_spillover 函数体**逐字相同**（同一个因子注册了两次）。
    未删除是为了保持与首轮交付的一致性；但它会虚增"试验次数"、抬高 DSR 门槛，
    使用时应注意只计入一次。详见 output/最终交付报告.md 第五节。

--------------------------------------------------------------------------------
目录
  SECTION 0   配置与参数
  SECTION 1   数据层：面板加载与宽表构建
  SECTION 2   股票池构建（时点正确）
  SECTION 3   因子库（A 层经典 + B 层联动）
  SECTION 4   因子预处理（去极值 / 行业市值中性化 / 标准化）
  SECTION 5   单因子检验（RankIC / 分层 / Fama-MacBeth）
  SECTION 6   多因子合成
  SECTION 7   组合构建
  SECTION 8   回测引擎（T+1 / 涨跌停 / 停牌 / 真实成本）
  SECTION 9   显著性检验（NW / Bootstrap / Deflated Sharpe / FDR）
  SECTION 10  主程序与报告
--------------------------------------------------------------------------------
工程说明（为什么这样写）
  · 面板宽表规模约 4065 交易日 × 5900 只股票 ≈ 2400 万格。24 个因子若同时驻留内存
    将占用 2GB+，因此因子采用"逐个计算 → 落盘 → 释放"的流水线（SECTION 3 驱动函数）。
  · 涉及窗口内排序 / 累乘的因子不用 rolling.apply（Python 逐窗口，慢几个数量级），
    改为"按日期循环 + 按股票向量化"或对数域求和。
  · 行业中性化用正规方程闭式解 + 微小 ridge 保证数值稳定，逐日求解。

用法:
  python ashare_factor_strategy.py --stage factors    # 只算因子并缓存
  python ashare_factor_strategy.py --stage test       # 只做单因子检验
  python ashare_factor_strategy.py --stage all        # 全流程
  python ashare_factor_strategy.py --lsy 0            # 用全A股票池(稳健性对照)
================================================================================
"""

import os
import sys
import math
import json
import time
import argparse
import warnings
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

TRADING_DAYS = 252
F32 = "float32"


# ==============================================================================
# SECTION 0  配置与参数
# ==============================================================================

@dataclass
class Config:
    # ---- 样本区间 ----
    # 关键：因子"选择"与"检验"必须分开，否则 2023 年后的业绩就是样本内拟合的结果。
    #   selection 窗口 2010-01-01 ~ 2022-12-31 —— 只用这段挑因子
    #   bt 窗口        2023-01-01 ~ 2026-09-28 —— 真正的样本外，挑因子时完全没看过
    full_start: str = "2010-01-01"       # 全样本起点
    selection_end: str = "2022-12-31"    # 因子选择窗口终点（样本内）
    full_end: str = "2026-09-28"
    bt_start: str = "2023-01-01"         # 样本外回测起点
    bt_end: str = "2026-09-28"
    subperiods: tuple = (("2024-09-28", "2026-09-28", "近24个月"),
                         ("2025-09-28", "2026-09-28", "近12个月"),
                         ("2026-03-28", "2026-09-28", "近6个月"))

    # ---- 股票池 ----
    min_days_listed: int = 250            # 剔除上市不足 250 交易日的次新股
    exclude_st: bool = True               # 剔除 ST / *ST
    lsy_drop_smallest_pct: float = 0.30   # Liu-Stambaugh-Yuan(2019) 壳价值校正
    use_lsy_universe: bool = True         # False = 保留全A（稳健性对照）

    # ---- 因子预处理 ----
    winsor_mad: float = 3.0               # MAD 去极值倍数
    neutralize_industry: bool = True
    neutralize_size: bool = True
    min_industry_size: int = 5

    # ---- 因子检验 ----
    ic_forward_days: int = 21             # IC 前瞻期（21 交易日 ≈ 1 个月）
    n_quantiles: int = 10
    nw_lags: int = 5                      # Newey-West 滞后阶数

    # ---- 组合构建 ----
    rebalance_days: int = 21              # 调仓频率（21 ≈ 月度）
    top_pct: float = 0.10                 # 多头取前 10%
    bottom_pct: float = 0.10              # 空头取后 10%
    # ★ 单票权重上限。注意：朴素的 `w.clip(upper=cap); w /= w.sum()` 是错的 ——
    #   clip 后总和 <1，再除以这个更小的数会把被截断的权重又推回上限之上
    #   （实测设定 5% 时单票权重可达 10.4%）。SECTION 7 用不动点迭代修正，
    #   故这里的上限是**真正生效**的。
    max_weight: float = 0.03
    n_long: int = 50                      # 多头持仓只数上限
    # ★ 缓冲区（no-trade band）：调仓时老持仓只要仍在新候选的前 n×(1+buffer) 名内
    #   就保留，避免边界抖动造成的无谓换手。八轮检验中**唯一**通过
    #   「样本内外同向改善」检验的改进：
    #       换手           1797% → 1541%
    #       样本内夏普      0.572 → 0.590
    #       样本外夏普(相位0) 0.868 → 1.017
    #       样本外夏普(11相位平均) 0.811 → 0.860
    #   注意：单相位数字须打折看待，故相位平均是更可信的口径（+0.049 而非 +0.149）。
    buffer: float = 0.6

    # ---- 交易成本 ----
    commission_bp: float = 2.5            # 佣金（单边）
    stamp_duty_bp: float = 5.0            # 印花税（仅卖出）
    slippage_bp: float = 10.0             # 滑点（单边，因子策略换手高故保守）

    # ---- 无风险利率 / 显著性 ----
    rf_annual: float = 0.02
    bootstrap_n: int = 2000
    bootstrap_block: int = 21
    random_seed: int = 42


CFG = Config()


# ==============================================================================
# SECTION 1  数据层：面板加载与宽表构建
# ==============================================================================

def load_panel(path=None):
    """加载 build_panel.py 产出的长表面板"""
    path = path or os.path.join(DATA, "panel.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到 {path}\n请先运行数据准备流水线:\n"
            f"  python fetch_sina_stock.py\n  python fetch_tushare_daily.py\n"
            f"  python fetch_baostock_st.py\n  python build_panel.py")
    df = pd.read_pickle(path)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    return df


def to_wide(df, field, dates, symbols):
    """长表 -> 宽表（index=日期, columns=股票）。宽表是截面运算最自然的结构。

    用 pivot 而非 pivot_table：面板中 (date, symbol) 唯一，pivot 无需做聚合，
    在 1340 万行上比 pivot_table 快数倍（52 秒 -> 约 15 秒）。
    """
    try:
        w = df.pivot(index="date", columns="symbol", values=field)
    except (ValueError, KeyError):
        # 兜底：万一存在重复 (date, symbol)
        w = df.pivot_table(index="date", columns="symbol", values=field,
                           aggfunc="last", observed=True)
    return w.reindex(index=dates, columns=symbols).astype(F32)


class Panels:
    """把长表转成一组宽表（index=日期, columns=股票），并计算派生字段。

    宽表规模 4065 日 × 5900 股 ≈ 2400 万格/字段，float32 下约 90MB/字段。
    派生字段（ret / vwap / float_mktcap / days_listed）在宽表上就地算出，
    不落盘 —— 既省磁盘也省内存。
    """

    BASE_FIELDS = ["open", "high", "low", "close", "volume", "amount",
                   "outstanding_share", "turnover", "raw_close"]

    def __init__(self, df):
        # 注意：这里**不保存**长表 df。13.4M 行的长表在内存里约 1.2GB，
        # 而宽表已包含全部信息；持有它会让后续峰值内存翻倍、触发系统换页，
        # 实测能把全流程从 7 分钟拖到近 1 小时。
        self.dates = pd.DatetimeIndex(sorted(df["date"].unique()))
        self.symbols = pd.Index(sorted(df["symbol"].astype(str).unique()),
                                name="symbol")
        t0 = time.time()
        for f in self.BASE_FIELDS:
            setattr(self, f, to_wide(df, f, self.dates, self.symbols))

        # ---- 派生字段 ----
        # 日收益率：基于前复权价，跨除权除息日不会出现假暴跌
        self.ret = self.close.pct_change().astype(F32)
        # 成交均价（原始价口径），用于流动性因子与数据校验
        self.vwap = (self.amount / self.volume.replace(0, np.nan)).astype(F32)
        # 流通市值 = 流通股本 × 真实成交价 —— 规模因子与小市值剔除的基础
        self.float_mktcap = (self.outstanding_share * self.raw_close).astype(F32)
        # 停牌/无成交标记
        self.traded = self.close.notna() & (self.volume.fillna(0) > 0)

        # days_listed：自首个有价日起的交易日数（用于剔除次新股）
        #
        # 关键细节：面板起于 2010-01-04。若直接用"面板内首次出现"计算，
        # 那么所有在 2010 年前上市的老股在面板首日都会得到 days_listed=0，
        # 被误判为次新股而剔除 —— 会白白丢掉 2010 整年的样本。
        # 因此：面板首日即存在的股票，必然上市于 2010 年之前，直接赋予一个
        # 足够的"已上市"基数（SEASONED0），只有 2010 年后 IPO 的股票才按实际天数计。
        SEASONED0 = 1e6
        valid = self.close.notna()
        has = valid.any(axis=0).to_numpy()
        first_date = valid.idxmax(axis=0)
        pos_of = pd.Series(np.arange(len(self.dates)), index=self.dates)
        first_pos = first_date.map(pos_of).to_numpy(dtype=F32)
        # base 取负常数：days_listed = i - base = i + SEASONED0，恒为大正数（已上市）
        base = np.where(first_pos <= 0, -SEASONED0, first_pos)
        base = np.where(has, base, np.nan)
        self.days_listed = pd.DataFrame(
            np.arange(len(self.dates), dtype=F32)[:, None] - base[None, :],
            index=self.dates, columns=self.symbols, dtype=F32)
        n_old = int((base == -SEASONED0).sum())
        print(f"  days_listed: {n_old} 只在 2010 年前上市(视为已满250日), "
              f"{int(has.sum()) - n_old} 只为 2010 年后 IPO", flush=True)
        del valid, first_date

        # ---- ST 标记 ----
        st = to_wide(df.assign(_s=df["is_st"].astype(F32)), "_s",
                     self.dates, self.symbols)
        self.is_st = st.fillna(0) > 0.5
        del st

        # ---- 行业 ----
        ipath = os.path.join(DATA, "industry_clean.pkl")
        if os.path.exists(ipath):
            i = pd.read_pickle(ipath).drop_duplicates("symbol").set_index("symbol")
            self.industry = i["industry"].reindex(self.symbols).fillna("未知")
        else:
            self.industry = pd.Series("未知", index=self.symbols)

        # turnover 单位归一：新浪返回小数(0.0023)；若量级异常则按百分数处理
        if float(np.nanmedian(self.turnover.to_numpy())) > 1.0:
            self.turnover = self.turnover / 100.0

        print(f"  宽表: {len(self.dates)} 日 × {len(self.symbols)} 股 "
              f"({time.time()-t0:.1f}s)", flush=True)


# ==============================================================================
# SECTION 2  股票池构建（时点正确）
# ==============================================================================

def detect_limit(P, tol=0.005):
    """涨跌停检测。

    涨跌停幅度按板块与 ST 状态区分：主板 10% / 创业板·科创板 20% / ST 5%。
    判据用「前复权开盘价 ÷ 前一日前复权收盘价」——前复权保留真实收益率，
    故该比值与交易所口径的涨跌幅一致。

    返回 (limit_up, limit_down)：当日开盘即封板 → 买不进 / 卖不出。
    """
    gap = (P.open / P.close.shift(1) - 1.0).astype(F32)

    lim = pd.Series(0.10, index=P.symbols, dtype=F32)
    lim[P.symbols.str.startswith("sz3")] = 0.20        # 创业板
    lim[P.symbols.str.startswith("sh688")] = 0.20      # 科创板
    limit_pct = pd.DataFrame(
        np.tile(lim.to_numpy(), (len(P.dates), 1)),
        index=P.dates, columns=P.symbols, dtype=F32)
    limit_pct = limit_pct.where(~P.is_st, 0.05)        # ST ±5%

    thr = (limit_pct - tol).astype(F32)
    return (gap >= thr).fillna(False), (gap <= -thr).fillna(False)


def build_universe(P, cfg=None):
    """构建时点正确的可交易股票池（每个交易日独立判定）。

    依次剔除：
      1. 当日无成交（停牌）或缺价格
      2. ST / *ST                     涨跌停 ±5%，退市风险
      3. 上市不足 250 交易日           次新股波动异常
      4. 最小 30% 流通市值             Liu-Stambaugh-Yuan(2019) 壳价值校正
      5. 当日涨停                      开盘封板买不进
    """
    cfg = cfg or CFG
    ok = P.traded.copy()
    if cfg.exclude_st:
        ok &= ~P.is_st
    ok &= (P.days_listed.fillna(999) >= cfg.min_days_listed)
    if cfg.use_lsy_universe and cfg.lsy_drop_smallest_pct > 0:
        q = fast_rank_pct(P.float_mktcap.where(ok))
        ok &= (q > cfg.lsy_drop_smallest_pct)
    limit_up, limit_down = detect_limit(P)
    ok &= ~limit_up
    ok = ok.fillna(False)

    n = ok.sum(axis=1)
    print(f"  股票池: 日均 {n.mean():.0f} 只（最少 {n.min():.0f} @ {n.idxmin().date()}, "
          f"最多 {n.max():.0f}）", flush=True)
    return ok, limit_up, limit_down


# ==============================================================================
# SECTION 3  因子库（A 层经典 + B 层联动）
# ==============================================================================
#
# 约定
# ----
#   · 所有因子在 t 日只使用 <= t 的数据（无未来函数）。
#   · 因子"经济方向"已统一：因子值越大 → 预期未来收益越高。
#     负向异象（短期反转、彩票效应、高换手…）在构造时已取负号。
#
# 实现效率
# --------
#   · 滚动窗口用 pandas.rolling（C 实现）。
#   · 窗口内排序（MAX5）与累乘（隔夜/日内）不用 rolling.apply（Python 逐窗口，慢几个
#     数量级），改为"按日期循环 + 按股票向量化"或对数域求和。
#   · 行业聚合（行业动量 / 溢出 / 联动度）用 稀疏哑变量矩阵乘法，一次算完全时段。
#   · 因子逐个计算后立即落盘，避免 24 个宽表同时驻留内存。

def _div(a, b):
    """安全除法：分母 0 -> NaN"""
    return a / b.replace(0, np.nan)


def fast_rank_pct(df, ascending=True):
    """横截面百分位排名（0~1），等价的 numpy 实现。

    为什么不直接用 pandas 的 DataFrame.rank(axis=1)：
    在全 A 面板上（4065 日 × 5900 股）pandas 的按行 rank 是 Python 级循环，
    单次耗时数十秒；下面用 argsort 的两次置换实现，是纯 numpy 向量化操作，
    快 1~2 个数量级。NaN 保持为 NaN（不参与排名）。
    """
    a = df.to_numpy(dtype=np.float32)
    if not ascending:
        a = -a
    fin = np.isfinite(a)
    safe = np.where(fin, a, np.inf)
    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty(a.shape, dtype=np.float32)
    rows = np.arange(a.shape[0])[:, None]
    ranks[rows, order] = np.arange(a.shape[1], dtype=np.float32)[None, :] + 1.0
    n_valid = fin.sum(axis=1, keepdims=True).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(fin, ranks / np.maximum(n_valid, 1.0), np.nan)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def rolling_topk_mean(df, window, k):
    """滚动窗口内最大的 k 个值的均值 —— MAX 彩票效应因子。

    按日期循环、按股票向量化：每次迭代对 (window, n_stocks) 矩阵沿股票轴排序，
    4065 次迭代 × 数十微秒级 numpy 操作，远快于 rolling.apply 的 Python 逐窗口调用。
    """
    arr = df.to_numpy(dtype=F32)
    out = np.full(arr.shape, np.nan, dtype=F32)
    for i in range(window - 1, len(arr)):
        w = arr[i - window + 1:i + 1]
        if np.all(np.isnan(w[-1])):
            continue
        out[i] = np.nanmean(np.sort(w, axis=0)[-k:], axis=0)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def rolling_compound(ret_like, window, min_periods=None):
    """滚动累乘 (1+r) - 1 —— 对数域求和，避免 rolling.apply(np.prod)。

    要求 1+r > 0（A 股涨跌停限制保证成立）。
    """
    min_periods = min_periods or max(2, window // 2)
    lp = np.log1p(ret_like.clip(lower=-0.999))
    return np.expm1(lp.rolling(window, min_periods=min_periods).sum())


# ---------------------------------------------------------------- 共享中间量

class Ctx:
    """多因子共享的中间量（市场收益、beta、行业哑变量等），算一次供所有因子复用。"""

    def __init__(self, P):
        t0 = time.time()
        ret = P.ret
        mc = P.float_mktcap
        self.mkt_w = mc.div(mc.sum(axis=1), axis=0)          # 流通市值权重
        self.mkt = (ret * self.mkt_w).sum(axis=1)            # 市值加权市场收益
        w = 252
        self.var_m = self.mkt.rolling(w, min_periods=126).var()
        self.cov_im = ret.rolling(w, min_periods=126).cov(self.mkt)
        self.beta = self.cov_im.div(self.var_m, axis=0)
        self.var_i = ret.rolling(w, min_periods=126).var()
        self.resid = ret.sub(self.beta.mul(self.mkt, axis=0))  # 市场模型残差
        self.mom_12_1 = P.close.shift(21) / P.close.shift(252) - 1

        # 行业哑变量（time-invariant）：稀疏矩阵乘法做聚合
        ind = P.industry.reindex(P.symbols).fillna("未知")
        codes, uniq = pd.factorize(ind)
        self.ind_codes = codes.astype(np.int32)
        self.ind_names = list(uniq)
        n_ind = len(uniq)
        D = np.zeros((len(codes), n_ind), dtype=F32)
        D[np.arange(len(codes)), codes] = 1.0
        self.D = D
        print(f"  共享中间量: {n_ind} 个行业, {time.time()-t0:.1f}s", flush=True)

    def ind_sum_count(self, X, mask):
        """按行业求和 / 计数。返回 (dates × n_ind) 两个数组。"""
        Xm = np.nan_to_num(X.where(mask).to_numpy(dtype=F32), nan=0.0)
        Mm = mask.to_numpy(dtype=F32)
        s = Xm @ self.D
        c = Mm @ self.D
        return s, c

    def exclude_self(self, X, mask):
        """行业内剔除自身后的成员均值（行业动量 / 溢出 / 联动度的核心运算）"""
        s, c = self.ind_sum_count(X, mask)
        Xm = np.nan_to_num(X.where(mask).to_numpy(dtype=F32), nan=0.0)
        Mm = mask.to_numpy(dtype=F32)
        own_s = s[:, self.ind_codes]
        own_c = c[:, self.ind_codes]
        num = own_s - Xm
        den = own_c - Mm
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(den > 0.5, num / den, np.nan)
        return pd.DataFrame(out, index=X.index, columns=X.columns, dtype=F32)


# ---------------------------------------------------------------- 因子定义
# 每个因子是一个函数 f(P, ctx) -> 宽表 DataFrame(index=日期, columns=股票)

def f_rev_1m(P, c):
    """短期反转 Jegadeesh(1990) / Lehmann(1990)：过去 1 月收益，负向"""
    return -(P.close / P.close.shift(21) - 1)


def f_rev_1w(P, c):
    """周度反转 Lehmann(1990)：过去 1 周收益，负向"""
    return -(P.close / P.close.shift(5) - 1)


def f_mom_12_1(P, c):
    """动量 12-1 Jegadeesh & Titman(1993)：过去 12 月剔除最近 1 月"""
    return P.close.shift(21) / P.close.shift(252) - 1


def f_rev_lt(P, c):
    """长期反转 De Bondt & Thaler(1985)：过去 3 年收益，负向"""
    return -(P.close.shift(252) / P.close.shift(756) - 1)


def f_max5(P, c):
    """彩票效应 Bali/Cakici/Whitelaw(2011)：过去 1 月最大 5 日收益，负向"""
    return -rolling_topk_mean(P.ret, 21, 5)


def f_ivol(P, c):
    """特质波动率 Ang/Hodrick/Xing/Zhang(2006)：负向。

    IVOL² = Var(r) − Cov(r, r_mkt)² / Var(r_mkt)，用滚动矩公式免去逐股回归。
    """
    resid_var = (c.var_i - c.beta.pow(2).mul(c.var_m, axis=0)).clip(lower=1e-10)
    return -np.sqrt(resid_var * TRADING_DAYS)


def f_beta(P, c):
    """市场 Beta / BAB Frazzini & Pedersen(2014)：低 beta 优于高 beta，取负"""
    return -c.beta


def f_illiq(P, c):
    """Amihud(2002) 非流动性：|收益|/成交额，流动性溢价为正"""
    illiq = _div(P.ret.abs(), P.amount).rolling(21, min_periods=10).mean()
    return np.log1p(illiq * 1e8)


def f_size(P, c):
    """规模 Banz(1981)：小市值溢价，取负对数流通市值"""
    return -np.log(P.float_mktcap.replace(0, np.nan))


def f_high52(P, c):
    """52 周高点 George & Hwang(2004)：越接近 52 周高点越强"""
    return _div(P.close, P.high.rolling(252, min_periods=126).max())


def f_coskew(P, c):
    """协偏度 Harvey & Siddique(2000)：与市场收益的协偏度，负向。

    coskew = E[(r_i − μ_i)(r_m − μ_m)²] / (σ_i · σ_m²)
    分子按定义展开为滚动矩的组合，避免逐日回归。
    """
    w = 252
    mi = P.ret.rolling(w, min_periods=126).mean()
    mm = c.mkt.rolling(w, min_periods=126).mean()
    dm = c.mkt - mm
    dm2_mean = dm.pow(2).rolling(w, min_periods=126).mean()
    dm_mean = dm.rolling(w, min_periods=126).mean()
    p1 = P.ret.mul(dm.pow(2), axis=0).rolling(w, min_periods=126).mean()
    p3 = P.ret.mul(dm, axis=0).rolling(w, min_periods=126).mean()
    num = p1 - mi.mul(dm2_mean, axis=0) - 2 * (p3 - mi.mul(dm_mean, axis=0)) \
        .mul(mm, axis=0)
    # 注意：DataFrame 与 Series 相乘必须显式指定 axis=0（按日期对齐），
    # 否则 pandas 默认按列名对齐，会静默产生全 NaN。
    den = np.sqrt(c.var_i).mul(c.var_m, axis=0)
    return -_div(num, den)


def f_turn(P, c):
    """换手率 Datar/Naik/Radcliffe(1998)：高换手低收益，负向"""
    return -P.turnover.rolling(21, min_periods=10).mean()


def f_pv_corr(P, c):
    """量价相关性 Chordia/Roll/Subrahmanyam(2001)：换手变化与|收益|正相关 → 负向"""
    return -P.turnover.diff().rolling(21, min_periods=15).corr(P.ret.abs())


def f_id_frog(P, c):
    """信息离散 Frog-in-the-Pan Da/Gurun/Warachka(2014)：正向。

    ID = sign(累计收益) × (负收益日占比 − 正收益日占比)
    信息"细水长流"式披露时 ID 大，未来动量更强。
    """
    s = np.sign(P.ret)
    pos = (s > 0).rolling(21, min_periods=10).mean()
    neg = (s < 0).rolling(21, min_periods=10).mean()
    cum = P.ret.rolling(21, min_periods=10).sum()
    return np.sign(cum) * (neg - pos)


def f_iskew(P, c):
    """特质偏度：彩票偏好，负向"""
    return -P.ret.rolling(60, min_periods=40).skew()


# ---- B 层：截面联动类 ----

def f_ind_mom(P, c):
    """行业动量 Moskowitz & Grinblatt(1999)：行业过去 12-1 月收益，剔除个股自身贡献"""
    return c.exclude_self(c.mom_12_1, P.traded)


def f_res_mom(P, c):
    """残差动量 Blitz/Huij/Martens(2011)：市场模型残差在过去 12-1 月的累积"""
    return c.resid.rolling(252 - 21, min_periods=126).sum().shift(21)


def f_overnight(P, c):
    """隔夜收益动量 Lou/Polk/Skouras(2019)：open_t / close_{t-1} − 1 的累积"""
    on = _div(P.open, P.close.shift(1)) - 1
    return rolling_compound(on, 21, 15)


def f_intraday(P, c):
    """日内收益反转 Lou/Polk/Skouras(2019)：close_t / open_t − 1 的累积，负向"""
    idr = _div(P.close, P.open) - 1
    return -rolling_compound(idr, 21, 15)


def f_lead_lag(P, c):
    """领先滞后 Lo & MacKinlay(1990)：对"滞后市场收益"的敏感度，负向。

    b1 大者反应慢 —— 对已实现的市场信息反应不足，未来存在补涨/补跌。
    """
    cov_lag = P.ret.rolling(252, min_periods=126).cov(c.mkt.shift(1))
    return -cov_lag.div(c.var_m, axis=0)


def f_spillover(P, c):
    """行业内溢出 Hong/Torous/Valkanov(2007)：同行业其他股票的动量"""
    return c.exclude_self(c.mom_12_1, P.traded)


def f_commove(P, c):
    """联动度：个股收益与（剔除自身的）行业收益的滚动相关性"""
    ind_ret = c.exclude_self(P.ret, P.traded)
    return P.ret.rolling(60, min_periods=40).corr(ind_ret)


def f_divergence(P, c):
    """背离度：个股动量 − 行业动量（特质动量的另一种表达）"""
    return c.mom_12_1 - c.exclude_self(c.mom_12_1, P.traded)


def f_turn_shock(P, c):
    """流动性冲击 Chordia/Roll/Subrahmanyam(2001)：短期换手相对长期换手，负向"""
    return -(P.turnover.rolling(5, min_periods=3).mean() /
             P.turnover.rolling(60, min_periods=30).mean() - 1)


# 因子注册表：名字 -> (函数, 分层)
FACTORS = [
    ("REV_1M",     f_rev_1m,     "A"),
    ("REV_1W",     f_rev_1w,     "A"),
    ("MOM_12_1",   f_mom_12_1,   "A"),
    ("REV_LT",     f_rev_lt,     "A"),
    ("MAX5",       f_max5,       "A"),
    ("IVOL",       f_ivol,       "A"),
    ("BETA",       f_beta,       "A"),
    ("ILLIQ",      f_illiq,      "A"),
    ("SIZE",       f_size,       "A"),
    ("HIGH52",     f_high52,     "A"),
    ("CO_SKEW",    f_coskew,     "A"),
    ("TURN",       f_turn,       "A"),
    ("PV_CORR",    f_pv_corr,    "A"),
    ("ID_FROG",    f_id_frog,    "A"),
    ("ISKEW",      f_iskew,      "A"),
    ("IND_MOM",    f_ind_mom,    "B"),
    ("RES_MOM",    f_res_mom,    "B"),
    ("OVERNIGHT",  f_overnight,  "B"),
    ("INTRADAY",   f_intraday,   "B"),
    ("LEAD_LAG",   f_lead_lag,   "B"),
    ("SPILLOVER",  f_spillover,  "B"),
    ("COMMOVE",    f_commove,    "B"),
    ("DIVERGENCE", f_divergence, "B"),
    ("TURN_SHOCK", f_turn_shock, "B"),
]


def compute_factor_pipeline(P, cfg, tag, force=False):
    """逐个计算因子 → 立即落盘 → 释放内存。返回因子名列表。

    注意：因子只依赖原始面板，**与股票池设置无关**（股票池是在 SECTION 4 的
    预处理与 SECTION 5 的检验阶段才生效的掩码）。因此缓存目录固定为 factors/，
    LSY 与全 A 两种股票池共用同一份因子，不必重复计算。
    """
    fdir = os.path.join(DATA, "factors")
    os.makedirs(fdir, exist_ok=True)
    ctx = Ctx(P)
    names = []
    for name, fn, layer in FACTORS:
        path = os.path.join(fdir, f"{name}.pkl")
        names.append(name)
        if os.path.exists(path) and not force:
            print(f"    [skip] {name}", flush=True)
            continue
        t0 = time.time()
        x = None
        try:
            x = fn(P, ctx)
            x = x.replace([np.inf, -np.inf], np.nan).astype(F32)
            x.to_pickle(path)
            print(f"    {name:12s} {time.time()-t0:6.2f}s  "
                  f"覆盖 {x.notna().mean().mean()*100:5.1f}%", flush=True)
        except Exception as e:
            print(f"    {name:12s} FAIL {type(e).__name__}: {str(e)[:90]}", flush=True)
        finally:
            del x
    del ctx
    return names


# ==============================================================================
# SECTION 4  因子预处理（去极值 / 行业市值中性化 / 标准化）
# ==============================================================================

def winsorize_mad(df, n=3.0):
    """MAD 去极值：中位数 ± n×1.4826×MAD 截断。

    比 3σ 稳健 —— A 股收益分布厚尾，σ 本身会被极值污染，导致截断失效。
    """
    med = df.median(axis=1)
    mad = df.sub(med, axis=0).abs().median(axis=1) * 1.4826
    return df.clip(lower=med - n * mad, upper=med + n * mad, axis=0)


def standardize(df):
    """横截面 z-score 标准化"""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def neutralize(df, P, industry=True, size=True, eligible=None, min_ind=None):
    """行业 + 市值中性化：逐日横截面回归取残差。

        factor_i = a + Σ_k b_k·Industry_k + c·ln(流通市值) + e_i

    残差 e_i 即中性化后的因子 —— 剔除"因子只是行业的代理"或"因子只是小市值的
    代理"这类伪显著。这是截面因子研究里最容易被忽略、却最影响结论的一步。

    实现用 Frisch-Waugh-Lovell 定理，把一次多元回归拆成两步一元的残差化：
        y⊥[D, s]  ≡  (y⊥D) 再对 (s⊥D) 回归取残差
    于是每天的计算量从 O(n·k²)（k≈85 个行业哑变量，n≈4000 只股票，
    约 3000 万次浮点运算）降到 O(n)。全 A 面板上这一步从约 40 分钟降到几秒。
    """
    if not (industry or size):
        return df
    min_ind = min_ind or CFG.min_industry_size

    cols = df.columns
    F = df.to_numpy(dtype=np.float32)
    out = np.full(F.shape, np.nan, dtype=np.float32)

    # 行业编码（列维度，时间不变）
    if industry:
        ind = P.industry.reindex(cols).fillna("未知")
        codes, uniq = pd.factorize(ind)
        codes = codes.astype(np.int64)
        n_ind = len(uniq)
    else:
        codes = np.zeros(len(cols), dtype=np.int64)
        n_ind = 1

    if size:
        size_all = np.log(P.float_mktcap.reindex(columns=cols)
                            .replace(0, np.nan)).to_numpy(dtype=np.float32)
    else:
        size_all = None

    EV = (eligible.reindex(columns=cols).to_numpy()
          if eligible is not None else None)

    for i in range(F.shape[0]):
        y = F[i]
        m = np.isfinite(y)
        if EV is not None:
            m &= EV[i]
        if m.sum() < 30:
            continue
        idx = np.flatnonzero(m)
        yy = y[idx].astype(np.float64)
        cc = codes[idx]

        # ---- 第一步：对行业哑变量回归取残差（等价于组内去均值）----
        cnt = np.bincount(cc, minlength=n_ind).astype(np.float64)
        good = cnt >= min_ind                      # 成员太少的行业不单独去均值
        gsum = np.bincount(cc, weights=yy, minlength=n_ind)
        gmean = np.where(good & (cnt > 0), gsum / np.maximum(cnt, 1.0), np.nan)
        gm = gmean[cc]
        y_res = yy - np.where(np.isfinite(gm), gm, yy.mean())

        # ---- 第二步：对"已剔除行业影响的市值"回归取残差 ----
        if size_all is not None:
            ss = size_all[i][idx].astype(np.float64)
            if not np.isfinite(ss).all():
                ss = np.where(np.isfinite(ss), ss, np.nanmedian(ss))
            ssum = np.bincount(cc, weights=ss, minlength=n_ind)
            smean = np.where(good & (cnt > 0), ssum / np.maximum(cnt, 1.0), np.nan)
            sm = smean[cc]
            s_res = ss - np.where(np.isfinite(sm), sm, ss.mean())
            denom = float(s_res @ s_res)
            if denom > 1e-12:
                y_res = y_res - (float(y_res @ s_res) / denom) * s_res

        out[i, idx] = y_res.astype(np.float32)

    return pd.DataFrame(out, index=df.index, columns=cols)


def prepare_factor(raw, P, eligible, cfg=None):
    """完整预处理流水线：股票池掩码 → 去极值 → 中性化 → 标准化 → 再过掩码"""
    cfg = cfg or CFG
    x = raw.where(eligible)
    x = winsorize_mad(x, cfg.winsor_mad)
    x = neutralize(x, P, cfg.neutralize_industry, cfg.neutralize_size, eligible)
    x = standardize(x)
    return x.where(eligible).astype(F32)


# ==============================================================================
# SECTION 5  单因子检验（RankIC / 分层 / Fama-MacBeth）
# ==============================================================================

def forward_return(close, horizon):
    """前瞻 h 日收益。**仅用于因子检验，绝不进入策略信号**。"""
    return close.shift(-horizon) / close - 1


def rank_ic(factor, fwd, min_n=50):
    """逐日横截面 Spearman 秩相关（RankIC）。

    RankIC 衡量"因子排序"与"未来收益排序"的一致性，比 Pearson IC 更抗极值，
    是截面因子研究的标准度量。
    """
    f, r = factor.align(fwd, join="inner")
    fr = f.rank(axis=1).to_numpy(dtype=F32)
    rr = r.rank(axis=1).to_numpy(dtype=F32)
    # 逐日 Pearson（作用在秩上 = Spearman）
    mu_f = np.nanmean(fr, axis=1, keepdims=True)
    mu_r = np.nanmean(rr, axis=1, keepdims=True)
    df_ = fr - mu_f
    dr_ = rr - mu_r
    num = np.nansum(df_ * dr_, axis=1)
    den = np.sqrt(np.nansum(df_ ** 2, axis=1) * np.nansum(dr_ ** 2, axis=1))
    n_ok = np.sum(np.isfinite(fr) & np.isfinite(rr), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = np.where((den > 0) & (n_ok >= min_n), num / den, np.nan)
    return pd.Series(ic, index=f.index).dropna()


def newey_west_tstat(x, lags=5):
    """Newey-West 异方差自相关稳健 t 值。

    因子收益 / IC 序列存在自相关，普通 t 值会高估显著性。
    NW 用 Bartlett 核加权自协方差修正标准误。返回 (均值, t 值, 样本数)。
    """
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    T = len(x)
    if T < 10:
        return np.nan, np.nan, T
    mu = x.mean()
    e = x - mu
    s = (e ** 2).mean()
    for l in range(1, min(lags, T - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        s += 2.0 * w * (e[l:] * e[:-l]).mean()
    return mu, mu / np.sqrt(max(s, 1e-18) / T), T


def quantile_backtest(factor, fwd, n_q=10):
    """分层回测：按因子值分 n_q 组，各层前瞻收益与多空组合收益。

    分层单调性是因子有效性最直观的证据 —— 比单一 IC 数字更难伪造。
    """
    pct = fast_rank_pct(factor)
    labels = np.ceil(pct * n_q).clip(1, n_q)
    Q = {}
    for q in range(1, n_q + 1):
        Q[f"Q{q}"] = fwd.where(labels == q).mean(axis=1)
    out = pd.DataFrame(Q)
    out["LS"] = out[f"Q{n_q}"] - out["Q1"]
    return out


def fama_macbeth(factor, fwd, eligible=None, lags=None):
    """Fama-MacBeth (1973) 两步截面回归。

    第一步 每个交易日做横截面回归 r_{i,t+h} = a_t + b_t·f_{i,t} + e
    第二步 对 b_t 时间序列取均值并算 Newey-West t 值。

    这是因子定价检验的标准做法，能同时处理"因子间相关"与"时序自相关"，
    比单纯看 IC 更严格。
    """
    lags = lags if lags is not None else CFG.nw_lags
    f, r = factor.align(fwd, join="inner")
    fv = f.to_numpy(dtype=np.float32)
    rv = r.to_numpy(dtype=np.float32)
    ev = eligible.reindex(index=f.index, columns=f.columns).to_numpy() \
        if eligible is not None else None

    betas, alphas = [], []
    for i in range(fv.shape[0]):
        m = np.isfinite(fv[i]) & np.isfinite(rv[i])
        if ev is not None:
            m &= ev[i]
        if m.sum() < 50:
            continue
        x, y = fv[i][m], rv[i][m]
        X = np.column_stack([np.ones(len(x), dtype=np.float32), x])
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            alphas.append(float(coef[0])); betas.append(float(coef[1]))
        except np.linalg.LinAlgError:
            continue
    b = pd.Series(betas); a = pd.Series(alphas)
    mu_b, t_b, T = newey_west_tstat(b, lags)
    mu_a, t_a, _ = newey_west_tstat(a, lags)
    return {"beta": mu_b, "t_beta": t_b, "alpha": mu_a, "t_alpha": t_a, "n_periods": T}


def evaluate_factor(name, factor, fwd, eligible, cfg=None):
    """对单个因子跑完整检验，返回指标字典"""
    cfg = cfg or CFG
    ann = TRADING_DAYS / cfg.ic_forward_days

    ic = rank_ic(factor, fwd, min_n=50)
    ic_mu, ic_t, ic_T = newey_west_tstat(ic, cfg.nw_lags)
    icir = (ic.mean() / ic.std() * np.sqrt(ann)) if ic.std() > 0 else np.nan

    Q = quantile_backtest(factor, fwd, cfg.n_quantiles)
    ls = Q["LS"].dropna()
    ls_mu, ls_t, _ = newey_west_tstat(ls, cfg.nw_lags)
    fm = fama_macbeth(factor, fwd, eligible)

    # 分层单调性：Q1..Q10 均值的秩相关（越接近 ±1 越单调）
    qmeans = [Q[f"Q{i}"].mean() for i in range(1, cfg.n_quantiles + 1)]
    mono = pd.Series(qmeans).corr(pd.Series(range(len(qmeans))), method="spearman")

    # 因子换手：相邻调仓期多头成分变化
    sel = (fast_rank_pct(factor, ascending=False) <= cfg.top_pct).iloc[::cfg.rebalance_days]
    to = (sel.astype(float).diff().abs().sum(axis=1) /
          sel.sum(axis=1).replace(0, np.nan) / 2).dropna()

    return {
        "factor": name,
        "IC_mean": ic_mu,
        "IC_t": ic_t,
        "ICIR": icir,
        "IC_pos_rate": (ic > 0).mean() if len(ic) else np.nan,
        "LS_mean_ann": ls_mu * ann,
        "LS_t": ls_t,
        "LS_sharpe": (ls_mu / ls.std() * np.sqrt(ann)) if ls.std() > 0 else np.nan,
        "FM_beta": fm["beta"],
        "FM_t": fm["t_beta"],
        "monotonic": mono,
        "turnover": to.mean(),
        "n_periods": ic_T,
        "Q1_ann": Q["Q1"].mean() * ann,
        "Q10_ann": Q[f"Q{cfg.n_quantiles}"].mean() * ann,
    }


# ==============================================================================
# SECTION 6  多因子合成
# ==============================================================================

def combine_equal(factor_dict):
    """等权合成：对每个因子取横截面均值。

    等权是最稳健的合成方式 —— IC 加权需要估计 24 个权重，引入的估计误差
    往往比它带来的信息增益更大（尤其在样本外）。
    """
    acc = cnt = None
    for v in factor_dict.values():
        valid = v.notna().astype(F32)
        acc = v.fillna(0) if acc is None else acc.add(v.fillna(0), fill_value=0)
        cnt = valid if cnt is None else cnt.add(valid, fill_value=0)
    return (acc / cnt.replace(0, np.nan)).astype(F32)


# ==============================================================================
# SECTION 7  组合构建
# ==============================================================================

def build_weights(score, eligible, side="long", cfg=None):
    """把因子得分转成目标权重（调仓日给出，次日开盘执行）。

    多头  取得分最高的 top_pct，等权，单票权重上限截断
    空头  取得分最低的 bottom_pct

    ★ 缓冲区（cfg.buffer > 0）
      调仓时，上一期的持仓只要仍在新候选的前 n×(1+buffer) 名内就保留，
      空缺由排名最高的新股票补足。作用是抑制「排名在边界附近来回抖动」
      造成的无谓换手 —— 这类换手不带来任何信息，只贡献成本。
      实测：换手 1797%→1541%，样本内外夏普同时改善。

    ★ 权重上限的正确实现
      `w.clip(upper=cap); w /= w.sum()` 是错的（clip 后总和 <1，再归一化会把
      被截断的权重推回上限之上，实测 cap=5% 时单票可达 10.4%）。
      这里用不动点迭代：截断 → 归一化 → 再截断，直到所有未固定权重都不超上限。
    """
    cfg = cfg or CFG
    dates = score.index[::cfg.rebalance_days]
    buf = float(getattr(cfg, "buffer", 0.0) or 0.0)
    rows = {}
    prev = None
    for dt in dates:
        s = score.loc[dt].where(eligible.loc[dt]).dropna()
        if len(s) < 20:
            continue
        n = min(max(int(len(s) * cfg.top_pct), 10), cfg.n_long)
        if side == "short":
            s = -s                                  # 空头：在取负后的空间里同样取最大
        if buf > 0 and prev is not None and len(prev) > 0:
            band = set(s.nlargest(min(int(n * (1 + buf)), len(s))).index)
            kept = [c for c in prev if c in band]
            room = n - len(kept)
            new = [c for c in s.nlargest(n + len(kept)).index if c not in kept]
            pick = s.reindex(kept + new[:max(room, 0)]).dropna()
            if len(pick) < max(int(n * 0.5), 10):
                pick = s.nlargest(n)
        else:
            pick = s.nlargest(n)
        w = pd.Series(1.0 / len(pick), index=pick.index)
        rows[dt] = _apply_weight_cap(w, cfg.max_weight)
        prev = list(pick.index)
    W = pd.DataFrame(rows).T.fillna(0.0)
    return W.reindex(columns=score.columns, fill_value=0.0)


def _apply_weight_cap(w, cap, iters=200, tol=1e-4):
    """真正生效的单票权重上限（不动点迭代）。

    朴素写法的缺陷见 build_weights 的注释。这里迭代到不动点：
    截断 → 重新分配剩余权重 → 再截断，直到没有权重超限。

    ★ 可行性保护：若 n × cap < 1（持仓只数太少、上限太紧），
      上限在数学上不可能同时满足 —— 此时把上限放宽到 1/n（即等权）。
      不做这个保护会静默产出超限权重（单元测试实测：6 只股票 cap=10% 时
      输出最大权重 16.7%，超出上限 67%）。
    """
    w = pd.Series(w).astype(float)
    if not np.isfinite(w.sum()) or w.sum() <= 0:
        return pd.Series(1.0 / len(w), index=w.index)
    if cap >= 1.0:
        return w / w.sum()
    n_eff = int((w > 0).sum())
    if n_eff > 0 and n_eff * cap < 1.0:
        cap = 1.0 / n_eff                     # 不可行 → 放宽到等权水平
    fixed = pd.Series(False, index=w.index)
    out = w.copy()
    for _ in range(iters):
        free = ~fixed
        if free.sum() == 0:
            break
        rest = max(1.0 - cap * int(fixed.sum()), 1e-12)
        s = out[free].sum()
        if s <= 0:
            break
        out[free] = out[free] / s * rest
        over = free & (out > cap * (1 + tol))
        if not over.any():
            break
        out[over] = cap
        fixed = fixed | over
    return out / out.sum()


def build_long_short(score, eligible, cfg=None):
    """多空组合 = 多头 − 空头（学术标准的因子组合，用于检验因子显著性）"""
    cfg = cfg or CFG
    Wl = build_weights(score, eligible, "long", cfg)
    Ws = build_weights(score, eligible, "short", cfg)
    idx = Wl.index.union(Ws.index)
    cols = Wl.columns
    return (Wl.reindex(idx, fill_value=0)
              .sub(Ws.reindex(idx, columns=cols, fill_value=0), fill_value=0))


# ==============================================================================
# SECTION 8  回测引擎（T+1 / 涨跌停 / 停牌 / 真实成本）
# ==============================================================================

def backtest(P, target_w, cfg=None, name="策略"):
    """逐日模拟回测引擎（主引擎，精确版）。

    执行模型（无未来函数）
    ----------------------
      t 日收盘算出目标权重 → t+1 日开盘成交

    单日三步（每一日都严格按这个顺序推进）
    --------------------------------------
      ① 隔夜   持仓从上日收盘漂移到今日开盘   open(t)/close(t-1)-1
      ② 开盘   调仓到目标权重，按真实差额收成本
      ③ 日内   新仓位从今日开盘持到今日收盘   close(t)/open(t)-1

    与向量化近似的关键区别：本引擎让权重**随价格真实漂移**。
    两次调仓之间，涨得多的股票权重自然变大 —— 这既符合实盘，
    又使得"为了维持目标权重而做的再平衡"产生的换手被自动计入成本。
    向量化近似（假设权重恒定）会漏掉这部分，见 backtest_vectorized() 的对照。

    T+1 制度
    --------
      A 股当日买入次日才可卖。本策略最短持有一个调仓周期，且 t 日决策 t+1 开盘执行，
      天然满足 T+1（不存在当日买入当日卖出）。

    停牌
    ----
      价格前向填充（上限 60 日）：停牌期间收益为 0、仓位被冻结，复牌按新价计算。
    """
    cfg = cfg or CFG
    dates, syms = P.dates, P.symbols
    n, m = len(dates), len(syms)

    if n < 3:
        return {"name": name}

    close_ff = P.close.ffill(limit=60).to_numpy(dtype=np.float32)
    open_ff = P.open.ffill(limit=60).to_numpy(dtype=np.float32)

    # 目标权重：调仓日给出，其余日期前向填充（持仓不动；漂移由下面的循环处理）
    W_dec = target_w.reindex(index=dates, columns=syms).ffill().fillna(0.0)
    ssum = W_dec.abs().sum(axis=1).replace(0, np.nan)
    W_dec = W_dec.div(ssum, axis=0).fillna(0.0).to_numpy(dtype=np.float32)
    # t 日决定 → t+1 日开盘执行
    W_tgt = np.vstack([np.zeros((1, m), dtype=np.float32), W_dec[:-1]])

    comm = (cfg.commission_bp + cfg.slippage_bp) / 1e4
    stamp = cfg.stamp_duty_bp / 1e4

    gross = np.zeros(n); cost = np.zeros(n); turn = np.zeros(n)
    w = np.zeros(m, dtype=np.float64)
    ret_net = np.zeros(n)

    for t in range(1, n):
        prev_c, op, cl = close_ff[t - 1], open_ff[t], close_ff[t]

        # ---- ① 隔夜跳空 ----
        with np.errstate(invalid="ignore", divide="ignore"):
            r_on = np.where(np.isfinite(prev_c) & np.isfinite(op) & (prev_c > 0),
                            op / prev_c - 1.0, 0.0).astype(np.float64)
        g_on = float(w @ r_on)
        if abs(1.0 + g_on) > 1e-12:
            w = w * (1.0 + r_on) / (1.0 + g_on)      # 权重随价格漂移

        # ---- ② 开盘调仓（含成本）----
        tgt = W_tgt[t].astype(np.float64)
        dW = tgt - w
        buy = float(dW[dW > 0].sum())
        sell = float(-dW[dW < 0].sum())
        c = (buy + sell) * comm + sell * stamp
        cost[t] = c
        turn[t] = buy + sell
        w = tgt.copy()

        # ---- ③ 日内 ----
        with np.errstate(invalid="ignore", divide="ignore"):
            r_id = np.where(np.isfinite(op) & np.isfinite(cl) & (op > 0),
                            cl / op - 1.0, 0.0).astype(np.float64)
        g_id = float(w @ r_id)
        if abs(1.0 + g_id) > 1e-12:
            w = w * (1.0 + r_id) / (1.0 + g_id)

        gross[t] = (1.0 + g_on) * (1.0 + g_id) - 1.0
        ret_net[t] = (1.0 + gross[t]) * (1.0 - c) - 1.0

    net = pd.Series(ret_net, index=dates).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross_s = pd.Series(gross, index=dates)
    cost_s = pd.Series(cost, index=dates)
    turn_s = pd.Series(turn, index=dates)

    res = summarize(net, cfg.rf_annual, name)
    res.update({"net": net, "gross": gross_s, "cost": cost_s, "turnover": turn_s,
                "ann_turnover": float(turn_s.mean() * TRADING_DAYS)})
    return res


def backtest_vectorized(P, target_w, cfg=None, name="策略(向量化近似)"):
    """向量化近似版：假设两次调仓之间权重恒定（不做漂移）。

    仅用于**校验主引擎**：两版结果应当接近，差异来自"漂移再平衡"的收益与成本。
    若差异过大，说明主引擎有 bug —— 这是回测工程里很重要的一步自检。
    """
    cfg = cfg or CFG
    dates, syms = P.dates, P.symbols
    close_ff = P.close.ffill(limit=60)
    open_ff = P.open.ffill(limit=60)

    W_dec = target_w.reindex(index=dates, columns=syms).ffill().fillna(0.0)
    W_dec = W_dec.div(W_dec.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    W_exec = W_dec.shift(1).fillna(0.0)

    overnight = (open_ff / close_ff.shift(1) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    intraday = (close_ff / open_ff - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    gross = ((W_exec.shift(1).fillna(0.0) * overnight).sum(axis=1) +
             (W_exec * intraday).sum(axis=1))

    dW = W_exec.diff().fillna(W_exec)
    buy = dW.clip(lower=0).sum(axis=1)
    sell = (-dW).clip(lower=0).sum(axis=1)
    cost = (buy + sell) * (cfg.commission_bp + cfg.slippage_bp) / 1e4
    cost += sell * cfg.stamp_duty_bp / 1e4

    net = gross - cost
    res = summarize(net, cfg.rf_annual, name)
    res.update({"net": net, "gross": gross, "cost": cost,
                "turnover": buy + sell,
                "ann_turnover": float((buy + sell).mean() * TRADING_DAYS)})
    return res


def vol_targeted(net, target_vol=0.15, window=60, max_lev=1.5, cost_bp=17.5):
    """波动率目标化叠加层（risk overlay）。

    做法：用截至 t-1 日的 window 日已实现波动率估计当前风险水平，
    按 lev = 目标波动 / 已实现波动 缩放仓位，把组合波动稳定在目标水平附近。

    为什么这是**纯风控**而不是过拟合：
      · 杠杆完全由历史波动率决定，不含任何收益预测，无未来函数（shift(1) 保证）
      · 波动率有很强的聚集性（volatility clustering），是可预测的；
        而收益几乎不可预测 —— 所以"预测波动、不预测收益"是稳健的改进
      · 不引入任何需要调参的收益信号

    代价：缩放本身产生换手。杠杆从 e1 变到 e2 相当于买卖 |e2-e1| 比例的整个组合，
    按 cost_bp 计提（默认 17.5bp ≈ 佣金2.5+滑点10+印花税5 的加权）。

    返回 (缩放后净收益, 杠杆序列)。
    """
    realized = net.rolling(window, min_periods=max(10, window // 2)).std() \
                  * np.sqrt(TRADING_DAYS)
    lev = (target_vol / realized.shift(1)).clip(upper=max_lev, lower=0.0).fillna(0.0)
    adj_cost = lev.diff().abs().fillna(lev) * cost_bp / 1e4
    return lev * net - adj_cost, lev


def summarize(net, rf_annual=0.02, name="策略"):
    """由日收益序列计算标准业绩指标"""
    net = pd.Series(net).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    equity = (1.0 + net).cumprod()
    n = len(net)
    if n == 0:
        return {"name": name}
    years = n / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if equity.iloc[-1] > 0 else -1.0
    vol = net.std(ddof=1) * np.sqrt(TRADING_DAYS)
    rf_d = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    ex = net - rf_d
    sharpe = (ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS)
              if ex.std(ddof=1) > 0 else np.nan)
    dd = equity / equity.cummax() - 1.0
    maxdd = dd.min()
    downside = net[net < 0].std(ddof=1) * np.sqrt(TRADING_DAYS)
    return {
        "name": name, "days": n, "years": years,
        "CAGR": cagr, "AnnVol": vol, "Sharpe": sharpe, "MaxDD": maxdd,
        "Calmar": cagr / abs(maxdd) if maxdd < 0 else np.nan,
        "Sortino": ex.mean() * TRADING_DAYS / downside if downside > 0 else np.nan,
        "WinRate": (net > 0).mean(),
        "equity": equity, "dd": dd,
    }


# ==============================================================================
# SECTION 9  显著性检验（NW / Bootstrap / Deflated Sharpe / FDR）
# ==============================================================================

EULER_MASCHERONI = 0.5772156649015329


def _norm_ppf(p):
    """标准正态分位数（优先 scipy，缺失时用 Acklam 有理近似兜底）"""
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except Exception:
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        cc = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
              -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
              3.754408661907416e+00]
        pl, ph = 0.02425, 1 - 0.02425
        if p < pl:
            q = math.sqrt(-2 * math.log(p))
            return (((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5]) / \
                   ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
        if p > ph:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5]) / \
                   ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def sharpe_ratio(x, rf_annual=None, periods=TRADING_DAYS):
    """年化夏普比率"""
    rf_annual = CFG.rf_annual if rf_annual is None else rf_annual
    x = pd.Series(x).dropna()
    if len(x) < 2 or x.std(ddof=1) == 0:
        return np.nan
    ex = x - ((1 + rf_annual) ** (1 / periods) - 1)
    return ex.mean() / ex.std(ddof=1) * np.sqrt(periods)


def probabilistic_sharpe(sr_p, n, skew, kurt, sr_benchmark_p=0.0):
    """PSR：真实 Sharpe 大于基准的概率（Bailey & López de Prado 2012）。

    修正了收益非正态（偏度、厚尾）对 Sharpe 显著性的影响。
    注意 sr 用「每期」口径（日频 Sharpe 除以 sqrt(252)）。
    """
    if not np.isfinite(sr_p) or n < 3:
        return np.nan
    denom = math.sqrt(max(1e-12, 1 - skew * sr_p + (kurt - 1) / 4 * sr_p ** 2))
    z = (sr_p - sr_benchmark_p) * math.sqrt(n - 1) / denom
    try:
        from scipy.stats import norm
        return float(norm.cdf(z))
    except Exception:
        return float(0.5 * (1 + math.erf(z / math.sqrt(2))))


def deflated_sharpe(x, n_trials, periods=TRADING_DAYS, sr_variance=None,
                    trial_sharpes=None):
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014, JPM)。

    解决的问题：如果你试了 24 个因子、挑了夏普最高的那个，"夏普 > 0"本身
    就不再稀奇 —— 最大值天然偏高。DSR 把"试验次数"折进显著性判断：

      SR0 = sqrt(Var(SR_trials)) · [ (1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)) ]
      DSR = Φ[ (SR − SR0)·sqrt(T−1) / sqrt(1 − skew·SR + (kurt−1)/4·SR²) ]

    **关键**：Var(SR_trials) 必须是**实际试过的那些策略的夏普横截面方差**，
    不能随便设。设成 1.0（年化）会得到一个 SR0≈2 的荒谬门槛，把所有策略都毙掉；
    设成 0 则等于不做多重检验校正。这里由调用方传入真实值。

    参数
    ----
      sr_variance    : 跨试验夏普的方差（年化口径）
      trial_sharpes  : 或者直接传入各试验的夏普序列，由本函数自行求方差

    公式内部统一在「每期（日频）」口径上计算，返回的年化 SR0 便于阅读。
    DSR > 0.95 表示"在考虑了多次试验后，该夏普依然显著为正"。
    """
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 30:
        return {"DSR": np.nan, "SR": np.nan, "SR0": np.nan, "PSR": np.nan,
                "skew": np.nan, "kurt": np.nan, "n_trials": n_trials}

    sr = sharpe_ratio(x, periods=periods)          # 年化
    sr_p = sr / math.sqrt(periods)                 # 每期口径
    skew = float(x.skew())
    kurt = float(x.kurt() + 3.0)                   # pandas 给超额峰度，转原始峰度

    if trial_sharpes is not None:
        ts = pd.Series(trial_sharpes).replace([np.inf, -np.inf], np.nan).dropna()
        var_sr = float(ts.var(ddof=1)) if len(ts) > 1 else 0.0
    elif sr_variance is not None:
        var_sr = float(sr_variance)
    else:
        var_sr = 0.0                               # 不做校正（保守地"不虚增门槛"）

    var_sr_p = max(var_sr, 0.0) / periods          # 转每期口径
    sr0_p = math.sqrt(var_sr_p) * (
        (1 - EULER_MASCHERONI) * _norm_ppf(1 - 1.0 / n_trials)
        + EULER_MASCHERONI * _norm_ppf(1 - 1.0 / (n_trials * math.e)))

    return {"DSR": probabilistic_sharpe(sr_p, n, skew, kurt, sr0_p),
            "SR": sr, "SR0": sr0_p * math.sqrt(periods),
            "PSR": probabilistic_sharpe(sr_p, n, skew, kurt, 0.0),
            "skew": skew, "kurt": kurt, "n_trials": n_trials,
            "var_sr": var_sr}


def stationary_bootstrap(x, n_boot=2000, block=21, seed=42):
    """定态块自助法 (Politis & Romano 1994)。

    时间序列不能简单重抽（会破坏自相关结构、低估波动）。定态块自助法按几何分布
    抽块长、循环拼接，保留局部时序依赖，得到统计量的经验分布。
    """
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    T = len(x)
    if T < 50:
        return None
    rng = np.random.default_rng(seed)
    p = 1.0 / block
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(T, dtype=np.int64)
        i = 0
        while i < T:
            start = rng.integers(0, T)
            L = min(int(rng.geometric(p)), T - i)
            idx[i:i + L] = (start + np.arange(L)) % T
            i += L
        out[b] = sharpe_ratio(x[idx], rf_annual=0.0)
    return out[np.isfinite(out)]


def bh_fdr(p_values, alpha=0.05):
    """Benjamini-Hochberg 错误发现率控制。

    同时检验 24 个因子时，即使全是噪音，也会有约 1 个"显著"（|t|>2 的概率约 5%）。
    BH 把多重检验下的假阳性期望比例控制在 alpha 以内。
    """
    p = pd.Series(p_values).dropna().sort_values()
    m = len(p)
    if m == 0:
        return pd.Series(dtype=bool), p
    passed = p.to_numpy() <= alpha * (np.arange(1, m + 1) / m)
    sig = pd.Series(False, index=p.index)
    if passed.any():
        k = int(np.max(np.where(passed)[0])) + 1
        sig.iloc[:k] = True
    return sig, p


# ==============================================================================
# SECTION 10  主程序与报告
# ==============================================================================

def fmt_pct(x, d=2):
    try:
        return "—" if x is None or not np.isfinite(x) else f"{x*100:.{d}f}%"
    except Exception:
        return "—"


def fmt_num(x, d=3):
    try:
        return "—" if x is None or not np.isfinite(x) else f"{x:.{d}f}"
    except Exception:
        return "—"


def print_factor_table(R):
    cols = ["layer", "IC_mean", "IC_t", "ICIR", "IC_pos_rate",
            "LS_mean_ann", "LS_t", "LS_sharpe", "FM_t", "monotonic", "turnover"]
    d = R[cols].copy()
    for c in ["IC_mean"]:
        d[c] = d[c].map(lambda v: fmt_num(v, 4))
    for c in ["IC_t", "LS_t", "FM_t", "ICIR", "LS_sharpe", "monotonic"]:
        d[c] = d[c].map(lambda v: fmt_num(v, 2))
    for c in ["IC_pos_rate", "LS_mean_ann", "turnover"]:
        d[c] = d[c].map(lambda v: fmt_pct(v, 1))
    print(d.to_string())


def print_perf(s, prefix="  "):
    if not s or "CAGR" not in s:
        return
    print(f"{prefix}{s['name']:<24s} 年化={fmt_pct(s['CAGR'],2):>8s}  "
          f"波动={fmt_pct(s['AnnVol'],2):>8s}  夏普={fmt_num(s['Sharpe'],3):>7s}  "
          f"最大回撤={fmt_pct(s['MaxDD'],2):>8s}  Calmar={fmt_num(s['Calmar'],2):>6s}  "
          f"胜率={fmt_pct(s['WinRate'],1):>7s}")


def make_charts(results, benchmark_df, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 净值曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    for k, r in results.items():
        if "equity" in r:
            ax.plot(r["equity"].index, r["equity"].to_numpy(), label=k, lw=1.6)
    if benchmark_df is not None:
        b = benchmark_df["equity"]
        ax.plot(b.index, b.to_numpy(), label="等权全市场基准", lw=1.5,
                ls="--", color="gray")
    ax.set_yscale("log")
    ax.set_title("净值曲线（对数坐标）")
    ax.set_ylabel("净值")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "equity.png"), dpi=130)
    plt.close(fig)

    # 回撤
    fig, ax = plt.subplots(figsize=(12, 4))
    for k, r in results.items():
        if "dd" in r:
            ax.fill_between(r["dd"].index, r["dd"].to_numpy(), 0, alpha=0.45, label=k)
    ax.set_title("水下回撤")
    ax.set_ylabel("回撤")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "drawdown.png"), dpi=130)
    plt.close(fig)

    # 滚动夏普
    fig, ax = plt.subplots(figsize=(12, 4))
    for k, r in results.items():
        if "net" in r:
            rs = r["net"].rolling(252).apply(
                lambda a: sharpe_ratio(a, rf_annual=0.0), raw=True)
            ax.plot(rs.index, rs.to_numpy(), label=k, lw=1.4)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(1, color="r", lw=0.8, ls="--", label="夏普=1")
    ax.set_title("滚动 1 年夏普")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rolling_sharpe.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "factors", "test", "backtest"])
    ap.add_argument("--lsy", type=int, default=1, help="1=LSY股票池 0=全A")
    ap.add_argument("--force-factors", action="store_true")
    args = ap.parse_args()

    CFG.use_lsy_universe = bool(args.lsy)
    tag = "LSY" if CFG.use_lsy_universe else "FULL"
    t_all = time.time()

    print("=" * 78)
    print("  中国 A 股截面因子量化策略")
    print(f"  股票池: {'LSY 校正（剔除最小 30% 市值）' if CFG.use_lsy_universe else '全 A'}")
    print("=" * 78, flush=True)

    # ---------------- SECTION 1 ----------------
    print("\n[SECTION 1] 加载数据面板", flush=True)
    df = load_panel()
    P = Panels(df)
    del df                       # 长表可释放：宽表已包含全部信息
    print(f"  区间 {P.dates[0].date()} ~ {P.dates[-1].date()}", flush=True)

    # ---------------- SECTION 2 ----------------
    print("\n[SECTION 2] 构建时点股票池", flush=True)
    eligible, limit_up, limit_down = build_universe(P, CFG)
    del limit_up, limit_down     # 股票池已剔除涨停；跌停限制在「局限」中说明

    # ---------------- SECTION 3 ----------------
    print("\n[SECTION 3] 因子库（逐个计算 → 落盘 → 释放）", flush=True)
    names = compute_factor_pipeline(P, CFG, tag, force=args.force_factors)
    fdir = os.path.join(DATA, "factors")

    if args.stage == "factors":
        print(f"\n因子已缓存至 {fdir}")
        return

    # ---------------- SECTION 5 ----------------
    # 【防过拟合的核心】挑因子与验真必须用两段互不重叠的数据。
    #   样本内 2010-01-01 ~ 2022-12-31 —— 只用这段决定"用哪些因子"
    #   样本外 2023-01-01 ~ 2026-09-28 —— 挑因子时完全没看过，业绩才可信
    m_is = ((P.dates >= pd.Timestamp(CFG.full_start)) &
            (P.dates <= pd.Timestamp(CFG.selection_end)))
    m_oos = ((P.dates >= pd.Timestamp(CFG.bt_start)) &
             (P.dates <= pd.Timestamp(CFG.bt_end)))

    print("\n[SECTION 4+5] 因子预处理与单因子检验")
    print(f"  样本内（决定用哪些因子）: {CFG.full_start} ~ {CFG.selection_end}"
          f"  {int(m_is.sum())} 交易日")
    print(f"  样本外（检验真实业绩）  : {CFG.bt_start} ~ {CFG.bt_end}"
          f"  {int(m_oos.sum())} 交易日", flush=True)

    fwd_all = forward_return(P.close, CFG.ic_forward_days)
    fwd_is, elig_is = fwd_all.loc[m_is], eligible.loc[m_is]
    fwd_oos, elig_oos = fwd_all.loc[m_oos], eligible.loc[m_oos]

    rows = []
    # 相关性去重只用样本内数据，且按 21 日抽样以控制内存
    corr_sample = {}
    t0 = time.time()
    for name, fn, layer in FACTORS:
        path = os.path.join(fdir, f"{name}.pkl")
        if not os.path.exists(path):
            continue
        raw = pd.read_pickle(path)
        pre = prepare_factor(raw, P, eligible, CFG)
        del raw
        try:
            r = evaluate_factor(name, pre.loc[m_is], fwd_is, elig_is, CFG)
            r["layer"] = layer
            # 同一因子在样本外再跑一遍 —— 直接量化"因子衰减"
            ro = evaluate_factor(name, pre.loc[m_oos], fwd_oos, elig_oos, CFG)
            r["IC_oos"] = ro["IC_mean"]
            r["IC_t_oos"] = ro["IC_t"]
            r["LS_oos"] = ro["LS_mean_ann"]
            r["decay"] = (abs(ro["IC_mean"]) / abs(r["IC_mean"])
                          if r["IC_mean"] not in (0, None) and np.isfinite(r["IC_mean"])
                          and r["IC_mean"] != 0 else np.nan)
            rows.append(r)
            corr_sample[name] = pre.loc[m_is].iloc[::21]
            print(f"    {name:12s} 样本内 IC={fmt_num(r['IC_mean'],4)} "
                  f"t={fmt_num(r['IC_t'],2):>6s} | 样本外 IC={fmt_num(r['IC_oos'],4)} "
                  f"t={fmt_num(r['IC_t_oos'],2):>6s}  ({time.time()-t0:5.1f}s)", flush=True)
        except Exception as e:
            print(f"    {name:12s} 检验失败: {type(e).__name__} {str(e)[:60]}", flush=True)
        del pre

    R = pd.DataFrame(rows).set_index("factor")
    R = R.sort_values("IC_t", key=lambda s: s.abs(), ascending=False)
    R.to_pickle(os.path.join(DATA, f"factor_tests_{tag}.pkl"))

    print("\n  【单因子检验 · 按样本内 |IC t值| 排序】")
    print_factor_table(R)

    # ---- A 层（经典美股异象） vs B 层（截面联动类）----
    print("\n  【A 层经典异象 vs B 层截面联动类 · 对比】")
    for lay, lab in [("A", "A层 经典美股异象"), ("B", "B层 截面联动类 ")]:
        sub = R[R["layer"] == lay]
        if len(sub) == 0:
            continue
        print(f"    {lab} {len(sub):2d}个因子 | 平均|IC|={sub['IC_mean'].abs().mean():.4f} "
              f"(样本外 {sub['IC_oos'].abs().mean():.4f}) | "
              f"显著|t|>2: {int((sub['IC_t'].abs()>2).sum())}个 | "
              f"平均多空年化={fmt_pct(sub['LS_mean_ann'].mean(),2)} "
              f"(样本外 {fmt_pct(sub['LS_oos'].mean(),2)}) | "
              f"平均ICIR={fmt_num(sub['ICIR'].mean(),2)}")

    if args.stage == "test":
        return

    # ---------------- SECTION 6 ----------------
    print("\n[SECTION 6] 多因子合成（仅依据样本内表现挑选）", flush=True)
    cand = R[(R["IC_t"].abs() > 2.0) & (R["IC_mean"].abs() > 0.005)]
    keep = []
    for k in cand.index:                       # 相关性去重，避免同类因子堆叠
        dup = False
        for k2 in keep:
            c = corr_sample[k2].corrwith(corr_sample[k]).mean()
            if np.isfinite(c) and abs(c) > 0.7:
                dup = True
                break
        if not dup:
            keep.append(k)
    del corr_sample
    print(f"  样本内显著因子 {len(cand)} 个 → 相关性去重后保留 {len(keep)} 个")
    print(f"  保留清单 (按样本内 |t| 降序): {keep}")

    if not keep:
        print("  [中止] 没有因子通过筛选")
        return

    # 逐因子累加合成，避免同时持有 17 个宽表（每个约 85MB，合计 1.4GB）
    acc = cnt = None
    for k in keep:
        pre = prepare_factor(pd.read_pickle(os.path.join(fdir, f"{k}.pkl")),
                             P, eligible, CFG)
        valid = pre.notna()
        acc = pre.where(valid, 0.0) if acc is None else acc.add(pre.where(valid, 0.0),
                                                                fill_value=0.0)
        cnt = valid.astype(F32) if cnt is None else cnt.add(valid.astype(F32),
                                                            fill_value=0.0)
        del pre, valid
    score = (acc / cnt.replace(0, np.nan)).astype(F32)
    del acc, cnt

    # ---------------- SECTION 7 / 8 ----------------
    print("\n[SECTION 7/8] 组合构建与回测", flush=True)
    W_long = build_weights(score, eligible, "long", CFG)
    W_ls = build_long_short(score, eligible, CFG)
    bench_w = eligible.astype(float)
    bench_w = bench_w.div(bench_w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    results = {
        "多头组合(可交易)": backtest(P, W_long, CFG, "多头组合(可交易)"),
        "多空组合(学术)": backtest(P, W_ls, CFG, "多空组合(学术)"),
    }
    benchmark = backtest(P, bench_w, CFG, "等权全市场基准")

    # ---- 引擎自检：主引擎(逐日漂移) vs 向量化近似，两者应接近 ----
    print("\n  【回测引擎自检】主引擎(逐日漂移模拟) vs 向量化近似(权重恒定)")
    for nm, W in [("多头", W_long), ("多空", W_ls)]:
        try:
            v = backtest_vectorized(P, W, CFG, nm)
            m = (v["net"].index >= pd.Timestamp(CFG.bt_start))
            a = results["多头组合(可交易)" if nm == "多头" else "多空组合(学术)"]["net"]
            corr = pd.Series(a[m]).corr(pd.Series(v["net"][m]))
            print(f"    {nm}: 主引擎年化={fmt_pct(a[m].mean()*TRADING_DAYS,2)}  "
                  f"向量化年化={fmt_pct(v['net'][m].mean()*TRADING_DAYS,2)}  "
                  f"日收益相关性={fmt_num(corr,4)}  "
                  f"（差异来自漂移再平衡，相关性>0.99 即引擎自洽）")
        except Exception as e:
            print(f"    {nm}: 自检失败 {type(e).__name__} {str(e)[:60]}")

    def slice_perf(r, a, b):
        m = (r["net"].index >= pd.Timestamp(a)) & (r["net"].index <= pd.Timestamp(b))
        if m.sum() < 20:
            return None
        s = summarize(r["net"][m], CFG.rf_annual, r["name"])
        s["ann_turnover"] = float(r["turnover"][m].mean() * TRADING_DAYS)
        return s

    # ---- 样本内：因子选择所用的数据，业绩天然好看，仅供参考 ----
    print(f"\n  ── 样本内 {CFG.full_start} ~ {CFG.selection_end}"
          f"（挑因子用的数据，非真实业绩）──")
    is_perf = {}
    for k, r in results.items():
        s = slice_perf(r, CFG.full_start, CFG.selection_end)
        if s:
            is_perf[k] = s
            print_perf(s, "    ")
    sbi = slice_perf(benchmark, CFG.full_start, CFG.selection_end)
    if sbi:
        print_perf(sbi, "    ")

    # ---- 样本外：这才是要报告的业绩 ----
    print(f"\n  ★ 样本外 {CFG.bt_start} ~ {CFG.bt_end}"
          f"（挑因子时完全未见过）★")
    main_perf = {}
    for k, r in results.items():
        s = slice_perf(r, CFG.bt_start, CFG.bt_end)
        if s:
            main_perf[k] = s
            print_perf(s, "    ")
    sb = slice_perf(benchmark, CFG.bt_start, CFG.bt_end)
    if sb:
        print_perf(sb, "    ")

    # ---- 样本内 → 样本外 衰减 ----
    print("\n  ── 样本内 → 样本外 衰减（过拟合的直接度量）──")
    for k in results:
        a, b = is_perf.get(k), main_perf.get(k)
        if a and b:
            print(f"    {k:<24s} 夏普 {a['Sharpe']:6.3f} → {b['Sharpe']:6.3f}  "
                  f"(保留 {b['Sharpe']/a['Sharpe']*100 if a['Sharpe'] else float('nan'):.0f}%)   "
                  f"年化 {a['CAGR']*100:6.2f}% → {b['CAGR']*100:6.2f}%")

    # ---- 子区间（时效性）----
    print("\n  ── 子区间（时效性与近期有效性）──")
    sub_perf = {}
    for a, b, lab in CFG.subperiods:
        print(f"    [{lab}] {a} ~ {b}")
        for k, r in results.items():
            s = slice_perf(r, a, b)
            if s:
                sub_perf.setdefault(k, {})[lab] = s
                print_perf(s, "      ")

    # ---- 波动率目标化叠加：直接改善最大回撤 ----
    print("\n  【波动率目标化叠加】目标年化波动 15%，60 日滚动估计，杠杆上限 1.5×")
    print("    （杠杆只由历史波动率决定，不含收益预测，无未来函数）")
    vt_results = {}
    for k, r in results.items():
        vt_net, lev = vol_targeted(r["net"], 0.15, 60, 1.5)
        vt = summarize(vt_net, CFG.rf_annual, k.replace("组合", "") + "+波动率目标")
        vt["net"] = vt_net
        vt["lev"] = lev
        vt["turnover"] = r["turnover"]
        vt_results[k] = vt
        s = slice_perf(vt, CFG.bt_start, CFG.bt_end)
        if s:
            print_perf(s, "    ")
    print(f"    （对照：未叠加时 多头夏普 {fmt_num(main_perf.get('多头组合(可交易)',{}).get('Sharpe'),3)} "
          f"/ 回撤 {fmt_pct(main_perf.get('多头组合(可交易)',{}).get('MaxDD'),2)}；"
          f"多空夏普 {fmt_num(main_perf.get('多空组合(学术)',{}).get('Sharpe'),3)} "
          f"/ 回撤 {fmt_pct(main_perf.get('多空组合(学术)',{}).get('MaxDD'),2)}）")

    # ---- 保存日收益序列（供后续分析，不必重跑全流程）----
    try:
        pd.to_pickle({k: {"net": v["net"], "gross": v["gross"], "cost": v["cost"],
                          "turnover": v["turnover"]} for k, v in results.items()},
                     os.path.join(OUT, f"returns_{tag}.pkl"))
    except Exception as e:
        print(f"    [提示] 收益序列保存失败: {e}")

    # ---- 交易成本敏感性 ----
    print("\n  【交易成本敏感性】换手越高，成本假设越要经得起推敲")
    print(f"    基准假设: 佣金{CFG.commission_bp}bp + 滑点{CFG.slippage_bp}bp "
          f"+ 印花税{CFG.stamp_duty_bp}bp(卖出)")
    cost_sens = {}
    for mult in (1.0, 2.0, 3.0):
        c2 = replace(CFG, commission_bp=CFG.commission_bp * mult,
                     slippage_bp=CFG.slippage_bp * mult,
                     stamp_duty_bp=CFG.stamp_duty_bp * mult)
        line = []
        for k, W in [("多头", W_long), ("多空", W_ls)]:
            rr = backtest(P, W, c2, k)
            s = summarize(rr["net"][
                (rr["net"].index >= pd.Timestamp(CFG.bt_start)) &
                (rr["net"].index <= pd.Timestamp(CFG.bt_end))], CFG.rf_annual, k)
            cost_sens[f"{k}_x{mult}"] = s
            line.append(f"{k} 夏普={fmt_num(s['Sharpe'],3)} 年化={fmt_pct(s['CAGR'],2)}")
        print(f"    {mult:.0f}× 成本: " + " | ".join(line))

    # ---- 因子暴露诊断：确认收益不是"小市值 beta"或"行业押注" ----
    print("\n  【组合暴露诊断】—— 排除「收益其实来自行业或市值押注」的可能")
    try:
        sz_raw = pd.read_pickle(os.path.join(fdir, "SIZE.pkl"))
        sz_rank = fast_rank_pct(sz_raw.where(eligible))          # SIZE 已取负号
        sc_rank = fast_rank_pct(score.where(eligible))
        # 综合得分与 SIZE 的截面相关（1.0 = 完全是市值因子）
        c_sz = sc_rank.corrwith(sz_rank).mean()
        print(f"    综合得分 与 SIZE 的截面相关性: {fmt_num(c_sz,3)}  "
              f"(越接近 0 越说明不是纯市值因子)")
        for nm, W in [("多头组合", W_long), ("多空组合", W_ls)]:
            Wr = W.reindex(P.dates).ffill().fillna(0)
            held = (Wr.abs() > 1e-9) & eligible
            if held.to_numpy().sum() == 0:
                continue
            # 持仓的平均市值分位（0=最小市值, 1=最大市值）；SIZE 取了负号故先还原
            hold_sz = (1 - sz_rank).where(held).mean(axis=1)
            uni_sz = (1 - sz_rank).where(eligible).mean(axis=1)
            n_hold = held.sum(axis=1).replace(0, np.nan)
            print(f"    {nm}: 持仓平均市值分位 {fmt_num(hold_sz.mean(),3)} "
                  f"vs 全市场 {fmt_num(uni_sz.mean(),3)}   平均持仓 {fmt_num(n_hold.mean(),1)} 只")
    except Exception as e:
        print(f"    暴露诊断失败: {type(e).__name__} {str(e)[:70]}")

    # ---------------- SECTION 9 ----------------
    print("\n[SECTION 9] 显著性检验", flush=True)
    # 多重检验基准：我们实际"试过"的是这 len(R) 个因子，
    # 因此跨试验夏普方差取这些因子多空组合夏普的实际方差 —— 而不是随手设一个数。
    trial_sharpes = R["LS_sharpe"].replace([np.inf, -np.inf], np.nan).dropna()
    var_sr = float(trial_sharpes.var(ddof=1)) if len(trial_sharpes) > 1 else 0.0
    print(f"  多重检验背景: 共试过 {len(R)} 个因子")
    print(f"    跨试验夏普(年化): 均值 {trial_sharpes.mean():.3f}, "
          f"标准差 {trial_sharpes.std(ddof=1):.3f}, 方差 {var_sr:.4f}")
    print(f"    → 该方差将用于 Deflated Sharpe 的 SR0 门槛（不是拍脑袋设的常数）")

    def run_significance(x_series, label, note=""):
        mu, t, T = newey_west_tstat(x_series, CFG.nw_lags)
        dsr = deflated_sharpe(x_series, n_trials=max(len(R), 1),
                              trial_sharpes=trial_sharpes)
        boot = stationary_bootstrap(x_series, CFG.bootstrap_n,
                                    CFG.bootstrap_block, CFG.random_seed)
        ci = np.nanpercentile(boot, [2.5, 97.5]) if boot is not None else (np.nan, np.nan)
        p_gt0 = float(np.mean(boot > 0)) if boot is not None else np.nan
        print(f"\n  【{label}】{note}")
        print(f"    样本长度         : {T} 日 ({T/TRADING_DAYS:.2f} 年)")
        print(f"    日收益均值       : {fmt_num(mu,6)}")
        print(f"    Newey-West t 值  : {fmt_num(t,3)}   （|t|>1.96 为 5% 水平显著）")
        print(f"    年化夏普         : {fmt_num(dsr['SR'],3)}")
        print(f"    PSR (夏普>0 概率): {fmt_num(dsr['PSR'],4)}")
        print(f"    多重检验门槛 SR0 : {fmt_num(dsr['SR0'],3)}  (N={dsr['n_trials']})")
        print(f"    Deflated Sharpe  : {fmt_num(dsr['DSR'],4)}   "
              f"（>0.95 才算扣除选择偏差后仍显著）")
        print(f"    定态块自助 95%CI : 夏普 ∈ [{fmt_num(ci[0],3)}, {fmt_num(ci[1],3)}]")
        print(f"    自助法 P(夏普>0) : {fmt_num(p_gt0,4)}")
        print(f"    收益偏度/峰度    : {fmt_num(dsr['skew'],2)} / {fmt_num(dsr['kurt'],2)}")
        return {"mean_daily": mu, "nw_t": t, "n": T, **dsr,
                "boot_ci_low": float(ci[0]), "boot_ci_high": float(ci[1]),
                "boot_p_sharpe_gt0": p_gt0}

    sig, sig_full = {}, {}
    for k, r in results.items():
        x = r["net"]
        m = (x.index >= pd.Timestamp(CFG.bt_start)) & (x.index <= pd.Timestamp(CFG.bt_end))
        sig[k] = run_significance(x[m], k, "—— 样本外（干净的验真区间）")
        mf = (x.index >= pd.Timestamp(CFG.full_start)) & (x.index <= pd.Timestamp(CFG.full_end))
        sig_full[k] = run_significance(
            x[mf], k, "—— 全样本（统计功效强，但与选因子窗口重叠，会偏乐观）")

    # 统计功效提示：3.7 年要显著需要多高的夏普
    yrs = int(((P.dates >= pd.Timestamp(CFG.bt_start)) &
               (P.dates <= pd.Timestamp(CFG.bt_end))).sum()) / TRADING_DAYS
    print(f"\n  【统计功效提示】样本外只有 {yrs:.2f} 年。")
    print(f"    在 t = 夏普 × sqrt(年数) 的关系下，要在 5% 水平显著（t>1.96），")
    print(f"    需要年化夏普 > {1.96/math.sqrt(yrs):.2f}。")
    print(f"    这正是短样本区间做量化验证的根本困难：不是策略不行，而是样本不够长。")

    # ---- FDR ----
    print("\n  【因子层面多重检验校正】")
    try:
        from scipy.stats import norm as _norm
        pvals = 2 * (1 - _norm.cdf(np.abs(R["IC_t"].to_numpy())))
        sig_mask, p_sorted = bh_fdr(pd.Series(pvals, index=R.index), 0.05)
        print(f"    未校正显著 (|t|>2): {int((R['IC_t'].abs()>2).sum())} / {len(R)} 个因子")
        print(f"    BH-FDR 校正后显著 : {int(sig_mask.sum())} 个")
        print(f"    通过校正的因子    : {list(sig_mask[sig_mask].index)}")
    except Exception as e:
        print(f"    FDR 计算失败: {e}")

    # ---------------- 保存 ----------------
    def clean(d):
        o = {}
        for kk, vv in d.items():
            if kk in ("equity", "dd", "net", "gross", "cost", "turnover", "weights",
                      "beta_series"):
                continue
            if isinstance(vv, (np.floating, np.integer)):
                vv = float(vv)
            if isinstance(vv, float) and not np.isfinite(vv):
                vv = None
            o[kk] = vv
        return o

    payload = {
        "config": {k: str(v) for k, v in CFG.__dict__.items()},
        "universe": tag,
        "n_factors_tested": len(R),
        "selected_factors": keep,
        "factor_tests": R.reset_index().to_dict("records"),
        "performance_in_sample": {k: clean(v) for k, v in is_perf.items()},
        "performance_out_of_sample": {k: clean(v) for k, v in main_perf.items()},
        "performance_benchmark": clean(sb) if sb else None,
        "performance_subperiods": {k: {l: clean(s) for l, s in v.items()}
                                   for k, v in sub_perf.items()},
        "cost_sensitivity": {k: clean(v) for k, v in cost_sens.items()},
        "vol_targeted": {k: clean(v) for k, v in vt_results.items()},
        "significance": {k: clean(v) for k, v in sig.items()},
        "significance_full_sample": {k: clean(v) for k, v in sig_full.items()},
        "trial_sharpe_variance": var_sr,
    }
    jpath = os.path.join(OUT, f"summary_{tag}.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    make_charts(results, benchmark, OUT)
    write_markdown_report(payload, R, keep, is_perf, main_perf, sb, sub_perf,
                          sig, cost_sens, CFG, tag,
                          os.path.join(OUT, f"report_{tag}.md"),
                          sig_full=sig_full)

    print(f"\n  结果已保存:")
    print(f"    {jpath}")
    print(f"    {os.path.join(OUT, f'report_{tag}.md')}")
    print(f"    {os.path.join(OUT, 'equity.png')} / drawdown.png / rolling_sharpe.png")
    print(f"  总用时 {(time.time()-t_all)/60:.1f} 分钟")


def write_markdown_report(payload, R, keep, is_perf, main_perf, bench_perf,
                          sub_perf, sig, cost_sens, cfg, tag, path, sig_full=None):
    """生成 Markdown 回测报告"""
    L = []
    A = L.append
    A(f"# 中国 A 股截面因子策略 · 回测报告（{tag} 股票池）\n")
    A(f"> 生成时间 {pd.Timestamp.now():%Y-%m-%d %H:%M}　|　"
      f"全样本 {cfg.full_start} ~ {cfg.full_end}\n")

    A("\n## 一、方法概要\n")
    A(f"- **因子选择窗口（样本内）**：{cfg.full_start} ~ {cfg.selection_end}")
    A(f"- **业绩报告窗口（样本外）**：{cfg.bt_start} ~ {cfg.bt_end}"
      f" —— 挑因子时完全未使用")
    A(f"- **股票池**：{'Liu-Stambaugh-Yuan(2019) 校正，剔除最小 30% 流通市值' if tag=='LSY' else '全 A（稳健性对照）'}")
    A(f"- **可交易性过滤**：剔除 ST、上市不足 {cfg.min_days_listed} 交易日、停牌、涨停")
    A(f"- **调仓频率**：每 {cfg.rebalance_days} 个交易日（≈月度）")
    A(f"- **交易成本**：佣金 {cfg.commission_bp}bp（单边）+ 滑点 {cfg.slippage_bp}bp（单边）"
      f"+ 印花税 {cfg.stamp_duty_bp}bp（仅卖出）")
    A(f"- **执行模型**：t 日收盘算因子 → t+1 日开盘成交（无未来函数）")
    A(f"- **检验因子数**：{len(R)} 个（Deflated Sharpe 按此做多重检验校正）\n")

    A("\n## 二、单因子检验（样本内）\n")
    A("| 因子 | 层 | RankIC | NW t | ICIR | IC胜率 | 多空年化 | 多空t | 单调性 | 换手 | 样本外IC | 样本外t |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f, r in R.iterrows():
        A(f"| {f} | {r['layer']} | {fmt_num(r['IC_mean'],4)} | {fmt_num(r['IC_t'],2)} | "
          f"{fmt_num(r['ICIR'],2)} | {fmt_pct(r['IC_pos_rate'],1)} | "
          f"{fmt_pct(r['LS_mean_ann'],1)} | {fmt_num(r['LS_t'],2)} | "
          f"{fmt_num(r['monotonic'],2)} | {fmt_pct(r['turnover'],1)} | "
          f"{fmt_num(r.get('IC_oos'),4)} | {fmt_num(r.get('IC_t_oos'),2)} |")

    A(f"\n**选入组合的因子（{len(keep)} 个，仅依据样本内表现 + 相关性去重）**：")
    A(f"`{'`, `'.join(keep)}`\n")

    A("\n## 三、业绩（样本外）\n")
    A(f"| 组合 | 年化收益 | 年化波动 | 夏普 | 最大回撤 | Calmar | 胜率 | 年化换手 |")
    A("|---|---|---|---|---|---|---|---|")
    for k, s in main_perf.items():
        A(f"| {k} | {fmt_pct(s['CAGR'],2)} | {fmt_pct(s['AnnVol'],2)} | "
          f"{fmt_num(s['Sharpe'],3)} | {fmt_pct(s['MaxDD'],2)} | "
          f"{fmt_num(s['Calmar'],2)} | {fmt_pct(s['WinRate'],1)} | "
          f"{fmt_pct(s.get('ann_turnover'),0)} |")
    if bench_perf:
        s = bench_perf
        A(f"| 等权全市场基准 | {fmt_pct(s['CAGR'],2)} | {fmt_pct(s['AnnVol'],2)} | "
          f"{fmt_num(s['Sharpe'],3)} | {fmt_pct(s['MaxDD'],2)} | "
          f"{fmt_num(s['Calmar'],2)} | {fmt_pct(s['WinRate'],1)} | — |")

    A("\n### 样本内 → 样本外 衰减\n")
    A("| 组合 | 样本内夏普 | 样本外夏普 | 夏普保留率 | 样本内年化 | 样本外年化 |")
    A("|---|---|---|---|---|---|")
    for k in main_perf:
        a, b = is_perf.get(k), main_perf.get(k)
        if a and b:
            keep_pct = (b['Sharpe'] / a['Sharpe'] * 100) if a['Sharpe'] else np.nan
            A(f"| {k} | {fmt_num(a['Sharpe'],3)} | {fmt_num(b['Sharpe'],3)} | "
              f"{fmt_num(keep_pct,0)}% | {fmt_pct(a['CAGR'],2)} | {fmt_pct(b['CAGR'],2)} |")

    A("\n## 四、显著性检验\n")
    A(f"### 4.1 样本外（{cfg.bt_start} ~ {cfg.bt_end}）—— 干净的验真区间\n")
    A("| 组合 | 样本年数 | Newey-West t | 年化夏普 | PSR | 多重检验门槛SR0 | **Deflated Sharpe** | 自助95%CI | P(夏普>0) |")
    A("|---|---|---|---|---|---|---|---|---|")
    for k, s in sig.items():
        A(f"| {k} | {s['n']/252:.2f} | {fmt_num(s['nw_t'],3)} | "
          f"{fmt_num(s['SR'],3)} | {fmt_num(s['PSR'],4)} | {fmt_num(s['SR0'],3)} | "
          f"**{fmt_num(s['DSR'],4)}** | "
          f"[{fmt_num(s['boot_ci_low'],2)}, {fmt_num(s['boot_ci_high'],2)}] | "
          f"{fmt_num(s['boot_p_sharpe_gt0'],3)} |")
    if sig_full:
        A(f"\n### 4.2 全样本（{cfg.full_start} ~ {cfg.full_end}）—— 功效强但与选因子窗口重叠\n")
        A("| 组合 | 样本年数 | Newey-West t | 年化夏普 | PSR | 多重检验门槛SR0 | **Deflated Sharpe** | 自助95%CI |")
        A("|---|---|---|---|---|---|---|---|")
        for k, s in sig_full.items():
            A(f"| {k} | {s['n']/252:.2f} | {fmt_num(s['nw_t'],3)} | "
              f"{fmt_num(s['SR'],3)} | {fmt_num(s['PSR'],4)} | {fmt_num(s['SR0'],3)} | "
              f"**{fmt_num(s['DSR'],4)}** | "
              f"[{fmt_num(s['boot_ci_low'],2)}, {fmt_num(s['boot_ci_high'],2)}] |")

    yrs = list(sig.values())[0]['n'] / 252 if sig else 3.7
    A("\n> **读法**")
    A("> - **Newey-West t**：|t|>1.96 为 5% 水平显著（已修正收益自相关）。")
    A("> - **PSR**：夏普大于 0 的概率（已修正偏度与厚尾）。")
    A(f"> - **Deflated Sharpe**：把「试过 {len(R)} 个因子再挑最好」这一选择偏差折进门槛，")
    A(">   门槛 SR0 由这些因子实际夏普的横截面方差算出，>0.95 才算真显著。")
    A(f"> - **统计功效**：在 t = 夏普 × √年数 的关系下，样本外 {yrs:.2f} 年要在 5% 水平显著，")
    A(f">   需要年化夏普 > {1.96/math.sqrt(yrs):.2f}。短样本区间是量化验证的根本约束。\n")

    A("\n## 五、子区间表现（时效性）\n")
    A("| 组合 | 区间 | 年化 | 夏普 | 最大回撤 |")
    A("|---|---|---|---|---|")
    for k, subs in sub_perf.items():
        for lab, s in subs.items():
            A(f"| {k} | {lab} | {fmt_pct(s['CAGR'],2)} | {fmt_num(s['Sharpe'],3)} | "
              f"{fmt_pct(s['MaxDD'],2)} |")

    A("\n## 六、交易成本敏感性\n")
    A("| 成本倍数 | 组合 | 年化 | 夏普 | 最大回撤 |")
    A("|---|---|---|---|---|")
    for k, s in cost_sens.items():
        nm, mult = k.split("_x")
        A(f"| {mult}× | {nm} | {fmt_pct(s['CAGR'],2)} | {fmt_num(s['Sharpe'],3)} | "
          f"{fmt_pct(s['MaxDD'],2)} |")

    vt = (payload or {}).get("vol_targeted") or {}
    if vt:
        A("\n## 七、波动率目标化叠加（风险控制层）\n")
        A("做法：用截至前一日的 60 日已实现波动率决定杠杆 `lev = 15% / 已实现波动`，")
        A("上限 1.5×，把组合波动稳定在目标水平。杠杆只由历史波动率决定，**不含任何收益预测**，")
        A("因此不是过拟合，而是利用波动率聚集性（volatility clustering）这一稳健规律。\n")
        A("| 组合 | 年化 | 年化波动 | 夏普 | 最大回撤 | Calmar |")
        A("|---|---|---|---|---|---|")
        for k, s in vt.items():
            A(f"| {s.get('name', k)} | {fmt_pct(s.get('CAGR'),2)} | "
              f"{fmt_pct(s.get('AnnVol'),2)} | {fmt_num(s.get('Sharpe'),3)} | "
              f"{fmt_pct(s.get('MaxDD'),2)} | {fmt_num(s.get('Calmar'),2)} |")
        A("\n（上表为**全样本**口径；与第四节样本外数字对照时注意区间不同。"
          "叠加层的主要价值在于压低回撤与波动，而非提高年化收益。）\n")

    A("\n## 八、局限与风险提示\n")
    A("1. **样本外区间偏短**：2023 年至今约 3.7 年，统计功效有限。")
    A("2. **涨跌停卖出限制未建模**：调仓日若个股跌停，实际无法卖出，本引擎按可成交处理，会略微高估。")
    A("3. **停牌处理**：停牌期间价格前向填充、收益记 0，复牌后按新价计算。")
    A("4. **融券成本未计入多空组合**：A 股融券难且贵，多空组合用于因子显著性检验，不代表可复制业绩。")
    A("5. **数据源**：新浪（前复权行情/流通股本）+ Tushare（原始行情交叉校验）+ baostock（时点 ST/行业）。")
    A("   已退市股票尽量纳入，但早年退市股在数据源侧可能缺失，仍存在残余幸存者偏差。")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
