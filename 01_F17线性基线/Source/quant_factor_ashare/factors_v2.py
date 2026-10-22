# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
factors_v2.py — 由论文精读新增的因子
================================================================================
本模块只放**严格照论文实现**的新因子，每一个都注明出处、公式与实现要点。

已实现
------
CGO / CGO_V   Grinblatt & Han (2005, JFE 78(2), 311-339)
              "Prospect theory, mental accounting, and momentum"

  论文模型（其式 (1)，原文递推）：
      R_{t+1} = V_t · P_t + (1 − V_t) · R_t
  其中 R 为「参考价格」（投资者的平均成本基础），V_t 为换手率，
  P_t 为价格。经济含义：当期换手的投资者把成本更新到现价 P_t，
  未换手的投资者保留原参考价，故参考价是全部历史价格的
  **换手率加权几何衰减平均**。

  论文中用于回归的变量（其 Table 3 note 与 NBER WP w8734 式 (25)）：
      g_{t−1} = (P_{t−2} − R_{t−1}) / P_{t−2}
      即「资本利得突出量 Capital Gains Overhang」= 未实现盈亏的相对幅度。

  ★ 实现要点（子代理逐行核对原文后确认，避免了我原本的错误假设）：
    1. **没有可校准的 delta 参数**。原文全文无 delta/decay/calibration 字样；
       衰减完全由股票自身换手率 V_t 内生决定，是逐股逐期时变的。
    2. 归一化常数 k 是「逐股逐期」的，不是常数；但在 A 股换手率水平下
       k ≈ 0.995 ~ 1.000，**可忽略**，直接用朴素递推即可。
    3. 原文要求 260 周历史，缺失则整条剔除。本实现要求 ≥ 52 周
       （A 股周换手率远高于美股，(1−V)^52 已衰减到可忽略）。
    4. 原文**未讨论**拆股与换手率缺失的处理。本实现用前复权价
       （P.close 已是 qfq），停牌期参考价冻结，上市前自然为 NaN。
    5. ★ 方向：**做多高 CGO、做空低 CGO**（g 越大 → 未实现盈利越多 →
       处置效应使卖压不足 → 未来收益越高）。
    6. ★ 原文报告**1月符号反转**（g 系数 −0.0117, t=−4.95），
       策略层面需注意。本实现不做日历择时，仅在报告中披露。

  论文实测（周频 Fama-MacBeth，1967-1996）：
      g 平均系数 0.0040 (t=7.79)；2-11月 0.0050 (t=9.42)；1月 −0.0117 (t=−4.95)
      月度收益版本 0.0127 (t=5.42)；按 60% 分位差换算约 12.5%/年
      骏马赛：中期动量 r_{−52:−5} 系数由 0.0014 (t=3.57) 降到 −0.0002 (t=−0.68)
      —— 即 CGO 吸收了动量。这与我们样本里 MOM_12_1 失效（IC=−0.0045）互补。
      交互项 V_{−1}·g 把 t 值提到 15.64，故一并实现为 CGO_V。
"""

import numpy as np
import pandas as pd

from ashare_factor_strategy import F32

MIN_WEEKS = 52          # 参考价格递推的最小历史周数


def _ref_price(P, weekly=True):
    """计算 Grinblatt-Han 参考价格 R_t（与 P.close 同一价格空间）。

    递推： R_t = V_{t-1}·P_{t-1} + (1 − V_{t-1})·R_{t-1}
    """
    if weekly:
        Pw = P.close.resample("W-FRI").last()
        Vw = P.turnover.resample("W-FRI").sum()
    else:
        Pw, Vw = P.close, P.turnover.copy()

    # 换手率单位已是小数；异常值（>100%）视为缺失
    Vw = Vw.where((Vw >= 0) & (Vw < 1.0))
    Vw = Vw.fillna(0.0)

    cols = Pw.columns
    idx = Pw.index
    p = Pw.to_numpy(dtype=np.float64)
    v = Vw.to_numpy(dtype=np.float64)
    n, m = p.shape

    R = np.full((n, m), np.nan, dtype=np.float64)
    hist = np.zeros(m, dtype=np.int64)     # 各股已积累的有效周数

    prev_R = np.full(m, np.nan, dtype=np.float64)
    prev_P = np.full(m, np.nan, dtype=np.float64)
    prev_V = np.zeros(m, dtype=np.float64)

    for i in range(n):
        pi = p[i]
        vi = v[i]
        finite_p = np.isfinite(pi)

        # 首次出现价格 -> 参考价初始化为此价
        seed = finite_p & ~np.isfinite(prev_R)
        prev_R = np.where(seed, pi, prev_R)
        hist = np.where(seed, 1, hist)

        newR = prev_V * prev_P + (1.0 - prev_V) * prev_R
        # 停牌（当期无价）时参考价冻结不更新
        newR = np.where(finite_p, newR, prev_R)
        R[i] = newR

        cont = finite_p & np.isfinite(prev_R) & ~seed
        hist = np.where(cont, hist + 1, hist)

        prev_R = np.where(finite_p, newR, prev_R)
        prev_P = np.where(finite_p, pi, prev_P)
        prev_V = vi

    R = pd.DataFrame(R, index=idx, columns=cols)
    R = R.where(pd.DataFrame(np.tile(hist, (n, 1)), index=idx, columns=cols) >= MIN_WEEKS)
    return R, hist


def _to_daily(x_weekly, P):
    """周频 -> 日频（前向填充），使其能进入日频因子流水线"""
    return x_weekly.reindex(P.dates, method="ffill").astype(F32)


def compute_cgo(P):
    """CGO：资本利得突出量 g = (P − R)/P。做多高 g、做空低 g。"""
    R, _ = _ref_price(P, weekly=True)
    Pw = P.close.resample("W-FRI").last()
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (Pw - R) / Pw
    return _to_daily(g, P)


def compute_cgo_v(P):
    """CGO 的换手率交互项 V·g —— 原文中 t 值最高（15.64）的形式。

    含义：突出量要「有人真的持有」才有定价含义，故按换手率加权。
    P.close 在周内取最后一日；换手率同样取最后一周的合计。
    """
    R, _ = _ref_price(P, weekly=True)
    Pw = P.close.resample("W-FRI").last()
    Vw = P.turnover.resample("W-FRI").sum().where(lambda x: (x >= 0) & (x < 1.0)).fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (Pw - R) / Pw
    return _to_daily(Vw * g, P)


def compute_cgo_m(P):
    """月度口径的 CGO（GH 用周频；此处给月度稳健性对照，取月末值）"""
    R, _ = _ref_price(P, weekly=True)
    Pm = P.close.resample("ME").last()
    Rm = R.reindex(Pm.index, method="ffill")
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (Pm - Rm) / Pm
    return g.reindex(P.dates, method="ffill").astype(F32)


# ------------------------------------------------------------------------------
# ★ 方向修正：A 股的 CGO 与美股相反
# ------------------------------------------------------------------------------
# 实测量化结果（预处理后 RankIC，21 日前瞻，LSY 股票池）：
#     CGO    样本内 IC = −0.0470 (t = −5.15, ICIR = −0.42)   样本外 IC = −0.0350 (t = −2.08)
#     CGO_V  样本内 IC = −0.0314 (t = −3.34)                 样本外 IC = −0.0212 (t = −1.41)
#     CGO_M  样本内 IC = −0.0254 (t = −2.82)                 样本外 IC = −0.0258 (t = −1.43)
#
# 即：在 A 股，**未实现盈利越大 → 未来收益越低**，与 Grinblatt-Han 在美国发现的
# 「做多高 CGO」方向相反。这并非实现错误，而是与已发表文献一致：
#
#   · Cheema & Nartea (2017, International Review of Economics and Finance 50,
#     85-97) 用 Cooper-Gutierrez-Hameed (2004) 的市场状态规则复现中国股市，
#     发现**中国动量跟随 DOWN 市场**，与美股 UP 市场结论完全相反
#     （UP−DOWN = −1.33%/月, t = −2.62），且反向由**输家腿**驱动。
#   · 本文实测 MOM_12_1 的 IC = −0.0045（动量失效/反向）、REV_1M 的 IC = +0.0497
#     （短期反转强），与上述一致：A 股是反转主导的市场。
#
# 因此本模块提供**取负号**的版本 CGO_R，使其与「因子值越大 → 预期收益越高」的
# 全库约定一致。符号的确定完全在**样本内窗口（2010-2022）**完成，样本外
# t=−2.08 是对该符号选择的独立确认，不构成事后择优。
def compute_cgo_r(P):
    """CGO 的反转口径（A 股适用）：因子值越大 → 预期收益越高。

    等价于「未实现亏损越大 → 未来反弹越强」，
    与短期反转 / 处置效应在 A 股的表现一致。
    """
    return -compute_cgo(P)


def cgo_diagnostics(P):
    """CGO 分布诊断：与 Grinblatt-Han 原文报告的分布对照，用于捕捉实现错误。

    原文（美股，周频，1962-1996）：mean 0.056, median 0.106, sd 0.251,
                                  p10 = −0.281, p90 = +0.312
    """
    R, hist = _ref_price(P, weekly=True)
    Pw = P.close.resample("W-FRI").last()
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (Pw - R) / Pw
    v = g.to_numpy(dtype=np.float64)
    v = v[np.isfinite(v)]
    q = np.percentile(v, [1, 10, 25, 50, 75, 90, 99])
    return {
        "n_valid": int(v.size),
        "mean": float(np.mean(v)), "sd": float(np.std(v)),
        "p1": float(q[0]), "p10": float(q[1]), "p25": float(q[2]),
        "median": float(q[3]), "p75": float(q[4]), "p90": float(q[5]),
        "p99": float(q[6]),
        "frac_g_gt_0": float((v > 0).mean()),
        "frac_g_lt_neg1": float((v < -1).mean()),
        "frac_g_gt_1": float((v > 1).mean()),
        "max_hist_weeks": int(hist.max()),
        "n_stocks_ge52w": int((hist >= MIN_WEEKS).sum()),
    }


EXTRA_FACTORS = {
    "CGO": compute_cgo,
    "CGO_V": compute_cgo_v,
    "CGO_M": compute_cgo_m,
    "CGO_R": compute_cgo_r,
}


# ------------------------------------------------------------------------------
# 由中国 A 股文献精读新增的两个换手率族因子
# ------------------------------------------------------------------------------
# 依据（子代理逐篇核对全文后给出）：
#
# (1) Jansen, Swinkels & Zhou (2021, Pacific-Basin Finance Journal 68, 101607)
#     检验 32 个异象（2000-2019），发现 **ABN_TURN = 过去 20 日平均换手 / 过去 250 日
#     平均换手** 是**唯一在市值加权组合上仍显著**的交易类异象：
#         EW 1.72%/月 (t=5.50)，VW 1.33%/月 (t=3.22)
#     而 TURN(250日) 的 VW 只有 0.69%/月 (t=1.56)、ILLIQ 0.71 (t=1.50) 均不显著。
#
# (2) Hou, Qiao & Zhang (2023, "Finding Anomalies in China", 426 个异象)
#     发现 **vturn1-1 = 换手率的「波动」**（不是水平）是全部 426 个异象中
#     经 CH-3 调整后**存活最强的**：
#         raw 1.58%/月 (t=3.45)，CH-3 α 1.38%/月 (t=3.38)，CH-4 α 0.71%
#     我们原有的 TURN_SHOCK 只是 5日/60日 的换手冲击（水平），**没有换手稳定性因子**。
#
# (3) Liu, Stambaugh & Yuan (2019) 的 CH-4 模型第四个因子 PMO 就建立在
#     异常换手率上（past-month turnover / past-year turnover，做多低、做空高），
#     与 ABN_TURN 同源。故这两项都不是数据挖掘产物，而是已被权威期刊验证的因子。

def compute_abn_turn(P):
    """异常换手率 ABN_TURN = 20日均换手 / 250日均换手，负向。

    换手率相对自身长期水平偏高 → 短期关注度/投机情绪过高 → 未来收益低。
    与已有 TURN_SHOCK（5日/60日）的区别：horizon 更长、更稳定，
    且是文献中在市值加权口径下唯一显著的交易类异象。
    """
    short = P.turnover.rolling(20, min_periods=15).mean()
    long = P.turnover.rolling(250, min_periods=150).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = short / long.replace(0, np.nan)
    return (-ratio).astype(F32)


def compute_vturn(P):
    """换手率波动 VTURN：过去 20 日换手率的变异系数，负向。

    Hou-Qiao-Zhang (2023) 在 426 个异象中发现它是经 CH-3 调整后存活最强的因子
    （CH-3 α 1.38%/月, t=3.38）。经济含义：换手率忽高忽低意味着交易由
    不稳定的注意力/情绪驱动，而非稳定基本面——这类股票未来收益更低。
    用变异系数（std/mean）而非纯 std，以剔除换手率水平本身的量纲影响。
    """
    m = P.turnover.rolling(20, min_periods=15).mean()
    s = P.turnover.rolling(20, min_periods=15).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = s / m.replace(0, np.nan)
    return (-cv).astype(F32)


EXTRA_FACTORS.update({"ABN_TURN": compute_abn_turn, "VTURN": compute_vturn})
