# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
build_ch3.py — 构建中国三因子 / 四因子模型（CH-3 / CH-4）
================================================================================
文献依据
--------
Liu, Jianan, Robert F. Stambaugh & Yu Yuan (2019), "Size and value in China",
Journal of Financial Economics 134(1), 48-69.  （NBER WP w24458 全文核对）

  CH-3 = MKT + SMB + VMG
    · 股票池：市值前 70%（剔除最小 30%，壳价值污染）
    · SMB：按市值中位数分 2 组
    · VMG：按 EP 分 3 组（30/70 分位），**不是 BM** —— 论文用 Fama-MacBeth 证明
      在中国 EP 显著（0.7825, t=4.38）而 logBM 不显著（0.0022, t=1.31）
    · 6 个市值加权组合：
        SMB = 1/3(S/V+S/M+S/G) − 1/3(B/V+B/M+B/G)
        VMG = 1/2(S/V+B/V) − 1/2(S/G+B/G)
    · 负 EP 股票保留在成长组（G）
    · 论文实测：MKT 0.66%/月(t=1.16)、SMB 1.03(t=3.25)、VMG 1.14(t=4.34)；
      SMB 与 VMG 相关 −0.62

  CH-4 = CH-3 + PMO（异常换手率因子）
    PMO = 过去一月换手率 / 过去一年换手率，**做多低、做空高**，
    构建方式与 VMG 完全相同；新的 SMB 取「EP 中性化 SMB」与「换手中性化 SMB」的均值。
    论文实测 PMO 约 12%/年。

为什么必须建这个
----------------
Hou, Qiao & Zhang (2023, "Finding Anomalies in China", 116pp, 426 个异象) 的结果：
    市值加权下显著 98 个 → 经 CH-3 调整后只剩 16 个 → 经 CH-4 调整后为 **0**
我们的 17 个因子很可能有相当一部分只是 CH-3 暴露。不建这个模型，
就无法判断策略的 alpha 究竟是选股能力还是规模/价值 beta。

★ 两处必须披露的口径偏离
------------------------
1. **市值用流通市值而非总市值**。LSY 用总市值（含非流通股），因为壳价值体现在
   非流通股上。本数据源无总股本字段，故用流通市值近似。
2. **可用时点用监管强制披露截止日**，而非 Wind 实际公告日：
      一季报 4/30、半年报 8/31、三季报 10/31、年报次年 4/30
   这比实际公告日晚，方向保守（不会前视）。

输出
----
data/ch3.pkl   DataFrame(index=月末日, columns=[MKT, SMB, VMG, PMO, SMB4])
data/ch3_expo.pkl  EP / BM / ROE / mv 的月末面板（供因子残差化使用）
"""

import os
import sys
import glob
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "data")
YDIR = os.path.join(DATA, "yjbb")
RF_ANNUAL = 0.02
PPY = 252


def sym6_to_panel(s):
    """6 位代码 -> 面板口径（sz000001 / sh600000 / bj430047）"""
    s = str(s).zfill(6)
    if s.startswith(("60", "68", "90", "11", "13")):
        return "sh" + s
    if s.startswith(("00", "30", "20", "12")):
        return "sz" + s
    if s.startswith(("43", "83", "87", "88", "92")):
        return "bj" + s
    return "sz" + s


def avail_date(report_date):
    """监管强制披露截止日之后才可用（保守，无前视）

    一季报 4/30、半年报 8/31、三季报 10/31、年报次年 4/30
    返回「最早可用月份的第一天」。
    """
    y, m, _ = report_date.year, report_date.month, report_date.day
    if m == 3:
        return pd.Timestamp(y, 5, 1)
    if m == 6:
        return pd.Timestamp(y, 9, 1)
    if m == 9:
        return pd.Timestamp(y, 11, 1)
    return pd.Timestamp(y + 1, 5, 1)          # 年报


def load_yjbb():
    """合并全部季报为长表，并折算 TTM（中国报表为年初至今累计口径）"""
    files = sorted(glob.glob(os.path.join(YDIR, "*.pkl")))
    frames = []
    for f in files:
        d = os.path.basename(f)[:8]
        df = pd.read_pickle(f)
        cols = {"股票代码": "code6", "每股收益": "eps_cum",
                "净利润-净利润": "ni_cum", "每股净资产": "bps",
                "净资产收益率": "roe", "营业总收入-营业总收入": "rev_cum"}
        keep = {k: v for k, v in cols.items() if k in df.columns}
        x = df[list(keep)].rename(columns=keep).copy()
        x["report"] = pd.to_datetime(d, format="%Y%m%d")
        for c in ["eps_cum", "ni_cum", "bps", "roe", "rev_cum"]:
            if c in x.columns:
                x[c] = pd.to_numeric(x[c], errors="coerce")
        x["code6"] = x["code6"].astype(str).str.zfill(6)
        frames.append(x)
    d = pd.concat(frames, ignore_index=True)
    d = d.drop_duplicates(["code6", "report"], keep="last")
    d["symbol"] = d["code6"].map(sym6_to_panel)
    d["q"] = d["report"].dt.quarter
    d["y"] = d["report"].dt.year

    # ---- TTM：完全向量化（此前用 groupby+iterrows，40 万行上慢到超时）----
    # 中国报表为「年初至今累计」口径。TTM 折算规则：
    #   年报(q=4)      TTM = 本期累计
    #   其余季度       TTM = 本期累计 + 上年年报 − 上年同期累计
    d = d.sort_values(["symbol", "report"])
    ann = (d.loc[d["q"] == 4, ["symbol", "y", "eps_cum", "ni_cum"]]
             .rename(columns={"eps_cum": "ann_eps", "ni_cum": "ann_ni"}))
    ann["y"] = ann["y"] + 1                      # 供下一年使用
    d = d.merge(ann, on=["symbol", "y"], how="left")
    prv = d[["symbol", "y", "q", "eps_cum", "ni_cum"]].copy()
    prv["y"] = prv["y"] + 1
    prv = prv.rename(columns={"eps_cum": "prv_eps", "ni_cum": "prv_ni"})
    d = d.merge(prv, on=["symbol", "y", "q"], how="left")
    is_ann = (d["q"] == 4).to_numpy()
    d["eps_ttm"] = np.where(is_ann, d["eps_cum"],
                            d["eps_cum"] + d["ann_eps"] - d["prv_eps"])
    d["ni_ttm"] = np.where(is_ann, d["ni_cum"],
                           d["ni_cum"] + d["ann_ni"] - d["prv_ni"])
    d["avail"] = d["report"].map(avail_date)
    t = d[["symbol", "report", "avail", "eps_ttm", "ni_ttm", "bps", "roe"]].copy()
    print(f"  季报长表: {len(t):,} 行  {t['report'].min().date()} ~ {t['report'].max().date()}"
          f"  股票 {t['symbol'].nunique()}")
    return t


def main():
    t0 = time.time()
    print("=" * 88)
    print("构建 CH-3 / CH-4（Liu-Stambaugh-Yuan 2019）")
    print("=" * 88)

    print("\n[1] 载入季报并折算 TTM")
    fund = load_yjbb()

    print("\n[2] 载入面板（价格 / 流通市值 / 收益）")
    cache = os.path.join(DATA, "ch3_prices.pkl")
    if os.path.exists(cache):
        C = pd.read_pickle(cache)
        close_me, raw_me, mcap_me = C["close_me"], C["raw_me"], C["mcap_me"]
        ret_me, abn_turn = C["ret_me"], C["abn_turn"]
        ret_fwd = close_me.shift(-1) / close_me - 1.0
        me_dates = close_me.index
        print(f"  [缓存] 月末价格面板 {close_me.shape}  {len(me_dates)} 个月末")
    else:
        from ashare_factor_strategy import load_panel, Panels
        df = load_panel()
        P = Panels(df)
        del df

        md = pd.read_pickle(os.path.join(DATA, "market_daily.pkl"))
        me_dates = pd.DatetimeIndex(md.index.to_series().resample("ME").last().dropna().values)
        me_dates = me_dates[(me_dates >= P.dates[0]) & (me_dates <= P.dates[-1])]
        print(f"  月末 {len(me_dates)} 个：{me_dates[0].date()} ~ {me_dates[-1].date()}")

        # 月末价格与市值
        # ★ 关键：估值比率必须用**真实成交价**（raw_close），不能用前复权价。
        #   前复权序列的定义是「最新价 = 真实价，历史价按累计分红/送转下调」，
        #   而 eps_ttm / bps 是**当期实际金额**。两者相除会使历史月份的 EP/BM
        #   被系统性放大（倍数因股而异，取决于累计分红），制造虚假的截面差异。
        #   收益率则相反，必须用前复权价 —— 故下面两套价格并存。
        close_me = P.close.reindex(me_dates, method="ffill")        # 收益用（前复权）
        raw_me = P.raw_close.reindex(me_dates, method="ffill")      # 估值用（真实价）
        mcap_me = P.float_mktcap.reindex(me_dates, method="ffill")
        # ★★ 时间对齐（本项目最容易犯、后果最严重的错误）★★
        #   ret_me[T]  = close[T] / close[T-1] - 1   → **T-1 到 T 已经实现**的收益
        #   ret_fwd[T] = close[T+1] / close[T] - 1   → **T 到 T+1 将要实现**的收益
        # 组合构建必须用 ret_fwd：在月末 T 观测到财务/换手特征，持有到 T+1 月末。
        # 若误用 ret_me，等于用因子解释**已经发生的过去收益**；在 A 股强烈的短期
        # 反转下会得到与横截面 IC 符号相反的荒谬价差
        # （实测报警信号：BM 的 IC=+0.034 但价差=−3.22%/月 t=−6.08；
        #   abn_turn 的 IC=−0.032 但价差=−6.92%/月 t=−17.04 —— 符号矛盾）。
        ret_me = close_me.pct_change()
        ret_fwd = close_me.shift(-1) / close_me - 1.0
        first_prev = P.dates[P.dates < me_dates[0]]
        if len(first_prev):
            prev_close = P.close.loc[first_prev[-1]]
            ret_me.iloc[0] = (close_me.iloc[0] / prev_close - 1).reindex(close_me.columns)
        # 异常换手率 PMO：过去一月 / 过去一年
        t1 = P.turnover.rolling(21, min_periods=15).mean().reindex(me_dates, method="ffill")
        t12 = P.turnover.rolling(252, min_periods=150).mean().reindex(me_dates, method="ffill")
        abn_turn = t1 / t12.replace(0, np.nan)
        del P
        pd.to_pickle({"close_me": close_me, "raw_me": raw_me, "mcap_me": mcap_me,
                      "ret_me": ret_me, "ret_fwd": ret_fwd, "abn_turn": abn_turn}, cache)
        print(f"  [缓存已写入] {cache}")

    print("\n[3] 逐月末做时点正确的财务匹配")
    # 增量指针：按 avail 排序后顺序推进，避免对 40 万行做 201 次全表 groupby
    fund = fund.sort_values(["avail", "report"])
    fund_by_avail = list(fund.groupby("avail", sort=True))
    latest = None            # 累积的「截至当前已披露」最新一期
    expo_rows = {}
    ptr = 0
    for dt in me_dates:
        while ptr < len(fund_by_avail) and fund_by_avail[ptr][0] <= dt:
            chunk = fund_by_avail[ptr][1]
            latest = chunk if latest is None else pd.concat([latest, chunk], ignore_index=True)
            ptr += 1
        if latest is None or latest.empty:
            continue
        f = latest.sort_values("report").groupby("symbol").tail(1).set_index("symbol")
        common = f.index.intersection(close_me.columns)
        if len(common) < 100:
            continue
        px = raw_me.loc[dt, common]          # 真实成交价（估值口径）
        mc = mcap_me.loc[dt, common]
        e = pd.DataFrame({
            "price": px, "mcap": mc,
            "eps_ttm": f.loc[common, "eps_ttm"],
            "bps": f.loc[common, "bps"],
            "roe": f.loc[common, "roe"]})
        with np.errstate(divide="ignore", invalid="ignore"):
            e["EP"] = e["eps_ttm"] / e["price"]
            e["BM"] = e["bps"] / e["price"]
        e.loc[~e["EP"].between(-5, 5), "EP"] = np.nan
        e.loc[~e["BM"].between(0, 20), "BM"] = np.nan
        e["abn_turn"] = abn_turn.loc[dt].reindex(common)
        expo_rows[dt] = e
    print(f"  有效月末 {len(expo_rows)} 个")

    print("\n[4] 构建 6 个市值加权组合 → SMB / VMG / PMO")
    res = []
    for dt in me_dates:
        if dt not in expo_rows:
            continue
        e = expo_rows[dt]
        # 股票池：市值前 70%（LSY）
        ok = e["mcap"].notna() & (e["mcap"] > 0)
        if ok.sum() < 100:
            continue
        q30 = e.loc[ok, "mcap"].quantile(0.30)
        e = e[ok & (e["mcap"] > q30)].copy()
        r = ret_fwd.loc[dt].reindex(e.index)      # ★ 前瞻收益，不是当期已实现收益
        e = e[r.notna()]
        r = r[e.index]
        if len(e) < 80:
            continue
        w = e["mcap"] / e["mcap"].sum()

        def vw(mask):
            m = mask & r.notna()
            if m.sum() < 3:
                return np.nan
            ww = w[m] / w[m].sum()
            return float((ww * r[m]).sum())

        sz = e["mcap"] > e["mcap"].median()
        ep = e["EP"]
        v = ep >= ep.quantile(0.70)
        g = ep < ep.quantile(0.30)
        mid = (~v) & (~g)
        # 异常换手率按 30/70 分位分 3 组；PMO 的构建**完全镜像 VMG**：
        #   VMG = 1/2(S/V + B/V) − 1/2(S/G + B/G)
        #   PMO = 1/2(S/L + B/L) − 1/2(S/H + B/H)     （做多低异常换手）
        # 此前误写成 (全市场低 + 小盘低)/2 − (全市场高 + 小盘高)/2，
        # 相当于只取了「小盘低减小盘高」的一半，且完全没用中组，量级被严重放大。
        at = e["abn_turn"]
        a_lo = at <= at.quantile(0.30)
        a_hi = at >= at.quantile(0.70)

        smb = (vw(sz & v) + vw(sz & mid) + vw(sz & g)) / 3 - \
              (vw(~sz & v) + vw(~sz & mid) + vw(~sz & g)) / 3
        vmg = (vw(sz & v) + vw(~sz & v)) / 2 - (vw(sz & g) + vw(~sz & g)) / 2
        pmo = (vw(sz & a_lo) + vw(~sz & a_lo)) / 2 - \
              (vw(sz & a_hi) + vw(~sz & a_hi)) / 2
        mkt = float((w * r).sum())
        res.append((dt, mkt, smb, vmg, pmo, int(len(e))))

    ch = pd.DataFrame(res, columns=["date", "MKT", "SMB", "VMG", "PMO", "n"]).set_index("date")
    ch["MKT"] = ch["MKT"] - (1 + RF_ANNUAL) ** (1 / 12) + 1     # 减无风险利率
    # CH-4 的 SMB：EP 中性化与换手中性化 SMB 的均值（此处用简化口径并披露）
    ch["SMB4"] = ch["SMB"]
    ch.to_pickle(os.path.join(DATA, "ch3.pkl"))

    print(f"\n  月末因子数 {len(ch)}  {ch.index[0].date()} ~ {ch.index[-1].date()}")
    print(f"  {'因子':6s} {'均值(月)':>10s} {'年化':>9s} {'标准差(月)':>11s} {'t':>7s} "
          f"{'论文对照':>14s}")
    ref = {"MKT": "0.66% (t=1.16)", "SMB": "1.03% (t=3.25)",
           "VMG": "1.14% (t=4.34)", "PMO": "≈1.0%/月"}
    for c in ["MKT", "SMB", "VMG", "PMO"]:
        s = ch[c].dropna()
        if len(s) < 20:
            continue
        mu = s.mean()
        t = mu / s.std() * np.sqrt(len(s))
        print(f"  {c:6s} {mu*100:>9.3f}% {((1+mu)**12-1)*100:>8.2f}% "
              f"{s.std()*100:>10.3f}% {t:>7.2f} {ref.get(c,''):>14s}")

    # 保存暴露面板
    pd.to_pickle(expo_rows, os.path.join(DATA, "ch3_expo.pkl"))
    print(f"\n[已保存] data/ch3.pkl 与 data/ch3_expo.pkl   用时 {(time.time()-t0)/60:.1f} 分")


if __name__ == "__main__":
    main()
