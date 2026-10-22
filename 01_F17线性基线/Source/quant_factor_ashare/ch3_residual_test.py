# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
ch3_residual_test.py — 因子对 CH-3 / CH-4 的存活检验
================================================================================
这是本轮最关键的检验。

文献依据
--------
Hou, Qiao & Zhang (2023), "Finding Anomalies in China"（426 个异象）：
    市值加权下显著            98 个
    经 CAPM 调整后            97 个
    经 CH-3 调整后            16 个
    经 CH-4 调整后            **0 个**
Liu, Stambaugh & Yuan (2019) 的 CH-3 = MKT + SMB + VMG，CH-4 再加 PMO。

要回答的问题
------------
我们那 17 个入选因子里，有多少在控制 CH-3 / CH-4 之后仍然显著？
如果绝大多数只是规模 + 价值的暴露，那么策略的"alpha"本质上是 beta，
其实盘价值与风险特征就完全不同 —— 这比再优化几个基点重要得多。

方法
----
1. 每月末在 LSY 股票池（市值前 70%）内，按**预处理后**的因子值取前 30% / 后 30%，
   等权多空，得到该因子的月度多空收益序列（200 个月）。
2. 依次对以下模型回归，取 alpha 及其 Newey-West t 值：
       原始          R = a
       CAPM          R = a + b·MKT
       CH-3          R = a + b1·MKT + b2·SMB + b3·VMG
       CH-4          R = a + b1·MKT + b2·SMB + b3·VMG + b4·PMO
3. 同样的检验施加于**策略本身**（多头组合与多空组合）。

注意
----
预处理已做过「行业 + 市值中性化」，这本身会剥离一部分 SMB 暴露，
因此残余的 CH-3 暴露应主要来自价值（VMG）与市场（MKT）维度。
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
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
PREP = os.path.join(DATA, "prep")

F17 = ["TURN", "MAX5", "INTRADAY", "ILLIQ", "COMMOVE", "TURN_SHOCK", "REV_1W",
       "ISKEW", "IVOL", "OVERNIGHT", "LEAD_LAG", "CO_SKEW", "BETA", "REV_LT",
       "SIZE", "HIGH52", "RES_MOM"]
EXTRA = ["CGO", "CGO_R", "ABN_TURN", "VTURN"]
ALLF = F17 + EXTRA
NW_LAGS = 6


def main():
    print("=" * 96)
    print("因子对 CH-3 / CH-4 的存活检验")
    print("=" * 96)

    C = pd.read_pickle(os.path.join(DATA, "ch3_prices.pkl"))
    close_me = C["close_me"]
    ret_fwd = close_me.shift(-1) / close_me - 1.0
    expo = pd.read_pickle(os.path.join(DATA, "ch3_expo.pkl"))
    ch = pd.read_pickle(os.path.join(DATA, "ch3.pkl"))
    months = [d for d in ch.index if d in expo]

    print(f"\n月末 {len(months)} 个  {months[0].date()} ~ {months[-1].date()}")
    print(f"CH-3/4 因子标准差（月，%）：" +
          "  ".join(f"{c}={ch[c].std()*100:.2f}" for c in ["MKT", "SMB", "VMG", "PMO"]))

    # 模型定义
    MODELS = {
        "原始":  [],
        "CAPM":  ["MKT"],
        "CH-3":  ["MKT", "SMB", "VMG"],
        "CH-4":  ["MKT", "SMB", "VMG", "PMO"],
    }

    def run_reg(series, tag):
        """对四个模型分别回归，返回 {模型: (alpha月%, t, R2)}"""
        out = {}
        for mname, cols in MODELS.items():
            d = pd.concat({"y": series, **{c: ch[c] for c in cols}}, axis=1).dropna()
            if len(d) < 40:
                continue
            X = sm.add_constant(d[cols]) if cols else pd.DataFrame(
                {"const": np.ones(len(d))}, index=d.index)
            m = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
            out[mname] = (float(m.params["const"]) * 100,
                          float(m.tvalues["const"]),
                          float(m.rsquared))
        return out

    # ---------------- 因子多空收益 ----------------
    print("\n" + "=" * 96)
    print("1  各因子月度多空收益（前30% − 后30%，等权）对 CH-3 / CH-4 的回归")
    print("=" * 96)
    print(f"  {'因子':10s} | {'原始α':>9s} {'t':>6s} | {'CAPM α':>9s} {'t':>6s} | "
          f"{'CH-3 α':>9s} {'t':>6s} | {'CH-4 α':>9s} {'t':>6s} | 判定")
    print("  " + "-" * 94)

    results = {}
    for nm in ALLF:
        p = os.path.join(PREP, nm + ".pkl")
        if not os.path.exists(p):
            continue
        fac = pd.read_pickle(p)
        rows = []
        for dt in months:
            e = expo.get(dt)
            if e is None:
                continue
            x = fac.loc[dt].reindex(e.index) if dt in fac.index else None
            if x is None:
                continue
            r = ret_fwd.loc[dt].reindex(e.index)
            m = x.notna() & r.notna()
            if m.sum() < 80:
                continue
            xx, rr = x[m], r[m]
            hi = xx >= xx.quantile(0.70)
            lo = xx <= xx.quantile(0.30)
            if hi.sum() < 5 or lo.sum() < 5:
                continue
            rows.append((dt, float(rr[hi].mean() - rr[lo].mean())))
        if len(rows) < 40:
            continue
        # 用 (日期, 值) 对构造序列，保证与 ch 的月索引正确对齐（跳过缺月）
        s = pd.Series(dict(rows)).sort_index()
        r = run_reg(s, nm)
        results[nm] = {k: [round(v[0], 4), round(v[1], 2), round(v[2], 4)]
                       for k, v in r.items()}
        def cell(k):
            if k not in r:
                return f"{'—':>9s} {'—':>6s}"
            return f"{r[k][0]:>+9.3f} {r[k][1]:>+6.2f}"
        c3 = r.get("CH-3", (np.nan, np.nan, np.nan))
        c4 = r.get("CH-4", (np.nan, np.nan, np.nan))
        if abs(c4[1]) >= 1.96:
            verdict = "★ CH-4 后仍显著"
        elif abs(c3[1]) >= 1.96:
            verdict = "○ 仅 CH-3 后显著"
        else:
            verdict = "× CH-3 即被吸收"
        print(f"  {nm:10s} | {cell('原始')} | {cell('CAPM')} | {cell('CH-3')} | "
              f"{cell('CH-4')} | {verdict}")

    # ---------------- 策略本身 ----------------
    print("\n" + "=" * 96)
    print("2  策略组合对 CH-3 / CH-4 的回归（月末口径）")
    print("=" * 96)
    rp = os.path.join(OUT, "v2_returns.pkl")
    import pickle
    if os.path.exists(rp):
        with open(rp, "rb") as fh:
            R = pickle.load(fh)
        for key, lab in [("net", "改进型多头组合(净)"), ("ls_net", "改进型多空组合(净)"),
                         ("baseline_net", "原版多头组合(净)")]:
            if key not in R:
                continue
            daily = R[key]
            mm = daily.resample("ME").apply(lambda x: (1 + x).prod() - 1)
            rr = mm.reindex(ch.index).dropna()
            r = run_reg(rr, lab)
            results["策略|" + lab] = {k: [round(v[0], 4), round(v[1], 2), round(v[2], 4)]
                                     for k, v in r.items()}
            line = "  ".join(f"{k}: α={v[0]:+.3f}% (t={v[1]:+.2f}) R²={v[2]:.3f}"
                             for k, v in r.items())
            print(f"  {lab:22s} {line}")

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 96)
    print("3  汇总：有多少因子在 CH-3 / CH-4 后存活？")
    print("=" * 96)
    n_capm = None
    tot = sum(1 for k in results if not k.startswith("策略"))
    for mn in ["原始", "CAPM", "CH-3", "CH-4"]:
        n = sum(1 for k, v in results.items()
                if not k.startswith("策略") and mn in v and abs(v[mn][1]) >= 1.96)
        print(f"  经 {mn:5s} 调整后 |t|>=1.96 的因子数：{n} / {tot}")
    print(f"\n  论文对照（HQZ 2023，426 个异象）：原始 98 → CAPM 97 → CH-3 16 → CH-4 0")

    with open(os.path.join(OUT, "ch3_residual.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\n[已保存] {os.path.join(OUT,'ch3_residual.json')}")


if __name__ == "__main__":
    main()
