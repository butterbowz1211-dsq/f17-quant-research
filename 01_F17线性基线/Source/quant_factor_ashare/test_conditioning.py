# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
test_conditioning.py — 情绪/流动性状态条件化异常收益（Stambaugh-Yu-Yuan 2012 式）
================================================================================
动机
----
risk_attribution.py 已经排除两条路：
  × Daniel & Moskowitz (2016) 的「熊市×波动」panic 状态（对本策略无效，t=0.90）
  × 股指期货 beta 对冲（A 股贴水成本吞掉全部收益）
剩下的方向是**时变暴露**：异常收益对投资者情绪的敏感度。

文献
----
Stambaugh, Yu & Yuan (2012, JFE 104(2), 288-302) "The short of it: Investor
sentiment and anomalies"：把情绪当作**时间序列条件变量**去调节**截面异常**的收益。
Baker & Wurgler (2006, JF) 给出情绪指数的构造配方（六个代理的主成分）。

可用代理（全部由本地面板导出，见 build_market_ext.py）
  turnover_agg  总成交额/总流通市值 —— A 股最经典的情绪指标，也是 BW 的 turnover 代理
  ipo_n         新上市股票数        —— BW 的 IPO 数量代理
  issuance      净股权发行/总市值   —— BW 的 equity share in new issues 代理
  dispersion    截面收益标准差      —— 意见分歧 / 套利限制
  small_big     小盘减大盘          —— 规模风格状态
  illiq_agg     聚合 Amihud         —— 市场流动性
  comove_ew     平均相关性          —— 联动性
  breadth       上涨股占比          —— 市场宽度
  n_stock       参与股票数          —— 已发现 t=-3.03
  另加 DM 的两个对照变量以免遗漏

方法学纪律（关键）
------------------
1. 所有状态变量用**截至 t-1 月末**的信息，预测第 t 月策略收益。
2. 状态变量的标准化（z-score）参数只用**样本内 2010-2022**估计。
3. 规则的选择（选哪个变量、什么形式）只在**样本内**做。
4. 选定后原样搬到**样本外 2023-2026** 报告，不回头调整。
   —— 这样整个叠加层只消耗 1 次试验次数，不会污染多重检验。
"""

import os
import sys
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")

PPY = 252
RF = 0.02
IS = ("2010-01-01", "2022-12-31")
OOS = ("2023-01-01", "2026-12-31")

STATE_VARS = [
    "turnover_agg", "ipo_n", "issuance", "dispersion", "small_big",
    "illiq_agg", "comove_ew", "breadth", "n_stock", "mkt_vol126", "mkt_24m",
]


def perf(r, name=""):
    r = pd.Series(r).dropna()
    if len(r) < 20:
        return {"name": name}
    eq = (1 + r).cumprod()
    yrs = len(r) / PPY
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(PPY)
    dd = eq / eq.cummax() - 1
    mdd = float(dd.min())
    return {"name": name, "CAGR": round(float(cagr), 4), "AnnVol": round(float(vol), 4),
            "Sharpe": round(float((r.mean() * PPY - RF) / vol), 4) if vol > 0 else None,
            "MaxDD": round(mdd, 4),
            "Calmar": round(float(cagr / abs(mdd)), 3) if mdd < 0 else None,
            "n": int(len(r))}


def build_states():
    m = pd.read_pickle(os.path.join(DATA, "market_daily2.pkl"))
    d = pd.DataFrame(index=m.index)
    d["mkt_vw"] = m["mkt_vw"]
    d["turnover_agg"] = m["turnover_agg"]
    d["ipo_n"] = m["ipo_n"]
    d["issuance"] = m["issuance"].clip(upper=m["issuance"].quantile(0.99))
    d["dispersion"] = m["dispersion"]
    d["illiq_agg"] = m["illiq_agg"]
    d["comove_ew"] = m["comove_ew"]
    d["breadth"] = m["breadth"]
    d["n_stock"] = m["n_stock"]
    d["small_big"] = m["small_big"]

    # 月度聚合
    mo = pd.DataFrame(index=d.resample("ME").last().index)
    mo["turnover_agg"] = d["turnover_agg"].resample("ME").mean()
    mo["ipo_n"] = d["ipo_n"].resample("ME").sum()
    mo["issuance"] = d["issuance"].resample("ME").sum()
    mo["dispersion"] = d["dispersion"].resample("ME").mean()
    mo["illiq_agg"] = d["illiq_agg"].resample("ME").mean()
    mo["comove_ew"] = d["comove_ew"].resample("ME").mean()
    mo["breadth"] = d["breadth"].resample("ME").mean()
    mo["n_stock"] = d["n_stock"].resample("ME").last()
    mo["small_big"] = d["small_big"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    mo["mkt_vw"] = d["mkt_vw"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    mo["mkt_vol126"] = d["mkt_vw"].rolling(126).std().resample("ME").last() * np.sqrt(PPY)
    mo["mkt_24m"] = (1 + mo["mkt_vw"]).rolling(24).apply(np.prod, raw=True) - 1

    # 派生：变化率版本
    for c in ["turnover_agg", "ipo_n", "dispersion", "illiq_agg", "comove_ew",
              "breadth", "n_stock"]:
        mo[c + "_chg"] = mo[c].pct_change(12)
    return mo


def main():
    print("=" * 84)
    print("情绪 / 流动性状态条件化检验（Stambaugh-Yu-Yuan 2012 式）")
    print("=" * 84)

    mo = build_states()
    with open(os.path.join(OUT, "returns_LSY.pkl"), "rb") as fh:
        R = pickle.load(fh)
    LONG = "多头组合(可交易)"
    LS = "多空组合(学术)"
    r_long = R[LONG]["net"].dropna()
    r_ls = R[LS]["net"].dropna()
    ml = r_long.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    mls = r_ls.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    candidates = STATE_VARS + [c + "_chg" for c in
                               ["turnover_agg", "ipo_n", "dispersion", "illiq_agg",
                                "comove_ew", "breadth", "n_stock"]]

    # 样本内 z-score 参数
    is_mask = (mo.index >= IS[0]) & (mo.index <= IS[1])
    Z = pd.DataFrame(index=mo.index)
    for c in candidates:
        if c not in mo.columns:
            continue
        mu, sd = mo.loc[is_mask, c].mean(), mo.loc[is_mask, c].std()
        if not np.isfinite(sd) or sd == 0:
            continue
        Z[c] = (mo[c] - mu) / sd
    Z = Z.shift(1)          # 只用 t-1 月末信息

    result = {"univariate": {}, "quintile_IS": {}, "chosen": None, "overlay": {}}

    # ---------------- 1. 单变量预测回归 ----------------
    print("\n" + "=" * 84)
    print("1  单变量预测回归：y_t = a + b*z_{t-1} + e   (NW lag=6)")
    print("=" * 84)
    print(f"{'状态变量':22s} {'样本内 β':>11s} {'t':>7s} {'R²':>7s} | "
          f"{'样本外 β':>11s} {'t':>7s} {'R²':>7s}")
    print("-" * 84)
    rows = []
    for c in Z.columns:
        rec = {}
        for lab, (s, e) in [("IS", IS), ("OOS", OOS)]:
            for y, ynm in [(ml, "long"), (mls, "ls")]:
                dd = pd.concat({"y": y, "x": Z[c]}, axis=1).dropna()
                dd = dd.loc[(dd.index >= s) & (dd.index <= e)]
                if len(dd) < 30:
                    continue
                mm = sm.OLS(dd["y"], sm.add_constant(dd["x"])).fit(
                    cov_type="HAC", cov_kwds={"maxlags": 6})
                rec[f"{lab}_{ynm}"] = (float(mm.params["x"]), float(mm.tvalues["x"]),
                                       float(mm.rsquared))
        if "IS_long" in rec:
            b, t, r2 = rec["IS_long"]
            o = rec.get("OOS_long", (np.nan, np.nan, np.nan))
            flag = "  <<<<" if abs(t) > 2.0 else ""
            print(f"{c:22s} {b:>+11.4f} {t:>+7.2f} {r2:>7.4f} | "
                  f"{o[0]:>+11.4f} {o[1]:>+7.2f} {o[2]:>7.4f}{flag}")
            rows.append((c, abs(t), rec))
        result["univariate"][c] = {k: [round(v[0], 5), round(v[1], 2), round(v[2], 4)]
                                   for k, v in rec.items()}

    # ---------------- 2. 样本内分位数检验 ----------------
    print(f"\n{'='*84}\n2  按状态变量分 5 组的月度收益（样本内 2010-2022，多头）\n{'='*84}")
    print(f"{'状态变量':22s} {'Q1(低)':>9s} {'Q2':>9s} {'Q3':>9s} {'Q4':>9s} {'Q5(高)':>9s}"
          f" {'Q5-Q1':>9s} {'t':>7s}")
    print("-" * 84)
    for c in Z.columns:
        dd = pd.concat({"y": ml, "x": Z[c]}, axis=1).dropna()
        dd = dd.loc[(dd.index >= IS[0]) & (dd.index <= IS[1])]
        if len(dd) < 60:
            continue
        try:
            q = pd.qcut(dd["x"], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        g = dd.groupby(q)["y"].mean()
        hi = dd.loc[q == q.max(), "y"]
        lo = dd.loc[q == q.min(), "y"]
        diff = hi.mean() - lo.mean()
        se = np.sqrt(hi.var() / len(hi) + lo.var() / len(lo))
        t = diff / se * np.sqrt(1.0) if se > 0 else np.nan
        vals = [f"{g.get(i, np.nan):>+9.4f}" for i in range(5)]
        print(f"{c:22s} " + " ".join(vals) + f" {diff:>+9.4f} {t:>+7.2f}")
        result["quintile_IS"][c] = {"q": [round(float(g.get(i, np.nan)), 5) for i in range(5)],
                                    "hi_minus_lo": round(float(diff), 5),
                                    "t": round(float(t), 2)}

    # ---------------- 3. 挑选叠加规则（仅用样本内） ----------------
    print(f"\n{'='*84}\n3  构造时变暴露规则（只在样本内挑选）\n{'='*84}")
    print("  规则形式：  w_t = clip(1 - k·z_{t-1}^*, 0, 1)   对多头组合按 w 缩放")
    print("  其中 z* 取样本内 |t| 最大的状态变量，k 在样本内网格搜索使 Sharpe 最大\n")

    # 用样本内挑：|t| 最大者
    rows_sorted = sorted(rows, key=lambda r: -r[1])
    best_var = rows_sorted[0][0]
    print(f"  样本内 |t| 最大：{best_var}  (t={rows_sorted[0][2]['IS_long'][1]:+.2f})")

    def apply_overlay(r_daily, z_monthly, k, mode="linear", cap=1.0):
        """把月度状态转成日频权重并作用于日收益"""
        s = z_monthly.dropna()
        w = (1.0 - k * s).clip(0.0, cap)
        w_d = w.reindex(r_daily.index, method="ffill").shift(1).fillna(1.0)
        return r_daily * w_d

    rr = r_long.loc[IS[0]:IS[1]]
    grid = {}
    for var in [best_var] + [r[0] for r in rows_sorted[1:4]]:
        for k in [0.0, 0.15, 0.30, 0.45, 0.60, 0.80]:
            p = perf(apply_overlay(rr, Z[var], k), f"{var}|k={k}")
            grid[f"{var}|{k}"] = p
    # 样本内最优
    best_key = max([k for k in grid if grid[k].get("Sharpe") is not None],
                   key=lambda k: grid[k]["Sharpe"])
    bvar, bk = best_key.split("|")
    bk = float(bk)
    print(f"  样本内最优：变量={bvar}  k={bk}  → Sharpe {grid[best_key]['Sharpe']:.3f} "
          f"(基准 {perf(rr)['Sharpe']:.3f})")
    print(f"\n  样本内网格（前 12 名）：")
    for kk in sorted(grid, key=lambda x: -grid[x]["Sharpe"])[:12]:
        p = grid[kk]
        print(f"    {kk:28s} 年化 {p['CAGR']:>7.2%}  波动 {p['AnnVol']:>7.2%}  "
              f"Sharpe {p['Sharpe']:>6.3f}  回撤 {p['MaxDD']:>8.2%}")

    result["chosen"] = {"var": bvar, "k": bk,
                        "IS_sharpe_base": perf(rr)["Sharpe"],
                        "IS_sharpe_overlay": grid[best_key]["Sharpe"]}

    # ---------------- 4. 样本外验证（原样搬过去） ----------------
    print(f"\n{'='*84}\n4  样本外验证：把样本内选定的规则原样应用到 2023-2026\n{'='*84}")
    ro = r_long.loc[OOS[0]:OOS[1]]
    base_oos = perf(ro, "多头 样本外 原始")
    over_oos = perf(apply_overlay(ro, Z[bvar], bk), f"多头 样本外 +{bvar}(k={bk})")
    print(f"  {'方案':34s} {'年化':>8s} {'波动':>8s} {'夏普':>7s} {'最大回撤':>9s} {'Calmar':>7s}")
    for p in [base_oos, over_oos]:
        print(f"  {p['name']:34s} {p['CAGR']:>8.2%} {p['AnnVol']:>8.2%} {p['Sharpe']:>7.3f} "
              f"{p['MaxDD']:>9.2%} {str(p['Calmar']):>7s}")
    result["overlay"]["long_oos"] = {"base": base_oos, "overlay": over_oos}

    # 多空也跑一遍
    rls_o = r_ls.loc[OOS[0]:OOS[1]]
    b_ls = perf(rls_o, "多空 样本外 原始")
    o_ls = perf(apply_overlay(rls_o, Z[bvar], bk), f"多空 样本外 +{bvar}(k={bk})")
    print()
    for p in [b_ls, o_ls]:
        print(f"  {p['name']:34s} {p['CAGR']:>8.2%} {p['AnnVol']:>8.2%} {p['Sharpe']:>7.3f} "
              f"{p['MaxDD']:>9.2%} {str(p['Calmar']):>7s}")
    result["overlay"]["ls_oos"] = {"base": b_ls, "overlay": o_ls}

    with open(os.path.join(OUT, "conditioning.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n[已保存] {os.path.join(OUT,'conditioning.json')}")


if __name__ == "__main__":
    main()
