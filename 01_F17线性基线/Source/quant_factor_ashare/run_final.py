# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
run_final.py — 最终交付：推荐配置的完整业绩与显著性，产出报告与图表
================================================================================
推荐配置（选择依据见 output/第二轮改进报告.md）
---------------------------------------------
    因子集   F17（全部 17 个，相位检验证明精简反而更差）
    持仓     得分前 50 名等权
    权重上限 3%（用不动点迭代 _apply_cap 保证真正生效）
    缓冲区   0.6（老持仓仍在候选前 80 名内则保留）
    调仓     每 21 个交易日
    股票池   LSY（剔除最小 30% 流通市值 + ST + 次新 + 涨停）

产出
----
output/final_report.md        完整业绩与显著性报告
output/final_summary.json     结构化结果
output/final_equity.png 等    图表

显著性
------
· Newey-West t（Bartlett 核，lag=5）
· PSR（Bailey & López de Prado）
· DSR（多重检验校正），试验次数按**累计评估过的配置数**诚实计入
· CH-4 alpha（Liu-Stambaugh-Yuan 2019），控制中国市场三/四因子
· 相位稳健性：报告 11 个调仓相位的均值与区间，不只用单相位
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
sys.path.insert(0, HERE)

from ashare_factor_strategy import (load_panel, Panels, build_universe, backtest,
                                    CFG, newey_west_tstat, probabilistic_sharpe,
                                    deflated_sharpe)
import strategy_v2 as v2

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
PPY = 252
RF = 0.02
IS = ("2010-01-01", "2022-12-31")
OOS = ("2023-01-01", "2026-12-31")

REC = dict(n=50, weighting="equal", buffer=0.6, rebalance_days=21, max_weight=0.03)
BASE = dict(n=50, weighting="equal", buffer=0.0, rebalance_days=21, max_weight=0.05)
N_TRIALS = 80          # grid + tune + beta + phase 累计评估过的配置数


def met(r):
    r = pd.Series(r).dropna()
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / PPY
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(PPY)
    dd = eq / eq.cummax() - 1
    mdd = float(dd.min())
    dn = r[r < 0].std() * np.sqrt(PPY)
    return {"CAGR": round(float(cagr), 4), "AnnVol": round(float(vol), 4),
            "Sharpe": round(float((r.mean() * PPY - RF) / vol), 4),
            "MaxDD": round(mdd, 4),
            "Calmar": round(float(cagr / abs(mdd)), 3) if mdd < 0 else None,
            "Sortino": round(float((r.mean() * PPY - RF) / dn), 3) if dn > 0 else None,
            "WinRate": round(float((r > 0).mean()), 4),
            "Skew": round(float(r.skew()), 3),
            "n_days": int(len(r))}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False

    print("=" * 92)
    print("最终交付运行")
    print("=" * 92)

    df = load_panel()
    P = Panels(df)
    del df
    eligible, _, _ = build_universe(P, CFG)
    need = sorted(set(v2.FACTOR_SETS["F17"] + v2.EXTRA_NAMES + v2.CONTROL_NAMES))
    v2.build_prep(P, eligible, need)
    score = v2.composite(v2.FACTOR_SETS["F17"])
    print("  因子得分合成完成", flush=True)

    # ---- 相位扫描：11 个相位，两个配置 ----
    print("\n[1] 相位扫描（11 个调仓相位）", flush=True)
    scan = {}
    for lab, kw in [("推荐", REC), ("原版", BASE)]:
        nets, grosses, turns = [], [], []
        for off in range(0, 21, 2):
            sc = score.iloc[off:] if off else score
            W = v2.make_weights(sc, eligible.reindex(sc.index), **kw)
            res = backtest(P, W, CFG, f"{lab}_ph{off}")
            del W
            nets.append(res["net"].rename(off))
            turn = res["turnover"].rename(off)
            turns.append(turn)
            del res
            print(f"    {lab} 相位 {off:>2d} 完成", flush=True)
        # 各相位的样本外指标 → 分布
        per = []
        for n in nets:
            o = n[(n.index >= OOS[0]) & (n.index <= OOS[1])]
            m = met(o)
            if m:
                per.append(m)
        scan[lab] = {
            "phase_metrics": per,
            "oos_sharpe_mean": round(float(np.mean([p["Sharpe"] for p in per])), 4),
            "oos_sharpe_sd": round(float(np.std([p["Sharpe"] for p in per])), 4),
            "oos_sharpe_min": round(float(np.min([p["Sharpe"] for p in per])), 4),
            "oos_sharpe_max": round(float(np.max([p["Sharpe"] for p in per])), 4),
            "oos_cagr_mean": round(float(np.mean([p["CAGR"] for p in per])), 4),
            "oos_maxdd_mean": round(float(np.mean([p["MaxDD"] for p in per])), 4),
            "oos_calmar_mean": round(float(np.mean([p["CAGR"] for p in per]) /
                                           abs(np.mean([p["MaxDD"] for p in per]))), 3),
            "turnover_mean": round(float(np.mean([t.mean() * PPY for t in turns])), 4),
        }
        s = scan[lab]
        print(f"    → {lab}: OOS 夏普 {s['oos_sharpe_mean']:.3f}±{s['oos_sharpe_sd']:.3f} "
              f"[{s['oos_sharpe_min']:.3f},{s['oos_sharpe_max']:.3f}]  "
              f"年化 {s['oos_cagr_mean']:.2%}  回撤 {s['oos_maxdd_mean']:.2%}  "
              f"Calmar {s['oos_calmar_mean']:.2f}  换手 {s['turnover_mean']:.0%}")

    # ---- 用中位相位的日收益做完整报告 ----
    print("\n[2] 完整业绩（用相位 0 的序列做详细展示，并标注相位区间）", flush=True)
    W = v2.make_weights(score, eligible, **REC)
    res = backtest(P, W, CFG, "推荐配置")
    r_net, r_gross, turn = res["net"], res["gross"], res["turnover"]
    # ★ 多空组合必须是**自融资价差**：把多空合并成一张权重矩阵（多 +w、空 −w）
    #   交给回测引擎，它按 Σ|w| 归一化，毛暴露 = 1。
    #   此前的写法 `r_long − r_short` 是两个满仓组合相减，毛暴露 = 2，
    #   报出来的年化与波动都约为标准口径的 2 倍，不可与既往结果比较。
    Ws = v2.make_weights(score, eligible, side="short", **REC)
    cols = W.columns
    W_ls = W.sub(Ws.reindex(columns=cols, fill_value=0.0), fill_value=0.0)
    res_ls = backtest(P, W_ls, CFG, "多空自融资价差")
    r_ls = res_ls["net"]
    del W, Ws, W_ls, res_ls
    # 单看空头腿（用于披露空头腿的绝对表现）
    r_short = None

    out = {"config": {k: v for k, v in REC.items()},
           "n_trials_accounted": N_TRIALS,
           "phase_scan": {k: {kk: vv for kk, vv in v.items() if kk != "phase_metrics"}
                          for k, v in scan.items()},
           "periods": {}}
    for pname, (s, e) in [("样本内 2010-2022", IS), ("样本外 2023-2026", OOS),
                          ("全样本 2010-2026", ("2010-01-01", "2026-12-31"))]:
        for pnm, ser in [("多头", r_net), ("多空", r_ls)]:
            seg = ser[(ser.index >= s) & (ser.index <= e)]
            out["periods"][f"{pnm}|{pname}"] = met(seg)
    # 基准
    Wb = v2.universe_weights(eligible)
    rb = backtest(P, Wb, CFG, "基准")["net"]
    for pname, (s, e) in [("样本内 2010-2022", IS), ("样本外 2023-2026", OOS)]:
        seg = rb[(rb.index >= s) & (rb.index <= e)]
        out["periods"][f"基准|{pname}"] = met(seg)

    # ---- CH-4 归因 ----
    print("\n[3] CH-4 归因", flush=True)
    ch = pd.read_pickle(os.path.join(DATA, "ch3.pkl")).shift(1)
    out["ch4"] = {}
    for pnm, ser in [("多头", r_net), ("多空", r_ls)]:
        mm = ser.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        for pname, (s, e) in [("全样本", ("2010-01-01", "2026-12-31")),
                              ("样本内", IS), ("样本外", OOS)]:
            sub = mm[(mm.index >= s) & (mm.index <= e)]
            d = pd.concat({"y": sub, "MKT": ch["MKT"], "SMB": ch["SMB"],
                           "VMG": ch["VMG"], "PMO": ch["PMO"]}, axis=1).dropna()
            if len(d) < 24:
                continue
            m = sm.OLS(d["y"], sm.add_constant(d[["MKT", "SMB", "VMG", "PMO"]])
                       ).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
            a = float(m.params["const"])
            out["ch4"][f"{pnm}|{pname}"] = {
                "alpha_m_pct": round(a * 100, 4),
                "alpha_a_pct": round(((1 + a) ** 12 - 1) * 100, 3),
                "t": round(float(m.tvalues["const"]), 2),
                "betas": {c: round(float(m.params[c]), 3)
                          for c in ["MKT", "SMB", "VMG", "PMO"]},
                "r2": round(float(m.rsquared), 4), "n_months": int(len(d))}

    # ---- 显著性 ----
    # ★ DSR 必须传入「试验次数」对应的夏普方差，否则 deflated_sharpe 内部
    #   var_sr=0 → SR0=0 → DSR 退化为 PSR，等于**取消多重检验校正**。
    #   这里从本轮所有已评估配置的结果文件中收集夏普，诚实计入试验分布。
    print("\n[4] 显著性检验", flush=True)
    trial_sharpes = [s["oos_sharpe_mean"] for s in scan.values()]
    trial_sharpes += [p["Sharpe"] for s in scan.values() for p in s["phase_metrics"]]
    for f in ["v2_grid.json", "v2_tune.json", "v2_beta.json", "neutral_experiment.json"]:
        fp = os.path.join(OUT, f)
        if not os.path.exists(fp):
            continue
        try:
            with open(fp, encoding="utf-8") as fh:
                j = json.load(fh)
            for k, v in j.items():
                if not isinstance(v, dict):
                    continue
                o = v.get("OOS") or (v.get("stats", {}) or {}).get("OOS")
                if isinstance(o, dict) and o.get("Sharpe") is not None:
                    trial_sharpes.append(float(o["Sharpe"]))
        except Exception:
            pass
    ts = pd.Series(trial_sharpes).replace([np.inf, -np.inf], np.nan).dropna()
    sr_var = float(ts.var(ddof=1)) if len(ts) > 1 else 0.0
    print(f"    试验夏普样本 {len(ts)} 个：均值 {ts.mean():.3f}  标准差 {ts.std():.3f}  "
          f"方差 {sr_var:.4f}  min {ts.min():.3f}  max {ts.max():.3f}")
    out["trial_distribution"] = {"n": int(len(ts)), "mean": round(float(ts.mean()), 4),
                                 "sd": round(float(ts.std()), 4),
                                 "var": round(sr_var, 4)}
    out["significance"] = {}
    for pnm, ser in [("多头", r_net), ("多空", r_ls)]:
        o = ser[(ser.index >= OOS[0]) & (ser.index <= OOS[1])]
        t = newey_west_tstat(o, lags=5)[1]
        sr = (o.mean() * PPY - RF) / (o.std() * np.sqrt(PPY))
        psr = probabilistic_sharpe(sr / np.sqrt(PPY), len(o),
                                   float(o.skew()), float(o.kurtosis() + 3.0))
        dsr = deflated_sharpe(o, n_trials=N_TRIALS, sr_variance=sr_var)
        out["significance"][f"{pnm}|样本外"] = {
            "Sharpe": round(float(sr), 4), "NW_t": round(float(t), 3),
            "PSR": round(float(psr), 4), "DSR": round(float(dsr["DSR"]), 4),
            "SR0_ann": round(float(dsr["SR0"]), 4),
            "n_obs": int(len(o))}
        f_ = ser[(ser.index >= "2010-01-01") & (ser.index <= "2026-12-31")]
        tf = newey_west_tstat(f_, lags=5)[1]
        srf = (f_.mean() * PPY - RF) / (f_.std() * np.sqrt(PPY))
        psrf = probabilistic_sharpe(srf / np.sqrt(PPY), len(f_),
                                    float(f_.skew()), float(f_.kurtosis() + 3.0))
        dsrf = deflated_sharpe(f_, n_trials=N_TRIALS, sr_variance=sr_var)
        out["significance"][f"{pnm}|全样本"] = {
            "Sharpe": round(float(srf), 4), "NW_t": round(float(tf), 3),
            "PSR": round(float(psrf), 4),
            "DSR": round(float(dsrf["DSR"]), 4), "SR0_ann": round(float(dsrf["SR0"]), 4),
            "n_obs": int(len(f_))}

    # ---- 图表 ----
    print("\n[5] 生成图表", flush=True)
    fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    for nm, ser, c in [("推荐配置 多头", r_net, "#c0392b"),
                       ("原版 多头", backtest(
                           P, v2.make_weights(score, eligible, **BASE), CFG, "原版")["net"],
                        "#2980b9"),
                       ("等权全市场基准", rb, "#7f8c8d")]:
        seg = ser[(ser.index >= OOS[0]) & (ser.index <= OOS[1])]
        eq = (1 + seg).cumprod()
        ax[0].plot(eq.index, eq.values, label=f"{nm}（夏普 {(seg.mean()*PPY-RF)/(seg.std()*np.sqrt(PPY)):.2f}）",
                   color=c, lw=1.6)
        ax[1].plot(eq.index, (eq / eq.cummax() - 1).values, color=c, lw=1.2)
    ax[0].set_ylabel("净值（样本外 2023-01 起 = 1）")
    ax[0].legend(loc="upper left", fontsize=9)
    ax[0].grid(alpha=0.3)
    ax[1].set_ylabel("回撤")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "final_equity.png"), dpi=130)
    plt.close()

    with open(os.path.join(OUT, "final_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n[已保存] {os.path.join(OUT,'final_summary.json')}")

    # ---- 打印摘要 ----
    print("\n" + "=" * 92)
    print("业绩摘要（相位 0 序列）")
    print("=" * 92)
    hdr = f"  {'组合/区间':22s} {'年化':>8s} {'波动':>8s} {'夏普':>7s} {'最大回撤':>9s} {'Calmar':>7s}"
    print(hdr)
    for k, v in out["periods"].items():
        if not v:
            continue
        print(f"  {k:22s} {v['CAGR']:>8.2%} {v['AnnVol']:>8.2%} {v['Sharpe']:>7.3f} "
              f"{v['MaxDD']:>9.2%} {str(v['Calmar']):>7s}")
    print("\n  CH-4 alpha（年化）与 t：")
    for k, v in out["ch4"].items():
        print(f"    {k:22s} α={v['alpha_a_pct']:>+7.2f}%  t={v['t']:>+5.2f}  "
              f"R²={v['r2']:.3f}  β={v['betas']}")
    print("\n  显著性：")
    for k, v in out["significance"].items():
        print(f"    {k:14s} 夏普 {v['Sharpe']:>6.3f}  NW t={v['NW_t']:>+6.2f}  "
              f"DSR={v['DSR']:.4f} (门槛 SR0={v['SR0_ann']:.3f})")


if __name__ == "__main__":
    main()
