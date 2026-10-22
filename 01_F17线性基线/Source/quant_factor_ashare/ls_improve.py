# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
ls_improve.py — 多空组合的改进实验（冲击 Sharpe>1.2）
================================================================================
为什么攻多空
------------
当前水平（相位 0，样本外）：
    多头  夏普 1.017  回撤 −25.43%  Calmar 0.847
    多空  夏普 1.078  回撤 −12.25%  Calmar 1.21
多空已满足 MaxDD<15% 与 Calmar>0.9，**只差夏普 0.12**。

三个着手点（均由前面 CH-3/CH-4 的检验直接指出）
----------------------------------------------
1. **因子权重**：现在 17 个因子等权。CH-4 存活检验给出了每个因子的风险调整后 alpha
   （COMMOVE +0.652 t=6.37 … MAX5 +0.280 t=1.70），据此加权可能优于等权。
   但等权的稳健性有理论支持（权重估计误差常大于信息增益），必须实测。
2. **中和 β_VMG = 0.511**：多空组合现在是一个隐性价值赌注（全样本 β_VMG 0.419、
   样本外 0.511）。在截面上对 EP 分位做残差化可把它压掉。
3. **相位平均**：多空此前只有相位 0 的数字，必须给出相位分布。

方法学纪律
----------
· 权重方案只在**样本内**挑选，样本外只作报告；
· 一律给相位平均与区间，不报单相位最优；
· 所有配置计入试验次数，供 DSR 使用。
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
sys.path.insert(0, HERE)

from ashare_factor_strategy import (load_panel, Panels, build_universe, backtest,
                                    CFG, fast_rank_pct)
import strategy_v2 as v2

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
PPY, RF = 252, 0.02
IS = ("2010-01-01", "2022-12-31")
OOS = ("2023-01-01", "2026-12-31")
REC = dict(n=50, weighting="equal", buffer=0.6, rebalance_days=21, max_weight=0.03)


def met(seg):
    seg = pd.Series(seg).dropna()
    if len(seg) < 20:
        return None
    eq = (1 + seg).cumprod()
    cagr = eq.iloc[-1] ** (PPY / len(seg)) - 1
    vol = seg.std() * np.sqrt(PPY)
    dd = float((eq / eq.cummax() - 1).min())
    return {"Sharpe": float((seg.mean() * PPY - RF) / vol), "CAGR": float(cagr),
            "AnnVol": float(vol), "MaxDD": dd,
            "Calmar": float(cagr / abs(dd)) if dd < 0 else None, "n": int(len(seg))}


def neutralise_score(score, eligible, extra):
    """在截面上把得分对额外暴露做残差化（FWL：逐日一元回归取残差）"""
    out = score.copy()
    # 注意：DataFrame.to_numpy() 在单块存储下返回**只读视图**，
    # 直接赋值会抛 "assignment destination is read-only"，必须显式拷贝。
    S = np.array(score.to_numpy(dtype=np.float64), copy=True)
    E = np.asarray(eligible.to_numpy(), dtype=bool)
    for name, X in extra.items():
        Xa = np.asarray(X.reindex(index=score.index, columns=score.columns)
                        .to_numpy(dtype=np.float64))
        for i in range(S.shape[0]):
            m = np.isfinite(S[i]) & E[i] & np.isfinite(Xa[i])
            if m.sum() < 50:
                continue
            x = Xa[i][m]
            x = x - x.mean()
            d = float(x @ x)
            if d <= 1e-12:
                continue
            y = S[i][m]
            y = y - y.mean()
            b = float(y @ x) / d
            S[i][m] = y - b * x
    return pd.DataFrame(S, index=score.index, columns=score.columns)


def main():
    print("=" * 100)
    print("多空组合改进实验")
    print("=" * 100)
    df = load_panel()
    P = Panels(df)
    del df
    eligible, _, _ = build_universe(P, CFG)
    need = sorted(set(v2.FACTOR_SETS["F17"] + v2.EXTRA_NAMES + v2.CONTROL_NAMES))
    v2.build_prep(P, eligible, need)

    # ---- 因子权重：从 CH-4 存活检验读取风险调整后 alpha ----
    # ★★ 前视控制（关键）★★
    # 第一版直接用 ch3_residual.json 里的 CH-4 alpha 作权重，但那是**全样本**
    # （2010-2026，含样本外）估计的 —— 用在样本外上就是前视，结果不可信。
    # 这里改为**只用样本内窗口 2010-2022** 重新估计每个因子的 CH-4 alpha，
    # 再据此加权，然后原样应用到样本外。这才是干净的样本外检验。
    names = v2.FACTOR_SETS["F17"]
    ch_raw = pd.read_pickle(os.path.join(DATA, "ch3.pkl")).shift(1)
    expo0 = pd.read_pickle(os.path.join(DATA, "ch3_expo.pkl"))
    C0 = pd.read_pickle(os.path.join(DATA, "ch3_prices.pkl"))
    cm = C0["close_me"]
    ret_fwd = cm.shift(-1) / cm - 1.0
    months = [d for d in ch_raw.index if d in expo0]
    is_months = [d for d in months if IS[0] <= str(d.date()) <= IS[1]]
    print(f"\n[1] 因子权重方案（CH-4 alpha 只用样本内 {len(is_months)} 个月估计）")
    print("    正在估计各因子的样本内 CH-4 alpha ...", flush=True)
    w_alpha = []
    alpha_detail = {}
    for n in names:
        fp = os.path.join(DATA, "prep", n + ".pkl")
        if not os.path.exists(fp):
            w_alpha.append(0.0)
            continue
        fac = pd.read_pickle(fp)
        rows = []
        for dt in is_months:
            e = expo0.get(dt)
            if e is None or dt not in fac.index:
                continue
            x = fac.loc[dt].reindex(e.index)
            r = ret_fwd.loc[dt].reindex(e.index)
            m = x.notna() & r.notna()
            if m.sum() < 80:
                continue
            xx, rr = x[m], r[m]
            hi, lo = xx >= xx.quantile(0.70), xx <= xx.quantile(0.30)
            if hi.sum() < 5 or lo.sum() < 5:
                continue
            rows.append((dt, float(rr[hi].mean() - rr[lo].mean())))
        del fac
        if len(rows) < 40:
            w_alpha.append(0.0)
            continue
        s = pd.Series(dict(rows)).sort_index()
        d = pd.concat({"y": s, "MKT": ch_raw["MKT"], "SMB": ch_raw["SMB"],
                       "VMG": ch_raw["VMG"], "PMO": ch_raw["PMO"]}, axis=1).dropna()
        if len(d) < 40:
            w_alpha.append(0.0)
            continue
        mm = sm.OLS(d["y"], sm.add_constant(d[["MKT", "SMB", "VMG", "PMO"]])
                    ).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        a = float(mm.params["const"])
        alpha_detail[n] = (round(a * 100, 4), round(float(mm.tvalues["const"]), 2))
        w_alpha.append(max(a, 0.0))
    w_alpha = np.array(w_alpha, dtype=float)
    if w_alpha.sum() <= 0:
        w_alpha = np.ones(len(names))
    w_alpha = w_alpha / w_alpha.sum()
    print("    样本内 CH-4 alpha（月%）与权重：")
    for n, w in sorted(zip(names, w_alpha), key=lambda x: -x[1]):
        ad = alpha_detail.get(n, (float("nan"), float("nan")))
        print(f"      {n:11s} α={ad[0]:>+7.3f}%  t={ad[1]:>+5.2f}   权重 {w:6.4f}")
    print(f"    等权：每个 1/17 = {1/17:.4f}")

    SCORES = {
        "等权": v2.composite(names),
        "CH4alpha加权": v2.composite(names, weights=w_alpha),
    }

    # ---- EP 分位（用于中和 VMG）----
    print("\n[2] 构造 EP 分位面板（用于中和 β_VMG）")
    expo = pd.read_pickle(os.path.join(DATA, "ch3_expo.pkl"))
    rebal = SCORES["等权"].index[::REC["rebalance_days"]]
    ep_rows = {}
    for dt in rebal:
        src = [d for d in expo if d <= dt]
        if src:
            ep_rows[dt] = expo[src[-1]]["EP"].rank(pct=True)
    ep_pct = pd.DataFrame(ep_rows).T.reindex(columns=P.symbols).reindex(SCORES["等权"].index)
    ep_pct = ep_pct.ffill()
    lnmv = np.log(P.float_mktcap.where(P.float_mktcap > 0))
    print(f"    EP 分位 {ep_pct.shape}")

    print("\n[3] 变体构造（EP 残差化 + 市值残差化）", flush=True)
    VARIANTS = {}
    for sn, sc in SCORES.items():
        VARIANTS[f"{sn}|原始"] = sc
        VARIANTS[f"{sn}|中和EP+市值"] = neutralise_score(
            sc, eligible, {"ep": ep_pct, "lnmv": lnmv})
    del lnmv

    # ---- 相位扫描 ----
    ch = pd.read_pickle(os.path.join(DATA, "ch3.pkl")).shift(1)
    PHASES = list(range(0, 21, 3))       # 7 个相位
    print(f"\n[4] 相位扫描（{len(PHASES)} 个相位） × {len(VARIANTS)} 变体 × 2 口径", flush=True)
    results = {}
    for vn, sc in VARIANTS.items():
        for mode in ["多头", "多空"]:
            iss, ooss, dd_o, cg_o, wst = [], [], [], [], []
            for off in PHASES:
                s2 = sc.iloc[off:] if off else sc
                el2 = eligible.reindex(s2.index)
                Wl = v2.make_weights(s2, el2, side="long", **REC)
                if mode == "多空":
                    Ws = v2.make_weights(s2, el2, side="short", **REC)
                    Wl = Wl.sub(Ws.reindex(columns=Wl.columns, fill_value=0.0),
                                fill_value=0.0)
                    del Ws
                r = backtest(P, Wl, CFG, f"{vn}|{mode}|{off}")["net"]
                del Wl
                i = met(r[(r.index >= IS[0]) & (r.index <= IS[1])])
                o = met(r[(r.index >= OOS[0]) & (r.index <= OOS[1])])
                if i:
                    iss.append(i["Sharpe"])
                if o:
                    ooss.append(o["Sharpe"])
                    dd_o.append(o["MaxDD"])
                    cg_o.append(o["CAGR"])
                wst.append(r)
                del r
            if not ooss:
                continue
            # CH-4 回归用相位 0 的序列
            r0 = wst[0]
            mm = r0.resample("ME").apply(lambda x: (1 + x).prod() - 1)
            sub = mm[(mm.index >= OOS[0]) & (mm.index <= OOS[1])]
            d = pd.concat({"y": sub, "MKT": ch["MKT"], "SMB": ch["SMB"],
                           "VMG": ch["VMG"], "PMO": ch["PMO"]}, axis=1).dropna()
            ch4 = {}
            if len(d) > 24:
                m = sm.OLS(d["y"], sm.add_constant(d[["MKT", "SMB", "VMG", "PMO"]])
                           ).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
                ch4 = {"alpha_a_pct": round(((1 + float(m.params["const"])) ** 12 - 1) * 100, 2),
                       "t": round(float(m.tvalues["const"]), 2),
                       "betas": {c: round(float(m.params[c]), 3)
                                 for c in ["MKT", "SMB", "VMG", "PMO"]},
                       "r2": round(float(m.rsquared), 3)}
            results[f"{vn}|{mode}"] = {
                "IS_sharpe_mean": round(float(np.mean(iss)), 4),
                "IS_sharpe_sd": round(float(np.std(iss)), 4),
                "OOS_sharpe_mean": round(float(np.mean(ooss)), 4),
                "OOS_sharpe_sd": round(float(np.std(ooss)), 4),
                "OOS_sharpe_min": round(float(np.min(ooss)), 4),
                "OOS_sharpe_max": round(float(np.max(ooss)), 4),
                "OOS_cagr_mean": round(float(np.mean(cg_o)), 4),
                "OOS_maxdd_mean": round(float(np.mean(dd_o)), 4),
                "OOS_calmar_mean": round(float(np.mean(cg_o) / abs(np.mean(dd_o))), 3),
                "ch4_oos": ch4, "n_phases": len(ooss)}
            t = results[f"{vn}|{mode}"]
            print(f"    {vn:24s} {mode}  IS {t['IS_sharpe_mean']:.3f} | "
                  f"OOS {t['OOS_sharpe_mean']:.3f}±{t['OOS_sharpe_sd']:.3f} "
                  f"[{t['OOS_sharpe_min']:.3f},{t['OOS_sharpe_max']:.3f}]  "
                  f"年化 {t['OOS_cagr_mean']:>7.2%}  回撤 {t['OOS_maxdd_mean']:>7.2%}  "
                  f"Calmar {t['OOS_calmar_mean']:.2f}  "
                  f"CH4α {ch4.get('alpha_a_pct', float('nan')):>6.2f}% "
                  f"(t={ch4.get('t', float('nan')):>4.2f})  "
                  f"β_VMG {ch4.get('betas', {}).get('VMG', float('nan')):>6.3f}", flush=True)
            del wst

    # ---- 按样本内挑选 ----
    print("\n[5] 按样本内夏普选择（样本外只作报告）")
    oosonly = {k: v for k, v in results.items() if v["n_phases"] >= 5}
    best_is = max(oosonly, key=lambda k: oosonly[k]["IS_sharpe_mean"])
    b = oosonly[best_is]
    print(f"    样本内最优 → {best_is}   IS {b['IS_sharpe_mean']:.3f}  →  "
          f"OOS {b['OOS_sharpe_mean']:.3f}  回撤 {b['OOS_maxdd_mean']:.2%}  "
          f"Calmar {b['OOS_calmar_mean']:.2f}")
    multi = {k: v for k, v in oosonly.items() if k.endswith("多空")}
    if multi:
        best_ls = max(multi, key=lambda k: k)
        print(f"    多空候选中 OOS 夏普最高（仅报告，不据此选择）："
              f"{max(multi, key=lambda k: multi[k]['OOS_sharpe_mean'])} "
              f"{multi[max(multi, key=lambda k: multi[k]['OOS_sharpe_mean'])]['OOS_sharpe_mean']:.3f}")

    with open(os.path.join(OUT, "ls_improve.json"), "w", encoding="utf-8") as fh:
        json.dump({"results": results, "alpha_weights": dict(zip(names, w_alpha.round(4))),
                   "is_selected": best_is}, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n[已保存] {os.path.join(OUT,'ls_improve.json')}")


if __name__ == "__main__":
    main()
