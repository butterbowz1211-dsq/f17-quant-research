# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
fm_weights.py — 用 Fama-MacBeth 多变量系数合成因子（此前从未试过）
================================================================================
为什么这是最后一个有理论依据的合成方法
--------------------------------------
此前试过的因子合成方式：
  · **等权**（在用）—— 最稳健，但完全忽略因子的信息量与相关性
  · **CH-4 alpha 加权** —— 只在空头腿有效（已验证）
  · **加/减因子** —— 均无增量（已验证）
  · **组合层面约束** —— 均失败（已验证）

**从未试过**：Fama-MacBeth (1973) 的**多变量**截面回归系数。
我一直只用 FM 做**逐因子**检验（单变量），却没用它的**联合**系数来合成。
这正是商业 alpha 模型的标准做法：
    每期截面回归   r_{i,t+1} = Σ_k β_{k,t} · f_{k,i,t} + ε
    取时序平均      ω_k = mean_t(β_{k,t})
    ω 即因子权重（自动考虑了因子的信息量与彼此的相关性 —— 多重共线性的因子
    会被自动压低权重，这正是等权做不到的）。

方法学纪律
----------
· 权重**只用样本内 2010-2022** 的截面回归估计，样本外原样应用；
· 因子值已是「去极值 → 行业+市值中性化 → z-score」后的标准值；
· 加入岭正则（ridge）以应对因子间多重共线性： ω = (X'X + λI)^{-1}X'y；
· 全部方案做相位平均，并报告样本内外方向是否一致。
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

from ashare_factor_strategy import load_panel, Panels, build_universe, backtest, CFG
import strategy_v2 as v2

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
PPY, RF = 252, 0.02
IS = ("2010-01-01", "2022-12-31")
OOS = ("2023-01-01", "2026-12-31")
REC = dict(n=50, weighting="equal", buffer=0.6, rebalance_days=21, max_weight=0.03)
PHASES = list(range(0, 21, 3))


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
            "Calmar": float(cagr / abs(dd)) if dd < 0 else None}


def main():
    print("=" * 104)
    print("Fama-MacBeth 多变量系数合成因子")
    print("=" * 104)
    df = load_panel()
    P = Panels(df)
    del df
    eligible, _, _ = build_universe(P, CFG)
    names = v2.FACTOR_SETS["F17"]
    need = sorted(set(names + v2.EXTRA_NAMES + v2.CONTROL_NAMES))
    v2.build_prep(P, eligible, need)

    # ---- 按**月末**取因子面板与前瞻收益 ----
    # ★ 不能拿「每 21 个交易日」去和 ret_fwd 取交集：ret_fwd 的索引是月末日期，
    #   而 21 日网格极少落在月末，交集只剩 11 个点（首次运行即踩到这个坑，
    #   FM 权重只用了 6 期估计，完全不可用）。直接用月末即可，也更符合 FM 的月度口径。
    C = pd.read_pickle(os.path.join(DATA, "ch3_prices.pkl"))
    cm = C["close_me"]
    ret_fwd = cm.shift(-1) / cm - 1.0
    months = [d for d in cm.index if d in P.dates and d in eligible.index]
    print(f"\n[1] 月末调仓日 {len(months)} 个  {months[0].date()} ~ {months[-1].date()}")
    assert len(months) > 100, f"沿用日过少({len(months)})，检查日期对齐"

    # 预先取每个因子的值并向量化：一次读取一个因子，按调仓日抽取
    print("    抽取因子值 ...", flush=True)
    Xall = {}
    for n in names:
        f = pd.read_pickle(os.path.join(DATA, "prep", n + ".pkl"))
        # 因子是日频宽表，取每月末（月末若非交易日则向前取最近交易日，与该因子自身的
        # 有效值对齐；用 reindex(method="ffill") 保证不会取到未来值）
        Xall[n] = f.reindex(index=cm.index, method="ffill").reindex(
            index=months).to_numpy(dtype=np.float64)
        del f
    Y = ret_fwd.reindex(index=months).to_numpy(dtype=np.float64)
    EL = eligible.reindex(index=months).to_numpy()
    print(f"    X {Xall[names[0]].shape}  Y {Y.shape}")

    is_idx = [i for i, d in enumerate(months) if IS[0] <= str(d.date()) <= IS[1]]
    print(f"    样本内调仓日 {len(is_idx)} 个")

    # ---- 逐期截面回归，取时序平均系数 ----
    def fm_weights(lam=0.0, min_n=200):
        B = []
        for i in is_idx:
            y = Y[i]
            cols = []
            mask = EL[i] & np.isfinite(y)
            Xm = []
            for n in names:
                x = Xall[n][i]
                mask = mask & np.isfinite(x)
                Xm.append(x)
            if mask.sum() < min_n:
                continue
            X = np.column_stack([x[mask] for x in Xm])
            yy = y[mask]
            # 标准化列，使岭惩罚尺度可比
            X = (X - X.mean(0)) / (X.std(0) + 1e-12)
            yy = (yy - yy.mean()) / (yy.std() + 1e-12)
            if lam > 0:
                A = X.T @ X + lam * np.eye(X.shape[1])
                b = np.linalg.solve(A, X.T @ yy)
            else:
                b, *_ = np.linalg.lstsq(X, yy, rcond=None)
            B.append(b)
        if not B:
            return None, 0, None
        B = np.array(B)
        w = B.mean(axis=0)
        t = w / (B.std(axis=0) / np.sqrt(len(B)) + 1e-12)
        return w, len(B), t

    print("\n[2] 估计 Fama-MacBeth 权重（只用样本内）")
    variants = {}
    for lam in [0.0, 0.05, 0.2, 1.0]:
        w, nobs, t = fm_weights(lam)
        if w is None:
            continue
        variants[f"FM岭λ={lam}"] = (w, t, nobs)
        print(f"\n    λ={lam}  (n={nobs} 期)")
        order = np.argsort(-np.abs(w))
        for j in order[:10]:
            print(f"      {names[j]:11s} ω={w[j]:>+8.4f}  t={t[j]:>+6.2f}")
        nz = int((np.abs(w) > 1e-6).sum())
        print(f"      非零权重因子数 {nz}/{len(names)}   |ω| 和 {np.abs(w).sum():.3f}")

    # ---- 组合对比 ----
    print("\n[3] 相位扫描对比（7 个相位）", flush=True)
    SETS = {"等权（基准）": None}
    for k, (w, t, nobs) in variants.items():
        SETS[k] = np.abs(w)
    res = {}
    ch = pd.read_pickle(os.path.join(DATA, "ch3.pkl")).shift(1)
    for sn, wv in SETS.items():
        sc = v2.composite(names) if wv is None else v2.composite(names, weights=wv)
        for mode in ["多头", "多空"]:
            iss, ooss, dds, cgs, keep0 = [], [], [], [], None
            for off in PHASES:
                s2 = sc.iloc[off:] if off else sc
                el2 = eligible.reindex(s2.index)
                Wl = v2.make_weights(s2, el2, side="long", **REC)
                if mode == "多空":
                    Ws = v2.make_weights(s2, el2, side="short", **REC)
                    Wl = Wl.sub(Ws.reindex(columns=Wl.columns, fill_value=0.0),
                                fill_value=0.0)
                    del Ws
                r = backtest(P, Wl, CFG, f"{sn}|{mode}|{off}")["net"]
                del Wl
                i = met(r[(r.index >= IS[0]) & (r.index <= IS[1])])
                o = met(r[(r.index >= OOS[0]) & (r.index <= OOS[1])])
                if i: iss.append(i["Sharpe"])
                if o: ooss.append(o["Sharpe"]); dds.append(o["MaxDD"]); cgs.append(o["CAGR"])
                if off == 0:
                    keep0 = r
                del r
            if not ooss:
                continue
            mm = keep0.resample("ME").apply(lambda x: (1 + x).prod() - 1)
            sub = mm[(mm.index >= OOS[0]) & (mm.index <= OOS[1])]
            d = pd.concat({"y": sub, "MKT": ch["MKT"], "SMB": ch["SMB"],
                           "VMG": ch["VMG"], "PMO": ch["PMO"]}, axis=1).dropna()
            ch4 = {}
            if len(d) > 24:
                m = sm.OLS(d["y"], sm.add_constant(d[["MKT", "SMB", "VMG", "PMO"]])
                           ).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
                ch4 = {"alpha_a_pct": round(((1 + float(m.params["const"])) ** 12 - 1) * 100, 2),
                       "t": round(float(m.tvalues["const"]), 2)}
            res[f"{sn}|{mode}"] = {
                "IS_sharpe_mean": round(float(np.mean(iss)), 4),
                "OOS_sharpe_mean": round(float(np.mean(ooss)), 4),
                "OOS_sharpe_sd": round(float(np.std(ooss)), 4),
                "OOS_sharpe_min": round(float(np.min(ooss)), 4),
                "OOS_sharpe_max": round(float(np.max(ooss)), 4),
                "OOS_cagr_mean": round(float(np.mean(cgs)), 4),
                "OOS_maxdd_mean": round(float(np.mean(dds)), 4),
                "OOS_calmar_mean": round(float(np.mean(cgs) / abs(np.mean(dds))), 3),
                "ch4_oos": ch4}
            t_ = res[f"{sn}|{mode}"]
            print(f"    {sn:16s} {mode}  IS {t_['IS_sharpe_mean']:.3f} | "
                  f"OOS {t_['OOS_sharpe_mean']:.3f}±{t_['OOS_sharpe_sd']:.3f} "
                  f"[{t_['OOS_sharpe_min']:.3f},{t_['OOS_sharpe_max']:.3f}]  "
                  f"年化 {t_['OOS_cagr_mean']:>7.2%}  回撤 {t_['OOS_maxdd_mean']:>7.2%}  "
                  f"Calmar {t_['OOS_calmar_mean']:.2f}  "
                  f"CH4α {ch4.get('alpha_a_pct', float('nan')):>6.2f}%"
                  f"(t={ch4.get('t', float('nan')):>4.2f})", flush=True)

    out = {"results": res,
           "weights": {k: dict(zip(names, np.round(v[0], 5).tolist())) for k, v in variants.items()},
           "tstats": {k: dict(zip(names, np.round(v[1], 3).tolist())) for k, v in variants.items()}}
    with open(os.path.join(OUT, "fm_weights.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n[已保存] {os.path.join(OUT,'fm_weights.json')}")


if __name__ == "__main__":
    main()
