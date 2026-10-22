# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
strategy_v2.py — 改进实验平台
================================================================================
本脚本承载由论文阅读导出的全部可落地改进，并在**统一、无未来函数**的口径下
横向比较。所有实验共用同一份股票池、同一份回测引擎、同一套成本假设。

改进来源与依据
--------------
1. 因子集精简（多重检验 + 样本外衰减）
   Di 与本文诊断：17 个入选因子里 LEAD_LAG / BETA / HIGH52 / RES_MOM 的样本外
   IC 已归零或反向。Harvey, Liu & Zhu (2016, RFS) 指出在多重检验下应提高 t 门槛，
   减少试验次数本身就能降低 Deflated Sharpe 的门槛 SR0。

2. 组合构建宽度（本次核心假设）
   现行 top_pct=10% 且 n_long=50，在日均 2500+ 只的池子里实际只取前 ~1.5% 等权。
   → 特质波动被放大（全样本 σ_ε=9.26%），且截断了排序信息。
   Grinold-Kahn 基本法则 IR ≈ IC·√BR：提高广度 BR 是提升 IR 最直接的途径。

3. 换手控制
   实测年化换手 1771%，成本拖累 2.66%/年，吃掉 0.118 夏普。
   → 缓冲区（no-trade band）+ 信号平滑。

4. Daniel & Moskowitz (2016) 的诚实结论
   panic 状态叠加层在本策略无效（γ_int 符号相反，t=0.90），故**不纳入**。
   但沿用其方法学纪律：报告分样本、做 spanning 检验。

5. Stambaugh-Yu-Yuan (2012) 的情绪条件化
   实测仅边际有效（唯一显著变量样本外不复现），作为**可选叠加层**保留，不进入主口径。

6. Grinblatt-Han (2005) 的资本利得突出量（CGO）
   文献周频 FM 系数 0.0040 (t=7.79)，且能吸收中期动量。由日频价与换手率完全可算，
   是本轮唯一新增的因子。

用法
----
  python strategy_v2.py --stage prep          # 构建预处理因子缓存（约 10-20 分钟）
  python strategy_v2.py --stage grid          # 跑组合构建网格
  python strategy_v2.py --stage final         # 跑最终方案 + 显著性
"""

import os
import sys
import json
import time
import pickle
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ashare_factor_strategy import (          # noqa: E402
    load_panel, Panels, build_universe, prepare_factor, backtest, CFG, F32,
    probabilistic_sharpe, deflated_sharpe, newey_west_tstat,
)
from factors_v2 import EXTRA_FACTORS            # noqa: E402

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
PREP = os.path.join(DATA, "prep")
os.makedirs(PREP, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

PPY = 252
RF = 0.02
IS = ("2010-01-01", "2022-12-31")
OOS = ("2023-01-01", "2026-12-31")

# 现行 17 因子
F17 = ["TURN", "MAX5", "INTRADAY", "ILLIQ", "COMMOVE", "TURN_SHOCK", "REV_1W",
       "ISKEW", "IVOL", "OVERNIGHT", "LEAD_LAG", "CO_SKEW", "BETA", "REV_LT",
       "SIZE", "HIGH52", "RES_MOM"]
# 剔除样本外已死/反向者
F13 = [f for f in F17 if f not in ("LEAD_LAG", "BETA", "HIGH52", "RES_MOM")]
# 只留 ICIR > 1.6 且有明确理论依据者
F8 = ["TURN", "MAX5", "INTRADAY", "ILLIQ", "COMMOVE", "TURN_SHOCK", "REV_1W", "ISKEW"]
# 只留最强 5 个
F5 = ["TURN", "MAX5", "INTRADAY", "ILLIQ", "COMMOVE"]
# 叠加 Grinblatt-Han (2005) 的资本利得突出量
#   CGO_R 为 A 股适用方向（符号修正，见 factors_v2.py 的论证）
F13C = F13 + ["CGO"]
F8C = F8 + ["CGO"]
F13CR = F13 + ["CGO_R"]
F8CR = F8 + ["CGO_R"]
# 叠加中国文献验证的换手率族因子（JSZ 2021 的 ABN_TURN；HQZ 2023 的 vturn1-1）
F19 = F17 + ["ABN_TURN", "VTURN"]
FACTOR_SETS = {"F17": F17, "F13": F13, "F8": F8, "F5": F5,
               "F13C": F13C, "F8C": F8C, "F13CR": F13CR, "F8CR": F8CR,
               "F19": F19}
EXTRA_NAMES = ["CGO", "CGO_R", "CGO_V", "CGO_M", "ABN_TURN", "VTURN"]
# 对照用因子（需一并预处理以便比较 IC）
CONTROL_NAMES = ["MOM_12_1", "HIGH52", "REV_1M", "PV_CORR", "ID_FROG",
                 "DIVERGENCE", "IND_MOM", "SPILLOVER"]


# ==============================================================================
def build_prep(P, eligible, names, force=False):
    """逐个因子预处理（掩码→去极值→行业市值中性化→标准化）并落盘。

    逐个处理而非一次性全载：中性化后每个宽表 4065×5221 float32 ≈ 81MB，
    24 个同时驻留会逼近 2GB，触发 Windows 换页。

    factors_v2.EXTRA_FACTORS 里的因子（CGO 等）没有预生成的原始缓存，
    直接由面板现算。
    """
    fdir = os.path.join(DATA, "factors")
    todo = [n for n in names if force or not os.path.exists(os.path.join(PREP, n + ".pkl"))]
    if not todo:
        print(f"  [缓存] {len(names)} 个预处理因子已全部就绪")
        return
    print(f"  需处理 {len(todo)} 个: {todo}", flush=True)
    for i, n in enumerate(todo, 1):
        t0 = time.time()
        if n in EXTRA_FACTORS:
            raw = EXTRA_FACTORS[n](P)
        else:
            raw = pd.read_pickle(os.path.join(fdir, n + ".pkl"))
        x = prepare_factor(raw, P, eligible, CFG)
        x.to_pickle(os.path.join(PREP, n + ".pkl"))
        del raw, x
        print(f"    [{i}/{len(todo)}] {n:12s} {time.time()-t0:.1f}s", flush=True)


def rank_ic_report(P, eligible, name, fwd_days=22, step=21):
    """单因子 RankIC 快报（样本内 / 样本外分开），用于新增因子的方向校验。"""
    x = pd.read_pickle(os.path.join(PREP, name + ".pkl"))
    fwd = P.close.shift(-fwd_days) / P.close - 1.0
    recs = []
    for dt in x.index[::step]:
        a = x.loc[dt].where(eligible.loc[dt])
        b = fwd.loc[dt].where(eligible.loc[dt])
        m = a.notna() & b.notna()
        if int(m.sum()) < 50:
            continue
        av, bv = a[m], b[m]
        recs.append((dt, av.rank().corr(bv.rank())))
    if not recs:
        return {}
    s = pd.Series(dict(recs))
    out = {"n_periods": int(len(s))}
    for lab, (st, en) in [("IS", IS), ("OOS", OOS)]:
        sub = s[(s.index >= st) & (s.index <= en)]
        if len(sub) < 10:
            continue
        ic = float(sub.mean())
        t = ic / sub.std() * np.sqrt(len(sub)) if sub.std() > 0 else np.nan
        out[lab] = {"IC": round(ic, 5), "t": round(float(t), 2),
                    "ICIR": round(float(ic / sub.std()), 3) if sub.std() > 0 else None,
                    "pos_rate": round(float((sub > 0).mean()), 3)}
    return out



def composite(names, weights=None, fill_na=True):
    """多因子合成：对标准ized残差做加权平均。

    等权是默认 —— 权重估计误差往往大于其信息增益（见 combine_equal 注释）。
    weights 给定时按权重加权。
    """
    acc = cnt = None
    for i, n in enumerate(names):
        w = 1.0 if weights is None else float(weights[i])
        v = pd.read_pickle(os.path.join(PREP, n + ".pkl"))
        valid = v.notna().astype(F32) * w
        vv = (v.fillna(0) * w).astype(F32)
        acc = vv if acc is None else acc.add(vv, fill_value=0)
        cnt = valid if cnt is None else cnt.add(valid, fill_value=0)
        del v, vv, valid
    return (acc / cnt.replace(0, np.nan)).astype(F32)


# ==============================================================================
# 组合构建
# ==============================================================================
def make_weights(score, eligible, n=50, weighting="equal", max_weight=0.05,
                 rebalance_days=21, buffer=0.0, smooth=0.0, side="long", mix=0.0,
                 pool=None):
    """把因子得分转成目标权重。

    参数
    ----
    n            最终持仓只数
    pool         候选池大小（默认 = n）。设成大于 n 时，先取得分前 pool 名作为候选，
                 在池内按得分加权，再取权重最大的 n 只。
                 这样保留了「向极端倾斜」的信息，又比直接取前 n 名平滑，
                 实测能同时改善样本外夏普与回撤。
    weighting    'equal' 等权 | 'score' 向高分端倾斜
    mix          倾斜强度 0~1：w = (1-mix)·等权 + mix·得分权重
    max_weight   单票上限（_apply_cap 用不动点迭代保证真正生效）
    buffer       缓冲区比例：老持仓只要仍在候选前 n*(1+buffer) 内就保留
    smooth       信号平滑系数 α
    side         'long' 取高分 | 'short' 取低分
    """
    if smooth > 0:
        score = score.ewm(alpha=smooth, adjust=False).mean()

    pool = int(pool or n)
    work = score if side == "long" else -score

    rebal = work.index[::rebalance_days]
    prev = None
    rows = {}
    for dt in rebal:
        s = work.loc[dt].where(eligible.loc[dt]).dropna()
        if len(s) < 20:
            continue
        n_eff = min(n, len(s))
        p_eff = min(max(pool, n_eff), len(s))

        if buffer > 0 and prev is not None and len(prev) > 0:
            keep_band = set(s.nlargest(min(int(p_eff * (1 + buffer)), len(s))).index)
            kept = [c for c in prev if c in keep_band]
            room = p_eff - len(kept)
            newp = [c for c in s.nlargest(p_eff + len(kept)).index if c not in kept]
            pick_idx = kept + newp[:max(room, 0)]
            cand = s.reindex(pick_idx).dropna()
            if len(cand) < max(int(p_eff * 0.5), 10):
                cand = s.nlargest(p_eff)
        else:
            cand = s.nlargest(p_eff)

        w = _weight_within(cand, weighting, mix)
        if len(w) > n_eff:
            w = w.nlargest(n_eff)          # 池内加权后再截断到 n 只
        w = _apply_cap(w, max_weight)
        rows[dt] = w
        prev = list(cand.index)

    W = pd.DataFrame(rows).T.fillna(0.0)
    return W.reindex(columns=score.columns, fill_value=0.0)


def _apply_cap(w, cap, iters=200, tol=1e-4):
    """真正生效的单票权重上限（不动点迭代）。

    朴素写法 `w.clip(upper=cap); w /= w.sum()` 是错的：clip 之后总和小于 1，
    再除以这个更小的数会把被截断的权重又推回上限之上（实测 cap=5% 时
    单票权重可达 10.4%）。这里改为迭代到不动点：截断 → 归一化 → 再截断，
    直到所有未固定的权重都不超过上限。
    """
    w = pd.Series(w).astype(float)
    if not np.isfinite(w.sum()) or w.sum() <= 0:
        return pd.Series(1.0 / len(w), index=w.index)
    if cap >= 1.0:
        return w / w.sum()
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


def _weight_within(pick, weighting, mix):
    """在已选中的股票集合内部分配权重。

    得分权重 s = (score − 集合下四分位)，截断到非负后除以均值 —— 归一化使得
    mix 在 0~1 之间有稳定的经济含义（而不受 z 分量纲影响）。
    """
    eq = pd.Series(1.0, index=pick.index)
    if weighting == "equal":
        return eq
    if mix <= 0.0:
        return eq
    thr = pick.quantile(0.25)
    s = (pick - thr).clip(lower=0.0)
    m = float(s.mean())
    if not np.isfinite(m) or m <= 1e-12:
        return eq
    s = s / m
    return ((1.0 - mix) * eq + mix * s).clip(lower=1e-6)


def make_weights_beta(score, eligible, beta_raw, n=50, pool=200, beta_target=None,
                      rebalance_days=21, max_weight=0.03):
    """带 β 目标约束的组合构建。

    动机
    ----
    risk_attribution.py 已证明：本策略多头组合 β≈0.93、与市场相关 0.91，
    −25.49% 的样本外回撤里 −15.60pp 来自市场 β，−9.89pp 来自自身的小盘/非流动性暴露。
    在不能做空股指期货（贴水成本）的前提下，**唯一能直接压低回撤的手段是把
    组合的事前 β 压到目标值**。

    方法
    ----
    在得分前 pool 名的候选池内，对权重施加指数倾斜并解出倾斜系数：
        w_i ∝ s_i · exp(−λ·β_i)          s.t.  Σ w_i·β_i = β_target
    λ 用二分法求解（组合 β 关于 λ 单调递减）。pool 设得比 n 大，才有腾挪空间；
    最后只保留权重最大的 n 只，从而在不牺牲集中度的前提下降低 β。

    为什么不用「持有现金」降 β：那只是等比例降杠杆，夏普与 Calmar 不变。
    本方法是在**同一批高 alpha 股票内部**换成低 β 的标的，理论上可同时改善夏普与回撤。
    """
    cols = score.columns
    rebal = score.index[::rebalance_days]
    rows = {}
    for dt in rebal:
        s = score.loc[dt].where(eligible.loc[dt]).dropna()
        if len(s) < 20:
            continue
        b = beta_raw.loc[dt].reindex(s.index) if dt in beta_raw.index else None
        if b is None:
            continue
        ok = b.notna()
        s, b = s[ok], b[ok]
        if len(s) < 20:
            continue
        cand = s.nlargest(min(pool, len(s)))
        bc = b.reindex(cand.index).fillna(float(b.median()))
        thr = cand.quantile(0.25)
        base = (cand - thr).clip(lower=0.0)
        if float(base.sum()) <= 0:
            base = pd.Series(1.0, index=cand.index)
        base = base / base.sum()
        bs = base.to_numpy(dtype=float)
        bb = bc.to_numpy(dtype=float)

        if beta_target is None:
            wv = bs
        else:
            def pbeta(lam):
                v = bs * np.exp(-lam * bb)
                t = v.sum()
                return float(v @ bb / t) if t > 0 else np.nan

            if not np.isfinite(pbeta(0.0)) or pbeta(0.0) <= beta_target:
                lam = 0.0
            else:
                lo, hi = 0.0, 1.0
                while pbeta(hi) > beta_target and hi < 1e4:
                    hi *= 2.0
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    if pbeta(mid) > beta_target:
                        lo = mid
                    else:
                        hi = mid
                lam = 0.5 * (lo + hi)
            v = bs * np.exp(-lam * bb)
            wv = v / v.sum()
        w = pd.Series(wv, index=cand.index).nlargest(min(n, len(cand)))
        w = _apply_cap(w, max_weight)
        rows[dt] = w

    W = pd.DataFrame(rows).T.fillna(0.0)
    return W.reindex(columns=cols, fill_value=0.0)
    """等权全市场基准"""
    e = eligible.astype(F32)
    e = e.div(e.sum(axis=1).replace(0, np.nan), axis=0)
    return e.fillna(0.0)


# ==============================================================================
# 评估
# ==============================================================================
def universe_weights(eligible):
    """等权全市场基准"""
    e = eligible.astype(F32)
    e = e.div(e.sum(axis=1).replace(0, np.nan), axis=0)
    return e.fillna(0.0)


def stats(net, gross=None, turnover=None, label=""):
    r = pd.Series(net).dropna()
    if len(r) < 30:
        return {"name": label}
    eq = (1 + r).cumprod()
    yrs = len(r) / PPY
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(PPY)
    dd = eq / eq.cummax() - 1
    mdd = float(dd.min())
    o = {"name": label, "CAGR": round(float(cagr), 4), "AnnVol": round(float(vol), 4),
         "Sharpe": round(float((r.mean() * PPY - RF) / vol), 4) if vol > 0 else None,
         "MaxDD": round(mdd, 4),
         "Calmar": round(float(cagr / abs(mdd)), 3) if mdd < 0 else None,
         "WinRate": round(float((r > 0).mean()), 4), "n": int(len(r))}
    if gross is not None:
        g = pd.Series(gross).dropna()
        gv = g.std() * np.sqrt(PPY)
        o["gross_Sharpe"] = round(float((g.mean() * PPY - RF) / gv), 4) if gv > 0 else None
    if turnover is not None:
        o["ann_turnover"] = round(float(pd.Series(turnover).dropna().mean() * PPY), 4)
    return o


def slice_stats(net, gross=None, turnover=None, label=""):
    out = {}
    for lab, (s, e) in [("IS", IS), ("OOS", OOS)]:
        m = (net.index >= s) & (net.index <= e)
        out[lab] = stats(net[m],
                         gross[m] if gross is not None else None,
                         turnover[m] if turnover is not None else None,
                         f"{label}|{lab}")
    return out


def run_variant(P, eligible, score, label, bt_kwargs, verbose=True):
    t0 = time.time()
    W = make_weights(score, eligible, **bt_kwargs)
    res = backtest(P, W, CFG, label)
    net, gross, tv = res["net"], res["gross"], res["turnover"]
    parts = slice_stats(net, gross, tv, label)
    if verbose:
        for lab in ["IS", "OOS"]:
            p = parts[lab]
            print(f"    {label:42s} {lab:3s}  年化 {p['CAGR']:>7.2%}  波动 {p['AnnVol']:>6.2%}  "
                  f"夏普 {p['Sharpe']:>6.3f}  回撤 {p['MaxDD']:>8.2%}  "
                  f"Calmar {str(p['Calmar']):>5s}  换手 {p.get('ann_turnover', float('nan')):>7.1%}"
                  f"  毛夏普 {p.get('gross_Sharpe', float('nan')):>5.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return parts, res


# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="grid",
                    choices=["prep", "cgo", "grid", "tune", "beta", "phase", "final"])
    ap.add_argument("--force-prep", action="store_true")
    args = ap.parse_args()

    t_all = time.time()
    print("=" * 96)
    print(f"strategy_v2  阶段={args.stage}")
    print("=" * 96)

    print("\n[1] 加载面板与股票池", flush=True)
    df = load_panel()
    P = Panels(df)
    del df
    eligible, limit_up, limit_down = build_universe(P, CFG)

    need = sorted(set(sum(FACTOR_SETS.values(), []) + EXTRA_NAMES + CONTROL_NAMES))
    print(f"\n[2] 预处理因子缓存（{len(need)} 个）", flush=True)
    build_prep(P, eligible, need, force=args.force_prep)

    if args.stage == "prep":
        print(f"\n完成，用时 {time.time()-t_all:.0f}s")
        return

    if args.stage == "cgo":
        print("\n[3] 新增因子 CGO 校验（含分布诊断，防止实现错误）", flush=True)
        from factors_v2 import cgo_diagnostics
        diag = cgo_diagnostics(P)
        print("    CGO 分布诊断（对照 GH 原文美股：mean 0.056 / median 0.106 / "
              "sd 0.251 / p10 −0.281 / p90 +0.312）")
        print(f"      有效样本 {diag['n_valid']:,}   mean {diag['mean']:+.4f}  "
              f"median {diag['median']:+.4f}  sd {diag['sd']:.4f}")
        print(f"      p1 {diag['p1']:+.4f}  p10 {diag['p10']:+.4f}  p25 {diag['p25']:+.4f}  "
              f"p75 {diag['p75']:+.4f}  p90 {diag['p90']:+.4f}  p99 {diag['p99']:+.4f}")
        print(f"      g>0 占比 {diag['frac_g_gt_0']:.3f}   g<−1 占比 {diag['frac_g_lt_neg1']:.5f}   "
              f"g>1 占比 {diag['frac_g_gt_1']:.5f}")
        print(f"      满足 {52} 周历史的股票数 {diag['n_stocks_ge52w']}", flush=True)

        cgo_ic = {"diagnostics": diag}
        for n in EXTRA_NAMES:
            r = rank_ic_report(P, eligible, n)
            cgo_ic[n] = r
            isr, oosr = r.get("IS", {}), r.get("OOS", {})
            print(f"    {n:8s} 样本内 IC={isr.get('IC')} (t={isr.get('t')}, "
                  f"ICIR={isr.get('ICIR')}, 正IC占比={isr.get('pos_rate')})  |  "
                  f"样本外 IC={oosr.get('IC')} (t={oosr.get('t')})", flush=True)
        print("\n    对照：现有因子样本内/样本外 IC")
        for n in ["TURN", "MAX5", "INTRADAY", "ILLIQ", "COMMOVE", "REV_1W",
                  "MOM_12_1", "HIGH52"]:
            r = rank_ic_report(P, eligible, n)
            cgo_ic[n] = r
            isr, oosr = r.get("IS", {}), r.get("OOS", {})
            print(f"    {n:8s} 样本内 IC={isr.get('IC')} | 样本外 IC={oosr.get('IC')}", flush=True)
        with open(os.path.join(OUT, "cgo_ic.json"), "w", encoding="utf-8") as fh:
            json.dump(cgo_ic, fh, ensure_ascii=False, indent=2, default=str)

        print("\n[4] 加入 CGO 前后的多空对比（n=150 等权，月度调仓）", flush=True)
        cmp_res = {}
        for sname in ["F13", "F13C", "F13CR", "F8", "F8C", "F8CR"]:
            sc = composite(FACTOR_SETS[sname])
            cmp_res[sname] = run_variant(P, eligible, sc, f"{sname} n=150", 
                                         dict(n=150, weighting="equal", rebalance_days=21))[0]
        with open(os.path.join(OUT, "v2_cgo.json"), "w", encoding="utf-8") as fh:
            json.dump(cmp_res, fh, ensure_ascii=False, indent=2, default=str)
        print(f"\n完成，用时 {time.time()-t_all:.0f}s")
        return

    print(f"\n[3] 合成因子得分", flush=True)
    scores = {}
    for sname, fl in FACTOR_SETS.items():
        t0 = time.time()
        scores[sname] = composite(fl)
        print(f"    {sname:5s} ({len(fl)} 因子) 完成 {time.time()-t0:.0f}s", flush=True)

    # 基准
    print(f"\n[4] 基准：等权全市场", flush=True)
    Wb = universe_weights(eligible)
    rb = backtest(P, Wb, CFG, "等权全市场")
    for lab in ["IS", "OOS"]:
        m = (rb["net"].index >= dict(IS=IS, OOS=OOS)[lab][0]) & \
            (rb["net"].index <= dict(IS=IS, OOS=OOS)[lab][1])
        p = stats(rb["net"][m], label=f"基准|{lab}")
        print(f"    {'等权全市场基准':42s} {lab:3s}  年化 {p['CAGR']:>7.2%}  波动 {p['AnnVol']:>6.2%}  "
              f"夏普 {p['Sharpe']:>6.3f}  回撤 {p['MaxDD']:>8.2%}")

    results = {}

    if args.stage == "grid":
        print(f"\n[5] 组合构建网格", flush=True)

        # --- 5a 持仓宽度：双向探测（此前"加宽有利"的假设已被 n=150 的结果否定，
        #         故同时向 n<50 收窄，检验 alpha 是否更集中在极端尾部）---
        print("\n  --- 5a 持仓宽度（F17 等权，月度调仓）---")
        for n in [15, 20, 30, 40, 50, 75, 100, 150, 200]:
            lab = f"F17 n={n}"
            results[lab] = run_variant(P, eligible, scores["F17"], lab,
                                       dict(n=n, weighting="equal", rebalance_days=21))[0]

        # --- 5b 加权方式（得分加权 = 在入选集合内进一步向极端集中）---
        print("\n  --- 5b 加权方式 ---")
        for n in [30, 50, 80]:
            for wmode in ["equal", "score"]:
                lab = f"F17 n={n} {wmode}"
                results[lab] = run_variant(P, eligible, scores["F17"], lab,
                                           dict(n=n, weighting=wmode, rebalance_days=21))[0]

        # --- 5c 因子集（在最有希望的宽度上）---
        print("\n  --- 5c 因子集（n=50 等权）---")
        for sname in FACTOR_SETS:
            lab = f"{sname} n=50"
            results[lab] = run_variant(P, eligible, scores[sname], lab,
                                       dict(n=50, weighting="equal", rebalance_days=21))[0]

        # --- 5d 换手控制 ---
        print("\n  --- 5d 换手控制（F17 n=50，年化换手基数约 1700%）---")
        for kw in [dict(rebalance_days=21),
                   dict(rebalance_days=21, buffer=0.3),
                   dict(rebalance_days=21, buffer=0.6),
                   dict(rebalance_days=21, buffer=1.0),
                   dict(rebalance_days=21, smooth=0.4),
                   dict(rebalance_days=21, smooth=0.7),
                   dict(rebalance_days=21, buffer=0.3, smooth=0.4),
                   dict(rebalance_days=42, buffer=0.3),
                   dict(rebalance_days=10, buffer=0.3)]:
            lab = "F17 n=50 " + ",".join(f"{k}={v}" for k, v in kw.items())
            results[lab] = run_variant(P, eligible, scores["F17"], lab,
                                       dict(n=50, weighting="equal", **kw))[0]

        # --- 5e 因子相关性（诊断冗余：解释为何加 CGO 无增量）---
        print("\n  --- 5e 预处理因子的截面相关性（时序平均）---", flush=True)
        names = ["TURN", "MAX5", "INTRADAY", "ILLIQ", "COMMOVE", "TURN_SHOCK",
                 "REV_1W", "ISKEW", "IVOL", "OVERNIGHT", "CGO_R", "REV_LT",
                 "SIZE", "CO_SKEW"]
        panels = {}
        for n in names:
            try:
                panels[n] = pd.read_pickle(os.path.join(PREP, n + ".pkl"))
            except FileNotFoundError:
                pass
        corr_acc = None
        cnt = 0
        for dt in list(panels.values())[0].index[::63]:
            sub = pd.DataFrame({k: v.loc[dt] for k, v in panels.items()}).dropna(how="all")
            c = sub.corr(min_periods=100)
            corr_acc = c if corr_acc is None else corr_acc.add(c, fill_value=0)
            cnt += 1
        if cnt:
            cm = (corr_acc / cnt).round(3)
            print(cm.to_string())
            cm.to_pickle(os.path.join(OUT, "prep_factor_corr.pkl"))
            hi = [(a, b, cm.loc[a, b]) for i, a in enumerate(names) for b in names[i+1:]
                  if a in cm.index and b in cm.columns and abs(cm.loc[a, b]) > 0.6]
            print(f"\n    |相关系数| > 0.6 的因子对：{len(hi)}")
            for a, b, v in sorted(hi, key=lambda x: -abs(x[2]))[:12]:
                print(f"      {a:10s} ~ {b:10s} {v:+.3f}")
            results["_corr_high_pairs"] = [{"a": a, "b": b, "r": float(v)}
                                          for a, b, v in hi]
        del panels

    if args.stage == "tune":
        # 在 5a/5b/5d 的发现上做二维联合寻优：
        #   集中度 mix（0=等权 … 1=纯得分加权） × 缓冲区 buffer × 持仓数 n
        # 目的：确认「得分加权」与「缓冲区」的收益是稳健的，而不是某一点的偶然。
        print(f"\n[5] 联合寻优：集中度 × 缓冲区 × 持仓数", flush=True)
        print(f"    {'n':>4s} {'mix':>5s} {'buffer':>7s} | {'IS夏普':>7s} {'OOS夏普':>8s} "
              f"{'OOS回撤':>9s} {'OOS Calmar':>10s} {'OOS年化':>8s} {'换手':>8s} {'有效只数':>9s}")
        print("    " + "-" * 88)
        best = None
        for n in [20, 30, 50, 80]:
            for mix in [0.0, 0.25, 0.5, 0.75, 1.0]:
                for buf in [0.0, 0.6]:
                    lab = f"n{n}_mix{mix}_buf{buf}"
                    part, _ = run_variant(P, eligible, scores["F17"], lab,
                                          dict(n=n, weighting="score", mix=mix,
                                               buffer=buf, rebalance_days=21),
                                          verbose=False)
                    i, o = part["IS"], part["OOS"]
                    # 有效持仓只数（逆 Herfindahl），用于识别「集中度收益是否只靠少数票」
                    Wchk = make_weights(scores["F17"], eligible, n=n, weighting="score",
                                        mix=mix, buffer=buf, rebalance_days=21)
                    nz = Wchk[Wchk > 0]
                    eff = float((1.0 / (Wchk ** 2).sum(axis=1).replace(0, np.nan)).mean())
                    mxw = float(Wchk.max().max())
                    results[lab] = {**part, "eff_names": round(eff, 1),
                                    "max_w": round(mxw, 4)}
                    print(f"    {n:>4d} {mix:>5.2f} {buf:>7.1f} | {i['Sharpe']:>7.3f} "
                          f"{o['Sharpe']:>8.3f} {o['MaxDD']:>9.2%} {str(o['Calmar']):>10s} "
                          f"{o['CAGR']:>8.2%} {o['ann_turnover']:>8.0%} {eff:>9.1f}", flush=True)
                    key = o["Sharpe"]
                    if best is None or key > best[0]:
                        best = (key, lab, i, o, eff, mxw)
                    del Wchk
        print(f"\n    样本外夏普最高：{best[1]}  夏普 {best[0]:.3f}  "
              f"回撤 {best[3]['MaxDD']:.2%}  有效只数 {best[4]:.1f}  单票最大权重 {best[5]:.2%}")
        results["_best"] = {"label": best[1], "oos_sharpe": best[0],
                            "eff_names": round(best[4], 1), "max_w": round(best[5], 4)}

        # --- 权重上限敏感性（防止「集中度收益」其实是单票赌博）---
        print(f"\n  --- 单票权重上限敏感性（取上表最优的 n/mix/buffer）---", flush=True)
        import re as _re
        m_ = _re.match(r"n(\d+)_mix([\d.]+)_buf([\d.]+)", best[1])
        bn, bmix, bbuf = int(m_.group(1)), float(m_.group(2)), float(m_.group(3))
        for cap in [0.02, 0.03, 0.05, 0.08, 0.12, 1.0]:
            lab = f"cap{cap}"
            part, _ = run_variant(P, eligible, scores["F17"], lab,
                                  dict(n=bn, weighting="score", mix=bmix, buffer=bbuf,
                                       max_weight=cap, rebalance_days=21), verbose=False)
            o = part["OOS"]
            results["cap_" + lab] = part
            print(f"    上限 {cap:>5.1%}  OOS 夏普 {o['Sharpe']:>6.3f}  回撤 {o['MaxDD']:>8.2%}  "
                  f"Calmar {str(o['Calmar']):>5s}  年化 {o['CAGR']:>7.2%}", flush=True)

    if args.stage == "beta":
        import statsmodels.api as sm
        print(f"\n[5] β 目标化组合构建（直接针对最大回撤）", flush=True)
        # f_beta = −c.beta，取负还原为原始 β
        beta_raw = -pd.read_pickle(os.path.join(DATA, "factors", "BETA.pkl"))
        beta_raw = beta_raw.astype(F32)
        print(f"    β 面板: {beta_raw.shape}  中位数 {float(np.nanmedian(beta_raw.to_numpy())):.3f}",
              flush=True)
        mkt = pd.read_pickle(os.path.join(DATA, "market_daily.pkl"))["mkt_vw"]

        def eval_beta(lab, W):
            res = backtest(P, W, CFG, lab)
            del W
            r = res["net"]
            out = {"stats": slice_stats(r, res["gross"], res["turnover"], lab)}
            for tag, (st, en) in [("IS", IS), ("OOS", OOS)]:
                d = pd.concat({"p": r, "m": mkt.reindex(r.index)}, axis=1).dropna()
                d = d[(d.index >= st) & (d.index <= en)]
                if len(d) < 100:
                    continue
                mm = sm.OLS(d["p"], sm.add_constant(d["m"])).fit(
                    cov_type="HAC", cov_kwds={"maxlags": 5})
                out[f"beta_{tag}"] = {
                    "beta": round(float(mm.params["m"]), 3),
                    "alpha_ann": round(float(mm.params["const"]) * PPY, 4),
                    "alpha_t": round(float(mm.tvalues["const"]), 2)}
            return out, res

        bt = {}
        for tg in [None, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            for pool in ([200] if tg is None else [200, 400]):
                lab = f"β目标={tg} pool={pool}"
                W = make_weights_beta(scores["F17"], eligible, beta_raw, n=50,
                                      pool=pool, beta_target=tg, rebalance_days=21,
                                      max_weight=0.03)
                o, _ = eval_beta(lab, W)
                bt[lab] = o
                i, os_ = o["stats"]["IS"], o["stats"]["OOS"]
                print(f"    {lab:26s} IS β={o.get('beta_IS',{}).get('beta')} "
                      f"夏普 {i['Sharpe']:>6.3f} | "
                      f"OOS β={o.get('beta_OOS',{}).get('beta')} "
                      f"夏普 {os_['Sharpe']:>6.3f} 年化 {os_['CAGR']:>7.2%} "
                      f"回撤 {os_['MaxDD']:>8.2%} Calmar {str(os_['Calmar']):>5s}", flush=True)
                del o

        with open(os.path.join(OUT, "v2_beta.json"), "w", encoding="utf-8") as fh:
            json.dump(bt, fh, ensure_ascii=False, indent=2, default=str)
        print(f"\n[已保存] {os.path.join(OUT,'v2_beta.json')}")
        print(f"总用时 {time.time()-t_all:.0f}s")
        return

    if args.stage == "phase":
        # 决定性检验：改进到底是真的，还是调仓相位的运气？
        # 原版在相位 0 得到 OOS 夏普 1.017，但平移相位后降到 0.75~0.94。
        # 因此必须在**全部 21 个相位上求平均**再比较，否则是用单点幸运值下结论。
        from ashare_factor_strategy import newey_west_tstat, deflated_sharpe, probabilistic_sharpe
        print(f"\n[5] 新增换手率族因子校验", flush=True)
        ic_tbl = {}
        for n in ["ABN_TURN", "VTURN", "TURN", "TURN_SHOCK", "REV_1W"]:
            r = rank_ic_report(P, eligible, n)
            ic_tbl[n] = r
            i2, o2 = r.get("IS", {}), r.get("OOS", {})
            print(f"    {n:10s} 样本内 IC={i2.get('IC')} (t={i2.get('t')}, ICIR={i2.get('ICIR')})"
                  f"  |  样本外 IC={o2.get('IC')} (t={o2.get('t')})", flush=True)

        print(f"\n[6] 相位平均对比（调仓相位平移，消除「相位运气」）", flush=True)
        CONFIGS = {
            "原版 n50 等权 cap5% 无缓冲":
                dict(n=50, weighting="equal", buffer=0.0, rebalance_days=21, max_weight=0.05),
            "改进B n50 等权 cap3% 缓冲0.6":
                dict(n=50, weighting="equal", buffer=0.6, rebalance_days=21, max_weight=0.03),
            "池200 得分加权 n50 cap3% 无缓冲":
                dict(n=50, pool=200, weighting="score", mix=1.0, buffer=0.0,
                     rebalance_days=21, max_weight=0.03),
            "池200 得分加权 n50 cap3% 缓冲0.3":
                dict(n=50, pool=200, weighting="score", mix=1.0, buffer=0.3,
                     rebalance_days=21, max_weight=0.03),
            "池200 得分加权 n50 cap3% 缓冲0.6":
                dict(n=50, pool=200, weighting="score", mix=1.0, buffer=0.6,
                     rebalance_days=21, max_weight=0.03),
            "池400 得分加权 n50 cap3% 缓冲0.6":
                dict(n=50, pool=400, weighting="score", mix=1.0, buffer=0.6,
                     rebalance_days=21, max_weight=0.03),
        }
        phase_tbl = {}
        for sname in ["F17"]:
            sc_all = scores[sname]
            for lab, kw in CONFIGS.items():
                key = f"{sname}|{lab}"
                shs_i, shs_o, dd_o, cg_o, tv_o = [], [], [], [], []
                for off in range(0, 21, 2):        # 11 个相位
                    sc = sc_all.iloc[off:] if off else sc_all
                    W = make_weights(sc, eligible.reindex(sc.index), **kw)
                    res = backtest(P, W, CFG, key)
                    del W
                    for tag, shs, dds, cgs, tvs in [("IS", shs_i, None, None, None),
                                                    ("OOS", shs_o, dd_o, cg_o, tv_o)]:
                        st, en = dict(IS=IS, OOS=OOS)[tag]
                        r = res["net"]
                        r = r[(r.index >= st) & (r.index <= en)]
                        if len(r) < 100:
                            continue
                        vol = r.std() * np.sqrt(PPY)
                        shs.append((r.mean() * PPY - RF) / vol)
                        if dds is not None:
                            eq = (1 + r).cumprod()
                            dds.append(float((eq / eq.cummax() - 1).min()))
                            cgs.append(float(eq.iloc[-1] ** (PPY / len(r)) - 1))
                            tvs.append(float(res["turnover"].reindex(r.index).mean() * PPY))
                    del res
                if shs_o:
                    phase_tbl[key] = {
                        "IS_sharpe_mean": round(float(np.mean(shs_i)), 4),
                        "IS_sharpe_sd": round(float(np.std(shs_i)), 4),
                        "OOS_sharpe_mean": round(float(np.mean(shs_o)), 4),
                        "OOS_sharpe_sd": round(float(np.std(shs_o)), 4),
                        "OOS_sharpe_min": round(float(np.min(shs_o)), 4),
                        "OOS_sharpe_max": round(float(np.max(shs_o)), 4),
                        "OOS_cagr_mean": round(float(np.mean(cg_o)), 4),
                        "OOS_maxdd_mean": round(float(np.mean(dd_o)), 4),
                        "OOS_maxdd_worst": round(float(np.min(dd_o)), 4),
                        "OOS_calmar_mean": round(float(np.mean(cg_o) / abs(np.mean(dd_o))), 3),
                        "OOS_turnover_mean": round(float(np.mean(tv_o)), 4),
                        "n_phases": len(shs_o)}
                    t = phase_tbl[key]
                    print(f"    {key:40s} IS {t['IS_sharpe_mean']:.3f}±{t['IS_sharpe_sd']:.3f} | "
                          f"OOS {t['OOS_sharpe_mean']:.3f}±{t['OOS_sharpe_sd']:.3f} "
                          f"[{t['OOS_sharpe_min']:.3f},{t['OOS_sharpe_max']:.3f}]  "
                          f"年化 {t['OOS_cagr_mean']:>7.2%}  回撤 {t['OOS_maxdd_mean']:>7.2%}  "
                          f"Calmar {t['OOS_calmar_mean']:.2f}  换手 {t['OOS_turnover_mean']:>6.0%}",
                          flush=True)
                del shs_i, shs_o
            del sc_all
        out = {"phase": phase_tbl, "ic": ic_tbl}
        with open(os.path.join(OUT, "v2_phase.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2, default=str)
        print(f"\n[已保存] {os.path.join(OUT,'v2_phase.json')}")
        print(f"总用时 {time.time()-t_all:.0f}s")
        return

    if args.stage == "final":
        from ashare_factor_strategy import (newey_west_tstat, probabilistic_sharpe,
                                            deflated_sharpe)
        import statsmodels.api as sm

        # ------------------------------------------------------------------
        # 方法学纪律（本项目最重要的自我约束）
        # ------------------------------------------------------------------
        # grid + tune 两个阶段共评估了约 80 个配置。如果按**样本外夏普**挑参数，
        # 样本外就被用于选择，不再干净。因此本阶段：
        #   · 候选配置刻意收缩到 6 个，且每个都有先验理由；
        #   · 主口径按**样本内夏普**选择，样本外只作报告；
        #   · 同时报告「机制驱动」的选择（缓冲区属于交易成本控制，不是拟合参数）；
        #   · DSR 的试验次数用**累计评估过的全部配置数**，不隐瞒。
        print(f"\n[5] 最终候选配置（IS 与 OOS 同时报告，选择只依据 IS）", flush=True)
        CANDS = {
            "A 基准 n50 等权": dict(n=50, weighting="equal", buffer=0.0),
            "B n50 等权 +缓冲0.6": dict(n=50, weighting="equal", buffer=0.6),
            "C n30 等权 +缓冲0.6": dict(n=30, weighting="equal", buffer=0.6),
            "D n20 等权 +缓冲0.6": dict(n=20, weighting="equal", buffer=0.6),
            "E n30 mix0.5 +缓冲0.6": dict(n=30, weighting="score", mix=0.5, buffer=0.6),
            "F n50 mix0.5 +缓冲0.6": dict(n=50, weighting="score", mix=0.5, buffer=0.6),
        }
        KW = dict(rebalance_days=21, max_weight=0.03)
        fin = {}
        for lab, kw in CANDS.items():
            part, res = run_variant(P, eligible, scores["F17"], lab, {**KW, **kw})
            fin[lab] = {"params": kw, "stats": part, "res": res}
        # 样本内最优
        is_best = max(CANDS, key=lambda k: fin[k]["stats"]["IS"]["Sharpe"])
        print(f"\n    按样本内夏普选择 → {is_best}  "
              f"(IS {fin[is_best]['stats']['IS']['Sharpe']:.3f})  "
              f"其样本外夏普 {fin[is_best]['stats']['OOS']['Sharpe']:.3f}")
        fin["_is_selected"] = is_best

        # ---- 基准（原始已交付版本：n=50 等权、无缓冲、5% 上限）----
        part0, res0 = run_variant(P, eligible, scores["F17"], "原版 n50 等权 cap5%",
                                  dict(n=50, weighting="equal", buffer=0.0,
                                       rebalance_days=21, max_weight=0.05))
        fin["原版"] = {"stats": part0, "res": res0}

        # ------------------------------------------------------------------
        print(f"\n[6] 稳健性检验", flush=True)
        oos = (lambda s: s[(s.index >= OOS[0]) & (s.index <= OOS[1])])

        # --- 6a 调仓相位：把调仓日整体平移 0..20 天，看结果是否依赖特定相位 ---
        print("\n  --- 6a 调仓相位稳健性（调仓起始日平移）---")
        base_lab = is_best
        ph = []
        for off in range(0, 21, 2):
            sc = scores["F17"].iloc[off:] if off else scores["F17"]
            W = make_weights(sc, eligible.reindex(sc.index), **{**KW, **CANDS[base_lab]})
            r = backtest(P, W, CFG, f"phase{off}")["net"]
            o = oos(r)
            if len(o) < 100:
                continue
            eq = (1 + o).cumprod()
            cagr = eq.iloc[-1] ** (252 / len(o)) - 1
            vol = o.std() * np.sqrt(252)
            dd = float((eq / eq.cummax() - 1).min())
            ph.append((off, (o.mean() * 252 - RF) / vol, cagr, dd))
            print(f"    起始偏移 {off:>2d} 日:  OOS 夏普 {(o.mean()*252-RF)/vol:>6.3f}  "
                  f"年化 {cagr:>7.2%}  回撤 {dd:>8.2%}", flush=True)
        if ph:
            arr = np.array([[p[1], p[2], p[3]] for p in ph])
            print(f"    → 夏普 均值 {arr[:,0].mean():.3f}  标准差 {arr[:,0].std():.3f}  "
                  f"min {arr[:,0].min():.3f}  max {arr[:,0].max():.3f}")
            fin["phase_robust"] = {"sharpe_mean": round(float(arr[:, 0].mean()), 4),
                                   "sharpe_sd": round(float(arr[:, 0].std()), 4),
                                   "sharpe_min": round(float(arr[:, 0].min()), 4),
                                   "sharpe_max": round(float(arr[:, 0].max()), 4),
                                   "n_phases": len(ph)}

        # --- 6a-2 相位平均对比：原版 vs 选定方案（避免用单一相位的幸运值下结论）---
        print("\n  --- 6a-2 相位平均对比（21 个调仓相位上求平均，消除相位运气）---")
        phase_cmp = {}
        for lab in ["原版", base_lab]:
            kw = dict(KW) if lab == "原版" else {**KW, **CANDS[base_lab]}
            if lab == "原版":
                kw = dict(n=50, weighting="equal", buffer=0.0, rebalance_days=21,
                          max_weight=0.05)
            elif lab == "B n50 等权 +缓冲0.6":
                kw = dict(n=50, weighting="equal", buffer=0.6, rebalance_days=21,
                          max_weight=0.03)
            shs, dds, cagrs = [], [], []
            for off in range(0, 21, 2):
                sc = scores["F17"].iloc[off:] if off else scores["F17"]
                W = make_weights(sc, eligible.reindex(sc.index), **kw)
                r = backtest(P, W, CFG, f"{lab}_ph{off}")["net"]
                o = oos(r)
                if len(o) < 100:
                    continue
                eq = (1 + o).cumprod()
                vol = o.std() * np.sqrt(252)
                shs.append((o.mean() * 252 - RF) / vol)
                cagrs.append(eq.iloc[-1] ** (252 / len(o)) - 1)
                dds.append(float((eq / eq.cummax() - 1).min()))
            if shs:
                phase_cmp[lab] = {
                    "sharpe_mean": round(float(np.mean(shs)), 4),
                    "sharpe_sd": round(float(np.std(shs)), 4),
                    "sharpe_min": round(float(np.min(shs)), 4),
                    "sharpe_max": round(float(np.max(shs)), 4),
                    "cagr_mean": round(float(np.mean(cagrs)), 4),
                    "maxdd_mean": round(float(np.mean(dds)), 4),
                    "n_phases": len(shs)}
                print(f"    {lab:24s} 夏普 均值 {np.mean(shs):.3f} (sd {np.std(shs):.3f}, "
                      f"区间 {np.min(shs):.3f}~{np.max(shs):.3f})  "
                      f"年化均值 {np.mean(cagrs):>7.2%}  回撤均值 {np.mean(dds):>8.2%}")
        fin["phase_compare"] = phase_cmp

        # --- 6a-3 情绪条件化（按中国文献预测的方向：高情绪后加仓）---
        # 依据：Cheema, Nartea & Man (2020, International Review of Finance) 发现
        # 中国 MAX 溢价**只在情绪高企后显著**：高情绪后 +1.85%/月 (t=3.83)，
        # 低情绪后 +0.22% (t=0.51)，FF 调整后 alpha 差 2.37%/月 (t=4.21)。
        # 本策略重仓 MAX5/TURN，故应检验「高情绪加仓」而非「高情绪减仓」。
        # 情绪代理只用可构造的两个 Baker-Wurgler 代理（换手率 + IPO 数量）：
        # 子代理核实 BW 六代理中 TURN 与 NIPO 可直接构造，CEFD/PD-ND/S 不可。
        print("\n  --- 6a-3 情绪条件化（中国文献方向：高情绪加仓）---")
        md = pd.read_pickle(os.path.join(DATA, "market_daily2.pkl"))
        mo = md.resample("ME").last().index
        mo = pd.DataFrame(index=mo)
        mo["turn"] = md["turnover_agg"].resample("ME").mean()
        mo["turn"] = np.log(mo["turn"]) - np.log(mo["turn"]).rolling(60).mean()  # BW 的 5 年去趋势
        mo["ipo"] = md["ipo_n"].resample("ME").sum()
        m_is = (mo.index >= IS[0]) & (mo.index <= IS[1])
        Zs = pd.DataFrame(index=mo.index)
        for c in ["turn", "ipo"]:
            mu, sd = mo.loc[m_is, c].mean(), mo.loc[m_is, c].std()
            Zs[c] = (mo[c] - mu) / sd
        Zc = Zs.loc[m_is].dropna()
        X = Zc.to_numpy() - Zc.to_numpy().mean(axis=0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        load = vt[0]
        if np.corrcoef(X @ load, Zc["turn"])[0, 1] < 0:
            load = -load
        S = pd.Series(Zs.to_numpy() @ load, index=Zs.index)
        S = (S - S.loc[m_is].mean()) / S.loc[m_is].std()
        med = S.expanding(min_periods=36).median().shift(1)      # 扩展中位数，无前视
        hi_flag = (S.shift(1) > med).astype(float)
        sent_res = {"loadings": [round(float(x), 4) for x in load],
                    "n_high_months": int(hi_flag.sum())}
        rB = fin[base_lab]["res"]["net"]
        for kk in [0.0, 0.25, 0.5, 1.0]:
            sc_d = (1.0 + kk * hi_flag).reindex(rB.index, method="ffill").shift(1).fillna(1.0)
            rr = (rB * sc_d).dropna()
            row = {}
            for lab2, (st, en) in [("IS", IS), ("OOS", OOS)]:
                o = rr[(rr.index >= st) & (rr.index <= en)]
                if len(o) < 100:
                    continue
                eq = (1 + o).cumprod()
                vol = o.std() * np.sqrt(252)
                row[lab2] = {"sharpe": round(float((o.mean() * 252 - RF) / vol), 4),
                             "cagr": round(float(eq.iloc[-1] ** (252 / len(o)) - 1), 4),
                             "maxdd": round(float((eq / eq.cummax() - 1).min()), 4)}
            sent_res[f"k={kk}"] = row
            if row:
                i2, o2 = row.get("IS", {}), row.get("OOS", {})
                print(f"    高情绪加仓 k={kk:<4.2f}  IS 夏普 {i2.get('sharpe')}  |  "
                      f"OOS 夏普 {o2.get('sharpe')}  年化 {o2.get('cagr'):.2%}  "
                      f"回撤 {o2.get('maxdd'):.2%}")
        fin["sentiment_conditioning"] = sent_res

        # --- 6b alpha/beta 分解：确认改善来自 alpha 而非仅仅是低 beta ---
        print("\n  --- 6b 对市场（流通市值加权）的 alpha/beta 分解（OOS）---")
        mkt = pd.read_pickle(os.path.join(DATA, "market_daily.pkl"))["mkt_vw"]
        abr = {}
        for lab in ["原版"] + list(CANDS):
            r = fin[lab]["res"]["net"]
            d = pd.concat({"p": r, "m": mkt.reindex(r.index)}, axis=1).dropna()
            d = d[(d.index >= OOS[0]) & (d.index <= OOS[1])]
            if len(d) < 100:
                continue
            mm = sm.OLS(d["p"], sm.add_constant(d["m"])).fit(
                cov_type="HAC", cov_kwds={"maxlags": 5})
            a, b = float(mm.params["const"]), float(mm.params["m"])
            abr[lab] = {"alpha_ann": round(a * 252, 4), "alpha_t": round(float(mm.tvalues["const"]), 2),
                        "beta": round(b, 3), "beta_t": round(float(mm.tvalues["m"]), 2),
                        "r2": round(float(mm.rsquared), 4)}
            print(f"    {lab:24s} α={a*252:>+7.2%}/年 (t={mm.tvalues['const']:>+5.2f})  "
                  f"β={b:>5.3f} (t={mm.tvalues['m']:>+6.2f})  R²={mm.rsquared:.4f}")
        fin["alpha_beta_OOS"] = abr

        # --- 6c 自举：改善是否显著？（重叠块自举，块长 21 日）---
        print("\n  --- 6c 分块自举：选定方案 vs 原版的 OOS 夏普差（2000 次，块长21日）---")
        r_new = oos(fin[base_lab]["res"]["net"])
        r_old = oos(fin["原版"]["res"]["net"])
        d = (r_new - r_old).dropna()
        rng = np.random.default_rng(CFG.random_seed)
        n = len(d)
        nb = int(np.ceil(n / 21))
        diffs = []
        for _ in range(CFG.bootstrap_n):
            starts = rng.integers(0, n, nb)
            idx = np.concatenate([np.arange(s0, min(s0 + 21, n)) for s0 in starts])[:n]
            dd = d.iloc[idx]
            rn = r_new.iloc[idx]
            ro = r_old.iloc[idx]
            if dd.std() == 0:
                continue
            sn = (rn.mean() * 252 - RF) / (rn.std() * np.sqrt(252))
            so = (ro.mean() * 252 - RF) / (ro.std() * np.sqrt(252))
            diffs.append(sn - so)
        diffs = np.array(diffs)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        pval = float((diffs <= 0).mean())
        print(f"    夏普差 均值 {diffs.mean():+.3f}  95%区间 [{lo:+.3f}, {hi:+.3f}]  "
              f"P(差<=0)={pval:.4f}")
        fin["bootstrap_sharpe_diff"] = {"mean": round(float(diffs.mean()), 4),
                                        "ci95": [round(float(lo), 4), round(float(hi), 4)],
                                        "p_le_0": round(pval, 4), "n_boot": int(len(diffs))}

        # --- 6d 成本敏感性 ---
        print("\n  --- 6d 成本敏感性（选定方案，交易成本倍数）---")
        base_cfg = dict(commission_bp=CFG.commission_bp, stamp_duty_bp=CFG.stamp_duty_bp,
                        slippage_bp=CFG.slippage_bp)
        cost_tbl = {}
        sc = scores["F17"]
        W = make_weights(sc, eligible, **{**KW, **CANDS[base_lab]})
        for mult in [1.0, 1.5, 2.0, 3.0]:
            CFG.commission_bp = base_cfg["commission_bp"] * mult
            CFG.stamp_duty_bp = base_cfg["stamp_duty_bp"] * mult
            CFG.slippage_bp = base_cfg["slippage_bp"] * mult
            r = backtest(P, W, CFG, f"cost{mult}")["net"]
            o = oos(r)
            vol = o.std() * np.sqrt(252)
            eq = (1 + o).cumprod()
            sh = (o.mean() * 252 - RF) / vol
            dd = float((eq / eq.cummax() - 1).min())
            cost_tbl[mult] = {"sharpe": round(float(sh), 4), "maxdd": round(dd, 4)}
            print(f"    成本 ×{mult:<4.1f}  OOS 夏普 {sh:>6.3f}  回撤 {dd:>8.2%}")
        for k, v in base_cfg.items():
            setattr(CFG, k, v)

        # --- 6e 显著性（选定方案）---
        print("\n  --- 6e 显著性检验（选定方案，样本外多空/多头）---")
        # 多空口径：多头腿 − 空头腿
        Ws = make_weights(scores["F17"], eligible, side="short", **{**KW, **CANDS[base_lab]})
        r_sh = backtest(P, Ws, CFG, "空头腿")["net"]
        r_ls = (fin[base_lab]["res"]["net"] - r_sh).dropna()
        n_trials = 80          # grid + tune 累计评估的配置数（诚实计入）
        sig = {}
        for nm, ser in [("多头", fin[base_lab]["res"]["net"]), ("多空", r_ls)]:
            o = oos(ser)
            t = newey_west_tstat(o, lags=5)[1]        # 返回 (mean, t, n)
            sr_p = (o.mean() * 252 - RF) / (o.std() * np.sqrt(252))
            psr = probabilistic_sharpe(sr_p / np.sqrt(252), len(o),
                                       float(o.skew()), float(o.kurtosis() + 3.0))
            dsr = deflated_sharpe(o, n_trials=n_trials)
            sig[nm] = {"sharpe": round(float(sr_p), 4),
                       "nw_t": round(float(t), 3),
                       "PSR": round(float(psr), 4),
                       "DSR": round(float(dsr["DSR"]), 4),
                       "SR0_ann": round(float(dsr["SR0"]), 4),
                       "ann_turnover": round(float(fin[base_lab]["res"]["turnover"]
                                                   .reindex(o.index).mean() * PPY), 4),
                       "CAGR": round(float((1 + o).prod() ** (252 / len(o)) - 1), 4),
                       "MaxDD": round(float(((1 + o).cumprod() /
                                             (1 + o).cumprod().cummax() - 1).min()), 4)}
            print(f"    {nm}: 夏普 {sr_p:.3f}  NW t={t:.3f}  PSR={psr:.4f}  "
                  f"DSR={dsr['DSR']:.4f} (门槛 SR0={dsr['SR0']:.3f})  "
                  f"年化 {sig[nm]['CAGR']:.2%}  回撤 {sig[nm]['MaxDD']:.2%}  "
                  f"换手 {sig[nm]['ann_turnover']:.0%}")
        fin["significance"] = sig
        fin["n_trials_accounted"] = n_trials

        with open(os.path.join(OUT, "v2_final.json"), "w", encoding="utf-8") as fh:
            json.dump({k: (v if not isinstance(v, dict) or "res" not in v else
                           {"params": v.get("params"), "stats": v["stats"]})
                       for k, v in fin.items()}, fh, ensure_ascii=False, indent=2, default=str)
        # 保存选定方案的日收益，供后续使用
        import pickle as _pk
        with open(os.path.join(OUT, "v2_returns.pkl"), "wb") as fh:
            _pk.dump({"selected": base_lab,
                      "net": fin[base_lab]["res"]["net"],
                      "gross": fin[base_lab]["res"]["gross"],
                      "turnover": fin[base_lab]["res"]["turnover"],
                      "cost": fin[base_lab]["res"]["cost"],
                      "baseline_net": fin["原版"]["res"]["net"],
                      "ls_net": r_ls, "short_net": r_sh}, fh)
        print(f"\n[已保存] {os.path.join(OUT,'v2_final.json')} 与 v2_returns.pkl")

    with open(os.path.join(OUT, f"v2_{args.stage}.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n[已保存] {os.path.join(OUT, f'v2_{args.stage}.json')}")
    print(f"总用时 {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
