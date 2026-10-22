# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
波动率目标化叠加分析 (vt_analysis.py)

复用主流程已缓存的因子与已选因子清单，只重做「组合构建 → 回测 → 波动率叠加」，
避免重复 10 分钟的因子检验。产出对比表与图表。

用法: python vt_analysis.py [--lsy 1]
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ashare_factor_strategy as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lsy", type=int, default=1)
    ap.add_argument("--target-vol", type=float, default=0.15)
    args = ap.parse_args()

    S.CFG.use_lsy_universe = bool(args.lsy)
    tag = "LSY" if S.CFG.use_lsy_universe else "FULL"

    sp = os.path.join(S.OUT, f"summary_{tag}.json")
    if not os.path.exists(sp):
        print(f"未找到 {sp}，请先运行主流程"); return
    with open(sp, encoding="utf-8") as f:
        summary = json.load(f)
    keep = summary["selected_factors"]
    print(f"[{tag}] 已选因子 {len(keep)} 个: {keep}", flush=True)

    print("加载面板 ...", flush=True)
    df = S.load_panel()
    P = S.Panels(df)
    del df
    eligible, _, _ = S.build_universe(P, S.CFG)

    fdir = os.path.join(S.DATA, "factors")
    acc = cnt = None
    for k in keep:
        pre = S.prepare_factor(pd.read_pickle(os.path.join(fdir, f"{k}.pkl")),
                               P, eligible, S.CFG)
        v = pre.notna()
        acc = pre.where(v, 0.0) if acc is None else acc.add(pre.where(v, 0.0), fill_value=0.0)
        cnt = v.astype("float32") if cnt is None else cnt.add(v.astype("float32"), fill_value=0.0)
        del pre, v
    score = (acc / cnt.replace(0, np.nan)).astype("float32")
    del acc, cnt
    print("组合得分构建完成", flush=True)

    W_long = S.build_weights(score, eligible, "long", S.CFG)
    W_ls = S.build_long_short(score, eligible, S.CFG)

    res = {"多头组合": S.backtest(P, W_long, S.CFG, "多头组合"),
           "多空组合": S.backtest(P, W_ls, S.CFG, "多空组合")}

    def sl(x, a, b):
        m = (x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))
        return x[m]

    B0, B1 = S.CFG.bt_start, S.CFG.bt_end
    rows = []
    print("\n" + "=" * 100)
    print(f"  [{tag}] 波动率目标化叠加效果（样本外 {B0} ~ {B1}）")
    print("=" * 100)
    print(f"  {'组合':<22}{'口径':<10}{'年化':>9}{'波动':>9}{'夏普':>8}{'最大回撤':>10}{'Calmar':>8}")
    print("-" * 100)
    out = {}
    for k, r in res.items():
        for mode in ("原始", f"波动率目标{args.target_vol*100:.0f}%"):
            if mode == "原始":
                net = r["net"]; lev = pd.Series(1.0, index=r["net"].index)
            else:
                net, lev = S.vol_targeted(r["net"], args.target_vol, 60, 1.5)
            st = S.summarize(sl(net, B0, B1), S.CFG.rf_annual, k + "+" + mode)
            out[f"{k}|{mode}"] = st
            print(f"  {k:<22}{mode:<10}{st['CAGR']*100:>8.2f}%{st['AnnVol']*100:>8.2f}%"
                  f"{st['Sharpe']:>8.3f}{st['MaxDD']*100:>9.2f}%{st['Calmar']:>8.2f}")
        print("-" * 100)

    # 全样本对照
    print(f"\n  全样本口径 ({S.CFG.full_start} ~ {S.CFG.full_end}):")
    for k, r in res.items():
        for mode in ("原始", f"波动率目标{args.target_vol*100:.0f}%"):
            net = r["net"] if mode == "原始" else S.vol_targeted(r["net"], args.target_vol, 60, 1.5)[0]
            st = S.summarize(sl(net, S.CFG.full_start, S.CFG.full_end), S.CFG.rf_annual, k)
            print(f"  {k:<14}{mode:<14} 年化={st['CAGR']*100:6.2f}%  夏普={st['Sharpe']:6.3f}  "
                  f"回撤={st['MaxDD']*100:7.2f}%")

    # 显著性
    print(f"\n  显著性（样本外，Newey-West）:")
    for k, r in res.items():
        for mode in ("原始", f"波动率目标{args.target_vol*100:.0f}%"):
            net = r["net"] if mode == "原始" else S.vol_targeted(r["net"], args.target_vol, 60, 1.5)[0]
            x = sl(net, B0, B1)
            mu, t, T = S.newey_west_tstat(x, S.CFG.nw_lags)
            print(f"  {k:<14}{mode:<14} NW t = {t:6.3f}   "
                  f"{'显著(|t|>1.96)' if abs(t) > 1.96 else '不显著'}")

    # 保存
    pd.to_pickle({f"{k}|{m}": out[f"{k}|{m}"] for k in res
                  for m in ("原始", f"波动率目标{args.target_vol*100:.0f}%")},
                 os.path.join(S.OUT, f"voltarget_{tag}.pkl"))
    with open(os.path.join(S.OUT, f"voltarget_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating))
                            and np.isfinite(vv) else None)
                       for kk, vv in v.items() if kk in
                       ("name", "CAGR", "AnnVol", "Sharpe", "MaxDD", "Calmar",
                        "Sortino", "WinRate", "years")}
                   for k, v in out.items()}, f, ensure_ascii=False, indent=2)
    print(f"\n  已保存 -> output/voltarget_{tag}.json")


if __name__ == "__main__":
    main()
