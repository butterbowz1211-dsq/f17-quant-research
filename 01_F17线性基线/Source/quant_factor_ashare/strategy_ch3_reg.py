# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
strategy_ch3_reg.py — 策略组合本身对 CH-3 / CH-4 的归因
================================================================================
ch3_residual_test.py 已证明**因子层面** 16/21 存活 CH-4。本节回答更直接的问题：

    策略组合（而非单个因子）的收益，有多少能被 CH-3 / CH-4 解释？
    即：所谓 alpha，扣掉中国市场的规模与价值 beta 之后还剩多少？

同时分样本内(2010-2022) / 样本外(2023-2026) 报告，因为因子选择用的是样本内窗口。
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
NW = 6

ch = pd.read_pickle(os.path.join(DATA, "ch3.pkl"))
# ★ 时间对齐：ch3 的因子在索引 t 上代表的是 **t → t+1** 的收益（组合在月末 t 形成、
#   持有到 t+1 月末）；而策略的月度收益在索引 t 上是 **t-1 → t**（截至 t 的过去一月）。
#   因此必须把 ch 下移一期再对齐，否则等于拿「下个月的市场收益」解释「这个月的策略收益」。
#   报警信号：不平移时多头组合的 CAPM β_MKT = 0.066、R² = 0.004，
#   而日频回归测得 β = 0.93 —— 月频不可能与日频差这么多。
ch = ch.shift(1)
with open(os.path.join(OUT, "returns_LSY.pkl"), "rb") as fh:
    R = pickle.load(fh)

MODELS = {"原始": [], "CAPM": ["MKT"], "CH-3": ["MKT", "SMB", "VMG"],
          "CH-4": ["MKT", "SMB", "VMG", "PMO"]}
SPLITS = {"全样本": ("2010-01-01", "2026-12-31"),
          "样本内": ("2010-01-01", "2022-12-31"),
          "样本外": ("2023-01-01", "2026-12-31")}

out = {}
print("=" * 100)
print("策略组合对 CH-3 / CH-4 的回归（月末，Newey-West lag=6）")
print("=" * 100)

for key, lab in [("多头组合(可交易)", "多头组合"), ("多空组合(学术)", "多空组合")]:
    daily = R[key]["net"].dropna()
    mm = daily.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    print(f"\n--- {lab} ---")
    print(f"  {'区间':6s} {'模型':6s} {'α(月%)':>9s} {'α(年%)':>9s} {'t':>7s} "
          f"{'β_MKT':>8s} {'β_SMB':>8s} {'β_VMG':>8s} {'β_PMO':>8s} {'R²':>7s}")
    for snm, (s, e) in SPLITS.items():
        sub = mm[(mm.index >= s) & (mm.index <= e)]
        for mn, cols in MODELS.items():
            d = pd.concat({"y": sub, **{c: ch[c] for c in cols}}, axis=1).dropna()
            if len(d) < 24:
                continue
            X = sm.add_constant(d[cols]) if cols else pd.DataFrame(
                {"const": np.ones(len(d))}, index=d.index)
            m = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": NW})
            a = float(m.params["const"])
            bs = {c: float(m.params[c]) for c in cols}
            out[f"{lab}|{snm}|{mn}"] = {
                "alpha_m": round(a * 100, 4), "alpha_a": round(((1 + a) ** 12 - 1) * 100, 3),
                "t": round(float(m.tvalues["const"]), 2),
                "betas": {k: round(v, 3) for k, v in bs.items()},
                "r2": round(float(m.rsquared), 4), "n": int(len(d))}
            star = "  ★" if abs(m.tvalues["const"]) >= 1.96 else ""
            print(f"  {snm:6s} {mn:6s} {a*100:>+9.3f} {((1+a)**12-1)*100:>+9.2f} "
                  f"{m.tvalues['const']:>+7.2f} "
                  f"{bs.get('MKT', float('nan')):>8.3f} {bs.get('SMB', float('nan')):>8.3f} "
                  f"{bs.get('VMG', float('nan')):>8.3f} {bs.get('PMO', float('nan')):>8.3f} "
                  f"{m.rsquared:>7.4f}{star}")

with open(os.path.join(OUT, "strategy_ch3_reg.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=2)
print(f"\n[已保存] {os.path.join(OUT,'strategy_ch3_reg.json')}")
