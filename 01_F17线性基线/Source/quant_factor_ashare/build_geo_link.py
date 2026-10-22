# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
build_geo_link.py — 地理联动因子（Parsons, Sabbatucci & Titman 2020, RFS 33(10)）
================================================================================
文献依据
--------
Parsons, Sabbatucci & Titman (2020), "Geographic Lead-Lag Effects",
Review of Financial Studies 33(10), 4721-4770.（AEA 2020 会议版全文 65pp 核对）

  同一城市（BEA 经济区）、**不同行业**的公司之间，过去收益能预测未来收益：
      Fama-MacBeth 城市系数 0.061 (t=5.11)；行业 0.243 (t=11.71)（城市约为行业 1/4）
      多空 raw 42bp/月 (t=3.65)，CAPM α 47bp (t=4.16)，FF5 α 42bp (t=3.32)
  **最关键**：效应**对分析师覆盖度完全平坦**（p=0.877）、对规模/成交量也平坦，
      而行业 lead-lag 会随覆盖度从 0.28 崩到 0.098 → **与标准因子近乎正交**。

  ★ 三个必须遵守的口径
    1. **信号用等权，交易组合用市值加权**（信号也市值加权会损失 40-50%）
    2. **必须剔除自身行业**，否则退化成行业动量
    3. 每个 lead 组合至少 10 家（A 股城市规模更小，放宽到 5 并披露）

  已知缺陷：数据源只报**当前**注册地，历史年份存在错分（论文自述 5-10%）。

产出
----
data/company_info.pkl        总部省份/城市/员工数/主营业务文本
data/factors/GEO_LINK.pkl    同城市·不同行业其他公司过去 1 月均值收益（等权）—— 主因子
data/factors/GEO_PROV.pkl    同省·不同行业（省份粒度对照）
data/factors/GEO_CITY.pkl    同城市·全部（不剔行业，对照）
data/factors/GEO_IND.pkl     同行业·跨城市（即行业动量，对照）
"""

import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "data")
FDIR = os.path.join(DATA, "factors")
os.makedirs(FDIR, exist_ok=True)

MIN_PEERS = 5
LOOKBACK = 21


def ts_to_panel(ts_code):
    num, ex = str(ts_code).split(".")
    return {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(ex.upper(), "sz") + num


def fetch_company():
    cache = os.path.join(DATA, "company_info.pkl")
    if os.path.exists(cache):
        c = pd.read_pickle(cache)
        print(f"  [缓存] company_info.pkl {c.shape}")
        return c
    import subprocess
    import tushare as ts
    tok = os.environ.get("TUSHARE_TOKEN") or subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "[Environment]::GetEnvironmentVariable('TUSHARE_TOKEN','User')"],
        capture_output=True, text=True).stdout.strip()
    df = ts.pro_api(tok).stock_company(
        exchange="", fields="ts_code,province,city,office,main_business,employees")
    df["symbol"] = df["ts_code"].map(ts_to_panel)
    for c in ["city", "province"]:
        df[c] = df[c].astype(str).str.strip().replace({"nan": "", "None": ""})
    df.loc[df["city"] == "", "city"] = df.loc[df["city"] == "", "province"]
    df["city"] = df["city"].replace("", "未知")
    df["province"] = df["province"].replace("", "未知")
    df["employees"] = pd.to_numeric(df["employees"], errors="coerce")
    df = df[["symbol", "ts_code", "province", "city", "employees",
             "main_business", "office"]]
    df.to_pickle(cache)
    print(f"  [新建] company_info.pkl {df.shape}  城市数 {df['city'].nunique()}  "
          f"未知 {int((df['city'] == '未知').sum())}")
    return df


def agg_peers(x, codes_a, codes_b=None, min_peers=MIN_PEERS, label=""):
    """对每只股票，求「与自身在 codes_a 同类、且与 codes_b 不同子类」的其他公司的 x 均值。

    用哑变量矩阵乘法一次算完全时段：
        s_A = X@D_A, c_A = M@D_A           同类成员的和与计数
        若给了 codes_b：再按 (codes_a, codes_b) 组合分组的和与计数 s_AB, c_AB
        股票 i 的同侪 = (s_A[a_i] − s_AB[ab_i]) / (c_A[a_i] − c_AB[ab_i])
    注意 codes_b=None 时**不能**用「减去一个常量子类」实现 —— 那样 s_AB≡s_A，
    分子分母同时为 0。必须直接返回 s_A/c_A。
    """
    cols = x.columns
    Xv = np.nan_to_num(x.to_numpy(dtype=np.float32), nan=0.0)
    Mv = x.notna().to_numpy(dtype=np.float32)

    def dum(codes):
        # ★ 必须显式给 index=cols。写成 pd.Series(codes).reindex(cols) 会因
        #   两个索引类型不同（RangeIndex vs 股票代码字符串索引）而全部变成 NaN，
        #   fillna 之后只剩一个类 —— 表现为「1 个主类」且信号全为 NaN。
        cc, uq = pd.factorize(pd.Series(codes, index=cols).fillna("_"))
        D = np.zeros((len(cols), len(uq)), dtype=np.float32)
        D[np.arange(len(cols)), cc] = 1.0
        return D, cc.astype(np.int64), len(uq)

    D_a, ci_a, n_a = dum(codes_a)
    s_a = Xv @ D_a
    c_a = Mv @ D_a
    if codes_b is None:
        num = s_a[:, ci_a]
        den = c_a[:, ci_a]
        print(f"    {label}: {n_a} 个类（不剔子类）", flush=True)
    else:
        comb = (pd.Series(codes_a, index=cols).fillna("_").astype(str) + "|" +
                pd.Series(codes_b, index=cols).fillna("_").astype(str)).to_numpy()
        D_ab, ci_ab, n_ab = dum(comb)
        s_ab = Xv @ D_ab
        c_ab = Mv @ D_ab
        num = s_a[:, ci_a] - s_ab[:, ci_ab]
        den = c_a[:, ci_a] - c_ab[:, ci_ab]
        print(f"    {label}: {n_a} 个主类 / {n_ab} 个子类（已剔子类）", flush=True)

    # 剔除自身：同侪均值中把公司自己减掉
    own = x.to_numpy(dtype=np.float32)
    own_valid = np.isfinite(own)
    num = num - np.where(own_valid, np.nan_to_num(own, nan=0.0), 0.0)
    den = den - own_valid.astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den >= min_peers, num / np.maximum(den, 1e-9), np.nan)
    return pd.DataFrame(out, index=x.index, columns=cols, dtype=np.float32)


def main():
    t0 = time.time()
    print("=" * 92)
    print("构建地理联动因子（Parsons-Sabbatucci-Titman 2020）")
    print("=" * 92)

    print("\n[1] 公司总部信息")
    comp = fetch_company()
    cu = comp.drop_duplicates("symbol").set_index("symbol")
    city = cu["city"]
    prov = cu["province"]

    print("\n[2] 载入面板并构造滞后信号")
    from ashare_factor_strategy import load_panel, Panels, build_universe, CFG
    df = load_panel()
    P = Panels(df)
    del df
    eligible, _, _ = build_universe(P, CFG)
    cols = P.symbols
    city_map = city.reindex(cols).fillna("未知").to_numpy()
    prov_map = prov.reindex(cols).fillna("未知").to_numpy()
    ind_map = P.industry.reindex(cols).fillna("未知").to_numpy()
    print(f"    城市覆盖 {int((pd.Series(city_map)!='未知').sum())}/{len(cols)}  "
          f"城市数 {pd.Series(city_map).nunique()}")

    # ★ 信号必须滞后：用**过去 21 日**均值收益，并把当期置为不可用
    mom = P.ret.rolling(LOOKBACK, min_periods=int(LOOKBACK * 0.7)).mean()
    # 再滞后 1 日，确保调仓日只用已实现信息
    mom = mom.shift(1)
    ret = P.ret
    industry = P.industry
    dates = P.dates
    del P
    print(f"    信号构造完成（{LOOKBACK} 日均值，滞后 1 日）", flush=True)

    print("\n[3] 同侪聚合", flush=True)
    sigs = {}
    sigs["GEO_LINK"] = agg_peers(mom, city_map, ind_map, label="同城市·不同行业")
    sigs["GEO_PROV"] = agg_peers(mom, prov_map, ind_map, label="同省·不同行业")
    sigs["GEO_CITY"] = agg_peers(mom, city_map, None, label="同城市·全部")
    sigs["GEO_IND"] = agg_peers(mom, ind_map, None, label="同行业·跨城市")
    for k, v in sigs.items():
        v.to_pickle(os.path.join(FDIR, k + ".pkl"))
        nz = float(np.isfinite(v.to_numpy()).mean())
        print(f"  [已保存] {k:10s} {v.shape}  有效占比 {nz:.1%}")

    # 与既有因子的相关性（判断是否正交）
    print("\n[4] 与既有代表性因子的截面相关性（时序平均）")
    peers = ["COMMOVE", "LEAD_LAG", "TURN", "MAX5"]
    for nm in peers:
        fp = os.path.join(DATA, "prep", nm + ".pkl")
        if not os.path.exists(fp):
            continue
        q = pd.read_pickle(fp)
        acc, cnt = None, 0
        for dt in sigs["GEO_LINK"].index[::63]:
            a = sigs["GEO_LINK"].loc[dt]
            b = q.loc[dt] if dt in q.index else None
            if b is None:
                continue
            d = pd.concat({"a": a, "b": b}, axis=1).dropna()
            if len(d) < 200:
                continue
            c = d["a"].corr(d["b"])
            acc = c if acc is None else acc + c
            cnt += 1
        if cnt:
            print(f"    GEO_LINK ~ {nm:9s} 平均相关 {acc/cnt:+.3f}  (n={cnt})")
        del q

    with open(os.path.join(DATA, "geo_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"min_peers": MIN_PEERS, "lookback": LOOKBACK, "lag_days": 1,
                   "n_city": int(pd.Series(city_map).nunique()),
                   "city_coverage": int((pd.Series(city_map) != "未知").sum()),
                   "n_stocks": int(len(city_map))}, fh, ensure_ascii=False, indent=2)
    print(f"\n完成，用时 {(time.time()-t0)/60:.1f} 分")


if __name__ == "__main__":
    main()
