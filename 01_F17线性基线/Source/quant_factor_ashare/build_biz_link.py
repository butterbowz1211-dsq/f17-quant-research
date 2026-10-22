# -*- coding: utf-8 -*-
# Research project author: Gilmour Jia
"""
build_biz_link.py — 业务相似度联动因子（Lee-Sun-Wang-Zhang 2019 的可行代理）
================================================================================
文献依据
--------
Lee, Charles M.C., Stephen Sun, Rongfei Wang & Ran Zhang (2019),
"Technological Links and Predictable Returns", JFE 132(3), 76-96.

  论文用**专利类别分布重叠**（Jaffe 1986 的 427 个 USPTO 类别的非中心相关）
  构建技术联动网络，再按联动公司的过去收益排序：
      等权多空 117bp/月 (t=5.47) = 14.1%/年；市值加权 69bp (t=3.19)
      Fama-MacBeth 系数 0.643 (t=4.18)，对规模/BM/盈利/资产增长/R&D/反转/动量
      以及**行业动量**均稳健
  **关键设计结果**：TECH 加权 1.17% > 等权 1.07% > 市值加权 0.39%
      —— 相似度加权本身在起作用，不是简单平均。

本实现
------
我们没有专利数据，但有 tushare `stock_company` 的 **main_business（主营业务描述文本）**。
用它计算**字符二元组余弦相似度**作为技术/经济关联的代理：

    sim(i,j) = |Bg(i) ∩ Bg(j)| / sqrt(|Bg(i)| · |Bg(j)|)      Bg = 字符二元组集合

每家股票取相似度最高的 K 家作为「联动同伴」（剔除自身），行归一化得到联动权重矩阵 P，
信号 = P @ (过去 21 日均值收益)。

两个版本
  BIZ_LINK    全部前 K 家同伴（对应 Lee et al. 原口径）
  BIZ_LINK_X  前 K 家中**剔除同行业**（更可能与既有行业类因子正交）

  ⚠️ 严重的前视局限（必须披露，且这是本因子结果不可全信的原因）
  数据源只提供**当前**主营业务描述。用它定义 2010 年的联动网络，等于用「后来才
  披露的业务调整」去构建当时的网络。若某公司在 2015 年转型，其 2010 年的联动
  关系会被按 2026 年的文本确定 —— 这是真实的前视。
  因此本因子的任何正向结果都必须用**时点文本**（如历年年报的「主营业务」段落）
  重新验证后才能采信。本脚本的作用是**先判断有没有信号**，而不是直接采用。
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "data")
FDIR = os.path.join(DATA, "factors")
os.makedirs(FDIR, exist_ok=True)

TOPK = 20
LOOKBACK = 21
MIN_CHARS = 8


def bigrams(txt):
    t = "".join(ch for ch in str(txt) if "\u4e00" <= ch <= "\u9fff")
    if len(t) < MIN_CHARS:
        return set()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def main():
    t0 = time.time()
    print("=" * 96)
    print("构建业务相似度联动因子（Lee et al. 2019 代理）")
    print("=" * 96)

    comp = pd.read_pickle(os.path.join(DATA, "company_info.pkl"))
    comp = comp.drop_duplicates("symbol")

    from ashare_factor_strategy import load_panel, Panels, build_universe, CFG
    df = load_panel()
    P = Panels(df)
    del df
    eligible, _, _ = build_universe(P, CFG)
    cols = P.symbols

    c = comp.set_index("symbol").reindex(cols)
    txt = c["main_business"].fillna("").astype(str)
    print(f"  股票 {len(cols)}  有业务描述 {int((txt.str.len() >= MIN_CHARS).sum())}")

    # ---- 字符二元组稀疏矩阵 ----
    print("\n[1] 构建字符二元组矩阵", flush=True)
    grams = [bigrams(t) for t in txt]
    vocab = {}
    rows, colsi = [], []
    for i, g in enumerate(grams):
        for bg in g:
            j = vocab.get(bg)
            if j is None:
                j = len(vocab); vocab[bg] = j
            rows.append(i); colsi.append(j)
    n, B = len(grams), len(vocab)
    X = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, colsi)),
                      shape=(n, B))
    nnz_per = np.asarray(X.getnnz(axis=1)).ravel()
    print(f"    二元组词表 {B}  矩阵 {X.shape}  非零 {X.nnz}  "
          f"每股平均 {nnz_per.mean():.1f} 个二元组")
    del grams, rows, colsi

    # ---- 逐行取 top-K 相似（避免 5221×5221 稠密矩阵）----
    print(f"\n[2] 逐股计算相似度并取前 {TOPK} 名同伴", flush=True)
    # ★ CSR 的 indptr 必须恰好 n+1 个元素：indptr[0]=0，之后每处理完一行
    #   就 append 当前的 len(indices)。首版在空行分支里 append(0)，导致
    #   indptr 长度只有 n → scipy 报 "index pointer size 5221 should be 5222"。
    indptr, indices, data = [0], [], []
    for i in range(n):
        if nnz_per[i] == 0:
            indptr.append(len(indices))
            continue
        sim = (X[i] @ X.T).toarray().ravel()          # |交集|（二元向量内积）
        denom = np.sqrt(nnz_per * max(nnz_per[i], 1))
        sim = sim / np.maximum(denom, 1e-9)
        sim[i] = -1.0                                  # 剔除自身
        k = min(TOPK, max(int((sim > 0).sum()), 1))
        idx = np.argpartition(-sim, k - 1)[:k]
        idx = idx[sim[idx] > 0]
        if idx.size:
            v = sim[idx]
            v = v / v.sum()
            indices.extend(idx.tolist())
            data.extend(v.tolist())
        indptr.append(len(indices))
        if (i + 1) % 1000 == 0:
            print(f"      {i+1}/{n}  {time.time()-t0:.0f}s", flush=True)
    Pm = sp.csr_matrix((np.array(data, dtype=np.float32), np.array(indices),
                        np.array(indptr)), shape=(n, n))
    print(f"    联动矩阵 {Pm.shape}  非零 {Pm.nnz}  平均每行 {Pm.nnz/n:.1f}")
    del X

    # ---- 剔除同行业的版本 ----
    ind_codes, _ = pd.factorize(P.industry.reindex(cols).fillna("未知"))
    same = sp.csr_matrix(
        (np.ones(n, dtype=np.float32), (np.arange(n), ind_codes)), shape=(n, n))
    # 逐行剔除同行业同伴后重新归一化
    print("\n[3] 构建剔除同行业的版本", flush=True)
    indptr2, indices2, data2 = [0], [], []
    Pc = Pm.tocsr()
    for i in range(n):
        s = Pc.indices[Pc.indptr[i]:Pc.indptr[i + 1]]
        v = Pc.data[Pc.indptr[i]:Pc.indptr[i + 1]]
        m = ind_codes[s] != ind_codes[i]
        keep, kv = s[m], v[m]
        if keep.size:
            kv = kv / kv.sum()
            indices2.extend(keep.tolist()); data2.extend(kv.tolist())
        indptr2.append(len(indices2))
    Pmx = sp.csr_matrix((np.array(data2, dtype=np.float32), np.array(indices2),
                         np.array(indptr2)), shape=(n, n))
    print(f"    剔除同行业后 非零 {Pmx.nnz}  平均每行 {Pmx.nnz/n:.1f}")

    # ---- 信号 = P @ 过去 21 日均值收益 ----
    print("\n[4] 计算联动信号", flush=True)
    mom = P.ret.rolling(LOOKBACK, min_periods=int(LOOKBACK * 0.7)).mean().shift(1)
    M = np.nan_to_num(mom.to_numpy(dtype=np.float32), nan=0.0)      # n × T -> 转置
    V = np.isfinite(mom.to_numpy(dtype=np.float32)).astype(np.float32)
    for nm, PM in [("BIZ_LINK", Pm), ("BIZ_LINK_X", Pmx)]:
        # num[t,i] = Σ_j PM[i,j]·M[t,j] = (M @ PMᵀ)[t,i]
        # M 是 (T × n)，PM 是 (n × n)，故用 M @ PM.T，避免转置混乱。
        num = M @ PM.T                         # T × n
        den = V @ PM.T
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)
        out = out * eligible.to_numpy()        # 只保留可交易股票
        F = pd.DataFrame(out.astype(np.float32), index=mom.index, columns=cols)
        F.to_pickle(os.path.join(FDIR, nm + ".pkl"))
        print(f"  [已保存] {nm}  {F.shape}  有效占比 {float(np.isfinite(out).mean()):.1%}")

    # ---- 与既有因子相关性 ----
    print("\n[5] 与既有代表因子的截面相关性（时序平均）")
    for nm in ["BIZ_LINK", "BIZ_LINK_X"]:
        F = pd.read_pickle(os.path.join(FDIR, nm + ".pkl"))
        for peer in ["COMMOVE", "LEAD_LAG", "IND_MOM", "TURN"]:
            fp = os.path.join(DATA, "prep", peer + ".pkl")
            if not os.path.exists(fp):
                continue
            q = pd.read_pickle(fp)
            acc, cnt = None, 0
            for dt in F.index[::63]:
                if dt not in q.index:
                    continue
                d = pd.concat({"a": F.loc[dt], "b": q.loc[dt]}, axis=1).dropna()
                if len(d) < 200:
                    continue
                cc = d["a"].corr(d["b"])
                acc = cc if acc is None else acc + cc
                cnt += 1
            if cnt:
                print(f"    {nm:11s} ~ {peer:9s} {acc/cnt:+.3f}  (n={cnt})")
            del q
        del F
    print(f"\n完成，用时 {(time.time()-t0)/60:.1f} 分")


if __name__ == "__main__":
    main()
