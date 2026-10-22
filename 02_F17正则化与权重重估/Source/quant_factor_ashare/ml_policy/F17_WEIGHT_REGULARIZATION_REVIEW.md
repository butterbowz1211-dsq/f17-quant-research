<!-- Research project author: Gilmour Jia. Cited works retain their original authors. -->

# F17 因子权重拟合与正则化：顶刊方法复现与实证结论

日期：2026-10-21
数据：F17 的 17 个已清洗、行业/市值中性化、标准化因子面板；决策日每 21 个交易日；标签为下一交易日开盘买入、21 个交易日后开盘卖出。
开发期：2016-01-01—2022-12-31；验证期：2023-01-01—2026-08-12。
代码：tools/f17_weight_lab.py、tools/f17_weight_deploy.py；结果：artifacts/f17_weight_lab_full_*、artifacts/f17_weight_deploy_*。

## 1. 问题

F17 基线不是拟合模型，而是 17 个因子的等权平均：

- workspace/review_workspace/quant_factor_ashare_review/run_group_research.py:44-62 逐因子累加 base += clean、count += valid，最后 base/count。
- workspace/source_workspace/quant_factor_ashare/ml_policy/run_baseline_f17.py 只读取 score_F17.pkl，不估计权重。
- source_snapshot/ashare_factor_strategy.py:768-842 的中性化是 FWL 行业去均值 + 市值一元残差，没有 ridge；模块注释里的“微小 ridge”与实现不一致。

因此要回答的是：用顶刊的因子权重/收缩方法，能否在样本外稳定超过这个等权先验？

## 2. 文献与实现

### 2.1 Brandt, Santa-Clara & Valkanov (2009, RFS)

组合权重直接参数化为：

w_i,t = wbar_i,t + (1/N_t) θ' xhat_i,t（Eq.3）

xhat 是截面标准化因子，θ 跨股票、跨时间常数。目标函数是组合效用/Sharpe：

max_θ (1/T) Σ u(r_p,t+1)，r_p = Σ w_i r_i（Eq.5-6）

线性 policy 下 θ 就是各因子多空组合的权重（Eq.7）。§2.3.5 明确建议把 θ 向基准收缩；本文实现为在估计的 θ 与等权单位向量之间做 θ_shrink = (1-a) θ + a θ_eq，a 由训练窗内时间折验证 Sharpe 选择。

### 2.2 Kozak, Nagel & Santosh (2020, JFE)

SDF 系数 b 的 naive GMM 解是 b = Σ^-1 μ（Eq.17）。经济先验 μ ~ N(0, (κ²/τ)Σ^η)，取 η=2， posterior：

b = (Σ + γI)^-1 μ（Eq.18-22）

在 PC 空间等价于把第 j 个 PC 的系数乘 d_j/(d_j+γ)，低特征值方向收缩更强。等价惩罚为：

min (μ-Σb)'(μ-Σb) + γ b'Σb（Eq.26）

γ 用 3 折 CV 选，OOS R² 为 1 - (μ2-Σ2b)'(μ2-Σ2b)/(μ2'μ2)（Eq.30）。本文把 17 个因子各自做成 top/bottom 20% 多空因子收益，估计 Σ、μ 后代入上式；γ 用训练窗内 3 折 CV 的因子收益 R² 选。

### 2.3 Freyberger, Neuhierl & Weber (2020, RFS)

先把特征做截面 rank 变换到 [0,1]，再用二次样条：

p1=1, p2=c, p3=c², pk=max(c-t_{k-3},0)²（Eq.4）

自适应 group lasso 两步：

1. min ||y-Xb||² + λ1 Σ_s ||b_s||2（Eq.5），BIC 选 λ1；
2. 权重 w_s=(Σ_k β_sk²)^(-1/2)（Eq.6），解 min ||y-Xb||² + λ2 Σ_s w_s ||b_s||2（Eq.7），BIC 选 λ2；
3. 对选中特征 OLS 重估。

本文实现了线性自适应 lasso（第一段 ridge 初值、特征缩放后 lasso），没有实现 rank+样条，因为当前问题是“权重能否超过等权”，不是“非线性形状是否重要”；样条会引入更多参数，放在下一步。

### 2.4 Ledoit-Wolf 与 DeMiguel 等

Ledoit & Wolf (2004) 的协方差收缩说明：在样本协方差噪声大时，向结构化目标收缩能降低组合权重误差；这支持 ridge/向等权收缩，但也意味着收缩目标的选择比惩罚强度更重要。DeMiguel, Garlappi & Uppal (2009, RFS) 的 1/N 基准说明：参数估计误差大时，等权极难被样本估计权重击败。本文结果与这两个结论一致。

## 3. 实验设计

- 年度 refit，5 年滚动训练窗；训练标签的退出日必须严格早于 1 月 1 日，避免前视。
- 超参只用训练窗内 3 个 expanding-window 时间折选择，验证期 2023-2026 不参与任何选择。
- 因子在每个训练/测试截面内重新标准化；标签做截面 rank 标准化。
- 主评价：逐日 rank IC（HAC(3) t 值）、top/bottom decile 多空（21 日非重叠）、换手与 25bp 单边成本后的净 Sharpe。
- 两种评价域：
  1. complete-case：17 个因子都非缺的股票，和旧 probe 对齐；
  2. full-coverage：每个因子在当日可观测股票内标准化，缺失填 0（截面均值），再加权。这是实盘可部署的打分方式。

## 4. 结果

### 4.1 Complete-case：只有“向等权收缩”的模型略有增益

| 模型 | 开发 IC | ΔIC | 开发 LS Sharpe | 验证 IC | ΔIC | 验证 LS Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| equal | .0945 | — | 1.301 | .1031 | — | .443 |
| OLS | .0920 | -.0025 | 1.563 | .0869 | -.0162 | .208 |
| ridge | .0910 | -.0034 | 1.541 | .1005 | -.0027 | .353 |
| ridge→equal | .0988 | +.0044 | 1.456 | .1056 | +.0025 | .609 |
| lasso | .0930 | -.0015 | 1.627 | .0877 | -.0154 | .211 |
| elastic net | .0938 | -.0007 | 1.541 | .0905 | -.0126 | .334 |
| Kozak SDF | .0875 | -.0069 | 1.498 | .0885 | -.0146 | .428 |
| Brandt policy | .1002 | +.0057 | 1.361 | .1070 | +.0039 | .621 |

只有 ridge→equal 和 Brandt 在开发期、验证期同时提高 IC 与多空 Sharpe。增益幅度是 IC +0.25~0.39 个点、Sharpe +0.17~0.18，不是数量级。OLS、lasso、elastic net、Kozak 在验证期都明显变差；lasso 的开发期 Sharpe 高，但验证期塌到 .211，是典型的过拟合。

选参也说明问题：

- ridge 的 λ 多数年份顶到 1000（2016/2018/2023/2025/2026），即最优是“几乎不要拟合”；
- ridge→equal 的 λ 多数年份也是 1000，2019=1、2021=.03、2022=.01、2023=300、2026=1；λ 越小越接近 OLS，验证期反而更差；
- Brandt 的 shrink 在 2016-2018、2020、2022-2026 都选 1.0，即完全收缩到等权；只有 2019、2021 选 0；
- Kozak 的 γ 多数年份在 .01-.1，但 2016/2025 顶到 100；因子收益的 OOS R² 经常为负，SDF 权重接近噪声。

### 4.2 Full-coverage：增益消失，等权仍是验证期最好的可部署分数

把权重应用到全部合资格股票（缺失因子填 0），并用 25bp 单边成本计算多空净 Sharpe：

| 模型 | 开发 IC | 开发净 Sharpe | 验证 IC | 验证净 Sharpe |
|---|---:|---:|---:|---:|
| 生产 F17（可用因子平均） | .1071 | 1.667 | .1137 | .858 |
| equal | .1061 | 1.583 | .1144 | .942 |
| ridge→equal | .1076 | 1.871 | .1133 | .931 |
| Brandt | .1070 | 1.696 | .1144 | .942 |
| OLS | .1016 | 1.867 | .0997 | .530 |
| lasso | .1030 | 1.783 | .1011 | .292 |
| Kozak | .0989 | 1.615 | .0999 | .723 |

关键变化：

- complete-case 上 ridge→equal 的验证 IC 是 .1056（Δ+.0025），但 full-coverage 上变成 .1133（Δ-.0011，t=-0.86）；
- Brandt 在验证期 full-coverage 与 equal 完全相同（.1144 / .942），因为 2023-2026 的 shrink 全部选 1.0；此时它的权重就是等权，部署打分与等权同序。complete-case 表里 Brandt 的 .1070 来自“同一等权在 complete-case 掩码内重新标准化因子”的排序差异，不是独立的选股能力。
- 开发期 ridge→equal 的净 Sharpe 1.871 高于 equal 1.583，但验证期 .931 低于 equal .942；开发期增益不能外推；
- 生产 F17 的 full-coverage IC 与 equal 接近，但净 Sharpe 略低，说明“可用因子平均”与“完整截面标准化后等权”各有细微差异，等权并不吃亏。

### 4.3 为什么

1. 17 个因子已经 winsorize、行业/市值中性化、标准化；等权本身就是很强的收缩先验。
2. 日截面约 1000 只、5 年窗二十多万观测，线性权重估计误差不大；正则化能带来的偏差-方差改善很有限。
3. 真正有效的“正则化”是向等权收缩，不是向零收缩，也不是 L1 稀疏。ridge→equal 和 Brandt 的增益都来自这一点；一旦 full-coverage，收缩到等权就等于等权。
4. L1/自适应 lasso 在验证期失败，因为因子间高度相关，稀疏解选中的是一组不稳定的代理因子；Kozak 失败，因为 17 个因子多空收益的协方差矩阵噪声大，SDF 权重没有经济约束。
5. 开发期 Sharpe 高但验证期 IC 低的模型（OLS/lasso）是过拟合；不能只看开发期。

## 5. 结论

不能认为拟合权重+正则化能稳健改善 F17。
在 complete-case 评价下，ridge→equal 与 Brandt 有 +0.25~0.39 个 IC 点、+0.17~0.18 Sharpe 的样本外增益，但统计上弱（验证 IC t 约 1.49；Brandt 的 t 2.77 来自 complete-case 重标准化后的排序差异，部署时与等权同序），且一旦按实盘可部署的 full-coverage 方式打分，增益消失，验证期等权反而略好。

因此：

- 不建议把 F17 的等权换成拟合权重；当前证据不支持。
- 如果必须做稳健性版本，用 ridge→equal，λ 取 0.1-1，预期 IC 改善小于 0.005，不要期待 Sharpe 数量级提升；Brandt shrink 在多数年份会选 1.0，等于等权。
- 不要把时间继续花在 F17 权重调参上；同样的精力更适合新因子、组合构建（换手/成本/风险约束）和非线性形状（Freyberger 的 rank+样条）。
- 本结论只针对 F17 的 17 个因子和当前样本；不否定“用 ML 学非线性映射”或“用更多特征做权重学习”。

## 6. 复现

    # complete-case 权重实验
    & 'workspace/source_workspace/PatchTST_A股复现/.venv/Scripts/python.exe' 'workspace/source_workspace/quant_factor_ashare/ml_policy/tools/f17_weight_lab.py' --years 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 --models equal ols ridge ridge_eq lasso enet kozak brandt --out 'workspace/source_workspace/quant_factor_ashare/ml_policy/artifacts/f17_weight_lab_full'

    # full-coverage 部署评价
    & 'workspace/source_workspace/PatchTST_A股复现/.venv/Scripts/python.exe' 'workspace/source_workspace/quant_factor_ashare/ml_policy/tools/f17_weight_deploy.py'

注意：运行前需要 pyarrow 的 bootstrap；tools/f17_weight_lab.py:47-54 已处理。不要用系统 Python，它没有 numpy/torch/pyarrow。

## 7. 文献文件

workspace/source_workspace/quant_factor_ashare/ml_policy/literature/：

- brandt_parametric_portfolio.pdf（47 页）
- kozak_shrinking_cross_section.pdf（58 页）
- freyberger_dissecting_characteristics.pdf（68 页）
- campbell_thompson.pdf（32 页）
- ledoit_wolf_honey.pdf（22 页）
- ledoit_wolf_2004b.pdf（20 页）
- demiguel_naive_2009.pdf（25 页）