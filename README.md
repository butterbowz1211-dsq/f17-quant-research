# F17 Quantitative Research

**Author and research lead: Gilmour Jia**

An evidence-based comparison of an equal-weight F17 signal, regularized factor weights, direct neural portfolio policies, and literature-inspired machine-learning variants in the Chinese A-share market. A separate neural-network branch predicts returns from F17 features before constructing portfolios.

本项目整理一条从透明基准到正则化和机器学习的量化研究路径。研究重点包括模型设计、参数依据、训练与验证安排，以及预测信息如何转化为组合表现。所有结果保留股票池、样本期间、费用和评价频率的差异；现有实验尚未证明复杂模型具有稳定、可实施的超额收益。

## Research contributions

- Established an interpretable baseline using the available-factor mean of 17 standardized characteristics, with explicit portfolio and execution rules.
- Compared equal weights with OLS, Ridge, shrinkage toward equal weights, Lasso, Elastic Net and literature-inspired coefficient estimators.
- Preserved two distinct learning routes: direct stock/cash allocation and supervised F17 return prediction.
- Examined chronological training, ranking objectives, temperature, portfolio truncation, turnover controls and valuation-feature extensions, including negative results.
- Reconciled saved returns with standard Sharpe, maximum drawdown and Calmar calculations, and documented label-timing, missing-label selection and execution limitations.

See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for the Chinese contribution statement and its evidence boundaries.

## Contents

| Directory | Content |
| --- | --- |
| [00_论文](00_论文/) | Chinese manuscript, bilingual abstracts, tables and reference verification |
| [01_F17线性基线](01_F17线性基线/) | Equal-weight F17 and execution diagnostics |
| [02_F17正则化与权重重估](02_F17正则化与权重重估/) | Factor-coefficient estimation and shrinkage experiments |
| [03_原始机器学习组合政策](03_原始机器学习组合政策/) | Original direct portfolio-policy experiments |
| [04_文献改进机器学习](04_文献改进机器学习/) | Policy v2/v3 and the separate F17 return-prediction branch |
| [05_研究文档与文献笔记](05_研究文档与文献笔记/) | Literature notes and background research |
| [00_说明与核验](00_说明与核验/) | Anonymized provenance, metric checks and dependency lists |

## Selected findings

| Comparison, within its own protocol | Result | Interpretation |
| --- | --- | --- |
| Equal-weight vs Ridge toward equal, 42 validation periods, 25 bp name-turnover scenario | Sharpe 0.942 vs 0.931 | Shrinkage did not outperform the matched equal-weight control in this diagnostic |
| F17 vs NN softmax, v1 daily gross returns, 2021–2026 | Sharpe 1.143 vs 0.841; MDD −24.63% vs −33.53% | Greater model flexibility did not improve this branch's daily risk-adjusted result |
| v4 NN vs matched F17, 58 periodic gross observations | CAGR 23.07% vs 20.17%; Sharpe 1.014 vs 1.084 | Higher return coincided with weaker risk-adjusted statistics; future-label availability affects selection |

Rows are separate experimental protocols and must not be ranked across tables. Period-end drawdown does not measure intraperiod daily drawdown. The 2023–2026 policy holdout tests the original p1 recipe, not every v2/v3 modification.

## Reproduce saved daily metrics

Python 3, standard library only:

```bash
python "00_说明与核验/verify_archived_daily_metrics.py"
```

This verifies 73 archived daily-return series without executing strategy code. Full retraining requires external market data, compatible dependencies and path configuration. Historical scripts use neutral `workspace/review_workspace`, `workspace/source_workspace` and `workspace/user_home` locations after privacy cleaning; they require configuration before use. No model was retrained for this release.

## Authorship, provenance and privacy

Gilmour Jia is the project author and research lead. AI tools assisted implementation, file organization, metric verification, literature checking and manuscript preparation. Cited papers, third-party methods and any retained upstream notices keep their original attribution. Method names do not imply that the cited authors endorsed this adaptation.

This release removes local account paths and private conversation excerpts, and standardizes document metadata. See [PRIVACY.md](PRIVACY.md). Raw market data, account configuration and credentials are not distributed. The original historical implementation is archived for inspection; known limitations are documented rather than silently presented as corrected experiments.

The repository does not grant rights to third-party literature or data. No additional open-source license is assigned in this release.
