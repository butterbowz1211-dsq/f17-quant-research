<!-- Research project author: Gilmour Jia. Cited works retain their original authors. -->

# 复现指南

## 环境

本机可用的解释器是 `PatchTST_A股复现/.venv`（Python 3.12，torch 2.11.0+cu128）。项目缓存是
pandas 3 / pyarrow 序列化，需要把审计目录里的 pyarrow 放进 `PYTHONPATH`：

```powershell
$env:PYTHONPATH='workspace/review_workspace/quant_factor_ashare_review/.deps'
$env:PYTHONUNBUFFERED="1"
$py="workspace/source_workspace/PatchTST_A股复现/.venv/Scripts/python.exe"
```

本机没有 pytest，测试用自带运行器：

```powershell
& $py tools/run_tests.py            # 全部
& $py tools/run_tests.py test_policy
```

## 第一步：数据准备（一次性，约 5 分钟）

`qlp/data.build_prep()` 把 13M 行面板旋转成日期×证券矩阵并缓存到 `artifacts/prep_arrays.npz`。
这一步是纯数据准备，不含标签、不含模型选择，所以跨折复用不会泄漏。

## 第二步：训练与评估

```powershell
& $py run_policy.py --tag p1 --folds 3 4 --seeds 0 1 2 --epochs 40 --patience 12 --lr 5e-3 --batch 16 --rebalance 21 --temperature 1.0 --temp-end 0.1 --cap 0.03 --l2 1e-3 --dropout 0.10 --partial 1.0 --band 0.0 --out artifacts/p1_report.json
```

关键参数的含义：

| 参数 | 含义 |
|---|---|
| `--rebalance 21` | 决策间隔（交易日）；标签也取这一段 |
| `--temperature` / `--temp-end` | 温度退火起止值，控制持仓广度 |
| `--top-k K` | 硬性最多 K 个持仓（含现金）；0 表示不限制 |
| `--cap` | 单票上限，训练与硬投影使用同一上限 |
| `--partial` | 部分调整系数，1.0 表示完全调到目标 |
| `--band` | 不交易带（占净值比例），0 表示关闭 |
| `--l2` / `--dropout` | 编码器正则 |

输出：`artifacts/<tag>_report.json`、`artifacts/<tag>_fold<k>_carry.npy`（目标权重网格）、
`artifacts/<tag>_fold<k>_daily.csv`（测试期逐日账本）。

## 折的定义

`qlp/splits.expanding_folds` 从数据首年开始扩窗：6 年训练 / 2 年验证 / 1 年测试。
注意它是按面板首年（2010）锚定的，所以 **fold0 的测试年是 2018**；测试年 2021 在 fold3、
2022 在 fold4。用 `--folds 3 4` 跑真正的未污染样本外。

## 汇报口径

- 训练用的是可微代理目标，它从不作为业绩数字汇报；
- 所有净值、回撤、换手、成本都来自 `qlp/execution.run_target_weight_engine`，其中含整手、
  涨跌停不可成交、停牌、1% 参与率上限、佣金、印花税（2023-08-28 前后不同）与双向滑点；
- 多随机种子在目标权重层平均，然后只跑一次账本，避免成本被重复计提；
- 2023 年及以后的结果一律标注 `CONTAMINATED`，只能当探索，不能当样本外。
