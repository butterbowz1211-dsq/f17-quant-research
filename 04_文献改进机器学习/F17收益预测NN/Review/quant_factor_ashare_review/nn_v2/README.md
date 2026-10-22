<!-- Research project author: Gilmour Jia. Cited works retain their original authors. -->

# 浅层 NN v2：独立代码交付

**2026-10-21 当前结论：v4（17 因子、34 维）是唯一保留的 NN 研究主线，但没有显著战胜 F17；v5（加 EP/BP/SP/DIV、42 维）和估值 overlay 都是负结果，已归档。** v4 NN 的 58 日 top50 为 23.07%/Sharpe 1.014/-24.44%，F17 为 20.17%/1.084/-19.35%；net25 为 19.92%/0.895 对 17.83%/0.971。v5 的 IC 从 0.1167 降到 0.1107（t -3.28）；overlay 的 2023–2025 top50 从 21.76% 降到 13.75%。详见 [最终报告](最终报告_2026-10-21.md) 与 [最终结论](最终结论.md)。

**当前状态：v4 与 v5 各 45 次 NN、8 次 Ridge 全部完成，0 failed；37 项单元测试通过。** 当前主模型是 v4；v5 与 overlay 不采用。严格数据/执行验收仍未通过，execution_accepted=false；所有收益都是研究读数，不是实盘结果。

2026-10-21（Asia/Shanghai）。本目录是新的 NN 研究入口，原项目、原 F17 因子线性打分策略及旧 NN 均未覆盖。原文件的逐项 SHA256 验证见 `audit/original_code_verification.json`。

先读：

- [最终报告（2026-10-21）](最终报告_2026-10-21.md)
- [最终结论](最终结论.md)
- [研究训练放行说明](研究训练放行说明.md)
- [MCP 安装与补数验收](MCP安装与补数验收.md)
- [AkShare 实测与可用性](AkShare接口实测.md)
- [数据与执行审计](数据与执行审计.md)
- [实验协议](实验协议.md)
- [因子字典](因子字典.md)
- [费用与交易规则来源](费用与交易规则.md)

## 环境与运行

现有训练 Python：

`workspace/source_workspace/PatchTST_A股复现/.venv/Scripts/python.exe`

实测 Python 3.12.14、PyTorch 2.11.0+cu128、CUDA 可用、pandas 3.0.6、numpy 2.5.4。`runtime.py` 只读引用已有 pyarrow 26.0.0 补充库和 DLL 搜索目录，不安装或改写原环境。

在 PowerShell 中运行：

```powershell
Set-Location 'workspace/review_workspace\quant_factor_ashare_review\nn_v2'
$nnPython = 'workspace/source_workspace\PatchTST_A股复现\.venv\Scripts\python.exe'
& $nnPython -m unittest test_nn_v2 test_research -v
& data_mcp/.venv/Scripts/python.exe -m unittest test_supplement.py -v
```

已有 CUDA 烟雾检查结果在 `engineering/smoke_result.json`。其 checkpoint 文件名明确标注 `SYNTHETIC_ONLY`，只用于检查梯度、设备、保存和重载，不能用于金融预测。

## 文件职责

| 文件 | 职责 |
|---|---|
| `data.py` | 精确 t+1 / t+22 标签、标签到期过滤、median/缺失标记、缓存签名、来源合并 |
| `factors.py` | 独立实现共同的 17 个原 F17 因子定义，未导入会写缓存的原 F17 模块 |
| `prepare.py` | 从已验证源数据重建因子、股票池、样本及成交输入；保留清洗前后因子 |
| `supplement.py` | 独立 BaoStock 全市场日级 ST/交易状态补充；断点续传、年度分区、覆盖台账，不写执行验收 |
| `model.py` | v4 34→32→16→1；v5 42→32→16→1（由 research_ext/train_v5.py 内存 patch）；Huber、日期等权、AdamW、完整 bundle、Ridge 对照 |
| `train.py` | v4/v5 各 45 次 NN + 8 次 Ridge；首折 6 候选×3 seed、年度 final；已完成任务可校验后重用。v1 的 51/9 为历史预算 |
| `portfolio.py` | 排名缓冲、固定秩映射 softmax、等权诊断、上限分配；本报告主表未使用这条读数 |
| `execution.py` | 原始价格股数账本、逐笔费用、受限成交、现金/应收款/锁定股份 |
| `metrics.py` / `evaluate.py` | 标准 Sharpe、HAC、配对区块区间、连续年度账本和门槛 |
| `audit_sources.py` / `repair_sources.py` | 原数据只读审计、独立来源并集 |
| `probe_akshare.py` / `analyze_akshare.py` | 有限接口探测、证券名单匹配与候选事件整理 |
| `research_ext/quick_readout.py` | 每日 top50 等权、22 日标签、rank IC/Sharpe/回撤读数 |
| `research_ext/build_v5_package.py` | 从 v4 标签 + 估值快照构建 42 维 v5 包，不改顶层源码 |
| `research_ext/train_v5.py` | 内存 patch 42 维后调用 train.run，不修改冻结代码 |
| `research_ext/overlay_value_v4.py` | 只读 v4 预测的估值 overlay 检验 |
| `experiment_spec.json` | 代码协议冻结；严格数据验收仍阻塞，v4/v5 按 research_with_limitations 放行 |

## 已实际生成的数据

`derived/source_union_v1/` 包含 2010—2026 共 17 个年度 Parquet，合计 14,428,034 行，约 900 MB。它保留 Tushare 独有证券、未复权 OHLC、来源标记、ST 快照日期及未知状态。缺失的复权价、股本和公司行动没有编造；`adjustment_verified=False`，明确 **NOT_ACCEPTED**。它是排错中间表，不是 v4/v5 的直接训练输入。

`derived/research_training_v4/`：17 因子 + 17 缺失标记 = 34 维；150 个形成日（2013-02-19 至 2025-12-31），342,390 行、4,493 个证券、327,663 条有效标签。

`derived/research_training_v5/`：与 v4 共用 y、formation_time、symbols；新增 EP/BP/SP/DIV 及其缺失标记，42 维。

`derived/research_model_v4/` 与 `derived/research_model_v5/`：各 45 次 NN、8 次 Ridge 全部完成；含 `predictions_2021..2025.csv`、`experiment_ledger.jsonl`、`run_manifest.json` 和模型 bundle。

`research_ext/overlay_v4/`：overlay 的 `summary.json`、`daily_overlay.csv`、`lambda_grid.csv`、`yearly_overlay.csv`、`top50_composition.csv`。

v1 的 51/9、65 日结果仍保留在 `analysis/research_v1/`，仅作历史，不与 v4/v5 混表。

## 当前研究复现（v4/v5/overlay）

v4 训练（17 因子、34 维）：

```powershell
& $nnPython train.py --package derived/research_training_v4 --out derived/research_model_v4 --device cuda --mode research_with_limitations --last-year 2025
```

v5 构建与训练（42 维，不改顶层源码）：

```powershell
& $nnPython research_ext/build_v5_package.py --source derived/research_training_v4 --output derived/research_training_v5
& $nnPython research_ext/train_v5.py --package derived/research_training_v5 --out derived/research_model_v5 --device cuda --mode research_with_limitations --last-year 2025
```

overlay（只读 v4 预测，不重训）：

```powershell
& $nnPython research_ext/overlay_value_v4.py --out research_ext/overlay_v4
```

`train.py` 会检查 ledger 并复用已完成任务；v4/v5 的 45 次 NN、8 次 Ridge 是内部调度，不是手工逐次命令。当前研究读数统一走 `research_ext/quick_readout.py` 的每日 top50 等权口径，不使用 `portfolio.py` 的 softmax/缓冲读数。

## 补齐数据后的正式入口

先按 `数据接口契约.md` 提供独立的验证源目录，再执行以下阶段。路径为参数，不会写入旧项目：

```powershell
& $nnPython prepare.py --source '完整验证源目录' --output '新的样本包目录'
& $nnPython train.py --package '通过验收的样本包目录' --out '新的实验目录' --device cuda
& $nnPython evaluate.py --package '通过验收的样本包目录' --runs '实验目录' --output '新的评估目录'
```

`prepare.py` 输出的验收状态默认 `REVIEW_REQUIRED`。审核必须有七类对应的本地证据文件和哈希，不能只将一个开关改成 PASS。`train.py` 检查关卡后才启动拟合，禁止静默加载旧缓存。已完成任务只会校验重用；失败或中断任务需要显式调查、保留原台账并记录恢复方案，不会自动改参数重试。

新增实验或修复使冻结的代码/数据签名改变时必须另建实验版本。当前总入口是独立 NN v2；原 `ml_policy/run_policy.py` 仍是旧模型入口。

## AkShare 环境

为实际验证接口，在本目录 `.akshare-env/` 中隔离安装了 AkShare 1.19.1，未改变训练环境。原始查询 CSV/JSON 保存在 `akshare_probe/` 和 `akshare_probe_retry1/`。

```powershell
& .akshare-env/Scripts/python.exe probe_akshare.py
& $nnPython analyze_akshare.py
```

探测结果有缓存；原请求失败和重试均保留。重新探测应指定新的 `--out`，避免混淆不同抓取日期。未创建自动监控，没有实盘订单。
