<!-- Research project author: Gilmour Jia. Cited works retain their original authors. -->

# 独立 A 股因子研究复核

先读 [审计报告](审计报告.md) 和 [联动因子第二轮研究](联动因子第二轮研究.md)。全部新文件位于本目录，原 `source_workspace/quant_factor_ashare` 只读。第二轮使用 Review 自带 Python 3.12，并在本目录 `.deps` 安装 pyarrow；不依赖 Tushare。

## 使用

在 PowerShell 进入本目录后：

```powershell
$quantPython = 'workspace/user_home/.cache/research-runtimes/research-primary-runtime/dependencies/python/python.exe'
$env:PYTHONPATH = "$PWD/.deps"
$env:PYTHONIOENCODING = 'utf-8'
& $quantPython -m unittest -v test_robust_factor_core test_linkage_research
& $quantPython run_research_review.py
```

下方旧示例中的 `python` 可替换为 `& $quantPython`。

第二轮使用 `run_group_research.py` 生成四组候选分数，`run_group_execution.py` 执行共同设定；严格模式遇陈旧估值会停止。历史诊断结果位于 `linkage_round2/execution_diagnostic`。外部联动计算入口为 `build_linkage_signals.py`，必须先提供符合其模式的关系与收益数据，当前披露下载不能直接冒充已完成的 PIT 输入。

`download_public_disclosures.py` 下载公开研报和分部，`normalize_public_disclosures.py` 输出标准化快照；`audit_disclosure_samples.py` 保存 PDF 样本。巨潮分区下载任务及日志在 `disclosure_harness_task.txt`、`linkage_round2/disclosures/harness_events.jsonl`。应查看下载状态和 `normalized/quality_report.json`，不能仅凭存在 CSV 判断历史完整。

巨潮正式入口是 `download_cninfo_partitioned.py`：日期二分、排序方向补集、父子区间对账，未对齐的区间保持 incomplete。DeepSeek 生成初稿后由主代理修正完整性传递、续传状态和并发保存，并增加 `test_disclosure_partitions.py` 的 5 项测试。样本 2011-01/02 已校验 433 个唯一公告。`finish_disclosures.py` 续传全范围并在结束后整理；`disclosure_status.py` 只读显示进度。下载日志在 `linkage_round2/disclosures/full_download.log`；`complete_requested_window` 只表示请求子窗口完成，`complete` 才表示整个预设日期范围完成。

默认是严格模式：遇到持仓超过 60 个交易日没有估值会停止。现有源数据会触发这个检查；这是已知数据问题，不应通过删除检查伪装修好。

仅重现本次研究诊断情景（明确允许持仓沿用陈旧估值）：

```powershell
python run_research_review.py --allow-stale
python audit_all_factors.py
python audit_risk_model.py
```

结果写入本目录 `results`；再次运行会覆盖同名诊断结果，原文件不受影响。`audit_risk_model.py` 依赖主重跑产生的收益 CSV。运行实盘前不应使用 `--allow-stale` 绕过数据修复。

可用 `python run_research_review.py --help` 查看源路径、相位和子集参数。默认读取原目录的大面板与因子 pickle 缓存，请只读取受信任的本地 pickle 文件。全量计算需要足够内存，耗时因设备而异。

## 文件

|文件|用途|
|---|---|
|robust_factor_core.py|共同样本 RankIC、训练标签 purge、HAC、FDR、bootstrap、DSR 诊断、月份回归及事件回测|
|run_research_review.py|冻结 F17 分数，复现旧回测并比较新基线与权重缓冲候选|
|audit_all_factors.py|23 个不重复因子的统一检验；不代表全部历史搜索族|
|audit_risk_model.py|月度索引与超额收益修正，沿用并明确标注原风险因子的局限|
|test_robust_factor_core.py|19 个针对性单元测试；从冻结副本抽取少量纯函数复现旧错误|
|results/review_results.json|重跑配置、持仓诊断、配对检验与主要指标|
|results/performance.csv|各相位及各窗口绩效|
|results/factor_family23.csv|因子 IC、HAC21/63、BH/BY|
|results/risk_model_audit.json|相同收益序列的回归口径对照|
|source_manifest.json|审计开始时原文档/代码等文件的 SHA256|
|source_verification.json|交付前源文件哈希复核|

不要把 `source_snapshot` 中的旧脚本作为新引擎启动。新引擎保留旧缓存信号用于控制比较，没有声称修复所有因子构造。它仍使用复权价格单位与分数股、近似交易限制、固定统计 rf；没有实现精确公司行动与开盘竞价账本。

`family_max_mean_test` 是固定、完整候选族的共同区块重采样诊断，不是完整 SPA 实现；`dsr_iid` 明确为 IID 诊断。没有足够历史试验记录时，不应把它们输出解释成对所有策略搜索已完成校正。
