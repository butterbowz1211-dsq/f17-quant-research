<!-- Research project author: Gilmour Jia. Cited works retain their original authors. -->

# BaoStock 日级状态补充（supplement_v1）

本文件记录 `supplement.py` 的真实覆盖、接入方式和验收边界。它不是执行验收报告；执行验收仍以 `execution_accepted=false` 和根目录 `acceptance.json` 为准。

## 目的

原 `source_union_v1` 只有月末 ST 快照，且缺少数证券的上市/退市日期。BaoStock 独立补充日级 `isST`、`tradestatus` 和原始价量，用于降低 ST 误判，不修改旧源，也不把未知状态填成 `False`。

## 产物

- `derived/supplement_v1/security_master.parquet`：BaoStock `query_stock_basic` 主表，`type=1` 股票参与下载。
- `bars/<symbol>.parquet`：2010-01-01—2026-09-28 的单只日线；字段 `date,code,raw_open,raw_high,raw_low,raw_close,raw_pre_close,raw_volume,raw_amount,turnover,is_st,daily_status`。
- `by_year/<year>.parquet`：供 `prepare_research.py --supplement` 读取的年度分区。
- `coverage.csv`：逐证券行数、首末日期、停牌/ST/未知天数、`status=ok/empty/not_downloaded`。
- `manifest.json`：字段、行数、年度文件 SHA256、状态计数、`complete`、`execution_accepted=false`。
- `download_ledger.jsonl`：逐证券成功/空/错误/跳过记录，用于断点续传和重试。

## 覆盖口径

`ok` 表示已落盘且至少一行；`empty` 表示供应商在请求窗口明确无行情（退市后无数据），不等于“缺失已补齐”；`not_downloaded` 表示尚未成功请求，不能当作通过。历史最后日期早于 2026-09-28 是合法结果，不能按丢页递归切分。

## 当前状态（2026-10-21 11:17）

用户决定改为采购数据后，全量 BaoStock 下载已停止。当前 `derived/supplement_v1` 是**部分目录**：

- 主表 5,563 只，但仅 361 只有 bars（含此前跳过/已完成的证券）；按最新状态去重为 ok 195、empty 9、skipped 166，其余 5,193 只未下载。
- 尚未运行全量 `build`，因此没有全量 `coverage.csv`/`manifest.json`；不得把该目录当作全市场补充。
- `derived/research_training_v3` 使用 `supplement_smoke`（仅 4 只）做接口验证，158 个形成日、357,883 行、346,742 个有效标签，`execution_accepted=false`；它不是正式训练包。
- 供应商数据要求见 `DATA_VENDOR_REQUIREMENTS.md`。

## 接入 prepare_research

`prepare_research.py` 的 `supplemental()` 把 supplement 原始表追加到来源优先级最后，按 `(date,symbol)` 覆盖同日字段；`metadata()` 用 supplement 主表补上市/退市日期。规范化层新增 `st_source`：

| `st_source` | 含义 |
|---|---|
| 1 | 该行有日级 `isST` |
| 2 | 该行仍使用月末快照 |
| 0 | ST 状态未知 |

`quality_report.json` 记录 `eligible_st_source`、`eligible_rows` 和 `supplement` 摘要（含 `complete`）。缺失日级状态不会填成 `False`。

## 验收边界

- 本补充不生成 `market.parquet`/`actions.json`，不标记 `execution_accepted=true`。
- 没有公司行动台账、退市终值结算和分红税假设；退市/并购样本的标签仍需单独处理。
- `empty` 和 `not_downloaded` 都不能在执行验收中当作完整覆盖。
- `adjustment_verified` 仍为 false；原始价量不等于可执行复权价。

## 测试

- `test_supplement.py`：7/7 通过（data_mcp venv）。
- `test_nn_v2.py` + `test_research.py`：37/37 通过（训练 venv）。
