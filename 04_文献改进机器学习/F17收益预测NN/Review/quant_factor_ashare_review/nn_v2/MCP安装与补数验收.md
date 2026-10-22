<!-- Research project author: Gilmour Jia. Cited works retain their original authors. -->

# MCP 安装、实际补数与训练准入

本页记录此前“严格验收后再训练”的阶段。用户随后明确修改要求，本次已继续补数、预处理并启动带局限的研究训练；最新结果见 [研究训练放行说明](研究训练放行说明.md)。以下的 0/51 是历史时点状态。

2026-10-20，Asia/Shanghai。**BaoStock MCP 已安装、注册且通过真实 MCP 协议查询；严格数据验收仍未通过，正式 NN 训练为 0/51。**

用户本轮要求自行安装 MCP、继续推进到训练，随后明确停止尝试低权限 Tushare，并确认“继续按原协议补齐，验收通过再正式训练”。本轮遵守这个范围，没有改用探索训练，也没有解除任务书第 6.3 节的准入要求。

## 安装与连接证据

- Review 注册名：`nn-ashare-data`，STDIO，已启用。配置读取结果：`data_mcp/registration.json`。
- 上游：[maimai-hqw/ashare-mcp](https://github.com/maimai-hqw/ashare-mcp)，包版本 0.1.0；BaoStock 0.9.4；MCP SDK 固定为 1.30.0。
- 安装位置：本目录 `data_mcp/.venv/`。源码 ZIP、展开源码和依赖锁分别位于 `data_mcp/vendor/`、`data_mcp/requirements.lock.txt`；ZIP SHA256 已记录在 `derived/mcp_evidence_v1/assessment.json`。
- 安装前检查了入口、依赖、网络域名和文件写入行为。实际启动器只保留 10 个相关读取工具，未开放任意文件下载、实时筛选或交易能力。
- 使用 MCP `initialize`、`tools/list`、`tools/call` 完成验证；对应原始 JSON 和服务日志已保存。这与仅安装 Python 数据包或只建议插件不同。
- 注册方式依据 [Review 官方 MCP 文档](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)。插件目录未搜索到匹配的 A 股插件，因此安装本地开源 MCP。

第三方声明没有给 MCP SDK 设上限，首次解析到了不兼容的 2.3.0；已固定到其 FastMCP 代码所需的 1.x。失败日志保留。全证券基本表请求曾超时，改为逐证券查询；2024-11-06 全市场请求也发生传输失败，失败返回没有被当作空数据或完整结果。

启动器增加 18 秒 socket 超时、分页错误和空传输拒收。测试确认发生分页中断时不会接受部分表。

## 实际拿到的数据

| 项目 | 实测结果 | 可用边界 |
|---|---|---|
| 270 个遗漏沪深代码 | **270/270** 取回基本资料，无空返回 | 证券基本资料不等于退市现金结算 |
| 证券身份分类 | 266 个终止目录匹配、3 个代码变更、1 个 CDR | 266 个仍需逐事件终值/换股核验 |
| 日线补充样本 | **12,332 个唯一证券×日期键**，原始来源副本合计 13,070 行 | 是验证样本，不是完整全市场日级包 |
| 历史状态 | 12,332 条含日级 ST 字段；514 条供应商标记停牌 | 日级状态没有证明具体开盘许可或复牌时刻 |
| 新旧 ST 差异 | 样本内 **2,129 条**与旧月度快照不同 | 不将这一个样本的差异率外推到全市场 |
| 600000 原价与前复权 | 4,555 个日期；OHLC 比例相对离差超过 0.5% 的日期为 0 | 只核验这一证券，不代表全体公司行动通过 |
| 早期历史 | 600000 取到 **2008-01-02** 起行情和早期日历 | 能补长窗口，但其他证券及全截面尚未补齐 |
| 分红 | 取得登记、除权、派息、红股上市和股息字段 | 净额、完整送转比例与账户税收口径仍需审核 |
| 原三条零 OHLC | 688089、688143、688173 的 2024-11-06 均返回停牌状态 | 供应商填充的参考价不能冒充可成交开盘价 |

所有成功/失败的请求参数、返回字段、抓取时间和文件哈希均保留，来源表中的未知值没有补成零或 False。

### 证券代码衔接

| 原代码 | 后续代码/类型 | 日期 | 已核验依据 |
|---|---|---|---|
| 000022 | 001872 | 2018-12-26 | [公司实施公告](https://static.cninfo.com.cn/finalpage/2018-12-26/1205690369.PDF)，原持仓数量不变 |
| 000043 | 001914 | 2019-12-16 | [交易所指数调整公告](https://www.szse.cn/disclosure/notice/general/t20191211_572534.html)；仍需完整股份权利衔接核验 |
| 300114 | 302132 | 2025-02-17 | [公司代码变更公告](https://static.cninfo.com.cn/finalpage/2025-02-07/1222485220.PDF)，原持仓数量不变 |
| 689009 | 存托凭证 CDR | 2020-10-29 上市 | [上交所上市公告](https://www.sse.com.cn/disclosure/announcement/listing/c/c_20201027_5242851.shtml)，不属于本协议普通 A 股范围 |

BaoStock 的 `type=1` 同时出现在 689009，因此不能仅靠这一枚举把所有记录当作普通 A 股。

302132 接口还返回了 2025-02-17 之前以 300114 交易的行情。规范化层保留 `source_symbol`、稳定 `security_id` 和 `active_code_on_date`；9 条不属于该日有效代码的记录已标记隔离，原始返回没有删掉。此处仅完成补充数据层处理，尚未将完整证券身份映射整合进正式样本包和实际股数账本。

600112 的日级 ST 变更与本轮停止 Tushare 前已取得的曾用名样本进行了对照。其终止日期存在价格全空、ST=0 的占位行；已单独标记，不能报告成“退市日摘帽”。

### 分红和停牌的限制

600000 的 2023 年分红返回税前每股 0.32 元，但税后字段为“0.288或0.32”。规范化代码没有任选其中一个值。`net_cash_per_share` 保持缺失，`verified=False`。某些送转字段为空时也没有自动推断零。

停牌日供应商可能填充前值 OHLC。补充表保留供应商原记录，但 `open_status=unknown`、`adjustment_verified=False`，没有生成买卖许可，也没有将这些数据写成已验证执行价格。

## 下一步准入结果

已把以上资料整理为 `derived/mcp_evidence_v1/`：

- `security_master_candidates.csv`、`code_changes.json`：证券资料与代码关系。
- `daily_observations.parquet`、`daily_distinct_candidates.parquet`：保留来源副本与去重后的候选记录。
- `distribution_candidates.csv`：未验收的公司行动。
- `cross_source_comparison.csv`、`sh600112_st_transitions.csv`：交叉核查。
- `assessment.json`、`manifest.json`：统计、原始证据和版本哈希。
- `acceptance.json`、`training_preflight.json`：严格关卡检查。

**已执行正式训练入口的准入检查，结果 BLOCKED，未启动任何正式拟合。** 这不等于已经完成全部数据补齐。

剩余工作分两类：

1. **已找到可用接口、但尚未完成全量处理**：全市场日级 ST/状态下载和对账、全证券早期历史、完整复权与分红覆盖、证券身份衔接、完整 17 因子及标签包。现有少量成功样本不能代表这些工作已完成。
2. **当前所装 MCP 的标准表不能直接补齐**：退市/并购终值及换股实际结算、特殊配股权利、开盘时段停复牌与实际限价例外、账户适用的红利税和到账规则。这些仍需要逐事件公告、交易所/结算资料或其他可核验数据源。不能把复权因子、日级停牌标记或终止上市日期当作替代证据。

用户已选择严格协议，因此没有用当前资料启动探索训练，没有把验收开关改成 PASS，也没有宣称 NN 优于 F17。

## 检查与运行

新增 **8 项测试通过**（`data_mcp/tests.log`）：日期契约、原价/复权分离、单位、未知状态、代码别名隔离、税后歧义、重复键及分页失败拒收。原训练代码 SHA256 与冻结版本一致；原项目 92 个 Python 文件再次核验全部未变（`data_mcp/original_code_verification.json`）。

Tushare 停止前已有少量成功返回，也暴露了每小时限制和接口无权限；在用户明确要求停止后没有再发送 Tushare 请求。其本地适配器未注册为启用的 Review MCP。

复核本轮补充数据：

```powershell
Set-Location 'workspace/review_workspace\quant_factor_ashare_review\nn_v2'
& data_mcp/.venv/Scripts/python.exe -X utf8 -m unittest discover -s data_mcp -p 'test_*.py' -v
& 'workspace/source_workspace/PatchTST_A股复现/.venv/Scripts/python.exe' -X utf8 data_mcp/assess.py
```

这两个命令只做本地验证，不请求 Tushare、不启动训练。正式入口仍是 README 所列 `prepare.py → train.py → evaluate.py`，必须提供验收通过的新数据包。
