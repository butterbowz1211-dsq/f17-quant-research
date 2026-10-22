<!-- Research project author: Gilmour Jia. Cited works retain their original authors. -->

# 渠道审计：bridge profile 的 AKShare / Playwright 能力核查

生成时间：2026-10-10（本机 AKShare 时间工具返回 2026-10-10T20:14+08:00）
任务边界：有界只读核查，未运行任何全量下载、未做全日期遍历、未安装工具、未搜索新 MCP、未启动其他代理。

## 1. 实际工作目录与环境

| 项目 | 实测结果 |
| --- | --- |
| PowerShell `Get-Location` | `workspace/review_workspace\quant_factor_ashare_review\bridge_channel_audit` |
| .NET `Directory.GetCurrentDirectory()` | `workspace/review_workspace\quant_factor_ashare_review\bridge_channel_audit` |
| 目录现状 | 仅 `.dsh-bridge`（初始），本次新增 `bridge_probe.txt`；未触碰父目录任何文件 |

说明：`Get-Process -Id $PID).Path` 返回的是 pwsh 可执行文件路径（`C:\Program Files\PowerShell\7\pwsh.exe`），不是 cwd；cwd 以上面两项为准，二者一致。

## 2. 验证文件（写入 + 读回）

- 文件：`bridge_probe.txt`（位于当前子任务目录）
- 写入内容：`桥接验证：中文与工作目录正确`
- 读回内容：完全一致（1 行，无乱码、无 BOM 异常）
- 结论：bridge 通道的**中文编码与目录解析均正常**，写入范围严格限于当前目录。

## 3. 实际注入的可用工具（按注入的工具名核对）

### 3.1 AKShare MCP（server: `akshare`）实际注入的工具
| 工具名 | 覆盖面 |
| --- | --- |
| `get_realtime_data` | 实时行情（A/B/H） |
| `get_hist_data` | 历史 K 线 + 技术指标 |
| `get_income_statement` | 利润表 |
| `get_balance_sheet` | 资产负债表 |
| `get_cash_flow` | 现金流量表 |
| `get_financial_metrics` | 三大报表关键财务指标 |
| `get_news_data` | 个股新闻 |
| `get_inner_trade_data` | 内部人（高管）交易 |
| `get_time_info` | 当前时间 / 最近交易日 |

### 3.2 Playwright MCP（server: `playwright`）
- 实际注入：**是**。已看到 `browser_navigate`、`browser_snapshot`、`browser_click`、`browser_evaluate`、`browser_network_requests`、`browser_take_screenshot` 等浏览器工具。
- 本次按要求**未启动浏览器**，因此这只是「工具已注入」，**不等于**浏览器会话可用或能取到数据。

## 4. 本次调用及结果证据（只读，共 1 次 MCP 调用）

| # | 调用 | 结果 |
| --- | --- | --- |
| 1 | `akshare.get_time_info()` | 成功返回 `{"iso_format":"2026-10-10T20:14:01.826695+08:00","timestamp":1791634441.826695,"last_trading_day":"2026-10-09"}` |
| 2 | `akshare.get_news_data` / `get_hist_data` 等业务取数 | **未调用**（按「最多 2 次必要只读调用」上限与最小化原则省略，避免无意义全量/遍历取数） |

证据分层（必须区分）：
- **已验证**：AKShare MCP 服务端可连通并有真实返回（第 1 条）。
- **未验证**：财务三表、新闻、内部人交易等具体接口在某只股票上的取数成功率。工具注入 ≠ 取数成功。

## 5. 针对两类历史披露需求的结论

### (1) 历史研报 + 分析师身份 / 预测年度
**不能解决。** 注入的工具集里没有任何研报（盈利预测 / 分析师 / 评级）类接口。`get_news_data` 是新闻资讯，不含分析师姓名、机构、预测年度与研报原文；`get_financial_metrics` 只是已实现财务指标，不是预测值。因此该缺口在 bridge 的 AKShare profile 下**无可用渠道**。

### (2) 历史业务分部、客户/供应商关系、公告原文/首次披露日期
**不能解决。** 没有分部报表（segment）、客户/供应商集中度、公告全文或首次披露日期类接口。间接替代全部不可行：
- `get_news_data` → 二手报道，非公告原文，且无首次披露日期字段；
- `get_inner_trade_data` → 仅内部人交易，与客户/供应商无关；
- `get_income_statement` / `get_balance_sheet` / `get_cash_flow` → 合并报表合计口径，通常无分部拆分、无客户/供应商明细、无披露时点版本信息。

### 与主任务已知缺口的对应关系
| 主任务缺口 | 本次核查能否补上 |
| --- | --- |
| 巨潮同时间戳分页不稳定 | 否（AKShare profile 无巨潮公告类接口） |
| 最新财务历史值缺少版本/时点（point-in-time） | 否（三表接口未暴露公告日期/修订版本维度，需另验证，但接口签名里没有该参数） |
| 需要 A 股历史披露（研报、分部、客户供应商、公告原文） | 否，bridge 的 AKShare profile 不覆盖 |

## 6. 建议（不执行，仅记录）

若确需上述历史披露数据，应另找专门渠道（如研报库 / 公告全文与披露时点数据库 / 分部与供应链数据集，或经允许的第三方数据源），本 bridge profile 的 AKShare MCP 不构成新增可行渠道。Playwright 虽已注入，但启动浏览器抓取属于新任务，且需另行评估反爬、稳定性与合规，本次不启动。

## 7. 合规与副作用声明

- 仅在当前目录新增 `bridge_probe.txt` 与 `channel_audit.md`。
- 未修改父目录脚本、原策略、配置或任何已有下载缓存。
- 未启动/终止/修改父目录中正在运行的下载进程。
- 未读取或输出任何凭证（本次未使用 Tushare 或任何 token）。
