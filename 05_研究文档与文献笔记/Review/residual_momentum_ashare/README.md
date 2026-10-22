<!-- Research project author: Gilmour Jia. Cited works retain their original authors. -->

# A股残差动量：带中文注释的最小研究实现

先看 `results/报告.md`。这是已经运行的固定参数研究，不是可直接实盘的完整交易系统。
原 `source_workspace/quant_factor_ashare` 只读；没有调用付费数据接口、没有参数搜索。

## 快速运行

在本目录打开PowerShell：

```powershell
# 有Python环境时，先安装依赖；已有依赖无需重复安装。
python -m pip install -r requirements.txt
python -m unittest -v test_research
python research.py
# 无PATH中的Python时，使用本项目提供的启动脚本。
.\run.ps1
# 严格模式保留因子诊断，但在缺少完整执行数据时拒绝出具可交易回测。
.\run.ps1 --strict
```

`--source` 可换数据目录，`--end` 默认2026-09-30。实际数据截至2026-09-28。
首次准备环境时额外安装的pyarrow在本项目 `.deps`；不修改系统Python。

## 阅读代码的顺序

1. `Config`：先冻结100万元、50只、月度调仓、98%目标仓位等假设。
2. `load_data`：读取17个原始年度行情，合并5491只沪深股票；不以现存股票名单过滤历史。
3. `build_signals`：用上月末规模/BM分组得到本土MKT、SMB、HML代理，再拟合36个月OLS。
4. `factor_tests`：检查预测分数与未来收益的关系，分组组合只作诊断，不假定能融券。
5. `simulate`：逐日账户账本，月末信号下一交易日开盘执行，非调仓日保持实际持股数量。
6. `performance`、`risk_attribution`：五个窗口的净收益表现与三因子风险归因。

每个函数和重要执行片段都有中文注释。`inspect_data.py` 是只读的数据字段检查辅助脚本。

## 因子定义及时间

- 股票超额月收益对MKT、SMB、HML回归，含截距。所有拟合数据不晚于信号日。
- 风险因子用本地流通市值和账面市值比构造，SMB小减大，HML高BM减低BM；非严格原论文复制。
- 信号为最近12个月去掉最新一个月后的11个月残差均值/标准差。
- 每月正向选分数最高的50只，无反向试验、无行业联动叠加、无择时。
- 研究股票池排除未知ST、ST、BM非正、缺失规模和底部30%规模股票；上市年龄规则见代码。
- 实際有效交易从2013-02-01开始，2010—2013年初为风险因子及回归预热。不能把预热现金算作低波动策略历史。
- 最后不完整月份不生成新月度因子；已有仓位仍每日估值至行情末日。

## 统计口径

- 月度RankIC：共同有效股票上Spearman相关，至少50只；信号日为月末。
- 标签：信号后下一交易日开盘，到下次月度执行日开盘；相邻标签不重叠。
- 由于最后标签尚未完整实现，近半年只有5个可检验月，近一年11个月。
- t值使用Newey–West/Bartlett HAC，带宽6个月；另报12个月带宽敏感性及正态渐近双侧p。
- IC胜率=RankIC>0的月份占比，不是交易胜率；报告相对50%的HAC检验。
- 连续6个月区块bootstrap（999次，固定随机种子）给IC均值和胜率95%区间。
- ICIR=月均IC/月IC标准差，未年化；另有五分组收益、Q5-Q1均值与t、分年度IC。
- 5个回看窗口里有足够样本的IC均值p值做BY校正；这不是对全部历史研究搜索的校正。
- 月数不足12不报显著性；12个月以上仍需注意小样本近似。窗口互相重叠。
- 夏普用日收益，252交易日年化，固定rf=2%；现金不计息。回撤使用窗口内重新归一的净值，含起始1。
- 窗口表现来自同一条连续策略曲线的截取，不在各窗口起点清仓重开；一年内CAGR是年化外推。
- 风险归因使用已实现完整月份、6个月HAC，并输出同池市值加权无成本参考的IR；不是沪深300比较。

## A股交易表达和无法精确实现的部分

已表达：多头且现金非负；T+1可卖数量；普通板100股整数买入；科创板200股起、1股递增；零股清仓；
历史创业板10%→20%；主板ST在2026-07-06前5%、此后10%；开盘封涨停不买、封跌停不卖；无有效报价不成交；
卖出印花税在2023-08-28由10bp变5bp；双边综合佣金3bp且每笔至少5元；单边滑点10bp；
下单金额不超过严格滞后的20日平均成交额0.1%。受阻订单取消，下一月再决策，不隐式每日再平衡。

以下缺口不能由代码凭空修复：

- 逐日ST、官方逐日涨跌停价、复牌/退市特殊规则、开盘竞价深度缺失。
- 除权参考价收益是研究代理。现有行情没有完整分红/送转/配股现金流，代码通过等价股数换算近似；
  由此产生的分数持仓不是现实证券登记。买入指令按真实原始价格和板块单位计算，仍不代表真实公司行动账本。
- 规模/BM旧缓存未逐份核验实际公告日和历史更正；重新构造行情没有消除基本面覆盖选择偏差。
- 长期停牌或缺报沿用旧价会低估不确定性。`carry`为旧价账面情景，`zero`为超过60日暂按零估值、恢复报价再恢复价值的压力情景。
  两情景都不是公允价值真相；零估值还会改变后续下单金额，所以并非固定仓位收益下界。
- 因子成分当月缺失收益按0保留原权重，记录最大缺失权重；没有按未来收益是否存在筛选形成期股票。
- 交易税费、估值和成交都未达到严格实盘级别，`--strict`会明确拒绝认证，而不是悄悄放行。

## 输出文件

|文件|用途|
|---|---|
|results/报告.md|中文结果、五个绩效窗口和数据质量诊断|
|results/performance.csv|两种估值情景的夏普、最大回撤、年化收益、换手和现金比例|
|results/factor_statistics.csv|RankIC、HAC t/p、ICIR、胜率及区块区间、BY q|
|results/monthly_ic.csv|逐月IC、样本覆盖和五分组收益|
|results/yearly_ic.csv|分年度稳定性|
|results/risk_attribution.csv|三因子代理alpha、HAC t、beta、参考IR|
|results/daily_carry.csv、daily_zero.csv|可独立复算绩效的日度净值、收益、现金和旧价持仓比例|
|results/trades.csv|逐笔研究成交、数量、费用和对应信号日|
|results/monthly_scores.pkl、risk_factors.csv|信号与风险因子中间结果|
|results/input_manifest.json|原始输入SHA256，便于追溯本次数据版本|
|results/summary.json|机器可读配置、质量限制和结果|

## 参考

- [Residual Momentum原论文](https://repub.eur.nl/pub/22252/ResidualMomentum-2011.pdf)
- [A股异象研究](https://pure.eur.nl/ws/portalfiles/portal/58642799/Anomalies_in_the_China_A_share_market.pdf)
- [上交所交易规则](https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/exchange/c/c_20260424_10816482.shtml)
- [科创板申报数量](https://edu.sse.com.cn/tib/qa/c/4761956.shtml)
- [印花税减半公告](https://m.mof.gov.cn/czxw/202308/t20230827_3904226.htm)
- [2026-07-06主板ST涨跌幅调整](https://www.csrc.gov.cn/shanghai/c105566/c7643909/content.shtml)
- 同工作区的 `quant_factor_ashare_review/审计报告.md`：本地数据与旧实现的问题来源。
