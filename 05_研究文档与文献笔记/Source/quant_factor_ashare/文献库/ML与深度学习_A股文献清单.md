<!-- Research project author: Gilmour Jia. Cited works retain their original authors. -->

# 机器学习 × 中国A股：面向「可复现」的文献清单

> 生成时间：2026-10-11
> 背景：本项目文献库此前 **113 个文件 / 50 篇 PDF 中，机器学习论文 0 篇**；
> 代码侧 **18 个文件使用 statsmodels，0 处使用 sklearn/torch/xgboost/lightgbm**，
> 即整套方法论是纯线性的（OLS / Newey-West / Fama-MacBeth / 因子打分）。
> 本文档用于补齐这条线。
>
> **核验标准**：所有条目经 Crossref / OpenAlex / RePEc / arXiv / GitHub API 核验。
> 未能核实的内容一律显式标注 `UNVERIFIED` —— 请把它们当作待办事项，不要当作事实。
> 本环境对 ScienceDirect / INFORMS / SSRN 返回 403，故**多数原文的数据章节未能直读**。

---

## 0. 先看这一节：三条结论

1. **顶刊里"中国 + ML"的论文其实很少。** 对 JFE 全刊检索 "chinese" 得中国相关论文 30 篇，
   其中 ML 类**只有 Leippold et al. (2022) 一篇**；RFS / JF / Review of Finance /
   Journal of Econometrics / Journal of Financial Econometrics 检索 "china + machine learning"
   均**无命中**。管理科学（Management Science）命中两篇。
   → 这条路不是红海，但也意味着可参考的顶刊范式不多。

2. **文献读得再多，也替代不了代码里那一行 `import sklearn`。**
   本项目当前的 DSR 未通过（0.059 / 0.079，门槛 0.95），正是"24 个因子做线性组合已到顶"的
   典型症状。线性组合无法利用因子间的交互效应，每个因子只能贡献一个加性分量。

3. **有一条几乎零数据门槛的路径可以直接开跑**（见 §3 路线①）：
   Qlib 公开数据 + MASTER + HIST，全部有官方代码、数据可直接下载、无需任何机构订阅。

---

## 1. 主表（中国A股 / 中国市场）

### 1.1 顶刊核心（必读）

| 论文 | 出处 | 做什么 | 数据可得性 | 代码 | 复现难度 | 核验 |
|---|---|---|---|---|---|---|
| **Leippold, Wang & Zhou**, *Machine learning in the Chinese stock market* | **JFE 145(2): 64–82, 2022**<br>https://doi.org/10.1016/j.jfineco.2021.08.017 | 1,160 个信号 + 多种 ML 算法做 A 股横截面收益预测。**流动性/基本面是最重要预测变量（与美国相反）**；零售主导提升短期可预测性；大盘股与国企长期更可预测；扣交易成本后仍显著 | ⚠️ 数据厂商 **UNVERIFIED**（ScienceDirect 403，只读到摘要）。第三方复现仓库称用 CSMAR，属二手信息。**替代路径可行**：tushare Pro 2000 积分档可自建大部分特征 | ❌ **未找到官方代码**（GitHub API 检索作者名与标题变体均无；作者主页全部不可达）。第三方非官方复现：`https://github.com/MEIQI-Lee/Machine-Learning-in-the-Chinese-Stock-Market` | **高** | ✅ 书目已核验（Crossref + RePEc + SFI 三处一致） |
| **Li, Liu, Liu & Wei**, *Replicating and Digesting Anomalies in the Chinese A-Share Market* | **Management Science 70(8): 5066–5090, 2024**<br>https://doi.org/10.1287/mnsc.2023.4904 | 复现 **469 个**异象变量；**83.37% 不显著**；CH3/CH4/q-factor 表现最好；证明**"全A断点+等权"会过度加权微盘股** | 中国A股。摘要明确写明在线附录与数据可在 DOI 处获取；正文开放获取 | ✅ **有**：官方在线附录 + 数据 | **中** | ✅ 已核验。⚠️ **附录数据粒度 UNVERIFIED** —— 见 §4 待办 3 |
| **Liu, Stambaugh & Yuan**, *Size and value in China* | **JFE 134(1): 48–69, 2019**<br>https://doi.org/10.1016/j.jfineco.2019.03.008 | 中国 CH-3 因子模型（剔除最小 30% 市值以规避借壳壳价值）。非 ML，但是几乎所有 A 股 ML 论文的基准 | 中国A股（CSMAR/Wind）。NBER 工作论文版公开：https://www.nber.org/papers/w24458 | ❌ 未找到因子数据下载站 | 中 | ✅ 已核验 |

> ### ★ 对本项目最重要的一条交叉印证
>
> Li-Liu-Liu-Wei (2024) 用 **469 个异象的复现**独立得出"全A断点+等权会过度加权微盘股"。
> 这与本项目 `最终结论.md` 中的判断**完全一致**：
>
> > 全 A 更高收益的来源正是 Liu-Stambaugh-Yuan (2019) 指出的"壳价值"微盘股 ——
> > 这部分实际上买不到。因此 LSY 口径是更可信的主口径。
>
> **本项目当初的口径选择，经得起 Management Science 级别论文的检验。**
>
> **可直接利用的一点**：那 469 个异象中 83% 不显著 —— 这是一份现成的**"已证伪清单"**。
> 在继续试新因子之前，先用它筛一遍现有 24 个因子池，比再试 24 个新因子更有价值。

### 1.2 中国A股 ML 的相关论文（按期刊层级）

| 论文 | 出处 | 做什么 | 核验 |
|---|---|---|---|
| Jiang, Ma & Zhu, *Fundamental characteristics, machine learning, and stock price crash risk* | **Journal of Financial Markets 69: 100908, 2024**<br>https://doi.org/10.1016/j.finmar.2024.100908 | 公司特征 + ML 预测 A 股**崩盘风险（尾部风险）**；盈利性与价值/成长类特征贡献最大；国企内、低政策不确定性期表现更好 | ✅ 已核验。⚠️ 数据厂商 UNVERIFIED |
| Zhou, Fan & Xue, *How Much Can Machines Learn Finance from Chinese Text Data?* | **Management Science 70(12): 8962–8987, 2024**<br>https://doi.org/10.1287/mnsc.2022.01468 | 中文文本数据 + ML 做金融预测 | ⚠️ 仅书目核验；**摘要与数据源完全未获取** |
| Wang & Zhu, *A comparison of factor models in China* | **Journal of Empirical Finance 79: 101548, 2024**<br>https://doi.org/10.1016/j.jempfin.2024.101548 | 105 个异象 → 23 个显著异象作测试资产，比较 11 个因子模型；**q-factor 最优**，IPCA5/FF6 次之 | ✅ 已核验。**绿色 OA**，SSRN 版 https://doi.org/10.2139/ssrn.4604327 |
| Ma, Leong & Jiang, *A latent factor model for the Chinese stock market* | **IRFA 87: 102555, 2023**<br>https://doi.org/10.1016/j.irfa.2023.102555 | 用 **IPCA**（工具化主成分分析）为中国市场建潜因子模型；market beta / 盈利 / 动量最关键 | ✅ 已核验 |
| **Pan, Long, Wang & Xie**, *Nonlinear asset pricing in Chinese stock market: A deep learning approach* | **IRFA 87: 102627, 2023**<br>https://doi.org/10.1016/j.irfa.2023.102627 | 用 **LSTM** 刻画五因子的非线性定价结构；样本外拟合与多空策略优于 FF5 | ✅ 已核验。**★ 数据需求最低**：输入是组合级/指数级序列，最可能纯 tushare/akshare 复现 |
| Liao & Ma, *From fundamental signals to stock volatility: A machine learning approach* | **Pacific-Basin Finance Journal 84: 102283, 2024**<br>https://doi.org/10.1016/j.pacfin.2024.102283 | 会计特征聚合"基本面风险"预测个股波动率；神经网络优于线性 | ✅ 已核验 |
| Ma, Liao & Jiang, *Timing the factor zoo via deep learning: Evidence from China* | **Accounting & Finance 63(1): 485–505, 2023**<br>https://doi.org/10.1111/acfi.13033 | 深度学习做**因子择时**（factor timing） | ✅ 已核验（Crossref 年份 2022 为在线优先） |
| Yao, Xia & Liu, *Six-factor asset pricing and portfolio investment via deep learning* | **Pacific-Basin Finance Journal 76: 101886, 2022**<br>https://doi.org/10.1016/j.pacfin.2022.101886 | 深度学习六因子资产定价与组合投资 | ✅ 已核验 |
| Wang, Wang & Wu, *Forecasting stock market return with anomalies: Evidence from China* | **International Journal of Forecasting 41(3): 1278–1295, 2025**<br>https://doi.org/10.1016/j.ijforecast.2024.12.007 | 132 个异象组合 + 收缩类学习预测市场收益；**多空异象组合在中国无预测力（与美国相反）** | ✅ 已核验 |
| Meng, Qi, Wang & Zhao, *Disciplining the factor zoo: Identifying pricing factors in the Chinese stock market* | **Economic Modelling 155: 107370, 2026**<br>https://doi.org/10.1016/j.econmod.2025.107370 | SDF 框架下用 **double-selection LASSO** 筛选中国定价因子 | ✅ 已核验 |
| Gharghori & Nguyen, *Which factors in China? A pre-registered report* | **Pacific-Basin Finance Journal 91: 102562, 2025**<br>https://doi.org/10.1016/j.pacfin.2024.102562 | Barillas-Shanken spanning test 比较 LSY(2019) 与 FF(2018) 谁能定价中国股票 | ✅ 已核验 |

### 1.3 机器学习会议论文 + 公开代码（工程基线，非经济学证据）

| 项目 | 出处 | 做什么 | 数据 | 代码 | 复现难度 | 核验 |
|---|---|---|---|---|---|---|
| **MASTER** | **AAAI 2024, 38(1): 162–170**<br>https://doi.org/10.1609/aaai.v38i1.27767 | **市场引导的股票 Transformer**，建模瞬时/跨期股票相关性与动态特征选择 | 中文A股 CSI300/CSI800；**仓库直接发布预处理数据** | ✅ https://github.com/SJTU-DMTai/MASTER （541★） | **低** | ✅ 已核验（AAAI OJS：卷38第1期、页码162–170、DOI、作者单位） |
| **HIST** | arXiv 预印本, 2021<br>https://arxiv.org/abs/2110.13716 | **概念导向的图神经网络**股票趋势预测（含隐概念） | **Qlib 公开中文数据 Alpha360，CSI100/CSI300 —— 完全可下载** | ✅ https://github.com/Wentao-Xu/HIST （305★，README 含完整复现命令） | **低** | ✅ arXiv 元数据已核验。⚠️ 正式发表 venue UNVERIFIED（疑纯预印本） |
| **TRA** | **KDD 2021, pp. 1017–1026**<br>https://doi.org/10.1145/3447548.3467358 | 用最优传输把样本路由到多个"交易模式"预测器 | 中文A股（厂商 UNVERIFIED） | ✅ 已并入 https://github.com/microsoft/qlib | 低-中 | ✅ 已核验。⚠️ **正式标题是 "Temporal Routing Adaptor and Optimal Transport"**，不是常被引用的 "…Adapting Network" |
| **AlphaGen** | **KDD 2023, pp. 5476–5486**<br>https://doi.org/10.1145/3580305.3599831 | 强化学习生成**一组互补的**公式化 alpha 因子 | 走 Qlib 数据管线 → 可下载 | ✅ https://github.com/ICT-FinD-Lab/alphagen （1,260★） | 低-中 | ✅ 已核验 |
| **Qlib** | arXiv 2020<br>https://arxiv.org/abs/2009.11189 | 开源 AI 量化投资平台；Alpha158/Alpha360 因子集 + CSI300/CSI500 基准 | **中文A股数据可直接下载**；社区镜像 https://github.com/chenditc/investment_data | ✅ 全部开源 | **最低** | ✅ 已核验 |

---

## 2. 明确排除的论文（省时间的部分）

| 论文 / 类别 | 排除原因 |
|---|---|
| ~~Gu, Kelly & Xiu (2020), RFS 33(5): 2223–2273~~ | ⚠️ **此条判断已更正（见 §14）**。我原先写"复现不可行"是**错的** —— 作者公开分发了处理后特征面板 `datashare.zip`（实测 1.56 GB 可下载），**完全绕开 WRDS**。仍非中国样本，但数据可得 |
| Kelly, Pruitt & Su (2019), IPCA, JFE 134(3): 501–524 | 美国数据。IPCA 方法本身可用中国数据自建（见 Ma-Leong-Jiang 2023） |
| Chen, Pelger & Zhu, *Deep Learning in Asset Pricing*, Management Science | 美国 CRSP 全样本 |
| Avramov, Cheng & Metzker (2023), Management Science 69(5): 2587–2619 | 美国 CRSP/Compustat，核心贡献依赖 Compustat 财务口径 |
| Hanauer & Kalsbach (2023), Emerging Markets Review 55: 101022 | 46 国面板，数据厂商应为 Refinitiv/Datastream → **个人拿不到**。可读不可复现 |
| Cakici, Fieberg, Metko & Zaremba (2023), JEDC 155: 104725 | 同上，46 国 × 148 特征国际面板，商业库依赖 |
| Fang et al. (2026), JBF 184: 107636（中国公司债 ML） | 研究对象是**公司债**而非股票；中国信用债成交/估值数据个人获取难度远高于A股行情 |
| 各类 FRL / IRFA / Entropy / IEEE Access 的"A股 + LSTM 预测股价"论文 | 期刊层级不足，且绝大多数只报 RMSE/MAPE（**价格水平预测**），无横截面定价检验，学术与复现价值都低。除非只为练手 |
| "Quantum Kernels and the Cross-Section of Stock Returns"（Shen, arXiv）等预印本 | 声称用中国A股 point-in-time 数据，值得自行追，但数据来源与发表状态均未验到 |

---

## 3. 复现路线（按摩擦从低到高排）

> **排序理由**：本项目当前的瓶颈**不是读文献**（已有 50 篇 PDF），
> 而是**代码里一行 ML 都没有**。所以应按摩擦排序，而不是按论文地位排序。

### 路线① 几乎零门槛，立刻能跑（建议从这里开始）

```powershell
pip install pyqlib
# 下载 qlib_bin.tar.gz（社区镜像，README 明示数据源之一为 tushare）
# 解压到 ~/.qlib/qlib_data/cn_data
# 然后 clone MASTER 与 HIST，按各自 README 运行
```

| 项目 | 仓库 | 说明 |
|---|---|---|
| Qlib 数据 | https://github.com/chenditc/investment_data | GitHub Release 形式发布 `qlib_bin.tar.gz`，含 provenance manifest 与校验脚本 |
| MASTER | https://github.com/SJTU-DMTai/MASTER | 直接发布预处理数据与核心代码 |
| HIST | https://github.com/Wentao-Xu/HIST | README 给出逐条复现命令，`python learn.py` |

**目标**：先把「数据管线 → 训练 → IC/RankIC 评估 → 回测」整条链路打通，再谈方法创新。
8GB 显存足够（本机 RTX 5060 Laptop 已装好 torch 2.11.0+cu128，见 `PatchTST_A股复现/03_环境搭建记录.md`）。

⚠️ **注意**：MASTER / HIST 是 **ML 会议论文，不是资产定价论文**。它们只报 IC/RankIC/准确率，
**不做**严格的交易成本、卖空约束、容量分析与因子定价检验。
**当工程基线用，不要当经济学证据引用。**
MASTER 作者还声明仓库链接的 Qlib 版本不由他们维护，存在不一致风险。

### 路线② 枢纽论文，但工程量巨大

**Leippold, Wang & Zhou (2022), JFE 145(2)** —— 唯一发在 JFE 的中国A股 ML 资产定价论文。

最短路径：
1. 用 **Qlib + chenditc/investment_data** 下载 CN 数据，跑通 Alpha158/LightGBM 基线（1 天内）
2. 用 **tushare Pro 2000 积分档（200 元/年）** 拉 `daily` / `daily_basic` / `income` /
   `balancesheet` / `cashflow` / `fina_indicator`，构造论文的 91 个公司特征 + 11 个宏观因子
   （宏观可从 akshare 补齐）
3. 参照第三方复现仓库的 notebook 结构做 OLS / ENet / PLS / GBRT / RF / NN1-5 对比

**最大卡点**：
- **特征工程规模** —— 1,160 个信号是从 91 个基础特征做交互与非线性变换来的，
  纯工程量极大。原文未公开代码，只能做**近似复现**而非逐位复现。
- **财报 point-in-time 对齐** —— tushare 的 `fina_indicator` 必须按 **`ann_date`** 而非 `end_date`
  对齐，否则前视偏差会毁掉全部结论。

### 路线③ 读，但不复现 —— ⚠️ **此条已被更正，见 §14**

~~Gu-Kelly-Xiu (2020)、Kelly-Pruitt-Su (2019)、Chen-Pelger-Zhu 等全部锁在 CRSP/Compustat 后面。
方法论必读，复现不可行。~~

**这句话有一半是错的。** Gu-Kelly-Xiu 与 Chen-Pelger-Zhu **都公开分发了处理后数据**，
可以复现。详细更正与代价见 **§14**。

---

## 4. 待办：必须自己确认的事（我无法核实的）

按优先级排：

1. **★ CSMAR 是否对个人开放？** 子代理尝试访问 gtarsc.com 及各子域**全部失败**（SSL/超时），
   既不能证实也不能否定。**这决定了一批用 CSMAR 的中国 ML 论文对你是否可用**，
   值得花 10 分钟自己查。
2. **★ Li-Liu-Liu-Wei (2024) 在线附录里的"data"到底是什么粒度？**
   摘要说附录与数据在 DOI 处，但 INFORMS 对本环境和主 agent 都返回 **403（Cloudflare）**，
   无法下载核对。**这是复现前必须确认的第一件事**：是完整的 469 变量特征面板，
   还是只有构造说明/示例文件？在浏览器里打开一次即可看到。
   → https://doi.org/10.1287/mnsc.2023.4904
3. **Leippold et al. (2022) 到底用哪家数据**（CSMAR / Wind / RESSET / 混合）与样本期。
   ScienceDirect 403，只读到摘要。**UNVERIFIED**
4. **Leippold et al. 是否存在官方代码/数据仓库。** GitHub API 检索无果；作者主页全部不可达。
   注意这是**"未找到"而非"已证实不存在"**。
5. **HIST 是否正式发表。** Crossref 按题名无任何记录；arXiv 确认 2021-10-26 提交。
   倾向纯预印本，但无法排除发表于未被 Crossref 收录的会议。
6. **MASTER / TRA / AlphaGen 的原始数据厂商**，以及论文实验是否用了同一份公开数据。
7. **akshare 的A股覆盖边界** —— 仅确认文档站存在且相关接口在文档中出现，
   **未逐一实测**其起始年份、复权口径与稳定性。
8. **Hanauer & Kalsbach (2023) 与 Cakici et al. (2023) 的数据厂商** —— 均为推测，未从原文证实。

### 已核实的一条关键事实（决定复现路径）

**tushare 积分与权限**（来源：https://tushare.pro/document/1?doc_id=108 、
https://tushare.pro/document/1?doc_id=290）：

| 积分档 | 价格 | 可获得 |
|---|---|---|
| 120 | 免费 | 仅**非复权**日线 |
| **2000** | **200 元/年** | 周线/月线/**复权行情 `pro_bar`** / `daily_basic` / `income` / `balancesheet` / `cashflow` / `fina_indicator` / `dividend` |
| 5000 | 500 元/年 | 常规数据无总量上限 |
| 分钟数据 | 2000 元/年 | 需单独开权限（港美股亦需单独开） |

---

## 5. 检索方法（便于自己继续追）

**已穷举的期刊级检索**（OpenAlex `primary_location.source.issn` + `title_and_abstract.search`，按被引排序）：

| 期刊 | ISSN | 结果 |
|---|---|---|
| Journal of Financial Economics | 0304-405X | 全刊搜 "chinese" → 中国相关 30 篇，ML 类**只有 Leippold et al. 一篇** |
| Review of Financial Studies | 0893-9454 | 无 ML × 中国股票 |
| Journal of Finance | 0022-1082 | 无 |
| Review of Finance | 1572-3097 | 无 |
| Journal of Econometrics | 0304-4076 | 无 |
| Journal of Financial Econometrics | 1479-8409 | 无 |
| Management Science | 0025-1909 | Li-Liu-Liu-Wei (2024)、Zhou-Fan-Xue (2024) |
| Journal of Banking & Finance | 0378-4266 | 仅公司债论文 |
| Journal of Empirical Finance | 0927-5398 | 仅文本分析/债券 |
| Journal of Financial Markets | 1386-4181 | Jiang-Ma-Zhu (2024) |

**最高产的技巧：引文网络爬取。**
抓 RePEc 的 Leippold et al. 被引页面（https://ideas.repec.org/r/eee/jfinec/v145y2022i2p64-82.html ，
176 条被引），用 `Management Science|Journal of Finance|Review of Financial Studies|Journal of Econometrics|Journal of Banking|Journal of Financial Markets` 过滤。

**GitHub 检索关键词**（GitHub Search API，按 star 排序）：
`chinese stock market machine learning` / `asset pricing china machine learning` /
`china a-share factor` / `HIST stock trend forecasting` /
`MASTER market guided stock transformer` / `alphagen formulaic alpha` / `qlib chinese`

**建议继续追**：
1. 从 Leippold et al. (2022) 与 Li-Liu-Liu-Wei (2024) 的**被引列表**再滚一轮
2. 盯 **Fuwei Jiang（蒋富威，中央财经大学）** 与 **Tian Ma（马田）** 这条线 ——
   中国 ML 资产定价最高产的组合，产出横跨 JFM / PBFJ / IRFA / A&F
3. 关键词模板：`"Chinese A-share" + (machine learning | deep learning | neural network) + (cross-section | anomaly | factor zoo | realized volatility | crash risk)`，
   检索源用 **OpenAlex + RePEc**（比 Google Scholar 稳定）

---

## 6. 与现有项目的关系（为什么这条线值得补）

本项目当前的实证结果已经**指向** ML：

| 现有证据 | 指向 |
|---|---|
| DSR 未通过（0.059 / 0.079 vs 门槛 0.95） | 24 个因子的**线性组合已到顶** —— 线性组合无法利用因子间交互效应 |
| **B 层截面联动类因子样本外 IC 优于样本内**（0.0190 → 0.0248） | 截面信息在 A 股**真实有效**，而这是手工构造的关联特征 |
| COMMOVE 样本外 IC 0.0476、**t = 13.24**，B 层最强 | 同上 —— 截面关联是最有潜力的方向 |
| MOM_12_1 为负、HIGH52 为负、TURN 最强 | 与文献预测一致 → 不是代码 bug，是真实市场特征 |

**由此得到一个谱系**：

| | 截面建模方式 |
|---|---|
| **PatchTST**（`PatchTST_A股复现/`，环境已跑通） | **主动丢弃**跨股票信息（channel-independence） |
| **MASTER / HIST** | **显式建模**截面相关性与概念共享 |
| **本项目 B 层因子**（COMMOVE / geo link / biz link） | **手工构造**截面关联特征 |

本项目**同时握着这条谱系的两端**。这不是"选一个"的问题，而是一个可做的实验设计：
**手工联动因子 vs 图学习自动联动 vs 完全独立的时序模型**，在同一数据面板上做三重对照。
现有的 NW / PSR / DSR / 自助法检验框架可以直接复用。

⚠️ **一个必须先知道的陷阱**：ML **不会自动**改善 DSR。DSR 惩罚的是"试验次数"，
若调 50 组超参，有效试验次数会爆炸，SR0 门槛反而更高。
正确做法是**预先承诺一个设定**（例如"GBRT + 现有 24 因子，只做一次 walk-forward"），
把模型当作**一个**假设而不是 50 个。这样 ML 才可能在**降低**试验次数的同时提升组合效率。

---

# 第二部分：金融深度学习方法论文（有公开代码）

> 核验方式：arXiv Atom API（作者/标题/comment/journal_ref）+ GitHub REST API
> （仓库存在性 / license / stars / 最后 push）+ raw.githubusercontent 抓 README 原文 + 数据集页直抓。
> 未能核验的一律标注 `UNVERIFIED`。

## 7. A股横截面选股 / 图模型（与本项目最直接相关）

| 标题 | 出处 | 方法核心 | 数据 | 代码（已核验） | 难度 |
|---|---|---|---|---|---|
| **StockMixer** | **AAAI 2024**（仓库自述） | **纯 MLP** 多尺度（时间/股票/指标混合）架构，**明确反对"必须上 Transformer"** | 中国 A 股 | ✅ https://github.com/SJTU-DMTai/StockMixer （383★，**无 LICENSE**，push 2024-03-19） | 中 |
| **THGNN** | **ACM CIKM 2022** | 时序**异构图**：公司-公司 + 公司-新闻关系，动态图学习 | 中国 A 股 + 新闻 | ✅ https://github.com/TongjiFinLab/THGNN （**GPL-3.0**，128★） | 中 |
| **MASTER** | **AAAI 2024, 38(1): 162–170** | 市场引导的股票 Transformer，建模瞬时+跨期股票相关性 | 中国 A 股（Qlib） | ✅ https://github.com/SJTU-DMTai/MASTER （MIT，541★）。⚠️ `SJTU-Quant/MASTER` 会 302 重定向 | 中 |
| **HIST** | arXiv 2021 | 概念图（共享信息）+ 个股图双向建模 | 中国 A 股（Qlib Alpha360） | ✅ https://github.com/Wentao-Xu/HIST （305★，无 LICENSE） | 中 |
| **RSR**（Temporal Relational Ranking） | **ACM TOIS 37(2), Art. 27, 2019** | 时序 + 关系（知识图谱）联合排序做选股 | NASDAQ/NYSE；**数据获取路径 UNVERIFIED** | ✅ https://github.com/fulifeng/Temporal_Relational_Stock_Ranking （**AGPL-3.0**，528★） | 中 |

> **StockMixer 值得单独提一句**：它用纯 MLP 在 A 股上打平/超过 Transformer，是**进入 ML 摩擦最小的架构**
> （没有注意力、没有图、调试成本最低）。如果目标是"先让代码里出现一个能跑的 ML 模型"，
> StockMixer 比 MASTER 更适合当第一步。

### ★ 一个对你截面研究特别重要的对照：iTransformer

| 标题 | 出处 | 为什么对你重要 | 代码 |
|---|---|---|---|
| **iTransformer** | **ICLR 2024 Spotlight** | **把变量维当作 token** —— 这是 **PatchTST 在注意力轴上的正反面**。PatchTST 用 channel-independence **主动丢弃**跨变量信息；iTransformer 反过来，让注意力在变量维上运作。**这正好是你"截面 vs 独立"问题的架构级对照实验** | ✅ https://github.com/thuml/iTransformer （MIT，2,227★） |

你已经跑通了 PatchTST 的环境。**iTransformer 是同一批代码风格、同一个实验框架下的"另一半"** ——
两者放在一起就是一个干净的消融：跨变量注意力到底值多少。这比再找一个新模型更有说服力。

## 8. 金融基础模型（K 线 / OHLCV 序列）

| 标题 | 出处 | 说明 | 数据/权重 | 代码 | 难度 |
|---|---|---|---|---|---|
| **Kronos** | **AAAI 2026**（README 2025-11-10 公告） | 两阶段：① K 线 OHLCV 分层离散 tokenizer ② decoder-only 生成式基础模型，专为金融高噪声设计 | ⚠️ **预训练语料 = 45 家全球交易所 K 线，不公开**；权重开源 Kronos-mini 4.1M / small 24.7M / base 102.3M；**large 499.2M 未开源**。**README 自带 A 股微调示例** | ✅ https://github.com/shiyu-coder/Kronos （**MIT**，40,481★） | 低（用权重微调）；**预训练不可复现** |
| Chronos | arXiv 2024 | 数值离散成 token + T5 式语言模型，零样本 | 权重公开 | ✅ https://github.com/amazon-science/chronos-forecasting （Apache-2.0，6,042★） | 低-中 |
| TimesFM | arXiv 2023 | decoder-only 时序基础模型 | 权重公开 | ✅ https://github.com/google-research/timesfm （Apache-2.0，34,182★） | 低-中 |
| Moirai | arXiv 2024 | 多变量通用掩码编码器 | 权重公开 | ✅ https://github.com/SalesforceAIResearch/uni2ts （Apache-2.0，1,604★） | 中（8GB 跑 large 会紧） |
| Lag-Llama | arXiv 2023 | 滞后特征 + Llama 架构的概率预测 | 权重公开 | ✅ https://github.com/time-series-foundation-models/lag-llama （Apache-2.0，1,600★） | 低-中 |

**已核实的一条纠正**：网上常有人说"通用时序基础模型没在金融数据上评测过" —— **这是错的**。
已核到多篇专门评测（TimeGPT/TimesFM/Moirai/Chronos 对上 NBEATS/PatchTST/iTransformer/KAN，
按 rolling-origin 协议 + 随机游走基准）。但**PatchTST 原文的 LTSF 基准确实不含金融数据**，
这两件事不矛盾，不要混为一谈。

⚠️ **国内网络注意**：huggingface.co 从本机直连失败。用 Kronos / Chronos 等需要先配 HF 镜像。

## 9. LOB / 高频（另一条研究线，供了解）

| 标题 | 出处 | 代码 | 难度 |
|---|---|---|---|
| **TLOB**（双注意力 Transformer for LOB） | arXiv 2025 | ✅ https://github.com/LeonardoBerti00/TLOB （**MIT**，180★）。**README 明写 "requires Windows"**；仓库自带 FI-2010 部分折 zip + horizon 1/2/5/10 预训练 ckpt + 同时实现 DeepLOB / MLPLOB / BiNCTABL 基线 | **低** |
| DeepLOB | **IEEE TSP 67(11), 2019** | ✅ zcakhaa 实现（613★，**无 LICENSE**） | 低 |
| TransLOB（Multi-Horizon LOB） | arXiv 2105.10430 | ✅ zcakhaa 实现（MIT，216★） | 低 |
| Axial-LOB | arXiv 2212.01807 | ✅ 存在（60★，无 LICENSE，**官方性存疑**） | 低-中 |

> 这条线**不是你现在的战场** —— 你的项目是日频横截面因子，LOB 是日内微观结构，数据、频率、
> 评价指标全都不同。但 **TLOB 是 Windows 原生、数据自带、带预训练权重**，如果你想找一个
> "小而完整、当天能跑通"的金融深度学习项目来建立手感，它是最合适的。
> DeepLOB 的 notebook 是绝大多数 LOB 论文的对照基线，读它有长期价值。

## 10. 工程基座（可以直接调用 PatchTST，省去自己搭）

| 项目 | 说明 | 代码 |
|---|---|---|
| **Time-Series-Library (TSLib)** | 统一复现 PatchTST / iTransformer / TimesNet / DLinear 等 | ✅ https://github.com/thuml/Time-Series-Library （MIT，12,942★） |
| GluonTS | AWS，内置 PatchTST | ✅ https://github.com/awslabs/gluonts （Apache-2.0，5,245★） |
| NeuralForecast | Nixtla，内置 PatchTST | ✅ https://github.com/Nixtla/neuralforecast （Apache-2.0，4,282★） |
| RevIN | PatchTST 依赖的可逆实例归一化（本项目已实测其语义） | ✅ https://github.com/ts-kim/RevIN （MIT，443★） |

## 11. 公开可下载的金融数据集（已核验许可）

| 名称 | 内容 | 许可 | 获取 | 核验状态 |
|---|---|---|---|---|
| **FI-2010** | 5 只北欧股票 × 10 天 LOB；每行 144 维特征 + 5 列标签（1=涨/2=平/3=跌）；含三种归一化与 9 折 anchored CV | **CC BY 4.0**（从 Etsin 页面内嵌元数据核验，非猜测） | 数据集页 https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649 ；**或直接用 TLOB 仓库内自带的折 zip** | ✅ 页面/作者/发布日/许可已核验；⚠️ 直接下载 URL 未提取到 |
| **Qlib 中国A股数据** | A 股日线 + 1 分钟线、美股日线；Alpha158/Alpha360 因子集 | **MIT** | `python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn` | ✅ 命令与市场覆盖从官方文档核验 |
| Binance 公共数据 | 现货 + 合约 aggTrades / klines / trades，按日或月 zip，带 CHECKSUM | 仓库 MIT（数据集另有 TERMS 文件） | https://data.binance.vision/ ；下载脚本 https://github.com/binance/binance-public-data | ✅ **已核验：没有 order book 深度快照** —— 想要加密 LOB 必须自己用 WebSocket 录 |
| Kaggle: Optiver Realized Volatility | 美股 book/trade → 预测已实现波动率 | Kaggle 竞赛规则 | https://www.kaggle.com/competitions/optiver-realized-volatility-prediction | ✅ 页面存在已核验 |
| Kaggle: Jane Street Market Prediction | 匿名化金融特征 + 交易目标 | Kaggle 竞赛规则 | https://www.kaggle.com/competitions/jane-street-market-prediction | ✅ 页面存在已核验 |
| Kaggle: Optiver - Trading at the Close | 收盘竞价数据 | Kaggle 竞赛规则 | https://www.kaggle.com/competitions/optiver-trading-at-the-close | ✅ 页面存在已核验 |

**数据可得性结论**：对私人研究者完全开放的组合是
**FI-2010（CC BY 4.0）+ Qlib A股/美股 + Binance + Kaggle 竞赛数据 + 自己录的 LOB**。

⚠️ **两个不要引用的说法**：
- **LOBSTER 的免费额度**：站点是 JS 单页应用，正文抓不到，许可条款 **UNVERIFIED**。只核验到 URL 存在。
- **FinTSB**（arXiv:2502.18834）：基准本身不错，但 README 明写数据须从 `quantapi.eastmoney.com`
  下载（需账号），且**预处理代码"camera-ready 后才开源"** → 当下个人无法端到端复现，已排除。

## 12. ★ 幻觉引用纠正（写东西之前务必对照）

这些是**实测确认**的高频误引，抄错会直接毁掉文献列表的可信度：

| 常被误引的 arXiv ID | 它实际是什么 | 正确 ID |
|---|---|---|
| `2107.13173`（被当成 TransLOB） | *New Metrics to Evaluate … Personalized Federated Learning* | **2105.10430** = TransLOB |
| `2410.09850`（被当成 Financial Fine-tuning a LTSM） | *Can GANs Learn the Stylized Facts of Financial Time Series?* | **2412.09880** |
| `2312.05673`（被当成 TimesURL） | *Modeling Homophily in Exponential-Family Random Graph Models* | **2312.15709** |
| `2212.05153`（被当成 TimeMAE） | *Algorithmic progress in computer vision* | **2303.00320** |
| `2107.03502`（被当成 TS-TCC） | CSDI（扩散插补） | TS-TCC 的 arXiv ID 未核到 |
| `github.com/SJTU-Quant/MASTER` | 会 302 重定向 | 真实仓库 **SJTU-DMTai/MASTER** |

另：TRA 的正式标题是 **"Temporal Routing Adaptor and Optimal Transport"**，
不是常被引用的 "Temporal Routing **Adapting Network**"。

## 13. 关于 LOB 这条线的一个判断

第二路把 TLOB 排在第 1 名，理由是"最像 PatchTST"。**我部分不同意，请你自己权衡**：

- TLOB 确实是**最像 PatchTST 的架构论文**（分块 + 双轴注意力 + 多 horizon），
  而且是 Windows 原生、数据自带、带预训练权重 —— **工程上最容易跑通**。
- 但它的研究对象是 **LOB（日内限价订单簿）**，而你的项目是**日频横截面因子**。
  两者的数据、频率、评价指标完全不同。跑通 TLOB **不会**让你现有项目前进一步。
- 结论：**把它当"建立手感的一次性练习"可以，当"项目主线"不行。**
  你的主线应该还是 §7 的横截面模型（StockMixer / MASTER / iTransformer）。

---

# 第三部分：更正与最终建议

## 14. ⚠️ 更正：顶刊 ML 资产定价论文的可复现性

我在 §2 与 §3 路线③ 里写过"Gu-Kelly-Xiu / Chen-Pelger-Zhu 等全部锁在 CRSP/Compustat 后面，
复现不可行"。**这个判断是错的，现更正。**

### 14.1 更正内容（均为实测）

| 论文 | 我原先的判断 | 实际 |
|---|---|---|
| **Gu, Kelly & Xiu (2020), RFS 33(5): 2223–2273** | 数据不可得 | ❌ **错**。作者公开分发处理后特征面板 `datashare.zip`，**实测 HTTP 200、1,555,896,289 字节（1.56 GB）**，内含 `datashare.csv`（约 1.5 GB）+ `readme.txt`。**完全绕开 WRDS** |
| **Chen, Pelger & Zhu (2024), Management Science 70(2): 714–750** | 锁在 CRSP | ❌ **错**。Pelger 官方 Data and Code 页提供个股收益/特征/宏观时序（Google Drive）+ SDF 时序（Dropbox）；代码在 GitHub（82★/60 fork，**无 LICENSE**） |

**但代价要如实说**：

- GXK 的 GitHub `xiubooth/ML_Codes` **只有 `Simu_Matlab/` 模拟代码，没有实证复现代码** → Table 1 的实证流程要自己重写
- `datashare.csv` 是**单个约 1.5 GB 的 CSV**，内存吃紧，需分块或做 dtype 优化
- Chen-Pelger-Zhu 是 **TensorFlow 1.12 + Python 3.6**，Windows 上很难装 → 实际做法是用 PyTorch 重写（这反而是好的 adapt 切入点）

### 14.2 这类论文到底有多少能复现

5 篇核心论文中：**2/5 完全可复现，1/5 部分可复现，2/5 被数据门槛挡住**。

**公开可下载（均已实测）**

| 数据源 | 实测 | 关键限制 |
|---|---|---|
| GXK `datashare.zip` | **200，1.56 GB** | 处理后数据，无法从零重建 |
| Chen-Pelger-Zhu 数据 | 官方 Google Drive / Dropbox | 未下载核对字段 |
| openassetpricing.com | 209 特征宽表（1.6 GB zip）+ 212 组组合收益 | ⚠️ **重建原始信号仍需 WRDS** |
| Ken French Data Library | 页面 + `25_Portfolios_5x5_CSV.zip`（528 KB）均 200 | ⚠️ **只有组合收益，没有个股数据** |
| SEC EDGAR XBRL Frames API | **200，770 KB JSON** | 需设 User-Agent；**含申报日期**（point-in-time 可用）；⚠️ 无除权/退市处理 |
| SEC EDGAR 财报数据集 | `2019q1.zip` **200，95.9 MB** | 季度基本面，含申报日期 |

**付费闸门（实测结论）**

| 数据源 | 谁在用 |
|---|---|
| WRDS / CRSP | GXK 原始构建、KPS/IPCA、FGI 因子动物园 |
| Compustat | 所有会计类特征 |
| IBES | 分析师预期类特征（SUE、预测修正） |
| OptionMetrics / 13F | Chen-Zimmermann 部分信号类别 |

> **分水岭不是"用不用 CRSP"，而是"作者愿不愿意把处理后的面板放出来"。**
> 而且这个比例在上升 —— 最有影响力的两篇都主动分发了。

### 14.3 ★ 一个免费数据绕不过去的短板（必须知道）

**没有任何免费源提供退市收益（delisting returns）。** 因此用 Yahoo / Stooq 等自建的美股回测
**一定有幸存者偏差**，小盘股与困境股的结论会系统性偏乐观。**这一点无法靠技巧绕过。**

> **对照你自己的项目**：你在 A 股上**有**完整的退市与停牌处理（`build_panel.py` 那条管线），
> 而且你自己如实记录了"146 只早年退市股在新浪通道无数据"的残余偏差。
> **在这一点上你的数据条件比美股免费源更好** —— 这是 A 股自建面板的一个真实优势，
> 也是"用中国数据做 ML 资产定价"比"复现美股论文"更靠谱的具体理由。

### 14.4 更正后的路线③ 逐篇判定

| 论文 | 更正后 |
|---|---|
| **Gu, Kelly & Xiu (2020)** | ✅ **可复现**（数据公开）→ 见 §15，**这是最值得先做的一步** |
| **Chen, Pelger & Zhu (2024)** | ✅ **可复现**（数据 + 代码公开），但需 PyTorch 重写 |
| Kelly, Pruitt & Su (2019, IPCA) | ⚠️ **算法包公开**（`bkelly-lab/ipca`，MIT，266★）但**不含论文数据**，示例只用 Grunfeld 数据 → 算法现成、数据自备 |
| Kelly, Malamud & Zhou (2024, JF) | ⚠️ 数据需求最小（15 个月度预测变量 + 市场超额收益），但**无官方代码**；且 CRSP 市值加权指数不免费，需用 French 市场因子代理，会有细微差异 |
| Feng, Giglio & Xiu (2020, JF) | ❌ **仍不可复现**。官方构建脚本明确要求 "WRDS account with subscription to CRSP, Compustat and IBES" |
| Chen & Zimmermann (2022) | ⚠️ **用现成数据可以（难度低）**，**从零重建不可能**（README：`Signals/pyCode/` "Downloads data from WRDS and elsewhere"） |

> **引用注意**：openassetpricing.com 官网**自己的 BibTeX 把卷号写成 `27`**，
> 但 GitHub README 与 RePEc 均为 **Critical Finance Review 11(2), 207–264, 2022**。
> **引用请用 11(2)**（官网 BibTeX 有误）。

### 14.5 其他核验结论

- arXiv API 检索 `abs:"contrastive" AND abs:"asset pricing"` 与
  `abs:"self-supervised" AND abs:"return prediction"` **均返回空** → 该方向在 arXiv 上未见经核验的工作
- Google 系站点（`sites.google.com` / `docs.google.com`）在本机**连接失败** →
  Goyal-Welch 预测变量数据集的确切 URL **未能核验**
- AQR 公开数据集本机**不可达（HTTP 000）**
- CSMAR：有公开代码仓库 `Quantactix/ChinaAShareEquityCharacteristics`，但 README 明确
  `./data/csmar_tables` 存放 CSMAR 原始数据，**输出数据未上传** → CSMAR 获取条件仍未确认

**遗留 UNVERIFIED（不要当事实用）**：
GXK `datashare.csv` 的具体列名与年份范围（`readme.txt` 在 zip 内未解出）；
Chen-Pelger-Zhu 数据文件的字段与年份（链接已确认，未下载核对）；
Chen-Zimmermann 数据本身的 LICENSE（代码是 GPL-2.0，数据授权未见声明）。

---

## 15. ★ 最终建议：把 GXK 的 datashare 当作"已知正确答案"的练兵场

**这条建议的性价比高于本文档前面所有内容。**

本项目现在的瓶颈**不是缺文献**（已有 50 篇 PDF + 本文档），而是**代码里一行 ML 都没有**。
而在自己的面板上第一次写 ML，你无法区分"模型不行"和"我写错了" ——
这正是你在 `PatchTST_A股复现` 里已经识别过的问题（所以要先对齐 ETTh1 再碰 A 股）。

**GXK 的 `datashare.csv` 就是这条路的"ETTh1"：**

| | PatchTST 线 | ML 资产定价线 |
|---|---|---|
| 已知正确的参照 | ETTh1 论文 Table 3（0.375 / 0.399） | GXK 论文 Table 1（OLS/PCR/PLS/GBRT/NN 的预测 R²） |
| 数据 | `ETTh1.csv` | `datashare.zip`（1.56 GB，公开） |
| 通过标准 | MSE 对齐量级 | R² 的**排序与量级**对齐 |
| 通过之后 | 才去碰 A 股 | 才去碰你的 24 因子面板 |

**具体步骤**

1. 下载 `datashare.zip`（1.56 GB）→ 解出 `datashare.csv`
2. 按 `readme.txt` 确认列与样本期（**这一步本文档未核验过，是第一件要确认的事**）
3. 先用子样本（如 1990–2016）跑通，避免 1.5 GB 单文件的内存问题
4. 复刻 GXK Table 1：OLS / PCR / PLS / GBRT / NN1–NN5 的样本外 R²
5. **对齐之后**，把同一套代码换成你的 A 股 24 因子面板

⚠️ 记住 §6 的陷阱：**预先承诺一个设定**，不要调 50 组超参，否则 DSR 门槛会更高。

**为什么值得花这个时间**：你在自己的 A 股面板上**根本没有"已知正确答案"可以对照** ——
没有一篇论文告诉过你"这 24 个因子用非线性组合应该拿到多少 R²"。
GXK 提供了这个锚点，而且是**真实的、已发表的、可下载的**锚点。

这比直接在自建面板上跑 GBRT 更慢，但**结果可解释**。
而"结果可解释"正是本项目一贯的标准（DSR 未通过也如实记录、失败实验也如实记录）。

---

## 16. 操作提醒：本机网络间歇性不稳定

实测同一批站点在不同时刻结果相反：

| 目标 | 子代理测试 | 主 agent 复测 |
|---|---|---|
| `github.com` | 超时 | **HTTP 200（728 ms）** |
| `raw.githubusercontent.com` | 可达 | **超时** |
| `papers.ssrn.com` | 超时 | 超时 ✓ |
| `data.sec.gov` | 可达 | **HTTP 200** ✓ |
| Ken French Data Library | 可达 | **HTTP 200** ✓ |

**实践含义**：不是"某些站点不通"，而是**网络本身不稳定**。
`git clone` 与大文件下载**值得重试几次再放弃**，不要一次失败就判定不可行。

