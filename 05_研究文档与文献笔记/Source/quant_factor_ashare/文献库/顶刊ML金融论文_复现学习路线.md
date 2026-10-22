<!-- Research project author: Gilmour Jia. Cited works retain their original authors. -->

# 顶刊 ML 金融论文 · 复现学习路线

> **本文档的筛选标准只有一条**：复现这篇论文，**我能学到什么具体的技能/概念/方法**。
>
> 不是"这篇论文对我的策略有没有用"，不是"有没有 alpha"。目标是**通过学习复现来掌握
> ML 在金融里的实际用法**。
>
> **档次标注口径**（严格，不注水）：
> - **顶级** = Journal of Finance / Journal of Financial Economics / Review of Financial Studies
> - **一流** = Management Science / Review of Finance / Journal of Econometrics / JFQA
> - **领域优质** = JBF / JFM / JEF / JEDC / JFEC / IJF / Quantitative Finance / IEEE TSP / ACM TOIS / CFR
> - **顶会** = NeurIPS / ICML / ICLR / KDD / AAAI / ACL / CIKM（**会议论文，不是"刊"**）
>
> 所有条目经 Crossref / OpenAlex / RePEc 核验；未核实项标注 `UNVERIFIED`。
> 数据可得性的详细核验见同目录 `ML与深度学习_A股文献清单.md`。

---

## 0. 先看这张地图：ML 在金融里有哪几块

| 方向 | 核心问题 | 代表论文（本文档覆盖） |
|---|---|---|
| **A. 横截面收益预测** | 用特征预测个股收益 | Gu-Kelly-Xiu (2020, RFS) |
| **B. 条件因子模型 / 潜因子** | 因子载荷随特征变化 | Kelly-Pruitt-Su (2019, JFE)；GXK (2021, JoE) |
| **C. 经济学约束进损失函数** | 让模型满足无套利 | Chen-Pelger-Zhu (2024, MS) |
| **D. 高维理论** | P>T 时会怎样 | Kelly-Malamud-Zhou (2024, JF) |
| **E. 因子研究方法论** | 新因子有没有增量信息 | Feng-Giglio-Xiu (2020, JF) |
| **F. 大规模复现与异象** | 469 个异象里几个是真的 | Li-Liu-Liu-Wei (2024, MS)；Chen-Zimmermann (2022, CFR) |
| **G. 中国市场** | A 股的 ML 结论与美股不同 | Leippold-Wang-Zhou (2022, JFE) |
| **H. 其他问题** | 波动率 / 崩盘风险 / 信用债 | Jiang-Ma-Zhu (2024, JFM) 等 |
| **I. 微观结构 / LOB** | 日内订单簿 | DeepLOB (2019, IEEE TSP) |
| **J. 序列模型架构** | 时序预测的网络设计 | PatchTST (ICLR 2023)；iTransformer (ICLR 2024) |
| **K. 文本 / NLP / LLM** | 把文字变成变量 | ⟳ 检索中 |
| **L. 波动率 / 信用 / 期权 / 组合优化** | ⟳ 检索中 | ⟳ 检索中 |

---

## 1. ★ 主干：Gu, Kelly & Xiu (2020)

**Empirical Asset Pricing via Machine Learning**
**Review of Financial Studies 33(5): 2223–2273** · 档次：**顶级** · [链接](https://academic.oup.com/rfs/article/33/5/2223/5758276)

### 复现它，你能学到（这是全清单里最重要的一篇）

1. **滚动窗口训练协议** —— 每年重训一次、只用当时可得的信息预测下一年。
   **这是金融 ML 与 Kaggle 式 ML 最大的区别，也是初学者最容易做错的地方。**
   学会它，你就永久性地获得了"识别未来函数"的能力。
2. **把公司特征整理成 ML 可用的面板** —— 缺失值处理、**逐月横截面标准化**、异常值缩尾。
   这三件事在通用 ML 教程里没人教你，但它们是金融数据的日常。
3. **模型横向对比的框架** —— OLS / PCR / PLS / GBRT / RF / NN1–NN5。
   学会判断"什么时候简单模型就够了"，而不是默认上深度网络。
4. **用样本外 R²（R²_oos）作主指标**，并理解它和"能不能赚钱"之间的距离。
5. **变量重要性归因** —— 怎么算出"哪些特征真正贡献了预测力"。
6. **从预测到组合** —— 十分位组合、多空、扣成本后的 alpha。
7. **代码要自己写**：作者的 GitHub `xiubooth/ML_Codes` **只有 `Simu_Matlab/` 模拟代码，
   没有实证代码** —— 所以你会被迫真正理解每一步，而不是抄。

**数据**：作者公开分发 `datashare.zip`（实测 **1.56 GB，HTTP 200，完全绕开 WRDS**）。
⚠️ 内含单个约 1.5 GB 的 `datashare.csv`，需分块或 dtype 优化。

> **为什么应该从这篇开始**：它是这个领域的**奠基石与标准模板**。后面每一篇论文
> （IPCA、autoencoder、deep learning asset pricing）都在回应或扩展它。
> 而且它是唯一一篇"数据公开 + 目标（Table 1 的 R²）明确 + 代码要自己写"三者齐备的。

---

## 2. 条件因子模型：把"因子"本身交给模型

### 2.1 Kelly, Pruitt & Su (2019) — IPCA
**Characteristics are covariances: A unified model of risk and return**
**Journal of Financial Economics 134(3): 501–524** · 档次：**顶级** · DOI 10.1016/j.jfineco.2019.05.001

**复现它能学到**：
- **条件因子模型的核心思想**：因子载荷 β 不是常数，而是公司特征的函数 `β(z)`
- **工具化主成分分析（IPCA）的推导与交替最小二乘（ALS）实现** —— 亲手推一遍 ALS 的收敛条件
- **潜因子与可观测因子的关系**：IPCA 同时估计出"因子"与"因子收益"，理解为什么这比"先定因子再回归"更强
- **学术级代码的工程组织**：`bkelly-lab/ipca`（**MIT 许可**）是通用估计器包，学习学者怎么写可复用代码

**代码**：✅ 公开（MIT）⚠️ **但不含论文数据**，示例只用 Grunfeld 数据 → **数据自备**
→ 这反而是好事：你可以**用自己的 A 股特征面板**跑 IPCA。

### 2.2 Gu, Kelly & Xiu (2021) — Autoencoder Asset Pricing Models
**Journal of Econometrics 222(1): 429–450** · 档次：**一流**

**复现它能学到**：
- **条件自编码器**：用神经网络**同时**学因子与载荷，把"降维"和"定价"合为一步
- **无监督目标与金融目标的结合方式** —— 这是自监督在金融里最干净的范例
- **因子个数 K 怎么选** —— 潜因子模型的核心超参，论文有系统讨论
- **一篇论文的数据能给下一篇用**：直接复用 GXK 的 `datashare.zip`

**代码**：❌ 未发现官方代码库 → 需要自己实现（学习价值高）
**数据**：可复用 GXK 的公开面板

---

## 3. ★ 最有价值的一课：把经济学约束写进损失函数

### Chen, Pelger & Zhu (2024) — Deep Learning in Asset Pricing
**Management Science 70(2): 714–750** · 档次：**一流** · DOI 10.1287/mnsc.2023.4695

**复现它能学到**：
- **把无套利条件写进损失函数** —— 网络输出被约束为不会产生静态套利。
  **这是"金融 + ML"区别于"把 ML 拿来跑金融数据"的分水岭。** 学会这个，你的模型才叫
  "金融模型"而不是"套了个壳的神经网络"。
- **对抗式构造测试资产**：把 GAN 的思想用于"生成最难被定价的测试组合"
- **SDF（随机折现因子）的估计** —— 从"预测收益"升级到"估计定价核"，这是资产定价的核心对象
- **把 TensorFlow 1.x 代码重写成 PyTorch** —— 真实的工程训练

**数据**：✅ 作者官方分发（Google Drive + Dropbox），**完全绕开 WRDS**
**代码**：✅ 公开（GitHub，82★，**无 LICENSE**）
⚠️ 原代码是 **TensorFlow 1.12 + Python 3.6**，Windows 上很难装 → 实际做法是用 PyTorch 重写

---

## 4. 理论直觉：为什么"越复杂越好"有时成立

### Kelly, Malamud & Zhou (2024) — The Virtue of Complexity in Return Prediction
**Journal of Finance 79(1): 459–503** · 档次：**顶级** · DOI 10.1111/jofi.13298

**复现它能学到**：
- **随机傅里叶特征（RFF）** —— 把线性模型变成非线性，又不失去闭式解。极其实用的技巧
- **双重下降（double descent）与 P > T 区间** —— **深度学习理论里最重要的现象之一**，
  这篇用最小的数据量把它讲清楚。理解它，你就理解了"为什么参数量超过样本量还能 work"
- **岭回归在高维下的行为**，以及它与"复杂度"的关系
- **理论直觉 vs 实证结果如何互相印证**

**数据需求极小**：只要约 15 个月度预测变量 + 市场超额收益，**数据量以 KB 计、几秒跑完**
→ **最适合"先把理论吃透再动手"**的一篇
⚠️ 无官方代码（`UNVERIFIED`）；CRSP 市值加权指数不免费，需用 French 市场因子代理（有细微差异）

---

## 5. 因子研究方法论（可直接迁移到任何因子研究）

### Feng, Giglio & Xiu (2020) — Taming the Factor Zoo
**Journal of Finance 75(3): 1327–1370** · 档次：**顶级** · DOI 10.1111/jofi.12883

**复现它能学到**：
- **双重选择 LASSO（double-selection）** —— 检验"一个新因子相对于已有因子是否有增量信息"的
  标准方法。**这是元技能**：学会它，你就能判断自己手上的一堆因子里哪些是冗余的
- **"因子动物园"问题的形式化**：怎么在几百个候选因子里做统计上站得住的筛选
- **模型选择与假设检验的结合**（机器学习方法 + 计量经济学推断）

**数据**：❌ 官方构建脚本明确要求 "WRDS account with subscription to CRSP, Compustat and IBES"
→ **方法论必学，原文数据不可复现**（但方法可以用 A 股自建面板实现）

---

## 6. 大规模复现：异象里到底几个是真的

### 6.1 Li, Liu, Liu & Wei (2024) — Replicating and Digesting Anomalies in the Chinese A-Share Market
**Management Science 70(8): 5066–5090** · 档次：**一流** · DOI 10.1287/mnsc.2023.4904

**复现它能学到**：
- **大规模异象复现的方法论**：**469 个变量**如何统一定义、统一构造、统一检验
  （结论：**83.37% 不显著**）
- **研究设计选择如何改变结论** —— 断点（主板 vs 全A）与加权（市值 vs 等权）会让结论翻转。
  学会这个，你看任何异象论文都会先问这两个问题
- **多重检验控制**：复现几百个异象时怎么处理假发现
- **一份现成的"已证伪清单"**，可直接用来筛自己的因子池

**数据**：✅ 摘要明确写明在线附录与数据可在 DOI 处获取
⚠️ **附录数据的粒度未核实**（INFORMS 对自动化访问返回 403）—— 需人工在浏览器确认

### 6.2 Chen & Zimmermann (2022) — Open Source Cross-Sectional Asset Pricing
**Critical Finance Review 11(2): 207–264** · 档次：**领域优质** · [项目站](https://www.openassetpricing.com/)

**复现它能学到**：
- **开源科研的工作方式**：数据 + 代码 + 复现脚本怎么组织、怎么版本化、怎么写 provenance
- **212 个预测变量 / 209 个公司级特征的统一框架**（宽表 1.6 GB + 212 组组合收益，可直接下载）
- **学会使用别人公开的面板**，而不必从 WRDS 重建
- 社区 Python 下载包 `openassetpricing` 的用法（2025-10 更新已把信号翻译成 Python）

⚠️ 引用注意：官网**自己的 BibTeX 把卷号写成 `27`**，GitHub README 与 RePEc 均为 **11(2)**。
⚠️ 用现成数据难度低，但**从零重建原始信号仍需 WRDS**

---

## 7. 中国市场（ML 结论与美股不同）

### Leippold, Wang & Zhou (2022) — Machine learning in the Chinese stock market
**Journal of Financial Economics 145(2): 64–82** · 档次：**顶级** · DOI 10.1016/j.jfineco.2021.08.017

**复现它能学到**：
- **特征工程的规模化**：**1,160 个信号**如何从 91 个基础特征做交互与非线性变换生成
  —— 这是"特征工程"这四个字最实在的一课
- **中国市场的特殊性如何改变 ML 的结论**：**流动性/基本面是最重要的预测变量，与美国相反**；
  零售主导提升短期可预测性；大盘股与国企长期更可预测
- **扣交易成本后是否仍然显著** —— 学会把成本作为结论的一部分而不是附录
- **无官方代码**（`UNVERIFIED`，但检索无果）→ **必须从零搭**，工作量大但学习价值最高

**数据**：⚠️ 数据厂商未核实；**替代路径**：tushare Pro 2000 积分档（200 元/年）自建
**第三方非官方复现**：`github.com/MEIQI-Lee/Machine-Learning-in-the-Chinese-Stock-Market`（91 特征 + 11 宏观因子）

### 补充（档次较低，但复现难度低、作为热身有用）

| 论文 | 档次 | 复现能学到 |
|---|---|---|
| Pan, Long, Wang & Xie (2023), *Nonlinear asset pricing in Chinese stock market: A deep learning approach*, **IRFA 87: 102627** | 领域 | **数据需求最低**（组合级/指数级输入）→ 最可能纯 tushare/akshare 复现。学会用 **LSTM 替代线性因子模型** |
| Wang & Zhu (2024), *A comparison of factor models in China*, **Journal of Empirical Finance 79: 101548** | 领域优质 | 学会用**测试资产**比较因子模型（105 异象 → 23 显著）；11 个模型横评 |
| Ma, Leong & Jiang (2023), *A latent factor model for the Chinese stock market*, **IRFA 87: 102555** | 领域 | **把 IPCA 用到 A 股**：学会"算法现成、数据自备"的工作方式 |
| Jiang, Ma & Zhu (2024), *Fundamental characteristics, machine learning, and stock price crash risk*, **Journal of Financial Markets 69: 100908** | 领域优质 | 学会把 ML 用于**尾部风险/崩盘风险**而非收益预测；分位数与极端事件建模 |
| Wang, Wang & Wu (2025), *Forecasting stock market return with anomalies: Evidence from China*, **International Journal of Forecasting 41(3): 1278–1295** | 领域优质 | 学会**收缩类方法**（shrinkage）预测市场收益；一个反直觉结论：多空异象组合在中国无预测力 |

---

## 8. 微观结构 / 限价订单簿（另一个战场）

### Zhang, Zohren & Roberts (2019) — DeepLOB
**IEEE Transactions on Signal Processing 67(11)** · 档次：**领域优质（信号处理顶刊）**
DOI 10.1109/TSP.2019.2907260 · [arXiv:1808.03668](https://arxiv.org/abs/1808.03668)

**复现它能学到**：
- **CNN + LSTM 处理限价订单簿**的完整流程：CNN 抓**空间结构**（40 档价量）+ LSTM 抓**时序**
- **LOB 数据的标签构造**：三分类（涨/平/跌）以及 **horizon bias** 问题
  （TLOB 那篇 2025 年的论文专门指出并修正了这个问题 —— 学会识别前人的标注缺陷）
- **FI-2010 数据集**的使用与 **9 折 anchored CV**（时序数据的交叉验证怎么做才对）
- 这是**日内/微观结构方向的第一课**，也是绝大多数 LOB 论文的对照基线

**数据**：✅ FI-2010 **CC BY 4.0**（已从 Etsin 页面元数据核验），
或直接用 `LeonardoBerti00/TLOB`（**MIT**）仓库内自带的折 zip
**代码**：✅ 有多份社区实现（`zcakhaa` 的 notebook，613★，⚠️ **无 LICENSE 文件**）

### 进阶：TLOB（arXiv 2025，非期刊）
**复现它能学到**：双注意力（**空间档位 + 时间轴**）Transformer 的设计；
**如何改进标注方法来消除 horizon bias**。
工程上最友好：**README 明写 requires Windows**，仓库自带 FI-2010 数据与 horizon 1/2/5/10 预训练权重，
并同时实现 DeepLOB / MLPLOB / BiNCTABL 三个基线 → **天然是一个对照实验平台**。
⚠️ 这是 arXiv 预印本，**不是顶刊**

---

## 9. 序列模型架构（PatchTST 这一条线）

| 论文 | 档次 | 复现能学到 |
|---|---|---|
| **PatchTST**（Nie et al., ICLR 2023） | **顶会** | patching（把连续时刻打包成 token）+ channel-independence；**输入表示比网络容量更重要**这一课；以及**金融数据被作者主动剔除**这个研究伦理问题 |
| **iTransformer**（Liu et al., ICLR 2024 Spotlight） | **顶会** | **把变量维当 token** —— PatchTST 在注意力轴上的正反面。**"跨变量注意力值多少"这个问题的干净消融** |
| **DLinear**（Zeng et al., AAAI 2023 Oral） | **顶会** | 一个线性层打爆所有 Transformer；学会**质疑复杂模型的必要性** |
| **MASTER**（Li et al., AAAI 2024, 38(1): 162–170） | **顶会** | 市场引导的截面 Transformer；**瞬时 + 跨期股票相关性**怎么建模 |
| **StockMixer**（Xu et al., AAAI 2024，venue 仅仓库自述） | **顶会** | **纯 MLP** 多尺度混合，明确反对"必须上 Transformer" |
| **HIST**（Xu et al., arXiv 2021） | 预印本 | 概念图 + 个股图的**双向建模**；⚠️ venue 未核实 |
| **TRA**（Lin et al., KDD 2021, pp. 1017–1026） | **顶会** | 用**最优传输**把样本路由到多个"交易模式"预测器 —— 解决"单一模型混合多种市场模式"的问题 |

⚠️ 注意区分：**ICLR / AAAI / KDD 是顶会，不是"顶刊"**。这个方向的论文主要在会议而非期刊。

---

## 10. 建议的学习顺序（按"学到什么"排，不是按档次排）

| 阶段 | 复现哪篇 | 为什么是这个顺序 |
|---|---|---|
| **1** | **Gu-Kelly-Xiu (2020, RFS)** | 建立整个领域的**标准模板**：滚动窗口、横截面标准化、R²_oos、变量重要性。数据公开、目标明确（Table 1 的 R²）。**没有它，后面每一篇你都会读得似是而非** |
| **2** | **Kelly-Malamud-Zhou (2024, JF)** | 补**理论直觉**。数据以 KB 计、几秒跑完，但把双重下降和 P>T 讲透。**先懂"为什么"，再看"怎么做"** |
| **3** | **Chen-Pelger-Zhu (2024, MS)** | 学**把经济学约束写进损失函数**。这是"金融 + ML"与"ML 套金融数据"的分水岭。顺便用 PyTorch 重写 TF1 代码 |
| **4** | **Kelly-Pruitt-Su (2019, JFE) → GXK (2021, JoE)** | 从"预测收益"升级到"**条件因子模型**"。IPCA 有 MIT 代码可读，autoencoder 要自己写 |
| **5** | **Feng-Giglio-Xiu (2020, JF)** | 学**因子研究方法论**（双重选择 LASSO）。方法可用 A 股自建面板实现 |
| **6** | **Li-Liu-Liu-Wei (2024, MS)** 或 **Leippold (2022, JFE)** | 学**大规模复现**或**特征工程规模化**。前者有数据、后者没有代码 —— 按你想练什么选 |
| **7** | 按兴趣分支 | 波动率 / 信用 / 期权 / 组合优化 / 微观结构 / 文本（后两个方向检索中） |

---

## 11. 一个必须知道的前提：免费数据绕不过去的短板

**没有任何免费源提供退市收益（delisting returns）。** 因此用 Yahoo / Stooq 等自建的
**美股**回测一定有幸存者偏差，小盘股与困境股的结论会系统性偏乐观。**无法靠技巧绕过。**

> 这构成一个实际判断：**在这些论文里，用中国数据复现的比用美股数据复现的更可靠** ——
> A 股有完整的退市、停牌、ST 记录可用。所以 Leippold (2022, JFE) 与 Li-Liu-Liu-Wei (2024, MS)
> 不只是"数据拿得到"，而是**数据质量本身更好**。

---

## 12. 文本 / NLP / LLM 在金融（部分完成）

### 12.1 ⚠️ 先看这条警告：一篇高被引论文已被作者撤稿

**Kim, Muhn & Nikolaev, *Financial Statement Analysis with Large Language Models*, arXiv:2407.17866**

这篇论文声称 **GPT-4 在预测盈利变化方向上超过专业分析师**，是"LLM 做财报分析"里被引用最多的一篇，
也是很多人会本能推荐你复现的对象。**但它已被作者主动撤稿。**

arXiv `arxiv:comment` 字段原文（**已用 arXiv API 逐字核验**）：

> "A co-author identified inconsistencies in the data and analyses while attempting to replicate past
> analyses from the working paper. Accordingly, we have **temporarily withdrawn the working paper**
> from circulation while we review the research findings."

三点要记住：
1. **它从未进过期刊** —— 始终是 arXiv 工作论文，没有经过同行评审
2. 撤稿理由是**复现时发现数据与分析不一致** —— 即"复现失败"
3. **不要把它当作复现对象**，除非你明确知道自己是在检验一篇已撤稿的工作论文

> 这件事本身就是一堂课：**"高被引" ≠ "可靠"。** 判断一篇论文能不能作为复现对象，
> 除了看 venue，还要查**当前状态**（是否撤稿、有无更正、有无 failed replication）。
> 你打算用"复现论文"来学 ML，那么**核查论文状态就是这项技能的一部分**。

### 12.2 ⚠️ FinBERT 有三个不同来源，不能混为一谈

同名 "FinBERT" 实际上是**三篇不同的论文**，引用时极易张冠李戴（均已核验）：

| 来源 | 发表状态 | 说明 |
|---|---|---|
| **Araci (2019)**, arXiv:1908.10063 | ❌ **硕士论文，无同行评审、无期刊 DOI** | 但 HuggingFace 上的 `ProsusAI/finbert` 权重最常由此而来 |
| **Yang, UY & Huang (2020)**, arXiv:2006.08097 | ❌ 无同行评审 | 代码 `github.com/yya518/FinBERT` |
| **Huang, Wang & Yang (2023)** | ✅ **Contemporary Accounting Research 40(2): 806–841**<br>DOI 10.1111/1911-3846.12832 | **唯一进入正式期刊的一篇**（会计学一流期刊）。已用 Crossref 核验 |
| Liu et al. (2020) | IJCAI 2020，DOI 10.24963/ijcai.2020/622 | 又一篇同名的 |

**实践含义**：如果你要"复现 FinBERT"，先想清楚复现的是**哪一篇**。
最常被下载的权重来自一篇**没有同行评审的硕士论文** —— 这本身就是一个需要知道的事实。

### 12.3 ✅ 已用 Crossref 核验的顶刊锚点（文本方向）

这批是"文本作为数据"这条线的骨架，全部为**顶级/一流**，DOI 已逐条核对：

| 论文 | 期刊+卷期页 | 档次 | **复现它能学到什么** |
|---|---|---|---|
| **Tetlock (2007)** | **JF 62(3): 1139–1168** | **顶级** | 把媒体文本变成可交易的**情绪变量**；词典法最原始的形态；**文本→收益的因果链怎么建立** |
| **Loughran & McDonald (2011)** | **JF 66(1): 35–65** | **顶级** | ★ **通用词典在金融文本上会失效** —— "liability"、"tax"、"cost" 在金融语境里不是负面词。**这是文本分析最重要的一课**；学会自己构造领域词典 |
| **Loughran & McDonald (2014)** | **JF 69(4): 1643–1671** | **顶级** | **词典的构造方法论**：词表如何选取、如何验证；10-K 文本的规模与清洗 |
| **Li (2008)** | **JAE 45(2–3): 221–247** | 一流（会计顶刊） | 年报**可读性**的度量（Fog index）；文本特征与公司基本面的关系 |
| **Li (2010)** | **JAR 48(5): 1049–1102** | 一流（会计顶刊） | **朴素贝叶斯 + 前向选择**做文本分类；小样本下的特征选择 |
| **Cohen, Malloy & Nguyen (2020)** | **JF 75(3): 1371–1415** | **顶级** | **LDA 主题模型**用于 10-K 文本**变化检测**；把"文本变化"本身当作信号 |
| **Sautner, van Lent, Vilkov & Zhang (2023)** | **JF 78(3): 1449–1498** | **顶级** | 用**财报电话会文本**度量气候风险暴露；**文本 → 期权定价**的连接 |
| **Bybee, Kelly, Manela & Xiu (2024)** | **JF 79(5): 3105–3147** | **顶级** | ★ 用 **LDA 从报纸全文构造"Business News"因子**；文本主题 → 宏观与市场。**"文本 + 资产定价"的顶配范例**，且作者正是 ML 资产定价那一系（Kelly、Xiu） |
| **Hassan et al. (2019)** | **QJE 134(4): 2135–2202** | **顶级**（经济学期刊 top-5） | 从电话会文本构造**公司层面政治风险**指标；损失函数式的文本打分方法 |
| **Baker, Bloom & Davis (2016)** | **QJE 131(4): 1593–1636** | **顶级**（top-5） | **关键词计数构造 EPU 指数** —— 最朴素但影响最广的文本指标。**学会"简单方法 + 大规模验证"的力量** |

### 12.4 数据可得性（已实测）

| 语料 | 状态 |
|---|---|
| **SEC EDGAR 全文检索 API** | ✅ **免费开放**（实测 `efts.sec.gov/LATEST/search-index` 返回 200 + JSON）→ 10-K / 10-Q / 8-K 全文可拿，**且带申报日期**（point-in-time 可用） |
| **财报电话会议记录（transcripts）** | ❌ **无免费全量档**。SEC 不强制提交 transcript；WRDS / StreetEvents 属机构订阅 |
| **huggingface.co** | ❌ 本机网络被阻断 → FinBERT 权重下载受阻（需镜像） |
| GDELT API | ⚠️ 返回 429（限流，非阻断） |
| 美联储 / FOMC / BIS / ECB 演讲库 | ✅ 可访问 |
| sec.gov/dera 与 Archives 全索引 | ❌ 403 |

> **由此得出一条重要的现实判断**：文本方向的**免费可用部分集中在"监管申报文件"**（EDGAR 全文），
> 而**电话会文本、新闻全文这类语料是付费墙**。所以如果你想动手，起点应该是
> **Li (2008/2010)、Loughran-McDonald (2011/2014)、Cohen-Malloy-Nguyen (2020)** 这一系
> —— 它们用的都是 10-K，而 10-K 恰好是免费的。

---

## 12.5 文本方向之后：另外七个方向（已完成，另见独立文件）

本文档第 1–12 节覆盖的是**横截面收益预测 + 文本**两条线。
另外**七个方向**的完整清单（22 篇，逐条 Crossref/OpenAlex 核验）见：
→ [`ML金融复现课程清单.md`](../../ML金融复现课程清单.md)

| 方向 | 首篇推荐 | 档次 | 为什么从这里开始 |
|---|---|---|---|
| 波动率预测 | Bucci (2020), *J. Financial Econometrics* 18(3) 502–531 | 领域优质 | 数据最易得、ground truth 最清楚；笔记本几小时跑完，一次拿到"已实现测度 → 样本外划分 → QLIKE/DM 检验 → 消融"完整工作流 |
| 期权定价/对冲 | **Buehler et al. (2019), *Quantitative Finance* 19(8) 1271–1291** | 领域优质 | **官方代码（作者本人维护，353★）+ 零数据依赖**（GBM/Heston 模拟即可）—— 全部论文中复现摩擦最小的一篇 |
| 信用风险 | Mai et al. (2019), *EJOR* 274(2) 743–758 | 领域优质 | 逼你面对"标签不可得"，学会自己从退市/重组公告造标签 —— 这项技能比模型本身值钱 |
| 公司金融 | Bao et al. (2020), *J. Accounting Research* 58(1) 199–235 | 一流（会计顶刊） | 标签（SEC AAER）与特征（EDGAR XBRL API）**全免费**，是该方向数据可得性最好的严格论文 |
| 投资组合 | Ban, El Karoui & Lim (2018), *Management Science* 64(3) 1136–1154 | 一流 | 教"决策损失 ≠ 预测损失"，这是**全部端到端金融 ML 的理论基石** |
| 市场微观结构 | Kolm, Turiel & Westray (2023), *Mathematical Finance* 33(4) 1044–1081 | 领域优质 | 完整的 LOB → 特征 → 多跨度标签 → 网络流水线；先读 Cont et al. (2014) 拿 OFI 正确定义 |
| 银行/系统性风险 | Petropoulos et al. (2020), *Int. J. Forecasting* 36(3) 1092–1113 | 领域优质 | **FDIC Call Reports + Failed Bank List 全免费** —— 该方向唯一不依赖付费数据库的高质量论文 |

**技能依赖链（建议顺序）**：波动率 → 期权/对冲 → 信用风险 → 公司金融 → 投资组合 → 微观结构 → 银行风险。

---

## 12.6 一个重量级的证据：JF 在 **1994 年**就发过神经网络论文

> **Hutchinson, J. M., Lo, A. W., & Poggio, T. (1994).**
> *A Nonparametric Approach to Pricing and Hedging Derivative Securities Via Learning Networks.*
> **The Journal of Finance, 49(3), 851–889.** DOI `10.1111/j.1540-6261.1994.tb00081.x`
> —— 本次已用 Crossref 独立复核（题名/期刊/卷期页/年份全部一致）。

这篇论文的意义远超它本身。它证明了一件对"我该学什么"这个问题**至关重要**的事：

**顶级金融期刊从不排斥神经网络 —— 它排斥的是"无法验证的结论"。**

Hutchinson-Lo-Poggio 能被 JF 接受，是因为它满足一个后来被反复验证的条件：

| 条件 | 这篇 1994 年论文怎么做的 |
|---|---|
| **有 ground truth** | 用 Black-Scholes **解析解**生成训练集，所以"网络学得准不准"是一个可以**定量回答**的问题 |
| **有经济含义** | 学的是定价函数与 delta（对冲比率），不是"预测股价" |
| **有失效边界** | 明确报告网络在深度实值/虚值、临近到期区域的误差放大 |
| **可用真实数据外部验证** | 用 S&P 500 期货期权做样本外检验 |

**对照推论（这是本文件最重要的判断之一）**：

> 为什么"用 LLM 预测股票收益"几乎进不了顶刊，而"用神经网络给期权定价"1994 年就进了 JF？
> **不是因为 LLM 太新，而是因为前者无法满足上表第一行。**
>
> 期权定价有解析解做标尺；股票收益没有。没有标尺时，"我的模型预测更准"这句话在统计上无法与"我挖了 469 个异象里的一个"区分开
> —— 这正是 §6 里 Li-Liu-Liu-Wei (2024) 量化的那个问题（**83.37% 的已发表异象不显著**）。

**顺带纠正一个常见误解**：这份 22 篇清单横跨 7 个方向，**没有一篇**是"用 LLM 预测收益"。
顶刊里的 ML 是**小网络 + 严格计量 + 明确标尺**（GXK 用的是带正则化的线性/树模型，
Hutchinson 用的是 RBF，Bucci 用的是单隐层网络）—— **模型复杂度从来不是这些论文的卖点**。

---

## 13. 待补（检索进行中）

| 方向 | 状态 |
|---|---|
| 波动率 / 信用风险 / 期权 / 组合 / 微观结构 / 银行 / 公司金融 | ✅ **已完成** → 见 §12.5 与 [`ML金融复现课程清单.md`](../../ML金融复现课程清单.md) |
| **文本 / NLP / LLM 在金融** | ◐ 部分完成（见 §12），完整表格、免费语料清单、学习路线待补。仍在核验：ACL/EMNLP 金融基准（FinQA / TAT-QA / ConvFinQA / FLUE / PIXIU）、EDGAR-CORPUS、分析师报告文本可得性 |
| **Transformer/LLM 在金融顶刊的真实分布** | ⟳ 检索中（核心问题：顶刊到底有没有、有多少；以及"LLM 预测收益"的前视偏差是否已有正式批评文献） |


---

## 附：数据可得性速查（已实测）

| 数据源 | 状态 | 用于哪些论文 |
|---|---|---|
| GXK `datashare.zip` | ✅ 公开，**1.56 GB，HTTP 200** | GXK 2020 / GXK 2021 (autoencoder) |
| Chen-Pelger-Zhu 数据 | ✅ 官方 Google Drive / Dropbox | Chen-Pelger-Zhu 2024 |
| openassetpricing.com | ✅ 209 特征宽表 + 212 组组合收益 | Chen-Zimmermann 2022 |
| Ken French Data Library | ✅ HTTP 200（⚠️ **只有组合收益，没有个股数据**） | Kelly-Malamud-Zhou 2024（代理指数） |
| SEC EDGAR XBRL Frames API | ✅ HTTP 200，**含申报日期**（point-in-time 可用） | 文本/基本面方向 |
| SEC EDGAR 财报数据集 | ✅ `2019q1.zip` 200，95.9 MB | 同上 |
| FI-2010 | ✅ **CC BY 4.0** | DeepLOB / TLOB |
| Qlib 中国A股数据 | ✅ MIT，`get_data.py qlib_data --region cn` | A 股方向 |
| tushare Pro 2000 积分档 | ⚠️ 200 元/年（含 `daily_basic` / 三大报表 / `fina_indicator`） | Leippold 复现 |
| WRDS / CRSP / Compustat / IBES | ❌ 付费 | Feng-Giglio-Xiu 2020（不可复现） |
| LOBSTER | ⚠️ **许可未核验，不要引用具体免费额度** | — |
| AQR 公开数据集 | ❌ 本机不可达（HTTP 000） | — |
