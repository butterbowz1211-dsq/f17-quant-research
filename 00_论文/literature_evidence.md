<!-- Research project author: Gilmour Jia. Cited works retain their original authors. -->

# F17 研究的文献证据与引用核查

核查日期：2026-10-21（Asia/Shanghai）。任务为支撑现有实验的学术论文写作，不是开展系统综述。文献只为研究设计和解释提供理论或先验证据；本项目最终绩效必须来自保存的代码、参数和收益序列。

## 检索与纳入范围

先检查 source_workspace/quant_factor_ashare 的 文献库、papers、ml_policy/papers、ml_policy/literature，并追溯 ml_policy/README.md 指向的 Documents/Review/2026-10-16/new-chat/outputs/reading_manifest.json。使用 bundled Python 与 pypdf 提取首页和指定关键词所在页；再以完整题名和 DOI 搜索出版社、作者或机构页面。纳入与四类模型的经济动机、统计估计、模型监督和组合实施直接相关的16篇文献。排除错配、无效PDF及无关NBER编号，不以文件名当正文身份。未实施完整数据库检索或撤稿数据库筛查，不宣称文献穷尽或投稿认证。

期刊分类：6篇 Journal of Financial Economics；6篇 The Review of Financial Studies；2篇 The Journal of Finance；1篇 Journal of Empirical Finance；1篇 Management Science。JFE、JF、RFS、Management Science 不可混称同一期刊。

## 可写入论文的逻辑链

**固定F17的意义。** 多个有经济解释的因子先分别去极值、标准化与中性化，再合成一个透明打分规则，使复杂模型面对一个能解释、低估计自由度的对照。Liu et al. (2019) 和 Bali et al. (2011) 可支撑中国壳价值筛选及彩票偏好类因子的经济背景；它们不规定本项目应有17个因子，更不证明17个因子和固定权重为总体最优。DeMiguel et al. (2009) 支持把简单等权规则作为有竞争力的基准，但该文1/N是资产权重，本研究1/17是因子分数权重，属于统计思路类比，须明确对象差异。

**为什么检验正则化和权重重估。** 等权牺牲了按信息强弱配置的灵活性，同时减少了估计误差。可用Ridge或向等权先验收缩检验偏差与方差权衡；Kozak et al. (2020) 和 Ledoit and Wolf (2003) 提供收缩思想，但前者针对SDF系数/特征组合收益，后者针对风险协方差，均非逐股回归Ridge的同名实现。Brandt et al. (2009) 直接权重政策与逐股预测回归也不相同。本项目权重实验的负结果能说明当前因子、样本、目标与选参方式未获得可重复增量，不能说明所有最优权重或正则化理论无效。

**为什么引入机器学习。** Gu et al. (2020) 为非线性交互、正则化、Huber损失及时间划分提供方法背景，Leippold et al. (2022) 提供A股研究先例。若F17因子存在状态依赖或非线性相互作用，共享MLP可以作为挑战者；其优势是表达能力，而非必然更高收益。网络预测目标、IC、头部组合收益和扣费收益并不等价。若保存结果显示IC提高但Sharpe下降，论文应解释评价对象错配，不能把预测改进写成投资价值已经改善。

**为什么采用文献启发的改进。** Brandt et al. (2009)、Simon et al. (2026)、Jensen et al. (2026) 使研究从先预测后配权延伸到直接以组合效用或扣费收益学习权重。Gârleanu and Pedersen (2013)、Novy-Marx and Velikov (2016) 为部分调仓、买入持有不同门槛和不交易区提供依据。它们只支撑“应考虑成本和持仓延续”，不推出50%调仓、3%上限、25bp费用或0.25%带宽为理论最优。Cohen and Frazzini (2008) 的客户供应商联系提供关系信息研究动机；本项目历史收益相关图属于统计关系代理，不能直接解释为真实供应链传播。

**为何要报告失败和边界。** Daniel and Moskowitz (2016) 表明动量风险存在状态依赖，Cederburg et al. (2020) 则提示动态管理的样本内优势未必转化为实时样本外优势。应据本项目结果给出限定性结论：哪一种映射、哪一个时期、哪一个股票池和哪一个费用口径下增益未成立。不能把v5退化单因果归于“42维必然过拟合”或宣称“34维已经到上限”；现有样本只能支持观察到泛化或目标错配风险，这些机制仍属待检验解释。

## 重要的文件身份纠正

- ml_policy/papers/cong2021_alphaportfolio.pdf 实际是 Poh、Lim、Zohren、Roberts 的跨截面货币排序论文。旧 index.json 的标签错误，禁止以此PDF证明Cong的AlphaPortfolio。真正对应文件为 alphaportfolio_nber2026.pdf，题为 AlphaPortfolio: Goal-Oriented Investment Management through Deep Reinforcement Learning，Cong、Tang、Wang，NBER 35195（2026）。本主文参考文献清单未纳入它，避免不必要混淆。
- ml_policy/papers/simon2023_deep_ppp.pdf 无法作为有效PDF读取。本次改用出版社2026正式在线页面核验并限定阅读范围。
- ml_policy/literature/demiguel_naive_2009.pdf 和 Review/.../portfolio/DeMiguel2009_RFS.pdf 均不是可用正文。本次仅使用出版社元数据与作者公开摘要，不声称全文审阅。
- ml_policy/literature/ledoit_wolf_2004b.pdf 实际为2001作者稿、正式发表于2003年 Journal of Empirical Finance；不能按文件名写成2004的另一篇论文。
- ml_policy/papers/leippold2022_china_ml.pdf 的ZORA封面沿用Chinese Factor Zoo旧题名，正文及正式出版条目为 Machine learning in the Chinese stock market。
- 文献库/try1.pdf 是 Size and Value in China 讲义，try3.pdf 才是NBER正文。nber_w24454至nber_w24457是无关论文；正确相关编号为w24458。
- 文献库/menzly_jf.pdf 实际是Internet Appendix，并非正文。
- 文献库/DanielMoskowitz2016_MomentumCrashes_JFE.pdf 无效，DanielMoskowitz2016_MomentumCrashes.pdf 才是可读正式期刊版。
- Review/.../core/JKMP_IEF2026.pdf 实际内容是2022公开稿。正式2026 RFS 39(10), 3035–3078 已通过出版社核验；引用一般方法可联系正式发表，但不可把旧稿结果数值冒充正式版。

## 文献条目与证据边界
### 1 liu2019

Liu, J., Stambaugh, R. F., & Yuan, Y. (2019). Size and value in China. Journal of Financial Economics, 134(1), 48–69. https://doi.org/10.1016/j.jfineco.2019.03.008

- 实际材料：workspace/source_workspace/quant_factor_ashare/文献库/try3.pdf。
- 版本与阅读范围：2018 NBER 工作论文，57页；try1.pdf是讲义，不当正文；PDF第2页摘要；正式期刊摘要核验。
- 支持的论点：中国极小市值股票可能含壳价值；以EP构造价值因子及剔除最小30%是可说明的文献选择。
- 适用边界：本文市值筛选口径与流通市值代理、研究股票池并不自动等同原文总市值及数据口径。原论文不能证明本研究不存在微盘或幸存偏差。
- 核验：作者、题名、年份、期刊、卷期页码（如已分配）与DOI已通过本地正文及出版社/作者检索结果交叉核对。DOI直接打开有服务不可达情况，不能声称全部进行了HTTP解析成功验证。撤稿状态未专门筛查。

### 2 bali2011

Bali, T. G., Cakici, N., & Whitelaw, R. F. (2011). Maxing out: Stocks as lotteries and the cross-section of expected returns. Journal of Financial Economics, 99(2), 427–446. https://doi.org/10.1016/j.jfineco.2010.08.014

- 实际材料：workspace/source_workspace/quant_factor_ashare/文献库/BaliCakiciWhitelaw2011_MaxingOut_NBERw14804.pdf。
- 版本与阅读范围：2009 NBER 工作论文，50页；PDF第2页摘要；正式期刊摘要核验。
- 支持的论点：美国样本中最大单日收益MAX与未来收益负相关，为反彩票特征提供经济动机。
- 适用边界：美国证据不能固定A股全部时期的MAX方向；本研究必须用保存的同口径IC与组合结果核验。
- 核验：作者、题名、年份、期刊、卷期页码（如已分配）与DOI已通过本地正文及出版社/作者检索结果交叉核对。DOI直接打开有服务不可达情况，不能声称全部进行了HTTP解析成功验证。撤稿状态未专门筛查。

### 3 daniel2016

Daniel, K., & Moskowitz, T. J. (2016). Momentum crashes. Journal of Financial Economics, 122(2), 221–247. https://doi.org/10.1016/j.jfineco.2015.12.002

- 实际材料：workspace/source_workspace/quant_factor_ashare/文献库/DanielMoskowitz2016_MomentumCrashes.pdf。
- 版本与阅读范围：正式期刊PDF，27页；PDF第1页摘要与第12页动态策略讨论；对应印刷221、232页。
- 支持的论点：动量收益与市场下跌、高波动及反弹状态相关，为状态条件化和回撤管理提供动机。
- 适用边界：原文多空动量和动态暴露不等于本研究纯多头F17；不能移植论文夏普，也不能断言任意条件权重改善有效。
- 核验：作者、题名、年份、期刊、卷期页码（如已分配）与DOI已通过本地正文及出版社/作者检索结果交叉核对。DOI直接打开有服务不可达情况，不能声称全部进行了HTTP解析成功验证。撤稿状态未专门筛查。

### 4 cederburg2020

Cederburg, S., O’Doherty, M. S., Wang, F., & Yan, X. S. (2020). On the performance of volatility-managed portfolios. Journal of Financial Economics, 138(1), 95–117. https://doi.org/10.1016/j.jfineco.2020.04.015

- 实际材料：workspace/source_workspace/quant_factor_ashare/文献库/COWY.pdf。
- 版本与阅读范围：正式期刊PDF，23页；PDF第1页摘要与引言；印刷95页。
- 支持的论点：103种股票策略上的波动管理并不系统优于原策略；样本内跨越回归与实时可实施投资表现须区分。
- 适用边界：用于限制动态风控的外推，而不是证明F17波动管理无效；本研究需自己的时间外验证。
- 核验：作者、题名、年份、期刊、卷期页码（如已分配）与DOI已通过本地正文及出版社/作者检索结果交叉核对。DOI直接打开有服务不可达情况，不能声称全部进行了HTTP解析成功验证。撤稿状态未专门筛查。

### 5 kozak2020

Kozak, S., Nagel, S., & Santosh, S. (2020). Shrinking the cross-section. Journal of Financial Economics, 135(2), 271–292. https://doi.org/10.1016/j.jfineco.2019.06.008

- 实际材料：workspace/user_home/Documents/Review/2026-10-16/new-chat/work/strategy_research/mlshrinking/KNS2020_SCS_author.pdf。
- 版本与阅读范围：2019-05-13作者稿，56页；另有2017 NBER版；PDF第1、3、25页；SDF先验与CV说明。
- 支持的论点：对特征组合收益的高维SDF估计进行经济约束收缩，尤其抑制低方差主成分方向，支持收缩作为强线性基准。
- 适用边界：原文SDF的风险溢价/协方差目标与逐股收益Ridge不同；用17因子多空收益构造近似实现应标为改编，不能称完整复现。
- 核验：作者、题名、年份、期刊、卷期页码（如已分配）与DOI已通过本地正文及出版社/作者检索结果交叉核对。DOI直接打开有服务不可达情况，不能声称全部进行了HTTP解析成功验证。撤稿状态未专门筛查。

### 6 leippold2022

Leippold, M., Wang, Q., & Zhou, W. (2022). Machine learning in the Chinese stock market. Journal of Financial Economics, 145(2), 64–82. https://doi.org/10.1016/j.jfineco.2021.08.017

- 实际材料：workspace/source_workspace/quant_factor_ashare/ml_policy/papers/leippold2022_china_ml.pdf。
- 版本与阅读范围：ZORA封面加2021排印稿，20页；封面旧题名Chinese Factor Zoo，正文/正式出版题名以本条为准；PDF第2–5页，摘要、引言、子样本讨论。
- 支持的论点：A股机器学习预测研究强调流动性、交易成本以及股票规模和投资者结构差异，为比较线性模型与NN提供同市场依据。
- 适用边界：其特征库、预测期限、样本与交易设定不同；文献支持研究问题，不能背书本项目特定网络宽度或保证NN超过F17。
- 核验：作者、题名、年份、期刊、卷期页码（如已分配）与DOI已通过本地正文及出版社/作者检索结果交叉核对。DOI直接打开有服务不可达情况，不能声称全部进行了HTTP解析成功验证。撤稿状态未专门筛查。

### 7 gu2020

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. The Review of Financial Studies, 33(5), 2223–2273. https://doi.org/10.1093/rfs/hhaa009

- 实际材料：workspace/source_workspace/quant_factor_ashare/ml_policy/papers/gu2020_gkx_nber.pdf。
- 版本与阅读范围：2018 NBER稿，2019-09修订，80页；PDF第2、10、12页；摘要、时间划分、Huber损失。
- 支持的论点：非线性交互可改善收益预测，正则化和独立时间验证是学习程序核心；Huber用于降低极端误差对目标的影响。
- 适用边界：这是一项美股预测比较，预测收益、组合收益和扣费可实施收益是不同评价对象；本研究34→32→16→1并非原文架构复刻。
- 核验：作者、题名、年份、期刊、卷期页码（如已分配）与DOI已通过本地正文及出版社/作者检索结果交叉核对。DOI直接打开有服务不可达情况，不能声称全部进行了HTTP解析成功验证。撤稿状态未专门筛查。

### 8 brandt2009

Brandt, M. W., Santa-Clara, P., & Valkanov, R. (2009). Parametric portfolio policies: Exploiting characteristics in the cross-section of equity returns. The Review of Financial Studies, 22(9), 3411–3447. https://doi.org/10.1093/rfs/hhp003

- 实际材料：workspace/source_workspace/quant_factor_ashare/ml_policy/papers/brandt2009_parametric_portfolio.pdf。
- 版本与阅读范围：2007-09作者稿，47页；PDF第1–2页及正式期刊摘要。
- 支持的论点：公司特征可直接参数化组合权重，利用投资者平均效用估计共享参数，从而直接针对经济目标学习。
- 适用边界：F17分数回归和其向等权收缩不是原文完整PPP；本项目多头、现金、容量、费用约束属于研究者的适配。
- 核验：作者、题名、年份、期刊、卷期页码（如已分配）与DOI已通过本地正文及出版社/作者检索结果交叉核对。DOI直接打开有服务不可达情况，不能声称全部进行了HTTP解析成功验证。撤稿状态未专门筛查。

### 9 demiguel2009

DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy?. The Review of Financial Studies, 22(5), 1915–1953. https://doi.org/10.1093/rfs/hhm075

- 实际材料：出版社或作者在线页面。
- 版本与阅读范围：本地同名PDF无效；本次依据期刊/作者公开摘要，未声称阅读全文；正式期刊元数据及作者SSRN摘要。
- 支持的论点：样本估计最优配置受估计误差制约，1/N是必要而有竞争力的基准。
- 适用边界：原文1/N是资产配置等权，本研究F17因子打分等权是类比而非同一命题；不能据此断言所有拟合权重无效。
- 核验：作者、题名、年份、期刊、卷期页码（如已分配）与DOI已通过本地正文及出版社/作者检索结果交叉核对。DOI直接打开有服务不可达情况，不能声称全部进行了HTTP解析成功验证。撤稿状态未专门筛查。

### 10 freyberger2020

Freyberger, J., Neuhierl, A., & Weber, M. (2020). Dissecting characteristics nonparametrically. The Review of Financial Studies, 33(5), 2326–2377. https://doi.org/10.1093/rfs/hhz123

- 实际材料：workspace/source_workspace/quant_factor_ashare/ml_policy/literature/freyberger_dissecting_characteristics.pdf。
- 版本与阅读范围：2017 NBER工作论文，68页；另有2018 CESifo稿；PDF第2–3页；组LASSO和二次样条说明。
- 支持的论点：非参数特征函数与自适应组LASSO可区分增量信息和冗余；单特征曲率与跨特征交互应分开讨论。
- 适用边界：本项目若只实现线性自适应LASSO、没有rank样条和分组惩罚，不应标为FNW完整复现。
- 核验：作者、题名、年份、期刊、卷期页码（如已分配）与DOI已通过本地正文及出版社/作者检索结果交叉核对。DOI直接打开有服务不可达情况，不能声称全部进行了HTTP解析成功验证。撤稿状态未专门筛查。

### 11 ledoit2003

Ledoit, O., & Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns with an application to portfolio selection. Journal of Empirical Finance, 10(5), 603–621. https://doi.org/10.1016/S0927-5398(03)00007-0

- 实际材料：workspace/source_workspace/quant_factor_ashare/ml_policy/literature/ledoit_wolf_2004b.pdf。
- 版本与阅读范围：2001-10作者稿，20页；文件名2004b与正文身份不符；PDF第1页；正式期刊元数据及摘要。
- 支持的论点：在样本协方差与结构化目标间收缩有助于降低风险估计误差。
- 适用边界：风险协方差收缩与收益系数Ridge、向因子等权先验收缩是不同对象，不能互换方法名。
- 核验：作者、题名、年份、期刊、卷期页码（如已分配）与DOI已通过本地正文及出版社/作者检索结果交叉核对。DOI直接打开有服务不可达情况，不能声称全部进行了HTTP解析成功验证。撤稿状态未专门筛查。

### 12 cohen2008

Cohen, L., & Frazzini, A. (2008). Economic links and predictable returns. The Journal of Finance, 63(4), 1977–2011. https://doi.org/10.1111/j.1540-6261.2008.01379.x

- 实际材料：workspace/source_workspace/quant_factor_ashare/文献库/cohen_frazzini_JF2008.pdf。
- 版本与阅读范围：正式期刊PDF，35页；PDF第1页摘要及引言；印刷1977页。
- 支持的论点：客户与供应商等经济联系可能因有限注意形成跨公司收益可预测性，为研究关系特征提供动机。
- 适用边界：相关系数近邻不是披露的客户供应商关系；本项目统计关系图不直接识别信息传播或因果供应链渠道。
- 核验：作者、题名、年份、期刊、卷期页码（如已分配）与DOI已通过本地正文及出版社/作者检索结果交叉核对。DOI直接打开有服务不可达情况，不能声称全部进行了HTTP解析成功验证。撤稿状态未专门筛查。

### 13 garleanu2013

Gârleanu, N., & Pedersen, L. H. (2013). Dynamic trading with predictable returns and transaction costs. The Journal of Finance, 68(6), 2309–2340. https://doi.org/10.1111/jofi.12080

- 实际材料：workspace/user_home/Documents/Review/2026-10-16/new-chat/work/strategy_research/portfolio/GarleanuPedersen_DynamicTrading.pdf。
- 版本与阅读范围：正式期刊PDF，32页；PDF第1、6页，部分调整及二次交易成本假设。
- 支持的论点：在可预测收益和交易成本下，已有持仓、信号衰减和未来目标影响交易速度，支持部分调整的经济动机。
- 适用边界：论文闭式最优解依赖特定成本和收益动态；本项目50%调仓只是预设或经验证参数，不可称理论最优值。
- 核验：作者、题名、年份、期刊、卷期页码（如已分配）与DOI已通过本地正文及出版社/作者检索结果交叉核对。DOI直接打开有服务不可达情况，不能声称全部进行了HTTP解析成功验证。撤稿状态未专门筛查。

### 14 novymarx2016

Novy-Marx, R., & Velikov, M. (2016). A taxonomy of anomalies and their trading costs. The Review of Financial Studies, 29(1), 104–147. https://doi.org/10.1093/rfs/hhv063

- 实际材料：workspace/user_home/Documents/Review/2026-10-16/new-chat/work/strategy_research/portfolio/NovyMarxVelikov2014_NBERw20721.pdf。
- 版本与阅读范围：2014 NBER工作论文，61页；PDF第2页摘要；期刊元数据核验。
- 支持的论点：买入与继续持有使用不同门槛能减轻交易成本，支持缓冲区和不交易带的设计动机。
- 适用边界：原文美国多空组合与本项目多头、整手、涨跌停约束不同；文献不证明25bp或0.25%带宽是正确参数。
- 核验：作者、题名、年份、期刊、卷期页码（如已分配）与DOI已通过本地正文及出版社/作者检索结果交叉核对。DOI直接打开有服务不可达情况，不能声称全部进行了HTTP解析成功验证。撤稿状态未专门筛查。

### 15 jensen2026

Jensen, T. I., Kelly, B., Malamud, S., & Pedersen, L. H. (2026). Machine learning and the implementable efficient frontier. The Review of Financial Studies, 39(10), 3035–3078. https://doi.org/10.1093/rfs/hhag022

- 实际材料：workspace/user_home/Documents/Review/2026-10-16/new-chat/work/strategy_research/core/JKMP_IEF2026.pdf。
- 版本与阅读范围：文件名2026，但实际为2022-09-30公开稿，69页；正式2026发表元数据本次另行核验；本地PDF第2、19页；2026期刊摘要。
- 支持的论点：投资评价应基于给定风险下扣除交易成本的收益，成本敏感的组合目标能改变学到的特征配置。
- 适用边界：不能把2022稿数值写成2026正式版数值；RF指随机特征而非随机森林。本文费用代理和A股执行器不等同论文机构级冲击模型。
- 核验：作者、题名、年份、期刊、卷期页码（如已分配）与DOI已通过本地正文及出版社/作者检索结果交叉核对。DOI直接打开有服务不可达情况，不能声称全部进行了HTTP解析成功验证。撤稿状态未专门筛查。

### 16 simon2026

Simon, F., Weibels, S., & Zimmermann, T. (2026). Deep parametric portfolio policies. Management Science. Advance online publication. https://doi.org/10.1287/mnsc.2025.00721

- 实际材料：出版社或作者在线页面。
- 版本与阅读范围：正式版2026-06-03在线发表；本地simon2023_deep_ppp.pdf无效，本次读出版社网页；出版社摘要、第1节与第5节；当前Articles in Advance。
- 支持的论点：以神经网络把公司特征映射为权重，直接按投资者效用训练；风险厌恶和交易摩擦约束复杂度价值。
- 适用边界：不能从其美股效用增益推断A股策略净夏普，也不能称本项目固定损失和约束复制其完整模型。
- 核验：作者、题名、年份、期刊、卷期页码（如已分配）与DOI已通过本地正文及出版社/作者检索结果交叉核对。DOI直接打开有服务不可达情况，不能声称全部进行了HTTP解析成功验证。撤稿状态未专门筛查。

## 引用使用建议

采用作者年份引用，参考文献按第一作者姓氏字母顺序排序；本文若未引用某条，应从正文末参考文献移除而保留在此审计清单。2026 Simon条目仍为在线优先，不虚构卷期页码。工作论文阅读版本与正式发表版本在此清单披露，正文引用正式条目时只使用一致且核实的概括，不移植未核对的最终版表格数值。

建议正文只用短的释义和本项目自己的分析，不复制已下载论文长段正文或表格。文献PDF无需进入四模型代码压缩包；保留此书目和DOI即可。pdf_inventory.json、literature_read_scope.json、extract_literature.py、literature_sections.py是内部审阅中间文件，建议不进入用户压缩包。

## 核验页面

- JFE正式条目：各条DOI链接及ScienceDirect题名页面，见references.json。
- DeMiguel作者摘要：https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1376199
- Jensen正式2026条目：https://academic.oup.com/rfs/article/39/10/3035/8524346
- Simon正式2026在线全文：https://pubsonline.informs.org/doi/full/10.1287/mnsc.2025.00721
- Brandt正式条目：https://academic.oup.com/rfs/article-abstract/22/9/3411/1572695
- Ledoit–Wolf正文身份：https://www.sciencedirect.com/science/article/pii/S0927539803000070

以上核验不构成全文方法复核或对本项目投资有效性的认证。

