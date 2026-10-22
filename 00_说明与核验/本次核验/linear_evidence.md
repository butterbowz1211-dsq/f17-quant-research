<!-- Research project author: Gilmour Jia. Cited works retain their original authors. -->

# F17 线性基线与正则化权重实验证据核查

核查日期 2026-10-21。所有原项目只读；本次没有重新训练模型。原始数值、逐年系数与所用源文件哈希见 `linear_models.json`。新增收益复算见 `verify_linear_metrics.py`；按原有函数从保存缓存和系数重构的 36 组 CAGR、Sharpe 与归档 CSV 最大绝对差为 4.44×10⁻¹⁶。

## 1 四模型中前两类的实际文件身份

- 第一类是 DSH `quant_factor_ashare/ashare_factor_strategy.py`、`strategy_v2.py`、`run_final.py` 对应的 F17。F17 是等权因子线性评分，不是已经拟合过回归系数的线性回归模型。
- 第二类的完整版本是 DSH `quant_factor_ashare/ml_policy/tools/f17_weight_lab.py` 和 `f17_weight_deploy.py`，配套 `F17_WEIGHT_REGULARIZATION_REVIEW.md` 及 `artifacts/f17_weight_lab_full_*`、`f17_weight_deploy_*`。前置 `f17_ridge_probe.py` 仅检验 IC，没有组合回撤结果，不宜拿它代替完整第二类模型。
- Review `robust_factor_core.py` 的 `weight_band=.25` 是组合持仓目标的不交易带，降低交易量；它不估计因子系数，不能称为 F17 因子权重正则化。应作为第一类 F17 的执行修订旁支。

下文路径简写：DSH=`workspace/source_workspace/quant_factor_ashare`；REVIEW=`workspace/review_workspace/quant_factor_ashare_review`。

## 2 F17 的确切定义与方向

`DSH/strategy_v2.py:82`、`REVIEW/run_research_review.py:24` 固定名单。原始因子已在函数内设定符号，合成时一律同号加权。HIGH52、RES_MOM 即使某段 IC 为负，也没有自动反号。

|因子|代码实际定义及偏好方向|源位置 ashare_factor_strategy.py|
|---|---|---|
|TURN|负的21日平均换手率，偏低换手|604|
|MAX5|负的过去21日最高5个日收益均值，偏低彩票性|547|
|INTRADAY|负的21日日内收益复合值，日内反转|650|
|ILLIQ|log1p(21日平均绝对收益/成交额×10⁸)，偏高非流动性|566|
|COMMOVE|60日个股收益与剔除自身行业收益的相关，正向|670|
|TURN_SHOCK|负的5日均换手/60日均换手减1，偏低冲击|681|
|REV_1W|负的5日价格收益，周度反转|532|
|ISKEW|负的60日原始日收益偏度，实际不是回归残差偏度|627|
|IVOL|负的市场模型残差波动年化估计，252日窗口|552|
|OVERNIGHT|21日隔夜收益复合值，正向|644|
|LEAD_LAG|负的个股收益对滞后一期市场收益的协方差/市场方差|656|
|CO_SKEW|代码滚动矩构造的负市场协偏度代理|582|
|BETA|负的252日市场 beta，偏低 beta|561|
|REV_LT|−(P[t−252]/P[t−756]−1)，实际为跳过最近一年的两年区间反转|542|
|SIZE|负对数流通市值，之后仍经市值中性化|572|
|HIGH52|当前收盘/252日最高价，正向|577|
|RES_MOM|过去231日市场模型残差和再滞后21日，正向|639|

预处理在 `ashare_factor_strategy.py:753–852`：股票池掩码 → 中位数±3×1.4826×MAD 去极值 → 行业与对数流通市值残差化 → 截面 z-score → 再掩码。行业为当前静态代理。`neutralize` 是行业内去均值加 FWL 一元残差回归，并没有 ridge 惩罚。SIZE 的原始规模暴露经市值中性化后不再是未经处理的纯规模因子；不应在论文中声称组合在该处保留完整小盘 beta。

`strategy_v2.py:161–177` 读取预处理因子，对于每只股票按可用因子数平均：S_it=Σ m_ikt z_ikt / Σ m_ikt。所有17个可用时每个权重1/17≈5.882%；缺失时分母变化。这里的因子权重与组合股票权重是两个层次。

基线组合 `run_final.py:57–58`：REC 为50只、等权、buffer=0.6、每21日决策、单股3%上限；早期 BASE 为buffer=0、cap=5%。正常满50股时目标每股2%，3%是上限而非固定仓位。排名缓冲表示老持仓仍在前80名即可保留，并按最新排名补足50只。此设计旨在保留横截面信息、避免估计17个自由系数，同时控制边界换手；这些是设计理由，不是已证明的普适最优参数。

股票池 `ashare_factor_strategy.py:372`：交易且上市满250日，剔除 ST、流通市值最小30%以及形成日涨停代理。样本面板13,408,542行、5,221证券、2010-01-04至2026-09-28。数据与研究选择已反复使用，因此2023年后应称重复验证期。

## 3 第一类模型必须区分旧引擎与修订引擎

归档 `DSH/output/final_summary.json`：后期 F17 多头2023–2026原表 CAGR21.53%、Sharpe1.0171、MDD−25.43%、Calmar0.847。根目录《最终结论》的18.40%为更早无该配置的结果，不能混成同一最终版本。

REVIEW保留同一F17分数后以统一统计口径复算。`results/review_results.json` 中2023-01-01至2026-09-28为906日；Sharpe扣固定年化2% rf。

|诊断版本|CAGR|Sharpe|日频最大回撤|Calmar|
|---|---:|---:|---:|---:|
|旧引擎复现 单相位0|21.525281%|1.018090|−25.427541%|0.846534|
|修正月度引擎 单相位0|14.006588%|0.942256|−16.136249%|0.868020|
|修正月度加不交易带 单相位0|13.665829%|0.940113|−15.587512%|0.876717|
|修正月度 21相位等初始资金合成|12.772470%|0.844373|−16.663295%|0.766503|
|修正月度加不交易带 21相位合成|12.420428%|0.846349|−16.008365%|0.775871|

旧引擎把目标向前填充后每日恢复权重，故与描述的月度调仓不同。新引擎仅决策次日开盘交易，保留持仓漂移并加入涨跌停/无价冻结、滞后ADV1%容量、现金预算、历史印花税。佣金2.5bp单边、滑点10bp单边；卖出印花税在2023-08-28前10bp，之后5bp。每相位1,000万元，21相位合成含义为2.1亿元，不能当成1,000万元单账户。

这些修订结果都显式允许陈旧价格估值；严格模式因持仓超过60日缺估值而停止。相位0最高陈旧持仓8.3908%；全部候选最高约21.28%。因此新净值也只是执行研究诊断，不能认定为实盘可复制收益。相位合成Sharpe增量0.001976，配对stationary bootstrap 95%区间，块长21为[−0.033245,0.032638]，块长63为[−0.036085,0.033531]，没有改善的统计证据。

## 4 第二类模型的目标函数与时间设计

模型共享17个缓存因子，标签为形成日t后t+1开盘到t+22开盘的21交易日收益。每21日一个形成日，缓存193×5,221×17。年度refit，过去5年滚动窗，退出日严格早于测试年的1月1日，至少24个训练形成日。开发评价2016–2022为81期；重复验证2023-01-19至2026-08-12为42期。

训练每个complete-case截面重新标准化x，对标签截面rank再z-score，之后把股票行堆叠。因而训练损失按股票观测等权，并非日期等权；大截面占更大权重。

以G=X'X/n、c=X'y/n，岭解为β=(G+λI)⁻¹(c+λβ₀)。β₀=0为ridge，β₀=(1/17,…,1/17)为ridge_eq。后者惩罚||β−β₀||²，体现向原等权先验收缩。系数用于排名，不要求和为1，也不是股票资金分配权重。

- OLS无惩罚。
- ridge/ridge_eq λ网格[.003,.01,.03,.1,.3,1,3,10,30,100,300,1000]，以内层平均验证RankIC选。
- lasso 使用L1；elastic net使用l1_ratio=.5；adaptive lasso函数存在，但归档full模型列表没有它，不能称本次已跑出其最终结果。
- Kozak式线性收缩使用每个因子top/bottom20%多空因子收益F，β=(Σ+γI)⁻¹μ，γ网格[.01,.03,.1,.3,1,3,10,30,100]，用验证矩条件R²选。它是针对17个因子收益的简化迁移，不是原论文完整SDF的精确复现。
- Brandt式先求参数组合 r_p=r_eq+Gθ 的样本Sharpe，θ归一化为单位L2，再对单位等权向量收缩，比例a∈[0,.25,.5,.75,1]以内层验证Sharpe选。部署将θ作为选股分数，最终十分位多空组合不是原论文的直接参数资金权重，应明确是方法启发的适配。

实际 `time_folds(n,3)` (`f17_weight_lab.py:185–195`) 将数据分成3块，但 `range(1,3)` 只给2个扩展窗验证折，不能照旧报告写3个验证折。年度边界purge存在；内层CV未显式清除退出日在下一验证形成日之后的训练标签。21日步长、t+22退出的组合存在边界标签可得性问题，须在局限中交代。

逐年超参原值存在JSON。验证期ridge_eq λ：2023=300，2024=1000，2025=1000，2026=1；Brandt a全为1，输出与等权同序。不能事后另推荐λ=.1–1并说这是同一验证结论。零先验ridge大λ只让β≈c/λ；排名不受共同缩放影响，它不等于原等权，也不能简单解释为不再拟合。

## 5 第二类模型的核验结果

主比较采用full-coverage归档口径：每个因子在当日有值股票内重新标准化，缺值置截面均值0，然后点乘已估β；production_f17仍为可用因子平均。组合是头尾各10%等权，每腿收益差；它不是F17原来的top50纯多头。未提供融券可得性与融券成本。

25bp成本是两腿新进入股票名称比例之和乘0.0025，未计权重漂移、资金借贷、冲击或真实订单。Sharpe rf=0、年化周期数252/21=12。下表MDD、Calmar是本次由保存系数与缓存补算，净值含初始1，观察频率为21日端点，不可当日频账户回撤。

|模型 验证42期 25bp|RankIC|CAGR|Sharpe|21日端点MDD|21日端点Calmar|
|---|---:|---:|---:|---:|---:|
|production F17|.113677|15.834711%|.858111|−21.345966%|.741813|
|同口径equal|.114436|17.802798%|.942462|−20.003512%|.889984|
|OLS|.099709|9.208652%|.529904|−24.703696%|.372764|
|ridge 零先验|.112244|11.406005%|.585389|−25.617086%|.445250|
|ridge 向等权|.113343|17.706142%|.931404|−20.204060%|.876366|
|lasso|.101080|4.095801%|.291707|−30.862418%|.132712|
|elastic net|.103395|7.151172%|.416942|−26.941551%|.265433|
|Kozak式收缩|.099869|11.850308%|.723138|−17.053307%|.694898|
|Brandt式全收缩|.114436|17.802798%|.942462|−20.003512%|.889984|

ridge_eq相对equal验证IC差−.00109276，HAC(3)t=−.855991；净Sharpe略低。因此没有稳健超过等权的证据。此处HAC以21日形成期为单位，不是每日lag3。

complete-case归档equal IC=.103108、LS毛Sharpe=.442472，ridge_eq IC=.105647、毛Sharpe=.608738，Brandt=.106962/.621145。该表不可作纯权重增量证明：`f17_weight_lab.py:487` 把equal覆盖为原X.mean，而其他模型用complete-case截面内重新标准化的X；预处理也随模型变化。Brandt验证a=1但complete-case仍不同，恰好揭示这点。推荐正文以full-coverage同处理比较为主，complete-case列作历史诊断。

此外 `f17_weight_deploy.py:133` 的base=elig & finite(Y)先按未来标签可用性筛选，之后才标准化评分；`ls_stats:83`也按标签有无决定选股域。故“full-coverage”仍仅指标签可观察样本，不等于完全可部署；这会排除未来无价股票且改变全截面标准化。论文必须将结果表标为历史标签组合诊断，不能照原报告称已部署投资组合。

## 6 建议打包的来源和依赖

第一模型目录建议完整保留以下轻量源文件：

- DSH `ashare_factor_strategy.py`, `strategy_v2.py`, `factors_v2.py`, `run_final.py`, `build_panel.py`, `run_all.py`；可追加原顶层全部研究 `.py` 作为历史附件，避免遗漏被否定实验。
- DSH `output/final_summary.json`, `output/final_multiwindow.json`, `output/summary_LSY.json`, `output/summary_FULL.json`，以及原说明文档。
- REVIEW `robust_factor_core.py`, `run_research_review.py`, `test_robust_factor_core.py`, `README.md`, `审计报告.md`；原代码读取 `source_snapshot/ashare_factor_strategy.py` 与 `source_snapshot/strategy_v2.py`，这两冻结副本须保留对应层级。
- REVIEW `results/performance.csv`, `review_results.json`, `legacy_phase0.csv`, `monthly_buffer_p00.csv`, `monthly_buffer_weightband_p00.csv`, `equal_capital_phase_blends.csv`, `candidate_returns.csv`；若容量许可全部results轻量CSV/JSON。

第一类复现全部行情依赖DSH `data/panel.pkl`、`data/prep/{17因子}.pkl`、`data/raw_daily/*.pkl`，体积大；不附时必须清楚声明包为源码与结果归档，不是离线完整行情重跑包。严格重跑会因旧价数据问题停止；`--allow-stale`只重现诊断。

第二模型建议保留：

- `ml_policy/tools/f17_weight_lab.py`, `f17_weight_deploy.py`, `f17_ridge_probe.py`。
- `ml_policy/artifacts/f17_weight_lab_full_weights.json`, `_metrics.json`, `_metrics.csv`, `_scores.npz`, `f17_weight_deploy_metrics.json`, `.csv`, `f17_ridge_probe_summary.json`, `.csv`, `f17_ridge_probe_ic.csv`。
- `ml_policy/artifacts/f17_panel_cache.npz`仅29.37MB，是本次复算的充分输入，值得一起附入。它包含X/Y/elig/score/date/pos，不是原始行情。
- 本次 `verify_linear_metrics.py`、`linear_models.json`、`linear_weight_performance_verified.csv`、`linear_weight_period_returns.csv`。
- `F17_WEIGHT_REGULARIZATION_REVIEW.md`保留为原始研究记录，同时附本核查说明以避免旧措辞误读。

原脚本存在绝对路径，移机需设置或改成配置路径；不能包装成无需配置即可运行。`f17_weight_lab.load_cache()`还读`data/prep/TURN.pkl`的完整交易日期索引以确定标签退出日，若只附缓存而要重新训练，需额外提供该索引或明确原路径依赖。Python依赖numpy、pandas、scipy、scikit-learn；原pandas3 pickle读取还用pyarrow及Windows DLL设置。纯结果复算脚本仅需numpy/pandas，但当前引用原始脚本函数与本机路径，打包时可用适配入口读取包内路径。

建议正文结论：F17的等权先验在现有17个标准化特征和有限时间样本内提供了竞争力强的参照。更复杂权重对开发期的拟合收益不能直接外推；在一致预处理和相同评价域中，向等权收缩没有显著改善，完全收缩回等权只是重现基准。执行修订、数据覆盖与标签可得性问题决定这些结果只能支持研究方法比较，不能证明实盘超额收益。
