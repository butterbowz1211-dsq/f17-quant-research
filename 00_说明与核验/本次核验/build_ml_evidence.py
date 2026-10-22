# Research project author: Gilmour Jia
from pathlib import Path
import ast, hashlib, json
import numpy as np
import pandas as pd

H=Path(__file__).resolve().parent
P=Path(r'workspace/source_workspace\quant_factor_ashare\ml_policy')
N=Path(r'workspace/review_workspace\quant_factor_ashare_review\nn_v2')

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def metric_table(d):
    rows=['| 模型与读出 | 区间 | CAGR | 标准Sharpe | 最大回撤 | Calmar | 原CAGR/波动 |',
          '|---|---|---:|---:|---:|---:|---:|']
    for _,r in d.iterrows():
        rows.append(f"| {r.model} | {r.start} 至 {r.end} | {r.cagr:.2%} | {r.sharpe_standard:.4f} | {r.max_drawdown:.2%} | {r.calmar:.4f} | {r.sharpe_cagr_over_vol:.4f} |")
    return '\n'.join(rows)

def main():
    d=pd.read_csv(H/'ml_daily_metrics_recomputed.csv')
    p=pd.read_csv(H/'nn_periodic_metrics_recomputed.csv')
    wanted=['p1_fold0','p1_fold1','beq_fold0','beq_fold1','bf17_fold0','bf17_fold1',
            'engine_daily_fold0__policy_softmax_all','engine_daily_fold0__ew_top500',
            'engine_daily_fold1__policy_softmax_all','engine_daily_fold1__ew_top500',
            'policy_v2_listnet_execfix_fold0_k500','policy_v2_listnet_a001_fold0_k500',
            'policy_v2_chrono_fold0_kall','policy_v2_topg500_fold0_k500',
            'policy_v3_chrono_score_fixedT06_fold0_k500',
            'policy_holdout_2013_2020_fold0_kall','policy_holdout_2013_2020_policy_softmax_all',
            'policy_holdout_2013_2020_ew_top500','policy_holdout_2013_2020_ew_all',
            'policy_holdout_2013_2020_vw_all']
    selected=d[d.model.isin(wanted)].copy()
    selected.to_csv(H/'ml_main_table.csv',index=False,encoding='utf-8-sig')
    # Execute only two inspected pure functions on synthetic inputs, never source imports.
    tree=ast.parse((P/'qlp/dataset.py').read_text(encoding='utf-8'))
    pure=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['daily_log_returns','forward_returns']],type_ignores=[])
    env={'np':np}
    exec(compile(pure,'audited_pure_label_functions','exec'),env)
    toy=np.array([[100.],[200.],[300.],[400.],[500.]])
    got=float(env['forward_returns'](toy,1,3,2)[0])
    expected=float(toy[4,0]/toy[2,0]-1)
    dates=pd.to_datetime(np.load(P/'artifacts/prep_arrays.npz',allow_pickle=False)['dates'])
    crosses=[]
    for endyear in range(2018,2023):
        loc=np.where((dates.year>=2013)&(dates.year<=endyear))[0]
        indices=np.arange(max(int(loc[0]),260),min(int(loc[-1])+1,len(dates)-22),21)
        for t in indices:
            if dates[t+21].year>endyear:
                crosses.append(dict(train_end_year=endyear,formation=str(dates[t].date()),
                                    actual_label_start=str(dates[t].date()),actual_label_end=str(dates[t+21].date())))
    check=dict(source='qlp/dataset.py:153-184',toy_open=toy[:,0].tolist(),start=1,k=2,
               implemented_return=got,documented_t_plus_1_return=expected,
               actual_formula='open[t+21]/open[t]-1 when all endpoints exist',
               intended_formula='open[t+22]/open[t+1]-1',training_boundary_crossings=crosses)
    (H/'policy_label_timing_audit.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    configs={}
    for name in ['p1','policy_v2_listnet_a001','policy_v2_listnet_execfix','policy_v2_chrono',
                 'policy_v2_topg500','policy_v3_chrono_score_fixedT06','policy_v3_chrono_ic_fixedT06',
                 'policy_v3_chrono_ic_anneal','policy_v3_chrono_anneal_last24',
                 'policy_v3_chrono_score_fixedT06_cost0','policy_v3_shuffle_score_fixedT06']:
        f=P/f'artifacts/{name}_report.json'
        if f.exists():
            data=read(f)
            configs[name]={'source':str(f),'config':data.get('config'), 'dataset':data.get('dataset'),
                           'folds':[{k:v for k,v in x.items() if k in ['index','label','top_names','n_train','n_valid','test_window','contamination','test','score_ic']} for x in data['folds']]}
    models={
      'scope':'Evidence audit, no training or strategy-source modifications, 2026-10-21 Asia/Shanghai',
      'current_source_snapshot':{
        'caveat':'These source findings apply to files observed on 2026-10-21. Source code evolved after historical runs. Without an exact archived code hash, findings cannot automatically be attributed to each historical run. Saved-return metrics are separate observed evidence.',
        'files':[{'path':str(P/rel), 'sha256':hashlib.sha256((P/rel).read_bytes()).hexdigest()} for rel in ['run_policy.py','qlp/policy.py','qlp/dataset.py','qlp/execution.py']],
        'daily_rebalancing':{
          'signal_refresh':'qlp/dataset.py:139-150 decision_index uses np.arange(..., rb), rb=21 in saved config.',
          'carry_targets':'run_policy.py:466-475 hold_forward fills out[t:stop]=grid[t] on every intervening date.',
          'wire_to_engine':'run_policy.py:648-658 ensemble grids -> hold_forward -> engine_inputs.',
          'daily_execution':'qlp/execution.py:141 loops over every date; :155 only tests valid NAV and t>0; :162 reads tg[t-1]; :189 blends toward target; :218-219 calculate each-day desired notional and delta.',
          'interpretation':'Signals refresh every 21 sessions; between signals the executor attempts daily restoration toward the carried target. This is execution frequency, not a claim that neural-network training runs daily.'},
        'cap_mismatch':{'training':'qlp/policy.py:60 cap_soft=False; :121-122 applies capped_simplex_soft only when true; run_policy.py:780-787 constructs config without enabling cap_soft.',
                        'readout':'run_policy.py:453-456 projects the network stock output to a hard cap; executor additionally applies feasible constraints.'}},
      'families':{
        '03_policy_original':{'name':'Relationship-enhanced direct-weight MLP p1','factors':10,'input':70,'hidden':[32,16],
          'parameters':2818,'definition':'Shared per-stock encoder; cash scalar logit; stock/cash softmax; hard executor caps stock weights. Signals refresh every 21 sessions but carried targets are rebalanced daily. cap_soft=False in training.',
          'objective_current_code':'Average portfolio arithmetic return minus 25bp turnover surrogate plus 0.5*L2*sum(encoder weight squared)',
          'data':'2010-01-04 to 2026-09-28; 5221 securities; eligible top1000 lagged-liquidity universe',
          'claim_boundary':'Descendant of F17 project, not a neural network using exactly the same 17 original factors.'},
        '04_policy_literature':{'name':'Policy v2/v3 ablations and quasi-holdout',
          'changes':['ListNet auxiliary loss','Chronological training','Top-G softmax with G=500','Fixed temperature 0.6','Validation rank IC stopping','Relative no-trade band'],
          'unrun':['Dynamic previous-weight/account-state branches have no validated result in research log'],
          'claim_boundary':'Exploratory literature-inspired modifications; selected 2021 results do not prove improvement or independent OOS.'},
        '04_nn_return_branch':{'name':'Independent F17-feature return-regression NN v1/v4/v5',
          'v4_input':34,'v4_factors':17,'v4_hidden':[32,16],'v4_parameters':1665,
          'v5_input':42,'v5_factors':21,'v5_parameters':1921,'v5_added':['EP','BP','SP','DIV'],
          'v4_winner':read(N/'derived/research_model_v4/winner.lock.json'),
          'v5_winner':read(N/'derived/research_model_v5/winner.lock.json'),
          'ridge_alpha':1.0,'loss':'Date-equal Huber(delta=1) on return labels scaled by training-date-equal lower weighted median absolute return.',
          'first_selection':'2013-2018 training, 2019-2020 validation, 2021 outer prediction; later years freeze lr/wd and reselect epochs on trailing two validation years.',
          'feature_processing':'Formation universe; 3*MAD*1.4826 winsorization; industry/size residualization; cross-sectional z-score ddof1; within-date median fill; missingness flags.',
          'v1_readout':'2021-01-04 to 2026-09-28 daily frictionless adjusted-price ledger, 50 names, retention rank80, 3% cap, NN softmax versus F17 equal; no future-label filter.',
          'v4_v5_readout':'58 formation dates 2021-01-22 to 2025-12-31; realized labels through 2026-02-03; conditional on future label availability; top50 equal; periodic diagnostic only.',
          'execution_accepted':False}},
      'policy_configs':configs,'daily_metrics':selected.where(pd.notna(selected),None).to_dict('records'),
      'nn_v1_daily_metrics':d[d.family.eq('nn_v1')].to_dict('records'),
      'nn_periodic_metrics':p.to_dict('records'),
      'label_timing_audit':check,'future_label_filter_audit':read(H/'nn_future_label_filter_summary.json'),
      'packaging':{
        '03_core_python':[str(x) for x in [P/'run_policy.py',P/'run_baselines.py',P/'run_baseline_f17.py']]+[str(x) for x in (P/'qlp').glob('*.py')],
        '03_results':[str(P/'artifacts'/n) for n in ['p1_report.json','p1_fold0_daily.csv','p1_fold1_daily.csv','beq_report.json','bf17_report.json','beq_fold0_daily.csv','beq_fold1_daily.csv','bf17_fold0_daily.csv','bf17_fold1_daily.csv','score_diagnostics.json','shuffle_control_fold0.json']],
        '04_policy_python':[str(P/'run_policy.py')]+[str(x) for x in (P/'qlp').glob('*.py')]+[str(P/'tools'/n) for n in ['score_sweep.py','score_readout.py','engine_from_scores.py','band_sweep.py','beta_decompose.py','holdout_readout.py','holdout_compare.py']],
        '04_policy_results':[str(x) for x in (P/'artifacts').glob('policy*_report.json')]+[str(x) for x in (P/'artifacts').glob('policy*_daily.csv')]+[str(x) for x in (P/'artifacts').glob('policy_holdout*.json')]+[str(x) for x in (P/'artifacts').glob('policy*.pt')]+[str(x) for x in (P/'artifacts').glob('policy*_scores.npz')],
        '04_nn_python':[str(x) for x in N.glob('*.py')]+[str(x) for x in (N/'research_ext').glob('*.py')]+[str(N/'analysis/research_v1'/n) for n in ['portfolio_diagnostic.py','portfolio_diagnostic_v4.py','diagnose.py','diagnose_v4.py']],
        '04_nn_config':[str(N/'fees.csv')],
        '04_nn_results_dirs':[str(N/f'derived/research_model_{v}') for v in ['v4','v5']]+[str(N/'analysis/research_v1/gross_portfolio_v1'),str(N/'research_ext/overlay_v4'),str(N/'research_ext/valuation_ic')],
        '04_nn_results_files':[str(N/'research_ext/final_report_metrics.json'),str(N/'analysis/research_v1/summary.json')],
        'large_data_excluded_but_required_for_training':[str(P.parent/'data/panel.pkl'),str(P/'artifacts/prep_arrays.npz'),str(N/'derived/research_training_v4'),str(N/'derived/research_training_v5'),str(N/'derived/valuation_at_formation_v1/valuation.parquet')],
        'environment':'Python3.12; numpy,pandas,scipy,torch,scikit-learn,pyarrow. CPU metric audit only requires numpy,pandas.'}}
    (H/'ml_models.json').write_text(json.dumps(models,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    content='''# 机器学习策略证据核查

核查日期为2026年10月21日，所有原始策略与结果只读。本文件不是回测重跑；指标由保存的日收益或周期收益重新计算。模型源代码是本次观察到的版本，不保证等同于每一个历史实验当时的代码版本。

## 1 四类归档中的机器学习对应关系

第三类以 DSH 的 ml_policy p1 为原始机器学习主线。它在F17项目基础上引入共享MLP、关系邻居、softmax与现金节点，直接形成组合权重；它使用10个基础价格量特征及关系状态共70维输入，不是“17个F17因子直接喂入网络”。第四类保留同一框架的文献驱动v2/v3实验以及后续quasi-holdout。Review 的 nn_v2 是独立的逐股票收益预测分支，使用17个F17因子加17个缺失标记，应该独立展示v1/v4/v5，不得把不同股票池、训练目标、评估方式的收益排成单一改善阶梯。

政策网络10个基础特征是5/20/60/120/252日收益、20/60日波动、20日Amihud、20日最大单日收益、成交额相对60日均值。最后一项在源码命名turnover但不是实际换手率；max5也实际为窗口最大单日收益，不是F17的前五个日收益均值。own值和缺失标记20维，邻居当期及滞后20日值/缺失率40维，关系图有效性1维，市场状态6维，账户状态3维，总计70维。网络70→32→16→1，2818参数（含现金logit），ReLU、dropout0.1。图基于过去120交易日去市场均值收益相关，最少80观察、最多10个正相关邻居。

## 2 权重与训练协议

p1预算记录为三个seed，lr0.005，max40 epoch，patience12，batch16，L2=0.001，温度由1.0到0.1退火，21日决策频率，3%单股目标上限，partial=1，absolute band=0。首先将各seed目标权重平均，再只运行一次成本账本。预算及实际配置以 artifacts/p1_report.json 为准。

固定宽度与dropout/L2意在降低有限日期样本的估计方差；3%上限约束集中风险；top500/topG与温度分别改变入选范围和权重集中度；相对目标权重的不交易带试图降低小额换手。这些是实验约束和候选超参，不是理论推导的最优权重，不能事后用表现最好来证明数值必然正确。

需要进一步限定：PolicyConfig.cap_soft默认False，run_policy.py:780–787未开启，所以3%仅施加于读出与硬执行，训练softmax可能大幅集中（p1验证seed0的mean_top_weight达0.5469）。旧报告声称“训练与执行相同可行集”不符实际。21日是信号更新频率；run_policy.py:466–475把目标权重填至下一信号日，qlp/execution.py:155起每天执行向上日目标的再平衡，并非“非决策日不交易”。这也解释高换手；训练每决策期成本代理与逐日再平衡账本存在差异。

逐日恢复目标的确切调用链为：qlp/dataset.py:139–150的decision_index按rb步长取得信号日期；run_policy.py:466–475的hold_forward在474行执行out[t:stop]=grid[t]；run_policy.py:648–658把集成目标送入hold_forward后送入engine_inputs；qlp/execution.py:141逐交易日循环，155行仅检查t>0与NAV有效，162行读取tg[t−1]，189行blend_targets，218–219行每天计算target_value与delta。因此准确术语是“21日更新信号并逐日尝试向沿用的目标权重再平衡”，并非网络每天重新训练。成交仍受不交易带、整手与流动性约束，不能说每天一定发生交易。

这些源代码事实是2026年10月21日当前源码快照核查。源码曾在历史实验之间修改，若某次实验未保存确切的当时代码哈希，不能把当前实现缺陷自动追溯为该次运行的已证实缺陷。历史报告配置和收益CSV是独立证据，应与当前源码观察分别措辞。ml_models.json的current_source_snapshot保留四个关键文件SHA256及逐项行号。

代码实际目标是逐期组合算术收益减交易成本代理，再加编码器L2；没有log1p。因此应称“净收益代理目标”，不能照旧文档称“精确扣费净值的对数增长最大化”。验证score为近似年化增长减0.5倍波动；该增长近似对算术代理收益使用expm1，并非真实复利账本。参考 run_policy.py:165–208、279–311、315–385，qlp/policy.py:218–244。

fold0实际训练2013–2018、验证2019–2020、测试2021；fold1训练2013–2019、验证2020–2021、测试2022。README/REPRO旧文档说fold3/4对应2021/22已过时，当前run_policy默认first_year=2013。报告中的1970日期是旧序列化问题，实际日收益CSV日期与fold.label支持2021/22。2021/22也已用于反复策略选择，当前论文不得继续称完全未触及样本外。

硬执行器记录整手、佣金2.5bp且最低5元、滑点10bp、卖出印花税2023-08-28前10bp之后5bp、滞后成交额1%参与率、停牌/代理涨跌停限制。它使用复权价记量，不能等同于原始成交价与真实公司行动账本；strict=false允许陈旧估值。训练25bp成本不能被简单称为“必然比所有实盘总成本保守”，因为成本计量对象、方向和市场冲击不同。

## 3 独立核查发现的时序及统计问题

第一，qlp/dataset.py:169–184所实现的 forward_returns 在完整报价下为 open[t+21]/open[t]−1，而文字契约称 open[t+22]/open[t+1]−1。合成价格100、200、300、400、500，t=1、k=2，实际函数给1.0，注释承诺的t+1入场收益为0.6666667。详见 policy_label_timing_audit.json。本次不改源码、不重训；该缺陷限制历史训练与IC结果的可交易解释。

第二，decision_index只按形成日限制折内范围；build_samples未按标签终点purge。合成边界检查显示年末形成日标签可跨训练年末。该问题不代表保存的硬账本收益被伪造，但不能据此声称严格无泄漏前推评估已完成。缺失开盘差分还被置零，可能低估也可能高估收益，旧注释“只能低估”不成立。

第三，policy原报告的Sharpe字段为CAGR/年化波动。本文重新计算标准Sharpe=日均收益/样本标准差×√252，rf=0；原读数另列。回撤纳入初始净值1，Calmar=CAGR/|最大回撤|。不同年、不同股票池、日账本与周期标签读数不可直接排名。原alpha的极大t值尚未全面复核，不宜据旧日志宣称已证明显著alpha。

第四，政策成本代理只接受当天股票轴的previous weights，离开当天股票池的旧持仓出售成本没有完整进入该代理；shuffle状态延续又并非实际时间顺序。chronological改进针对的是这一训练语义问题，其收益变化不能归因于单一因子。

## 4 保存日账本的统一指标

下表的“原CAGR/波动”只为与旧报告对账。原政策年内账本可能从之前的warm-up持仓延续，quasi-holdout读出则从现金开始；不能把差异全部解释为网络能力。

'''+metric_table(selected)+'''

2021原p1有正收益，但2022的同一配置损失明显；top500映射在2021的改善未稳定跨年。v3 chronological+固定T0.6+score的top500在2021降低回撤，但收益低于p1后处理top500。2023-01-03至2026-09-28的探索性holdout中，softmax CAGR约−0.12%，top500约2.08%，同池市值权重约7.41%；这些结果不支持文献驱动修改已经取得稳定超额收益。旧报告对“收益来自再平衡溢价”“梯度互相拉扯”“网络学不会慢变量”等解释只能作候选机制；目前没有充分的专门识别实验。

## 5 F17收益预测网络独立分支

v4为34→32→16→1，1665参数；v5加入EP/BP/SP/DIV及缺失标记变为42→32→16→1，1921参数。损失为日期等权Huber(delta1)，训练标签以训练截面等权绝对收益中位数缩放；AdamW，100轮上限，patience10，三个seed在收益单位平均。lr网格0.0003/0.001，weight_decay网格0.0001/0.001/0.01；两版都选lr0.001/wd0.01；Ridge候选0.0001/0.01/1，均选alpha1。v4/v5各完成45次NN与8次Ridge拟合，主报告保存53 complete。

首折超参只用2019–2020验证集，在输出2021预测前锁定（train.py:164–188）；因此“2021是选型年、纳入2021就污染”是旧报告表述错误。不过整个历史区间已被反复查看，SPEC也显式research_contaminated=true，所以仍只能称历史探索性外推。

v4/v5有58个形成日、175068有效标签；形成日2021-01-22至2025-12-31，最后一期入场2026-01-05，标签终点2026-02-03。不能说2026完全未使用。标签t+1至t+22是21个交易日持有，quick_readout.py按252/22年化，审计CSV同时保留该原口径和252/21敏感性。原周期回撤只在每个形成期末观察，无法捕捉期内日最大回撤；不等同于日频风险指标。

最关键的是 quick_readout.py:35–43 先筛掉未来标签缺失再选top50。实际58期内F17有48期替换119个名额，v4 NN有7期替换8个，v5 NN有7期替换9个。这是未来标签可得性条件化，会以不同强度改变对照。因此v4/v5表应明确标为“条件于标签可得性的周期收益诊断”，不能当成已验收可执行回测。尚不能断言其偏差方向，也不能把F17的较高Sharpe视为无偏比较结果。

v4原22口径毛收益：NN CAGR23.071%、Sharpe1.01369、期末MDD−24.436%；F17 CAGR20.173%、Sharpe1.08442、MDD−19.354%。v5 NN CAGR22.827%、Sharpe1.03507、MDD−23.378%。v5 IC低于v4，但其周期Sharpe和回撤略优；“v4在所有指标最好”“v5被完全证伪”均超出证据。v5对v4的IC下降跨年一致，仍不足以单独证明过拟合机制或估值因子总体无效。

原报告毛收益使用58期、扣25bp名称换手使用57期，差值不应全部归因于费用。nn_periodic_metrics_recomputed.csv提供同57期毛/净配对；名称换手扣费不是双边逐笔费用账本。overlay的2023–2025毛收益与净收益均删首期为34期，其配对t却用35期；引用需说明样本数差别。

v1有独立日账本：2021-01-04至2026-09-28，1391日。它按当时分数选股、不用未来标签筛选，top50/80缓冲，NN softmax而F17等权，毛收益无费用。结果如下，不能同v4周期诊断拼成改善幅度：

'''+metric_table(d[d.family.eq('nn_v1')])+'''

## 6 可复核输出与打包说明

ml_models.json提供模型结构、配置、精确指标与待打包源文件列表；ml_daily_metrics_recomputed.csv含73条日收益序列的统一指标；nn_periodic_metrics_recomputed.csv含36条周期指标；ml_main_table.csv是主表候选。recompute_ml_metrics.py可只读取原CSV重算，build_ml_evidence.py包含合成标签时序核查，均不训练。

应把当前源码快照与历史结果同时归档，明确“结果可复核”与“完整重训可复现”区别。ml_policy的p1报告未保留完整当时代码快照，当前源码已叠加v2/v3功能；p1原文件的carry.npy可复核目标权重，但每份约85MB。最新NN v4/v5完整checkpoint约数百KB、预测CSV各约22MB，建议全部保留，样本包与全市场原始行情独立登记为外部依赖。第四包还应保留v5/overlay负结果，不能只打包最优数字。

原入口建议：ml_policy/run_policy.py --tag p1 --folds 0 1 --first-year 2013 --seeds 0 1 2 --epochs 40 --patience 12 --lr 0.005 --batch 16 --rebalance 21 --temperature 1 --temp-end 0.1 --cap 0.03 --l2 0.001 --dropout 0.1 --partial 1 --band 0；重训应先修复上文标签与边界问题并创建新版本，不能覆盖历史结果。NN入口 train.py --package derived/research_training_v4 --out 新目录 --device cuda --mode research_with_limitations --last-year 2025；v5使用research_ext/build_v5_package.py和train_v5.py。两个分支依赖路径不相同，应保留原目录结构及路径说明。

## 7 可支持的论文结论

研究已完成从固定因子线性打分、正则化估计，到关系增强直接权重学习，再到文献驱动训练与组合层改造，以及逐股票Huber收益预测的多阶段探索。模型复杂度提供了表达交互关系的能力，但保存结果未证明稳定的风险调整收益改善。最有价值的发现是收益预测、组合映射、执行与统计口径必须共同识别；它是实证方法与负结果的研究，而不是已经确认的实盘alpha。未来需要在时序标签、缺失收益处置、样本冻结和执行验收修正后，以未参与选型的时间段重新验证。
'''
    (H/'ml_evidence.md').write_text(content,encoding='utf-8')
    print(json.dumps({'models':str(H/'ml_models.json'),'report':str(H/'ml_evidence.md'),'label_test':check,'table_rows':len(selected)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
