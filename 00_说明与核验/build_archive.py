# Research project author: Gilmour Jia
"""Archive frozen research sources and result evidence; never run source code.
All writes are confined to the package containing this script.
"""
from pathlib import Path
import ast, csv, hashlib, json, re, shutil

PKG=Path(__file__).resolve().parents[1]
AUDIT=PKG.parent/'audit'
DSH=Path(r'workspace/source_workspace/quant_factor_ashare')
REV=Path(r'workspace/review_workspace/quant_factor_ashare_review')
ML=DSH/'ml_policy'; NN=REV/'nn_v2'
A='01_F17线性基线'; B='02_F17正则化与权重重估'
C='03_原始机器学习组合政策'; D='04_文献改进机器学习'
plans={}; omitted=[]

def add(path,folder,base):
    p=Path(path)
    if not p.is_file():
        omitted.append({'source':str(p),'reason':'所选源文件未找到','bytes':''});return
    rel=Path(folder)/p.relative_to(base)
    plans[str(rel)]=p

def globadd(base,pat,folder,limit=20_000_000):
    for p in sorted(base.glob(pat)):
        if p.is_file():
            if p.stat().st_size>limit:
                omitted.append({'source':str(p),'reason':'大体积训练样本或账本数组未附','bytes':p.stat().st_size})
            else: add(p,folder,base)

def select():
    # Source files exclude provider downloaders, credential/account helpers, probes and temp scripts.
    exclude=('fetch_','probe_','retry_','validate_st')
    for p in DSH.glob('*.py'):
        if not p.name.startswith(exclude):add(p,A+'/Source/quant_factor_ashare',DSH)
    for name in ['最终结论.md']:
        add(DSH/name,A+'/Source/quant_factor_ashare',DSH)
    for p in (DSH/'output').iterdir():
        if p.is_file() and p.suffix.lower() in {'.json','.csv','.md','.png','.pkl'}:
            add(p,A+'/Source/quant_factor_ashare',DSH)
    for name in ['robust_factor_core.py','run_research_review.py','audit_all_factors.py','audit_risk_model.py',
                 'test_robust_factor_core.py','README.md','审计报告.md','run_group_research.py',
                 'run_group_execution.py','linkage_research.py','test_linkage_research.py','联动因子第二轮研究.md']:
        add(REV/name,A+'/Review/quant_factor_ashare_review',REV)
    for name in ['ashare_factor_strategy.py','strategy_v2.py','factors_v2.py']:
        add(REV/'source_snapshot'/name,A+'/Review/quant_factor_ashare_review',REV)
    for p in (REV/'results').glob('*'):
        if p.is_file() and p.suffix in {'.json','.csv'}:add(p,A+'/Review/quant_factor_ashare_review',REV)
    for p in (REV/'linkage_round2').glob('*.json'):
        add(p,A+'/Review/quant_factor_ashare_review',REV)
    for p in (REV/'linkage_round2').glob('*.csv'):
        add(p,A+'/Review/quant_factor_ashare_review',REV)
    for name in ['f17_weight_lab.py','f17_weight_deploy.py','f17_ridge_probe.py']:
        add(ML/'tools'/name,B+'/Source/quant_factor_ashare/ml_policy',ML)
    add(ML/'F17_WEIGHT_REGULARIZATION_REVIEW.md',B+'/Source/quant_factor_ashare/ml_policy',ML)
    for p in (ML/'artifacts').glob('f17*'):
        if (('full' in p.name or 'deploy' in p.name or 'ridge_probe' in p.name or p.name=='f17_panel_cache.npz')
                and p.suffix in {'.json','.csv','.npz'}):
            add(p,B+'/Source/quant_factor_ashare/ml_policy',ML)
    # The policy core is intentionally duplicated in both model folders, with the same source hashes.
    for category in [C,D+'/policy_v2_v3']:
        folder=category+'/Source/quant_factor_ashare/ml_policy'
        for name in ['run_policy.py','run_baselines.py','run_baseline_f17.py']:
            add(ML/name,folder,ML)
        for sub in ['qlp','tests']:
            for p in (ML/sub).glob('*.py'):add(p,folder,ML)
        for name in ['README.md','REPRO.md','REPORT_policy_v1.md','RESEARCH_LOG.md','SCORE_PORTFOLIO_AND_SOFTMAX_REVIEW.md']:
            add(ML/name,folder,ML)
    for p in (ML/'tools').glob('*.py'):
        if not p.name.startswith(('_','fetch','f17_')):
            add(p,D+'/policy_v2_v3/Source/quant_factor_ashare/ml_policy',ML)
    for p in (ML/'artifacts').rglob('*'):
        if not p.is_file():continue
        if p.suffix=='.npy' or p.name=='prep_arrays.npz':
            omitted.append({'source':str(p),'reason':'训练/逐日目标权重大数组未附；保存日收益与汇总已附','bytes':p.stat().st_size});continue
        if p.name.startswith('f17_'):continue
        if p.suffix in {'.json','.csv','.pt','.npz'} and p.stat().st_size<12_000_000:
            category=(D+'/policy_v2_v3' if p.name.startswith(('policy_','score_sweep','score_readout','engine_','beta_','band_','exec_'))
                       or p.parent.name.startswith('engine_') else C)
            add(p,category+'/Source/quant_factor_ashare/ml_policy',ML)
    nnfolder=D+'/F17收益预测NN/Review/quant_factor_ashare_review/nn_v2'
    nnexclude=('analyze_akshare','audit_sources','baidu_','probe_','repair_sources','supplement','_tmp','test_supplement')
    for p in NN.glob('*.py'):
        if not p.name.startswith(nnexclude):add(p,nnfolder,NN)
    # Imported data-loader dependencies retained as source, never executed here.
    for name in ['supplement.py','baidu_supplement.py','data_mcp/normalize.py']:
        add(NN/name,nnfolder,NN)
    for p in NN.glob('*.md'):
        if p.name in ['README.md','最终报告_2026-10-21.md','最终结论.md','实验协议.md','因子字典.md',
                      '研究训练放行说明.md','数据与执行审计.md','费用与交易规则.md','数据接口契约.md']:
            add(p,nnfolder,NN)
    for p in NN.glob('*.csv'):add(p,nnfolder,NN)
    for p in (NN/'research_ext').glob('*.py'):add(p,nnfolder,NN)
    for sub in ['derived/research_model_v4','derived/research_model_v5','runs/research_v1',
                'analysis/research_v1','research_ext/overlay_v4','research_ext/valuation_ic']:
        for p in (NN/sub).rglob('*'):
            if p.is_file() and p.suffix.lower() in {'.py','.json','.jsonl','.csv','.pt','.npz','.md'}:
                if p.stat().st_size<=12_000_000:add(p,nnfolder,NN)
                else:omitted.append({'source':str(p),'reason':'大体积辅助文件未附','bytes':p.stat().st_size})
    for sub in ['derived/research_training_v4','derived/research_training_v5']:
        for p in (NN/sub).iterdir():
            if p.is_file():
                if p.suffix=='.json' and p.stat().st_size<1_000_000:add(p,nnfolder,NN)
                else:omitted.append({'source':str(p),'reason':'重训样本不附，保留模型与已保存预测','bytes':p.stat().st_size})
    for name in ['final_report_metrics.json']:
        add(NN/'research_ext'/name,nnfolder,NN)
    evidence=['linear_evidence.md','linear_models.json','linear_weight_performance_verified.csv',
              'linear_weight_period_returns.csv','verify_linear_metrics.py','ml_evidence.md','ml_models.json',
              'ml_daily_metrics_recomputed.csv','ml_main_table.csv','ml_selected_nav.csv','nn_future_label_filter_audit.csv',
              'nn_future_label_filter_summary.json','nn_periodic_metrics_recomputed.csv','nn_periodic_nav.csv',
              'nn_v4_periodic_readout.csv','nn_v5_periodic_readout.csv','policy_label_timing_audit.json',
              'recompute_ml_metrics.py','build_ml_evidence.py']
    for name in evidence:add(AUDIT/name,'00_说明与核验/本次核验',AUDIT)
    for p in [DSH/'data/panel.pkl',DSH/'data/prep/TURN.pkl',REV/'linkage_round2/score_F17.pkl']:
        omitted.append({'source':str(p),'reason':'外部行情/大因子缓存依赖未附；原源目录保留','bytes':p.stat().st_size if p.exists() else ''})
    omitted.extend([{'source':str(DSH/'data/prep/{其余16个F17因子}.pkl'),'reason':'完整训练行情依赖未附','bytes':''},
                    {'source':str(DSH/'data/raw_daily/*.pkl'),'reason':'全市场原始行情未附','bytes':''}])

def scan(path):
    if path.suffix.lower() not in {'.py','.json','.jsonl','.txt','.md','.csv'}:return []
    raw=path.read_text(encoding='utf-8-sig',errors='replace')
    # Never log matched values. Only literal credential-like assignments are flagged.
    pattern=re.compile(r'''(?i)["']?\b(?:[\w]*token|api[_-]?key|access[_-]?key|secret[_-]?key|password|passwd|pwd|authorization)\b["']?\s*[:=]\s*["']([^"'\r\n]+)["']''')
    hits=[]
    # This constant is a Chinese user research-consent sentence, checked against
    # research.py's package authorization gate. It is not an authentication key.
    consent_tree=ast.parse((NN/'research.py').read_text(encoding='utf-8-sig'))
    consent_values={n.value.value for n in consent_tree.body if isinstance(n,ast.Assign)
                    and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)
                    and any(isinstance(t,ast.Name) and t.id=='AUTHORIZATION' for t in n.targets)}
    for no,line in enumerate(raw.splitlines(),1):
        for match in pattern.finditer(line):
            value=match.group(1)
            placeholders=('your_','your-','example','placeholder','<','redacted','none','null','api_token','test','dummy')
            harmless=any(x in value.lower() for x in placeholders) or value in {'','Bearer ','token'}
            old_consent=(path.name in {'experiment_spec.json','run_manifest.json'} and 'research_v1' in str(path)
                         and sum('\u4e00'<=c<='\u9fff' for c in value)>=10
                         and any(w in value for w in ['研究','允许','训练']))
            disposition=('research_consent_text_not_credential' if value in consent_values or old_consent else
                         'placeholder_or_example' if harmless else 'excluded_pending_credential_review')
            hits.append({'source':str(path),'line':no,'disposition':disposition})
    return hits

def csvwrite(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    select(); manifest=[];hits=[];checks=[]
    for rel,src in sorted(plans.items()):
        scan_hits=scan(src);hits.extend(scan_hits)
        if any(h['disposition'].startswith('excluded') for h in scan_hits):
            omitted.append({'source':str(src),'reason':'疑似凭据字面量，未复制且未输出其值','bytes':src.stat().st_size});continue
        if src.suffix=='.json':
            try:json.loads(src.read_text(encoding='utf-8-sig'))
            except Exception as e:
                if src.parent==AUDIT:raise RuntimeError('Invalid audit JSON: '+str(src)) from e
        dst=PKG/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        sh=hashlib.sha256(src.read_bytes()).hexdigest()
        assert sh==hashlib.sha256(dst.read_bytes()).hexdigest()
        manifest.append({'source':str(src),'relative_path':rel.replace('\\','/'),'sha256':sh,'bytes':src.stat().st_size})
        if dst.suffix=='.py':
            try:ast.parse(dst.read_text(encoding='utf-8-sig'));status='PASS';error=''
            except SyntaxError as e:status='FAIL';error=f'line {e.lineno}: {e.msg}'
            checks.append({'relative_path':rel.replace('\\','/'),'status':status,'error':error})
    control=PKG/'00_说明与核验'
    csvwrite(control/'source_manifest.csv',manifest,['source','relative_path','sha256','bytes'])
    csvwrite(control/'credential_scan.csv',hits,['source','line','disposition'])
    csvwrite(control/'omitted_dependencies.csv',omitted,['source','reason','bytes'])
    csvwrite(control/'python_ast_checks.csv',checks,['relative_path','status','error'])
    summary={'copied_files':len(manifest),'copied_bytes':sum(x['bytes'] for x in manifest),
             'python_files':len(checks),'ast_failures':[x for x in checks if x['status']=='FAIL'],
             'credential_candidate_hits':len(hits),'credential_excluded_files':len({x['source'] for x in hits if x['disposition'].startswith('excluded')}),
             'source_files_modified':False,'training_or_backtests_run':False,
             'omitted_records':len(omitted)}
    (control/'archive_verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
