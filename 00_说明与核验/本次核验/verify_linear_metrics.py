# Research project author: Gilmour Jia
"""Read saved F17 evidence; do not refit or change original project files.

Adds 21-trading-day sampled wealth drawdown and Calmar to the archived
factor-weight experiment. These are not daily account drawdowns.
"""
from pathlib import Path
import ast
import json
import math
import hashlib
import numpy as np
import pandas as pd

BASE = Path(r'workspace/source_workspace/quant_factor_ashare')
REVIEW = Path(r'workspace/review_workspace/quant_factor_ashare_review')
ART = BASE / 'ml_policy/artifacts'
OUT = Path(__file__).resolve().parent

def extract_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace

def drawdown_metrics(r):
    x = np.asarray(r, float)
    x = x[np.isfinite(x)]
    nav = np.r_[1., np.cumprod(1 + x)]
    mdd = float(np.min(nav / np.maximum.accumulate(nav) - 1))
    cagr = float(nav[-1] ** (12 / len(x)) - 1)
    return dict(n_periods=len(x), CAGR=cagr, MaxDD_21d=mdd,
                Calmar_21d=cagr/abs(mdd) if mdd else None,
                Sharpe_rf0=float(x.mean()/x.std(ddof=1)*math.sqrt(12)))

def main():
    source = BASE / 'ml_policy/tools/f17_weight_deploy.py'
    ns = extract_functions(source, {'full_score','ls_stats','perf'},
                           {'np':np,'pd':pd,'math':math,'H':21})
    with np.load(ART / 'f17_panel_cache.npz', allow_pickle=True) as z:
        X=z['X'].astype(np.float32);Y=z['Y'].astype(np.float64)
        elig=z['elig'].astype(bool);score_prod=z['score'].astype(np.float64)
        dates=pd.to_datetime([str(s) for s in z['dates']]); pos=z['pos'].copy()
        shape=list(X.shape)
    meta=json.loads((ART/'f17_weight_lab_full_weights.json').read_text(encoding='utf-8'))
    models=['equal','ols','ridge','ridge_eq','lasso','enet','kozak','brandt']
    scores={m:np.full(Y.shape,np.nan,dtype=np.float32) for m in models}
    scores['production_f17']=score_prod.astype(np.float32)
    for j,dt in enumerate(dates):
        base=elig[j] & np.isfinite(Y[j])
        if base.sum()<100: continue
        for m in models:
            info=meta['info'][m].get(str(dt.year))
            if info:
                scores[m][j,base]=ns['full_score'](X[j],base,np.asarray(info['w'],float))
    archive=pd.read_csv(ART/'f17_weight_deploy_metrics.csv')
    rows=[];series={};max_abs_error=0.
    for m,s in scores.items():
        for cost in (0.,25.):
            r,turn,gross=ns['ls_stats'](s,Y,elig,dates,cost)
            series[f'{m}_cost{cost:g}bp']=r
            for split,lo,hi in [('development_2016_2022','2016-01-01','2022-12-31'),
                                ('validation_2023_2026','2023-01-01','2026-12-31')]:
                mask=(dates>=lo)&(dates<=hi)
                p=drawdown_metrics(r[mask])
                old=archive[(archive.model==m)&(archive.cost_bp==cost)&(archive.split==split)].iloc[0]
                error=max(abs(p['CAGR']-old.ls_ann),abs(p['Sharpe_rf0']-old.ls_sharpe))
                max_abs_error=max(max_abs_error,error)
                rows.append(dict(split=split,model=m,cost_bp=cost,
                                 formation_start=str(dates[mask][0].date()),
                                 formation_end=str(dates[mask][-1].date()),
                                 ic=float(old.ic),mean_turnover=float(old.turnover),
                                 original_CAGR=float(old.ls_ann),original_Sharpe=float(old.ls_sharpe),
                                 **p))
    pd.DataFrame(series,index=dates).to_csv(OUT/'linear_weight_period_returns.csv',index_label='formation_date')
    pd.DataFrame(rows).to_csv(OUT/'linear_weight_performance_verified.csv',index=False)
    review=json.loads((REVIEW/'results/review_results.json').read_text(encoding='utf-8'))
    daily_rows=[x for x in review['performance'] if x['window']=='reused_validation_2023_2026' and x['phase']==0]
    perf_ns=extract_functions(REVIEW/'robust_factor_core.py',{'performance'},{'np':np})
    daily_checks=[]
    files={'original_daily_reset':'legacy_phase0.csv','monthly_buffer':'monthly_buffer_p00.csv',
           'monthly_buffer_weightband':'monthly_buffer_weightband_p00.csv'}
    for row in daily_rows:
        dd=pd.read_csv(REVIEW/'results'/files[row['strategy']],index_col=0,parse_dates=True)
        pp=perf_ns['performance'](dd.loc[dd.index>='2023-01-01','net'])
        error=max(abs(pp[k]-row[k]) for k in ['CAGR','Sharpe','MaxDD','Calmar'])
        daily_checks.append({'strategy':row['strategy'],'maximum_metric_error':error})
    dd=pd.read_csv(REVIEW/'results/equal_capital_phase_blends.csv',index_col=0,parse_dates=True)
    for name,row in review['phase_blends'].items():
        pp=perf_ns['performance'](dd.loc[dd.index>='2023-01-01',name])
        error=max(abs(pp[k]-row[k]) for k in ['CAGR','Sharpe','MaxDD','Calmar'])
        daily_checks.append({'strategy':name+'_blend','maximum_metric_error':error})
    relevant_paths=[source,ART/'f17_panel_cache.npz',ART/'f17_weight_lab_full_weights.json',
                    ART/'f17_weight_deploy_metrics.csv',REVIEW/'results/review_results.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in relevant_paths}
    payload={
      'verified_on':'2026-10-21',
      'scope':'Archived evidence reconstruction only; no model training; no source mutation',
      'second_model_resolution':'f17_weight_lab and f17_weight_deploy are full weight regularization experiments; f17_ridge_probe is preliminary; weight_band is portfolio execution optimization, not factor coefficient regularization',
      'F17':meta['names'],
      'linear_baseline_factor_weights':'1 / count of available factors; 1/17 only if all 17 are observed',
      'linear_baseline_portfolio':'long-only 50 names, rank buffer to 80, intended 21-day rebalance, equal weights, cap 3%',
      'cache_shape':shape,'cache_first_formation':str(dates.min().date()),
      'cache_last_formation':str(dates.max().date()),'positions_step_unique':np.unique(np.diff(pos)).tolist(),
      'complete_case_archived':json.loads((ART/'f17_weight_lab_full_metrics.json').read_text(encoding='utf-8')),
      'weight_full_coverage_verified':rows,
      'max_absolute_reproduction_error_CAGR_Sharpe':max_abs_error,
      'weights_and_hyperparameters_by_year':meta,
      'legacy_final_archived':json.loads((BASE/'output/final_summary.json').read_text(encoding='utf-8')),
      'corrected_daily_phase0':daily_rows,'corrected_daily_phase_blends':review['phase_blends'],
      'daily_csv_reproduction_checks':daily_checks,
      'corrected_daily_config':review['config'],
      'bootstrap_paired_weightband_Sharpe':review['paired_weightband_minus_baseline_sharpe'],
      'limitations':[
       'Original F17 engine carries target weights forward then resets daily, despite intended monthly rebalance.',
       'Corrected engine strict valuation mode fails; reported correction runs explicitly allow stale held prices.',
       'Full-coverage scores standardize only elig & finite future label; reported LS results are ex-post labelled-cross-section diagnostics, not executable accounts.',
       'Weight-model Sharpe uses rf=0 and 12 periods/year; daily corrected F17 Sharpe uses rf=2% and 252 days/year.',
       'Weight-model drawdowns newly computed here use 21-day endpoints and include initial NAV=1; they cannot measure intra-period daily drawdowns.',
       'time_folds(n,3) partitions three blocks but produces only two expanding-window validation folds.',
       'Annual training labels are purged at annual cutoff; inner CV does not explicitly purge labels crossing validation formation dates.',
       'Complete-case equal score is overwritten with original X mean, while fitted scores use complete-case restandardization; not a pure weight-only comparison.',
       'Saved final weight experiment does not include adaptive lasso, despite function being implemented.',
       '25bp cost uses turnover in names, not weight drift, short financing, stock lending availability or full execution fees.',
       '2023-2026 has been repeatedly inspected in research and is not untouched independent out-of-sample.',
      ],
      'source_sha256':hashes,
    }
    (OUT/'linear_models.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'max_error':max_abs_error,'shape':shape,'rows':len(rows),'daily_phase0':daily_rows,
       'validation_net': [r for r in rows if r['split']=='validation_2023_2026' and r['cost_bp']==25]},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
