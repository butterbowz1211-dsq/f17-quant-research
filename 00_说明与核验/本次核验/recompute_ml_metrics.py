# Research project author: Gilmour Jia
"""Audit saved return series only. Never imports or runs source strategies.

Daily metric convention: rf=0, 252 sessions, sample standard deviation,
initial NAV=1 included in drawdown, CAGR / abs(max drawdown) for Calmar.
Periodic NN label diagnostics are reported separately: archival 22-day
annualization and corrected 21-session holding-interval annualization.
Neither is an executed daily portfolio.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
POL = Path(r'workspace/source_workspace\quant_factor_ashare\ml_policy')
NN = Path(r'workspace/review_workspace\quant_factor_ashare_review\nn_v2')

def metrics(r, annual_periods=252):
    r = np.asarray(r, float)
    if not len(r) or not np.isfinite(r).all():
        raise ValueError('Missing returns cannot be silently dropped')
    nav = np.r_[1.0, np.cumprod(1+r)]
    dd = nav/np.maximum.accumulate(nav)-1
    cagr = nav[-1]**(annual_periods/len(r))-1
    vol = r.std(ddof=1)*np.sqrt(annual_periods)
    return dict(n=len(r), cagr=float(cagr), volatility=float(vol),
                sharpe_standard=float(r.mean()*annual_periods/vol),
                sharpe_cagr_over_vol=float(cagr/vol),
                max_drawdown=float(dd.min()), calmar=float(cagr/abs(dd.min())) if dd.min()<0 else None,
                hit=float((r>0).mean()), total_return=float(nav[-1]-1))

def main():
    daily_rows, curves = [], []
    source_paths=list((POL/'artifacts').glob('*_daily.csv'))
    source_paths+=list((POL/'artifacts').glob('engine_daily*/*.csv'))
    for p in sorted(source_paths):
        frame = pd.read_csv(p)
        if 'net' not in frame: continue
        dates = pd.to_datetime(frame.iloc[:,0])
        model=p.stem.replace('_daily','') if p.parent.name=='artifacts' else p.parent.name+'__'+p.stem
        row = dict(family='policy', model=model, source=str(p),
                   start=str(dates.iloc[0].date()),end=str(dates.iloc[-1].date()),
                   cost_status='source_hard_executor_net', annual_periods=252)
        row.update(metrics(frame.net))
        daily_rows.append(row)
        if p.name in ['p1_fold0_daily.csv','p1_fold1_daily.csv',
                       'policy_v3_chrono_score_fixedT06_fold0_k500_daily.csv',
                       'policy_holdout_2013_2020_fold0_kall_daily.csv',
                       'policy_holdout_2013_2020_ew_all_daily.csv',
                       'policy_holdout_2013_2020_vw_all_daily.csv']:
            nav=np.cumprod(1+frame.net.to_numpy())
            for date,r,n in zip(dates, frame.net, nav):
                curves.append(dict(model=row['model'],date=str(date.date()),return_=r,nav=n))
    v1=NN/'analysis/research_v1/gross_portfolio_v1'
    for p in sorted(v1.glob('*_daily_gross.csv')):
        frame=pd.read_csv(p)
        col='net' if 'net' in frame else 'return'
        if col not in frame:
            col='gross_return' if 'gross_return' in frame else 'ret'
        if col not in frame: continue
        dates=pd.to_datetime(frame.iloc[:,0])
        row=dict(family='nn_v1',model=p.stem,source=str(p),start=str(dates.iloc[0].date()),
                 end=str(dates.iloc[-1].date()),cost_status='frictionless_gross_daily',annual_periods=252)
        row.update(metrics(frame[col]))
        daily_rows.append(row)
    pd.DataFrame(daily_rows).to_csv(HERE/'ml_daily_metrics_recomputed.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(curves).to_csv(HERE/'ml_selected_nav.csv',index=False,encoding='utf-8-sig')

    periodic, diagnostic, periodic_curves = [], [], []
    for version in ('v4','v5'):
        root=NN/f'derived/research_model_{version}'
        ps=[pd.read_csv(root/f'predictions_{y}.csv') for y in range(2021,2026)]
        pred=pd.concat(ps,ignore_index=True)
        result=[]
        selections={name:[] for name in ('NN','Ridge','F17')}
        for date,g in pred.groupby('formation_time',sort=True):
            valid=g.label_reason.eq('ok') & np.isfinite(g[['NN','Ridge','F17','label']]).all(axis=1)
            good=g[valid]
            if len(good)<100: continue
            rec={'date':date,'n':len(good),'all':good.label.mean()}
            for name in ('NN','Ridge','F17'):
                chosen=good.nlargest(50,name)
                before=g[np.isfinite(g[name])].nlargest(50,name)
                missing=(~before.label_reason.eq('ok') | ~np.isfinite(before.label)).sum()
                diagnostic.append(dict(version=version,date=date,model=name,
                                       rows=len(g),valid_rows=len(good),
                                       missing_labels_before_selection=int(missing),
                                       replacements_due_to_label_filter=len(set(chosen.symbol)-set(before.symbol))))
                rec[name+'_top']=float(chosen.label.mean())
                rec[name+'_ic']=float(good[name].rank().corr(good.label.rank()))
                selections[name].append(set(chosen.symbol))
            result.append(rec)
        result=pd.DataFrame(result)
        result.to_csv(HERE/f'nn_{version}_periodic_readout.csv',index=False,encoding='utf-8-sig')
        for name in ('NN','Ridge','F17'):
            nav=np.cumprod(1+result[name+'_top'])
            for date,r,n in zip(result.date,result[name+'_top'],nav):
                periodic_curves.append(dict(version=version,model=name,date=date,period_return=r,nav=n))
            tos=np.array([1-len(a&b)/50 for a,b in zip(selections[name][:-1],selections[name][1:])])
            for convention,days in [('source_22',22),('holding_interval_21',21)]:
                for label,r in [('gross_all',result[name+'_top'].to_numpy()),
                                ('gross_matched57',result[name+'_top'].to_numpy()[1:]),
                                ('net25_matched57',result[name+'_top'].to_numpy()[1:]-tos*0.0025)]:
                    row=dict(family='nn_periodic',version=version,model=name,readout=label,
                             annualization=convention,annual_periods=252/days,
                             start=result.date.iloc[0 if label=='gross_all' else 1],end=result.date.iloc[-1],
                             rank_ic=float(result[name+'_ic'].mean()),mean_turnover=float(tos.mean()),
                             execution_accepted=False)
                    row.update(metrics(r,252/days));periodic.append(row)
    pd.DataFrame(periodic).to_csv(HERE/'nn_periodic_metrics_recomputed.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(diagnostic).to_csv(HERE/'nn_future_label_filter_audit.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(periodic_curves).to_csv(HERE/'nn_periodic_nav.csv',index=False,encoding='utf-8-sig')
    dg=pd.DataFrame(diagnostic)
    summary=dg.groupby(['version','model']).agg(
        dates=('date','size'),dates_affected=('missing_labels_before_selection',lambda x:int((x>0).sum())),
        selections_missing=('missing_labels_before_selection','sum'),
        substituted_names=('replacements_due_to_label_filter','sum')).reset_index().to_dict('records')
    (HERE/'nn_future_label_filter_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'daily_series':len(daily_rows),'periodic_rows':len(periodic),'label_audit':summary},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
