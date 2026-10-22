# Research project author: Gilmour Jia
"""Fixed four-candidate experiment. Reused history, NOT independent validation.

Run without arguments. Only reads DSH caches; outputs under linkage_round2.
Builds 17-factor reference, ten-factor equal, four-group equal, annual rolling
five-year ridge-to-equal group scores. No candidate/parameter selection on results.
"""
from pathlib import Path
import argparse
import gc
import json
import numpy as np
import pandas as pd
from linkage_research import fit_asof
from robust_factor_core import forward_open_return, rank_ic, hac_mean, fdr
from run_research_review import F17, digest, json_clean

HERE=Path(__file__).resolve().parent
OUT=HERE/'linkage_round2'
GROUPS={'liquidity':['TURN','ILLIQ','TURN_SHOCK'],
        'tail_risk':['MAX5','IVOL','ISKEW'],
        'short_horizon':['REV_1W','INTRADAY','OVERNIGHT'],
        'linkage':['COMMOVE']}


def standardized_rank(f):
    r=f.rank(axis=1,pct=True)
    return r.sub(r.mean(axis=1),axis=0).div(r.std(axis=1,ddof=0).replace(0,np.nan),axis=0).astype('float32')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(r'workspace/source_workspace\quant_factor_ashare'))
    args=ap.parse_args(); data=args.source/'data'; OUT.mkdir(exist_ok=True)
    reduced=sum(GROUPS.values(),[])
    spec={'created':'2026-10-10','groups':GROUPS,'candidates':['F17','F10_equal','group_equal','group_ridge'],
          'removed':['CO_SKEW','BETA','REV_LT','SIZE','HIGH52','RES_MOM','LEAD_LAG'],
          'ridge':1.,'training_years':5,'refit':'annual, labels exited strictly before first prediction date',
          'sample_dates':'each 21st trading date from panel origin; matched complete-factor IC sample',
          'minimum_training_dates':36,'evaluation_start':'2015-01-01',
          'status':'historically informed hypothesis; 2023+ already reused; NOT preregistration or new OOS',
          'new_external_linkage_factors':'not included: historical relationship datasets missing'}
    (OUT/'experiment_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    hashes={}; base=None; count=None; sums={}; nobs={}
    for name in F17:
        path=data/'prep'/f'{name}.pkl'; hashes[str(path)]=digest(path)
        f=pd.read_pickle(path)
        if base is None:
            dates,syms=f.index,f.columns; shape=f.shape
            base=np.zeros(shape,'float32');count=np.zeros(shape,'uint8')
            sums={g:np.zeros(shape,'float32') for g in GROUPS}
            nobs={g:np.zeros(shape,'uint8') for g in GROUPS}
        if not f.index.equals(dates) or not f.columns.equals(syms):
            raise ValueError('factor axes differ')
        a=f.to_numpy(); valid=np.isfinite(a); clean=np.where(valid,a,0)
        base+=clean;count+=valid
        for g,names in GROUPS.items():
            if name in names: sums[g]+=clean;nobs[g]+=valid
        print('Loaded',name,flush=True)
        del f,a,valid,clean
    base=np.divide(base,count,out=np.full(shape,np.nan,'float32'),where=count>0)
    baseline=pd.DataFrame(base,index=dates,columns=syms)
    baseline.to_pickle(OUT/'score_F17.pkl')
    # Common availability across all ten factors; no changing implicit group mix.
    complete=np.logical_and.reduce([nobs[g]==len(GROUPS[g]) for g in GROUPS])
    f10=sum(sums.values())/len(reduced)
    f10[~complete]=np.nan
    pd.DataFrame(f10,index=dates,columns=syms).to_pickle(OUT/'score_F10_equal.pkl')
    groups=[]
    for g in GROUPS:
        a=sums[g]/len(GROUPS[g]);a[~complete]=np.nan
        groups.append(standardized_rank(pd.DataFrame(a,index=dates,columns=syms)))
    del sums,nobs,base,count,baseline,f10;gc.collect()
    equal=sum(groups)/len(groups)
    equal.to_pickle(OUT/'score_group_equal.pkl');del equal
    panel=pd.read_pickle(data/'panel.pkl'); hashes[str(data/'panel.pkl')]=digest(data/'panel.pkl')
    # pivot works with categorical symbol columns, then align explicitly.
    op=panel.pivot(index='date',columns='symbol',values='open').reindex(index=dates,columns=syms)
    del panel;gc.collect()
    y=forward_open_return(op,21);del op
    pos=np.arange(0,len(dates)-22,21)
    moments=[]
    for p in pos:
        x=np.column_stack([g.iloc[p].to_numpy() for g in groups])
        ret=y.iloc[p].to_numpy();valid=np.isfinite(x).all(axis=1)&np.isfinite(ret)
        if valid.sum()<100: continue
        xx=x[valid].astype(float)
        yy=pd.Series(ret[valid]).rank(pct=True).to_numpy()
        yy=(yy-yy.mean())/yy.std()
        # Each date has equal influence irrespective of stock-universe size.
        moments.append((dates[p],dates[p+22],xx.T@xx/len(xx),xx.T@yy/len(xx),int(valid.sum())))
    weights=[]; learned=np.full(shape,np.nan,'float32')
    for year in range(2015,dates[-1].year+1):
        cutoff=pd.Timestamp(year=year,month=1,day=1)
        w,n_dates,last_exit=fit_asof(moments,cutoff)
        mask=(dates.year==year)
        learned[mask]=sum(float(w[k])*g.loc[mask].to_numpy() for k,g in enumerate(groups))
        weights.append({'prediction_year':year,'n_dates':n_dates,'last_label_exit':str(last_exit.date()),
                        **dict(zip(GROUPS,map(float,w)))})
    pd.DataFrame(learned,index=dates,columns=syms).to_pickle(OUT/'score_group_ridge.pkl')
    pd.DataFrame(weights).to_csv(OUT/'annual_weights.csv',index=False,encoding='utf-8-sig')
    del groups,learned;gc.collect()
    # Matched cross-sections, matched dates for every IC comparison.
    sample=pos[dates[pos]>='2015-01-01']
    target=y.iloc[sample].where(complete[sample]);del y,complete
    ics={}
    for name in spec['candidates']:
        fac=pd.read_pickle(OUT/f'score_{name}.pkl').iloc[sample].where(target.notna())
        ics[name]=rank_ic(fac,target)
        del fac;gc.collect()
    table=pd.DataFrame(ics).dropna(); table.to_csv(OUT/'paired_ic.csv',encoding='utf-8-sig')
    exit_dates=pd.Series(dates[pos+22],index=dates[pos])
    rows=[]
    for split,mask in [('development_2015_2022',exit_dates.reindex(table.index)<pd.Timestamp('2023-01-01')),
                       ('reused_validation_2023_2026',table.index>=pd.Timestamp('2023-01-01'))]:
        sub=table.loc[mask]
        for name in table:
            st=hac_mean(sub[name],3)
            d=hac_mean(sub[name]-sub.F17,3) if name!='F17' else None
            rows.append(dict(split=split,candidate=name,n=len(sub),IC=st['mean'],t_hac3=st['t'],
                             t_hac6=hac_mean(sub[name],6)['t'],delta_IC=None if d is None else d['mean'],
                             delta_t=None if d is None else d['t'],delta_p=None if d is None else d['p_two_sided']))
    result=pd.DataFrame(rows)
    for _,ix in result[result.candidate!='F17'].groupby('split').groups.items():
        result.loc[ix,'delta_q_BY_3']=fdr(result.loc[ix,'delta_p'],True)
    result.to_csv(OUT/'ic_summary.csv',index=False,encoding='utf-8-sig')
    (OUT/'input_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    (OUT/'score_coverage.json').write_text(json.dumps(json_clean({'common_label_coverage':float(target.notna().mean().mean()),
        'meaning':'fraction of all panel columns with ten factors and forward label on sampled dates'}),indent=2),encoding='utf-8')
    print(result.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
