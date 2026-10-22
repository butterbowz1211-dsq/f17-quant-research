# Research project author: Gilmour Jia
"""Execute the fixed four-score experiment using the audited event engine.

Default strict valuation. --allow-stale is only a conditional diagnostic.
Three prespecified phases (0,7,14); not a complete 21-phase robustness exercise.
"""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from robust_factor_core import ExecutionConfig,event_backtest,performance,bootstrap_sharpe
from run_research_review import json_clean

HERE=Path(__file__).resolve().parent
OUT=HERE/'linkage_round2'
NAMES=['F17','F10_equal','group_equal','group_ridge']


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(r'workspace/source_workspace\quant_factor_ashare'))
    ap.add_argument('--allow-stale',action='store_true')
    args=ap.parse_args();data=args.source/'data'
    out=OUT/('execution_diagnostic' if args.allow_stale else 'execution_strict');out.mkdir(parents=True,exist_ok=True)
    panel=pd.read_pickle(data/'panel.pkl')
    dates=pd.DatetimeIndex(sorted(panel.date.unique()))
    syms=pd.Index(sorted(panel.symbol.astype(str).unique()),name='symbol')
    row=dates.get_indexer(panel.date);col=syms.get_indexer(panel.symbol)
    def wide(name):
        a=np.full((len(dates),len(syms)),np.nan,'float32');a[row,col]=panel[name].to_numpy('float32')
        return pd.DataFrame(a,index=dates,columns=syms)
    op,cl=wide('open'),wide('close');adv=wide('amount').rolling(20,min_periods=15).mean().shift(1)
    traded=(wide('volume')>0)&cl.notna();st=wide('is_st').fillna(0)>.5
    mc=wide('raw_close')*wide('outstanding_share')
    first=np.argmax(cl.notna().to_numpy(),axis=0)
    age=np.arange(len(dates))[:,None]-np.where(first==0,-1e6,first)[None,:]
    eligible=traded&~st&(age>=250);eligible &= mc.where(eligible).rank(axis=1,pct=True)>.3
    # Preserve previous review's decision universe for a controlled comparison.
    old_lim=np.where(syms.str.startswith(('sz3','sh688')),.20,.10)
    eligible &= ~(op/cl.shift(1)-1 >= np.where(st,.05,old_lim)-.005)
    del panel,row,col,mc,age;gc.collect()
    ro=np.full(cl.shape,np.nan,'float32');pc=ro.copy()
    dpos={d:i for i,d in enumerate(dates)};spos={s:i for i,s in enumerate(syms)}
    for k,p in enumerate(sorted((data/'raw_daily').glob('*.pkl'))):
        if p.name.endswith('_failed.pkl'):continue
        r=pd.read_pickle(p);ss=r.ts_code.str[-2:].str.lower()+r.ts_code.str[:6]
        ii=pd.to_datetime(r.trade_date,format='%Y%m%d').map(dpos);jj=ss.map(spos)
        ok=ii.notna()&jj.notna();i=ii[ok].to_numpy(int);j=jj[ok].to_numpy(int)
        ro[i,j]=r.loc[ok,'open'].to_numpy('float32');pc[i,j]=r.loc[ok,'pre_close'].to_numpy('float32')
        if (k+1)%1000==0:print('Read raw files',k+1,flush=True)
    lim=np.full(cl.shape,.10,'float32');chi=np.asarray(syms.str.startswith('sz3'))
    star=np.asarray(syms.str.startswith('sh688'));bj=np.asarray(syms.str.startswith('bj'))
    after=np.asarray(dates>='2020-08-24');lim[:,star]=.20;lim[:,bj]=.30;lim[np.ix_(after,chi)]=.20
    special=np.broadcast_to(star|bj,cl.shape).copy();special[np.ix_(after,chi)]=True
    lim=np.where(st&~special,.05,lim)
    up=np.floor(pc*(1+lim)*100+.50001)/100;dn=np.floor(pc*(1-lim)*100+.50001)/100
    known=np.isfinite(ro)&(ro>0)&np.isfinite(pc)&(pc>0)
    buy=traded&known&(ro<up-.0001)&~st;sell=traded&known&(ro>dn+.0001)
    del ro,pc,lim,special,up,dn,known,st,traded;gc.collect()
    sample=dates>='2015-01-01'
    op,cl,eligible,buy,sell,adv=[x.loc[sample] for x in (op,cl,eligible,buy,sell,adv)]
    cfg=ExecutionConfig(allow_stale=args.allow_stale)
    ret={};rows=[];diags={}
    for name in NAMES:
        score=pd.read_pickle(OUT/f'score_{name}.pkl').loc[sample]
        for phase in [0,7,14]:
            key=f'{name}_p{phase:02d}'
            try:
                r=event_backtest(op,cl,score,eligible,buy,sell,adv,cfg,phase)
            except ValueError as exc:
                (out/'failure.json').write_text(json.dumps({'candidate':key,'error':str(exc),'status':'no deployable result'},indent=2),encoding='utf-8')
                raise
            daily=r['daily'];ret[key]=daily.net
            daily.to_csv(out/f'{key}.csv',encoding='utf-8-sig')
            r['events'].to_csv(out/f'{key}_events.csv',index=False,encoding='utf-8-sig')
            diags[key]={'max_stale':float(daily.stale_weight.max()),'cash_mean':float(daily.cash_weight.mean())}
            print(key,performance(daily.loc['2023-01-01':,'net']),flush=True)
        del score;gc.collect()
    ret=pd.DataFrame(ret);ret.to_csv(out/'daily_returns.csv',encoding='utf-8-sig')
    blends={}
    for name in NAMES:
        nav=(1+ret[[f'{name}_p{p:02d}' for p in [0,7,14]]]).cumprod().mean(axis=1)
        net=nav.pct_change(fill_method=None);net.iloc[0]=nav.iloc[0]-1;blends[name]=net
        for split,mask in [('development_2015_2022',net.index<'2023-01-01'),('reused_validation_2023_2026',net.index>='2023-01-01')]:
            rows.append({'candidate':name,'split':split,**performance(net.loc[mask])})
    blends=pd.DataFrame(blends);blends.to_csv(out/'phase_blends.csv',encoding='utf-8-sig')
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False,encoding='utf-8-sig')
    paired={}
    for name in NAMES[1:]:
        paired[name]={str(block):bootstrap_sharpe(blends.loc['2023-01-01':,name],blends.loc['2023-01-01':,'F17'],block=block)
                      for block in (21,63)}
    payload={'allow_stale':args.allow_stale,'phases':[0,7,14],'capital_per_phase':cfg.capital,
             'combined_capital':3*cfg.capital,'diagnostics':diags,'paired_sharpe':paired,
             'limits':['reused validation and historically chosen hypotheses','three phases only',
                       'source universe/PIT/data gaps inherited','fractional adjusted units and approximate execution limits',
                       'individual bootstrap intervals are not familywise search-adjusted']}
    (out/'diagnostics.json').write_text(json.dumps(json_clean(payload),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
