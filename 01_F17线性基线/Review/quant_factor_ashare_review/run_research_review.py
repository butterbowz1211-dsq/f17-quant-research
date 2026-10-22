# Research project author: Gilmour Jia
"""Read existing DSH caches, write only to this independent review directory.

Preserves F17 to isolate execution changes. Cached signals inherit source data
biases; no result from this runner is a new untouched out-of-sample result.
Default: all 21 phases, original engine phase 0, monthly buffer vs monthly band.
"""
import argparse
import ast
from dataclasses import replace, asdict
from pathlib import Path
from types import SimpleNamespace
import gc
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
from robust_factor_core import (ExecutionConfig,event_backtest,performance,hac_mean,
    bootstrap_sharpe,rank_ic,forward_open_return,purged_training_mask,fdr)

HERE=Path(__file__).resolve().parent
F17=['TURN','MAX5','INTRADAY','ILLIQ','COMMOVE','TURN_SHOCK','REV_1W','ISKEW',
     'IVOL','OVERNIGHT','LEAD_LAG','CO_SKEW','BETA','REV_LT','SIZE','HIGH52','RES_MOM']


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()


def load_definitions(path,names,namespace):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def json_clean(x):
    if isinstance(x,dict): return {str(k):json_clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [json_clean(v) for v in x]
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(float,np.floating)): return float(x) if np.isfinite(x) else None
    return x


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(r'workspace/source_workspace\quant_factor_ashare'))
    ap.add_argument('--output',type=Path,default=HERE/'results')
    ap.add_argument('--phases',default=','.join(map(str,range(21))))
    ap.add_argument('--audit-factors',default='TURN,COMMOVE,HIGH52,RES_MOM')
    ap.add_argument('--allow-stale',action='store_true',help='Research valuation carry only, prominently reported')
    ap.add_argument('--capital',type=float,default=10_000_000.)
    args=ap.parse_args()
    source=args.source.resolve(); out=args.output.resolve()
    if out==source or source in out.parents or not (out==HERE or HERE in out.parents):
        raise ValueError('output must be inside this review folder and outside source')
    out.mkdir(parents=True,exist_ok=True)
    phases=sorted(set(map(int,args.phases.split(','))))
    if any(not 0<=p<21 for p in phases): raise ValueError('phases must be 0..20')
    start=time.monotonic()
    data=source/'data'
    inputs={}
    def record(p):
        inputs[str(p.relative_to(source))]=dict(size=p.stat().st_size,sha256=digest(p))
    print('Loading panel and recording source input hashes...',flush=True)
    record(data/'panel.pkl')
    panel=pd.read_pickle(data/'panel.pkl')
    if panel.duplicated(['date','symbol']).any(): raise ValueError('duplicate price keys')
    dates=pd.DatetimeIndex(sorted(panel.date.unique()))
    symbols=pd.Index(sorted(panel.symbol.astype(str).unique()),name='symbol')
    row=dates.get_indexer(panel.date)
    if isinstance(panel.symbol.dtype,pd.CategoricalDtype):
        remap=symbols.get_indexer(panel.symbol.cat.categories)
        col=remap[panel.symbol.cat.codes.to_numpy()]
    else:
        col=symbols.get_indexer(panel.symbol)
    def wide(c):
        a=np.full((len(dates),len(symbols)),np.nan,dtype='float32')
        a[row,col]=panel[c].to_numpy(dtype='float32')
        print(f'  loaded {c}',flush=True)
        return pd.DataFrame(a,index=dates,columns=symbols)
    op,cl=wide('open'),wide('close')
    amount=wide('amount')
    adv=amount.rolling(20,min_periods=15).mean().shift(1).astype('float32')
    del amount
    vol=wide('volume'); traded=(vol>0)&cl.notna(); del vol
    mc=wide('raw_close')*wide('outstanding_share')
    st=wide('is_st').fillna(0)>.5
    first=np.argmax(cl.notna().to_numpy(),axis=0)
    age=np.arange(len(dates))[:,None]-np.where(first==0,-1e6,first)[None,:]
    eligible=traded & ~st & (age>=250)
    eligible &= mc.where(eligible).rank(axis=1,pct=True)>.3
    del age,mc
    panel_symbols=set(symbols)
    nrows=len(panel)
    del panel,row,col;gc.collect()
    # Legacy reconstruction only, to verify published phase-0 results.
    old_lim=np.where(symbols.str.startswith(('sz3','sh688')),.20,.10)
    gap=op/cl.shift(1)-1
    old_up=gap>=np.where(st,.05,old_lim)-.005
    old_eligible=eligible&~old_up
    del gap,old_up

    print(f'Panel {nrows:,} rows, {len(dates)} dates x {len(symbols)} symbols. Loading F17...',flush=True)
    acc=np.zeros(cl.shape,dtype=np.float32);count=np.zeros(cl.shape,dtype=np.uint8)
    for name in F17:
        p=data/'prep'/f'{name}.pkl';record(p)
        f=pd.read_pickle(p)
        if not f.index.equals(dates) or not f.columns.equals(symbols):
            raise ValueError(f'{name}: cache axes mismatch; rebuild required')
        a=f.to_numpy();valid=np.isfinite(a)
        acc+=np.where(valid,a,0);count+=valid
        del f,a,valid
    score=pd.DataFrame(np.divide(acc,count,out=np.full_like(acc,np.nan),where=count>0).astype('float32'),index=dates,columns=symbols)
    factor_coverage=dict(min_nonzero=int(count[count>0].min()),max=int(count.max()),
                         fraction_cells_all17=float((count==17).mean()))
    del acc,count;gc.collect()

    # Tushare raw open/pre_close avoids treating adjusted returns as legal limits.
    # Raw daily files cannot supply official up/down limit quotes or auction depth.
    raw_open=np.full(cl.shape,np.nan,dtype='float32')
    preclose=np.full(cl.shape,np.nan,dtype='float32')
    raw_symbols=set();raw_rows=0
    dpos={d:i for i,d in enumerate(dates)};spos={s:i for i,s in enumerate(symbols)}
    files=sorted((data/'raw_daily').glob('*.pkl'))
    print(f'Checking raw daily universe and execution reference prices ({len(files)} files)...',flush=True)
    for k,p in enumerate(files):
        if p.name.endswith('_failed.pkl'): continue
        r=pd.read_pickle(p)
        required={'ts_code','trade_date','open','pre_close'}
        if not required.issubset(r.columns): raise ValueError(f'{p.name}: missing {required-set(r.columns)}')
        sy=r.ts_code.str[-2:].str.lower()+r.ts_code.str[:6]
        raw_symbols.update(sy);raw_rows+=len(r)
        dd=pd.to_datetime(r.trade_date,format='%Y%m%d')
        ii=dd.map(dpos);jj=sy.map(spos);ok=ii.notna()&jj.notna()
        i=ii[ok].to_numpy(int);j=jj[ok].to_numpy(int)
        raw_open[i,j]=r.loc[ok,'open'].to_numpy('float32')
        preclose[i,j]=r.loc[ok,'pre_close'].to_numpy('float32')
        if (k+1)%1000==0: print(f'  raw files {k+1}/{len(files)}',flush=True)
    # Historical board schedule; official daily limit price input still needed for
    # special relisting/no-limit cases. New IPO excluded at decision, not a guarantee.
    lim=np.full(cl.shape,.10,dtype='float32')
    chinext=np.asarray(symbols.str.startswith('sz3'))
    star=np.asarray(symbols.str.startswith('sh688'))
    bj=np.asarray(symbols.str.startswith('bj'))
    lim[:,star]=.20;lim[:,bj]=.30
    after=np.asarray(dates>=pd.Timestamp('2020-08-24'))
    lim[np.ix_(after,chinext)]=.20
    special=np.broadcast_to(star|bj,cl.shape).copy()
    special[np.ix_(after,chinext)]=True
    lim=np.where(st & ~special,.05,lim)
    # Round to the exchange's one-cent price grid (positive half-up).
    up=np.floor(preclose*(1+lim)*100+.50001)/100
    down=np.floor(preclose*(1-lim)*100+.50001)/100
    known=np.isfinite(raw_open)&(raw_open>0)&np.isfinite(preclose)&(preclose>0)
    buy=traded&known&(raw_open<up-.0001)&~st
    sell=traded&known&(raw_open>down+.0001)
    known_frac=float(known[traded.to_numpy()].mean())
    del raw_open,preclose,lim,special,up,down,known,traded,st;gc.collect()

    old=load_definitions(HERE/'source_snapshot'/'ashare_factor_strategy.py',
                        {'backtest','summarize','rank_ic','newey_west_tstat'},
                        {'np':np,'pd':pd,'F32':'float32','TRADING_DAYS':252})
    weights=load_definitions(HERE/'source_snapshot'/'strategy_v2.py',
                            {'make_weights','_apply_cap','_weight_within'}, {'np':np,'pd':pd})
    W=weights['make_weights'](score,old_eligible,n=50,max_weight=.03,buffer=.6)
    cfg_old=SimpleNamespace(commission_bp=2.5,slippage_bp=10,stamp_duty_bp=5,rf_annual=.02)
    P=SimpleNamespace(dates=dates,symbols=symbols,open=op,close=cl)
    print('Reproducing original phase-0 engine using extracted pure functions...',flush=True)
    legacy=old['backtest'](P,W,cfg_old,'legacy')
    old_daily=pd.DataFrame({k:legacy[k] for k in ['net','gross','cost','turnover']})
    old_daily.to_csv(out/'legacy_phase0.csv',encoding='utf-8-sig')
    del W,legacy;gc.collect()

    windows={'development_2010_2022':dates<='2022-12-31',
             'reused_validation_2023_2026':dates>='2023-01-01',
             'last504_descriptive':np.arange(len(dates))>=len(dates)-504}
    summary=[]
    def add_summary(name,phase,daily):
        for window,mask in windows.items():
            seg=daily.loc[mask,'net']; perf=performance(seg)
            summary.append(dict(strategy=name,phase=phase,window=window,**perf,
                                ann_turnover=float(daily.loc[mask,'turnover'].mean()*252),
                                hac_excess=hac_mean(seg-((1.02)**(1/252)-1),lags=21)))
    add_summary('original_daily_reset',0,old_daily)
    # Match old universe for controlled comparisons. Execution-time restrictions
    # are new; upstream caches still contain source's historical industry and masks.
    cfg=ExecutionConfig(capital=args.capital,allow_stale=args.allow_stale)
    returns={}; ledger_diagnostics={}
    for name,c in [('monthly_buffer',cfg),('monthly_buffer_weightband',replace(cfg,weight_band=.25))]:
        for phase in phases:
            r=event_backtest(op,cl,score,old_eligible,buy,sell,adv,c,phase)
            daily=r['daily'];add_summary(name,phase,daily)
            key=f'{name}_p{phase:02d}';returns[key]=daily['net']
            daily.to_csv(out/f'{key}.csv',encoding='utf-8-sig')
            r['events'].to_csv(out/f'{key}_events.csv',index=False,encoding='utf-8-sig')
            ledger_diagnostics[key]=dict(blocked_orders=int(daily.blocked.sum()),
                    max_stale_weight=float(daily.stale_weight.max()),mean_cash=float(daily.cash_weight.mean()))
            print(f'{key}: reused validation SR={summary[-2]["Sharpe"]:.3f}; elapsed {time.monotonic()-start:.0f}s',flush=True)
    allret=pd.DataFrame(returns)
    allret.to_csv(out/'candidate_returns.csv',encoding='utf-8-sig')
    # Initial equal capital across independent sleeves: aggregate wealth paths,
    # letting sleeve capital drift. Never average phase Sharpe ratios or rebalance
    # sleeve capital daily without charging its costs. Each sleeve uses cfg.capital.
    blend={}
    for name in ['monthly_buffer','monthly_buffer_weightband']:
        navs=(1+allret[[c for c in allret if c.startswith(name+'_p')]]).cumprod()
        nav=navs.mean(axis=1)
        series=nav.pct_change(fill_method=None)
        series.iloc[0]=nav.iloc[0]-1
        blend[name]=series
    pd.DataFrame(blend).to_csv(out/'equal_capital_phase_blends.csv',encoding='utf-8-sig')
    oos=dates>='2023-01-01'
    paired={str(block):bootstrap_sharpe(blend['monthly_buffer_weightband'][oos],
                    blend['monthly_buffer'][oos],block=block) for block in (21,63)}

    print('Auditing factor IC: original vs pairwise ranks, HAC bandwidth and open labels...',flush=True)
    audits=[]
    close_label=cl.shift(-21)/cl-1
    open_label=forward_open_return(op,21)
    for name in filter(None,args.audit_factors.split(',')):
        fac=pd.read_pickle(data/'prep'/f'{name}.pkl')
        for win,mask in [('training_purged',purged_training_mask(dates,'2022-12-31',21)),('reused_validation',oos)]:
            f=fac.loc[mask];yc=close_label.loc[mask];yo=open_label.loc[mask]
            orig=old['rank_ic'](f,yc)
            fixed=rank_ic(f,yc)
            executable=rank_ic(f,yo)
            audits.append(dict(factor=name,window=win,
                original_ic_mean=float(orig.mean()),original_hac5_t=float(old['newey_west_tstat'](orig,5)[1]),
                paired_close_ic_hac21=hac_mean(fixed,21,21),
                next_open_ic_hac21=hac_mean(executable,21,21),
                next_open_ic_hac63=hac_mean(executable,63,21)))
            print(name,win,'IC',round(float(executable.mean()),5),'HAC21 t',audits[-1]['next_open_ic_hac21']['t'],flush=True)
        del fac,f,yc,yo,orig,fixed,executable;gc.collect()
    # No FDR decisions for selected subset: eligible search family must be complete.
    payload=dict(scope='Research reconstruction; validation reused; NOT new OOS or deployable performance',
        source=str(source),source_rows=nrows,source_symbols=len(symbols),raw_symbols=len(raw_symbols),
        raw_symbols_missing_from_panel=sorted(raw_symbols-panel_symbols),raw_rows=raw_rows,
        raw_reference_coverage=known_frac,factor_coverage=factor_coverage,
        config=asdict(cfg),phases=phases,performance=summary,ledger=ledger_diagnostics,
        phase_blends={name:performance(s[oos]) for name,s in blend.items()},
        paired_weightband_minus_baseline_sharpe=paired,factor_audit=audits,
        caveats=['cached F17 was selected after historical research',
                 'survivorship, current industry and monthly ST inherited',
                 'adjusted price units, fractional units, no auction depth or exact corporate action ledger',
                 'historical board-limit approximation, special cases need official daily limits',
                 'missing held valuations are carried only when explicitly allowed; no assumed delisting recovery',
                 'daily volume availability proxy is not an opening-time tradability snapshot',
                 'fixed 2% risk free for statistics; no interest accrual on idle cash',
                 'no claim of selection-adjusted significance from the fixed two-candidate experiment'])
    (out/'review_results.json').write_text(json.dumps(json_clean(payload),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (out/'input_manifest.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf-8')
    flat=[{k:v for k,v in s.items() if k!='hac_excess'} for s in summary]
    pd.DataFrame(flat).to_csv(out/'performance.csv',index=False,encoding='utf-8-sig')
    print(f'Done in {time.monotonic()-start:.0f}s. Outputs: {out}',flush=True)


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    main()
