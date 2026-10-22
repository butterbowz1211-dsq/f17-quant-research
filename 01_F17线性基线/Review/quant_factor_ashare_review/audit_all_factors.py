# Research project author: Gilmour Jia
"""Re-test the original 24-factor family (23 unique definitions) on next-open labels.

Family FDR is descriptive: excludes later newly searched factors and strategies.
Uses cached preprocessing; does not repair point-in-time industry or survivorship.
"""
from pathlib import Path
import argparse
import json
import gc
import numpy as np
import pandas as pd
from robust_factor_core import rank_ic,forward_open_return,purged_training_mask,hac_mean,fdr

HERE=Path(__file__).resolve().parent
NAMES=['REV_1M','REV_1W','MOM_12_1','REV_LT','MAX5','IVOL','BETA','ILLIQ','SIZE',
       'HIGH52','CO_SKEW','TURN','PV_CORR','ID_FROG','ISKEW','IND_MOM','RES_MOM',
       'OVERNIGHT','INTRADAY','LEAD_LAG','COMMOVE','DIVERGENCE','TURN_SHOCK']


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(r'workspace/source_workspace\quant_factor_ashare'))
    args=ap.parse_args();data=args.source/'data';out=HERE/'results';out.mkdir(exist_ok=True)
    panel=pd.read_pickle(data/'panel.pkl')
    dates=pd.DatetimeIndex(sorted(panel.date.unique()))
    symbols=pd.Index(sorted(panel.symbol.astype(str).unique()),name='symbol')
    row=dates.get_indexer(panel.date)
    remap=symbols.get_indexer(panel.symbol.cat.categories)
    col=remap[panel.symbol.cat.codes.to_numpy()]
    a=np.full((len(dates),len(symbols)),np.nan,dtype='float32')
    a[row,col]=panel.open.to_numpy('float32')
    label=forward_open_return(pd.DataFrame(a,index=dates,columns=symbols),21)
    del panel,row,col,a;gc.collect()
    masks={'training_purged':purged_training_mask(dates,'2022-12-31',21),
           'reused_validation':dates>='2023-01-01'}
    rows=[]
    for name in NAMES:
        fac=pd.read_pickle(data/'prep'/f'{name}.pkl')
        ic=rank_ic(fac,label)
        for split,mask in masks.items():
            s=ic.loc[mask]
            stat=hac_mean(s,21,21);slow=hac_mean(s,63,21)
            rows.append(dict(factor=name,split=split,IC=stat['mean'],t_hac21=stat['t'],
                             p_hac21=stat['p_two_sided'],t_hac63=slow['t'],n=stat['n']))
        print(name,rows[-1],flush=True)
        del fac,ic;gc.collect()
    table=pd.DataFrame(rows)
    for _,ix in table.groupby('split').groups.items():
        table.loc[ix,'q_BH_23']=fdr(table.loc[ix,'p_hac21'],False)
        table.loc[ix,'q_BY_23']=fdr(table.loc[ix,'p_hac21'],True)
    table.to_csv(out/'factor_family23.csv',index=False,encoding='utf-8-sig')
    (out/'factor_family23.json').write_text(json.dumps(table.to_dict('records'),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__': main()
