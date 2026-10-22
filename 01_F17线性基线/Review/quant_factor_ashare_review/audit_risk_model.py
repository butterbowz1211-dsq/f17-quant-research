# Research project author: Gilmour Jia
"""Audit calendar alignment/risk-free subtraction without modifying source CH data."""
import argparse
import json
from pathlib import Path
import pandas as pd
from robust_factor_core import monthly_alpha,ols_hac

HERE=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(r'workspace/source_workspace\quant_factor_ashare'))
    ap.add_argument('--results',type=Path,default=HERE/'results')
    args=ap.parse_args()
    results=args.results.resolve()
    if HERE not in results.parents: raise ValueError('results must be in review folder')
    ch=pd.read_pickle(args.source/'data'/'ch3.pkl')[['MKT','SMB','VMG','PMO']]
    # Sign correction makes SMB interpretable; flipping a regressor sign alone
    # does NOT change intercept, fitted values, or R-squared.
    ch['SMB']=-ch['SMB']
    out={'note':'CH4-like proxy only; source float-cap, financial vintages, forward-return missingness and SMB4 simplification remain',
         'factor_months':len(ch),'calendar_month_end_fraction':float(ch.index.is_month_end.mean()),'comparisons':{}}
    for name in ['legacy_phase0','monthly_buffer_p00','monthly_buffer_weightband_p00']:
        r=pd.read_csv(results/f'{name}.csv',index_col=0,parse_dates=True)['net']
        for split,start in [('full','2010-01-01'),('reused_validation','2023-01-01')]:
            seg=r.loc[start:]
            old_monthly=seg.resample('ME').apply(lambda x:(1+x).prod()-1)
            old=ols_hac(old_monthly,ch.shift(1),6)
            corrected=monthly_alpha(seg,ch)
            out['comparisons'][f'{name}/{split}']={'original_calendar_join_raw_y':old,
                                                  'period_join_excess_y':corrected}
    (results/'risk_model_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
