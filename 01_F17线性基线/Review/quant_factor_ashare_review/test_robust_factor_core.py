# Research project author: Gilmour Jia
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
import robust_factor_core as core


def old_functions():
    """Extract only audited pure functions; never execute source module top level."""
    p=Path(__file__).parent/'source_snapshot'/'ashare_factor_strategy.py'
    tree=ast.parse(p.read_text(encoding='utf-8-sig'))
    names={'backtest','summarize','rank_ic','newey_west_tstat'}
    module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    ns={'np':np,'pd':pd,'F32':'float32','TRADING_DAYS':252}
    exec(compile(module,str(p),'exec'),ns)
    return ns


class CoreTests(unittest.TestCase):
    def setup_market(self):
        self.dates=pd.bdate_range('2024-01-01',periods=6)
        op=pd.DataFrame([[10,10],[10,10],[12,10],[12,10],[12,10],[12,10]],index=self.dates,columns=['a','b'],dtype=float)
        cl=op.copy(); cl.iloc[1,0]=12
        score=pd.DataFrame([[2,1]]*6,index=self.dates,columns=op.columns)
        mask=score.notna(); adv=score*0+1e12
        return op,cl,score,mask,adv

    def run_market(self,op,cl,s,m,adv,**kwargs):
        cfg=core.ExecutionConfig(n=2,cap=1,rebalance=4,commission_bp=0,slippage_bp=0,capital=1000,participation=1)
        return core.event_backtest(op,cl,s,m,kwargs.get('buy',m),kwargs.get('sell',m),adv,cfg,audit_holdings=True)

    def test_monthly_means_no_daily_rebalance(self):
        op,cl,s,m,adv=self.setup_market()
        new=self.run_market(op,cl,s,m,adv)
        self.assertEqual(new['daily'].iloc[2]['turnover'],0)
        cfg=SimpleNamespace(commission_bp=0,slippage_bp=0,stamp_duty_bp=0,rf_annual=0)
        old=old_functions()['backtest'](SimpleNamespace(dates=op.index,symbols=op.columns,open=op,close=cl),
            pd.DataFrame([[.5,.5]],index=op.index[:1],columns=op.columns),cfg)
        self.assertGreater(old['turnover'].iloc[2],.09)

    def test_missing_open_freezes_sale_and_no_phantom_return(self):
        op,cl,s,m,adv=self.setup_market()
        op.iloc[5,0]=np.nan; cl.iloc[5,0]=np.nan
        s.iloc[4,0]=np.nan
        sell=m.copy(); sell.iloc[5,0]=False
        new=self.run_market(op,cl,s,m,adv,sell=sell)
        self.assertGreater(new['holdings'].iloc[5,0],0)
        self.assertEqual(new['daily'].iloc[5]['net'],0)

    def test_limit_up_cannot_buy(self):
        op,cl,s,m,adv=self.setup_market(); buy=m.copy(); buy.iloc[1]=False
        new=self.run_market(op,cl,s,m,adv,buy=buy)
        self.assertEqual(new['daily'].iloc[1]['cash_weight'],1)

    def test_cap_infeasible_leaves_cash(self):
        op,cl,s,m,adv=self.setup_market()
        cfg=core.ExecutionConfig(n=2,cap=.03,capital=1000,participation=1)
        r=core.event_backtest(op,op,s,m,m,m,adv,cfg,audit_holdings=True)
        self.assertGreater(r['daily'].iloc[1]['cash_weight'],.93)
        self.assertLessEqual(r['holdings'].iloc[1].max(),.03)

    def test_first_day_loss_in_drawdown(self):
        self.assertAlmostEqual(core.performance([-.1,0])['MaxDD'],-.1)

    def test_pairwise_rank_ic(self):
        f=pd.DataFrame([[1,2,3,np.nan]])
        y=pd.DataFrame([[3,2,1,100]])
        self.assertAlmostEqual(core.rank_ic(f,y,3).iloc[0],-1)
        self.assertNotAlmostEqual(old_functions()['rank_ic'](f,y,3).iloc[0],-1)

    def test_hac_matches_explicit_bartlett_matrix(self):
        x=np.random.default_rng(7).normal(size=50)+.1; L=7
        e=x-x.mean(); i=np.arange(len(x))
        kernel=np.maximum(0,1-np.abs(i[:,None]-i[None,:])/(L+1))
        se=np.sqrt(e@kernel@e/len(x)**2)
        self.assertAlmostEqual(core.hac_mean(x,L)['se'],se)
        self.assertGreaterEqual(core.hac_mean(x,5,horizon=22)['lags'],21)

    def test_purge(self):
        d=pd.bdate_range('2022-12-01',periods=70)
        m=core.purged_training_mask(d,'2022-12-31',5)
        for i in np.flatnonzero(m):
            self.assertLessEqual(d[i+6],pd.Timestamp('2022-12-31'))

    def test_forward_open_label(self):
        o=pd.DataFrame({'a':[1.,2.,3.,4.,5.]})
        self.assertEqual(core.forward_open_return(o,2).iloc[0,0],1.)

    def test_nonzero_p(self):
        self.assertAlmostEqual(core.monte_carlo_p(2,np.zeros(200)),1/201)

    def test_past_returns_do_not_change_with_future_scores(self):
        op,cl,s,m,adv=self.setup_market(); a=self.run_market(op,cl,s,m,adv)
        s.iloc[4:]*=-100; b=self.run_market(op,cl,s,m,adv)
        np.testing.assert_allclose(a['daily']['net'].iloc[:5],b['daily']['net'].iloc[:5])

    def test_tax_schedule(self):
        self.assertEqual(core.stamp_rate('2023-08-25'),.001)
        self.assertEqual(core.stamp_rate('2023-08-28'),.0005)

    def test_fdr(self):
        np.testing.assert_allclose(core.fdr([.01,.03,.2],False),[.03,.045,.2])

    def test_bankruptcy_or_invalid_series_rejected(self):
        with self.assertRaises(ValueError): core.performance([0,np.nan])
        with self.assertRaises(ValueError): core.performance([0,-1])

    def test_stale_holding_rejected(self):
        op,cl,s,m,adv=self.setup_market(); op.iloc[2:,0]=np.nan;cl.iloc[2:,0]=np.nan
        cfg=core.ExecutionConfig(n=2,cap=1,stale_limit=1)
        with self.assertRaises(ValueError): core.event_backtest(op,cl,s,m,m,m,adv,cfg)

    def test_month_period_alignment_and_excess_alpha(self):
        dates=pd.bdate_range('2018-01-01','2024-12-31')
        returns=pd.Series(.0003,index=dates)
        idx=pd.Series(dates,index=dates).resample('ME').last().to_numpy()
        fac=pd.DataFrame({'MKT':np.random.default_rng(1).normal(0,.02,len(idx))},index=pd.DatetimeIndex(idx))
        result=core.monthly_alpha(returns,fac)
        self.assertEqual(result['n'],len(idx)-1)

    def test_changing_smb_sign_does_not_change_alpha(self):
        rng=np.random.default_rng(1);x=pd.DataFrame({'SMB':rng.normal(size=70)})
        y=.02+.3*x.SMB+rng.normal(0,.01,len(x))
        a=core.ols_hac(y,x);b=core.ols_hac(y,-x)
        self.assertAlmostEqual(a['alpha_monthly'],b['alpha_monthly'])
        self.assertAlmostEqual(a['betas']['SMB'],-b['betas']['SMB'])

    def test_family_max_test_p_is_finite_nonzero(self):
        d=np.zeros((50,2))
        self.assertEqual(core.family_max_mean_test(d,99)['p'],1)

    def test_dsr_rejects_mismatched_dates(self):
        y=pd.Series(np.arange(50)*.0001)
        x=pd.DataFrame({'a':y,'b':y+.001},index=np.arange(1,51))
        with self.assertRaises(ValueError): core.dsr_iid(y,x)


if __name__=='__main__': unittest.main(verbosity=2)
