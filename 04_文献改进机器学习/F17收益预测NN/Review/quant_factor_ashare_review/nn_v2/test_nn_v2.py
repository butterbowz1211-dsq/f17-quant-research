# Research project author: Gilmour Jia
import runtime
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data import (DateSample, forward_label, allowed_sample, preprocess, validate_keys,
                  validate_market, canonical_union, cache_key, save_cache, load_cache,
                  decision_indices, split_samples)
from execution import Book, FeeSchedule, quantity
from factors import build_factors, top5
from metrics import metrics, newey_west, rank_ic, gates, paired_interval
from model import ReturnNN, fit, predict, save_bundle, load_bundle, date_huber, label_scale, ridge_fit, ridge_predict
from portfolio import cap_allocate, readout
from spec import SPEC, FEATURES
from spec import write_json, file_hash
from train import acceptance, Ledger, load_samples


def sample(date="2018-06-01", end="2018-07-03", n=40, seed=1):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,34)).astype(np.float32)
    y=(x[:,0]*.03+x[:,1]*.01).astype(np.float32)
    return DateSample(np.datetime64(date,"ns"),np.datetime64(date,"ns")+np.timedelta64(1,"D"),
                      np.datetime64(end,"ns"),np.array([f"sh{600000+i}" for i in range(n)]),
                      x,y,np.full(n,"ok",dtype="U40"),x[:,:17].mean(axis=1))


def bars(px=10., can_buy=True, can_sell=True, adv=1e9, close=None, symbol="sh600000"):
    return pd.DataFrame({"raw_open":[px],"raw_close":[px if close is None else close],
                         "can_buy":[can_buy],"can_sell":[can_sell],"adv_lag":[adv]},index=[symbol])


class LabelTests(unittest.TestCase):
    def setUp(self):
        self.dates=pd.bdate_range("2020-01-01",periods=40)
        self.p=(100+np.arange(40)**2).astype(float)[:,None]
        self.status=np.full(self.p.shape,"trading",dtype="U20")
        self.verified=np.ones(self.p.shape,bool)

    def test_exact_next_open_non_geometric(self):
        y,r,e,z=forward_label(self.p,self.dates,3,self.status,self.verified)
        self.assertAlmostEqual(y[0],self.p[25,0]/self.p[4,0]-1)
        self.assertNotAlmostEqual(y[0],self.p[24,0]/self.p[3,0]-1)
        self.assertEqual(e,self.dates[4].to_datetime64())
        self.assertEqual(z,self.dates[25].to_datetime64())

    def test_gap_is_not_zero(self):
        self.p[2,0]=np.nan
        self.p[1,0]=100
        self.p[22,0]=80
        y,r,*_=forward_label(self.p,self.dates,0,self.status,self.verified)
        self.assertTrue(np.isnan(y[0]))
        self.assertEqual(r[0],"unexplained_price_gap")
        self.status[2,0]="suspended"
        y,r,*_=forward_label(self.p,self.dates,0,self.status,self.verified)
        self.assertAlmostEqual(y[0],-.2)

    def test_missing_endpoint_delisting_and_adjustment(self):
        for which,expected in (("endpoint","missing_endpoint"),("delist","delisting_unresolved"),
                               ("adjustment","unverified_adjustment")):
            p,s,v=self.p.copy(),self.status.copy(),self.verified.copy()
            if which=="endpoint": p[22]=np.nan
            if which=="delist": s[10]="delisted"
            if which=="adjustment": v[12]=False
            y,r,*_=forward_label(p,self.dates,0,s,v)
            self.assertTrue(np.isnan(y[0])); self.assertEqual(r[0],expected)

    def test_unmatured_and_purge(self):
        y,r,*_=forward_label(self.p,self.dates,30,self.status,self.verified)
        self.assertTrue(np.isnan(y[0])); self.assertEqual(r[0],"not_matured")
        s=sample("2018-12-20","2019-01-23")
        self.assertFalse(allowed_sample(s,"2013-01-01","2018-12-31","2018-12-31"))
        self.assertFalse(allowed_sample(s,"2013-01-01","2018-12-31","2019-01-23",strict=True))
        self.assertTrue(allowed_sample(s,"2013-01-01","2018-12-31","2019-01-23"))

    def test_global_grid_not_year_reset(self):
        d=pd.bdate_range("2010-01-04","2026-09-28")
        ix=decision_indices(d)
        np.testing.assert_equal(np.diff(ix),21)
        self.assertTrue(all(i%21==0 for i in ix[d[ix].year==2021]))
        with self.assertRaises(ValueError): decision_indices(d[d.year==2021])

    def test_expanding_split_realization(self):
        ss=[sample("2018-12-01","2018-12-31"),sample("2018-12-20","2019-01-22"),
            sample("2019-06-01","2019-07-01"),sample("2020-12-10","2021-01-12"),
            sample("2020-12-29","2021-01-29"),sample("2021-01-20","2021-02-23")]
        train,valid,final,outer=split_samples(ss,2021)
        self.assertEqual(len(train),1); self.assertEqual(len(valid),2)
        self.assertEqual(len(final),4); self.assertEqual(len(outer),1)


class DataTests(unittest.TestCase):
    def test_prepare_verified_synthetic_package_and_keep_gate_closed(self):
        from prepare import prepare
        rng=np.random.default_rng(50)
        dates=pd.bdate_range("2010-01-04",periods=850)
        symbols=[f"sh{600000+i}" for i in range(50)]
        px=10*np.exp(np.cumsum(rng.normal(0,.01,(850,50)),axis=0))
        idx=pd.MultiIndex.from_product([dates,symbols],names=["date","symbol"])
        f=pd.DataFrame(index=idx).reset_index()
        p=px.ravel()
        for prefix in ("raw_","adjusted_"):
            for col,mult in (("open",.999),("close",1),("high",1.02),("low",.98)):
                f[prefix+col]=p*mult
        f["raw_volume"]=1e5;f["raw_amount"]=p*1e5
        f["exchange"]="SH";f["security_type"]="A";f["float_shares"]=1e8
        f["turnover"]=.01+rng.random(len(f))*.01;f["listing_date"]="2008-01-01"
        f["is_st"]=False;f["industry"]=np.tile(np.repeat(["a","b"],25),850)
        f["status"]="trading";f["adjustment_verified"]=True
        f["limit_up_price"]=p*1.10;f["limit_down_price"]=p*.90;f["no_limit"]=False
        with tempfile.TemporaryDirectory(dir=runtime.ROOT) as td:
            source=Path(td)/"source";source.mkdir()
            f.to_parquet(source/"market.parquet",index=False)
            pd.DataFrame({"date":dates}).to_csv(source/"calendar.csv",index=False)
            write_json(source/"actions.json",[])
            write_json(source/"source_acceptance.json",{"status":"PASS","synthetic_only":True,
                       "files":{n:file_hash(source/n) for n in ("market.parquet","calendar.csv","actions.json")}})
            output=Path(td)/"prepared"
            prepare(source,output)
            ss=load_samples(output/"samples.npz")
            self.assertGreater(len(ss),1)
            self.assertTrue(all(s.x.shape[1]==34 and np.isfinite(s.x).all() for s in ss))
            market=pd.read_parquet(output/"market.parquet")
            self.assertTrue({"date","symbol","can_buy","can_sell","adv_lag"}<=set(market))
            with self.assertRaises(ValueError):acceptance(output)

    def test_duplicate_keys(self):
        d=pd.DataFrame({"date":["2020-01-01"]*2,"symbol":["sh600000"]*2})
        with self.assertRaises(ValueError): validate_keys(d)

    def test_unit_check(self):
        row={"date":"2020-01-01","symbol":"sh600000","raw_volume":100,"raw_amount":1000}
        for p in ("raw_","adjusted_"):
            row.update({p+"open":10,p+"high":11,p+"low":9,p+"close":10})
        validate_market(pd.DataFrame([row]))
        row["raw_amount"]*=1000
        with self.assertRaises(ValueError): validate_market(pd.DataFrame([row]))

    def test_union_preserves_raw_only_and_price_source(self):
        s=pd.DataFrame({"symbol":["sh600000"],"date":["2020-01-01"],
                        "open":[5.],"high":[5.5],"low":[4.5],"close":[5.],
                        "volume":[100.],"amount":[1000.]})
        raw=pd.DataFrame({"ts_code":["600000.SH","600001.SH"],"trade_date":["20200101"]*2,
                          "open":[10.,20.],"high":[11.,21.],"low":[9.,19.],"close":[10.,20.],
                          "vol":[1.,2.],"amount":[1.,4.]})
        z=canonical_union(s,raw)
        self.assertEqual(len(z),2)
        self.assertEqual(z.raw_volume.tolist(),[100.,200.])
        self.assertEqual(z.raw_amount.tolist(),[1000.,4000.])
        self.assertTrue(np.isnan(z.adjusted_close.iloc[1]))
        self.assertEqual(z.raw_close_source.iloc[1],"tushare_daily")

    def test_cache_rejects_changed_config(self):
        with tempfile.TemporaryDirectory(dir=runtime.ROOT) as td:
            p=Path(td)/"cache.npz"
            key=cache_key("a","b",{"horizon":21})
            save_cache(p,key,x=np.ones(3))
            np.testing.assert_equal(load_cache(p,key)["x"],1)
            with self.assertRaises(ValueError): load_cache(p,cache_key("a","b",{"horizon":20}))

    def test_median_flags_and_f17_skip_missing(self):
        rng=np.random.default_rng(2)
        raw=rng.normal(size=(60,17))
        raw[0,2]=np.nan
        clean,x,score=preprocess(raw,np.ones(60,bool),rng.normal(size=60),np.repeat(["a","b"],30))
        self.assertEqual(x.shape,(60,34)); self.assertTrue(np.isfinite(x).all())
        self.assertEqual(x[0,19],1)
        self.assertAlmostEqual(x[0,2],np.nanmedian(clean[:,2]),places=6)
        self.assertAlmostEqual(score[0],np.nanmean(clean[0]))
        raw[:,0]=np.nan
        with self.assertRaises(ValueError): preprocess(raw,np.ones(60,bool),np.zeros(60),np.repeat("a",60))

    def test_future_prices_do_not_change_past_features(self):
        rng=np.random.default_rng(7)
        d=pd.bdate_range("2010-01-04",periods=820)
        cols=[f"sh{600000+i}" for i in range(6)]
        p=pd.DataFrame(10*np.exp(np.cumsum(rng.normal(0,.01,size=(820,6)),axis=0)),index=d,columns=cols)
        m={"adjusted_close":p,"adjusted_open":p*.998,"adjusted_high":p*1.01,
           "amount":p*100000,"turnover":p*.001,"float_cap":p*1e8,
           "industry":pd.Series(["a"]*3+["b"]*3,index=cols),"traded":p>0}
        a=build_factors(m)
        changed={k:v.copy() for k,v in m.items()}
        for k in ("adjusted_close","adjusted_open","adjusted_high","amount","turnover","float_cap"):
            changed[k].iloc[800:]*=2
        b=build_factors(changed)
        for k in a:
            np.testing.assert_allclose(a[k].iloc[:800],b[k].iloc[:800],equal_nan=True)
        r=pd.DataFrame({"x":np.arange(21.)})
        self.assertEqual(top5(r).iloc[-1,0],18.)


class PortfolioTests(unittest.TestCase):
    def test_caps_and_infeasible_fully_invested(self):
        w=cap_allocate([1,100,1000],.03)
        np.testing.assert_allclose(w,.03)
        self.assertAlmostEqual(w.sum(),.09)
        w=cap_allocate(np.exp(np.arange(50)/50),.03)
        self.assertLessEqual(w.max(),.03+1e-12); self.assertAlmostEqual(w.sum(),1)

    def test_readout_ties_units_and_buffer(self):
        s=pd.Series(np.arange(100.),index=[f"s{i:03}" for i in range(100)])
        a,p=readout(s,["s025"],"softmax")
        b,_=readout(s*100+5,["s025"],"softmax")
        self.assertIn("s025",p); np.testing.assert_allclose(a,b)
        w,p=readout(s*0,(),"softmax")
        self.assertEqual(p[0],"s000"); np.testing.assert_allclose(w,.02)
        self.assertEqual(readout(s.iloc[:10],["old"]),(None,["old"]))


class ExecutionTests(unittest.TestCase):
    def test_both_sides_individual_min_commission_and_zero(self):
        fees=FeeSchedule()
        for side in ("buy","sell"):
            a=fees.charge("2021-01-04","sh600000",side,1000)
            b=fees.charge("2021-01-04","sz000001",side,1000000)
            self.assertEqual(a["commission"],5)
            self.assertAlmostEqual(a["commission"]+b["commission"],273.7)
            self.assertAlmostEqual(a["slippage"],1)
            self.assertEqual(fees.charge("2021-01-04","sh600000",side,0)["total"],0)

    def test_fee_effective_days(self):
        f=FeeSchedule()
        a=f.charge("2022-04-28","sh600000","sell",100000)
        b=f.charge("2022-04-29","sh600000","sell",100000)
        self.assertEqual(a["transfer"],2);self.assertEqual(b["transfer"],1)
        a=f.charge("2023-08-27","sz000001","sell",100000)
        b=f.charge("2023-08-28","sz000001","sell",100000)
        self.assertEqual(a["stamp"],100); self.assertEqual(b["stamp"],50)
        self.assertAlmostEqual(a["handling"],4.87); self.assertAlmostEqual(b["handling"],3.41)
        with self.assertRaises(ValueError): f.charge("2014-01-01","sh600000","sell",100)
        with self.assertRaises(ValueError): f.charge("2023-01-01","bj830001","buy",100)

    def test_forced_exit_cannot_bypass_suspension_or_limit(self):
        b=Book(10000);b.shares["sh600000"]=155;b.marks["sh600000"]=10
        b.last_nav=b.nav()
        for date in ("2023-01-03","2023-01-04"):
            b.step(date,bars(can_sell=False),pd.Series(dtype=float))
            self.assertEqual(b.shares["sh600000"],155)
            self.assertEqual(len(b.orders),0)
        b.step("2023-01-05",bars(),pd.Series(dtype=float))
        self.assertEqual(b.shares["sh600000"],0)
        self.assertEqual(b.orders[-1]["shares"],155)

    def test_capacity_and_odd_lot(self):
        b=Book(10000);b.shares["sh600000"]=255;b.marks["sh600000"]=10;b.last_nav=b.nav()
        b.step("2023-01-03",bars(adv=20000),pd.Series(dtype=float))
        self.assertEqual(b.orders[0]["shares"],100)
        self.assertEqual(b.shares["sh600000"],155)
        self.assertLessEqual(b.orders[0]["notional"],1000)

    def test_raw_board_lot_and_cash(self):
        b=Book(10000)
        b.step("2023-01-03",bars(px=1),{"sh600000":.03})
        self.assertEqual(b.shares["sh600000"],300)
        self.assertAlmostEqual(b.cash,10000-300-5-.003-.3)
        self.assertGreaterEqual(b.cash,0)
        self.assertEqual(quantity("sh688001",199,"buy"),0)
        self.assertEqual(quantity("sh688001",201,"buy"),201)
        self.assertEqual(quantity("sh600000",299,"buy"),200)

    def test_no_trade_no_charge_and_cross_year_continuity(self):
        b=Book(10000000)
        b.step("2023-12-29",bars(px=10),{"sh600000":.03})
        q,c=b.shares.copy(),b.cash
        b.step("2024-01-02",bars(px=11),None)
        self.assertEqual(q,b.shares);self.assertEqual(c,b.cash)
        self.assertEqual(b.daily[-1]["total"],0)
        self.assertGreater(b.daily[-1]["net"],0)
        with self.assertRaises(ValueError): b.step("2024-01-02",bars(px=11))

    def test_suspension_reopening_loss_and_stale_rejection(self):
        b=Book(10000);b.shares["sh600000"]=100;b.marks["sh600000"]=100;b.last_nav=b.nav()
        b.step("2023-01-03",bars(px=np.nan,close=np.nan,can_buy=False,can_sell=False))
        b.step("2023-01-04",bars(px=80))
        self.assertAlmostEqual(b.daily[-1]["net"],-.1)
        b.stale["sh600000"]=60
        with self.assertRaises(ValueError): b.step("2023-01-05",bars(px=np.nan,close=np.nan))

    def test_corporate_cash_stock_and_locks(self):
        b=Book(0);b.shares["sh600000"]=100;b.marks["sh600000"]=20;b.last_nav=2000
        a={"id":"x","symbol":"sh600000","kind":"cash_stock_distribution","verified":True,
           "record_date":"2023-01-03","ex_date":"2023-01-04","pay_date":"2023-01-05",
           "stock_date":"2023-01-06","share_ratio":2.,"net_cash_per_share":2.,"gross_cash_per_share":2.}
        b.step("2023-01-03",bars(px=20),actions=[a])
        b.step("2023-01-04",bars(px=9),actions=[a])
        self.assertEqual(b.shares["sh600000"],200);self.assertEqual(b.locked["sh600000"],100)
        self.assertEqual(b.receivable,200);self.assertEqual(b.nav(),2000)
        b.step("2023-01-05",bars(px=9),actions=[a])
        self.assertEqual(b.cash,200);self.assertEqual(b.receivable,0)
        b.step("2023-01-06",bars(px=9),actions=[a])
        self.assertEqual(b.locked["sh600000"],0)

    def test_basket_affordability(self):
        names=[f"sh{600000+i}" for i in range(50)]
        f=pd.concat([bars(px=10,symbol=s) for s in names])
        b=Book(100000)
        b.step("2023-01-03",f,pd.Series(.02,index=names))
        self.assertGreaterEqual(b.cash,0)
        self.assertTrue(all(q%100==0 for q in b.shares.values()))
        self.assertAlmostEqual(b.nav(),100000-sum(o["total"] for o in b.orders))


class ModelTests(unittest.TestCase):
    def test_structure_and_unconstrained_output(self):
        m=ReturnNN()
        self.assertEqual(sum(isinstance(x,torch.nn.Linear) for x in m.modules()),3)
        self.assertFalse(any(isinstance(x,(torch.nn.Softmax,torch.nn.BatchNorm1d)) for x in m.modules()))
        with torch.no_grad():
            for p in m.parameters():p.zero_()
            m.layers[-1].bias.fill_(-2)
        self.assertTrue((predict(m,np.zeros((4,34)),.1) < 0).all())

    def test_mask_and_date_weight_invariance(self):
        s=sample(n=30);s.y[0]=np.nan;s.label_reason[0]="missing_endpoint"
        m=ReturnNN().eval()
        a=date_huber(m,s,1).item()
        s.x[0]=np.nan
        b=date_huber(m,s,1).item()
        self.assertAlmostEqual(a,b)
        s=sample(n=30)
        doubled=DateSample(s.formation_time,s.entry_time,s.label_end_time,np.tile(s.symbols,2),
                           np.tile(s.x,(2,1)),np.tile(s.y,2),np.tile(s.label_reason,2),np.tile(s.f17_score,2))
        self.assertAlmostEqual(date_huber(m,s,1).item(),date_huber(m,doubled,1).item(),places=7)
        self.assertAlmostEqual(label_scale([s]),label_scale([doubled]))

    def test_tiny_fit_gradient_bundle_and_ridge(self):
        train=[sample(seed=i) for i in range(4)]
        valid=[sample("2019-06-01","2019-07-01",seed=40)]
        m,sc,epoch,val,hist=fit(train,valid,.001,.0001,0,smoke=True)
        self.assertTrue(np.isfinite(val));self.assertGreater(epoch,0)
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in m.parameters()))
        with tempfile.TemporaryDirectory(dir=runtime.ROOT) as td:
            p=Path(td)/"tiny.pt"
            save_bundle(p,m,sc,0,epoch,{"data_version":"synthetic_engineering_only",
                        "code_version":"test","train_range":["2018-06-01","2018-06-01"],
                        "spec_hash":"test","hyperparameters":{"lr":.001,"weight_decay":.0001}})
            reload,b=load_bundle(p)
            np.testing.assert_array_equal(predict(m,valid[0].x,sc),
                                          predict(reload,valid[0].x,b["scalers"]["label_scale"]))
        beta,scale=ridge_fit(train,.0001)
        self.assertLess(np.mean((ridge_predict(beta,valid[0].x,scale)-valid[0].y)**2),1e-6)

    def test_training_rejects_no_acceptance_and_budget(self):
        with tempfile.TemporaryDirectory(dir=runtime.ROOT) as td:
            p=Path(td)
            with self.assertRaises(ValueError): acceptance(p)
            ledger=Ledger(p/"ledger.jsonl",{"spec_hash":"x"})
            ledger.begin("one","inner","NN",seed=0)
            with self.assertRaises(ValueError): ledger.begin("one","inner","NN",seed=1)
            for i in range(1,51):ledger.begin(str(i),"inner","NN",seed=0)
            with self.assertRaises(ValueError):ledger.begin("overflow","inner","NN",seed=0)

    def test_full_protocol_task_count_and_resume_without_refitting(self):
        """Mock optimization only: exercise actual scheduling, bundles and resume."""
        import train as runner
        samples=[]
        for year in range(2013,2027):
            for month in (2,6,10):
                samples.append(sample(f"{year}-{month:02}-01",f"{year}-{month+1:02}-10",seed=year+month))
        calls=[]
        def mock_fit(tr,va,lr,wd,seed,epochs=None,device="cpu",progress=None):
            calls.append((lr,wd,seed,epochs))
            m=ReturnNN().eval()
            return m,.1,(epochs or 2),(lr+wd if va is not None else None),[]
        with tempfile.TemporaryDirectory(dir=runtime.ROOT) as td:
            out=Path(td)/"test_only_run"
            with patch.object(runner,"acceptance",return_value={"synthetic_test":True}), \
                 patch.object(runner,"load_samples",return_value=samples), \
                 patch.object(runner,"fit",side_effect=mock_fit):
                runner.run(Path(td),out,"cpu")
                self.assertEqual(len(calls),51)
                ledger=[json.loads(s) for s in (out/"experiment_ledger.jsonl").read_text().splitlines()]
                self.assertEqual(sum(r["model"]=="Ridge" and r["state"]=="complete" for r in ledger),9)
                self.assertEqual(len({(lr,wd) for lr,wd,_,_ in calls[:18]}),6)
                self.assertEqual(len({(lr,wd) for lr,wd,_,_ in calls[18:]}),1)
                self.assertTrue(all(epochs==2 for _,_,_,epochs in calls if epochs is not None))
                runner.run(Path(td),out,"cpu")
                self.assertEqual(len(calls),51)  # Completed task reuse consumes no fits.


class StatisticsTests(unittest.TestCase):
    def test_hac_against_hand_calculation(self):
        beta,cov=newey_west([2,1,4,3],[-1,0,1,2],lags=1)
        np.testing.assert_allclose(beta,[2.2,.6],atol=1e-12)
        # Bread=[[.3,-.1],[-.1,.2]], meat=[[.8,.4],[.4,1.28]].
        np.testing.assert_allclose(cov,[[.0608,-.0216],[-.0216,.0432]],atol=1e-12)

    def test_sharpe_and_initial_drawdown(self):
        r=np.array([-.1,.02,.03])
        m=metrics(r)
        self.assertAlmostEqual(m["Sharpe"],r.mean()/r.std(ddof=1)*np.sqrt(252))
        self.assertAlmostEqual(m["max_drawdown"],-.1)
        self.assertNotAlmostEqual(m["Sharpe"],m["CAGR"]/m["volatility"])
        self.assertEqual(rank_ic([1,2,3,np.nan],[3,2,1,0]),-1)

    def test_gate_and_paired_series(self):
        r=np.array([.01,-.005,.003,.004,-.002]*8)
        m=metrics(r)
        self.assertFalse(gates(m,m,True)["passed_historical_gate"])
        ci=paired_interval(r,r,reps=20)
        np.testing.assert_allclose(list(ci.values()),np.zeros((3,2)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
