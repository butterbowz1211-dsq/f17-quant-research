# Research project author: Gilmour Jia
import unittest
import numpy as np
import pandas as pd
from linkage_research import analyst_edges, snapshot_edges, network_signal, segment_edges, fit_group_weights, fit_asof


class LinkageTests(unittest.TestCase):
    def test_analyst_dedup_and_future_exclusion(self):
        f=pd.DataFrame([
            ['x','A','2020-01-01','2020-01-01','FY1'],
            ['x','A','2020-01-02','2020-01-02','FY2'],
            ['x','B','2020-01-01','2020-01-01','FY1'],
            ['y','A','2020-01-01','2020-01-01','FY1'],
            ['y','B','2020-01-01','2020-03-01','FY1'],
            ['old','A','2018-01-01','2018-01-01','FY1'],
            ['old','B','2018-01-01','2018-01-01','FY1']],
            columns=['analyst_id','symbol','issued_at','available_at','horizon'])
        e=analyst_edges(f,'2020-02-01')
        self.assertEqual(len(e),2); self.assertTrue((e.weight==1).all())
        self.assertTrue((analyst_edges(f,'2020-04-01').weight==2).all())

    def test_atomic_snapshot_removes_old_edges(self):
        f=pd.DataFrame([['A','B',1,'2020-01-01','s1'],['A','C',1,'2020-02-01','s2'],
                        ['A',None,0,'2020-03-01','s3']],
                        columns=['target','peer','weight','available_at','snapshot_id'])
        self.assertEqual(snapshot_edges(f,'2020-01-15').peer.tolist(),['B'])
        self.assertEqual(snapshot_edges(f,'2020-02-15').peer.tolist(),['C'])
        self.assertTrue(snapshot_edges(f,'2020-03-15').empty)

    def test_missing_peer_not_zero_return(self):
        e=pd.DataFrame({'target':['A','A'],'peer':['B','C'],'weight':[1.,3.]})
        r=pd.Series({'B':.1})
        self.assertTrue(np.isnan(network_signal(e,r).loc['A','signal']))
        self.assertEqual(network_signal(e,r).loc['A','coverage'],.25)
        self.assertAlmostEqual(network_signal(e,pd.Series({'B':.1,'C':.2})).loc['A','signal'],.175)

    def test_self_link_rejected(self):
        with self.assertRaises(ValueError):
            network_signal(pd.DataFrame({'target':['A'],'peer':['A'],'weight':[1]}),pd.Series({'A':1}))

    def test_segment_sales_and_cap_weights(self):
        s=pd.DataFrame([['C','food',60,100,'2020-01-01'],['C','tech',40,100,'2020-01-01']],
                       columns=['target','industry','sales','total_sales','available_at'])
        p=pd.DataFrame([['A','food','2020-01-01'],['B','tech','2020-01-01'],['D','tech','2020-01-01']],
                       columns=['symbol','industry','available_at'])
        e=segment_edges(s,p,pd.Series({'A':10,'B':10,'D':30}),'2020-02-01').set_index('peer')
        np.testing.assert_allclose(e.loc[['A','B','D'],'weight'],[.6,.1,.3])
        with self.assertRaises(ValueError): segment_edges(s,p,pd.Series({'A':10,'B':10,'D':30}),'2019-01-01')

    def test_missing_segment_does_not_renormalize(self):
        s=pd.DataFrame([['C','food',60,100,'2020-01-01'],['C','tech',40,100,'2020-01-01']],
                       columns=['target','industry','sales','total_sales','available_at'])
        p=pd.DataFrame([['A','food','2020-01-01']],columns=['symbol','industry','available_at'])
        self.assertTrue(segment_edges(s,p,pd.Series({'A':10}),'2020-02-01').empty)

    def test_ridge_shrinks_and_is_feasible(self):
        w=fit_group_weights(np.eye(4),np.array([2.,0,0,0]),1.)
        self.assertAlmostEqual(w.sum(),1); self.assertTrue((w>=0).all())
        strong=fit_group_weights(np.eye(4),np.array([2.,0,0,0]),100.)
        self.assertLess(np.linalg.norm(strong-.25),np.linalg.norm(w-.25))
        np.testing.assert_allclose(fit_group_weights(np.eye(4),np.ones(4),1.),.25)

    def test_future_labels_cannot_change_fitted_weights(self):
        past=(pd.Timestamp('2019-11-01'),pd.Timestamp('2019-12-01'),np.eye(2),np.array([.2,.1]),100)
        boundary=(pd.Timestamp('2019-12-01'),pd.Timestamp('2020-01-01'),np.eye(2),np.array([999.,-999.]),100)
        future=(pd.Timestamp('2020-02-01'),pd.Timestamp('2020-03-01'),np.eye(2),np.array([-999.,999.]),100)
        w,n,last=fit_asof([past,boundary,future],'2020-01-01',min_periods=1)
        expected,_,_=fit_asof([past],'2020-01-01',min_periods=1)
        np.testing.assert_allclose(w,expected);self.assertEqual(n,1)
        self.assertLess(last,pd.Timestamp('2020-01-01'))


if __name__=='__main__': unittest.main()
