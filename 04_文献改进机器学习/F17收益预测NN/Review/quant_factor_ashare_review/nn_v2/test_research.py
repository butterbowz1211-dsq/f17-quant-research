# Research project author: Gilmour Jia
import runtime
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from factors import top5
from research import research_label, research_acceptance, CONTRACT, MODE, AUTHORIZATION
from prepare_research import coherent_prices, RAW, ADJ, active_code_mask
from train import acceptance
from spec import file_hash, write_json, freeze, code_hash


class ResearchTests(unittest.TestCase):
    def test_successor_backfill_and_predecessor_placeholder_are_inactive(self):
        f = pd.DataFrame({"date":pd.to_datetime(["2025-02-14","2025-02-17"]*2),
                          "symbol":["sz300114"]*2+["sz302132"]*2})
        self.assertEqual(active_code_mask(f).tolist(),[True,False,False,True])

    def test_top5_matches_full_window_reference_with_gaps(self):
        rng = np.random.default_rng(73)
        data = pd.DataFrame(rng.normal(size=(64,72)))
        data.iloc[20,3] = np.nan
        expected = data.rolling(21,min_periods=21).apply(lambda x:np.sort(x)[-5:].mean(),raw=True)
        np.testing.assert_allclose(top5(data),expected,equal_nan=True,atol=1e-14)

    def test_research_label_real_endpoints_and_failures(self):
        dates = pd.bdate_range("2020-01-01",periods=30)
        price = np.full((30,6),100.)
        price[22] = 80.
        suspended = np.zeros_like(price,dtype=bool)
        terminal = suspended.copy()
        boundary = suspended.copy()
        price[10,1:3] = np.nan
        suspended[10,2] = True
        suspended[22,3] = True
        terminal[21:,4] = True
        boundary[15,5] = True
        y,r,entry,end = research_label(price,dates,0,suspended,terminal,boundary)
        np.testing.assert_allclose(y[[0,2]],[-.2,-.2])
        self.assertTrue(np.isnan(y[[1,3,4,5]]).all())
        self.assertEqual(r.tolist(),["ok","unexplained_price_gap","ok",
                                   "suspended_endpoint","delisting_unresolved","code_change_unresolved"])
        self.assertEqual(entry,dates[1])
        self.assertEqual(end,dates[22])

    def test_coherent_rounding_not_return_clipping(self):
        f = pd.DataFrame([[10,11,9,10,10,1000,10000,4.01,4.4,3.6,4.0],
                          [10,11,9,10,10,1000,10000,5,4.4,3.6,4]],
                         columns=RAW+ADJ)
        bad,stats = coherent_prices(f)
        self.assertFalse(bad[0])
        self.assertAlmostEqual(f.adjusted_open.iloc[0],4.)
        self.assertTrue(bad[1])
        self.assertTrue(f[ADJ].iloc[1].isna().all())

    def test_research_waiver_does_not_open_strict_gate(self):
        with tempfile.TemporaryDirectory(dir=runtime.ROOT) as d:
            p = Path(d)
            required = ["samples.npz","calendar.csv","label_coverage.csv","security_coverage.csv",
                        "skipped_dates.json","factors_raw.parquet","factors_clean.parquet","sample_quality.parquet"]
            for name in required:
                (p/name).write_bytes(b"test fixture, not real training data")
            write_json(p/"authorization.json",{"user_instruction":AUTHORIZATION,"mode":MODE})
            write_json(p/"quality_report.json",{"checks":dict.fromkeys(
                ["finite_features","unique_keys","label_endpoints","purge_all_folds",
                 "all_17_factors","source_hashes"],True)})
            write_json(p/"package_manifest.json",{"training_contract":CONTRACT,"preparation_code":code_hash(),
                       "files":{x.name:file_hash(x) for x in p.iterdir()}})
            write_json(p/"research_acceptance.json",{
                "status":"RESEARCH_TRAINING_AUTHORIZED_WITH_LIMITATIONS","execution_accepted":False})
            accepted = research_acceptance(p)
            self.assertFalse(accepted["execution_accepted"])
            with self.assertRaises(ValueError):
                acceptance(p)
            frozen = freeze(p/"run",accepted)
            self.assertIn("NOT_verified",frozen["label"]["corporate_actions"])
            (p/"samples.npz").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                research_acceptance(p)


if __name__ == "__main__":
    unittest.main()
