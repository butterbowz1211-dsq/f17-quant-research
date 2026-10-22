# Research project author: Gilmour Jia
"""Contract tests for the final single-fit holdout split."""
from __future__ import annotations

import pandas as pd
import pytest

from qlp.splits import CONTAMINATED_FROM, holdout_fold


def _dates(start="2010-01-01", end="2026-09-28"):
    return pd.bdate_range(start, end)


def test_holdout_puts_train_valid_and_test_on_the_declared_boundaries():
    fold = holdout_fold(_dates(), first_year=2013, train_end=2020,
                        valid_years=2, test_start=2023)
    assert str(fold.train_start.date()) == "2013-01-01"
    assert str(fold.train_end.date()) == "2020-12-31"
    assert str(fold.valid_start.date()) == "2021-01-01"
    assert str(fold.valid_end.date()) == "2022-12-31"
    assert str(fold.test_start.date()) == "2023-01-02"
    assert str(fold.test_end.date()) == "2026-09-28"
    assert fold.contaminated is (2023 >= CONTAMINATED_FROM)


def test_holdout_rejects_a_test_that_overlaps_validation():
    with pytest.raises(ValueError):
        holdout_fold(_dates(), first_year=2013, train_end=2020,
                     valid_years=2, test_start=2022)


def test_holdout_rejects_a_test_window_too_short():
    with pytest.raises(ValueError):
        holdout_fold(_dates(end="2023-02-01"), first_year=2013, train_end=2020,
                     valid_years=2, test_start=2023, min_test_days=120)
