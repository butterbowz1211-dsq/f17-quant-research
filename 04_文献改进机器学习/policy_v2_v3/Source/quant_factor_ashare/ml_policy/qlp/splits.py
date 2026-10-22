# Research project author: Gilmour Jia
"""Strict expanding-window splits with an explicitly recorded, frozen boundary set.

Prior work in this project re-used 2023+ results to choose factors and
parameters, so those years can never be restored to out-of-sample status.
This module does not pretend otherwise: it marks each outer test year with
whether it was already used in earlier decisions, and reports the untouched
folds separately from the contaminated ones.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = ["Fold", "expanding_folds", "holdout_fold", "CONTAMINATED_FROM"]

# Years whose results were already inspected while choosing factors, weights and
# parameters. Any performance measured on these is exploratory, never OOS.
CONTAMINATED_FROM = 2023


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp      # inclusive last training date
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    contaminated: bool           # True if this test window was already used

    def label(self) -> str:
        return (f"fold{self.index}: train {self.train_start.date()}..{self.train_end.date()}"
                f" | valid {self.valid_start.date()}..{self.valid_end.date()}"
                f" | test {self.test_start.date()}..{self.test_end.date()}"
                f" | {'CONTAMINATED' if self.contaminated else 'untouched'}")


def expanding_folds(dates, *, train_years: int = 6, valid_years: int = 2,
                    test_years: int = 1, min_test_days: int = 120,
                    first_year: int | None = None) -> list[Fold]:
    """Year-by-year expanding folds over the supplied trading calendar.

    ``first_year`` sets the year the training window starts from.  It defaults to
    the first date present, which is the honest choice only when the panel is
    usable from its first row.  This panel is not: there are no stocks with 252
    listing days and 60 days of turnover before 2011, so training on 2010 adds
    nothing but empty cross-sections.  Anchoring on the usable start also puts
    the untouched test years where they belong (2021 and 2022) instead of
    silently labelling 2018 as "fold 0".
    """
    idx = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    if idx.empty:
        return []
    if first_year is not None:
        idx = idx[idx.year >= int(first_year)]
        if idx.empty:
            return []
    first_year = int(idx[0].year)
    last_year = int(idx[-1].year)
    folds: list[Fold] = []
    k = 0
    # The first test year starts after train + validation have been consumed.
    test_year = first_year + train_years + valid_years
    while test_year <= last_year:
        train_start = pd.Timestamp(f'{first_year}-01-01')
        train_end = pd.Timestamp(f'{test_year - valid_years - 1}-12-31')
        valid_start = pd.Timestamp(f'{test_year - valid_years}-01-01')
        valid_end = pd.Timestamp(f'{test_year - 1}-12-31')
        test_start = pd.Timestamp(f'{test_year}-01-01')
        test_end = pd.Timestamp(f'{test_year + test_years - 1}-12-31')
        test_days = idx[(idx >= test_start) & (idx <= test_end)]
        if len(test_days) >= min_test_days:
            folds.append(Fold(
                index=k,
                train_start=train_start, train_end=train_end,
                valid_start=valid_start, valid_end=valid_end,
                test_start=test_days[0], test_end=test_days[-1],
                contaminated=test_year >= CONTAMINATED_FROM,
            ))
            k += 1
        test_year += test_years
    return folds


def holdout_fold(dates, *, first_year: int = 2013, train_end: int = 2020,
                 valid_years: int = 2, test_start: int | None = None,
                 min_test_days: int = 120) -> Fold:
    """One pre-specified train/validation/test split, with a multi-year test.

    ``expanding_folds`` is the walk-forward protocol; this is the single
    final-fit protocol.  Training stops at ``train_end``, validation covers the
    next ``valid_years`` calendar years, and the test window runs from
    ``test_start`` (default: the year after validation) to the last date in the
    panel.  Nothing about the test window is used to choose the split.

    The returned fold carries ``contaminated`` from ``CONTAMINATED_FROM``; it is
    metadata, not a claim that the test is clean OOS.  If earlier work chose
    factors on 2023+ data, the factor layer is still contaminated even though
    the model weights never saw these rows.
    """
    idx = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    if idx.empty:
        raise ValueError("no dates supplied")
    idx = idx[idx.year >= int(first_year)]
    if idx.empty:
        raise ValueError(f"no dates on or after {first_year}")
    if int(valid_years) < 1:
        raise ValueError("valid_years must be at least 1")
    train_start = pd.Timestamp(f"{int(first_year)}-01-01")
    train_end_ts = pd.Timestamp(f"{int(train_end)}-12-31")
    valid_start = pd.Timestamp(f"{int(train_end) + 1}-01-01")
    valid_end = pd.Timestamp(f"{int(train_end) + int(valid_years)}-12-31")
    test_start_year = (int(test_start) if test_start is not None
                       else int(train_end) + int(valid_years) + 1)
    test_start_ts = pd.Timestamp(f"{test_start_year}-01-01")
    if train_end_ts < train_start:
        raise ValueError("train_end precedes first_year")
    if test_start_ts <= valid_end:
        raise ValueError("test_start must be after the validation window")
    test_days = idx[(idx >= test_start_ts)]
    if len(test_days) < min_test_days:
        raise ValueError(f"test window has {len(test_days)} days, need {min_test_days}")
    return Fold(index=0, train_start=train_start, train_end=train_end_ts,
                valid_start=valid_start, valid_end=valid_end,
                test_start=test_days[0], test_end=test_days[-1],
                contaminated=test_start_year >= CONTAMINATED_FROM)