# Research project author: Gilmour Jia
"""Cross-sectional features that only use information available at formation time."""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["trailing_return", "realized_vol", "amihud_illiq", "max_daily_return",
           "cross_sectional_rank", "build_stock_features", "build_market_state"]

RETURN_WINDOWS = (5, 20, 60, 120, 252)
VOL_WINDOWS = (20, 60)


def _safe_log_ratio(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    ratio = a / b.where(b > 0)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    return np.log(ratio)


def trailing_return(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Log return over exactly ``window`` trading days, lagged by none."""
    return _safe_log_ratio(close, close.shift(window))


def realized_vol(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualised realised volatility of daily log returns."""
    ret = np.log(close).diff()
    return ret.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(252)


def amihud_illiq(close: pd.DataFrame, amount: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Average |return| per unit of currency turnover."""
    ret = np.log(close).diff().abs()
    daily = ret / amount.where(amount > 0)
    daily = daily.replace([np.inf, -np.inf], np.nan)
    return daily.rolling(window, min_periods=max(5, window // 2)).mean()


def max_daily_return(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Largest single-day return in the window (lottery proxy)."""
    ret = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    return ret.rolling(window, min_periods=max(3, window // 4)).max()


def cross_sectional_rank(frame: pd.DataFrame, clip: float = 1.0) -> pd.DataFrame:
    """Map each date's cross-section onto [-clip, clip] by rank."""
    ranks = frame.rank(axis=1, pct=True)
    return (ranks - 0.5) * 2.0 * clip


def build_stock_features(close: pd.DataFrame, raw_close: pd.DataFrame,
                         amount: pd.DataFrame, universe: pd.DataFrame | None = None,
                         *, rank: bool = True) -> dict:
    """Feature block plus a missingness mask, ranked within each date."""
    feats = {}
    for w in RETURN_WINDOWS:
        feats[f'ret_{w}'] = trailing_return(close, w)
    for w in VOL_WINDOWS:
        feats[f'vol_{w}'] = realized_vol(close, w)
    feats['illiq'] = amihud_illiq(close, amount, 20)
    feats['max5'] = max_daily_return(close, 20)
    feats['turnover'] = amount / amount.rolling(60, min_periods=20).mean()

    # Everything is expressed relative to the tradable cross-section of the day.
    if universe is not None:
        mask = universe.reindex_like(close).fillna(False)
        for k in list(feats):
            feats[k] = feats[k].where(mask)

    missing = {f'{k}_miss': v.isna().astype('float64') for k, v in feats.items()}
    if rank:
        feats = {k: cross_sectional_rank(v) for k, v in feats.items()}

    # Median-fill only after ranking, and only with the same day's cross-section:
    # the fill is a cross-sectional statement, not a time-series one.
    filled = {}
    for k, v in feats.items():
        med = v.median(axis=1)
        filled[k] = v.T.fillna(med).T
    return dict(features=filled, missing=missing)


def build_market_state(close: pd.DataFrame, universe: pd.DataFrame | None = None,
                       window: int = 60) -> pd.DataFrame:
    """Market-level series available at each date, shared by all stocks."""
    ret = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    if universe is not None:
        ret = ret.where(universe.reindex_like(ret).fillna(False))
    ew = ret.mean(axis=1, skipna=True)
    state = pd.DataFrame(index=close.index)
    state['mkt_ret_5'] = ew.rolling(5, min_periods=3).sum()
    state['mkt_ret_20'] = ew.rolling(20, min_periods=10).sum()
    state['mkt_ret_60'] = ew.rolling(60, min_periods=30).sum()
    state['mkt_vol_20'] = ew.rolling(20, min_periods=10).std() * np.sqrt(252)
    state['mkt_dispersion'] = ret.std(axis=1, skipna=True)
    state['n_tradable'] = universe.sum(axis=1) if universe is not None else close.notna().sum(axis=1)
    return state