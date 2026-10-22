# Research project author: Gilmour Jia
"""The 17 original F17 signal definitions; causal, no NN graph/amount proxy.

Original source remains untouched. CO_SKEW and ISKEW preserve actual F17
definitions, with their deviations from the names documented in the dictionary.
"""
import numpy as np
import pandas as pd
from spec import FACTORS


def divide(a, b):
    return a / b.replace(0, np.nan)


def compound(r, window=21, min_periods=15):
    return np.expm1(np.log1p(r.where(r > -1)).rolling(window, min_periods=min_periods).sum())


def top5(r):
    # Repair: old np.sort put NaNs into the top five. Require the full window
    # to avoid redefining MAX5 around missing observations. Shared for all models.
    a = r.to_numpy(dtype=float)
    result = np.full(a.shape, np.nan)
    if len(a) >= 21:
        # Bounded column chunks avoid millions of Python rolling callbacks.
        for start in range(0, a.shape[1], 64):
            windows = np.lib.stride_tricks.sliding_window_view(a[:, start:start+64], 21, axis=0)
            value = np.partition(windows, 16, axis=-1)[..., -5:].mean(axis=-1)
            value[~np.isfinite(windows).all(axis=-1)] = np.nan
            result[20:, start:start+64] = value
    return pd.DataFrame(result, index=r.index, columns=r.columns)


def build_factors(m):
    """m uses aligned DataFrames and contemporaneously available industry codes."""
    close, op, high = m["adjusted_close"], m["adjusted_open"], m["adjusted_high"]
    r = close.pct_change(fill_method=None)
    amount, turn, cap = m["amount"], m["turnover"], m["float_cap"]
    # Preserve original same-day close-cap-weighted market proxy, disclose it.
    weights = cap.div(cap.sum(axis=1), axis=0)
    market = (r * weights).sum(axis=1, min_count=1)
    vm = market.rolling(252, min_periods=126).var()
    cov = r.rolling(252, min_periods=126).cov(market)
    beta = cov.div(vm, axis=0)
    vi = r.rolling(252, min_periods=126).var()
    residual = r - beta.mul(market, axis=0)
    peers = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    industry = m["industry"]
    if isinstance(industry, pd.Series):
        industry = pd.DataFrame(np.tile(industry.reindex(close.columns).to_numpy(), (len(close), 1)),
                                index=close.index, columns=close.columns)
    for t in range(len(close)):
        groups = industry.iloc[t]
        rr = r.iloc[t].where(m["traded"].iloc[t])
        group_sum = rr.groupby(groups, dropna=False).transform("sum")
        group_count = rr.groupby(groups, dropna=False).transform("count")
        self_valid = rr.notna().astype(int)
        peers.iloc[t] = (group_sum - rr.fillna(0)) / (group_count - self_valid).replace(0, np.nan)
    mi = r.rolling(252, min_periods=126).mean()
    mm = market.rolling(252, min_periods=126).mean()
    dm = market - mm
    dm2_mean = dm.pow(2).rolling(252, min_periods=126).mean()
    dm_mean = dm.rolling(252, min_periods=126).mean()
    p1 = r.mul(dm.pow(2), axis=0).rolling(252, min_periods=126).mean()
    p3 = r.mul(dm, axis=0).rolling(252, min_periods=126).mean()
    co_num = p1 - mi.mul(dm2_mean, axis=0) - 2 * (p3 - mi.mul(dm_mean, axis=0)).mul(mm, axis=0)
    out = {
        "TURN": -turn.rolling(21, min_periods=10).mean(),
        "MAX5": -top5(r),
        "INTRADAY": -compound(divide(close, op) - 1),
        "ILLIQ": np.log1p(divide(r.abs(), amount).rolling(21, min_periods=10).mean() * 1e8),
        "COMMOVE": r.rolling(60, min_periods=40).corr(peers),
        "TURN_SHOCK": -(turn.rolling(5, min_periods=3).mean() /
                        turn.rolling(60, min_periods=30).mean() - 1),
        "REV_1W": -(close / close.shift(5) - 1),
        "ISKEW": -r.rolling(60, min_periods=40).skew(),
        "IVOL": -np.sqrt((vi - beta.pow(2).mul(vm, axis=0)).clip(lower=1e-10) * 252),
        "OVERNIGHT": compound(divide(op, close.shift(1)) - 1),
        "LEAD_LAG": -r.rolling(252, min_periods=126).cov(market.shift(1)).div(vm, axis=0),
        "CO_SKEW": -divide(co_num, np.sqrt(vi).mul(vm, axis=0)),
        "BETA": -beta,
        "REV_LT": -(close.shift(252) / close.shift(756) - 1),
        "SIZE": -np.log(cap.where(cap > 0)),
        "HIGH52": divide(close, high.rolling(252, min_periods=126).max()),
        "RES_MOM": residual.rolling(231, min_periods=126).sum().shift(21),
    }
    return {f: out[f].replace([np.inf, -np.inf], np.nan) for f in FACTORS}
