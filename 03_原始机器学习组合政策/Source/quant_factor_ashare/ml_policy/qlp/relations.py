# Research project author: Gilmour Jia
"""Cross-sectional relation graph built only from information available at formation."""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["market_residual_returns", "rolling_neighbor_graph", "aggregate_neighbors"]


def market_residual_returns(close: pd.DataFrame, market: pd.Series | None = None) -> pd.DataFrame:
    """Daily returns with the market move removed.

    Common variation is subtracted so a positive correlation reflects
    co-movement beyond the market, which is what the linkage hypothesis is
    about. When ``market`` is not supplied it is the cross-sectional mean of
    that date's returns, which is computable in real time.
    """
    ret = close.pct_change(fill_method=None)
    ret = ret.replace([np.inf, -np.inf], np.nan)
    if market is None:
        market = ret.mean(axis=1, skipna=True)
    return ret.sub(market, axis=0)


def rolling_neighbor_graph(resid: pd.DataFrame, as_of: pd.Timestamp,
                           lookback: int = 120, min_obs: int = 80,
                           neighbors: int = 10, positive_only: bool = True):
    """Positive-correlation neighbours formed strictly before ``as_of``.

    Returns ``(indices, weights, valid)``. ``indices[k, j]`` is the j-th
    neighbour of stock k and ``weights[k, j]`` its normalised edge weight.
    Stocks with fewer than ``min_obs`` usable observations, or with no
    positive partner, are marked ``valid[k] = False`` rather than being given
    a silent zero vector.
    """
    window = resid.loc[resid.index < pd.Timestamp(as_of)].tail(lookback)
    n = window.shape[1]
    indices = np.zeros((n, neighbors), dtype=np.int32)
    weights = np.zeros((n, neighbors), dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    if window.empty:
        return indices, weights, valid

    values = window.to_numpy(dtype=np.float64)
    complete = np.isfinite(values)
    counts = complete.sum(axis=0)
    # Correlate on pairwise-complete observations: centre each column over the
    # rows that stock actually has, then treat a missing partner as no
    # contribution (zero after centring).
    with np.errstate(invalid='ignore'):
        col_mean = np.nanmean(np.where(complete, values, np.nan), axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0)
    filled = np.where(complete, values - col_mean, 0.0)
    norms = np.sqrt((filled ** 2).sum(axis=0))
    safe = norms > 0
    denom = np.outer(norms, norms)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.where(denom > 0, (filled.T @ filled) / denom, 0.0)
    np.fill_diagonal(corr, 0.0)
    corr = np.clip(np.nan_to_num(corr, nan=0.0), -1.0, 1.0)

    for k in range(n):
        if counts[k] < min_obs:
            continue
        row = corr[k].copy()
        row[~safe] = 0.0
        if positive_only:
            row[row <= 0] = 0.0
        if not row.any():
            continue
        take = np.argsort(-row, kind='stable')[:neighbors]
        vals = row[take]
        keep = vals > 0
        if not keep.any():
            continue
        take, vals = take[keep], vals[keep]
        indices[k, : len(take)] = take
        weights[k, : len(vals)] = vals / vals.sum()
        valid[k] = True
    return indices, weights, valid


def aggregate_neighbors(feature: np.ndarray, indices: np.ndarray,
                        weights: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Edge-weight-weighted mean of a neighbour feature block.

    ``feature`` is (n_stocks, n_features). Stocks without a valid graph get a
    zero vector; the caller knows via ``valid`` that this is absence of
    information, not a genuine zero reading.  Missing neighbour readings are
    renormalised away rather than read as zero, so a halted partner dilutes the
    average instead of dragging it towards a value it never had.
    """
    n_stocks, n_feat = feature.shape
    if indices.shape[0] != n_stocks:
        raise ValueError("graph and feature must share the stock axis")
    out = np.zeros((n_stocks, n_feat), dtype=np.float64)
    if n_stocks == 0 or indices.size == 0:
        return out
    gather = feature[indices].astype(np.float64, copy=False)      # (n, k, f)
    gather = np.where(np.isfinite(gather), gather, 0.0)
    w = np.where(valid[:, None], weights, 0.0).astype(np.float64)  # (n, k)
    used = np.where(np.isfinite(feature[indices]), w[:, :, None], 0.0).sum(axis=1)
    num = (gather * w[:, :, None]).sum(axis=1)
    den = np.where(used > 0, used, 1.0)
    out = num / den
    out[~valid] = 0.0
    return out


def correlation_from_residuals(filled: np.ndarray, lookback: int, min_obs: int):
    """Positive-part correlation graph from a residuals matrix, vectorised.

    Used by the dataset builder when the same window is needed for all dates at
    once.  Centring does not affect a correlation, so it can be done over the
    full sample; the lookback and the observation floor are enforced on the
    window itself.  Returns ``(corr, enough_obs)``.
    """
    values = np.asarray(filled, dtype=np.float64)
    # The window is centred on its own rows and a missing reading is replaced by
    # that column's window mean -- i.e. it contributes nothing after centring.
    # Doing this over a longer history instead would shift every pairwise
    # correlation by a few hundredths whenever a name has gaps, which is exactly
    # the size of the edge weights being estimated.  The observation floor is
    # part of the same window: counting the full history made this call allocate
    # a (t x n_symbols) boolean on every decision date and turned assembly into
    # an O(t) memory/time loop.
    sub = values[-int(lookback):]
    present = np.isfinite(sub)
    counts = present.sum(axis=0)
    enough = counts >= int(min_obs)
    with np.errstate(invalid='ignore'):
        col_mu = np.nanmean(np.where(present, sub, np.nan), axis=0)
    col_mu = np.nan_to_num(col_mu, nan=0.0)
    sub = np.where(np.isfinite(sub), sub - col_mu, 0.0)
    # Accumulate in double precision: the cross-product of a 120 x 5000 block in
    # float32 loses roughly three digits, the same order as the edge weights
    # this function exists to produce.
    norm = np.sqrt((sub ** 2).sum(axis=0))
    denom = np.outer(norm, norm)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.where(denom > 0, (sub.T @ sub) / denom, 0.0)
    corr = np.clip(np.nan_to_num(corr, nan=0.0), -1.0, 1.0)
    np.fill_diagonal(corr, 0.0)
    return corr.astype(np.float32), enough


def topk_from_corr(corr: np.ndarray, eligible: np.ndarray, trusted: np.ndarray,
                   enough: np.ndarray, neighbors: int):
    """Row-wise top-k positive partners, vectorised over the whole matrix.

    A partner only counts when it is inside the point-in-time eligible set and
    has enough observations of its own; the caller's own row is what matters,
    so an untrusted partner never receives weight even if it correlates.
    """
    n = corr.shape[0]
    work = np.array(corr, dtype=np.float64, copy=True)
    keep = eligible & trusted & enough
    work[:, ~keep] = 0.0
    work[~trusted, :] = 0.0
    k = int(min(neighbors, max(0, n - 1)))
    indices = np.zeros((n, k), dtype=np.int32)
    weights = np.zeros((n, k), dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    if k == 0 or not trusted.any():
        return indices, weights, valid
    cand = np.flatnonzero(trusted)
    sub = work[np.ix_(cand, np.arange(n))]
    take = np.argpartition(-sub, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(sub, take, axis=1)
    vals = np.where(vals > 0, vals, 0.0)
    order = np.argsort(-vals, axis=1, kind='stable')
    vals = np.take_along_axis(vals, order, axis=1)
    take = np.take_along_axis(take, order, axis=1)
    total = vals.sum(axis=1)
    good = total > 0
    indices[cand] = take
    weights[cand] = np.where(good[:, None], vals / np.where(total > 0, total, 1.0)[:, None], 0.0)
    valid[cand] = good
    return indices, weights, valid
