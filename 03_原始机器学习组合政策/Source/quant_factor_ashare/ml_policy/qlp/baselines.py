# Research project author: Gilmour Jia
"""Baseline target-weight builders, so every candidate is judged by one engine."""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["topn_target_weights", "equal_weight_targets",
           "equal_weight_grid", "topn_grid"]


def equal_weight_grid(a, t_list, cap: float = 0.03, cash: float = 0.0) -> np.ndarray:
    """Equal weight over each decision date's tradable universe.

    This is the honest null hypothesis for a policy that holds several hundred
    names: equal weighting costs almost nothing to compute, needs no model, and
    captures the same market beta.  A model that cannot beat it net of costs has
    not earned its complexity.  Weights are capped, and anything the cap forces
    out stays in cash, exactly as the policy's own targets do.

    ``cash`` holds back a fixed fraction before the names are weighted.  That is
    how a concentrated policy gets compared honestly: a 30-name book with a 3%
    cap is structurally long cash, so benchmarking it against a fully invested
    index measures the cash weight, not the selection.
    """
    grid = np.zeros((len(a.dates), len(a.symbols)), dtype=np.float64)
    diag = []
    for t in t_list:
        t = int(t)
        elig = a.tradable[t] & a.universe[t]
        ids = np.flatnonzero(elig)
        if ids.size == 0:
            continue
        budget = 1.0 - float(cash)
        w = min(cap, budget / ids.size)
        grid[t, ids] = w
        diag.append(dict(t=t, n_above_1e4=int(ids.size),
                         cash=float(1.0 - w * ids.size), max_w=float(w)))
    return grid, diag


def topn_grid(a, score: np.ndarray, t_list, n: int = 50, cap: float = 0.03,
              buffer: float = 0.6) -> np.ndarray:
    """Rank-and-hold targets from a pre-computed score matrix.

    ``score`` is (dates, symbols), higher is better.  The retention buffer keeps
    a name already held while it stays inside the widened band, which is what
    stops a rank-and-hold book from churning on rank noise.
    """
    grid = np.zeros((len(a.dates), len(a.symbols)), dtype=np.float64)
    held: set = set()
    band_n = int(np.ceil(n * (1.0 + buffer)))
    for t in t_list:
        t = int(t)
        elig = a.tradable[t] & a.universe[t]
        ids = np.flatnonzero(elig)
        if ids.size == 0:
            continue
        vals = np.where(np.isfinite(score[t][ids]), score[t][ids], -np.inf)
        order = ids[np.argsort(-vals, kind="stable")]
        band = set(order[:band_n].tolist())
        kept = [j for j in order if j in held and j in band][:n]
        chosen = kept + [j for j in order if j not in kept][: max(0, n - len(kept))]
        if not chosen:
            continue
        grid[t, chosen] = min(cap, 1.0 / len(chosen))
        held = set(chosen)
    return grid


def topn_target_weights(score: pd.DataFrame, eligible: pd.DataFrame,
                        n: int = 50, buffer: float = 0.6, cap: float = 0.03,
                        reserve: float = 0.0) -> pd.DataFrame:
    """Rank-and-hold targets with a retention buffer, as F17 does.

    ``score`` is a date x symbol cross-sectional score (higher is better) and
    ``eligible`` the point-in-time universe. A name already held inside the
    widened band is kept before new names are admitted, which is what the
    buffer is for. ``reserve`` is a fraction of NAV held back for costs and is
    applied uniformly, never normalised past ``cap``.
    """
    vals = score.reindex_like(eligible).to_numpy(dtype=np.float64)
    el = eligible.to_numpy(dtype=bool)
    vals = np.where(np.isfinite(vals), vals, -np.inf)
    good = el & np.isfinite(vals)
    n_dates, n_sec = vals.shape
    # Note: n * cap is deliberately allowed to fall short of 1.0. The baseline
    # caps each name at ``cap`` and leaves the remainder in cash, rather than
    # scaling weights up past the cap to force full investment.

    out = np.zeros((n_dates, n_sec), dtype=np.float64)
    band_n = int(np.ceil(n * (1.0 + buffer)))
    held = set()
    for t in range(n_dates):
        row = np.flatnonzero(good[t])
        if row.size == 0:
            continue
        order = row[np.argsort(-vals[t][row], kind="stable")]
        band = set(order[:band_n].tolist())
        kept = [j for j in order if j in held and j in band][:n]
        chosen = kept + [j for j in order if j not in kept][: max(0, n - len(kept))]
        if not chosen:
            continue
        out[t, chosen] = min(cap, 1.0 / len(chosen)) * (1.0 - reserve)
        held = set(chosen)
    return pd.DataFrame(out, index=eligible.index, columns=eligible.columns)


def equal_weight_targets(eligible: pd.DataFrame, cap: float = 0.03,
                         n: int = 50) -> pd.DataFrame:
    """One-in-N equal weighting of the first eligible names by column order."""
    el = eligible.to_numpy(dtype=bool)
    out = np.zeros(el.shape, dtype=np.float64)
    for t in range(el.shape[0]):
        row = np.flatnonzero(el[t])[:n]
        if row.size:
            out[t, row] = min(cap, 1.0 / row.size)
    return pd.DataFrame(out, index=eligible.index, columns=eligible.columns)
