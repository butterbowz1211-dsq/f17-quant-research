# Research project author: Gilmour Jia
"""Tests for the relation graph, including its guards against look-ahead."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qlp.relations import (aggregate_neighbors, market_residual_returns,
                           rolling_neighbor_graph)


def _resid(n_dates=200, n_sec=5, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2024-01-01', periods=n_dates)
    cols = [f'S{i:02d}' for i in range(n_sec)]
    # S00 and S01 share a common factor so they must end up as neighbours.
    common = rng.normal(0, 0.01, size=n_dates)
    data = rng.normal(0, 0.01, size=(n_dates, n_sec))
    data[:, 0] += common
    data[:, 1] += common
    return pd.DataFrame(data, index=dates, columns=cols)


def test_market_residual_removes_the_cross_sectional_mean():
    close = pd.DataFrame(np.exp(np.cumsum(np.random.default_rng(1).normal(
        0, 0.01, size=(100, 4)), axis=0)),
        index=pd.bdate_range('2024-01-01', periods=100),
        columns=list('ABCD'))
    resid = market_residual_returns(close)
    assert abs(float(resid.mean(axis=1).mean())) < 1e-12


def test_graph_uses_only_rows_before_the_cutoff():
    resid = _resid()
    cutoff = resid.index[120]
    idx_a, w_a, v_a = rolling_neighbor_graph(resid, cutoff)
    # Rewriting the future must not change a graph formed in the past.
    tampered = resid.copy()
    tampered.loc[tampered.index >= cutoff] = 0.0
    idx_b, w_b, v_b = rolling_neighbor_graph(tampered, cutoff)
    assert np.array_equal(idx_a, idx_b)
    assert np.allclose(w_a, w_b)
    assert np.array_equal(v_a, v_b)


def test_weights_are_normalised_and_positive():
    resid = _resid()
    idx, w, valid = rolling_neighbor_graph(resid, resid.index[-1])
    assert valid.any()
    for k in np.flatnonzero(valid):
        real = w[k][w[k] > 0]
        assert real.size > 0
        assert abs(real.sum() - 1.0) < 1e-9
        assert (real > 0).all()
        assert k not in idx[k][: real.size]


def test_correlated_pair_becomes_neighbours():
    resid = _resid()
    idx, w, valid = rolling_neighbor_graph(resid, resid.index[-1], neighbors=1)
    assert valid[0] and idx[0, 0] == 1
    assert valid[1] and idx[1, 0] == 0


def test_insufficient_history_is_flagged_not_zero_filled():
    resid = _resid(n_dates=200, n_sec=3)
    resid.iloc[:150, 2] = np.nan  # S02 has too few usable observations
    idx, w, valid = rolling_neighbor_graph(resid, resid.index[-1], min_obs=80)
    assert not valid[2]
    assert w[2].sum() == 0.0


def test_aggregate_neighbors_matches_manual_weighting():
    feature = np.arange(10, dtype=float).reshape(5, 2)
    idx = np.array([[1, 2], [0, 0], [0, 1], [0, 0], [0, 0]])
    w = np.array([[0.25, 0.75], [0, 0], [1.0, 0.0], [0, 0], [0, 0]])
    valid = np.array([True, False, True, False, False])
    out = aggregate_neighbors(feature, idx, w, valid)
    assert np.allclose(out[0], 0.25 * feature[1] + 0.75 * feature[2])
    assert np.allclose(out[2], feature[0])
    assert np.allclose(out[1], 0.0)  # invalid stays zero and is flagged


def _run_all():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f'  PASS  {name}')
        except Exception as exc:  # noqa: BLE001
            failed.append((name, exc))
            print(f'  FAIL  {name}: {type(exc).__name__}: {exc}')
    print()
    print(f'{len(tests) - len(failed)}/{len(tests)} tests passed')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(_run_all())
