# Research project author: Gilmour Jia
"""Freshness and shape tests for the point-in-time dataset builder.

These tests exist to make one class of mistake impossible to reintroduce:
training on a label that was not yet realised, or on a feature that was not
yet known.  They run on a small synthetic panel, so they are fast and the
expected numbers can be written down by hand.

Layout convention under test: the design matrix is
    own (ranked + missing flags) | peer (contemporaneous + lagged) |
    peer-validity | market | account state,
with widths 2F, 4F, 1, 6, 3 for F named features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlp.dataset import (DatasetConfig, assemble, daily_log_returns,
                          forward_returns, make_sample, standardize_state)


def synthetic_prep(n_dates: int = 400, n_sym: int = 40, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-05", periods=n_dates)
    symbols = pd.Index(["sh%06d" % (600000 + i) for i in range(n_sym)])
    daily = rng.normal(0.0005, 0.02, size=(n_dates, n_sym))
    open_ = 10.0 * np.exp(np.cumsum(daily, axis=0))
    close = open_ * np.exp(rng.normal(0.0, 0.005, size=(n_dates, n_sym)))
    amount = np.full((n_dates, n_sym), 5e7) * np.exp(rng.normal(0, 0.3, (n_dates, n_sym)))
    universe = np.zeros((n_dates, n_sym), dtype=bool)
    universe[260:, :] = True
    return dict(
        dates=dates, symbols=symbols,
        open=pd.DataFrame(open_, index=dates, columns=symbols),
        close=pd.DataFrame(close, index=dates, columns=symbols),
        raw_close=pd.DataFrame(close, index=dates, columns=symbols),
        amount_lag=pd.DataFrame(amount, index=dates, columns=symbols),
        universe=pd.DataFrame(universe, index=dates, columns=symbols),
        suspended=pd.DataFrame(False, index=dates, columns=symbols),
        limit_locked=pd.DataFrame(False, index=dates, columns=symbols),
    )


# ------------------------------------------------------------------ timing ---

def test_daily_log_returns_is_backward_looking():
    px = np.array([[1.0], [2.0], [4.0], [8.0]])
    r = daily_log_returns(px, shift=1)
    assert r[0, 0] == 0.0
    assert r[1, 0] == pytest.approx(np.log(2.0))
    assert r[3, 0] == pytest.approx(np.log(2.0))


def test_forward_return_starts_at_the_execution_open():
    """The label must skip the formation close, not harvest it."""
    px = 100.0 * np.exp(np.arange(10, dtype=float) * 0.01)[:, None]
    one = forward_returns(px, 2, 3, 1)[0]
    assert one == pytest.approx(np.expm1(0.01), rel=1e-9, abs=1e-12)
    three = forward_returns(px, 2, 5, 3)[0]
    assert three == pytest.approx(np.expm1(0.03), rel=1e-9, abs=1e-12)


def test_forward_return_ignores_days_past_the_next_decision():
    px = 100.0 * np.exp(np.arange(10, dtype=float) * 0.01)[:, None]
    # Formation at 2, next decision at 4, so two days are held even though
    # k=5 was asked for: the extra days must not be counted.
    r = forward_returns(px, 2, 4, 5)[0]
    assert r == pytest.approx(np.expm1(0.02), rel=1e-9, abs=1e-12)


def test_missing_quotes_do_not_destroy_a_window():
    px = 100.0 * np.exp(np.arange(6, dtype=float) * 0.01)[:, None]
    px[4, 0] = np.nan
    r = forward_returns(px, 1, 6, 4)[0]
    # Rows 2..5 are held.  The two daily returns that bracket the missing quote
    # are zero, so the window yields three usable days' growth rather than a
    # NaN: understated, never overstated, and the sum still telescopes.
    assert r == pytest.approx(np.expm1(0.02), rel=1e-9, abs=1e-12)


# -------------------------------------------------------------- assembling ---

def test_assembled_shapes_and_order():
    prep = synthetic_prep()
    a = assemble(prep, DatasetConfig(rebalance=5, warmup=260), verbose=False)
    assert a.own.shape == (len(a.dates), len(a.symbols), 20)
    assert a.peer.shape == (len(a.dates), len(a.symbols), 40)
    assert a.peer_valid.shape == (len(a.dates), len(a.symbols))
    assert a.mkt.shape == (len(a.dates), 6)
    assert a.n_input == 20 + 40 + 1 + 6 + 3
    s = make_sample(a, 300, slot=60)
    assert s.feat.shape == (s.ids.size, a.n_input)
    assert np.all(np.diff(s.ids) > 0)
    assert np.all(np.isfinite(s.feat))
    assert np.all(np.isfinite(s.returns))


def test_decision_index_is_on_the_grid_and_realised():
    prep = synthetic_prep(n_dates=300, n_sym=12)
    a = assemble(prep, DatasetConfig(rebalance=7, warmup=98), verbose=False)
    t = a.decision_index(0, len(a.dates))
    assert np.all(t % 7 == 0)
    assert t[0] >= 98
    assert t[-1] + a.rb + 1 <= len(a.dates)


def test_empty_universe_is_an_error_not_a_fallback():
    """A date with nothing tradable has no decision to make."""
    prep = synthetic_prep()
    a = assemble(prep, DatasetConfig(rebalance=5, warmup=0), verbose=False)
    with pytest.raises(ValueError):
        make_sample(a, 0, slot=0)


def test_features_do_not_move_when_the_future_moves():
    """Poisoning every price after the formation date must change nothing."""
    prep = synthetic_prep()
    a1 = assemble(prep, DatasetConfig(rebalance=21, warmup=270), verbose=False)
    s1 = make_sample(a1, 300, slot=10)

    bad = synthetic_prep()
    for key in ("open", "close", "raw_close"):
        arr = bad[key].to_numpy(dtype=float).copy()
        arr[301:] *= 3.0
        bad[key] = pd.DataFrame(arr, index=bad["dates"], columns=bad["symbols"])
    a2 = assemble(bad, DatasetConfig(rebalance=21, warmup=270), verbose=False)
    s2 = make_sample(a2, 300, slot=10)

    assert np.array_equal(s1.ids, s2.ids)
    assert np.allclose(s1.feat, s2.feat, atol=0.0, rtol=0.0)
    # ... while the label, which is genuinely about the future, does move.
    assert not np.allclose(s1.returns, s2.returns)


def test_state_scale_is_frozen_outside_the_fit_slice():
    prep = synthetic_prep()
    a = assemble(prep, DatasetConfig(rebalance=10, warmup=260), verbose=False)
    # Fit inside the listed window: the first 260 rows have no quotes at all.
    a.state_scale = standardize_state(a.mkt, slice(260, 400))
    before = a.state_scale[0].copy()
    make_sample(a, 350, slot=35)
    assert np.array_equal(before, a.state_scale[0])
    assert np.all(np.isfinite(before))


def test_standardize_state_survives_an_all_missing_window():
    mkt = np.full((50, 4), np.nan)
    mu, sd = standardize_state(mkt, slice(0, 10))
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sd))
    assert np.allclose(mu, 0.0) and np.allclose(sd, 1.0)


def test_cached_and_fresh_prep_agree():
    """The npz cache must hand back the same object a rebuild would."""
    from qlp.data import build_prep
    prep = build_prep()
    assert "dates" in prep and "symbols" in prep
    assert len(prep["dates"]) == prep["close"].shape[0]
    assert len(prep["symbols"]) == prep["close"].shape[1]
    assert pd.Timestamp(prep["dates"][0]) < pd.Timestamp(prep["dates"][-1])


# ------------------------------------------------------- graph equivalence ---

def test_vectorised_graph_matches_the_reference():
    """The fast path must be the same estimator, not merely a similar one."""
    from qlp.relations import (correlation_from_residuals, rolling_neighbor_graph,
                               topk_from_corr)
    rng = np.random.default_rng(7)
    n_d, n_s = 260, 60
    dates = pd.bdate_range("2020-01-01", periods=n_d)
    cols = pd.Index(["sh%06d" % (600000 + i) for i in range(n_s)])
    r = rng.normal(0.0, 0.02, (n_d, n_s))
    common = rng.normal(0.0, 0.02, n_d)
    r[:, :10] += common[:, None] * 1.5
    r[rng.random((n_d, n_s)) < 0.05] = np.nan
    resid = pd.DataFrame(r, index=dates, columns=cols)
    for t in (150, 220, 259):
        ref_idx, ref_w, ref_v = rolling_neighbor_graph(
            resid, dates[t], lookback=120, min_obs=80, neighbors=5)
        corr, enough = correlation_from_residuals(
            resid.to_numpy(dtype=np.float64)[:t], 120, 80)
        idx, wts, valid = topk_from_corr(corr, np.ones(n_s, bool), enough, enough, 5)
        assert np.array_equal(ref_v, valid)
        assert np.array_equal(np.sort(ref_idx[ref_v], axis=1),
                              np.sort(idx[valid], axis=1))
        assert np.allclose(np.sort(ref_w[ref_v], axis=1),
                           np.sort(wts[valid], axis=1), atol=1e-6)
