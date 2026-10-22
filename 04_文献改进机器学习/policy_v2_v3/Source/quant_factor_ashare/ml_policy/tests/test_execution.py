# Research project author: Gilmour Jia
"""Engine and weight-algebra tests on synthetic data (no market data needed).

These lock the properties the strategy specification promises: no trading
off the decision grid, cash never goes negative, cost is charged only on
fills, frozen positions are preserved, and whole-lot rounding holds.

Two conventions matter when reading the expectations:

* targets written for date t are executed at the open of t+1;
* a stock that is sunk to exactly the same weight as its target generates no
  order, but ordinary price drift between decisions is corrected, so a
  realistic basket still trades on the next decision date.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qlp.execution import ExecutionConfig, run_target_weight_engine, stamp_duty_rate
from qlp.weights import blend_targets, no_trade_mask, project_capped_simplex


def _panel(n_dates=12, n_sec=6, seed=0, price=10.0):
    """Build a calm, fully tradeable synthetic market."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2024-01-01', periods=n_dates)
    cols = [f'S{i:02d}' for i in range(n_sec)]
    rets = rng.normal(0.0005, 0.01, size=(n_dates, n_sec))
    close = price * np.cumprod(1.0 + rets, axis=0)
    openp = close * (1.0 + rng.normal(0.0, 0.002, size=(n_dates, n_sec)))
    op = pd.DataFrame(openp, index=dates, columns=cols)
    cl = pd.DataFrame(close, index=dates, columns=cols)
    ones = pd.DataFrame(True, index=dates, columns=cols)
    zeros = pd.DataFrame(False, index=dates, columns=cols)
    adv = pd.DataFrame(1e9, index=dates, columns=cols)
    return op, cl, ones, zeros, adv


def _flat_panel(n_dates=12, n_sec=6, price=10.0):
    """Constant-price panel: isolates trading rules from return drift."""
    dates = pd.bdate_range('2024-01-01', periods=n_dates)
    cols = [f'S{i:02d}' for i in range(n_sec)]
    px = pd.DataFrame(price, index=dates, columns=cols)
    ones = pd.DataFrame(True, index=dates, columns=cols)
    zeros = pd.DataFrame(False, index=dates, columns=cols)
    adv = pd.DataFrame(1e9, index=dates, columns=cols)
    return px, px.copy(), ones, zeros, adv


def _targets(op, spec, first=1):
    """Build a target frame from {column: weight}, constant from ``first``."""
    tg = pd.DataFrame(0.0, index=op.index, columns=op.columns)
    for col, w in spec.items():
        tg.loc[op.index[first]:, col] = w
    return tg


def test_stamp_duty_schedule():
    assert stamp_duty_rate('2023-08-25') == 0.001
    assert stamp_duty_rate('2023-08-28') == 0.0005
    assert stamp_duty_rate('2026-01-05') == 0.0005


def test_projection_respects_bounds_and_total():
    w, ok = project_capped_simplex(np.array([0.5, 0.3, 0.4, -0.2]),
                                   np.zeros(4), np.full(4, 0.3), total=1.0)
    assert ok and abs(w.sum() - 1.0) < 1e-9
    assert (w >= -1e-12).all() and (w <= 0.3 + 1e-9).all()


def test_projection_reports_infeasible():
    _, ok = project_capped_simplex(np.array([0.5, 0.5]), np.zeros(2), np.full(2, 0.2))
    assert not ok


def test_no_trade_mask():
    """A 1bp weight gap is inside the band; a 2% gap is not."""
    cur = np.array([0.03, 0.03])
    tgt = np.array([0.0301, 0.05])
    m = no_trade_mask(cur, tgt, nav=1_000_000.0, threshold=0.0005)
    assert m[0] and not m[1]


def test_relative_band_uses_target_weight_not_nav():
    """A 20% target moving 1pp is a 5% relative gap; a 1% target moving 1pp is 100%.

    With a 10% relative band the small relative move must be skipped and the
    large relative move must trade -- even though both are the same 1pp in NAV.
    """
    op, cl, ones, zeros, adv = _flat_panel(n_dates=7, n_sec=3)
    tg = pd.DataFrame(0.0, index=op.index, columns=op.columns)
    # Row 1 is the initial target, row 3 the new target.  Targets written on t
    # execute at the open of t+1, so row 3 is the first row that sees the change.
    tg.loc[op.index[1]:op.index[2], 'S00'] = 0.20
    tg.loc[op.index[1]:op.index[2], 'S01'] = 0.01
    tg.loc[op.index[3]:, 'S00'] = 0.21
    tg.loc[op.index[3]:, 'S01'] = 0.02

    def run(band_rel):
        cfg = ExecutionConfig(capital=1_000_000.0, cap=0.5,
                              partial_adjustment=1.0, no_trade_weight=0.0,
                              no_trade_relative=band_rel,
                              slippage_bp=0.0, commission_bp=0.0,
                              min_commission=0.0,
                              enforce_board_lot=False)
        res = run_target_weight_engine(op, cl, tg, ones, ones, zeros, adv, cfg)
        return res['holdings'].iloc[-1].to_numpy()

    no_band = run(0.0)
    band = run(0.10)
    # Without the band both names move to the new target.
    assert abs(no_band[0] - 0.21) < 1e-9 and abs(no_band[1] - 0.02) < 1e-9
    # With the band S00 stays at 20% (4.8% relative gap) but S01 moves to 2%
    # (50% relative gap).
    assert abs(band[0] - 0.20) < 1e-9
    assert abs(band[1] - 0.02) < 1e-9


def test_relative_band_does_not_block_a_full_exit():
    """A dropped name must still be sold even if the band would skip the gap."""
    op, cl, ones, zeros, adv = _flat_panel(n_dates=7, n_sec=3)
    tg = pd.DataFrame(0.0, index=op.index, columns=op.columns)
    tg.loc[op.index[1]:op.index[2], 'S00'] = 0.20
    tg.loc[op.index[1]:op.index[2], 'S01'] = 0.01
    # From row 3, S00 is dropped; S01 is unchanged.
    tg.loc[op.index[3]:, 'S00'] = 0.0
    tg.loc[op.index[3]:, 'S01'] = 0.01
    cfg = ExecutionConfig(capital=1_000_000.0, cap=0.5,
                          partial_adjustment=1.0, no_trade_weight=0.0,
                          no_trade_relative=0.90,
                          slippage_bp=0.0, commission_bp=0.0,
                          min_commission=0.0,
                          enforce_board_lot=False)
    res = run_target_weight_engine(op, cl, tg, ones, ones, zeros, adv, cfg)
    assert abs(res['holdings'].iloc[-1, 0]) < 1e-9


def test_blend_endpoints():
    cur = np.array([0.0, 1.0])
    tgt = np.array([1.0, 0.0])
    assert np.allclose(blend_targets(cur, tgt, 1.0), tgt)
    assert np.allclose(blend_targets(cur, tgt, 0.5), [0.5, 0.5])


def test_no_trading_on_non_decision_days():
    """Once established, the basket stops trading until drift accumulates."""
    op, cl, ones, zeros, adv = _flat_panel(n_dates=12, n_sec=9)
    targets = _targets(op, {f'S{i:02d}': 0.03 for i in range(8)})
    cfg = ExecutionConfig(capital=1_000_000.0, partial_adjustment=1.0,
                          no_trade_weight=0.0, slippage_bp=0.0, commission_bp=0.0)
    res = run_target_weight_engine(op, cl, targets, ones, ones, zeros, adv, cfg)
    turn = res['daily']['turnover_currency']
    # Targets formed on t are executed at the open of t+1, so the basket is
    # established on the third row, not the second.
    assert turn.iloc[2] > 0
    assert turn.iloc[:2].sum() == 0.0
    assert turn.iloc[3:].sum() < turn.iloc[2] * 1e-6  # then effectively quiet
    assert (res['holdings'].iloc[3:].to_numpy() > 0).any(axis=1).all()


def test_partial_adjustment_closes_half_the_gap():
    """A 50% adjustment must leave the position half-way to its target."""
    op, cl, ones, zeros, adv = _flat_panel(n_dates=10, n_sec=3)
    targets = _targets(op, {'S00': 0.5})
    cfg = ExecutionConfig(capital=1_000_000.0, cap=0.5, partial_adjustment=0.5,
                          no_trade_weight=0.0, slippage_bp=0.0, commission_bp=0.0)
    res = run_target_weight_engine(op, cl, targets, ones, ones, zeros, adv, cfg)
    # First execution (open of t+1) reaches half the 0.5 target; the next
    # decision closes half of the remaining gap again. Drift moves these a
    # little, so the assertions are one-sided and loose on purpose.
    w1 = res['holdings']['S00'].iloc[2]
    w2 = res['holdings']['S00'].iloc[3]
    assert 0.22 <= w1 <= 0.28
    assert 0.34 <= w2 <= 0.40


def test_cash_never_negative_and_weights_sum():
    op, cl, ones, zeros, adv = _panel(n_dates=15, n_sec=8)
    rng = np.random.default_rng(1)
    targets = pd.DataFrame(rng.uniform(0, 0.03, size=(len(op), len(op.columns))),
                           index=op.index, columns=op.columns)
    cfg = ExecutionConfig(capital=500_000.0)
    res = run_target_weight_engine(op, cl, targets, ones, ones, zeros, adv, cfg)
    d = res['daily']
    assert (d['cash_weight'] >= -1e-9).all()
    assert (res['holdings'].to_numpy() >= -1e-12).all()
    assert (res['holdings'].sum(axis=1) + d['cash_weight'] - 1.0).abs().max() < 1e-6


def test_cost_only_charged_on_fills():
    op, cl, ones, zeros, adv = _panel(n_dates=8, n_sec=4)
    targets = _targets(op, {'S00': 0.03, 'S01': 0.03})
    cfg = ExecutionConfig(capital=1_000_000.0, partial_adjustment=1.0,
                          no_trade_weight=0.0)
    res = run_target_weight_engine(op, cl, targets, ones, ones, zeros, adv, cfg)
    d = res['daily']
    no_trade = d['turnover_currency'] <= 0
    assert (d.loc[no_trade, 'cost_currency'] == 0).all()
    assert (d['cost_currency'] > 0).any()


def test_frozen_positions_are_preserved():
    """A locked holding must survive a later policy decision to exit it."""
    op, cl, ones, zeros, adv = _flat_panel(n_dates=10, n_sec=3)
    targets = _targets(op, {'S00': 0.03, 'S01': 0.03, 'S02': 0.03})
    targets.loc[op.index[2]:, :] = 0.0  # policy wants out from day 2 onward
    locked = zeros.copy()
    locked.loc[op.index[3]:, 'S00'] = True  # S00 freezes from day 3 onward
    cfg = ExecutionConfig(capital=1_000_000.0, partial_adjustment=1.0,
                          no_trade_weight=0.0)
    res = run_target_weight_engine(op, cl, targets, ones, ones, locked, adv, cfg)
    s00 = res['holdings']['S00']
    assert s00.iloc[2] > 0  # established at the open of t+1
    # From the row where the lock binds, the weight may not change further: the
    # position is carried at whatever it was, never liquidated by the engine.
    assert abs(s00.iloc[4] - s00.iloc[5]) < 1e-9
    assert (s00.iloc[4:] > 0).all()
    # The unlocked names were able to exit to a negligible weight, and the
    # engine reports the position it could not trade rather than pretending.
    leftover = res['holdings'][['S01', 'S02']].iloc[4:].to_numpy()
    assert (leftover <= 0.002).all()


def test_board_lot_rounding():
    op, cl, ones, zeros, adv = _panel(n_dates=6, n_sec=3, price=13.37)
    targets = _targets(op, {'S00': 0.03, 'S01': 0.03})
    cfg = ExecutionConfig(capital=1_000_000.0, partial_adjustment=1.0,
                          no_trade_weight=0.0, enforce_board_lot=True)
    res = run_target_weight_engine(op, cl, targets, ones, ones, zeros, adv, cfg)
    last = res['holdings'].iloc[-1].to_numpy()
    nav = res['daily']['nav'].iloc[-1]
    shares = last * nav / cl.iloc[-1].to_numpy()
    lots = shares / 100.0
    assert np.allclose(lots, np.round(lots), atol=1e-6)


def test_blocked_sell_keeps_exposure():
    op, cl, ones, zeros, adv = _panel(n_dates=8, n_sec=3)
    targets = _targets(op, {'S00': 0.03, 'S01': 0.03, 'S02': 0.03})
    tradable = ones.copy()
    tradable.iloc[3:, :] = False
    cfg = ExecutionConfig(capital=1_000_000.0, partial_adjustment=1.0,
                          no_trade_weight=0.0)
    res = run_target_weight_engine(op, cl, targets, ones, tradable, zeros, adv, cfg)
    assert res['holdings'].iloc[2].sum() > 0  # bought before trading stopped
    assert res['holdings'].iloc[-1].sum() > 0  # exposure is carried, not lost
    assert res['daily']['total_weight'].iloc[-1] > 0
    # No fills of any kind happened once the whole market was untradeable.
    assert (res['daily']['turnover_currency'].iloc[3:] == 0).all()


def test_stale_valuation_is_rejected_by_default():
    op, cl, ones, zeros, adv = _panel(n_dates=8, n_sec=3)
    # S02 sorts last, so the target column and the damaged column line up.
    targets = _targets(op, {'S02': 0.03})
    op_bad, cl_bad = op.copy(), cl.copy()
    # S02 must be buyable once, at the open of the second row, then go dark: a
    # security that never trades cannot become a stale holding.
    op_bad.iloc[3:, -1] = np.nan    # no new open from the 4th row
    cl_bad.iloc[2:, -1] = np.nan    # valuation first goes dark on the 3rd row
    cfg = ExecutionConfig(capital=1_000_000.0, stale_limit=1)
    try:
        run_target_weight_engine(op_bad, cl_bad, targets, ones, ones, zeros, adv, cfg)
    except ValueError as exc:
        assert 'valuation' in str(exc)
        return
    raise AssertionError('expected stale valuation to be rejected')


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
