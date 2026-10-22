# Research project author: Gilmour Jia
"""Reproduce the F17 baseline on the new hard executor, for a like-for-like reference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qlp.baselines import topn_target_weights  # noqa: E402
from qlp.data import build_prep  # noqa: E402
from qlp.execution import ExecutionConfig, run_target_weight_engine  # noqa: E402

SCORE_CACHE = Path(r'workspace/review_workspace/quant_factor_ashare_review/linkage_round2/score_F17.pkl')


def perf(daily: pd.DataFrame) -> dict:
    """Headline net-of-cost statistics from a daily ledger."""
    net = daily['net'].to_numpy(dtype=float)
    net = net[np.isfinite(net)]
    if net.size == 0:
        return {}
    nav = np.cumprod(1.0 + net)
    years = net.size / 252.0
    cagr = float(nav[-1] ** (1.0 / years) - 1.0) if years > 0 else float('nan')
    vol = float(np.std(net, ddof=1) * np.sqrt(252))
    sharpe = float(np.mean(net) / np.std(net, ddof=1) * np.sqrt(252)) if np.std(net, ddof=1) > 0 else float('nan')
    peak = np.maximum.accumulate(nav)
    dd = float((nav / peak - 1.0).min())
    return dict(days=int(net.size), cagr=cagr, vol=vol, sharpe=sharpe, max_drawdown=dd,
                final_nav=float(nav[-1]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2013-01-01')
    ap.add_argument('--end', default=None)
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--buffer', type=float, default=0.6)
    ap.add_argument('--cap', type=float, default=0.03)
    ap.add_argument('--rebalance', type=int, default=1)
    ap.add_argument('--partial', type=float, default=1.0)
    ap.add_argument('--band', type=float, default=0.0)
    ap.add_argument('--capital', type=float, default=10_000_000.0)
    ap.add_argument('--out', default=str(ROOT / 'artifacts' / 'baseline_f17.json'))
    # Default to the audit's documented diagnostic mode: prolonged missing quotes
    # are an unresolved data gap in this project. The resulting stale weight is
    # reported, and the run is labelled a diagnostic, never a tradable result.
    ap.add_argument('--strict', action='store_true',
                    help='reject held assets with no fresh valuation instead of carrying them')
    args = ap.parse_args(argv)

    prep = build_prep()
    close = prep['close']
    open_ = prep['open']
    raw = prep['raw_close']
    adv = prep['amount_lag']
    base_universe = prep['universe']
    limit_locked = prep['limit_locked']

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else close.index.max()
    keep = (close.index >= start) & (close.index <= end)
    idx = close.index[keep]
    cols = close.columns

    close = close.loc[idx]
    open_ = open_.loc[idx]
    adv = adv.loc[idx]
    eligible = base_universe.loc[idx]
    # Tradable at the next open: in the universe, has a quote, and not limit-locked.
    tradable = eligible & ~limit_locked.loc[idx] & np.isfinite(open_) & (open_ > 0)
    locked = pd.DataFrame(False, index=idx, columns=cols)

    if not SCORE_CACHE.exists():
        raise FileNotFoundError(SCORE_CACHE)
    score = pd.read_pickle(SCORE_CACHE).reindex(index=idx, columns=cols)

    targets = topn_target_weights(score, eligible, n=args.n, buffer=args.buffer,
                                  cap=args.cap)
    if args.rebalance > 1:
        # Decisions only on the grid: carry the last decision's target forward so
        # off-grid days have nothing to trade rather than liquidating the book.
        grid = np.zeros((len(idx), len(cols)), dtype=bool)
        grid[:: args.rebalance, :] = True
        grid_df = pd.DataFrame(grid, index=idx, columns=cols)
        targets = targets.where(grid_df).ffill().fillna(0.0)

    cfg = ExecutionConfig(capital=args.capital, cap=args.cap,
                          partial_adjustment=args.partial,
                          no_trade_weight=args.band, price_guard=0.0,
                          allow_stale=not args.strict)
    res = run_target_weight_engine(open_, close, targets, eligible, tradable,
                                   locked, adv, cfg)
    daily = res['daily']
    artifacts = Path(args.out).parent
    artifacts.mkdir(parents=True, exist_ok=True)
    daily.to_csv(artifacts / 'baseline_f17_daily.csv')

    turn = float(daily['turnover_currency'].sum()) / float(daily['nav'].mean())
    out = dict(
        config=vars(args),
        period=dict(start=str(idx.min().date()), end=str(idx.max().date()), days=int(len(idx))),
        performance=perf(daily),
        annual_two_way_turnover=turn / (len(idx) / 252.0),
        mean_cash_weight=float(daily['cash_weight'].mean()),
        mean_holdings=float(daily['n_holdings'].mean()),
        total_cost=float(daily['cost_currency'].sum()),
        total_cost_bp_of_capital=float(daily['cost_currency'].sum() / args.capital * 1e4),
        blocked_buy_days=int((daily['blocked_buy'] > 0).sum()),
        blocked_sell_days=int((daily['blocked_sell'] > 0).sum()),
        stale_weight_mean=float(daily['stale_weight'].mean()),
        stale_weight_max=float(daily['stale_weight'].max()),
        stale_weight_days=int((daily['stale_weight'] > 0).sum()),
        valuation_mode='diagnostic_carry_last_price' if not args.strict else 'strict',
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


if __name__ == '__main__':
    main()
