# Research project author: Gilmour Jia
"""Hard event-driven execution engine for direct-weight policies.

A policy emits *target* weights (stocks plus an explicit cash slot) for every
decision date. This engine turns those targets into an auditable account:

  * partial adjustment toward the target (never full rebalancing by default)
  * per-stock no-trade band measured in NAV notional
  * board-lot rounding, commission floor, slippage, historical stamp duty
  * lagged ADV capacity cap, applied as a proxy only
  * blocked orders are cancelled, not simulated as filled
  * unconfirmed sale proceeds never fund the same-open buys
  * held assets with no fresh valuation are frozen, never repriced from a stale open

The engine is deliberately independent of any model: it consumes only price
matrices, eligibility/tradability flags and target weights, so it can be tested
against synthetic data and reused for every candidate policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .weights import blend_targets, feasible_bounds, no_trade_mask, project_capped_simplex

__all__ = ["ExecutionConfig", "stamp_duty_rate", "run_target_weight_engine"]


@dataclass(frozen=True)
class ExecutionConfig:
    """Account, market and cost conventions for one backtest run."""

    capital: float = 10_000_000.0
    cap: float = 0.03
    partial_adjustment: float = 0.5
    no_trade_weight: float = 0.0025
    # Relative no-trade band: skip a stock trade when |target-current| is below
    # this fraction of the *target* weight.  The absolute band above is measured
    # in NAV notional; a 25bp notional band is larger than a 0.11% position and
    # freezes a 900-name book, so the relative band is the usable control.
    no_trade_relative: float = 0.0
    board_lot: int = 100
    commission_bp: float = 2.5
    min_commission: float = 5.0
    slippage_bp: float = 10.0
    participation: float = 0.01
    price_guard: float = 0.02
    stale_limit: int = 60
    allow_stale: bool = False
    cash_annual_return: float = 0.0
    enforce_board_lot: bool = True
    # A cash weight below this is treated as rounding dust, not a real decision.
    cash_dust: float = 1e-9

    def validate(self) -> None:
        if self.capital <= 0:
            raise ValueError("capital must be positive")
        if not 0 < self.cap <= 1:
            raise ValueError("cap must be in (0, 1]")
        if not 0 < self.partial_adjustment <= 1:
            raise ValueError("partial_adjustment must be in (0, 1]")
        if self.no_trade_weight < 0 or self.participation <= 0 or self.participation > 1:
            raise ValueError("invalid band or participation")
        if self.no_trade_relative < 0 or self.no_trade_relative > 1:
            raise ValueError("no_trade_relative must be in [0, 1]")
        if self.board_lot < 1 or self.price_guard < 0:
            raise ValueError("invalid board lot or price guard")
        if self.stale_limit < 1:
            raise ValueError("stale_limit must be >= 1")


def stamp_duty_rate(date) -> float:
    """A-share sell-side stamp duty, historical schedule for the 2010+ sample.

    On 2023-08-28 the rate was halved from 0.1% to 0.05%. The exact in-force
    date follows the Ministry of Finance announcement of 2023-08-27.
    """
    return 0.001 if pd.Timestamp(date) < pd.Timestamp('2023-08-28') else 0.0005


def _validate_axes(open_prices, close_prices, targets, eligible, locked):
    ref = open_prices.index, open_prices.columns
    for name, obj in (('close_prices', close_prices), ('targets', targets),
                      ('eligible', eligible), ('locked', locked)):
        if obj.index.equals(ref[0]) is False or obj.columns.equals(ref[1]) is False:
            raise ValueError(f"{name} axes differ from open_prices")


def run_target_weight_engine(open_prices, close_prices, targets, eligible,
                             tradable, locked, adv_lag, cfg=ExecutionConfig(),
                             cash_return_daily=0.0):
    """Execute target weights against real frictions and return a ledger.

    Parameters
    ----------
    open_prices, close_prices : DataFrame
        Consistent (adjusted) prices on a common date x security axis. Missing
        values mark a suspended or unavailable security; they are never filled.
    targets : DataFrame
        Policy target weights, one column per security. The *cash* target is
        passed separately via ``cash_target`` (see below) if the policy models
        cash explicitly; by default residual weight is cash.
    eligible : DataFrame[bool]
        May be held / selected at all on that date (point-in-time universe).
    tradable : DataFrame[bool]
        Actually tradeable at the next open (not suspended, not limit-locked).
    locked : DataFrame[bool]
        Position cannot be reduced for regulatory or settlement reasons.
    adv_lag : DataFrame[float]
        Trailing turnover in currency, lagged so no same-day information is used.
    """
    cfg.validate()
    _validate_axes(open_prices, close_prices, targets, eligible, locked)

    op = open_prices.to_numpy(dtype=float)
    cl = close_prices.to_numpy(dtype=float)
    tg = targets.to_numpy(dtype=float)
    el = eligible.to_numpy(dtype=bool)
    tv = tradable.to_numpy(dtype=bool)
    lk = locked.to_numpy(dtype=bool)
    adv = adv_lag.to_numpy(dtype=float)

    dates = open_prices.index
    n_dates, n_sec = op.shape
    if tg.shape != op.shape:
        raise ValueError("targets must match the price matrix shape")

    shares = np.zeros(n_sec, dtype=float)
    mark = np.full(n_sec, np.nan, dtype=float)
    stale = np.zeros(n_sec, dtype=int)
    cash = float(cfg.capital)
    last_nav = cash
    commission = cfg.commission_bp / 1e4
    slippage = cfg.slippage_bp / 1e4

    daily_rows = []
    event_rows = []
    holding_rows = []

    for t, date in enumerate(dates):
        valid_open = np.isfinite(op[t]) & (op[t] > 0)
        valid_close = np.isfinite(cl[t]) & (cl[t] > 0)

        # Mark to the open when available; otherwise carry the previous mark and
        # record staleness (never fabricate an open from a close).
        open_mark = np.where(valid_open, op[t], mark)
        holdings_value = shares * open_mark
        nav_open = cash + float(np.nansum(np.where(np.isfinite(holdings_value), holdings_value, 0.0)))

        cost = turnover = 0.0
        blocked_buy = blocked_sell = 0
        filled_buy = filled_sell = 0.0

        if t > 0 and np.isfinite(nav_open) and nav_open > 0:
            w_now = np.where(np.isfinite(holdings_value), holdings_value, 0.0) / nav_open
            cash_w = cash / nav_open

            # The policy forms targets from information available at the close of
            # t-1 and trades them at the open of t. A target may only be held when
            # the security is in the point-in-time universe, unless it is locked.
            desired = np.where(el[t - 1] | lk[t], tg[t - 1], 0.0)
            desired = np.where(np.isfinite(desired), desired, 0.0)
            # Negative policy outputs are not short positions in a long-only
            # mandate; the projection would clip them anyway, so drop them here
            # to keep the funding check below meaningful.
            desired = np.clip(desired, 0.0, None)
            # Everything already invested plus all settled cash is fundable, so
            # the stock budget is w_now + cash_w, not just cash. Weights sum to 1.
            stock_budget = float(np.clip(w_now.sum() + cash_w, 0.0, 1.0))
            if desired.sum() > stock_budget > 0:
                desired = desired * (stock_budget / desired.sum())
            elif stock_budget <= 0:
                desired = np.zeros(n_sec)

            allow_trade = tv[t] & ~lk[t]
            current_full = np.append(w_now, cash_w)
            desired_full = np.append(desired, max(0.0, 1.0 - desired.sum()))
            tradeable_full = np.append(allow_trade, True)
            locked_full = np.append(np.where(lk[t], w_now, 0.0), 0.0)

            lower, upper = feasible_bounds(tradeable_full, locked_full, cfg.cap, False)
            feasible, ok = project_capped_simplex(desired_full, lower, upper, total=1.0)
            if not ok:
                # Bounds cannot admit full investment (locked positions already
                # consume more than NAV allows); hold what we have.
                feasible = current_full.copy()

            blended = blend_targets(current_full, feasible, cfg.partial_adjustment)
            # The band is measured in notional and applies to stocks only; cash is
            # whatever weight the projection leaves with, never a traded position.
            if cfg.no_trade_relative > 0:
                # Compare the gap with the target weight, not with NAV.  A 1%
                # target moving to 1.1% is a 10% relative move and should trade;
                # a 20% target moving to 20.1% is 0.5% and should not.
                scale = np.maximum(np.abs(blended), 1e-9)
                skip = np.abs(blended - current_full) <= cfg.no_trade_relative * scale
            else:
                skip = no_trade_mask(current_full, blended, nav_open,
                                     cfg.no_trade_weight * nav_open)
            skip[-1] = False
            # A name the policy has dropped must be *exited*, not nibbled. Without
            # this, partial adjustment plus a band leaves a geometric residue that
            # never clears, so the account slowly fills with dust positions that
            # consume weight and pay costs without expressing any view.
            exited = np.minimum(desired_full, current_full) <= 0
            skip &= ~(exited & tradeable_full)
            target_w = np.where(skip, current_full, blended)
            # Cash is the residual of the *stock* targets we chose to hold, not a
            # target of its own.  Carrying blended cash through the projection
            # would hand any weight freed by a skipped stock trade to whichever
            # name has the most slack, silently defeating the band.
            target_w[-1] = max(0.0, 1.0 - target_w[:-1].sum())
            # Re-project after suppressing small trades so weights still sum to 1.
            target_w, _ = project_capped_simplex(
                target_w, lower, upper, total=1.0)

            target_value = target_w[:-1] * nav_open
            delta = target_value - holdings_value
            delta = np.where(np.isfinite(delta), delta, 0.0)

            capacity = np.where(np.isfinite(adv[t]) & (adv[t] > 0),
                                adv[t] * cfg.participation, 0.0)
            if cfg.price_guard > 0 and t > 0:
                # Reference value must be the raw formation-day close, so a
                # favourable gap is not used to reject the order.
                ref_close = np.where(np.isfinite(cl[t - 1]) & (cl[t - 1] > 0), cl[t - 1], np.nan)
                gap = np.divide(op[t], ref_close, out=np.ones(n_sec), where=np.isfinite(ref_close))
                offside = (np.abs(gap - 1.0) > cfg.price_guard) & np.isfinite(gap)
            else:
                offside = np.zeros(n_sec, dtype=bool)

            # --- sells first: only confirmed proceeds may fund buys ---
            px_open = np.where(valid_open, op[t], np.nan)
            sell_mask = (delta < 0) & allow_trade & ~offside
            sell_want = np.where(sell_mask, np.minimum(-delta, capacity), 0.0)
            sell_want = np.minimum(sell_want, holdings_value)
            held_shares = shares.copy()
            if cfg.enforce_board_lot:
                lots = np.floor(np.divide(sell_want, px_open * cfg.board_lot,
                                          out=np.zeros(n_sec),
                                          where=np.isfinite(px_open) & (px_open > 0)))
                sell_value = lots * cfg.board_lot * px_open
                sell_value = np.where(np.isfinite(sell_value), sell_value, 0.0)
                # A name the policy has dropped is *exited*: sell the whole
                # remaining position. Flooring a liquidation to whole lots would
                # strand an odd-lot residue that no later order can ever clear,
                # and the account would silently fill with untradeable dust.
                full_exit = (delta < 0) & (desired_full[:-1] <= 0) & (held_shares > 0)
                exact = held_shares * px_open
                sell_value = np.where(full_exit & np.isfinite(exact),
                                      np.maximum(sell_value, exact), sell_value)
            else:
                sell_value = np.where(np.isfinite(px_open), sell_want, 0.0)
            sell_value = np.minimum(sell_value, held_shares * px_open)
            sell_value = np.where(np.isfinite(sell_value) & (sell_value > 0), sell_value, 0.0)

            if sell_value.any():
                stamp = stamp_duty_rate(date)
                sell_fee = sell_value.sum() * (commission + slippage + stamp)
                # Only the names that actually sold may change. Dividing the whole
                # vector by a missing open price would turn uninvolved holdings
                # into NaN and then silently delete them.
                filled = sell_value > 0
                px_filled = np.where(filled, np.where(valid_open, op[t], np.nan), 1.0)
                delta_shares = np.where(filled, sell_value / px_filled, 0.0)
                held_shares = held_shares - np.nan_to_num(delta_shares, nan=0.0)
                cash += sell_value.sum() - sell_fee
                cost += float(sell_fee)
                turnover += float(sell_value.sum())
                filled_sell = float(sell_value.sum())

            blocked_sell = int(np.count_nonzero((delta < 0) & ~allow_trade))
            blocked_buy = int(np.count_nonzero((delta > 0) & ~allow_trade)) + int(np.count_nonzero((delta > 0) & offside))

            # --- buys, funded only by cash actually settled ---
            buy_mask = (delta > 0) & allow_trade & ~offside
            buy_want = np.where(buy_mask, np.minimum(delta, capacity), 0.0)
            if buy_want.any():
                fee_rate = commission + slippage
                px = np.where(valid_open, op[t], np.nan)
                lot_cost = px * cfg.board_lot
                # Round down to whole board lots before checking affordability.
                if cfg.enforce_board_lot:
                    lots = np.floor(np.divide(
                        buy_want, lot_cost,
                        out=np.zeros(n_sec), where=np.isfinite(lot_cost) & (lot_cost > 0)))
                    buy_value = lots * lot_cost
                else:
                    buy_value = np.where(np.isfinite(px) & (px > 0), buy_want, 0.0)
                buy_value = np.where(np.isfinite(buy_value) & (buy_value > 0), buy_value, 0.0)

                def _total_outlay(value):
                    """Cash needed for a basket: notional plus fees, floor applied."""
                    n_filled = int(np.count_nonzero(value > 0))
                    if n_filled == 0:
                        return 0.0, 0.0
                    fee = max(value.sum() * fee_rate, cfg.min_commission * n_filled)
                    return value.sum() + fee, fee

                outlay, buy_fee = _total_outlay(buy_value)
                # Scale back in whole lots until the basket is affordable.
                guard = 0
                while outlay > cash + 1e-6 and buy_value.sum() > 0 and guard < 64:
                    ratio = cash / outlay
                    if cfg.enforce_board_lot:
                        lots = np.floor(np.divide(
                            buy_value * ratio, lot_cost,
                            out=np.zeros(n_sec), where=np.isfinite(lot_cost) & (lot_cost > 0)))
                        buy_value = lots * lot_cost
                    else:
                        buy_value = buy_value * ratio
                    outlay, buy_fee = _total_outlay(buy_value)
                    guard += 1
                if buy_value.sum() > 0 and outlay <= cash + 1e-6:
                    filled = buy_value > 0
                    px_filled = np.where(filled, np.where(valid_open, op[t], np.nan), 1.0)
                    delta_shares = np.where(filled, buy_value / px_filled, 0.0)
                    held_shares = held_shares + np.nan_to_num(delta_shares, nan=0.0)
                    cash -= outlay
                    cost += float(buy_fee)
                    turnover += float(buy_value.sum())
                    filled_buy = float(buy_value.sum())

            # Committed state: trades changed held_shares, so it becomes the
            # account position that rolls into the next day.
            shares = held_shares

        # --- mark to close and roll the account forward ---
        shares = np.nan_to_num(shares, nan=0.0, posinf=0.0, neginf=0.0)
        # A missing close means "no new valuation", not "worthless": carry the
        # last known mark and count the staleness, so a suspended holding stays
        # on the book and is reported instead of silently vanishing.
        end_mark = np.where(valid_close, cl[t], mark)
        end_mark = np.where(np.isfinite(end_mark), end_mark, open_mark)
        end_value = shares * end_mark
        end_value = np.where(np.isfinite(end_value), end_value, 0.0)
        mark = end_mark
        stale = np.where(valid_close, 0, stale + 1)
        stale_weight = float(end_value[stale > cfg.stale_limit].sum())
        if stale_weight > 1e-4 and not cfg.allow_stale:
            raise ValueError(
                f"{date}: held asset lacks a fresh valuation for more than "
                f"{cfg.stale_limit} days; delisting/valuation data required"
            )

        cash *= (1.0 + cash_return_daily)
        nav = cash + float(end_value.sum())
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError(f"{date}: invalid NAV")

        daily_rows.append(dict(
            net=nav / last_nav - 1.0,
            nav=nav,
            cash_weight=cash / nav,
            cost=cost / last_nav,
            cost_currency=cost,
            turnover=turnover / nav,
            turnover_currency=turnover,
            blocked_buy=blocked_buy,
            blocked_sell=blocked_sell,
            filled_buy=filled_buy,
            filled_sell=filled_sell,
            n_holdings=int(np.count_nonzero(end_value > nav * 1e-10)),
            stale_weight=stale_weight / nav,
            total_weight=float(end_value.sum()) / nav,
        ))
        holding_rows.append(end_value / nav)
        event_rows.append(dict(
            date=str(pd.Timestamp(date).date()),
            turnover=turnover,
            cost=cost,
            blocked_buy=blocked_buy,
            blocked_sell=blocked_sell,
        ))
        last_nav = nav

    daily = pd.DataFrame(daily_rows, index=dates)
    events = pd.DataFrame(event_rows).set_index("date") if event_rows else pd.DataFrame()
    holdings = pd.DataFrame(holding_rows, index=dates, columns=open_prices.columns)
    return dict(daily=daily, events=events, holdings=holdings)
