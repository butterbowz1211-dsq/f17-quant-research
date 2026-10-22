# Research project author: Gilmour Jia
"""Raw-share, cash and corporate-entitlement ledger for NN comparisons only."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from spec import SPEC


class FeeSchedule:
    def __init__(self, path=None):
        self.table = pd.read_csv(path or Path(__file__).with_name("fees.csv"))

    def charge(self, date, symbol, side, notional):
        if side not in ("buy", "sell") or not np.isfinite(notional) or notional < 0:
            raise ValueError("Invalid order")
        if notional == 0:
            return dict(commission=0., broker_net=0., handling=0., regulatory=0.,
                        commission_floor=0., stamp=0., transfer=0., slippage=0., total=0.)
        if not symbol.startswith(("sh6", "sz0", "sz3")):
            raise ValueError("Unsupported exchange/security; no default fee fallback")
        d = str(pd.Timestamp(date).date())
        rows = self.table[(self.table.start <= d) & (self.table.end >= d) &
                          (self.table.exchange == symbol[:2].upper()) &
                          (self.table.security_type == "A")]
        if len(rows) != 1:
            raise ValueError(f"No unique fee schedule for {d} {symbol}")
        r = rows.iloc[0]
        broker = notional * 0.0002
        handling, regulatory = notional * r.handling, notional * r.regulatory
        base = broker + handling + regulatory
        commission = max(5.0, base)
        stamp = notional * r.stamp_sell if side == "sell" else 0.
        transfer, slippage = notional * r.transfer, notional * 0.001
        return dict(commission=commission, broker_net=broker, handling=handling,
                    regulatory=regulatory, commission_floor=commission-base,
                    stamp=stamp, transfer=transfer, slippage=slippage,
                    total=commission+stamp+transfer+slippage)


def quantity(symbol, desired, side, held=0, full_exit=False):
    q = max(0, int(np.floor(desired + 1e-8)))
    star = symbol.startswith("sh688")
    q = min(q, 100000 if star else 1000000)
    if side == "sell" and full_exit and q >= held:
        return int(held)
    if star:
        return q if q >= 200 else 0
    return q // 100 * 100


@dataclass
class Book:
    cash: float = 10000000.0
    shares: dict = field(default_factory=dict)
    locked: dict = field(default_factory=dict)
    marks: dict = field(default_factory=dict)
    stale: dict = field(default_factory=dict)
    receivable: float = 0.
    entitlements: dict = field(default_factory=dict)
    orders: list = field(default_factory=list)
    daily: list = field(default_factory=list)
    actions_log: list = field(default_factory=list)
    positions: list = field(default_factory=list)
    last_nav: float = 10000000.0
    last_date: object = None

    def __post_init__(self):
        self.last_nav = self.cash

    def nav(self):
        return self.cash + self.receivable + sum(q * self.marks[s] for s, q in self.shares.items() if q)

    def corporate_actions(self, date, actions, at_close=False):
        date = pd.Timestamp(date)
        if len({a["id"] for a in actions}) != len(actions):
            raise ValueError("Duplicate corporate-action ID")
        for a in actions:
            if not a.get("verified") or a.get("kind") != "cash_stock_distribution":
                raise ValueError("Unsupported/unverified corporate action; reconcile externally")
            terms = [a["net_cash_per_share"],a["gross_cash_per_share"],a["share_ratio"]]
            if (not np.isfinite(terms).all() or not 0 <= terms[0] <= terms[1] or terms[2] < 1):
                raise ValueError("Invalid cash/share terms")
            if not (pd.Timestamp(a["record_date"]) < pd.Timestamp(a["ex_date"]) <=
                    min(pd.Timestamp(a["pay_date"]), pd.Timestamp(a["stock_date"]))):
                raise ValueError("Invalid corporate-action chronology")
            key, s = a["id"], a["symbol"]
            if at_close:
                if date == pd.Timestamp(a["record_date"]):
                    self.entitlements[key] = self.shares.get(s, 0)
                continue
            entitled = self.entitlements.get(key, 0)
            cash = entitled * a["net_cash_per_share"]
            bonus = entitled * (a["share_ratio"] - 1)
            if bonus < 0 or abs(bonus - round(bonus)) > 1e-7:
                raise ValueError("Fractional/rights/reverse action requires explicit settlement")
            bonus = int(round(bonus))
            if date == pd.Timestamp(a["ex_date"]) and entitled:
                self.shares[s] = self.shares.get(s, 0) + bonus
                self.locked[s] = self.locked.get(s, 0) + bonus
                self.receivable += cash
                # Only a missing-price fallback; actual raw prices overwrite it.
                self.marks[s] = (self.marks[s] - a["gross_cash_per_share"]) / a["share_ratio"]
                self.actions_log.append(dict(date=str(date.date()), id=key, symbol=s,
                                             entitled=entitled, bonus=bonus, receivable=cash))
            if date == pd.Timestamp(a["pay_date"]):
                self.receivable -= cash
                self.cash += cash
            if date == pd.Timestamp(a["stock_date"]):
                self.locked[s] = self.locked.get(s, 0) - bonus

    def step(self, date, bars, target=None, actions=(), fees=None):
        """bars indexed by symbol: raw_open, raw_close, can_buy/sell, adv_lag.

        Boolean trade permissions must represent the open, not a test using the
        same day's future high/low/volume. All capacity is lagged information.
        """
        fees = fees or FeeSchedule()
        if self.last_date is not None and pd.Timestamp(date) <= pd.Timestamp(self.last_date):
            raise ValueError("Ledger dates must be strictly increasing")
        if bars.index.has_duplicates:
            raise ValueError("Duplicate bar symbol")
        self.corporate_actions(date, actions)
        for s, q in self.shares.items():
            if q and s in bars.index and np.isfinite(bars.loc[s, "raw_open"]) and bars.loc[s, "raw_open"] > 0:
                self.marks[s] = float(bars.loc[s, "raw_open"])
        opening_nav = self.nav()
        start_order = len(self.orders)
        blocked_buy = blocked_sell = 0
        capacity_limited = 0
        if target is not None:
            target = pd.Series(target, dtype=float)
            if (not np.isfinite(target).all() or (target < 0).any() or
                    target.sum() > 1+1e-10 or (target > 0.03+1e-10).any()):
                raise ValueError("Invalid target weights")
            names = sorted(set(target.index) | {s for s, q in self.shares.items() if q})
            # Compute a fixed target before sales. Never spend hypothetical sales.
            desired = {s: float(target.get(s, 0)) * opening_nav for s in names}
            buys = {}
            for side in ("sell", "buy"):
                for s in names:
                    held = self.shares.get(s, 0)
                    mark = self.marks.get(s, np.nan)
                    delta = desired[s] - (held * mark if held else 0)
                    if (side == "sell" and delta >= -1e-7) or (side == "buy" and delta <= 1e-7):
                        continue
                    row = bars.loc[s] if s in bars.index else None
                    valid = row is not None and np.isfinite(row.raw_open) and row.raw_open > 0
                    valid = valid and pd.notna(row["can_" + side]) and bool(row["can_" + side])
                    valid = valid and np.isfinite(row.adv_lag) and row.adv_lag > 0
                    if not valid:
                        blocked_buy += side == "buy"
                        blocked_sell += side == "sell"
                        continue
                    px = float(row.raw_open)
                    capacity = float(row.adv_lag) * 0.05
                    wanted = abs(delta)
                    capacity_limited += wanted > capacity + 1e-8
                    budget = min(wanted, capacity)
                    if side == "sell":
                        available = max(0, held - self.locked.get(s, 0))
                        q = quantity(s, min(budget/px, available), "sell", held=held,
                                     full_exit=desired[s] == 0 and available == held)
                        if q:
                            costs = fees.charge(date, s, side, q*px)
                            if self.cash + q*px < costs["total"]:
                                blocked_sell += 1
                                continue
                            self._fill(date, s, side, q, px, costs)
                    else:
                        q = quantity(s, budget/px, "buy")
                        if q:
                            buys[s] = [q, px]
                if side == "buy" and buys:
                    def outlay():
                        return sum(q*px+fees.charge(date, s, "buy", q*px)["total"]
                                   for s, (q, px) in buys.items() if q)
                    # Proportional cash reduction, then deterministic one-step
                    # fallback. Each reduction strictly decreases integer shares.
                    while outlay() > self.cash + 1e-8:
                        ratio = max(0, self.cash) / outlay()
                        changed = False
                        for s, (q, px) in buys.items():
                            new = quantity(s, q*ratio, "buy")
                            changed |= new < q
                            buys[s][0] = new
                        if not changed:
                            s = max((s for s in buys if buys[s][0]), key=lambda s: (buys[s][0]*buys[s][1], s))
                            buys[s][0] = quantity(s, buys[s][0] - (1 if s.startswith("sh688") else 100), "buy")
                    for s, (q, px) in buys.items():
                        if q:
                            self._fill(date, s, "buy", q, px, fees.charge(date, s, "buy", q*px))
        for s, q in self.shares.items():
            if not q:
                continue
            cl = bars.loc[s, "raw_close"] if s in bars.index else np.nan
            if np.isfinite(cl) and cl > 0:
                self.marks[s] = float(cl)
                self.stale[s] = 0
            else:
                self.stale[s] = self.stale.get(s, 0) + 1
                if self.stale[s] > 60:
                    raise ValueError(f"Stale valuation exceeds 60 sessions: {s}")
        self.corporate_actions(date, actions, at_close=True)
        nav = self.nav()
        if self.cash < -1e-7 or self.receivable < -1e-7 or nav <= 0:
            raise ValueError("Cash/entitlement/NAV invariant broken")
        orders = self.orders[start_order:]
        stale_value = sum(self.shares[s]*self.marks[s] for s in self.shares if self.stale.get(s, 0))
        daily = dict(date=str(pd.Timestamp(date).date()), nav=nav, net=nav/self.last_nav-1,
                     cash=self.cash, cash_weight=self.cash/nav, receivable=self.receivable,
                     holdings=sum(q > 0 for q in self.shares.values()),
                     turnover=sum(o["notional"] for o in orders)/opening_nav,
                     blocked_buy=blocked_buy, blocked_sell=blocked_sell,
                     capacity_limited=capacity_limited, stale_weight=stale_value/nav,
                     target_cash=None if target is None else float(1-target.sum()),
                     max_actual_weight=max([q*self.marks[s]/nav for s,q in self.shares.items() if q] or [0]))
        for k in ("commission", "broker_net", "handling", "regulatory", "commission_floor",
                  "stamp", "transfer", "slippage", "total"):
            daily[k] = sum(o[k] for o in orders)
        self.daily.append(daily)
        for s,q in self.shares.items():
            if q:
                self.positions.append(dict(date=str(pd.Timestamp(date).date()),symbol=s,shares=q,
                                      locked=self.locked.get(s,0),mark=self.marks[s],
                                      value=q*self.marks[s],stale_sessions=self.stale.get(s,0)))
        self.last_nav = nav
        self.last_date = pd.Timestamp(date)
        return daily

    def _fill(self, date, symbol, side, q, price, costs):
        value = q * price
        self.shares[symbol] = self.shares.get(symbol, 0) + (q if side == "buy" else -q)
        self.cash += -value-costs["total"] if side == "buy" else value-costs["total"]
        self.marks[symbol] = price
        self.orders.append(dict(date=str(pd.Timestamp(date).date()), symbol=symbol, side=side,
                                shares=q, price=price, notional=value, **costs))
