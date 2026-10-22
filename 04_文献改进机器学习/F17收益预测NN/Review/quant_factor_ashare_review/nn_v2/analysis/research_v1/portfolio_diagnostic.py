# Research project author: Gilmour Jia
"""Unlevered, frictionless adjusted-price diagnostic, NOT an execution backtest.

Fractional adjusted-price units are not real A-share holdings. Missing orders
remain unfilled. Last-observed quote valuation is explicit and audited; unresolved
terminal events or >60-session gaps abort rather than invent terminal proceeds.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import runtime
import argparse
import json
import numpy as np
import pandas as pd
from metrics import metrics
from portfolio import readout
from spec import SPEC, file_hash, code_hash, write_json
from research import research_acceptance


class PriceIndex:
    def __init__(self, capital=10_000_000.):
        self.cash = capital
        self.last_nav = capital
        self.units, self.marks, self.stale = {}, {}, {}
        self.daily, self.orders, self.gaps = [], [], []

    def step(self, date, op, close, suspended, terminal, target=None):
        universe = set(self.units) | (set(target) if target is not None else set())
        can_trade, opens = {}, {}
        for s in universe:
            if self.units.get(s,0) > 0 and terminal.get(s,False):
                raise ValueError(f"Unresolved held terminal event: {date} {s}")
            p = op.get(s,np.nan)
            observed = np.isfinite(p) and p > 0
            can_trade[s] = observed and not suspended.get(s,False) and not terminal.get(s,False)
            if observed:
                opens[s] = float(p)
            elif self.units.get(s,0) > 0:
                opens[s] = self.marks[s]
        open_nav = self.cash + sum(q*opens[s] for s,q in self.units.items())
        turnover, blocked = 0., 0
        if target is not None:
            if min(target.values(),default=0) < 0 or sum(target.values()) > 1+1e-10:
                raise ValueError("Invalid unlevered target")
            for s,q in list(self.units.items()):
                if not can_trade[s]:
                    if s not in target:
                        blocked += 1
                    continue
                desired = open_nav*target.get(s,0.)/opens[s]
                sell = max(q-desired,0.)
                if sell:
                    self.cash += sell*opens[s]
                    self.units[s] -= sell
                    turnover += sell*opens[s]/open_nav
                    self.orders.append({"date":str(date.date()),"symbol":s,"side":"sell",
                                        "adjusted_units":sell,"adjusted_price":opens[s]})
                    if self.units[s] < 1e-8:
                        del self.units[s]
            wants = {}
            for s,w in target.items():
                if not can_trade[s]:
                    blocked += 1
                    continue
                wants[s] = max(open_nav*w-self.units.get(s,0.)*opens[s],0.)
            need = sum(wants.values())
            scale = min(1.,max(self.cash,0.)/need) if need else 0.
            for s,notional in wants.items():
                buy = notional*scale
                if buy > 1e-8:
                    if s not in self.units:
                        self.stale[s], self.marks[s] = 0, opens[s]
                    self.units[s] = self.units.get(s,0.)+buy/opens[s]
                    self.cash -= buy
                    turnover += buy/open_nav
                    self.orders.append({"date":str(date.date()),"symbol":s,"side":"buy",
                                        "adjusted_units":buy/opens[s],"adjusted_price":opens[s]})
            after = self.cash + sum(q*opens[s] for s,q in self.units.items())
            if abs(after-open_nav) > max(1e-5,open_nav*1e-10) or self.cash < -1e-5:
                raise ValueError("Cash/position conservation failed")
            self.cash = max(0.,self.cash)
        stale_value, unknown_value, max_stale = 0., 0., 0
        for s,q in self.units.items():
            c = close.get(s,np.nan)
            p = op.get(s,np.nan)
            if np.isfinite(c) and c > 0:
                self.marks[s], self.stale[s] = float(c), 0
            else:
                self.stale[s] = self.stale.get(s,0)+1
                # The opening quote, if observed, is newer than yesterday's mark.
                if np.isfinite(p) and p > 0:
                    self.marks[s] = float(p)
                if s not in self.marks:
                    raise ValueError("Position has no observed quote for valuation")
                if self.stale[s] > 60:
                    raise ValueError(f"Unresolved >60-session held price gap: {date} {s}")
                value = q*self.marks[s]
                stale_value += value
                known = bool(suspended.get(s,False))
                if not known:
                    unknown_value += value
                self.gaps.append({"date":str(date.date()),"symbol":s,"adjusted_units":q,
                                  "mark":self.marks[s],"value":value,"known_suspension":known,
                                  "consecutive_missing_closes":self.stale[s]})
                max_stale = max(max_stale,self.stale[s])
        nav = self.cash + sum(q*self.marks[s] for s,q in self.units.items())
        if nav <= 0 or not np.isfinite(nav):
            raise ValueError("Invalid gross diagnostic NAV")
        self.daily.append({"date":str(date.date()),"nav":nav,"gross_return":nav/self.last_nav-1,
                           "cash_weight":self.cash/nav,"holdings":len(self.units),
                           "turnover":turnover,"blocked_orders":blocked,
                           "missing_close_weight":stale_value/nav,
                           "unknown_missing_close_weight":unknown_value/nav,
                           "max_consecutive_missing_closes":max_stale})
        self.last_nav = nav


def sanity_checks():
    # Price doubling before the next-open purchase is not earned by the signal.
    b = PriceIndex(1000.)
    b.step(pd.Timestamp("2021-01-04"),{"x":100.},{"x":100.},{},{})
    b.step(pd.Timestamp("2021-01-05"),{"x":200.},{"x":200.},{},{},{"x":1.})
    b.step(pd.Timestamp("2021-01-06"),{"x":200.},{"x":220.},{},{})
    assert np.allclose([r["gross_return"] for r in b.daily],[0.,0.,.1])
    # An unfilled suspended sale remains exposed and realizes its later drop.
    b = PriceIndex(1000.)
    b.step(pd.Timestamp("2021-01-04"),{"x":100.},{"x":100.},{},{},{"x":1.})
    b.step(pd.Timestamp("2021-01-05"),{}, {}, {"x":True},{},{})
    assert b.units["x"] == 10 and b.cash == 0
    b.step(pd.Timestamp("2021-01-06"),{"x":80.},{"x":80.},{},{},{})
    assert not b.units and b.cash == 800 and np.isclose(b.daily[-1]["gross_return"],-.2)
    assert np.isclose(metrics(np.array([.1,-.2,.25]))["max_drawdown"],-.2)


def summarize(daily):
    result = metrics(daily.gross_return)
    result["Calmar"] = result["CAGR"]/abs(result["max_drawdown"]) if result["max_drawdown"] < 0 else None
    result.update(mean_cash_weight=float(daily.cash_weight.mean()),
                  mean_holdings=float(daily.holdings.mean()),
                  annual_two_way_turnover=float(daily.turnover.mean()*252),
                  blocked_orders=int(daily.blocked_orders.sum()),
                  days_with_missing_held_close=int(daily.missing_close_weight.gt(0).sum()),
                  maximum_missing_close_weight=float(daily.missing_close_weight.max()),
                  maximum_unknown_close_weight=float(daily.unknown_missing_close_weight.max()),
                  max_consecutive_missing_closes=int(daily.max_consecutive_missing_closes.max()))
    return result


def run(output):
    sanity_checks()
    runs = ROOT/"runs/research_v1"
    package = ROOT/"derived/research_training_v1"
    normalized = ROOT/"derived/research_normalized_v3"
    manifest = json.loads((runs/"run_manifest.json").read_text(encoding="utf-8"))
    if manifest["code_hash"] != code_hash() or manifest["data_manifest"] != research_acceptance(package):
        raise ValueError("Frozen code or training inputs changed")
    if output.exists():
        raise ValueError("A new diagnostic version directory is required")
    predictions = []
    for y in range(2021,2027):
        p = runs/f"predictions_{y}.csv"
        if file_hash(p) != manifest["files"][p.name]:
            raise ValueError("Changed predictions")
        predictions.append(pd.read_csv(p))
    pred = pd.concat(predictions,ignore_index=True)
    for c in ("formation_time","entry_time"):
        pred[c] = pd.to_datetime(pred[c])
    if pred.duplicated(["formation_time","symbol"]).any() or not (pred.formation_time < pred.entry_time).all():
        raise ValueError("Invalid prediction keys or signal timing")
    # Do NOT filter by label_reason or future label availability when selecting.
    strategies = {"NN_softmax":("NN","softmax"),"F17":("F17","equal"),
                  "Ridge":("Ridge","softmax"),"NN_equal":("NN","equal")}
    targets, all_selected = {}, set()
    for name,(col,mode) in strategies.items():
        previous, targets[name] = [], {}
        for entry,g in pred.groupby("entry_time",sort=True):
            target,previous = readout(g.set_index("symbol")[col],previous,mode)
            targets[name][entry] = None if target is None else target.to_dict()
            if target is not None:
                all_selected.update(target.index)
    normalized_manifest = json.loads((package/"source_manifest.json").read_text(encoding="utf-8"))
    fields = ["date","symbol","adjusted_open","adjusted_close","suspended","terminal"]
    frames, source_hashes = [], {}
    for year in range(2021,2027):
        p = normalized/f"{year}.parquet"
        sha = file_hash(p)
        if sha != normalized_manifest["normalized_files"][p.name]:
            raise ValueError("Changed normalized price inputs")
        source_hashes[str(p)] = sha
        f = pd.read_parquet(p,columns=fields)
        frames.append(f[f.symbol.isin(all_selected)])
    bars = pd.concat(frames,ignore_index=True)
    if bars.duplicated(["date","symbol"]).any():
        raise ValueError("Duplicate observed price keys")
    by_date = {pd.Timestamp(d):g.set_index("symbol") for d,g in bars.groupby("date",sort=True)}
    dates = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(package/"calendar.csv").date))
    dates = dates[dates >= "2021-01-01"]
    books = {name:PriceIndex(SPEC["execution"]["capital"]) for name in strategies}
    changes = normalized_manifest["code_changes"]
    output.mkdir(parents=True)
    for d in dates:
        g = by_date.get(d)
        if g is None:
            op,close,suspended,terminal = {},{},{},{}
        else:
            op,close = g.adjusted_open.to_dict(),g.adjusted_close.to_dict()
            suspended,terminal = g.suspended.to_dict(),g.terminal.to_dict()
        for name,b in books.items():
            for c in changes:
                if d == pd.Timestamp(c["effective"]) and c["old"] in b.units:
                    raise ValueError(f"Cannot splice held adjusted-price code bases: {name} {c}")
            b.step(d,op,close,suspended,terminal,targets[name].get(d))
    reports, yearly = {}, []
    for name,b in books.items():
        df = pd.DataFrame(b.daily)
        df.to_csv(output/f"{name}_daily_gross.csv",index=False)
        pd.DataFrame(b.orders).to_csv(output/f"{name}_orders.csv",index=False)
        pd.DataFrame(b.gaps).to_csv(output/f"{name}_valuation_gaps.csv",index=False)
        reports[name] = summarize(df)
        reports[name]["last_day_missing_close_weight"] = float(df.missing_close_weight.iloc[-1])
        for year,g in df.groupby(pd.to_datetime(df.date).dt.year):
            yearly.append({"model":name,"year":int(year),"partial_year":year==2026,**summarize(g)})
    pd.DataFrame(yearly).to_csv(output/"yearly_metrics.csv",index=False)
    summary = {
        "scope":"UNVERIFIED_FRICTIONLESS_ADJUSTED_PRICE_PORTFOLIO_DIAGNOSTIC",
        "start":str(dates[0].date()),"end":str(dates[-1].date()),
        "costs_included":False,"real_execution_accepted":False,"qualification_decision":None,
        "holding_rules":SPEC["portfolio"],"rf_annual":0.,"periods_per_year":252,
        "timing":"signal after formation close; rebalance at next observed session open; daily closing valuation",
        "units":"fractional adjusted-price units, NOT actual A-share quantities",
        "no_future_label_filter":True,"continuous_book_no_annual_reset":True,
        "valuation":"observed adjusted close; else observed open; else carry last observed quote with gap flags",
        "missing_orders":"remain unfilled; no leverage; cash constrains buys",
        "unsupported_events":"abort on unresolved held terminal event, code-basis change, or >60-session close gap",
        "omitted_constraints":["verified opening limits","actual board lots","liquidity capacity",
                               "commission/stamp/transfer/slippage","verified cash/share entitlement ledger"],
        "data_limitations":"vendor adjustments, monthly ST proxies, static industry and incomplete security coverage; historical research contamination",
        "models":reports,"input_prices":source_hashes,"training_spec_hash":manifest["spec_hash"],
        "script_sha256":file_hash(__file__),"sanity_checks_passed":True}
    write_json(output/"summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"gross_portfolio_v1")
    args = ap.parse_args()
    run(args.output)
