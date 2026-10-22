# Research project author: Gilmour Jia
"""One continuous ledger for each predeclared readout, after all training."""
import runtime
import argparse
import json
from pathlib import Path
import pandas as pd
from execution import Book
from portfolio import readout
from metrics import metrics, paired_interval, gates, rank_ic
from train import acceptance
from spec import write_json, code_hash, file_hash


def evaluate(package, runs, output):
    accepted = acceptance(package)
    manifest = json.loads((runs/"run_manifest.json").read_text(encoding="utf-8"))
    if manifest["code_hash"] != code_hash() or manifest["data_manifest"] != accepted:
        raise ValueError("Code/data no longer match the completed training run")
    for name,sha in manifest["files"].items():
        if file_hash(runs/name) != sha:
            raise ValueError(f"Changed training artifact: {name}")
    if output.exists():
        raise ValueError("Use a new evaluation directory")
    output.mkdir(parents=True)
    pred = pd.concat([pd.read_csv(runs/f"predictions_{y}.csv") for y in range(2021,2027)],ignore_index=True)
    for c in ("formation_time","entry_time","label_end_time"):
        pred[c] = pd.to_datetime(pred[c])
    if pred.duplicated(["formation_time","symbol"]).any():
        raise ValueError("Duplicate predictions")
    market = pd.read_parquet(package/"market.parquet")
    market["date"] = pd.to_datetime(market.date)
    actions = json.loads((package/"actions.json").read_text(encoding="utf-8"))
    books = {k:Book() for k in ("NN_softmax","NN_equal","F17","Ridge")}
    previous = {k:[] for k in books}
    signal_at = {pd.Timestamp(d): g for d,g in pred.groupby("entry_time")}
    reports = {}
    # Include cash days from the start of 2021. The global grid is never reset.
    for date,bars in market[market.date >= "2021-01-01"].groupby("date",sort=True):
        bars = bars.set_index("symbol")
        p = signal_at.get(date)
        for name,book in books.items():
            target = None
            if p is not None:
                source = "NN" if name.startswith("NN") else name
                mode = "equal" if name in ("F17","NN_equal") else "softmax"
                target, previous[name] = readout(p.set_index("symbol")[source],previous[name],mode)
            book.step(date,bars,target,actions)
    yearly = []
    for name,book in books.items():
        daily = pd.DataFrame(book.daily)
        daily.to_csv(output/f"{name}_daily.csv",index=False)
        pd.DataFrame(book.orders).to_csv(output/f"{name}_orders.csv",index=False)
        pd.DataFrame(book.positions).to_csv(output/f"{name}_positions.csv",index=False)
        pd.DataFrame(book.actions_log).to_csv(output/f"{name}_corporate_actions.csv",index=False)
        reports[name] = metrics(daily.net)
        reports[name]["mean_cash_weight"] = float(daily.cash_weight.mean())
        reports[name]["mean_holdings"] = float(daily.holdings.mean())
        reports[name]["annual_turnover"] = float(daily.turnover.mean()*252)
        reports[name]["costs"] = {c:float(daily[c].sum()) for c in
                                ("commission","stamp","transfer","slippage","total")}
        reports[name]["blocked_orders"] = int(daily.blocked_buy.sum()+daily.blocked_sell.sum())
        reports[name]["max_stale_weight"] = float(daily.stale_weight.max())
        for year,g in daily.groupby(pd.to_datetime(daily.date).dt.year):
            yearly.append({"model":name,"year":int(year),"partial_year":year==2026,**metrics(g.net)})
    a = pd.DataFrame(books["NN_softmax"].daily).set_index("date").net
    b = pd.DataFrame(books["F17"].daily).set_index("date").net
    if not a.index.equals(b.index):
        raise ValueError("Nonaligned strategy returns")
    summary = {"models":reports,"qualification":gates(reports["NN_softmax"],reports["F17"],True),
               "paired_uncertainty":paired_interval(a,b),"research_contaminated":True}
    ic = []
    for date,g in pred.groupby("formation_time"):
        valid = g.label_reason.eq("ok")
        for name in ("NN","F17","Ridge"):
            ic.append({"date":str(date.date()),"model":name,"rank_IC":rank_ic(g.loc[valid,name],g.loc[valid,"label"]),
                       "stocks":len(g),"missing_label_fraction":float(1-valid.mean())})
    pd.DataFrame(ic).to_csv(output/"rank_IC.csv",index=False)
    pd.DataFrame(yearly).to_csv(output/"yearly_metrics.csv",index=False)
    write_json(output/"summary.json",summary)


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    for name in ("package","runs","output"):
        ap.add_argument("--"+name,type=Path,required=True)
    a=ap.parse_args()
    evaluate(a.package,a.runs,a.output)
