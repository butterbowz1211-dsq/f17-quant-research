# Research project author: Gilmour Jia
"""Read-only diagnostics of frozen predictions; no training or model selection."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import runtime
import datetime as dt
import json
import numpy as np
import pandas as pd
from metrics import rank_ic
from spec import file_hash, code_hash, write_json
from research import research_acceptance


def huber(residual):
    a = np.abs(residual)
    return float(np.where(a <= 1., residual**2/2, a-.5).mean())


def main():
    run = ROOT/"runs/research_v1"
    out = Path(__file__).resolve().parent
    manifest = json.loads((run/"run_manifest.json").read_text(encoding="utf-8"))
    if manifest["code_hash"] != code_hash():
        raise ValueError("Frozen training code changed")
    accepted = research_acceptance(ROOT/"derived/research_training_v1")
    if manifest["data_manifest"] != accepted:
        raise ValueError("Frozen input package changed")
    for name, sha in manifest["files"].items():
        if file_hash(run/name) != sha:
            raise ValueError(f"Changed completed-run artifact: {name}")
    ledger = [json.loads(x) for x in (run/"experiment_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    completed = [r for r in ledger if r["state"] == "complete"]
    if any(r["state"] == "failed" for r in ledger):
        raise ValueError("Failed fitting task in ledger")
    for model, expected in (("NN",51),("Ridge",9)):
        for state in ("started","complete"):
            if sum(r["model"] == model and r["state"] == state for r in ledger) != expected:
                raise ValueError("Unexpected fit count")
    daily, coverage = [], []
    for year in range(2021,2027):
        p = pd.read_csv(run/f"predictions_{year}.csv")
        if p.duplicated(["formation_time","symbol"]).any():
            raise ValueError("Duplicate prediction keys")
        if not np.isfinite(p[["NN","Ridge"]].to_numpy()).all():
            raise ValueError("Nonfinite prediction")
        # Ridge and all final NN seeds estimate precisely the same label scale
        # from the same matured final-fit labels, independently of outer labels.
        with np.load(run/f"ridge_{year}_final.npz",allow_pickle=False) as z:
            scale = float(z["scale"])
        for date,g in p.groupby("formation_time",sort=True):
            m = g.label_reason.eq("ok") & np.isfinite(g[["NN","Ridge","F17","label"]]).all(axis=1)
            good = g[m]
            coverage.append({"year":year,"formation_time":date,"all_rows":len(g),
                             "shared_valid_rows":len(good),"label_scale":scale})
            if len(good) < 3:
                if g.label_reason.eq("ok").any():
                    raise ValueError("Usable date lacks enough common predictor rows")
                continue
            row = {"year":year,"formation_time":date,"n":len(good),
                   "entry_time":g.entry_time.iloc[0],"label_end_time":g.label_end_time.iloc[0]}
            for name in ("NN","Ridge","F17"):
                ic = rank_ic(good[name],good.label)
                if ic is None:
                    raise ValueError("Undefined rank correlation")
                row[name+"_rank_ic"] = ic
            for name in ("NN","Ridge"):
                row[name+"_huber"] = huber((good[name].to_numpy()-good.label.to_numpy())/scale)
                row[name+"_mae"] = float(np.abs(good[name]-good.label).mean())
            row["zero_huber"] = huber(-good.label.to_numpy()/scale)
            daily.append(row)
    daily, coverage = pd.DataFrame(daily), pd.DataFrame(coverage)
    daily.to_csv(out/"date_diagnostics.csv",index=False)
    coverage.to_csv(out/"prediction_coverage.csv",index=False)
    summaries = []
    for year, frame in [*daily.groupby("year"), ("ALL",daily)]:
        row = {"period":str(year),"dates":len(frame),"shared_labels":int(frame.n.sum())}
        for name in ("NN","Ridge","F17"):
            col = frame[name+"_rank_ic"]
            row[name+"_mean_rank_ic"] = float(col.mean())
            row[name+"_positive_ic_fraction"] = float(col.gt(0).mean())
        for name in ("NN","Ridge","zero"):
            row[name+"_date_mean_huber"] = float(frame[name+"_huber"].mean())
        row["NN_beats_F17_dates"] = int((frame.NN_rank_ic > frame.F17_rank_ic).sum())
        row["NN_beats_Ridge_dates"] = int((frame.NN_rank_ic > frame.Ridge_rank_ic).sum())
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(out/"year_diagnostics.csv",index=False)
    winner = json.loads((run/"winner.lock.json").read_text(encoding="utf-8"))
    selection = json.loads((run/"validation_selection.json").read_text(encoding="utf-8"))
    ridge_winner = json.loads((run/"ridge_winner.lock.json").read_text(encoding="utf-8"))
    local = dt.timezone(dt.timedelta(hours=8))
    epoch_rows = []
    for year in range(2021,2027):
        finals = [r for r in completed if r["model"] == "NN" and r["task"].startswith(f"{year}_final_")]
        epoch_rows.append({"year":year,"seed_epochs":[r["best_epoch"] for r in finals]})
    summary = {
        "checked_at":dt.datetime.now(local).isoformat(),
        "finished_at":dt.datetime.fromisoformat(ledger[-1]["at"]).astimezone(local).isoformat(),
        "elapsed_fit_protocol_seconds":(dt.datetime.fromisoformat(ledger[-1]["at"]) -
                                        dt.datetime.fromisoformat(ledger[0]["at"])).total_seconds(),
        "NN_completed":51,"Ridge_completed":9,"failed":0,
        "run_artifacts_verified":len(manifest["files"]),
        "spec_hash":manifest["spec_hash"],
        "diagnostic_script_sha256":file_hash(__file__),
        "prediction_dates":len(coverage),"scored_dates":len(daily),
        "prediction_rows":int(coverage.all_rows.sum()),"scored_rows":int(daily.n.sum()),
        "winner":winner,"validation_candidates":selection,"ridge_winner":ridge_winner,
        "final_epochs":epoch_rows,"summaries":summaries,
        "metric":"Spearman correlation across stocks per formation date; dates equally weighted",
        "scope":"2021-2026 historical time extrapolation with research contamination",
        "F17":"independent research score on common cleaned factors, not original account performance",
        "execution_backtest_accepted":False,"no_additional_fits":True}
    write_json(out/"summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
