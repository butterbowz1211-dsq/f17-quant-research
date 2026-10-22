# Research project author: Gilmour Jia
"""Bounded 51-task expanding-window protocol. Requires a verified data package."""
import runtime
import argparse
import datetime as dt
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from data import DateSample, load_samples, split_samples
from model import (fit, save_bundle, load_bundle, predict, ridge_fit,
                   ridge_predict, ridge_validation)
from spec import SPEC, FEATURES, freeze, file_hash, digest, write_json


REQUIRED_CHECKS = (
    "security_coverage", "raw_prices_units", "adjustment_actions", "daily_status",
    "factor_lineage", "label_coverage", "execution_tests",
)


def acceptance(package):
    package = Path(package)
    p = package / "acceptance.json"
    if not p.exists():
        raise ValueError("Data acceptance missing. Formal training is blocked.")
    gate = json.loads(p.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise ValueError("Data/execution acceptance is not PASS. No formal fits started.")
    for key in REQUIRED_CHECKS:
        c = gate.get("checks", {}).get(key, {})
        evidence = (package/c.get("evidence", "__missing__")).resolve()
        if not evidence.is_relative_to(package.resolve()) or not evidence.is_file():
            raise ValueError(f"Missing local evidence: {key}")
        if c.get("status") != "PASS" or file_hash(evidence) != c.get("sha256"):
            raise ValueError(f"Invalid acceptance evidence: {key}")
    manifest = json.loads((package / "package_manifest.json").read_text(encoding="utf-8"))
    for name, sha in manifest["files"].items():
        p = (package/name).resolve()
        if not p.is_relative_to(package.resolve()) or file_hash(p) != sha:
            raise ValueError(f"Input package changed: {name}")
    if not {"samples.npz", "market.parquet", "actions.json"} <= manifest["files"].keys():
        raise ValueError("Incomplete package manifest")
    return {"package_manifest": manifest, "acceptance_sha256": file_hash(package/"acceptance.json")}


class Ledger:
    def __init__(self, path, frozen, nn_budget=51, ridge_budget=9):
        self.path, self.frozen = Path(path), frozen
        self.nn_budget, self.ridge_budget = int(nn_budget), int(ridge_budget)
        self.rows = [json.loads(s) for s in self.path.read_text(encoding="utf-8").splitlines()] if self.path.exists() else []
        if any(r["spec_hash"] != frozen["spec_hash"] for r in self.rows):
            raise ValueError("Experiment provenance changed")

    def append(self, task, stage, model, state, **kwargs):
        r = dict(task=task, stage=stage, model=model, state=state,
                 at=dt.datetime.now(dt.timezone.utc).isoformat(),
                 spec_hash=self.frozen["spec_hash"], **kwargs)
        with self.path.open("a",encoding="utf-8") as f:
            f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n")
        self.rows.append(r)

    def begin(self, task, stage, model, **config):
        previous = [r for r in self.rows if r["task"] == task]
        if previous:
            raise ValueError(f"Task {task} already attempted. Inspect ledger; explicit versioned recovery required.")
        limit = self.nn_budget if model == "NN" else self.ridge_budget
        count = len({r["task"] for r in self.rows if r["model"] == model})
        if count >= limit:
            raise ValueError("Frozen fit budget exhausted")
        self.append(task,stage,model,"started",config=config)


def run(package, out, device, mode="strict", last_year="2026"):
    last_year = int(last_year)
    if last_year < 2021:
        raise ValueError("last_year must be >= 2021")
    n_years = last_year - 2020
    nn_budget = 21 + 6 * (n_years - 1)
    ridge_budget = 3 + n_years
    if mode == "strict":
        manifest = acceptance(package)
    elif mode == "research_with_limitations":
        from research import research_acceptance, audit_sample_times
        manifest = research_acceptance(package)
    else:
        raise ValueError("Unknown training mode")
    samples = load_samples(package/"samples.npz")
    if mode == "research_with_limitations":
        audit_sample_times(samples, pd.to_datetime(pd.read_csv(package/"calendar.csv").date))
    # Validate every fold before spending any fit budget.
    for year in range(2021, last_year + 1):
        split_samples(samples, year)
    out.mkdir(parents=True,exist_ok=True)
    frozen = freeze(out,manifest)
    ledger = Ledger(out/"experiment_ledger.jsonl",frozen,nn_budget,ridge_budget)
    grid = [(lr,wd) for lr in SPEC["training"]["lr_grid"] for wd in SPEC["training"]["weight_decay_grid"]]
    def one(task, stage, train, valid, lr, wd, seed, epochs=None):
        completed = [r for r in ledger.rows if r["task"] == task and r["state"] == "complete"]
        if completed:
            record = completed[-1]
            path = out/(task+".pt")
            if file_hash(path) != record["checkpoint_sha256"]:
                raise ValueError(f"Changed checkpoint: {task}")
            m,b = load_bundle(path,device)
            return m,b["scalers"]["label_scale"],record["best_epoch"],record["validation_loss"]
        ledger.begin(task,stage,"NN",lr=lr,weight_decay=wd,seed=seed,epochs=epochs)
        t = time.monotonic()
        print(json.dumps({"event":"fit_started", "task":task, "device":device,
                          "mode":mode, "train_dates":len(train),
                          "train_labels":sum(int(s.valid.sum()) for s in train)}, ensure_ascii=False), flush=True)
        try:
            def progress(row):
                event = {"task":task, "seconds":round(time.monotonic()-t, 3), **row}
                with (out/"epoch_progress.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, allow_nan=False)+"\n")
                print(json.dumps(event, allow_nan=False), flush=True)
            model, scale, epoch, val, history = fit(train,valid,lr,wd,seed,epochs,device,progress=progress)
            path = out/(task+".pt")
            context = {"data_version": digest(manifest), "code_version": frozen["code_hash"],
                       "spec_hash": frozen["spec_hash"],
                       "train_range": [str(train[0].formation_time),str(train[-1].formation_time)],
                       "latest_label_end": str(max(s.label_end_time for s in train)),
                       "hyperparameters": {"lr":lr,"weight_decay":wd},
                       "training_mode": mode,
                       "data_contract": frozen.get("training_data_contract", {}),
                       "execution_accepted": mode == "strict"}
            save_bundle(path,model,scale,seed,epoch,context)
            reload, b = load_bundle(path,device)
            a = predict(model,train[0].x,scale,device)
            p = predict(reload,train[0].x,b["scalers"]["label_scale"],device)
            if not np.array_equal(a,p):
                raise ValueError("Saved/reloaded predictions differ")
            write_json(out/(task+"_history.json"),history)
            ledger.append(task,stage,"NN","complete",seconds=time.monotonic()-t,
                          best_epoch=epoch,validation_loss=val,checkpoint_sha256=file_hash(path))
            return reload,scale,epoch,val
        except Exception as e:
            ledger.append(task,stage,"NN","failed",seconds=time.monotonic()-t,reason=repr(e))
            raise
    winner = None
    ridge_alpha = None
    def one_ridge(task, stage, data, alpha, validation=None):
        done=[r for r in ledger.rows if r["task"]==task and r["state"]=="complete"]
        path=out/(task+".npz")
        if done:
            if file_hash(path)!=done[-1]["checkpoint_sha256"]:
                raise ValueError(f"Changed Ridge checkpoint: {task}")
            with np.load(path,allow_pickle=False) as z:
                return z["beta"],float(z["scale"]),done[-1].get("validation_loss")
        ledger.begin(task,stage,"Ridge",alpha=alpha)
        start=time.monotonic()
        try:
            beta,scale=ridge_fit(data,alpha)
            loss=ridge_validation(beta,scale,validation) if validation is not None else None
            np.savez(path,beta=beta,scale=scale,features=np.asarray(FEATURES),
                     alpha=alpha,spec_hash=frozen["spec_hash"])
            ledger.append(task,stage,"Ridge","complete",validation_loss=loss,
                          seconds=time.monotonic()-start,checkpoint_sha256=file_hash(path))
            return beta,scale,loss
        except Exception as e:
            ledger.append(task,stage,"Ridge","failed",seconds=time.monotonic()-start,reason=repr(e))
            raise
    for year in range(2021,last_year + 1):
        train,valid,final,outer = split_samples(samples,year)
        best_epochs = []
        if year == 2021:
            candidates = []
            for i,(lr,wd) in enumerate(grid):
                runs = [one(f"{year}_candidate{i}_seed{s}","inner",train,valid,lr,wd,s) for s in (0,1,2)]
                candidates.append({"index":i,"lr":lr,"weight_decay":wd,
                                   "loss":float(np.mean([r[3] for r in runs])),
                                   "epochs":[r[2] for r in runs]})
            best = min(candidates,key=lambda c:(c["loss"],c["index"]))
            winner = best["lr"],best["weight_decay"]
            best_epochs = best["epochs"]
            write_json(out/"validation_selection.json",candidates)
            # Written before any outer predictions or performance inspection.
            write_json(out/"winner.lock.json",{"lr":winner[0],"weight_decay":winner[1],
                       "seed_epochs_2021":best_epochs,"criterion":"mean_validation_Huber",
                       "spec_hash":frozen["spec_hash"]})
            ridge_candidates = []
            for i,alpha in enumerate(SPEC["ridge"]["alphas"]):
                task = f"ridge_candidate{i}"
                beta,scale,loss = one_ridge(task,"inner",train,alpha,valid)
                ridge_candidates.append((loss,i,alpha))
            ridge_alpha = min(ridge_candidates)[2]
            write_json(out/"ridge_winner.lock.json",{"alpha":ridge_alpha,"candidates":ridge_candidates})
        else:
            for seed in (0,1,2):
                r = one(f"{year}_inner_seed{seed}","inner",train,valid,*winner,seed)
                best_epochs.append(r[2])
        models = [one(f"{year}_final_seed{s}","final",final,None,*winner,s,best_epochs[s]) for s in (0,1,2)]
        task = f"ridge_{year}_final"
        beta,rs,_ = one_ridge(task,"final",final,ridge_alpha)
        rows=[]
        for s in outer:
            pred = np.mean([predict(m,s.x,scale,device) for m,scale,_,_ in models],axis=0)
            rp = ridge_predict(beta,s.x,rs)
            rows.append(pd.DataFrame({"formation_time":s.formation_time,"entry_time":s.entry_time,
                         "label_end_time":s.label_end_time,"symbol":s.symbols,"NN":pred,"Ridge":rp,
                         "F17":s.f17_score,"label":s.y,"label_reason":s.label_reason,
                         "research_contaminated":True}))
        pd.concat(rows,ignore_index=True).to_csv(out/f"predictions_{year}.csv",index=False)
    if sum(r["state"]=="complete" and r["model"]=="NN" for r in ledger.rows) != nn_budget:
        raise RuntimeError("Incomplete fit budget")
    if sum(r["state"]=="complete" and r["model"]=="Ridge" for r in ledger.rows) != ridge_budget:
        raise RuntimeError("Incomplete Ridge budget")
    write_json(out/"run_manifest.json",{"spec_hash":frozen["spec_hash"],
        "code_hash":frozen["code_hash"],"data_manifest":manifest,
        "last_prediction_year":last_year,"nn_budget":nn_budget,"ridge_budget":ridge_budget,
        "files":{p.name:file_hash(p) for p in out.iterdir() if p.is_file() and p.name!="run_manifest.json"}})


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--package",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    ap.add_argument("--mode",choices=["strict","research_with_limitations"],default="strict")
    ap.add_argument("--last-year",type=int,default=2026)
    a=ap.parse_args()
    run(a.package,a.out,a.device,a.mode,a.last_year)
