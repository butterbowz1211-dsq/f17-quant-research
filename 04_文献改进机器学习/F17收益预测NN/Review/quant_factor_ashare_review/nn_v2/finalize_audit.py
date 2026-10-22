# Research project author: Gilmour Jia
"""Freeze this code delivery and verify original sources remain byte-identical."""
import runtime
import json
from pathlib import Path
import pandas as pd
from spec import SPEC, code_hash, digest, file_hash, write_json


def main():
    root=runtime.ROOT
    before=json.loads((root/"audit/original_code_manifest.json").read_text(encoding="utf-8"))
    verification=[]
    for name,record in before["files"].items():
        current=file_hash(runtime.SOURCE/name)
        verification.append({"path":name,"before":record["sha256"],"after":current,
                             "unchanged":current==record["sha256"]})
    write_json(root/"audit/original_code_verification.json",{
        "files":len(verification),"all_unchanged":all(r["unchanged"] for r in verification),
        "records":verification})
    if not all(r["unchanged"] for r in verification):
        raise ValueError("An original source changed; inspect before delivery")
    panel=pd.read_pickle(runtime.SOURCE/"data/panel.pkl")
    price=panel[["open","high","low","close"]]
    invalid=(price<=0).any(axis=1) | (panel.high+1e-5<price[["open","close","low"]].max(axis=1)) | (panel.low-1e-5>price[["open","close","high"]].min(axis=1))
    panel.loc[invalid].to_csv(root/"audit/invalid_adjusted_ohlc_rows.csv",index=False)
    rev=pd.read_pickle(runtime.SOURCE/"data/factors/REV_LT.pkl").iloc[::21]
    absent=rev[(rev.index.year>=2013)&rev.isna().all(axis=1)].index
    write_json(root/"audit/REV_LT_missing_dates.json",{"all_missing_formation_dates":[str(d.date()) for d in absent]})
    freeze={**SPEC,"status":"CODE_PROTOCOL_FROZEN_DATA_BLOCKED",
        "source_manifest_sha256":file_hash(root/"audit/data_manifest.json"),
        "derived_union_manifest_sha256":file_hash(root/"derived/source_union_v1/manifest.json"),
        "akshare_assessment_sha256":file_hash(root/"akshare_assessment/assessment.json"),
        "code_hash":code_hash(),"formal_nn_fits":0,"formal_ridge_fits":0,
        "scope_update":"User approved code and gap-list delivery; requested bounded AkShare feasibility probe"}
    freeze["spec_hash"]=digest(freeze)
    write_json(root/"experiment_spec.json",freeze)
    write_json(root/"acceptance.json",{"status":"BLOCKED","formal_training_allowed":False,
        "reason":"Partial AkShare recovery does not establish complete PIT data and corporate-action execution",
        "evidence":"akshare_assessment/assessment.json","checks":{}})
    write_json(root/"prospective_config.json",{
        "enabled":False,"mode":"paper_research_only","allow_live_orders":False,
        "start_condition":["data_and_execution_acceptance_PASS","code_and_protocol_freeze",
                           "three_final_models_trained_on_matured_labels","fee_table_current",
                           "first_future_global_21_session_formation_after_freeze"],
        "capital":10000000,"phase":0,"prediction":"three_seed_mean_return",
        "readout":"rank_softmax_T1_cap3pct","historical_2021_2026_independent":False,
        "monitor_created":False})
    (root/"prediction_log_template.csv").write_text(
        "formation_time,entry_time,symbol,predicted_return,model_bundle_hash,data_hash,spec_hash,weight,target_cash,actual_shares,fill_price,commission,stamp,transfer,slippage,blocked_reason,label_end_time,realized_label\n",
        encoding="utf-8")
    (root/"experiment_ledger.jsonl").write_text(json.dumps({
        "stage":"delivery","model":"NN","state":"blocked_before_formal_fit",
        "spec_hash":freeze["spec_hash"],"formal_fits_used":0,"formal_fits_budget":51,
        "reason":"Data incomplete; user approved code/gap delivery"},ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"Original sources unchanged: {len(verification)} files; formal fits 0/51")


if __name__=="__main__":main()
