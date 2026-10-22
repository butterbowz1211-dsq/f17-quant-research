# Research project author: Gilmour Jia
"""Explicit, user-authorized training contract; never execution acceptance."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from spec import file_hash, code_hash

MODE = "research_with_limitations"
AUTHORIZATION = "Research training authorized with documented limitations."
CONTRACT = {
    "version": "nn_v2.3_research",
    "authorization": AUTHORIZATION,
    "execution_accepted": False,
    "preprocessing": {
        "all_missing_factor": "reject_date_and_report; retain_all_17_features",
        "industry": "existing_static_industry_proxy; identical_for_NN_and_research_F17",
    },
    "universe": {
        "security_master": "local_listing_dates; conservative_first_quote_fallback",
        "unknown_st": "exclude; Baidu daily name preferred; backward month-end proxy otherwise",
        "formation_limit_up": "exclude_using_Baidu_daily_limit_up_else_board_rate_preclose_proxy",
        "float_cap": "raw_close_times_Baidu_float_shares_else_Sina_shares_else_turnover_proxy",
    },
    "label": {
        "corporate_actions": "vendor_adjusted_price_research_proxy; NOT_verified_cash_share_settlement",
        "interior_missing": "allowed_only_with_observed_BaoStock_suspension",
        "missing": "NaN_with_reason; no_imputation; no_return_outlier_deletion",
        "price_precision": "raw_OHLC_scaled_by_same_day_vendor_adjusted_close_over_raw_close",
    },
    "calendar": {"source": "union_of_existing_market_trading_dates; BaoStock_overlap_audited"},
    "acceptance": {
        "training_authorized_with_limitations": True,
        "data_execution_pass_required": True,
        "future_confirmation": "not_started; execution_and_data_acceptance_still_required",
    },
}


def research_label(prices, dates, t, suspended, terminal, boundary):
    """No verified-adjustment claim. Use only coherent observed price endpoints."""
    e, z = t + 1, t + 22
    n = prices.shape[1]
    y = np.full(n, np.nan)
    reason = np.full(n, "ok", dtype="U40")
    calendar = np.asarray(dates, dtype="datetime64[ns]")
    entry = calendar[e] if e < len(calendar) else np.datetime64("NaT", "ns")
    end = calendar[z] if z < len(calendar) else np.datetime64("NaT", "ns")
    if z >= len(calendar):
        reason[:] = "not_matured"
        return y, reason, entry, end
    p = prices[e:z+1]
    good = np.isfinite(p) & (p > 0)
    reason[(~good & ~suspended[e:z+1]).any(axis=0)] = "unexplained_price_gap"
    reason[~good[0] | ~good[-1]] = "missing_endpoint"
    # Vendor carried prices on suspended endpoints are not tradable open labels.
    reason[suspended[e] | suspended[z]] = "suspended_endpoint"
    reason[terminal[t:z+1].any(axis=0)] = "delisting_unresolved"
    reason[boundary[t+1:z+1].any(axis=0)] = "code_change_unresolved"
    ok = reason == "ok"
    y[ok] = p[-1, ok] / p[0, ok] - 1
    return y, reason, entry, end


def research_acceptance(package):
    package = Path(package).resolve()
    gate = json.loads((package/"research_acceptance.json").read_text(encoding="utf-8"))
    manifest = json.loads((package/"package_manifest.json").read_text(encoding="utf-8"))
    if gate.get("status") != "RESEARCH_TRAINING_AUTHORIZED_WITH_LIMITATIONS":
        raise ValueError("Research training not authorized")
    if gate.get("execution_accepted") is not False or manifest.get("training_contract") != CONTRACT:
        raise ValueError("Research contract changed or misrepresents execution acceptance")
    if manifest.get("preparation_code") != code_hash():
        raise ValueError("Preparation/code version changed; rebuild into a new data version")
    required = {"samples.npz", "calendar.csv", "quality_report.json", "authorization.json",
                "label_coverage.csv", "security_coverage.csv", "skipped_dates.json",
                "factors_raw.parquet", "factors_clean.parquet", "sample_quality.parquet"}
    if not required <= manifest["files"].keys():
        raise ValueError("Incomplete research package")
    for name, sha in manifest["files"].items():
        p = (package/name).resolve()
        if not p.is_relative_to(package) or not p.is_file() or file_hash(p) != sha:
            raise ValueError(f"Research input changed: {name}")
    auth = json.loads((package/"authorization.json").read_text(encoding="utf-8"))
    if auth.get("user_instruction") != AUTHORIZATION or auth.get("mode") != MODE:
        raise ValueError("Missing specific user authorization")
    report = json.loads((package/"quality_report.json").read_text(encoding="utf-8"))
    for check in ("finite_features", "unique_keys", "label_endpoints", "purge_all_folds",
                  "all_17_factors", "source_hashes"):
        if report.get("checks", {}).get(check) is not True:
            raise ValueError(f"Research check failed: {check}")
    return {"training_mode": MODE, "package_manifest": manifest,
            "research_acceptance_sha256": file_hash(package/"research_acceptance.json"),
            "execution_accepted": False}


def audit_sample_times(samples, dates):
    dates = pd.DatetimeIndex(dates)
    positions = {d.to_datetime64(): i for i, d in enumerate(dates)}
    for s in samples:
        t = positions[s.formation_time]
        if t % 21:
            raise ValueError("Formation date not on the global grid")
        for offset, observed in ((1, s.entry_time), (22, s.label_end_time)):
            expected = dates[t+offset].to_datetime64() if t+offset < len(dates) else np.datetime64("NaT", "ns")
            if not ((np.isnat(expected) and np.isnat(observed)) or observed == expected):
                raise ValueError("Label timestamp mismatch")
