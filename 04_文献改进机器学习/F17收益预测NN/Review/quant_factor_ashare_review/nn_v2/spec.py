# Research project author: Gilmour Jia
"""Predeclared NN-only experiment. No F17 source modules are imported."""
import hashlib
import json
from pathlib import Path

FACTORS = [
    "TURN", "MAX5", "INTRADAY", "ILLIQ", "COMMOVE", "TURN_SHOCK",
    "REV_1W", "ISKEW", "IVOL", "OVERNIGHT", "LEAD_LAG", "CO_SKEW",
    "BETA", "REV_LT", "SIZE", "HIGH52", "RES_MOM",
]
FEATURES = FACTORS + [x + "_missing" for x in FACTORS]
SPEC = {
    "version": "nn_v2.1", "decision_date": "2026-10-20",
    "scope": "Independent NN research implementation; original F17 sources read-only",
    "factors": FACTORS, "features": FEATURES,
    "preprocessing": {
        "order": ["formation_universe", "MAD_clip_3x1.4826", "industry_size_residual",
                  "cross_section_zscore_ddof1", "formation_median", "missing_flags"],
        "min_cross_section": 30, "min_industry_size": 5, "eps": 1e-12,
        "all_missing_factor": "reject_date_and_block_acceptance",
        "constant_finite_factor": "zero_signal_keep_missing_flags",
        "industry": "historical_if_verified; static_proxy_must_be_disclosed_for_both",
    },
    "universe": {"listing_trading_days": 250, "drop_smallest_float_cap": 0.30,
                 "exclude_st": True, "unknown_st": "exclude_and_report",
                 "markets": ["SH_A", "SZ_A"], "formation_limit_up": "exclude",
                 "security_master": "historical_listing_and_delisting_required"},
    "label": {"horizon": 21, "entry_offset": 1, "end_offset": 22,
              "formula": "adjusted_open[t+22]/adjusted_open[t+1]-1",
              "missing": "NaN_with_reason; no_imputation",
              "boundary": "label_end_time <= cutoff; validation_end < first_outer_formation",
              "interior_missing": "allowed_only_when_verified_suspension",
              "corporate_actions": "verified_adjustment_and_explicit_cash_share_ledger_required"},
    "calendar": {"anchor": "2010-01-04", "step": 21, "phase": 0,
                 "source": "global_exchange_calendar", "reset_at_year": False,
                 "reason": "run_final.py detailed main series uses phase 0; no phase search"},
    "network": {"input": 34, "hidden": [32, 16], "dropout": 0.1,
                "activation": "ReLU", "output": "unconstrained_return"},
    "training": {"optimizer": "AdamW", "max_epochs": 100, "patience": 10,
                 "min_delta": 1e-6, "batch_dates": 4, "stock_chunk": 4096,
                 "epoch": "all_training_dates_once", "shuffle_training_dates": True,
                 "loss": "Huber", "delta": 1.0,
                 "label_scale": "max(date_equal_weighted_lower_median_abs_return,1e-4); no_centering; quantile_tolerance_1e-12",
                 "date_weight": "equal", "within_date_weight": "equal_valid_labels",
                 "seeds": [0, 1, 2], "lr_grid": [0.0003, 0.001],
                 "weight_decay_grid": [0.0001, 0.001, 0.01],
                 "selection": "mean_best_validation_Huber_over_3_seeds",
                 "tie": "lower_grid_index_exact_loss_tie; earlier_epoch_within_min_delta",
                 "final_fit": "each_seed_best_epoch; all_matured_labels_since_2013",
                 "prediction": "mean_3_predictions_in_return_units",
                 "max_nn_fit_tasks": 51, "first_grid_year": 2021},
    "folds": [{"year": y, "train_start": "2013-01-01",
               "train_end": f"{y-3}-12-31", "valid_start": f"{y-2}-01-01",
               "valid_end": f"{y-1}-12-31", "research_contaminated": True}
              for y in range(2021, 2027)],
    "ridge": {"alphas": [0.0001, 0.01, 1.0], "selection": "same_date_equal_validation_Huber",
              "objective": "date_equal_weighted_squared_error_plus_alpha_L2",
              "intercept_penalized": False, "first_fold_only": True,
              "max_fit_tasks": 9},
    "portfolio": {"n": 50, "retention_rank": 80, "minimum_candidates": 20,
                  "cap": 0.03, "main": "softmax", "temperature": 1.0,
                  "score_map": "selected_average_rank_scaled_to_[0,1]; all_tied_equal",
                  "ties": "symbol_ascending", "cash": "unallocated_cap_lots_fees_constraints",
                  "nn_equal_diagnostic": True, "ridge_readout": "same_softmax_as_NN",
                  "f17_readout": "skip_missing_mean_score_then_equal",
                  "buffer_state": "previous_selected_list; actual_holdings_remain_in_ledger"},
    "execution": {"capital": 10000000.0, "leverage": False, "net_commission": 0.0002,
                  "min_commission": 5.0, "commission_unit": "symbol_side_day_aggregated_fill",
                  "commission_includes": ["broker_net", "handling", "regulatory"],
                  "separate": ["stamp", "transfer", "slippage"],
                  "slippage": 0.001, "participation": 0.05,
                  "capacity": "lagged_20_day_mean_amount; no_future_daily_volume",
                  "buy_lot": "main_ChiNext_100; STAR_min200_increment1",
                  "sells": "partial_main_100; odd_remainder_only_full_exit_within_capacity",
                  "stale_mark": "carry_last_verified_close_adjusted_for_actions; reject_after_60_sessions",
                  "trade_price": "unadjusted_open", "annual_reset": False,
                  "fee_table_start": "2021-01-01",
                  "fee_table_scope": "actual_execution_period_2021_onward; no_pre2021_backtest"},
    "statistics": {"periods": 252, "rf_annual": 0.0, "rf_source": "explicit_zero_assumption",
                   "sharpe": "mean_daily_excess/std_daily_excess_ddof1*sqrt252",
                   "hac_lags": 21, "hac_small_sample_correction": False,
                   "paired_bootstrap": "circular_blocks", "block": 21,
                   "replicates": 2000, "seed": 42, "interval": [0.025, 0.975]},
    "acceptance": {"net_CAGR_gt_F17": True, "Sharpe_ge_F17": True,
                   "vol_ratio_max": 1.10, "drawdown_worsening_max": 0.03,
                   "data_execution_pass_required": True,
                   "future_confirmation": "not_completed; starts_after_verified_freeze"},
}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_hash():
    root = Path(__file__).resolve().parent
    paths = list(root.glob("*.py")) + [root/"fees.csv"]
    return digest({p.name: file_hash(p) for p in sorted(paths)})


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def freeze(root, data_manifest):
    """A freeze is immutable. Code/data changes require a new run directory."""
    root = Path(root)
    import copy
    record = copy.deepcopy(SPEC)
    if data_manifest.get("training_mode") == "research_with_limitations":
        from research import CONTRACT
        record["version"] = CONTRACT["version"]
        record["training_data_contract"] = CONTRACT
        for key in ("preprocessing", "universe", "label", "calendar", "acceptance"):
            record[key].update(CONTRACT[key])
    record.update(data_manifest=data_manifest, code_hash=code_hash())
    record["spec_hash"] = digest(record)
    path = root / "experiment_spec.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != record:
        raise RuntimeError("Frozen experiment changed; use a new version directory")
    write_json(path, record)
    return record
