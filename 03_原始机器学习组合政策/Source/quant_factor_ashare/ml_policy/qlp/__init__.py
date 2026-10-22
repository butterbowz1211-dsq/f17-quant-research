# Research project author: Gilmour Jia
"""Direct-weight portfolio policy research package."""
from .weights import (blend_targets, feasible_bounds, no_trade_mask,
                      project_capped_simplex)
from .execution import ExecutionConfig, run_target_weight_engine, stamp_duty_rate
from .relations import (aggregate_neighbors, market_residual_returns,
                        rolling_neighbor_graph)
from .features import build_market_state, build_stock_features, cross_sectional_rank
from .dataset import (Assembled, DatasetConfig, Sample, assemble, forward_returns,
                      make_sample, standardize_state)
from .policy import (DirectWeightPolicy, PolicyConfig, capped_simplex_soft,
                     net_growth_surrogate, topk_weights)
from .splits import Fold, expanding_folds, CONTAMINATED_FROM

__all__ = [
    "project_capped_simplex", "blend_targets", "no_trade_mask", "feasible_bounds",
    "ExecutionConfig", "run_target_weight_engine", "stamp_duty_rate",
    "rolling_neighbor_graph", "aggregate_neighbors", "market_residual_returns",
    "build_stock_features", "cross_sectional_rank", "build_market_state",
    "Assembled", "DatasetConfig", "Sample", "assemble", "forward_returns",
    "make_sample", "standardize_state",
    "DirectWeightPolicy", "PolicyConfig", "capped_simplex_soft",
    "net_growth_surrogate", "topk_weights",
    "Fold", "expanding_folds", "CONTAMINATED_FROM",
]
