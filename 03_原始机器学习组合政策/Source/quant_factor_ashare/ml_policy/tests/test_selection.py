# Research project author: Gilmour Jia
"""Contract tests for validation-window model selection.

The early-stopping statistic is part of the research protocol: if it silently
falls back to portfolio growth, the rank-IC experiment is not testing what the
report claims.  These tests keep the metric switch explicit and keep the IC
diagnostic on the validation cross-section.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from run_policy import evaluate_surrogate, selection_score, train_one_seed
from qlp.policy import DirectWeightPolicy, PolicyConfig

BASE = dict(n_own=21, n_neighbour=40, n_market=6, n_state=3, hidden=(8, 4))


def _valid(**over):
    out = dict(n_decisions=24, ann_return_approx=0.20, ann_vol_approx=0.10,
               mean_log_growth=0.01, n_ic=24, rank_ic=0.03, rank_ic_ir=0.50)
    out.update(over)
    return out


def test_selection_metric_switches_between_score_ic_and_growth():
    v = _valid()
    assert selection_score(v, metric="score") == pytest.approx(0.15)
    assert selection_score(v, metric="growth") == pytest.approx(0.01)
    assert selection_score(v, metric="ic") == pytest.approx(0.03)
    assert selection_score(v, metric="icir") == pytest.approx(0.50)


def test_missing_ic_is_not_allowed_to_win_by_default():
    assert selection_score({}, metric="ic") == float("-inf")
    assert selection_score(_valid(n_ic=0, rank_ic=None), metric="ic") == float("-inf")


def test_unknown_selection_metric_is_rejected():
    with pytest.raises(ValueError):
        selection_score(_valid(), metric="sharpe")


def test_evaluate_surrogate_reports_validation_rank_ic():
    cfg = PolicyConfig(**BASE)
    rng = np.random.default_rng(0)
    samples = [SimpleNamespace(feat=rng.normal(size=(40, cfg.n_input)).astype(np.float32),
                               ids=np.arange(40),
                               returns=(rng.normal(size=40) * 0.02).astype(np.float32))
               for _ in range(3)]
    p = DirectWeightPolicy(cfg)
    torch.manual_seed(0)
    out = evaluate_surrogate(p, samples, "cpu", rb=21, n_symbols=40)
    assert out["n_ic"] == 3
    assert np.isfinite(out["rank_ic"])
    assert np.isfinite(out["rank_ic_ir"])


def test_top_g_ic_is_scoped_to_the_names_that_can_receive_weight():
    """With top-G the excluded scores carry no portfolio gradient.

    Ranking all names would mix trained scores with initialisation noise, so the
    validation IC must be computed on the active set only.
    """
    cfg = PolicyConfig(**dict(BASE, top_g=20))
    rng = np.random.default_rng(1)
    samples = [SimpleNamespace(feat=rng.normal(size=(40, cfg.n_input)).astype(np.float32),
                               ids=np.arange(40),
                               returns=(rng.normal(size=40) * 0.02).astype(np.float32))]
    p = DirectWeightPolicy(cfg)
    out = evaluate_surrogate(p, samples, "cpu", rb=21, n_symbols=40)
    assert out["n_ic"] == 1
    assert np.isfinite(out["rank_ic"])


def test_last_selection_keeps_the_final_epoch():
    """The forced-final ablation must not silently fall back to early stopping."""
    cfg = PolicyConfig(**BASE)
    rng = np.random.default_rng(2)
    samples = [SimpleNamespace(feat=rng.normal(size=(40, cfg.n_input)).astype(np.float32),
                               ids=np.arange(40),
                               returns=(rng.normal(size=40) * 0.02).astype(np.float32))
               for _ in range(4)]
    res = train_one_seed(samples, samples, cfg, seed=0, epochs=3, lr=0.01,
                         batch=2, patience=1, temp_end=None, rb=21,
                         n_symbols=40, device="cpu", selection_metric="last",
                         verbose=False)
    assert res["best_epoch"] == 3
